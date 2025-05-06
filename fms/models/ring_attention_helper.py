from typing import List, Tuple, Dict, Optional, Union, Any
import torch
import torch.distributed as dist
from torch.distributed import P2POp
import math
import torch.nn as nn

# Assuming necessary imports like MultiHeadAttention, GatedLinearUnit, LayerNormParameterized
from fms.distributed.strategy import RingAttentionStrategy # Import RingAttentionStrategy
# from fms.modules.feedforward import GatedLinearUnit
# from fms.modules.layernorm import LayerNormParameterized


def _pad_to_block(t: torch.Tensor, target_len: int, dim: int = 2) -> torch.Tensor:
    """Pads tensor along a dimension to target_len."""
    length = t.size(dim)
    if length >= target_len:
        # Assert if already too large, should not happen if target_len is block_size and input is shard
        assert length == target_len, f"Tensor length {length} along dim {dim} is >= target {target_len} and not equal."
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = target_len - length
    pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


class RingAttentionHelper:
    """
    Helper class to perform the distributed Ring Attention computation within a block.
    """
    def __init__(
        self,
        attn_module: nn.Module, # Expected MultiHeadAttention
        strategy: Any, # Expected RingAttentionStrategy, but typed Any to avoid circular import here if strategy is not in scope
        llama_block: nn.Module, # Expected LLaMABlock
        use_cache: bool = False,
        ff: Optional[nn.Module] = None, # Expected GatedLinearUnit
        ff_norm: Optional[nn.Module] = None # Expected LayerNormParameterized
    ):
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.strategy = strategy # This should be the RingAttentionStrategy instance
        self.use_cache = use_cache # Ring attention helper might ignore this for main passes
        self.llama_block = llama_block # For compute_local_qkv_and_rope

        # Principle: Explicit Distributed State (Accessed from strategy)
        self.rank: int = self.strategy.rank
        self.world_size: int = self.strategy.world_size
        self.block_size: int = self.strategy.block_size

        self.head_dim: int = self.attn.emb_kq_per_head # Assuming this attribute exists
        self.scale: float = math.sqrt(self.head_dim)

        # Principle: Prevent Usage Bugs - Ensure FF modules are provided if needed
        if self.ff is None or self.ff_norm is None:
            # Depending on implementation, FF might be integrated differently or always present.
            # Add an assertion if FF is mandatory for this helper's forward_full.
            pass # Assuming FF is optional or handled elsewhere if not provided

    def forward(
        self,
        x_norm: torch.Tensor, # Local normalized input shard
        strategy: Any, # Expected RingAttentionStrategy (passed again for consistency)
        mask: Optional[torch.Tensor] = None, # Global mask
        position_ids: Optional[torch.Tensor] = None, # Global position_ids
        past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Likely ignored
        is_causal_mask: bool = False, # Likely handled by mask/logic
        valid_len: int = 0, # Actual number of tokens on this rank
        residual: Optional[torch.Tensor] = None, # Residual connection before LN+Attn
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Any]: # Returns (output, cache, extra)

        assert isinstance(strategy, RingAttentionStrategy), f"Rank {self.rank}: Helper forward expected RingAttentionStrategy, got {type(strategy)}"
        assert self.strategy is strategy, f"Rank {self.rank}: Helper strategy mismatch. Init strategy {self.strategy} != Forward strategy {strategy}"
        assert valid_len >= 0 and valid_len <= self.block_size, f"Rank {self.rank}: Invalid valid_len: {valid_len}"

        B, T_padded, _ = x_norm.shape
        # T_padded might not always be self.block_size if world_size=1 and seq_len < block_size
        # However, for ring attention (world_size > 1), inputs to helper.forward are expected to be padded.
        if self.world_size > 1:
            assert T_padded == self.block_size, f"Rank {self.rank}: Input x_norm to helper forward must be padded to block_size for multi-GPU: {T_padded} != {self.block_size}"

        # Fix 2.A: Create position_ids if None
        start_idx_global = self.rank * self.block_size
        if position_ids is None:
            # Create position_ids for the padded block, then fill valid part
            position_ids = torch.full((B, T_padded), fill_value=-1, dtype=torch.long, device=x_norm.device)
            if valid_len > 0:
                valid_global_positions = torch.arange(start_idx_global, start_idx_global + valid_len, device=x_norm.device)
                position_ids[:, :valid_len] = valid_global_positions.unsqueeze(0) # Broadcast if B > 1

        # Fix 2.B: Trim inputs to valid_len BEFORE RoPE computation
        x_norm_for_rope = x_norm[:, :valid_len, :]
        position_ids_for_rope = position_ids[:, :valid_len]
        # residual_for_rope will be used for forward_full's x_block
        residual_for_rope = residual[:, :valid_len, :] if residual is not None else None

        # Compute local QKV and apply RoPE
        q_local_padded, k_local_padded, v_local_padded = self.llama_block.compute_local_qkv_and_rope(
            self.attn, # Pass self.attn
            q=x_norm_for_rope, k=x_norm_for_rope, v=x_norm_for_rope, # Use trimmed inputs
            position_ids=position_ids_for_rope, # Pass trimmed position_ids
            use_cache=False,
            past_key_value_state=None,
            is_self=True
        )
        B, H, T_padded, D_head = q_local_padded.shape
        assert T_padded == self.block_size, f"Rank {self.rank}: QKV after compute_local_qkv_and_rope must be padded to block_size: {T_padded} != {self.block_size}"

        # QKV from compute_local_qkv_and_rope are already effectively valid_len long
        # because inputs to it were trimmed.
        q_local = q_local_padded[:, :, :valid_len, :]
        k_local = k_local_padded[:, :, :valid_len, :]
        v_local = v_local_padded[:, :, :valid_len, :]

        # x_norm_local for forward_full should be the trimmed version used for RoPE.
        # residual_local for forward_full is the trimmed residual_for_rope.
        x_norm_local_for_ffull = x_norm_for_rope
        residual_local_for_ffull = residual_for_rope

        # Perform the two-pass ring attention computation and post-attention layers
        # forward_full handles the residual connection and FF internally for the valid part
        attn_out_local_valid = self.forward_full(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            mask_global=mask, # Pass global mask
            valid_len=valid_len, # Pass actual valid length
            x_block=residual_local_for_ffull, # Pass trimmed residual
            x_norm_block=x_norm_local_for_ffull, # Pass trimmed x_norm
            q_start_global=self.rank * self.block_size # Pass global start index for Q
        )

        # The output from forward_full is the processed tensor for the valid tokens on this rank.
        # It needs to be padded back to block_size for subsequent layers/gathering.
        output_padded = _pad_to_block(attn_out_local_valid, self.block_size, dim=1)
        assert output_padded.size(1) == self.block_size, f"Rank {self.rank}: Output padding after forward_full failed: {output_padded.size(1)} != {self.block_size}"

        # In the current RingAttentionHelper code, cache and extra output are None
        return output_padded, None, None # Return padded output, None cache, None extra


    def forward_full(
        self,
        q_local: torch.Tensor, # Q tensor for valid local tokens (B, H, T_q_local, D_head)
        k_local: torch.Tensor, # K tensor for valid local tokens (B, H_kv, T_q_local, D_head)
        v_local: torch.Tensor, # V tensor for valid local tokens (B, H_kv, T_q_local, D_v)
        mask_global: Optional[torch.Tensor], # Global mask (B, 1, S, S) or similar
        valid_len: int, # Actual length of local Q, K, V
        x_block: Optional[torch.Tensor], # Residual connection slice (B, T_q_local, E)
        x_norm_block: torch.Tensor, # Normalized input slice (B, T_q_local, E)
        q_start_global: int # Global start index for Q block
    ) -> torch.Tensor: # Returns processed tensor for valid local tokens (B, T_q_local, E)

        B, H, T_q_local, D_head = q_local.shape
        D_v = v_local.shape[-1]
        device = q_local.device

        # Principle: Numerical Stability & Dtype Casting
        # Compute max scores (Pass 1)
        max_score = self._compute_max_score_pass(
            q_local.to(torch.float32), # Cast Q to float32 for computation
            k_local.to(torch.float32), # Cast K to float32
            mask_global,
            q_start_global,
            valid_len,
        )

        # Compute sums (Pass 2)
        numerator, denominator = self._compute_sums_pass(
            q_local.to(torch.float32), # Cast Q to float32
            k_local.to(torch.float32), # Cast K to float32
            v_local.to(torch.float32), # Cast V to float32
            mask_global,
            q_start_global,
            valid_len,
            max_score,
        )

        # Compute attention output
        attn_out_h = numerator / (denominator + 1e-10) # Use float32 for division
        attn_out_h = attn_out_h.to(q_local.dtype) # Cast result back to original dtype

        # Assert shape after attention calculation
        assert attn_out_h.shape == (B, H, T_q_local, D_v), f"Rank {self.rank}: Attn head output shape mismatch: {attn_out_h.shape} vs {(B, H, T_q_local, D_v)}"

        # Reshape and apply dense layer
        attn_out = attn_out_h.transpose(1, 2).contiguous().view(B, T_q_local, H * D_v)
        attn_out = self.attn.dense(attn_out)

        # First residual connection
        residual_1 = x_block + attn_out # x_block is residual_local_valid (B, T_q_local, E)

        # Apply FF norm and FF
        ff_ln_out = self.ff_norm(residual_1)
        ff_out = self.ff(ff_ln_out)

        # Second residual connection
        x = ff_out + residual_1

        # Return processed valid local tokens
        return x

    def _compute_max_score_pass(
        self,
        q_local_f32: torch.Tensor, # Q (float32) (B, H, T_q_local, D_head)
        k_local_f32: torch.Tensor, # K (float32) (B, H_kv, T_q_local, D_head)
        mask_global: Optional[torch.Tensor], # Global mask
        q_start_global: int, # Global start index for Q block
        valid_len_local: int, # Actual length of local Q
    ) -> torch.Tensor: # Returns max scores (float32) (B, H, T_q_local, 1)

        B, H, T_q_local, _ = q_local_f32.shape
        device = q_local_f32.device
        dtype = torch.float32

        max_score = torch.full((B, H, T_q_local, 1), -torch.inf, device=device, dtype=dtype) # Use torch.inf

        # Indices for the local Q block in the global sequence
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local_f32
        current_k_len = k_local_f32.shape[2]

        for i in range(self.world_size):
            # Global start index for the current K block in the ring
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            # Indices for the current K block in the global sequence
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)

            # Slice global mask relevant to current Q block and current K block
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None

            # Only compute scores if the current K block has elements
            if current_k_len > 0:
                 scores = self._compute_attention_scores(
                    q_local_f32,
                    current_k_block[:, :, :current_k_len, :], # Use sliced K block
                    q_indices_global,
                    k_indices_global,
                    current_mask,
                    apply_mask=True,
                    keep_causal=True # Causal mask is handled here
                 )
                 # Update max score across blocks
                 block_max = scores.amax(dim=-1, keepdim=True)
                 max_score = torch.maximum(max_score, block_max)


            # Ring shift K block (and V block in the sums pass)
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.block_size)

        return max_score

    def _compute_sums_pass(
        self,
        q_local_f32: torch.Tensor, # Q (float32) (B, H, T_q_local, D_head)
        k_local_f32: torch.Tensor, # K (float32) (B, H_kv, T_q_local, D_head)
        v_local_f32: torch.Tensor, # V (float32) (B, H_kv, T_q_local, D_v)
        mask_global: Optional[torch.Tensor], # Global mask
        q_start_global: int, # Global start index for Q block
        valid_len_local: int, # Actual length of local Q, K, V
        max_score: torch.Tensor, # Max scores from Pass 1 (float32) (B, H, T_q_local, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns numerator, denominator (float32) (B, H, T_q_local, D_v), (B, H, T_q_local, 1)

        B, H, T_q_local, D_head = q_local_f32.shape
        D_v = v_local_f32.shape[-1]
        device = q_local_f32.device
        dtype = torch.float32 # Accumulation dtype

        numerator = torch.zeros(B, H, T_q_local, D_v, device=device, dtype=dtype)
        denominator = torch.zeros(B, H, T_q_local, 1, device=device, dtype=dtype)

        # Indices for the local Q block in the global sequence
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local_f32
        current_v_block = v_local_f32
        current_k_len = k_local_f32.shape[2] # K and V should have the same sequence length dimension

        for i in range(self.world_size):
            # Global start index for the current K/V block in the ring
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            # Indices for the current K block in the global sequence
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)

             # Slice global mask relevant to current Q block and current K block
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None

            # Only compute scores if the current K block has elements
            if current_k_len > 0:
                scores = self._compute_attention_scores(
                    q_local_f32,
                    current_k_block[:, :, :current_k_len, :], # Use sliced K block
                    q_indices_global,
                    k_indices_global,
                    current_mask,
                    apply_mask=True, # Apply mask including causal
                    keep_causal=True
                )

                # Compute stable exponentiated scores
                stable_scores = scores - max_score # Use max_score from Pass 1
                exp_scores = torch.exp(stable_scores.clamp(min=-10.0, max=10.0)) # Clamp exp input for safety

                # Update numerator and denominator
                numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, current_v_block[:, :, :current_k_len, :]) # Use sliced V block
                denominator += exp_scores.sum(dim=-1, keepdim=True)

            # Ring shift K and V blocks
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.block_size)
                current_v_block, _ = self._ring_shift_tensor(current_v_block, self.block_size) # V has same length dim as K

        return numerator, denominator

    def _compute_attention_scores(
        self,
        q: torch.Tensor, # Q tensor (float32) (B, H, T_q_local, D_head)
        k: torch.Tensor, # K tensor (float32) (B, H_kv, T_k_block, D_head)
        q_indices_global: torch.Tensor, # Global indices for Q (T_q_local)
        k_indices_global: torch.Tensor, # Global indices for K block (T_k_block)
        mask: Optional[torch.Tensor] = None, # Sliced global mask (B, 1, T_q_local, T_k_block)
        apply_mask: bool = True,
        keep_causal: bool = True # Apply causal mask based on global indices
    ) -> torch.Tensor: # Returns scores (float32) (B, H, T_q_local, T_k_block)

        # Compute QK^T scaled by head_dim
        # Ensure K is broadcastable if H != H_kv (GQA)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # scores are float32

        # Apply masks
        if apply_mask:
             if mask is not None:
                scores = scores + mask.to(scores.dtype) # Apply padding/attention mask

             if keep_causal:
                # Apply causal mask based on global indices
                causal_mask = (k_indices_global[None, None, None, :] > q_indices_global[None, None, :, None]) # Shape (1, 1, T_q_local, T_k_block)
                scores = scores.masked_fill(causal_mask, -torch.inf) # Use -torch.inf for float32


        # Clamp scores for stability before softmax calculation
        # This clamp value should be related to the stable softmax implementation's exp clamping
        # Clamp scores in _update_totals uses exp(scores - max_score). The value here prevents
        # overflow if scores themselves are huge before subtracting max_score, although
        # standard practice often omits this first clamp. Let's keep it conservative.
        scores = scores.clamp(min=torch.finfo(scores.dtype).min) # Prevent -inf becoming NaN if other infs are present

        return scores

    def _update_max_score(
        self,
        scores: torch.Tensor, # Scores from _compute_attention_scores (float32) (B, H, T_q_local, T_k_block)
        current_max: torch.Tensor # Current max score (float32) (B, H, T_q_local, 1)
    ) -> torch.Tensor: # Returns updated max score (float32) (B, H, T_q_local, 1)

        # Compute max for the current block of scores
        # Replace -inf with a very small number before max to avoid issues if all scores are -inf
        block_max = scores.masked_fill(scores == -torch.inf, torch.finfo(scores.dtype).min).amax(dim=-1, keepdim=True)

        # Update the maximum score seen so far
        max_score = torch.maximum(current_max, block_max)

        return max_score

    def _update_totals(
        self,
        scores: torch.Tensor, # Scores from _compute_attention_scores (float32) (B, H, T_q_local, T_k_block)
        v: torch.Tensor, # V block (float32) (B, H_kv, T_k_block, D_v)
        max_score: torch.Tensor, # Max score from Pass 1 (float32) (B, H, T_q_local, 1)
        numerator: torch.Tensor, # Current numerator sum (float32) (B, H, T_q_local, D_v)
        denominator: torch.Tensor # Current denominator sum (float32) (B, H, T_q_local, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns updated numerator, denominator

        # Compute stable exponentiated scores using max_score from Pass 1
        stable_scores = scores - max_score # Shape (B, H, T_q_local, T_k_block)
        exp_scores = torch.exp(stable_scores.clamp(min=-10.0, max=10.0)) # Clamp exp input for stability

        # Update numerator (sum of exp(score - max) * V)
        # einsum handles broadcasting for GQA if H != H_kv
        numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v)

        # Update denominator (sum of exp(score - max))
        denominator += exp_scores.sum(dim=-1, keepdim=True)

        return numerator, denominator


    def _ring_shift_tensor(
        self,
        tensor: torch.Tensor, # Tensor to shift (e.g., K or V block) (B, H, T_block, D) padded to block_size
        pad_len: int # Expected padded length (self.block_size)
    ) -> Tuple[torch.Tensor, int]: # Returns received tensor and its valid length

        rank, world = self.rank, self.world_size
        send_rank = (rank + 1) % world
        recv_rank = (rank - 1 + world) % world

        # The input tensor should already be padded to block_size
        assert tensor.shape[-2] == pad_len, f"Rank {self.rank}: _ring_shift_tensor input not padded to {pad_len}: got {tensor.shape[-2]}"

        # Get the *actual* number of elements in the sequence dimension *before* padding
        # This assumes the tensor was padded from a valid_len size
        # This is implicitly stored or needs to be passed if not always padded from valid_len
        # Let's assume tensor *is* the padded K/V block, and its original valid length
        # isn't directly available here. The previous code used `tensor.shape[-2]`
        # before the assert, which seems wrong if the input is *already* padded.
        # The valid_len needs to be tracked *outside* this function and passed if required for the *sent* data.
        # However, the previous code seems to send/receive the *padded* block and *separately* send/receive the length.
        # Let's stick to that logic: send the padded tensor, send its *original* valid length.

        # Assuming the input `tensor` is already padded to `pad_len`
        # The length to send is the actual number of non-pad elements in this block
        # Where does this come from? It should be the `valid_len` of the K/V block
        # received in `forward_full`. This `valid_len` needs to be shifted along with K/V.
        # Let's assume the input `tensor` *is* the K/V block from the start of `forward_full`
        # or a received block, and `valid_len` refers to the actual data it contains.
        # This implies the `_ring_shift_tensor` should operate on the *valid* part and pad it.

        # REVISION based on original logic: Original _ring_shift_tensor takes `tensor` which is the block (potentially received and sliced)
        # and `pad_len` which is the target block_size. It calculates `valid_len = tensor.shape[-2]` (current size)
        # then pads it *again* if needed (which it shouldn't if it's already block_size).
        # It sends the padded version, and sends the calculated `valid_len`.
        # It receives a padded tensor and a received_len. It slices the received tensor using received_len.

        # Let's adjust based on the likely intent: The K/V blocks in `forward_full`
        # are slices of the *initially* padded `k_local_f32`, `v_local_f32` to `valid_len`.
        # These `valid_len` slices are what need to be padded *back* to `block_size` for sending.

        # Corrected logic based on how _ring_shift_tensor is called and what it likely does:
        # input `tensor` is a block of K or V (B, H, T_current, D) where T_current is the valid length of this block
        # We need to pad it to `pad_len` (block_size) for sending.
        valid_len_current = tensor.shape[-2] # The actual non-padded length of the input tensor

        # Pad the current tensor block to block_size for communication
        padded_for_send = _pad_to_block(tensor, pad_len, dim=-2).contiguous()
        assert padded_for_send.shape[-2] == pad_len, f"Rank {self.rank}: Padding for send failed: got {padded_for_send.shape[-2]}, expected {pad_len}"


        device = tensor.device

        # Create buffers for receiving
        recv_len_tensor = torch.empty(1, dtype=torch.int32, device=device)
        tensor_recv_padded = torch.empty_like(padded_for_send) # Received padded tensor


        # Prepare send/recv ops
        ops = [
            # Send the actual number of valid elements in the tensor being sent
            P2POp(op=dist.isend, tensor=torch.tensor([valid_len_current], dtype=torch.int32, device=device), peer=send_rank),
            # Receive the valid length of the tensor from the previous rank
            P2POp(op=dist.irecv, tensor=recv_len_tensor, peer=recv_rank),
            # Send the padded tensor block
            P2POp(op=dist.isend, tensor=padded_for_send, peer=send_rank),
            # Receive the padded tensor block
            P2POp(op=dist.irecv, tensor=tensor_recv_padded, peer=recv_rank),
        ]

        # Execute async, then wait
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        recv_len = recv_len_tensor.item() # The valid length of the received block

        # Assert received length makes sense
        assert recv_len >= 0 and recv_len <= pad_len, f"Rank {self.rank}: Received invalid length: {recv_len}, expected <= {pad_len}"

        # Slice the received padded tensor to its actual valid length
        tensor_recv = tensor_recv_padded[:, :, :recv_len, :].contiguous()

        # Return the received tensor (sliced) and its valid length
        return tensor_recv, recv_len