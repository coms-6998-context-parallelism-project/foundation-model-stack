from typing import List, Tuple, Dict, Optional, Union, Any
from fms.modules.attention import MultiHeadAttention
import torch
import torch.distributed as dist
from torch.distributed import P2POp
import math
import torch.nn.functional as F
import torch.nn as nn

# Assuming necessary imports like MultiHeadAttention, GatedLinearUnit, LayerNormParameterized
from fms.distributed.strategy import RingAttentionStrategy # Import RingAttentionStrategy
# from fms.modules.feedforward import GatedLinearUnit
# from fms.modules.layernorm import LayerNormParameterized

# Constants for numerical stability
EXP_CLAMP_MIN = -10.0
EXP_CLAMP_MAX = 10.0



# These functions are assigned as methods to the LLaMABlock class.
# `self` refers to the LLaMABlock instance.

def compute_local_qkv_and_rope(
    attn_data: MultiHeadAttention, # self.attn module
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: bool = False, # Likely ignored in ring context
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Likely ignored
    is_self: bool = True # Always True for self-attention
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Returns Q, K, V tensors

    B, T, E = q.shape
    q_out, k_out, v_out = attn_data.in_proj(q, k, v)

    assert q_out.shape == (B, T, E)
    assert k_out.shape == (B, T, E)
    assert v_out.shape == (B, T, E)

    H_q = attn_data.nheads
    H_kv = attn_data.kvheads
    D_kq = attn_data.emb_kq_per_head
    D_v = attn_data.emb_v_per_head

    queries = q_out.view(B, T, H_q, D_kq)
    keys    = k_out.view(B, T, H_kv, D_kq)
    values  = v_out.view(B, T, H_kv, D_v)

    assert queries.shape == (B, T, H_q, D_kq)
    assert keys.shape == (B, T, H_kv, D_kq)
    assert values.shape == (B, T, H_kv, D_v)

    if attn_data.position_encoder is not None and T > 0:
        assert position_ids is not None
        expected_pos_shape = (B, T)
        if not (T == 0 and position_ids.shape == (B, 0)) and position_ids.shape != expected_pos_shape:
                raise AssertionError(f"Expected position_ids shape {expected_pos_shape}, got {position_ids.shape}")

        valid_mask = position_ids != -1
        if valid_mask.any():
            max_pos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
            position_ids_safe = position_ids.clone()
            position_ids_safe[valid_mask] = position_ids_safe[valid_mask].clamp(0, max_pos - 1)

            queries_rope, keys_rope = attn_data.position_encoder.adjusted_qk(
                queries, keys, position_ids_safe
            )

            if valid_mask.all():
                queries = queries_rope
                keys = keys_rope
            else:
                mask_qkv = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(queries)
                queries = torch.where(mask_qkv, queries_rope, queries)
                keys    = torch.where(mask_qkv.expand_as(keys), keys_rope, keys)

    return (
        queries.permute(0, 2, 1, 3),
        keys.permute(0, 2, 1, 3),
        values.permute(0, 2, 1, 3)
    )



def _pad_to_block(t: torch.Tensor, target_len: int, dim: int = 2) -> torch.Tensor:
    """Pads tensor along a dimension to target_len."""
    length = t.size(dim)
    if length >= target_len:
        # Assert if already too large, should not happen if target_len is block_size and input is shard
        assert length == target_len, f"Tensor length {length} along dim {dim} is >= target {target_len} and not equal. Tensor shape: {t.shape}"
        return t
    
    # Normalize dim to be positive for the check
    actual_dim = dim
    if actual_dim < 0:
        actual_dim += t.ndim

    pad_shape = list(t.shape)
    # Ensure the dimension to pad is valid
    if not (0 <= actual_dim < len(pad_shape)):
        raise IndexError(f"Dimension {dim} (normalized to {actual_dim}) is out of bounds for tensor with shape {t.shape}")
    pad_shape[actual_dim] = target_len - length # Use actual_dim for modifying pad_shape
    pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim) # torch.cat handles original dim correctly


class RingAttentionHelper:
    """
    Helper class to perform the distributed Ring Attention computation within a block.
    It handles the two-pass algorithm for stable softmax calculation in a distributed
    manner, including communication of K/V blocks around the ring.
    The `forward` method is the main entry point, orchestrating QKV computation,
    RoPE application, and the two-pass attention followed by feed-forward layers.
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

        self.head_dim: int = self.attn.emb_kq_per_head
        # Fix 1: Match Scaling Behavior
        self.scale: float = self.attn.scale_factor or math.sqrt(self.head_dim)

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
        """
        Main forward pass for the RingAttentionHelper.
        Args:
            x_norm: Local normalized input shard, padded to block_size.
            strategy: The RingAttentionStrategy instance.
            mask: Global attention mask.
            position_ids: Global position_ids.
            valid_len: The number of actual (non-padding) tokens in x_norm for this rank.
            residual: The original input tensor (before normalization) for the residual connection.
        """
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
        # These will be of shape (B, H, valid_len, D_head) as inputs were trimmed
        q_local_unpadded, k_local_unpadded, v_local_unpadded = compute_local_qkv_and_rope(
            self.attn, # Pass self.attn
            q=x_norm_for_rope, k=x_norm_for_rope, v=x_norm_for_rope, # Use trimmed inputs
            position_ids=position_ids_for_rope, # Pass trimmed position_ids (RoPE applied here)
            use_cache=False,
            past_key_value_state=None,
            is_self=True
        )
        # q_local_unpadded, k_local_unpadded, v_local_unpadded are (B, H_orig_kv, valid_len, D_head/D_v)
        # Note: compute_local_qkv_and_rope already permutes to (B, H, S, D)

        # Fix 4: Ensure Matching KV Expansion
        # This needs to happen *after* RoPE and *before* passing to attention computation logic
        # if the original K/V heads are different from Q heads.
        kv_expansion = self.attn.nheads // self.attn.kvheads
        if kv_expansion != 1:
            # k_local_unpadded and v_local_unpadded are (B, H_kv, valid_len, D_head/D_v)
            # We need to expand H_kv to H_q for the attention computation if it's GQA/MQA
            k_local_unpadded = k_local_unpadded.unsqueeze(2).expand(-1, -1, kv_expansion, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_kq_per_head)
            v_local_unpadded = v_local_unpadded.unsqueeze(2).expand(-1, -1, kv_expansion, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_v_per_head)

        B, H, T_qkv_local, D_head = q_local_unpadded.shape
        assert T_qkv_local == valid_len, f"Rank {self.rank}: QKV length {T_qkv_local} after compute_local_qkv_and_rope does not match valid_len {valid_len}"

        # QKV from compute_local_qkv_and_rope are already effectively valid_len long
        # because inputs to it were trimmed.
        q_local = q_local_unpadded
        k_local = k_local_unpadded
        v_local = v_local_unpadded

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

        # Cache and extra output are None because Ring Attention, in this implementation,
        # does not support KV caching between forward passes (it's designed for training or full recomputation).
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
        """
        Performs the core two-pass ring attention computation (max_score and sums passes)
        and applies post-attention layers (dense, residual, FF norm, FF, residual).
        Operates on the 'valid_len' portion of the local shard.
        Args:
            q_local, k_local, v_local: Local Q, K, V tensors for valid tokens.
            mask_global: Global attention mask.
            valid_len: Actual length of local Q, K, V.
            x_block: Residual connection slice (original input for this block).
            x_norm_block: Normalized input slice (used for QKV computation).
            q_start_global: Global start index for the Q block.
        Returns:
            Processed tensor for valid local tokens.
        """

        B, H, T_q_local, D_head = q_local.shape
        D_v = v_local.shape[-1]
        device = q_local.device
        # Rank is available via self.rank

        # Fix 6: Numerical Precision Control
        compute_dtype = q_local.dtype if q_local.dtype in [torch.float32, torch.bfloat16, torch.float16] else torch.float32
        q_compute = q_local.to(compute_dtype)
        k_compute = k_local.to(compute_dtype)
        v_compute = v_local.to(compute_dtype)

        if self.rank == 0: # Assuming llama_block.layer_idx is not directly available here, print based on rank
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): q_compute norm = {torch.linalg.norm(q_compute.float()).item()}")
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): k_compute norm = {torch.linalg.norm(k_compute.float()).item()}")

        # Compute max scores (Pass 1)
        max_score = self._compute_max_score_pass(
            q_compute,
            k_compute,
            mask_global,
            q_start_global,
            valid_len,
        )
        if self.rank == 0:
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): max_score norm = {torch.linalg.norm(max_score.float()).item()}")

        # Fix 7: Softmax Implementation (Option to use F.softmax)
        # If choosing to use F.softmax for closer matching, the two-pass logic changes.
        # For now, let's keep the two-pass for numerical stability but ensure dtypes are handled.
        # If you want to switch, you'd compute all scores, then apply F.softmax.
        # The current two-pass is generally more robust for distributed settings.
        # We will apply the dtype change to the existing two-pass.

        # Compute sums (Pass 2) using compute_dtype
        numerator, denominator = self._compute_sums_pass(
            q_compute, k_compute, v_compute,
            mask_global,
            q_start_global,
            valid_len,
            max_score, # max_score is already in compute_dtype
        )
        if self.rank == 0:
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): numerator norm = {torch.linalg.norm(numerator.float()).item()}")
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): denominator norm = {torch.linalg.norm(denominator.float()).item()}")

        # Compute attention output in compute_dtype
        attn_out_h = numerator / (denominator + 1e-10) # Division in compute_dtype
        attn_out_h = attn_out_h.to(q_local.dtype) # Cast result back to original dtype
        if self.rank == 0:
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): attn_out_h (raw) norm = {torch.linalg.norm(attn_out_h.float()).item()}")

        # Fix 2: Match Dropout Behavior
        if self.attn.p_dropout and self.llama_block.training: # Check training mode of the parent LLaMABlock
            attn_out_h = F.dropout(attn_out_h, p=self.attn.p_dropout, training=True)

        # Assert shape after attention calculation
        assert attn_out_h.shape == (B, H, T_q_local, D_v), f"Rank {self.rank}: Attn head output shape mismatch: {attn_out_h.shape} vs {(B, H, T_q_local, D_v)}"

        # Reshape and apply dense layer
        attn_out = attn_out_h.transpose(1, 2).contiguous().view(B, T_q_local, H * D_v)
        attn_out = self.attn.dense(attn_out)
        if self.rank == 0:
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): attn_out after dense norm = {torch.linalg.norm(attn_out.float()).item()}")

        # First residual connection
        residual_1 = x_block + attn_out # x_block is residual_local_valid (B, T_q_local, E)
        if self.rank == 0:
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): residual_1 (after attn) norm = {torch.linalg.norm(residual_1.float()).item()}")

        # Apply FF norm and FF
        ff_ln_out = self.ff_norm(residual_1)
        ff_out = self.ff(ff_ln_out)
        if self.rank == 0:
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): ff_out (raw) norm = {torch.linalg.norm(ff_out.float()).item()}")

        # Second residual connection
        x = ff_out + residual_1
        if self.rank == 0:
            print(f"DEBUG (RingAttentionHelper.forward_full, Rank {self.rank}): x final (RingHelper output for valid_len) norm = {torch.linalg.norm(x.float()).item()}")


        # Return processed valid local tokens
        return x

    def _compute_max_score_pass(
        self,
        q_compute: torch.Tensor, # Q (compute_dtype) (B, H, T_q_local, D_head)
        k_compute: torch.Tensor, # K (compute_dtype) (B, H, T_q_local, D_head) - H should match Q after expansion
        mask_global: Optional[torch.Tensor], # Global mask
        q_start_global: int, # Global start index for Q block
        valid_len_local: int, # Actual length of local Q
    ) -> torch.Tensor: # Returns max scores (compute_dtype) (B, H, T_q_local, 1)
        """
        Pass 1 of Ring Attention: Compute maximum attention scores.
        Iteratively shifts K blocks around the ring and computes partial max scores.
        Args:
            q_compute: Local Q tensor (compute_dtype).
            k_compute: Local K tensor (compute_dtype), already expanded if GQA/MQA.
            mask_global: Global attention mask.
            q_start_global: Global start index for the Q block.
            valid_len_local: Actual length of the local Q tensor.
        Returns:
            Maximum scores tensor (compute_dtype).
        """

        B, H, T_q_local, _ = q_compute.shape
        device = q_compute.device
        dtype = q_compute.dtype # Use compute_dtype

        max_score = torch.full((B, H, T_q_local, 1), -torch.inf, device=device, dtype=dtype) # Use torch.inf

        # Indices for the local Q block in the global sequence
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_compute # Use the correctly typed variable
        current_k_len = k_compute.shape[2]

        for i in range(self.world_size):
            # Global start index for the current K block in the ring
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            # Indices for the current K block in the global sequence
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)

            # Slice global mask relevant to current Q block and current K block
            if mask_global is not None:
                current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len]
                assert current_mask.shape[-2] == T_q_local and current_mask.shape[-1] == current_k_len, \
                    f"Rank {self.rank}: Sliced mask shape {current_mask.shape} mismatch with Q_len {T_q_local}, K_len {current_k_len}"
            else:
                current_mask = None

            # Only compute scores if the current K block has elements
            if current_k_len > 0:
                 scores = self._compute_attention_scores(
                    q_compute,
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
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.block_size, current_k_len)

        return max_score

    def _compute_sums_pass(
        self,
        q_compute: torch.Tensor, # Q (compute_dtype) (B, H, T_q_local, D_head)
        k_compute: torch.Tensor, # K (compute_dtype) (B, H, T_q_local, D_head)
        v_compute: torch.Tensor, # V (compute_dtype) (B, H, T_q_local, D_v)
        mask_global: Optional[torch.Tensor], # Global mask
        q_start_global: int, # Global start index for Q block
        valid_len_local: int, # Actual length of local Q, K, V
        max_score: torch.Tensor, # Max scores from Pass 1 (compute_dtype) (B, H, T_q_local, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns numerator, denominator (compute_dtype)
        """
        Pass 2 of Ring Attention: Compute numerator and denominator for softmax.
        Iteratively shifts K and V blocks around the ring.
        Args:
            q_compute: Local Q tensor (compute_dtype).
            k_compute: Local K tensor (compute_dtype).
            v_compute: Local V tensor (compute_dtype).
            mask_global: Global attention mask.
            q_start_global: Global start index for the Q block.
            valid_len_local: Actual length of local Q, K, V.
            max_score: Max scores from Pass 1 (compute_dtype).
        Returns:
            Tuple of (numerator, denominator) tensors (compute_dtype).
        """

        B, H, T_q_local, D_head = q_compute.shape
        D_v = v_compute.shape[-1]
        device = q_compute.device
        dtype = q_compute.dtype # Accumulation dtype

        numerator = torch.zeros(B, H, T_q_local, D_v, device=device, dtype=dtype)
        denominator = torch.zeros(B, H, T_q_local, 1, device=device, dtype=dtype)

        # Indices for the local Q block in the global sequence
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_compute
        current_v_block = v_compute
        current_k_len = k_compute.shape[2] # K and V should have the same sequence length dimension

        for i in range(self.world_size):
            # Global start index for the current K/V block in the ring
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            # Indices for the current K block in the global sequence
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)

             # Slice global mask relevant to current Q block and current K block
            if mask_global is not None:
                current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len]
                assert current_mask.shape[-2] == T_q_local and current_mask.shape[-1] == current_k_len, \
                    f"Rank {self.rank}: Sliced mask shape {current_mask.shape} mismatch with Q_len {T_q_local}, K_len {current_k_len} in sums pass"
            else:
                current_mask = None

            # Only compute scores if the current K block has elements
            if current_k_len > 0:
                scores = self._compute_attention_scores(
                    q_compute,
                    current_k_block[:, :, :current_k_len, :], # Use sliced K block
                    q_indices_global,
                    k_indices_global,
                    current_mask,
                    apply_mask=True, # Apply mask including causal
                    keep_causal=True
                )

                # Compute stable exponentiated scores
                stable_scores = scores - max_score # Use max_score from Pass 1
                exp_scores = torch.exp(stable_scores.clamp(min=EXP_CLAMP_MIN, max=EXP_CLAMP_MAX)) # Clamp exp input for safety

                # Update numerator and denominator
                numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, current_v_block[:, :, :current_k_len, :]) # Use sliced V block
                denominator += exp_scores.sum(dim=-1, keepdim=True)

            # Ring shift K and V blocks
            if i < self.world_size - 1:
                # Store the valid length of the K/V pair *before* K is shifted.
                # This is the length that both current_k_block and current_v_block have.
                valid_len_of_current_kv_to_send = current_k_len

                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.block_size, valid_len_of_current_kv_to_send)
                # Now current_k_len is the length of the newly received K block.
                # We must send the *old* V block, which had length valid_len_of_current_kv_to_send.
                current_v_block, _ = self._ring_shift_tensor(current_v_block, self.block_size, valid_len_of_current_kv_to_send)
        return numerator, denominator

    def _compute_attention_scores(
        self,
        q: torch.Tensor, # Q tensor (compute_dtype) (B, H, T_q_local, D_head)
        k: torch.Tensor, # K tensor (compute_dtype) (B, H, T_k_block, D_head) - H should match Q
        q_indices_global: torch.Tensor, # Global indices for Q (T_q_local)
        k_indices_global: torch.Tensor, # Global indices for K block (T_k_block)
        mask: Optional[torch.Tensor] = None, # Sliced global mask (B, 1, T_q_local, T_k_block)
        apply_mask: bool = True,
        keep_causal: bool = True # Apply causal mask based on global indices
    ) -> torch.Tensor: # Returns scores (compute_dtype) (B, H, T_q_local, T_k_block)
        """
        Computes attention scores (Q @ K.T / sqrt(D_head)) and applies masks.
        Args:
            q: Q tensor (float32).
            k: K tensor (float32).
            q_indices_global: Global indices for Q.
            k_indices_global: Global indices for K.
            mask: Sliced global mask (padding/attention mask).
            apply_mask: Whether to apply the `mask`.
            keep_causal: Whether to apply the causal mask based on global indices.
        Returns:
            Attention scores tensor (compute_dtype).
        """
        B, H, T_q_local, D_head_q = q.shape
        _, _, T_k_block, D_head_k = k.shape

        # Compute QK^T scaled by head_dim
        # Ensure K is broadcastable if H != H_kv (GQA)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # scores are compute_dtype

        # Apply masks
        if apply_mask:
             if mask is not None:
                expected_mask_shape_full = (B, H, T_q_local, T_k_block)
                expected_mask_shape_broadcast_H = (B, 1, T_q_local, T_k_block)
                assert mask.shape == expected_mask_shape_full or mask.shape == expected_mask_shape_broadcast_H, \
                    f"Rank {self.rank}: Mask shape {mask.shape} is not compatible with scores shape {scores.shape}. Expected {expected_mask_shape_full} or {expected_mask_shape_broadcast_H}."
                scores = scores + mask.to(scores.dtype) # Apply padding/attention mask, ensure mask is broadcastable


             if keep_causal:
                # Apply causal mask based on global indices
                # Ensure mask is on the same device and boolean
                causal_mask_bool = (k_indices_global.to(device=q_indices_global.device)[None, None, None, :] > q_indices_global[None, None, :, None])
                # Fix 5: Verify Mask Application Consistency (using -torch.inf for float types)
                # The mask from SDPA is often additive (0 for keep, -inf for mask).
                # .masked_fill uses a boolean mask.
                scores = scores.masked_fill(causal_mask_bool, -torch.inf) # Use -torch.inf for float32


        # Clamp scores for stability before softmax calculation
        # This clamp value should be related to the stable softmax implementation's exp clamping
        # Clamp scores in _update_totals uses exp(scores - max_score). The value here prevents
        # overflow if scores themselves are huge before subtracting max_score, although
        # standard practice often omits this first clamp. Let's keep it conservative.
        scores = scores.clamp(min=torch.finfo(scores.dtype).min) # Prevent -inf becoming NaN if other infs are present

        return scores

    def _update_max_score(
        self,
        scores: torch.Tensor, # Scores from _compute_attention_scores (compute_dtype) (B, H, T_q_local, T_k_block)
        current_max: torch.Tensor # Current max score (compute_dtype) (B, H, T_q_local, 1)
    ) -> torch.Tensor: # Returns updated max score (compute_dtype) (B, H, T_q_local, 1)

        # Compute max for the current block of scores
        # Replace -inf with a very small number before max to avoid issues if all scores are -inf
        block_max = scores.masked_fill(scores == -torch.inf, torch.finfo(scores.dtype).min).amax(dim=-1, keepdim=True)

        # Update the maximum score seen so far
        max_score = torch.maximum(current_max, block_max)

        return max_score

    def _update_totals(
        self,
        scores: torch.Tensor, # Scores from _compute_attention_scores (compute_dtype) (B, H, T_q_local, T_k_block)
        v: torch.Tensor, # V block (compute_dtype) (B, H, T_k_block, D_v) - H should match Q
        max_score: torch.Tensor, # Max score from Pass 1 (compute_dtype) (B, H, T_q_local, 1)
        numerator: torch.Tensor, # Current numerator sum (compute_dtype) (B, H, T_q_local, D_v)
        denominator: torch.Tensor # Current denominator sum (compute_dtype) (B, H, T_q_local, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns updated numerator, denominator

        # Compute stable exponentiated scores using max_score from Pass 1
        stable_scores = scores - max_score # Shape (B, H, T_q_local, T_k_block)
        exp_scores = torch.exp(stable_scores.clamp(min=EXP_CLAMP_MIN, max=EXP_CLAMP_MAX)) # Clamp exp input for stability

        # Update numerator (sum of exp(score - max) * V)
        # einsum handles broadcasting for GQA if H != H_kv
        numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v)

        # Update denominator (sum of exp(score - max))
        denominator += exp_scores.sum(dim=-1, keepdim=True)

        return numerator, denominator


    def _ring_shift_tensor(
        self,
        tensor: torch.Tensor, # Tensor to shift (e.g., K or V block) (B, H, T_block, D) padded to block_size
        pad_len: int, # Expected padded length (self.block_size)
        valid_len_to_send: int # The actual number of valid elements in the tensor being sent
    ) -> Tuple[torch.Tensor, int]: # Returns received tensor and its valid length
        """
        Shifts a tensor block to the next rank in the ring and receives a block from the previous rank.
        Handles padding for sending and slicing after receiving.
        Args:
            tensor: The tensor block to send (e.g., K or V). This tensor should contain `valid_len_to_send`
                    actual data elements along its sequence dimension, and the rest can be padding.
            pad_len: The target length to pad the tensor to for communication (self.block_size).
            valid_len_to_send: The number of valid (non-padding) elements in the `tensor` to be sent.
        Returns:
            A tuple containing the received tensor (sliced to its valid length) and its valid length.
        """

        rank, world = self.rank, self.world_size
        send_rank = (rank + 1) % world
        recv_rank = (rank - 1 + world) % world

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
        # input `tensor` is a block of K or V (B, H, T_input, D) where T_input is the current length of this block,
        # which might already be padded or might be exactly valid_len_to_send.
        # We need to pad it to `pad_len` (block_size) for sending.
        # The `valid_len_to_send` argument explicitly tells us how many elements are valid.

        assert tensor.shape[-2] >= valid_len_to_send, \
            f"Rank {self.rank}: Tensor sequence length {tensor.shape[-2]} is less than valid_len_to_send {valid_len_to_send}"

        # Slice to valid_len_to_send before padding, if tensor is larger (e.g. if it was a previously received padded block)
        tensor_to_send = tensor[:, :, :valid_len_to_send, :]

        # Pad the current tensor block to block_size for communication
        padded_for_send = _pad_to_block(tensor_to_send, pad_len, dim=-2).contiguous()
        assert padded_for_send.shape[-2] == pad_len, \
            f"Rank {self.rank}: Padding for send failed. Expected dim -2 to be {pad_len}, got {padded_for_send.shape[-2]}. Original tensor shape: {tensor.shape}, valid_len_to_send: {valid_len_to_send}"

        device = tensor.device

        # Create buffers for receiving
        recv_len_tensor = torch.empty(1, dtype=torch.int32, device=device)
        tensor_recv_padded = torch.empty_like(padded_for_send) # Received padded tensor
        assert tensor_recv_padded.shape == padded_for_send.shape, \
            f"Rank {self.rank}: Shape mismatch between send buffer {padded_for_send.shape} and recv buffer {tensor_recv_padded.shape}"

        # Prepare send/recv ops
        ops = [
            # Send the actual number of valid elements in the tensor being sent
            P2POp(op=dist.isend, tensor=torch.tensor([valid_len_to_send], dtype=torch.int32, device=device), peer=send_rank),
            # Receive the valid length of the tensor from the previous rank
            P2POp(op=dist.irecv, tensor=recv_len_tensor, peer=recv_rank),
            # Send the padded tensor block
            P2POp(op=dist.isend, tensor=padded_for_send, peer=send_rank),
            # Receive the padded tensor block
            P2POp(op=dist.irecv, tensor=tensor_recv_padded, peer=recv_rank)
        ]

        # Execute async, then wait
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
            # Consider adding a timeout to req.wait(timeout_sec) if the API supports it,
            # or wrap in a try-except for dist.DistError for more robust error handling.
            # For now, relying on default error propagation.
            # Example:
            # try:
            #     req.wait(timeout=datetime.timedelta(seconds=30)) # Requires appropriate timeout value
            # except dist.DistError as e:
            #     raise RuntimeError(f"Rank {self.rank}: Distributed operation failed: {e}") from e

        recv_len = recv_len_tensor.item() # The valid length of the received block

        # Assert received length makes sense
        assert recv_len >= 0 and recv_len <= pad_len, f"Rank {self.rank}: Received invalid length: {recv_len}, expected <= {pad_len}"
        # Slice the received padded tensor to its actual valid length
        tensor_recv = tensor_recv_padded[:, :, :recv_len, :].contiguous()
        assert tensor_recv.shape[-2] == recv_len, f"Rank {self.rank}: Slicing received tensor failed. Expected dim -2 to be {recv_len}, got {tensor_recv.shape[-2]}. Received padded shape: {tensor_recv_padded.shape}"
        # Return the received tensor (sliced) and its valid length
        return tensor_recv, recv_len