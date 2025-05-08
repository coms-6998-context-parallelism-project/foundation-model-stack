from typing import Any, List, Mapping, Optional, Tuple, Union # Keep necessary types

from fms.modules.attention import MultiHeadAttention
import torch
import torch.nn as nn
import torch.distributed as dist

from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy # Need both for type hints
from typing import Any, List, Mapping, Optional, Tuple, Union # Keep necessary types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor # Explicitly import Tensor for type hinting
from torch.distributed import P2POp

import math


class RingAttentionKernel:
    @staticmethod
    def _get_accum_dtype(dtype: torch.dtype) -> torch.dtype:
        """Pick float32 for unsupported dtypes, else preserve fp16/bf16/f32"""
        return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

    @staticmethod
    def _get_clamp_bounds(accum_dtype: torch.dtype) -> Tuple[float, float]:
        """Compute (log_min+margin, log_max−margin) for stable exp()"""
        info = torch.finfo(accum_dtype)
        margin = 2.0  # As used in fms_extras
        return math.log(info.tiny) + margin, math.log(info.max) - margin

    @staticmethod
    def _pad_to_block(t: Tensor, target_len: int, dim: int) -> Tensor:
        """Zero‑pad any tensor along dim up to target_len"""
        current_len = t.size(dim)
        if current_len >= target_len:
            # If current length is greater, it might indicate an issue or need for slicing.
            # For strict padding, we assert it's not greater.
            assert current_len == target_len, f"Tensor dim {dim} (size {current_len}) already meets or exceeds target_len ({target_len}). Padding not applied or slicing needed."
            return t
        
        pad_shape = list(t.shape)
        pad_shape[dim] = target_len - current_len
        pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=dim)

    @staticmethod
    def _compute_qkv_and_rope(
        attn_module: MultiHeadAttention,
        x: Tensor,  # Expected to be trimmed input (B, T_valid, E)
        position_ids: Optional[Tensor]  # Expected to be trimmed (B, T_valid) or None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """in_proj → reshape heads → apply RoPE only on valid positions"""
        B, T_valid, E = x.shape
        # Assuming x is for q, k, v generation. If k,v are different, attn_module.in_proj handles it.
        q_out, k_out, v_out = attn_module.in_proj(x, None, None)

        Hq, Hkv = attn_module.nheads, attn_module.kvheads
        Dk, Dv = attn_module.emb_kq_per_head, attn_module.emb_v_per_head

        Q = q_out.view(B, T_valid, Hq, Dk)
        K = k_out.view(B, T_valid, Hkv, Dk)
        V = v_out.view(B, T_valid, Hkv, Dv)

        if attn_module.position_encoder and T_valid > 0:
            assert position_ids is not None, "position_ids must be provided for RoPE if T_valid > 0"
            
            valid_pos_mask = position_ids.ne(-1) # Check for sentinel values like -1
            if valid_pos_mask.any():
                clamped_position_ids = position_ids.clone()
                
                # Determine max_pos for RoPE clamping
                # Default to a common value if specific attributes are not found or are complex types.
                rope_max_len_default = 2048 # A sensible default
                max_pos_to_use = rope_max_len_default

                if hasattr(attn_module.position_encoder, 'max_seq_len'):
                    max_pos = attn_module.position_encoder.max_seq_len
                    if isinstance(max_pos, int):
                        max_pos_to_use = max_pos
                elif hasattr(attn_module.position_encoder, 'max_seq_len_cached') and isinstance(attn_module.position_encoder.max_seq_len_cached, dict):
                    # If max_seq_len_cached is a dict, try to get value for current device, or use a default.
                    # This part might need refinement based on how max_seq_len_cached is intended to be used here.
                    # For simplicity, we'll prefer max_seq_len if it's an int.
                    pass # Fallback to rope_max_len_default or max_seq_len if it was an int

                # Clamp only valid positions to be within the RoPE's expected range [0, max_pos-1]
                clamped_position_ids[valid_pos_mask] = clamped_position_ids[valid_pos_mask].clamp(min=0, max=max_pos_to_use - 1)
                
                # Apply RoPE. adjusted_qk typically expects (B, T, H, D) and (B, T) for pos_ids.
                q_rope, k_rope = attn_module.position_encoder.adjusted_qk(Q, K, clamped_position_ids)
                
                # Apply RoPE only where positions are valid (not -1)
                # Mask shape needs to be (B, T_valid, 1, 1) to broadcast with Q/K (B, T_valid, H, Dk)
                rope_apply_mask = valid_pos_mask.unsqueeze(-1).unsqueeze(-1)
                Q = torch.where(rope_apply_mask.expand_as(Q), q_rope, Q)
                K = torch.where(rope_apply_mask.expand_as(K), k_rope, K)

        # Permute to (B, H, T_valid, D)
        Q_permuted = Q.permute(0, 2, 1, 3)
        K_permuted = K.permute(0, 2, 1, 3)
        V_permuted = V.permute(0, 2, 1, 3)

        # GQA/MQA expansion for K, V
        if attn_module.nheads != attn_module.kvheads:
            num_key_value_groups = attn_module.nheads // attn_module.kvheads
            K_permuted = K_permuted.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1, -1).reshape(B, attn_module.nheads, T_valid, Dk)
            V_permuted = V_permuted.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1, -1).reshape(B, attn_module.nheads, T_valid, Dv)

        return Q_permuted, K_permuted, V_permuted

    @staticmethod
    def _ring_shift(
        tensor: Tensor, # (B, H, SeqLen, D)
        strategy: RingAttentionStrategy,
        current_valid_len: int  # valid length of the tensor being sent along SeqLen dimension
    ) -> Tuple[Tensor, int]:  # returns (next_tensor_trimmed, next_valid_len)
        """Communicate one KV block around the ring"""
        rank, world_size = strategy.rank, strategy.world_size
        block_size = strategy.block_size # Padded length for communication

        send_peer = (rank + 1) % world_size
        recv_peer = (rank - 1 + world_size) % world_size
        device = tensor.device
        
        # Tensor sequence dimension is dim 2 for (B, H, SeqLen, D)
        seq_dim = 2

        if current_valid_len == 0:
            # Create an empty tensor with 0 sequence length but correct other dims
            empty_payload_shape = list(tensor.shape)
            empty_payload_shape[seq_dim] = 0
            send_tensor_payload = torch.empty(*empty_payload_shape, dtype=tensor.dtype, device=device)
        else:
            # Slice the valid part of the tensor
            slicing_indices = [slice(None)] * tensor.ndim
            slicing_indices[seq_dim] = slice(0, current_valid_len)
            send_tensor_payload = tensor[tuple(slicing_indices)]

        # Pad the payload to block_size for sending
        send_tensor_padded = RingAttentionKernel._pad_to_block(send_tensor_payload, block_size, dim=seq_dim)
        send_tensor_padded = send_tensor_padded.contiguous() # P2P ops require contiguous tensors

        recv_tensor_padded = torch.empty_like(send_tensor_padded) # Receive into a similarly padded tensor
        
        recv_len_tensor = torch.empty(1, dtype=torch.int32, device=device)
        sent_len_tensor = torch.tensor([current_valid_len], dtype=torch.int32, device=device)

        ops = [
            P2POp(dist.isend, sent_len_tensor, peer=send_peer),
            P2POp(dist.irecv, recv_len_tensor, peer=recv_peer),
            P2POp(dist.isend, send_tensor_padded, peer=send_peer),
            P2POp(dist.irecv, recv_tensor_padded, peer=recv_peer),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        next_valid_len = recv_len_tensor.item()
        assert 0 <= next_valid_len <= block_size, f"Rank {rank}: invalid recv_len {next_valid_len} (block_size {block_size})"

        # Return the received tensor, trimmed to its actual valid length
        slicing_indices_recv = [slice(None)] * recv_tensor_padded.ndim
        slicing_indices_recv[seq_dim] = slice(0, next_valid_len)
        return recv_tensor_padded[tuple(slicing_indices_recv)].contiguous(), next_valid_len

    @staticmethod
    def _compute_attention_scores(
        q: Tensor, k: Tensor,  # q: (B,H,Tq,Dk), k: (B,H,Tk,Dk)
        q_indices_global: Tensor,  # (Tq) global indices for queries
        k_indices_global: Tensor,  # (Tk) global indices for keys of current block
        scale: float,
        mask: Optional[Tensor] = None,  # Pre-sliced mask, broadcastable to (B,H,Tq,Tk)
        is_causal_mask: bool = False
    ) -> Tensor:
        """Compute scaled dot‑product scores with mask/causal logic"""
        B, H, Tq, Dk_q = q.shape
        _B_k, _H_k, Tk, _Dk_k = k.shape

        if Tq == 0 or Tk == 0:  # No queries or no keys in the current block
            return torch.empty(B, H, Tq, Tk, device=q.device, dtype=q.dtype)

        scores = torch.matmul(q / scale, k.transpose(-2, -1))  # (B, H, Tq, Tk)

        if mask is not None:
            # Mask addition handles broadcasting (e.g., if mask is (B,1,Tq,Tk) and scores (B,H,Tq,Tk))
            scores = scores + mask.to(scores.dtype)

        if is_causal_mask:
            # q_indices_global: (Tq), k_indices_global: (Tk)
            # causal_condition: (Tq, Tk) where True means k is after q
            causal_condition = k_indices_global.unsqueeze(0) > q_indices_global.unsqueeze(1)
            # Expand to (1, 1, Tq, Tk) for broadcasting with scores (B, H, Tq, Tk)
            final_causal_mask_shape = causal_condition.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(final_causal_mask_shape, float('-inf'))
        return scores

    @staticmethod
    def _compute_max_pass(
        q_local: Tensor, k_local_initial: Tensor,  # local q (B,H,Tq_local,Dk), initial local k (B,H,Tk_local_init,Dk)
        mask_global: Optional[Tensor],  # Full mask (e.g. B, H, SeqMax, SeqMax) or (B, 1, SeqMax, SeqMax)
        q_start_global: int,  # Global start index for queries of this rank
        local_query_len: int,  # Tq_local: number of valid queries on this rank
        strategy: RingAttentionStrategy,
        scale: float,
        is_causal_mask_flag: bool
    ) -> Tensor:  # Returns max_scores (B, H, local_query_len, 1)
        """First ring pass: find per‑query global max score"""
        B, H, Tq_local, Dk = q_local.shape
        dev = q_local.device
        # Max usually done in same or higher precision as q/k for stability before exp
        accum_dtype = RingAttentionKernel._get_accum_dtype(q_local.dtype) 

        max_score_accum = torch.full((B, H, Tq_local, 1), torch.finfo(accum_dtype).min, device=dev, dtype=accum_dtype)

        if Tq_local == 0:  # No local queries, return min-filled tensor
            return max_score_accum

        q_indices_global_for_block = torch.arange(q_start_global, q_start_global + Tq_local, device=dev)

        # Ensure k_local_initial is on accum_dtype before the loop
        current_k_block = k_local_initial.to(accum_dtype) 
        current_k_block_valid_len = k_local_initial.shape[2] # K sequence length is dim 2

        for i in range(strategy.world_size):
            k_block_origin_rank = (strategy.rank - i + strategy.world_size) % strategy.world_size
            k_start_global_for_block = k_block_origin_rank * strategy.block_size

            if current_k_block_valid_len > 0:
                k_indices_global_for_block = torch.arange(
                    k_start_global_for_block,
                    k_start_global_for_block + current_k_block_valid_len,
                    device=dev
                )

                local_mask_slice = None
                if mask_global is not None:
                    # Slice the global mask for the current Q-K interaction
                    # q_slice covers [q_start_global, q_start_global + Tq_local)
                    # k_slice covers [k_start_global_for_block, k_start_global_for_block + current_k_block_valid_len)
                    q_slice_indices = slice(q_start_global, q_start_global + Tq_local)
                    k_slice_indices = slice(k_start_global_for_block, k_start_global_for_block + current_k_block_valid_len)
                    # mask_global expected shape (B, H_mask, SeqLen_Global, SeqLen_Global)
                    # H_mask can be 1 (for broadcasting) or H.
                    local_mask_slice = mask_global[:, :, q_slice_indices, k_slice_indices]
                
                s = RingAttentionKernel._compute_attention_scores(
                    q_local.to(accum_dtype),
                    current_k_block, # This is now guaranteed to be on accum_dtype
                    q_indices_global_for_block,
                    k_indices_global_for_block,
                    scale,
                    mask=local_mask_slice,
                    is_causal_mask=is_causal_mask_flag
                )  # s is (B, H, Tq_local, current_k_block_valid_len)
                
                if s.numel() > 0: # Ensure scores tensor is not empty before amax
                    max_score_accum = torch.maximum(max_score_accum, s.amax(dim=-1, keepdim=True))

            if i < strategy.world_size - 1: # Don't shift on the last iteration
                current_k_block, current_k_block_valid_len = RingAttentionKernel._ring_shift(
                    current_k_block, strategy, current_k_block_valid_len
                )
        return max_score_accum

    @staticmethod
    def _compute_sum_pass(
        q_local: Tensor, k_local_initial: Tensor, v_local_initial: Tensor,
        mask_global: Optional[Tensor],
        q_start_global: int,
        local_query_len: int,  # Tq_local
        max_score: Tensor,  # (B, H, local_query_len, 1) from _compute_max_pass
        exp_min: float, exp_max: float,  # Clamping bounds for exp
        strategy: RingAttentionStrategy,
        scale: float,
        accum_dtype: torch.dtype, # Accumulator data type (e.g. float32)
        is_causal_mask_flag: bool
    ) -> Tuple[Tensor, Tensor]:  # Returns (numerator, denominator)
        """Second ring pass: Kahan‑sum numerator & denominator"""
        B, H, Tq_local, Dk = q_local.shape
        _B_v, _H_v, _Tk_v_init, Dv = v_local_initial.shape # Get Dv from v_local_initial

        dev = q_local.device

        numerator = torch.zeros(B, H, Tq_local, Dv, device=dev, dtype=accum_dtype)
        denominator = torch.zeros(B, H, Tq_local, 1, device=dev, dtype=accum_dtype)
        # Kahan summation compensators
        num_comp = torch.zeros_like(numerator)
        den_comp = torch.zeros_like(denominator)

        if Tq_local == 0:
            return numerator, denominator

        q_indices_global_for_block = torch.arange(q_start_global, q_start_global + Tq_local, device=dev)

        current_k_block = k_local_initial.to(accum_dtype)
        current_v_block = v_local_initial.to(accum_dtype) # V also needs to be on accum_dtype for matmul
        current_kv_block_valid_len = k_local_initial.shape[2] # K and V blocks share valid length (dim 2)

        for i in range(strategy.world_size):
            k_block_origin_rank = (strategy.rank - i + strategy.world_size) % strategy.world_size
            k_start_global_for_block = k_block_origin_rank * strategy.block_size

            if current_kv_block_valid_len > 0:
                k_indices_global_for_block = torch.arange(
                    k_start_global_for_block,
                    k_start_global_for_block + current_kv_block_valid_len,
                    device=dev
                )

                local_mask_slice = None
                if mask_global is not None:
                    q_slice_indices = slice(q_start_global, q_start_global + Tq_local)
                    k_slice_indices = slice(k_start_global_for_block, k_start_global_for_block + current_kv_block_valid_len)
                    local_mask_slice = mask_global[:, :, q_slice_indices, k_slice_indices]

                s = RingAttentionKernel._compute_attention_scores(
                    q_local.to(accum_dtype), current_k_block,
                    q_indices_global_for_block, k_indices_global_for_block,
                    scale,
                    mask=local_mask_slice,
                    is_causal_mask=is_causal_mask_flag
                )  # s is (B, H, Tq_local, current_kv_block_valid_len)

                # Softmax part: exp(scores - max_score)
                # Handle case where max_score is -inf to prevent NaN from s(-inf) - max_score(-inf)
                # If max_score is -inf, all scores 's' that contributed to it must also have been -inf.
                # In this case, delta_scores should effectively be -inf (or a very small number for exp).
                # If s is also -inf, s - (-inf) is NaN. If s is finite, s - (-inf) is +inf.
                # A robust way is to ensure that if max_score is -inf, exp_stabilized_scores becomes 0.
                # The existing masked_fill handles this, but let's ensure delta_scores itself is not NaN.
                delta_scores = torch.where(torch.isneginf(max_score), torch.full_like(s, -torch.inf), s - max_score)
                clamped_delta = delta_scores.clamp(min=exp_min, max=exp_max) # Clamp for stable exp
                exp_stabilized_scores = torch.exp(clamped_delta)

                # If max_score was -inf (all original scores were -inf), exp_stabilized_scores should be 0
                exp_stabilized_scores = exp_stabilized_scores.masked_fill(torch.isneginf(max_score.expand_as(s)), 0.0)
                
                # Current V block for matmul: current_v_block is (B,H,current_kv_block_valid_len,Dv)
                # exp_stabilized_scores is (B,H,Tq_local, current_kv_block_valid_len)
                contrib_num = torch.matmul(exp_stabilized_scores, current_v_block)  # (B,H,Tq_local,Dv)
                contrib_den = exp_stabilized_scores.sum(dim=-1, keepdim=True)  # (B,H,Tq_local,1)

                # Kahan summation for numerator
                y_num = contrib_num - num_comp
                t_num = numerator + y_num
                num_comp = (t_num - numerator) - y_num
                numerator = t_num

                # Kahan summation for denominator
                y_den = contrib_den - den_comp
                t_den = denominator + y_den
                den_comp = (t_den - denominator) - y_den
                denominator = t_den
            
            if i < strategy.world_size - 1: # Don't shift on the last iteration
                current_k_block, received_k_len = RingAttentionKernel._ring_shift(
                    current_k_block, strategy, current_kv_block_valid_len
                )
                current_v_block, received_v_len = RingAttentionKernel._ring_shift(
                    current_v_block, strategy, current_kv_block_valid_len # Send V with same original valid len as K
                )
                assert received_k_len == received_v_len, \
                    f"Rank {strategy.rank}: Mismatch in received K ({received_k_len}) and V ({received_v_len}) lengths after ring shift."
                current_kv_block_valid_len = received_k_len
        
        return numerator, denominator

    @staticmethod
    def _three_pass(
        q_local_valid: Tensor, k_local_valid: Tensor, v_local_valid: Tensor, # Q,K,V for the local valid tokens
        mask_global: Optional[Tensor],  # Global attention mask
        strategy: RingAttentionStrategy,
        q_start_global: int,  # Global start index of the first query token on this rank
        local_query_len: int,  # Number of valid query tokens on this rank (q_local_valid.shape[2])
        scale: float,
        accum_dtype: torch.dtype,
        is_causal_mask_flag: bool
    ) -> Tensor:  # Returns attention output for valid tokens (B, H, local_query_len, Dv)
        """Orchestrates max_pass, sum_pass, finalize softmax & projection"""
        
        exp_min, exp_max = RingAttentionKernel._get_clamp_bounds(accum_dtype)

        # Pass 1: Compute max score
        max_scores = RingAttentionKernel._compute_max_pass(
            q_local_valid, k_local_valid, mask_global, q_start_global, local_query_len,
            strategy, scale, is_causal_mask_flag
        )  # (B, H, local_query_len, 1)

        # Pass 2: Compute numerator and denominator sums
        numerator, denominator = RingAttentionKernel._compute_sum_pass(
            q_local_valid, k_local_valid, v_local_valid, mask_global, q_start_global, local_query_len,
            max_scores, exp_min, exp_max, strategy, scale, accum_dtype, is_causal_mask_flag
        )  # num: (B,H,Tq_local,Dv), den: (B,H,Tq_local,1)

        # Pass 3: Finalize softmax (projection is handled outside this _three_pass method)
        if local_query_len == 0:  # No queries, return empty tensor of correct shape
            B, H, _, Dv = v_local_valid.shape # Get Dv from v_local_valid
            return torch.empty(B, H, 0, Dv, device=q_local_valid.device, dtype=q_local_valid.dtype)

        eps = torch.finfo(denominator.dtype).eps  # Use epsilon of accumulator's dtype for stability
        attn_output_valid = (numerator / (denominator + eps)).to(q_local_valid.dtype)  # Cast back to original q input dtype

        return attn_output_valid  # (B, H, local_query_len, Dv)

    @staticmethod
    def forward(
        x_norm: Tensor,  # (B, T_padded_block, E) - Normalized input
        residual: Tensor,  # (B, T_padded_block, E) - Original input before normalization (for residual connection)
        attn_module: MultiHeadAttention,
        ff: nn.Module,  # Feed-forward network (sub-layer)
        ff_norm: nn.Module,  # Normalization for the feed-forward network
        strategy: RingAttentionStrategy,
        valid_len: int,  # Number of valid tokens in x_norm for this rank's block
        mask: Optional[Tensor] = None,  # Global mask, e.g., (B, H, SeqMax, SeqMax) or (B, 1, SeqMax, SeqMax)
        position_ids: Optional[Tensor] = None,  # Global or per-rank block position_ids (B, T_padded_block)
        past_key_value_state: Optional[Tuple[Tensor,Tensor]] = None, # Consumed if applicable, not standard in basic ring
        is_causal_mask: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor,Tensor]]]:  # (output_padded_or_valid, new_cache_for_valid_tokens)
        """ End‑to‑end ring block forward """
        B, T_padded, E = x_norm.shape
        rank = strategy.rank
        world_size = strategy.world_size
        block_size = strategy.block_size
        
        if world_size > 1:
             assert T_padded == block_size, \
                 f"Input x_norm T_padded ({T_padded}) != block_size ({block_size}) for rank {rank} in distributed mode."

        q_start_global = rank * block_size

        # 1. Trim to valid_len
        x_trim = x_norm[:, :valid_len, :]  # (B, valid_len, E)
        # Residual is from *before* x_norm (block input)
        res_trim = residual[:, :valid_len, :] # (B, valid_len, E)

        pos_trim: Optional[Tensor] = None
        if position_ids is not None:
            # Assuming position_ids corresponds to x_norm (is already sharded or global and sliceable by T_padded)
            # If position_ids is (B, T_global_max), it needs careful slicing.
            # Simpler: assume position_ids is (B, T_padded) like x_norm.
            pos_trim = position_ids[:, :valid_len]
        elif valid_len > 0: # Create on-the-fly if not provided and there are valid tokens
            pos_trim_indices = torch.arange(q_start_global, q_start_global + valid_len, device=x_norm.device)
            pos_trim = pos_trim_indices.unsqueeze(0).expand(B, -1) # (B, valid_len)

        # 2. Compute QKV + RoPE for the valid tokens
        # q_valid, k_valid, v_valid are (B, H, valid_len, Dk/Dv)
        if valid_len > 0:
            q_valid, k_valid, v_valid = RingAttentionKernel._compute_qkv_and_rope(
                attn_module, x_trim, pos_trim
            )
        else: # No valid tokens, create empty Q,K,V
            Hq = attn_module.nheads # Assuming GQA expansion happens in _compute_qkv_and_rope, so Hq for all
            Dk, Dv = attn_module.emb_kq_per_head, attn_module.emb_v_per_head
            q_valid = torch.empty(B, Hq, 0, Dk, dtype=x_norm.dtype, device=x_norm.device)
            k_valid = torch.empty(B, Hq, 0, Dk, dtype=x_norm.dtype, device=x_norm.device)
            v_valid = torch.empty(B, Hq, 0, Dv, dtype=x_norm.dtype, device=x_norm.device)
        
        # Cache output: current K and V for this block's valid tokens
        # (k_valid, v_valid) are (B,H,valid_len,D)
        current_rank_cache: Optional[Tuple[Tensor, Tensor]] = (k_valid, v_valid) if valid_len > 0 else None
        
        # `past_key_value_state` is an input. Ring attention typically recomputes K/V or passes full blocks.
        # Standard ring attention as described here doesn't typically use `past_key_value_state` from a previous *global* step
        # to append local QKV. It's more about distributing a long sequence.
        # So, we produce `current_rank_cache` but don't explicitly combine with `past_key_value_state` here.

        # 3. Three-pass softmax across the ring
        scale = attn_module.scale_factor or math.sqrt(attn_module.emb_kq_per_head)
        accum_dtype = RingAttentionKernel._get_accum_dtype(q_valid.dtype)

        attn_output_valid = RingAttentionKernel._three_pass(
            q_valid, k_valid, v_valid, # Use locally computed K,V as the initial blocks for ring exchange
            mask, # Global mask
            strategy,
            q_start_global,
            valid_len, # local_query_len
            scale,
            accum_dtype,
            is_causal_mask
        )  # Returns (B, H, valid_len, Dv)

        # 4. Dense proj + dropout + FFN + residual (all on valid_len tensors)
        if valid_len > 0:
            # Reshape attention output for dense layer: (B, H, valid_len, Dv) -> (B, valid_len, H*Dv)
            attn_output_reshaped = attn_output_valid.transpose(1, 2).contiguous().view(B, valid_len, attn_module.nheads * attn_module.emb_v_per_head)
            
            # Dense projection
            # In FMS MultiHeadAttention, dropout is usually applied *after* this dense projection.
            attn_dense_out = attn_module.dense(attn_output_reshaped) # (B, valid_len, E)
            
            # Dropout (if attn_module includes it, e.g., fms.MHA applies it after dense via self.dropout)
            if hasattr(attn_module, 'dropout') and isinstance(attn_module.dropout, nn.Dropout):
                 attn_dense_out = attn_module.dropout(attn_dense_out)
            
            # First residual connection (x + Attention(LN(x)))
            x_after_attn = res_trim + attn_dense_out # (B, valid_len, E)

            # Feed-forward network part (h + MLP(LN(h)))
            ffn_input = ff_norm(x_after_attn)
            ffn_output = ff(ffn_input)
            
            # Second residual connection
            final_out_valid = x_after_attn + ffn_output # (B, valid_len, E)
        else: # No valid tokens on this rank
            final_out_valid = torch.empty(B, 0, E, dtype=x_norm.dtype, device=x_norm.device)

        # 5. Pad back to block_size if in distributed mode
        if world_size > 1:
            # Pad along sequence dimension (dim 1 for B,T,E)
            output_padded = RingAttentionKernel._pad_to_block(final_out_valid, block_size, dim=1)
        else: # Single GPU, no padding to block_size needed, return valid output directly
            output_padded = final_out_valid
            
        return output_padded, current_rank_cache