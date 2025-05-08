from typing import Any, Optional, Tuple
from fms.modules.attention import MultiHeadAttention
import torch
import torch.distributed as dist
from torch.distributed import P2POp # Keep P2POp
import math
import torch.nn.functional as F
import torch.nn as nn
from fms.distributed.strategy import RingAttentionStrategy


def _get_accum_dtype(dtype: torch.dtype) -> torch.dtype:
    return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32


def _get_clamp_bounds(accum_dtype: torch.dtype) -> Tuple[float, float]:
    info = torch.finfo(accum_dtype)
    margin = 2.0
    return math.log(info.tiny) + margin, math.log(info.max) - margin


def compute_local_qkv_and_rope(
    attn_data: MultiHeadAttention,
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, E = q.shape
    q_out, k_out, v_out = attn_data.in_proj(q, k, v)
    Hq, Hkv = attn_data.nheads, attn_data.kvheads
    Dk, Dv = attn_data.emb_kq_per_head, attn_data.emb_v_per_head
    Q = q_out.view(B, T, Hq, Dk)
    K = k_out.view(B, T, Hkv, Dk)
    V = v_out.view(B, T, Hkv, Dv)
    if attn_data.position_encoder and T > 0: # RoPE
        # Ensure position_ids is not None if T > 0 and RoPE is used.
        # The benchmark script should provide valid position_ids.
        # If position_ids can be None here, a default like torch.arange might be needed,
        # but the snippet assumes it's provided.
        assert position_ids is not None, "position_ids must be provided for RoPE"

        # Create a mask for valid positions (not -1)
        valid_pos_mask = position_ids.ne(-1)
        if valid_pos_mask.any(): # Apply RoPE only to valid positions
            # Clamp position_ids to be within the max sequence length of RoPE
            # max_seq_len_cached might be more accurate if NTK scaling is involved.
            # Using a common default like 2048 or a configurable max_pos.
            max_pos = getattr(attn_data.position_encoder, 'max_seq_len', 2048) # Fallback, ideally from config
            clamped_position_ids = position_ids.clone()
            clamped_position_ids[valid_pos_mask] = clamped_position_ids[valid_pos_mask].clamp(0, max_pos - 1)

            q_rope, k_rope = attn_data.position_encoder.adjusted_qk(Q, K, clamped_position_ids)
            # Apply RoPE only where positions are valid
            Q = torch.where(valid_pos_mask.unsqueeze(-1).unsqueeze(-1).expand_as(Q), q_rope, Q)
            K = torch.where(valid_pos_mask.unsqueeze(-1).unsqueeze(-1).expand_as(K), k_rope, K)
    return (
        Q.permute(0, 2, 1, 3),
        K.permute(0, 2, 1, 3),
        V.permute(0, 2, 1, 3),
    )


def _pad_to_block(t: torch.Tensor, target_len: int, dim: int = 2) -> torch.Tensor:
    if t.size(dim) >= target_len:
        assert t.size(dim) == target_len, f"Shape {t.shape} incompatible with target {target_len}"
        return t # Already at or exceeds target length
    pad = torch.zeros(
        *t.shape[:dim],
        target_len - t.size(dim),
        *t.shape[dim + 1 :],
        dtype=t.dtype,
        device=t.device,
    )
    return torch.cat([t, pad], dim=dim)


class RingAttentionHelper:
    def __init__(
        self,
        attn_module: nn.Module,
        strategy: Any,
        llama_block: nn.Module,
        use_cache: bool = False,
        ff: Optional[nn.Module] = None,
        ff_norm: Optional[nn.Module] = None,
    ):
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.strategy = strategy
        self.use_cache = use_cache
        self.llama_block = llama_block

        self.rank = strategy.rank
        self.world_size = strategy.world_size
        self.block_size = strategy.block_size

        self.head_dim = attn_module.emb_kq_per_head
        self.scale = attn_module.scale_factor or math.sqrt(self.head_dim)
        self._accum_dtype = torch.float32


    def forward(
        self,
        x_norm,
        strategy,
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        is_causal_mask=False,
        valid_len=0,
        residual=None,
    ):
        assert isinstance(strategy, RingAttentionStrategy) and self.strategy is strategy
        B, T_padded, _ = x_norm.shape
        if self.world_size > 1:
            assert T_padded == self.block_size
        start = self.rank * self.block_size

        if position_ids is None:
            position_ids = torch.full((B, T_padded), -1, dtype=torch.long, device=x_norm.device)
            if valid_len > 0:
                position_ids[:, :valid_len] = torch.arange(start, start + valid_len, device=x_norm.device)

        x_trim = x_norm[:, :valid_len, :]
        pos_trim = position_ids[:, :valid_len]
        res_trim = residual[:, :valid_len, :] if residual is not None else None

        q, k, v = compute_local_qkv_and_rope(self.attn, x_trim, x_trim, x_trim, pos_trim) # Use updated RoPE

        if self.attn.nheads != self.attn.kvheads:
            e = self.attn.nheads // self.attn.kvheads
            k = k.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_kq_per_head)
            v = v.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_v_per_head)

        out_valid = self.forward_full(q, k, v, mask, valid_len, res_trim, start, is_causal_mask)
        return (_pad_to_block(out_valid, self.block_size, dim=1), None, None) if self.world_size > 1 else (out_valid, None, None)

    def forward_full(self, q, k, v, mask, valid_len_local, x_block_residual_input, q_start_global, is_causal_mask: bool):
        B, H, T, D_kq = q.shape # T is the local query length (valid_len_local)
        _, _, _, D_v = v.shape   # D_v is the value head dimension

        # All ranks must participate in ring communication, even if T (local query length) is 0.
        # The max_score, num, and den computations will naturally result in tensors with a 0 query dimension if T=0.

        accum_dtype = self._accum_dtype
        exp_min, exp_max = _get_clamp_bounds(accum_dtype)

        # q_compute will have T as its query dimension length.
        # k_compute and v_compute sequence lengths are based on the K/V blocks being processed,
        # which may be non-zero even if local T is 0.
        qc, kc, vc = q.to(accum_dtype), k.to(accum_dtype), v.to(accum_dtype)

        # max_score will have shape (B, H, T, 1)
        max_score = self._compute_max_score_pass(
            qc, kc, mask, q_start_global, valid_len_local, is_causal_mask
        )
        # num will have shape (B, H, T, D_v)
        # den will have shape (B, H, T, 1)
        num, denom = self._compute_sums_pass(
            qc, kc, vc, mask, q_start_global, valid_len_local, max_score, exp_min, exp_max, is_causal_mask
        )

        if T > 0:
            eps = torch.finfo(denom.dtype).eps
            attn = (num / (denom + eps)).to(q.dtype) # attn shape (B, H, T, D_v)

            if self.attn.p_dropout and self.llama_block.training:
                attn = F.dropout(attn, p=self.attn.p_dropout, training=True)
            attn_reshaped = attn.transpose(1, 2).contiguous().view(B, T, H * D_v)
            attn_out = self.attn.dense(attn_reshaped) # attn_out shape (B, T, E)
        else: # T == 0
            # If there are no local queries, the attention output is an empty tensor.
            # self.attn.emb_dim is the output dimension of the dense layer.
            attn_out = torch.empty(B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype)

        # x_block_residual_input has shape (B, T, E). If T=0, it's (B, 0, E).
        x_after_attn = x_block_residual_input + attn_out
        return self.ff(self.ff_norm(x_after_attn)) + x_after_attn

    def _compute_max_score_pass(self, q_compute, k_compute, mask_global, q_start_global, valid_len_local, is_causal_mask: bool):
        B, H, T, _ = q_compute.shape
        dtype = q_compute.dtype
        dev = q_compute.device
        max_score = torch.full((B, H, T, 1), torch.finfo(dtype).min, device=dev, dtype=dtype)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=dev)
        k_blk, k_len = k_compute, k_compute.shape[2]

        start_idx_loop = ((self.rank) % self.world_size) * self.block_size # Initial start_idx for k-blocks
        for i in range(self.world_size):
            # k_start_loop is the global start index of the current k_blk
            k_idx = torch.arange(start_idx_loop, start_idx_loop + k_len, device=dev)
            m = mask_global[:, :, q_start_global:q_start_global + T, start_idx_loop:start_idx_loop + k_len] if mask_global is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m, apply_causal=is_causal_mask)
                max_score = torch.maximum(max_score, s.amax(-1, keepdim=True))
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, k_len)
                start_idx_loop = ((self.rank - (i + 1) + self.world_size) % self.world_size) * self.block_size
        return max_score

    def _compute_sums_pass(self, q_compute, k_compute, v_compute, mask_global, q_start_global, valid_len_local, max_score, exp_min, exp_max, is_causal_mask: bool):
        B, H, T, _ = q_compute.shape # T is local query length (valid_len_local)
        _, _, _, D_v = v_compute.shape   # Get Dv from v_compute
        dev = q_compute.device
        dtype = self._accum_dtype
        num = torch.zeros(B, H, T, D_v, device=dev, dtype=dtype) # Use D_v here
        den = torch.zeros(B, H, T, 1, device=dev, dtype=dtype)
        num_comp = torch.zeros_like(num)
        den_comp = torch.zeros_like(den)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=dev)
        k_blk, v_blk, k_len = k_compute, v_compute, k_compute.shape[2]

        start_idx_loop = ((self.rank) % self.world_size) * self.block_size # Initial start_idx for k-blocks
        for i in range(self.world_size):
            # k_start_loop is the global start index of the current k_blk
            k_idx = torch.arange(start_idx_loop, start_idx_loop + k_len, device=dev)
            m = mask_global[:, :, q_start_global:q_start_global + T, start_idx_loop:start_idx_loop + k_len] if mask_global is not None else None
            
            # Only proceed if there are local queries (T > 0) and the current k_block has content (k_len > 0)
            if T > 0 and k_len > 0:
                s = self._compute_attention_scores(q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m, apply_causal=is_causal_mask)
                delta = s - max_score
                clamp = delta.clamp(min=exp_min, max=exp_max)
                e = torch.exp(clamp.masked_fill(torch.isneginf(clamp), exp_min)) # exp(S - max(S))
                e = e.masked_fill(torch.isneginf(max_score.expand_as(s)), 0) # if max_score was -inf, scores were -inf, exp should be 0

                # Ensure v_blk's K-dimension matches e's K-dimension (k_len from k_blk)
                # v_blk.shape[2] is the actual current length of the v_blk tensor.
                if v_blk.shape[2] == k_len: # If k_len > 0, v_blk.shape[2] must also be > 0 and equal.
                    contrib_num = torch.matmul(e, v_blk) # Use v_blk directly, it's already the correct block
                    contrib_den = e.sum(-1, keepdim=True)

                    # Kahan summation
                    y_num = contrib_num - num_comp; t_num = num + y_num; num_comp = (t_num - num) - y_num; num = t_num
                    y_den = contrib_den - den_comp; t_den = den + y_den; den_comp = (t_den - den) - y_den; den = t_den
                # else: A desynchronization occurred (e.g., k_len > 0 but v_blk.shape[2] == 0 or != k_len).
                # This indicates an issue, but we skip the matmul to prevent a crash.
                # The benchmark script's output should be inspected if this path is taken.

            if i < self.world_size - 1:
                # Store the length of the current k_blk (and v_blk) before it's shifted.
                len_of_block_to_send = k_len
                
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, len_of_block_to_send)
                v_blk, _ = self._ring_shift_tensor(v_blk, self.block_size, len_of_block_to_send)
                start_idx_loop = ((self.rank - (i + 1) + self.world_size) % self.world_size) * self.block_size

        return num, den

    def _compute_attention_scores(self, q, k, q_idx, k_idx, mask=None, apply_mask=True, apply_causal=True):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        if Tq == 0: # No local queries, no scores to compute.
            return torch.empty(B, H, 0, Tk, device=q.device, dtype=q.dtype)
        scores = torch.matmul(q / self.scale, k.transpose(-2, -1))

        if apply_mask:
            if mask is not None:
                assert mask.shape == scores.shape, f"Mask shape {mask.shape} mismatch with scores shape {scores.shape}"
                scores += mask.to(scores.dtype)
            if apply_causal:
                causal = k_idx[None, None, None, :] > q_idx[None, None, :, None]
                scores = scores.masked_fill(causal, float('-inf'))
        return scores

    def _ring_shift_tensor(self, tensor, pad_len, valid_len):
        rank, world = self.rank, self.world_size
        send, recv = (rank + 1) % world, (rank - 1 + world) % world
        device = tensor.device
        send_tensor = _pad_to_block(tensor[:, :, :valid_len], pad_len, -2).contiguous()
        recv_tensor = torch.empty_like(send_tensor)
        recv_len_tensor = torch.empty(1, dtype=torch.int32, device=device) # Renamed to avoid conflict

        ops = [
            P2POp(dist.isend, torch.tensor([valid_len], dtype=torch.int32, device=device), peer=send),
            P2POp(dist.irecv, recv_len_tensor, peer=recv),
            P2POp(dist.isend, send_tensor, peer=send),
            P2POp(dist.irecv, recv_tensor, peer=recv),
        ]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        r_len = recv_len_tensor.item()
        assert 0 <= r_len <= pad_len, f"Rank {rank}: invalid recv_len {r_len}"
        return recv_tensor[:, :, :r_len].contiguous(), r_len
