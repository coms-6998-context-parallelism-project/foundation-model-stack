from typing import Tuple, Optional, Any
from fms.modules.attention import MultiHeadAttention
import torch
import torch.distributed as dist
from torch.distributed import P2POp
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
    position_ids: Optional[torch.Tensor] = None,
    is_self: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, E = q.shape
    q_out, k_out, v_out = attn_data.in_proj(q, k, v)
    assert q_out.shape == k_out.shape == v_out.shape == (B, T, E)

    H_q, H_kv = attn_data.nheads, attn_data.kvheads
    D_kq, D_v = attn_data.emb_kq_per_head, attn_data.emb_v_per_head

    queries = q_out.view(B, T, H_q, D_kq)
    keys = k_out.view(B, T, H_kv, D_kq)
    values = v_out.view(B, T, H_kv, D_v)

    if attn_data.position_encoder and T > 0:
        assert position_ids is not None
        # T is valid_len here, and position_ids are the global indices for these valid tokens.
        # These position_ids (pos_trim from caller) do not contain -1 padding markers.
        if position_ids.shape != (B, T) and not (T == 0 and position_ids.shape == (B, 0)):
            raise AssertionError(f"Expected shape {(B, T)}, got {position_ids.shape}")

        # Clamp global positions to be within RoPE's precomputed range.
        # attn_data.position_encoder is an instance of fms.modules.positions.RotaryEmbedding
        max_len_for_rope = attn_data.position_encoder.max_seq_len
        position_ids_clamped = position_ids.clamp(0, max_len_for_rope - 1)

        q_rope, k_rope = attn_data.position_encoder.adjusted_qk(queries, keys, position_ids_clamped)
        queries, keys = q_rope, k_rope # Direct assignment as all tokens in x_trim are valid

    return (
        queries.permute(0, 2, 1, 3),
        keys.permute(0, 2, 1, 3),
        values.permute(0, 2, 1, 3),
    )


def _pad_to_block(t: torch.Tensor, target_len: int, dim: int = 2) -> torch.Tensor:
    if t.size(dim) >= target_len:
        assert t.size(dim) == target_len, f"Shape {t.shape} incompatible with target {target_len}"
        return t
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
        ff: Optional[nn.Module] = None,
        ff_norm: Optional[nn.Module] = None,
    ):
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.strategy = strategy
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
        is_causal_mask=False,
        valid_len=0,
        residual=None,
    ) -> torch.Tensor:
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

        q, k, v = compute_local_qkv_and_rope(self.attn, x_trim, x_trim, x_trim, pos_trim)

        if self.attn.nheads != self.attn.kvheads:
            e = self.attn.nheads // self.attn.kvheads
            k = k.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_kq_per_head)
            v = v.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_v_per_head)

        out_valid = self.forward_full(q, k, v, mask, valid_len, res_trim, x_trim, start)
        return _pad_to_block(out_valid, self.block_size, dim=1) if self.world_size > 1 else out_valid

    def forward_full(self, q, k, v, mask, valid_len, x_block, x_norm_block, q_start_global):
        B, H, T, D = q.shape
        if T == 0:
            return torch.empty(B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype)

        dev = q.device
        accum_dtype = self._accum_dtype
        exp_min, exp_max = _get_clamp_bounds(accum_dtype)
        q_compute = q.to(accum_dtype)
        k_compute = k.to(accum_dtype)
        v_compute = v.to(accum_dtype)

        # Hoist q_idx computation as it's constant for these passes
        q_idx = torch.arange(q_start_global, q_start_global + T, device=dev)

        max_score = self._compute_max_score_pass(q_compute, k_compute, mask, q_idx, valid_len)
        num, denom = self._compute_sums_pass(
            q_compute, k_compute, v_compute, mask, q_idx, valid_len, max_score, exp_min, exp_max
        )

        eps = torch.finfo(denom.dtype).eps
        attn = (num / (denom + eps)).to(q.dtype)

        if self.attn.p_dropout and self.llama_block.training:
            attn = F.dropout(attn, p=self.attn.p_dropout, training=True)

        attn_out = self.attn.dense(attn.transpose(1, 2).contiguous().view(B, T, -1))
        x = x_block + attn_out
        return self.ff(self.ff_norm(x)) + x

    def _compute_max_score_pass(self, q_compute, k_compute, mask_global, q_idx, valid_len_local):
        B, H, T, _ = q_compute.shape
        dtype = q_compute.dtype
        dev = q_compute.device
        max_score = torch.full((B, H, T, 1), torch.finfo(dtype).min, device=dev, dtype=dtype)
        k_blk, k_len = k_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = mask_global[:, :, q_idx[0]:q_idx[-1]+1, k_start:k_start + k_len] if mask_global is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m)
                max_score = torch.maximum(max_score, s.amax(-1, keepdim=True))
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, k_len)
        return max_score

    def _compute_sums_pass(
        self, q_compute, k_compute, v_compute, mask_global, q_idx, valid_len_local, max_score, exp_min, exp_max
    ):
        B, H, T, Dv = q_compute.shape
        dev = q_compute.device
        dtype = self._accum_dtype
        num = torch.zeros(B, H, T, Dv, device=dev, dtype=dtype)
        den = torch.zeros(B, H, T, 1, device=dev, dtype=dtype)
        num_comp = torch.zeros_like(num) # Kahan summation compensator for numerator
        den_comp = torch.zeros_like(den) # Kahan summation compensator for denominator
        k_blk, v_blk, k_len = k_compute, v_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = mask_global[:, :, q_idx[0]:q_idx[-1]+1, k_start:k_start + k_len] if mask_global is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m)
                e = torch.exp((s - max_score).clamp(min=exp_min, max=exp_max))
                contrib_num = torch.einsum("bhqk,bhkd->bhqd", e, v_blk[:, :, :k_len])
                contrib_den = e.sum(-1, keepdim=True)

                # Kahan summation
                y_num = contrib_num - num_comp
                t_num = num + y_num
                num_comp = (t_num - num) - y_num
                num = t_num

                y_den = contrib_den - den_comp
                t_den = den + y_den
                den_comp = (t_den - den) - y_den
                den = t_den

            if i < self.world_size - 1:
                valid = k_len
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, valid)
                v_blk, _ = self._ring_shift_tensor(v_blk, self.block_size, valid)

        return num, den

    def _compute_attention_scores(self, q, k, q_idx, k_idx, mask=None, apply_mask=True, keep_causal=True):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        scores = torch.matmul(q / self.scale, k.transpose(-2, -1))

        if apply_mask:
            if mask is not None:
                scores += mask.to(scores.dtype)
            if keep_causal:
                causal = k_idx[None, None, None, :] > q_idx[None, None, :, None]
                scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        return scores

    def _ring_shift_tensor(self, tensor, pad_len, valid_len):
        rank, world = self.rank, self.world_size
        send, recv = (rank + 1) % world, (rank - 1 + world) % world
        device = tensor.device
        send_tensor = _pad_to_block(tensor[:, :, :valid_len], pad_len, -2).contiguous()
        recv_tensor = torch.empty_like(send_tensor)
        recv_len = torch.empty(1, dtype=torch.int32, device=device)

        ops = [
            P2POp(dist.isend, torch.tensor([valid_len], dtype=torch.int32, device=device), peer=send),
            P2POp(dist.irecv, recv_len, peer=recv),
            P2POp(dist.isend, send_tensor, peer=send),
            P2POp(dist.irecv, recv_tensor, peer=recv),
        ]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        r_len = recv_len.item()
        assert 0 <= r_len <= pad_len, f"Rank {rank}: invalid recv_len {r_len}"
        return recv_tensor[:, :, :r_len].contiguous(), r_len
