from typing import List, Tuple, Dict, Optional, Union, Any
from fms.modules.attention import MultiHeadAttention
import torch
import torch.distributed as dist
from torch.distributed import P2POp
import math
import torch.nn.functional as F
import torch.nn as nn
from fms.distributed.strategy import RingAttentionStrategy  # Import RingAttentionStrategy

# Helper functions for numeric stability

def _get_accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Choose an accumulation dtype based on input dtype.
    Float16, bfloat16, float32 are supported; others promote to float32.
    """
    return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32


def _get_clamp_bounds(accum_dtype: torch.dtype) -> Tuple[float, float]:
    """
    Return (min, max) log-space clamp bounds based on numeric limits of accum_dtype.
    """
    info = torch.finfo(accum_dtype)
    return math.log(info.tiny), math.log(info.max)

# Backward-compatible defaults
EXP_CLAMP_MIN = -10.0
EXP_CLAMP_MAX = 10.0


def compute_local_qkv_and_rope(
    attn_data: MultiHeadAttention,
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    assert queries.shape == (B, T, H_q, D_kq)
    assert keys.shape == (B, T, H_kv, D_kq)
    assert values.shape == (B, T, H_kv, D_v)

    if attn_data.position_encoder and T > 0:
        assert position_ids is not None
        if position_ids.shape != (B, T) and not (T == 0 and position_ids.shape == (B, 0)):
            raise AssertionError(f"Expected shape {(B, T)}, got {position_ids.shape}")

        valid_mask = position_ids != -1
        if valid_mask.any():
            max_pos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
            position_ids_safe = position_ids.clone()
            position_ids_safe[valid_mask] = position_ids_safe[valid_mask].clamp(0, max_pos - 1)
            q_rope, k_rope = attn_data.position_encoder.adjusted_qk(
                queries, keys, position_ids_safe
            )

            if valid_mask.all():
                queries, keys = q_rope, k_rope
            else:
                mask = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(queries)
                queries = torch.where(mask, q_rope, queries)
                keys = torch.where(mask.expand_as(keys), k_rope, keys)

    return (
        queries.permute(0, 2, 1, 3),
        keys.permute(0, 2, 1, 3),
        values.permute(0, 2, 1, 3),
    )


def _pad_to_block(t: torch.Tensor, target_len: int, dim: int = 2) -> torch.Tensor:
    """Pad tensor on `dim` to `target_len`."""
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
    """Performs distributed Ring Attention with RoPE, 2-pass softmax, and optional FF."""
    def __init__(
        self,
        attn_module: nn.Module,
        strategy: Any,
        llama_block: nn.Module,
        use_cache: bool = False,
        ff: Optional[nn.Module] = None,
        ff_norm: Optional[nn.Module] = None,
    ):
        self.attn, self.ff, self.ff_norm, self.strategy, self.use_cache, self.llama_block = (
            attn_module,
            ff,
            ff_norm,
            strategy,
            use_cache,
            llama_block,
        )
        self.rank, self.world_size, self.block_size = (
            strategy.rank,
            strategy.world_size,
            strategy.block_size,
        )
        self.head_dim = attn_module.emb_kq_per_head
        self.scale = attn_module.scale_factor or math.sqrt(self.head_dim)

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
            position_ids = torch.full(
                (B, T_padded), -1, dtype=torch.long, device=x_norm.device
            )
            if valid_len > 0:
                position_ids[:, :valid_len] = torch.arange(
                    start, start + valid_len, device=x_norm.device
                )

        x_trim = x_norm[:, :valid_len, :]
        pos_trim = position_ids[:, :valid_len]
        res_trim = residual[:, :valid_len, :] if residual is not None else None

        q, k, v = compute_local_qkv_and_rope(
            self.attn, x_trim, x_trim, x_trim, pos_trim
        )

        if self.attn.nheads != self.attn.kvheads:
            e = self.attn.nheads // self.attn.kvheads
            k = (
                k.unsqueeze(2)
                .expand(-1, -1, e, -1, -1)
                .reshape(B, self.attn.nheads, valid_len, self.attn.emb_kq_per_head)
            )
            v = (
                v.unsqueeze(2)
                .expand(-1, -1, e, -1, -1)
                .reshape(B, self.attn.nheads, valid_len, self.attn.emb_v_per_head)
            )

        out_valid = self.forward_full(
            q, k, v, mask, valid_len, res_trim, x_trim, start
        )
        if self.world_size == 1:  # If not distributed, no need to pad to block_size
            return out_valid, None, None
        else:  # Otherwise, pad for consistency in distributed setting before potential gather
            return _pad_to_block(out_valid, self.block_size, dim=1), None, None

    def forward_full(
        self, q, k, v, mask, valid_len, x_block, x_norm_block, q_start_global
    ):
        B, H, T, D = q.shape
        # If T is zero, return empty as before
        if T == 0:
            return torch.empty(
                B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype
            )

        # Promote to stable accumulation dtype
        accum_dtype = _get_accum_dtype(q.dtype)
        exp_min, exp_max = _get_clamp_bounds(accum_dtype)
        q_compute = q.to(accum_dtype)
        k_compute = k.to(accum_dtype)
        v_compute = v.to(accum_dtype)

        # Two-pass softmax for numerical stability
        max_score = self._compute_max_score_pass(
            q_compute, k_compute, mask, q_start_global, valid_len
        )
        num, denom = self._compute_sums_pass(
            q_compute,
            k_compute,
            v_compute,
            mask,
            q_start_global,
            valid_len,
            max_score,
            exp_min,
            exp_max,
        )
        attn = (num / (denom + 1e-10)).to(q.dtype)

        if self.attn.p_dropout and self.llama_block.training:
            attn = F.dropout(attn, p=self.attn.p_dropout, training=True)

        attn_out = self.attn.dense(
            attn.transpose(1, 2).contiguous().view(B, T, -1)
        )
        x = x_block + attn_out
        return self.ff(self.ff_norm(x)) + x

    def _compute_max_score_pass(
        self, q_compute, k_compute, mask_global, q_start_global, valid_len_local
    ):
        B, H, T, _ = q_compute.shape
        dev, dtype = q_compute.device, q_compute.dtype
        max_score = torch.full(
            (B, H, T, 1), -torch.inf, device=dev, dtype=dtype
        )
        q_idx = torch.arange(
            q_start_global, q_start_global + T, device=dev
        )
        k_blk, k_len = k_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = (
                ((self.rank - i + self.world_size) % self.world_size)
                * self.block_size
            )
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = (
                mask_global[
                    :, :, q_start_global : q_start_global + T, k_start : k_start + k_len
                ]
                if mask_global is not None
                else None
            )
            if k_len > 0:
                s = self._compute_attention_scores(
                    q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m
                )
                max_score = torch.maximum(
                    max_score, s.amax(-1, keepdim=True)
                )
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(
                    k_blk, self.block_size, k_len
                )
        return max_score

    def _compute_sums_pass(
        self,
        q_compute,
        k_compute,
        v_compute,
        mask_global,
        q_start_global,
        valid_len_local,
        max_score,
        exp_min,
        exp_max,
    ):
        B, H, T, Dv = q_compute.shape
        dev, dtype = v_compute.device, q_compute.dtype
        num = torch.zeros(B, H, T, Dv, device=dev, dtype=dtype)
        den = torch.zeros(B, H, T, 1, device=dev, dtype=dtype)
        q_idx = torch.arange(
            q_start_global, q_start_global + T, device=dev
        )
        k_blk, v_blk, k_len = k_compute, v_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = (
                ((self.rank - i + self.world_size) % self.world_size)
                * self.block_size
            )
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = (
                mask_global[
                    :, :, q_start_global : q_start_global + T, k_start : k_start + k_len
                ]
                if mask_global is not None
                else None
            )
            if k_len > 0:
                s = self._compute_attention_scores(
                    q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m
                )
                e = torch.exp((s - max_score).clamp(min=exp_min, max=exp_max))
                num += torch.einsum("bhqk,bhkd->bhqd", e, v_blk[:, :, :k_len])
                den += e.sum(-1, keepdim=True)
            if i < self.world_size - 1:
                valid = k_len
                k_blk, k_len = self._ring_shift_tensor(
                    k_blk, self.block_size, valid
                )
                v_blk, _ = self._ring_shift_tensor(
                    v_blk, self.block_size, valid
                )
        return num, den

    def _compute_attention_scores(
        self, q, k, q_idx, k_idx, mask=None, apply_mask=True, keep_causal=True
    ):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if apply_mask:
            if mask is not None:
                scores += mask.to(scores.dtype)
            if keep_causal:
                causal = (
                    k_idx[None, None, None, :] > q_idx[None, None, :, None]
                )
                scores = scores.masked_fill(causal, -torch.inf)

        return scores.clamp(min=torch.finfo(scores.dtype).min)

    def _ring_shift_tensor(self, tensor, pad_len, valid_len):
        rank, world = self.rank, self.world_size
        send, recv = (rank + 1) % world, (rank - 1 + world) % world
        device = tensor.device

        send_tensor = _pad_to_block(
            tensor[:, :, :valid_len], pad_len, -2
        ).contiguous()
        recv_tensor = torch.empty_like(send_tensor)
        recv_len = torch.empty(1, dtype=torch.int32, device=device)

        ops = [
            P2POp(
                dist.isend,
                torch.tensor([valid_len], dtype=torch.int32, device=device),
                peer=send,
            ),
            P2POp(dist.irecv, recv_len, peer=recv),
            P2POp(dist.isend, send_tensor, peer=send),
            P2POp(dist.irecv, recv_tensor, peer=recv),
        ]
        for r in dist.batch_isend_irecv(ops):
            r.wait()

        r_len = recv_len.item()
        assert 0 <= r_len <= pad_len, f"Rank {rank}: invalid recv_len {r_len}"
        return recv_tensor[:, :, :r_len].contiguous(), r_len
