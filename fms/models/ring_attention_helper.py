from typing import List, Tuple, Dict, Optional, Union, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import P2POp

from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import RingAttentionStrategy

EXP_CLAMP_MIN = -10.0
EXP_CLAMP_MAX = 10.0

def compute_local_qkv_and_rope(
    attn_data: MultiHeadAttention,
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    position_ids: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, E = q.shape
    if T == 0:
        H_q, H_kv = attn_data.nheads, attn_data.kvheads
        D_kq, D_v = attn_data.emb_kq_per_head, attn_data.emb_v_per_head
        empty_q = torch.empty(B, H_q, 0, D_kq, device=q.device, dtype=q.dtype)
        empty_k = torch.empty(B, H_kv, 0, D_kq, device=q.device, dtype=q.dtype)
        empty_v = torch.empty(B, H_kv, 0, D_v, device=q.device, dtype=q.dtype)
        return empty_q, empty_k, empty_v

    q_out, k_out, v_out = attn_data.in_proj(q, k, v)
    H_q, H_kv = attn_data.nheads, attn_data.kvheads
    D_kq, D_v = attn_data.emb_kq_per_head, attn_data.emb_v_per_head

    queries = q_out.view(B, T, H_q, D_kq)
    keys    = k_out.view(B, T, H_kv, D_kq)
    values  = v_out.view(B, T, H_kv, D_v)

    if attn_data.position_encoder is not None and position_ids is not None:
        max_pos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
        position_ids = position_ids.clamp(0, max_pos - 1)
        queries, keys = attn_data.position_encoder.adjusted_qk(queries, keys, position_ids)

    return queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3)

class RingAttentionHelper:
    def __init__(self, attn_module: nn.Module, strategy: Any, llama_block: nn.Module,
                 use_cache: bool = False, ff: Optional[nn.Module] = None, ff_norm: Optional[nn.Module] = None):
        self.attn, self.ff, self.ff_norm = attn_module, ff, ff_norm
        self.strategy, self.use_cache, self.llama_block = strategy, use_cache, llama_block
        self.rank, self.world_size, self.block_size = strategy.rank, strategy.world_size, strategy.block_size
        self.head_dim = attn_module.emb_kq_per_head
        self.scale = attn_module.scale_factor or math.sqrt(self.head_dim)

    def forward(self, x_norm, strategy, mask=None, position_ids=None, residual=None, valid_len=0,
                past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                is_causal_mask: bool = False):
        assert isinstance(strategy, RingAttentionStrategy) and self.strategy is strategy
        start = self.rank * self.block_size

        if position_ids is None:
            position_ids = torch.full((x_norm.size(0), valid_len), -1, dtype=torch.long, device=x_norm.device)
            position_ids[:, :] = torch.arange(start, start + valid_len, device=x_norm.device)

        q, k, v = compute_local_qkv_and_rope(self.attn, x_norm, x_norm, x_norm, position_ids)

        if self.attn.nheads != self.attn.kvheads:
            e = self.attn.nheads // self.attn.kvheads
            k = k.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(x_norm.size(0), self.attn.nheads, valid_len, self.attn.emb_kq_per_head)
            v = v.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(x_norm.size(0), self.attn.nheads, valid_len, self.attn.emb_v_per_head)

        out = self.forward_full(q, k, v, mask, valid_len, x_norm, x_norm, start)
        return out, None, None

    def forward_full(self, q, k, v, mask, valid_len, x_block, x_norm_block, q_start_global):
        B, H, T, _ = q.shape
        # Convert to compute dtype
        dtype = q.dtype if q.dtype in [torch.float32, torch.bfloat16, torch.float16] else torch.float32
        q_compute, k_compute, v_compute = q.to(dtype), k.to(dtype), v.to(dtype)

        # Always perform communication passes
        max_score = self._compute_max_score_pass(q_compute, k_compute, mask, q_start_global)
        num, denom = self._compute_sums_pass(q_compute, k_compute, v_compute, mask, q_start_global, max_score)

        # After communication, handle zero-length
        if T == 0:
            return torch.empty(B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype)

        attn = torch.nan_to_num(num / (denom + 1e-10), nan=0.0, posinf=0.0, neginf=0.0).to(q.dtype)
        if self.attn.p_dropout and self.llama_block.training:
            attn = F.dropout(attn, p=self.attn.p_dropout, training=True)

        attn_out = self.attn.dense(attn.transpose(1, 2).contiguous().view(B, T, -1))
        x = x_block + attn_out
        return self.ff(self.ff_norm(x)) + x

    def _compute_max_score_pass(self, q, k, mask_global, q_start_global):
        B, H, T, _ = q.shape
        max_score = torch.full((B, H, T, 1), -torch.inf, device=q.device, dtype=q.dtype)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=q.device)
        k_blk, k_len = k, k.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=q.device)
            m = mask_global[:, :, q_start_global:q_start_global+T, k_start:k_start+k_len] if mask_global is not None else None
            # Only compute scores when both query and key lengths are non-zero
            if T > 0 and k_len > 0:
                scores = self._compute_attention_scores(q, k_blk, q_idx, k_idx, m)
                max_score = torch.maximum(max_score, scores.amax(-1, keepdim=True))
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, k_len)
        return max_score

    def _compute_sums_pass(self, q, k, v, mask_global, q_start_global, max_score):
        B, H, T, _ = q.shape
        Dv = v.shape[-1]
        num = torch.zeros(B, H, T, Dv, device=q.device, dtype=q.dtype)
        den = torch.zeros(B, H, T, 1, device=q.device, dtype=q.dtype)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=q.device)
        k_blk, v_blk, k_len = k, v, k.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=q.device)
            m = mask_global[:, :, q_start_global:q_start_global+T, k_start:k_start+k_len] if mask_global is not None else None
            scores = self._compute_attention_scores(q, k_blk, q_idx, k_idx, m)
            exp = torch.exp((scores - max_score).clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX))
            num += torch.einsum("bhqk,bhkd->bhqd", exp, v_blk)
            den += exp.sum(-1, keepdim=True)
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, k_len)
                v_blk, _ = self._ring_shift_tensor(v_blk, k_len)
        return num, den

    def _compute_attention_scores(self, q, k, q_idx, k_idx, mask=None, apply_mask=True, keep_causal=True):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if apply_mask:
            if mask is not None:
                scores += mask.to(scores.dtype)
            if keep_causal:
                causal = (k_idx[None, None, None, :] > q_idx[None, None, :, None])
                scores = scores.masked_fill(causal, float('-inf'))
        return scores.clamp(min=torch.finfo(scores.dtype).min)

    def _ring_shift_tensor(self, tensor, valid_len):
        if self.world_size == 1:
            return tensor[:, :, :valid_len].contiguous(), valid_len

        B, H, T, *rest = tensor.shape
        device = tensor.device
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1 + self.world_size) % self.world_size
        send_tensor = tensor[:, :, :valid_len].contiguous()
        send_len = torch.tensor([valid_len], dtype=torch.int32, device=device)
        recv_tensor = torch.empty((B, H, self.block_size, *rest), dtype=tensor.dtype, device=device)
        recv_len = torch.empty(1, dtype=torch.int32, device=device)

        ops = [
            P2POp(dist.isend, send_len, peer=send_rank),
            P2POp(dist.irecv, recv_len, peer=recv_rank),
            P2POp(dist.isend, send_tensor, peer=send_rank),
            P2POp(dist.irecv, recv_tensor, peer=recv_rank)
        ]
        for op in dist.batch_isend_irecv(ops):
            op.wait()

        actual_len = recv_len.item()
        return recv_tensor[:, :, :actual_len].contiguous(), actual_len
