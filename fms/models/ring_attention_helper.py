from typing import List, Tuple, Dict, Optional, Union, Any
from fms.modules.attention import MultiHeadAttention
import torch
import torch.distributed as dist
from torch.distributed import P2POp
import math
import torch.nn.functional as F
import torch.nn as nn

from fms.distributed.strategy import RingAttentionStrategy # Import RingAttentionStrategy
EXP_CLAMP_MIN = -10.0
EXP_CLAMP_MAX = 10.0



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
        # Generate empty shaped tensors with correct dimensions
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

    if attn_data.position_encoder is not None:
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
                # Add missing arguments to match the call signature
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

        out = self.forward_full(q, k, v, mask, valid_len, residual, x_norm, start)
        return out, None, None

    def forward_full(self, q, k, v, mask, valid_len, x_block, x_norm_block, q_start_global):
        B, H, T, _ = q.shape
        if T == 0:
            return torch.empty(B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype)

        dtype = q.dtype if q.dtype in [torch.float32, torch.bfloat16, torch.float16] else torch.float32
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        max_score = self._compute_max_score_pass(q, k, mask, q_start_global)
        num, denom = self._compute_sums_pass(q, k, v, mask, q_start_global, max_score)
        attn = (num / (denom + 1e-10)).to(q.dtype)

        if self.attn.p_dropout and self.llama_block.training:
            attn = F.dropout(attn, p=self.attn.p_dropout, training=True)

        attn = self.attn.dense(attn.transpose(1, 2).contiguous().view(B, T, -1))
        x = x_block + attn
        return self.ff(self.ff_norm(x)) + x

    def _compute_max_score_pass(self, q_compute, k_compute, mask_global, q_start_global):
        B, H, T, _ = q_compute.shape
        dev, dtype = q_compute.device, q_compute.dtype
        max_score = torch.full((B, H, T, 1), -torch.inf, device=dev, dtype=dtype)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=dev)
        k_blk, k_len = k_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = (mask_global[:, :, q_start_global:q_start_global+T, k_start:k_start+k_len]
                 if mask_global is not None else None)
            if k_len > 0:
                scores = self._compute_attention_scores(q_compute, k_blk, q_idx, k_idx, m)
                max_score = torch.maximum(max_score, scores.amax(-1, keepdim=True))
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, k_len)
        return max_score

    def _compute_sums_pass(self, q_compute, k_compute, v_compute, mask_global, q_start_global, max_score):
        B, H, T, _ = q_compute.shape
        Dv, dev, dtype = v_compute.shape[-1], q_compute.device, q_compute.dtype
        num = torch.zeros(B, H, T, Dv, device=dev, dtype=dtype)
        den = torch.zeros(B, H, T, 1, device=dev, dtype=dtype)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=dev)
        k_blk, v_blk, k_len = k_compute, v_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = (mask_global[:, :, q_start_global:q_start_global+T, k_start:k_start+k_len]
                 if mask_global is not None else None)
            if k_len > 0:
                scores = self._compute_attention_scores(q_compute, k_blk, q_idx, k_idx, m)
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
                scores = scores.masked_fill(causal, -torch.inf)
        return scores.clamp(min=torch.finfo(scores.dtype).min)

    def _ring_shift_tensor(self, tensor, valid_len):
        # If world_size is 1, no shifting is needed.
        if self.world_size == 1:
            # Still slice for consistency
            return tensor[:, :, :valid_len].contiguous(), valid_len

        # Ensure send/recv always happens, even if valid_len == 0
        send_tensor = tensor[:, :, :valid_len].contiguous()
        send_len = torch.tensor([valid_len], dtype=torch.int32, device=tensor.device)
        recv_tensor = torch.empty((tensor.size(0), tensor.size(1), self.block_size, *tensor.shape[3:]),
                                dtype=tensor.dtype, device=tensor.device)
        recv_len = torch.empty(1, dtype=torch.int32, device=tensor.device)


        rank, world = self.rank, self.world_size
        send, recv = (rank + 1) % world, (rank - 1 + world) % world
        device = tensor.device
        print(f"[DEBUG Rank {rank}] _ring_shift_tensor: Sending to {send}, Receiving from {recv}. Input tensor shape: {tensor.shape}, valid_len to send: {valid_len}")
        send_tensor = tensor[:, :, :valid_len].contiguous()
        send_len = torch.tensor([valid_len], dtype=torch.int32, device=device)
        max_recv_len = self.block_size
        recv_tensor = torch.empty((tensor.size(0), tensor.size(1), max_recv_len, *tensor.shape[3:]),
                                  dtype=tensor.dtype, device=device)
        recv_len = torch.empty(1, dtype=torch.int32, device=device)

        print(f"[DEBUG Rank {rank}] _ring_shift_tensor: send_tensor shape: {send_tensor.shape}, send_len: {send_len.item()}, recv_tensor buffer shape: {recv_tensor.shape}")

        ops = [
            P2POp(dist.isend, send_len, peer=send),
            P2POp(dist.irecv, recv_len, peer=recv),
            P2POp(dist.isend, send_tensor, peer=send),
            P2POp(dist.irecv, recv_tensor, peer=recv)
        ]
        for op in dist.batch_isend_irecv(ops):
            print(f"[DEBUG Rank {rank}] _ring_shift_tensor: Waiting for op: {op}")
            op.wait()
            print(f"[DEBUG Rank {rank}] _ring_shift_tensor: Op completed: {op}")

        r_len = recv_len.item()
        print(f"[DEBUG Rank {rank}] _ring_shift_tensor: Received r_len: {r_len}. Slicing recv_tensor to this length.")
        return recv_tensor[:, :, :r_len].contiguous(), r_len
