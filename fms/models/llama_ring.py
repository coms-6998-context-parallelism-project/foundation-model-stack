import math
from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed import P2POp

from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy
from fms.modules.attention import MultiHeadAttention


class RingAttentionKernel:
    @staticmethod
    def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
        return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

    @staticmethod
    def _pad_block(x: Tensor, length: int, dim: int) -> Tensor:
        cur = x.size(dim)
        if cur == length:
            return x
        pad_shape = list(x.shape); pad_shape[dim] = length - cur
        pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=dim)

    @staticmethod
    def _compute_qkv_and_rope(
        attn: MultiHeadAttention,
        x: Tensor,
        pos_ids: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, T, E = x.shape
        q, k, v = attn.in_proj(x, None, None)
        nh, nk = attn.nheads, attn.kvheads
        dk, dv = attn.emb_kq_per_head, attn.emb_v_per_head

        Q = q.view(B, T, nh, dk)
        K = k.view(B, T, nk, dk)
        V = v.view(B, T, nk, dv)

        if attn.position_encoder and T > 0:
            assert pos_ids is not None
            mask = pos_ids.ne(-1)
            if mask.any():
                pid = pos_ids.clone()
                max_len = getattr(attn.position_encoder, "max_seq_len", 2048)
                pid[mask] = pid[mask].clamp(0, max_len - 1)
                q_r, k_r = attn.position_encoder.adjusted_qk(Q, K, pid)
                m = mask.unsqueeze(-1).unsqueeze(-1)
                Q = torch.where(m.expand_as(Q), q_r, Q)
                K = torch.where(m.expand_as(K), k_r, K)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        if nh != nk:
            repeat = nh // nk
            K = K.unsqueeze(2).expand(-1, -1, repeat, -1, -1).reshape(B, nh, T, dk)
            V = V.unsqueeze(2).expand(-1, -1, repeat, -1, -1).reshape(B, nh, T, dv)

        return Q, K, V

    @staticmethod
    def _ring_shift(
        x: Tensor,
        strategy: RingAttentionStrategy,
        length: int
    ) -> Tuple[Tensor, int]:
        r, w, bs = strategy.rank, strategy.world_size, strategy.block_size
        send_to, recv_from = (r + 1) % w, (r - 1 + w) % w
        d = 2

        if length == 0:
            shape = list(x.shape); shape[d] = 0
            to_send = torch.empty(*shape, dtype=x.dtype, device=x.device)
        else:
            idx = [slice(None)] * x.ndim; idx[d] = slice(0, length)
            to_send = x[tuple(idx)]

        to_send = RingAttentionKernel._pad_block(to_send, bs, d).contiguous()
        to_recv = torch.empty_like(to_send)
        recv_len = torch.empty(1, torch.int32, device=x.device)
        send_len = torch.tensor([length], torch.int32, device=x.device)

        ops = [
            P2POp(dist.isend, send_len, peer=send_to),
            P2POp(dist.irecv, recv_len, peer=recv_from),
            P2POp(dist.isend, to_send, peer=send_to),
            P2POp(dist.irecv, to_recv, peer=recv_from),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

        nl = recv_len.item()
        idx2 = [slice(None)] * to_recv.ndim; idx2[d] = slice(0, nl)
        return to_recv[tuple(idx2)].contiguous(), nl

    @staticmethod
    def _attn_scores(
        Q: Tensor,
        K: Tensor,
        qi: Tensor,
        ki: Tensor,
        scale: float,
        mask: Optional[Tensor] = None,
        causal: bool = False
    ) -> Tensor:
        B, H, Tq, _ = Q.shape
        _, _, Tk, _ = K.shape
        if Tq == 0 or Tk == 0:
            return torch.empty(B, H, Tq, Tk, device=Q.device, dtype=Q.dtype)

        scores = torch.matmul(Q / scale, K.transpose(-2, -1))
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        if causal:
            future = ki.unsqueeze(0) > qi.unsqueeze(1)
            scores = scores.masked_fill(future.unsqueeze(0).unsqueeze(0), float("-inf"))
        return scores

    @staticmethod
    def _max_pass(
        Ql: Tensor,
        Kl: Tensor,
        mask: Optional[Tensor],
        q0: int,
        ql_len: int,
        strategy: RingAttentionStrategy,
        scale: float,
        causal: bool
    ) -> Tensor:
        B, H, Tq, Dk = Ql.shape
        ad = RingAttentionKernel._accum_dtype(Ql.dtype)
        max_score = torch.full((B, H, Tq, 1), torch.finfo(ad).min, device=Ql.device, dtype=ad)
        if Tq == 0:
            return max_score

        qi = torch.arange(q0, q0 + Tq, device=Ql.device)
        Ql_ad, K_ad = Ql.to(ad), Kl.to(ad)
        k_len = Kl.shape[2]

        for i in range(strategy.world_size):
            kr = (strategy.rank - i + strategy.world_size) % strategy.world_size
            ks = kr * strategy.block_size
            if k_len > 0:
                ki = torch.arange(ks, ks + k_len, device=Ql.device)
                lm = mask[:, :, slice(q0, q0 + Tq), slice(ks, ks + k_len)] if mask is not None else None
                s = RingAttentionKernel._attn_scores(Ql_ad, K_ad, qi, ki, scale, lm, causal)
                if s.numel():
                    max_score = torch.maximum(max_score, s.amax(dim=-1, keepdim=True))
            if i < strategy.world_size - 1:
                K_ad, k_len = RingAttentionKernel._ring_shift(K_ad, strategy, k_len)

        return max_score

    @staticmethod
    def _sum_pass(
        Ql: Tensor,
        Kl: Tensor,
        Vl: Tensor,
        mask: Optional[Tensor],
        q0: int,
        ql_len: int,
        max_score: Tensor,
        strategy: RingAttentionStrategy,
        scale: float,
        ad: torch.dtype,
        causal: bool
    ) -> Tuple[Tensor, Tensor]:
        B, H, Tq, Dk = Ql.shape
        dv = Vl.shape[-1]
        num = torch.zeros(B, H, Tq, dv, device=Ql.device, dtype=ad)
        den = torch.zeros(B, H, Tq, 1, device=Ql.device, dtype=ad)
        if Tq == 0:
            return num, den

        # Calculate exp_min and exp_max directly here
        finfo_ad = torch.finfo(ad)
        margin = 2.0
        current_exp_min, current_exp_max = math.log(finfo_ad.tiny) + margin, math.log(finfo_ad.max) - margin

        qi = torch.arange(q0, q0 + Tq, device=Ql.device)
        Ql_ad, K_blk, V_blk = Ql.to(ad), Kl.to(ad), Vl.to(ad)
        k_len = Kl.shape[2]

        for i in range(strategy.world_size):
            kr = (strategy.rank - i) % strategy.world_size
            ks = kr * strategy.block_size
            if k_len > 0:
                ki = torch.arange(ks, ks + k_len, device=Ql.device)
                lm = mask[:, :, slice(q0, q0 + Tq), slice(ks, ks + k_len)] if mask is not None else None
                s = RingAttentionKernel._attn_scores(Ql_ad, K_blk, qi, ki, scale, lm, causal)
                delta = torch.where(torch.isneginf(max_score), torch.full_like(s, -math.inf), s - max_score)
                exp_s = torch.exp(delta.clamp(min=current_exp_min, max=current_exp_max))
                exp_s = exp_s.masked_fill(torch.isneginf(max_score), 0.0)
                num += torch.matmul(exp_s, V_blk)
                den += exp_s.sum(dim=-1, keepdim=True)
            if i < strategy.world_size - 1:
                K_blk, k_len = RingAttentionKernel._ring_shift(K_blk, strategy, k_len)
                V_blk, _     = RingAttentionKernel._ring_shift(V_blk, strategy, k_len)

        return num, den

    @staticmethod
    def _three_pass(
        Ql: Tensor,
        Kl: Tensor,
        Vl: Tensor,
        mask: Optional[Tensor],
        strategy: RingAttentionStrategy,
        q0: int,
        ql_len: int,
        scale: float,
        ad: torch.dtype,
        causal: bool
    ) -> Tensor:
        max_score = RingAttentionKernel._max_pass(Ql, Kl, mask, q0, ql_len, strategy, scale, causal)

        
        num, den = RingAttentionKernel._sum_pass(
            Ql, Kl, Vl, mask, q0, ql_len, max_score,
            strategy, scale, ad, causal # Ensure ad and causal are passed here
        )
        if ql_len == 0:
            B, H, _, dv = Vl.shape
            return torch.empty(B, H, 0, dv, device=Ql.device, dtype=Ql.dtype)
        eps = torch.finfo(den.dtype).eps
        return (num / (den + eps)).to(Ql.dtype)

    @staticmethod
    def forward(
        x_norm: Tensor, 
        residual: Tensor,
        attn_module: MultiHeadAttention,
        ff: nn.Module, 
        ff_norm: nn.Module, 
        strategy: RingAttentionStrategy,
        valid_len: int, 
        mask: Optional[Tensor] = None, 
        position_ids: Optional[Tensor] = None, 
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None, 
        is_causal_mask: bool = False, 
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T_pad, E = x_norm.shape
        r, w, bs = strategy.rank, strategy.world_size, strategy.block_size
        if w > 1:
            assert T_pad == bs

        q0 = r * bs
        x_slice = x_norm[:, :valid_len, :]
        res_slice = residual[:, :valid_len, :]

        if position_ids is not None:
            pid = position_ids[:, :valid_len]
        elif valid_len > 0:
            idx = torch.arange(q0, q0 + valid_len, device=x_norm.device)
            pid = idx.unsqueeze(0).expand(B, -1)
        else:
            pid = None

        if valid_len > 0:
            Q, K, V = RingAttentionKernel._compute_qkv_and_rope(attn_module, x_slice, pid)
            cache = (K, V)
        else:
            nh, dv = attn_module.nheads, attn_module.emb_v_per_head
            Q = K = torch.empty(B, nh, 0, attn_module.emb_kq_per_head, device=x_norm.device, dtype=x_norm.dtype)
            V = torch.empty(B, nh, 0, dv, device=x_norm.device, dtype=x_norm.dtype)
            cache = None

        scale = attn_module.scale_factor or math.sqrt(attn_module.emb_kq_per_head)
        ad = RingAttentionKernel._accum_dtype(Q.dtype)

        attn_out = RingAttentionKernel._three_pass(Q, K, V, mask, strategy, q0, valid_len, scale, ad, is_causal_mask) # Use is_causal_mask
        if valid_len > 0:
            proj = attn_out.transpose(1, 2).reshape(B, valid_len, -1)
            out_attn = attn_module.dense(proj)
            if hasattr(attn_module, "dropout") and isinstance(attn_module.dropout, nn.Dropout):
                out_attn = attn_module.dropout(out_attn)
            x_attn = res_slice + out_attn
            x_ff = ff(ff_norm(x_attn))
            valid_out = x_attn + x_ff
        else:
            valid_out = torch.empty(B, 0, E, device=x_norm.device, dtype=x_norm.dtype)

        output = RingAttentionKernel._pad_block(valid_out, bs, dim=1) if w > 1 else valid_out
        return output, cache
