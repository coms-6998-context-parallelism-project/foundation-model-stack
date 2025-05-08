import math
from typing import Any, List, Mapping, Optional, Tuple, Union  # Keep necessary types

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor  # Explicitly import Tensor for type hinting
from torch.distributed import P2POp

from fms.distributed.strategy import (  # Need both for type hints
    DistributedStrategy,
    RingAttentionStrategy,
)
from fms.modules.attention import MultiHeadAttention


class RingAttentionKernel:
    @staticmethod
    def _get_accum_dtype(dtype: torch.dtype) -> torch.dtype:
        return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

    @staticmethod
    def _get_clamp_bounds(accum_dtype: torch.dtype) -> Tuple[float, float]:
        info = torch.finfo(accum_dtype)
        m = 2.0
        return math.log(info.tiny) + m, math.log(info.max) - m

    @staticmethod
    def _pad_to_block(t: Tensor, target_len: int, dim: int) -> Tensor:
        cur = t.size(dim)
        if cur >= target_len:
            assert cur == target_len
            return t
        pad_shape = list(t.shape)
        pad_shape[dim] = target_len - cur
        pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=dim)

    @staticmethod
    def _compute_qkv_and_rope(
        attn_module: MultiHeadAttention,
        x: Tensor,
        position_ids: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, T, E = x.shape
        q, k, v = attn_module.in_proj(x, None, None)
        Hq, Hkv = attn_module.nheads, attn_module.kvheads
        Dk, Dv = attn_module.emb_kq_per_head, attn_module.emb_v_per_head
        Q = q.view(B, T, Hq, Dk)
        K = k.view(B, T, Hkv, Dk)
        V = v.view(B, T, Hkv, Dv)
        if attn_module.position_encoder and T > 0:
            assert position_ids is not None
            mask = position_ids.ne(-1)
            if mask.any():
                pid = position_ids.clone()
                max_pos = getattr(attn_module.position_encoder, 'max_seq_len', 2048)
                pid[mask] = pid[mask].clamp(0, max_pos - 1)
                q_r, k_r = attn_module.position_encoder.adjusted_qk(Q, K, pid)
                m = mask.unsqueeze(-1).unsqueeze(-1)
                Q = torch.where(m.expand_as(Q), q_r, Q)
                K = torch.where(m.expand_as(K), k_r, K)
        Qp = Q.permute(0, 2, 1, 3)
        Kp = K.permute(0, 2, 1, 3)
        Vp = V.permute(0, 2, 1, 3)
        if attn_module.nheads != attn_module.kvheads:
            ng = attn_module.nheads // attn_module.kvheads
            Kp = Kp.unsqueeze(2).expand(-1,-1,ng,-1,-1).reshape(B,attn_module.nheads,T,Dk)
            Vp = Vp.unsqueeze(2).expand(-1,-1,ng,-1,-1).reshape(B,attn_module.nheads,T,Dv)
        return Qp, Kp, Vp

    @staticmethod
    def _ring_shift(
        tensor: Tensor,
        strategy: RingAttentionStrategy,
        cur_len: int
    ) -> Tuple[Tensor, int]:
        r, w = strategy.rank, strategy.world_size
        bs = strategy.block_size
        sp = (r + 1) % w
        rp = (r - 1 + w) % w
        dev = tensor.device
        d = 2
        if cur_len == 0:
            shp = list(tensor.shape)
            shp[d] = 0
            send = torch.empty(*shp, dtype=tensor.dtype, device=dev)
        else:
            idx = [slice(None)] * tensor.ndim
            idx[d] = slice(0, cur_len)
            send = tensor[tuple(idx)]
        send = RingAttentionKernel._pad_to_block(send, bs, dim=d).contiguous()
        recv = torch.empty_like(send)
        recv_len = torch.empty(1, dtype=torch.int32, device=dev)
        sent_len = torch.tensor([cur_len], dtype=torch.int32, device=dev)
        ops = [
            P2POp(dist.isend, sent_len, peer=sp),
            P2POp(dist.irecv, recv_len, peer=rp),
            P2POp(dist.isend, send, peer=sp),
            P2POp(dist.irecv, recv, peer=rp),
        ]
        for req in dist.batch_isend_irecv(ops): req.wait()
        nl = recv_len.item()
        assert 0 <= nl <= bs
        idx2 = [slice(None)] * recv.ndim
        idx2[d] = slice(0, nl)
        return recv[tuple(idx2)].contiguous(), nl

    @staticmethod
    def _compute_attention_scores(
        q: Tensor, k: Tensor,
        qi: Tensor, ki: Tensor,
        scale: float,
        mask: Optional[Tensor] = None,
        causal: bool = False
    ) -> Tensor:
        B, H, Tq, _ = q.shape
        _, _, Tk, _ = k.shape
        if Tq==0 or Tk==0:
            return torch.empty(B,H,Tq,Tk, device=q.device, dtype=q.dtype)
        sc = torch.matmul(q/scale, k.transpose(-2,-1))
        if mask is not None:
            sc = sc + mask.to(sc.dtype)
        if causal:
            cond = ki.unsqueeze(0) > qi.unsqueeze(1)
            sc = sc.masked_fill(cond.unsqueeze(0).unsqueeze(0), float('-inf'))
        return sc

    @staticmethod
    def _compute_max_pass(
        ql: Tensor, kl0: Tensor,
        mask: Optional[Tensor],
        q0: int, ql_len: int,
        strategy: RingAttentionStrategy,
        scale: float,
        causal_flag: bool
    ) -> Tensor:
        B,H,Tq,Dk = ql.shape
        dev = ql.device
        ad = RingAttentionKernel._get_accum_dtype(ql.dtype)
        msa = torch.full((B,H,Tq,1), torch.finfo(ad).min, device=dev, dtype=ad)
        if Tq==0:
            return msa
        qi = torch.arange(q0, q0+Tq, device=dev)
        # Convert ql and kl0 to accum_dtype once before the loop
        ql_ad = ql.to(ad)
        kb = kl0.to(ad) 
        kl_len = kl0.shape[2]
        for i in range(strategy.world_size):
            kr = (strategy.rank - i + strategy.world_size) % strategy.world_size
            ks = kr * strategy.block_size
            if kl_len>0:
                ki = torch.arange(ks, ks+kl_len, device=dev)
                lm = None
                if mask is not None:
                    qs = slice(q0, q0+Tq)
                    kslice = slice(ks, ks+kl_len)
                    lm = mask[:, :, qs, kslice]
                s = RingAttentionKernel._compute_attention_scores(
                    ql_ad, kb, qi, ki, scale, mask=lm, causal=causal_flag
                )
                if s.numel():
                    msa = torch.maximum(msa, s.amax(dim=-1, keepdim=True))
            if i < strategy.world_size-1:
                kb, kl_len = RingAttentionKernel._ring_shift(kb, strategy, kl_len)
        return msa

    @staticmethod
    def _compute_sum_pass(
        q_local: Tensor,
        k_local_initial: Tensor,
        v_local_initial: Tensor,
        mask_global: Optional[Tensor],
        q_start_global: int,
        local_query_len: int,
        max_score: Tensor,
        exp_min: float,
        exp_max: float,
        strategy: RingAttentionStrategy,
        scale: float,
        accum_dtype: torch.dtype,
        is_causal_mask_flag: bool
    ) -> Tuple[Tensor, Tensor]:
        B, H, Tq, Dk = q_local.shape
        Dev = q_local.device
        numerator = torch.zeros(B, H, Tq, v_local_initial.shape[-1], device=Dev, dtype=accum_dtype)
        denominator = torch.zeros(B, H, Tq, 1, device=Dev, dtype=accum_dtype) # Kahan comp removed

        if Tq == 0:
            return numerator, denominator
        q_idx = torch.arange(q_start_global, q_start_global + Tq, device=Dev)
        # Convert q_local, k_local_initial, v_local_initial to accum_dtype once before the loop
        q_local_ad = q_local.to(accum_dtype)
        k_blk = k_local_initial.to(accum_dtype) 
        v_blk = v_local_initial.to(accum_dtype) 
        cur_len = k_local_initial.shape[2]
        for i in range(strategy.world_size):
            kr = (strategy.rank - i) % strategy.world_size
            ks = kr * strategy.block_size
            if cur_len > 0:
                ki = torch.arange(ks, ks + cur_len, device=Dev)
                lm = None
                if mask_global is not None:
                    qs = slice(q_start_global, q_start_global + Tq)
                    kslice = slice(ks, ks + cur_len)
                    lm = mask_global[:, :, qs, kslice]
                s = RingAttentionKernel._compute_attention_scores(
                    q_local_ad, k_blk, q_idx, ki, scale,
                    mask=lm, causal=is_causal_mask_flag
                )
                delta = torch.where(torch.isneginf(max_score), torch.full_like(s, -torch.inf), s - max_score)
                clamped = delta.clamp(min=exp_min, max=exp_max)
                exp_s = torch.exp(clamped)
                exp_s = exp_s.masked_fill(torch.isneginf(max_score.expand_as(s)), 0.0)
                # Direct accumulation instead of Kahan summation
                numerator = numerator + torch.matmul(exp_s, v_blk)
                denominator = denominator + exp_s.sum(dim=-1, keepdim=True)
            if i < strategy.world_size - 1:
                k_blk, kl = RingAttentionKernel._ring_shift(k_blk, strategy, cur_len)
                v_blk, vl = RingAttentionKernel._ring_shift(v_blk, strategy, cur_len)
                assert kl == vl
                cur_len = kl
        return numerator, denominator

    @staticmethod
    def _three_pass(
        q_local_valid: Tensor,
        k_local_valid: Tensor,
        v_local_valid: Tensor,
        mask_global: Optional[Tensor],
        strategy: RingAttentionStrategy,
        q_start_global: int,
        local_query_len: int,
        scale: float,
        accum_dtype: torch.dtype,
        is_causal_mask_flag: bool
    ) -> Tensor:
        exp_min, exp_max = RingAttentionKernel._get_clamp_bounds(accum_dtype)
        max_scores = RingAttentionKernel._compute_max_pass(
            q_local_valid, k_local_valid, mask_global,
            q_start_global, local_query_len,
            strategy, scale, is_causal_mask_flag
        )
        numerator, denominator = RingAttentionKernel._compute_sum_pass(
            q_local_valid, k_local_valid, v_local_valid,
            mask_global, q_start_global, local_query_len,
            max_scores, exp_min, exp_max,
            strategy, scale, accum_dtype, is_causal_mask_flag
        )
        if local_query_len == 0:
            B, H, _, Dv = v_local_valid.shape
            return torch.empty(B, H, 0, Dv, device=q_local_valid.device, dtype=q_local_valid.dtype)
        eps = torch.finfo(denominator.dtype).eps
        return (numerator / (denominator + eps)).to(q_local_valid.dtype)

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
        B, T_padded, E = x_norm.shape
        rank, world = strategy.rank, strategy.world_size
        bs = strategy.block_size
        if world > 1:
            assert T_padded == bs
        q_start = rank * bs
        x_trim = x_norm[:, :valid_len, :]
        res_trim = residual[:, :valid_len, :]
        if position_ids is not None:
            pos_trim = position_ids[:, :valid_len]
        elif valid_len > 0:
            idx = torch.arange(q_start, q_start + valid_len, device=x_norm.device)
            pos_trim = idx.unsqueeze(0).expand(B, -1)
        else:
            pos_trim = None
        if valid_len > 0:
            qv, kv, vv = RingAttentionKernel._compute_qkv_and_rope(attn_module, x_trim, pos_trim)
        else:
            Hq = attn_module.nheads
            Dk, Dv = attn_module.emb_kq_per_head, attn_module.emb_v_per_head
            qv = torch.empty(B, Hq, 0, Dk, dtype=x_norm.dtype, device=x_norm.device)
            kv = torch.empty(B, Hq, 0, Dk, dtype=x_norm.dtype, device=x_norm.device)
            vv = torch.empty(B, Hq, 0, Dv, dtype=x_norm.dtype, device=x_norm.device)
        cache = (kv, vv) if valid_len > 0 else None
        scale = attn_module.scale_factor or math.sqrt(attn_module.emb_kq_per_head)
        accum_dtype = RingAttentionKernel._get_accum_dtype(qv.dtype)
        attn_out = RingAttentionKernel._three_pass(
            qv, kv, vv, mask, strategy, q_start,
            valid_len, scale, accum_dtype, is_causal_mask
        )
        if valid_len > 0:
            proj = attn_out.transpose(1, 2).contiguous().view(B, valid_len, -1)
            dense_out = attn_module.dense(proj)
            if hasattr(attn_module, 'dropout') and isinstance(attn_module.dropout, nn.Dropout):
                dense_out = attn_module.dropout(dense_out)
            x_attn = res_trim + dense_out
            ffn_out = ff(ff_norm(x_attn))
            out_valid = x_attn + ffn_out
        else:
            out_valid = torch.empty(B, 0, E, dtype=x_norm.dtype, device=x_norm.device)
        if world > 1:
            output = RingAttentionKernel._pad_to_block(out_valid, bs, dim=1)
        else:
            output = out_valid
        return output, cache
