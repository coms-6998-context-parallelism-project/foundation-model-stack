from typing import Any, Optional, Tuple
import sys
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import RingAttentionStrategy

def compare_tensors(label, t1, t2, threshold=1e-6):
    if t1.dtype != t2.dtype:
        print(f"[{label}] WARNING: dtype mismatch {t1.dtype} vs {t2.dtype}. Casting to float32.")
        t1, t2 = t1.float(), t2.float()
    diff = (t1 - t2).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    count = (diff > threshold).sum().item()
    print(f"[{label}] shape={t1.shape}, max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}, diff>{threshold}: {count}")
    if count > 0:
        flat = diff.view(-1)
        for idx in torch.topk(flat, min(5, flat.numel())).indices:
            idxu = list(torch.unravel_index(idx, t1.shape))
            v1, v2 = t1[tuple(idxu)].item(), t2[tuple(idxu)].item()
            print(f"    idx={idxu}, t1={v1:.4e}, t2={v2:.4e}, diff={abs(v1-v2):.4e}")

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
    if attn_data.position_encoder and T > 0:
        mask = position_ids.ne(-1)
        if mask.any():
            maxpos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
            pid = position_ids.clone().clamp(0, maxpos - 1)
            q_r, k_r = attn_data.position_encoder.adjusted_qk(Q, K, pid)
            Q = torch.where(mask.unsqueeze(-1).unsqueeze(-1), q_r, Q)
            K = torch.where(mask.unsqueeze(-1).unsqueeze(-1).expand_as(K), k_r, K)
    return Q.permute(0, 2, 1, 3), K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3)

def _pad_to_block(t: torch.Tensor, target_len: int, dim: int = 2) -> torch.Tensor:
    if t.size(dim) >= target_len:
        assert t.size(dim) == target_len
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
        debug_dict_populate: Optional[dict] = None,
        debug_key_prefix_populate: str = "",
        layer_idx: int = -1,
        debug_print_values: bool = False,
        debug_tolerance: float = 1e-3,
    ):
        assert isinstance(strategy, RingAttentionStrategy) and self.strategy is strategy
        B, Tp, _ = x_norm.shape
        assert Tp == self.block_size if self.world_size > 1 else True
        can_debug = (
            self.llama_block.config.debug_mode and
            layer_idx == self.llama_block.config.debug_target_layer and
            self.rank in [0, 1] and
            debug_dict_populate is not None
        )
        start = self.rank * self.block_size
        if position_ids is None:
            position_ids = torch.full((B, Tp), -1, dtype=torch.long, device=x_norm.device)
            if valid_len > 0:
                position_ids[:, :valid_len] = torch.arange(start, start + valid_len, device=x_norm.device)
        x_trim = x_norm[:, :valid_len, :]
        pos_trim = position_ids[:, :valid_len]
        res_trim = residual[:, :valid_len, :] if residual is not None else None
        if can_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_x_norm"] = x_trim.clone().cpu()
            if res_trim is not None:
                debug_dict_populate[f"{debug_key_prefix_populate}_residual_input"] = res_trim.clone().cpu()
        q, k, v = compute_local_qkv_and_rope(self.attn, x_trim, x_trim, x_trim, pos_trim)
        if can_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_q_local"] = q.clone().cpu()
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_k_local"] = k.clone().cpu()
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_v_local"] = v.clone().cpu()
        if self.attn.nheads != self.attn.kvheads:
            e = self.attn.nheads // self.attn.kvheads
            k = k.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_kq_per_head)
            v = v.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_v_per_head)
        out_valid = self.forward_full(
            q, k, v, mask, valid_len, res_trim, start, is_causal_mask,
            can_debug, debug_dict_populate, debug_key_prefix_populate, layer_idx
        )
        if can_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_block_output"] = out_valid.clone().cpu()
        return (_pad_to_block(out_valid, self.block_size, dim=1), None, None) if self.world_size > 1 else (out_valid, None, None)

    def forward_full(
        self,
        q, k, v, mask, valid_len_local, x_block_residual_input,
        q_start_global, is_causal_mask: bool,
        can_debug: bool = False,
        debug_dict_populate: Optional[dict] = None,
        debug_key_prefix_populate: str = "",
        layer_idx: int = -1
    ):
        B, H, T, Dk = q.shape
        _, _, _, Dv = v.shape
        accum_dtype = self._accum_dtype
        exp_min, exp_max = _get_clamp_bounds(accum_dtype)
        qc, kc, vc = q.to(accum_dtype), k.to(accum_dtype), v.to(accum_dtype)
        max_score = self._compute_max_score_pass(
            qc, kc, mask, q_start_global, valid_len_local,
            is_causal_mask, can_debug, debug_dict_populate, debug_key_prefix_populate, layer_idx
        )
        num, den = self._compute_sums_pass(
            qc, kc, vc, mask, q_start_global, valid_len_local, max_score, exp_min, exp_max,
            is_causal_mask, can_debug, debug_dict_populate, debug_key_prefix_populate, layer_idx
        )
        if T > 0:
            eps = torch.finfo(den.dtype).eps
            attn_manual = (num / (den + eps)).to(q.dtype)
            attn = attn_manual
            if self.attn.p_dropout > 0 and self.llama_block.training:
                attn = F.dropout(attn, p=self.attn.p_dropout, training=True)
            if can_debug:
                debug_dict_populate[f"{debug_key_prefix_populate}_context_raw"] = attn.clone().cpu()
            attn_out = self.attn.dense(attn.transpose(1, 2).contiguous().view(B, T, H * Dv))
            if can_debug:
                debug_dict_populate[f"{debug_key_prefix_populate}_attn_out_dense"] = attn_out.clone().cpu()
        else:
            attn_out = torch.empty(B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype)
            if can_debug:
                debug_dict_populate[f"{debug_key_prefix_populate}_context_raw"] = torch.empty(B, H, 0, Dv, device=q.device, dtype=q.dtype).cpu()
                debug_dict_populate[f"{debug_key_prefix_populate}_attn_out_dense"] = attn_out.clone().cpu()
        x_after = x_block_residual_input + attn_out
        if can_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_out_residual"] = x_after.clone().cpu()
        ff_ln_out = self.ff_norm(x_after)
        if can_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_ff_ln_out"] = ff_ln_out.clone().cpu()
        ff_out_raw = self.ff(ff_ln_out)
        if can_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_ff_out_raw"] = ff_out_raw.clone().cpu()
        return ff_out_raw + x_after

    def _compute_max_score_pass(self, q, k, mask, q_start, valid_len, is_causal_mask, can_debug, debug_dict, prefix, layer_idx):
        B, H, T, _ = q.shape
        dev, dtype = q.device, q.dtype
        max_score = torch.full((B, H, T, 1), torch.finfo(dtype).min, device=dev, dtype=dtype)
        q_idx = torch.arange(q_start, q_start + T, device=dev)
        k_blk, k_len = k, k.shape[2]
        start_idx = ((self.rank) % self.world_size) * self.block_size
        for i in range(self.world_size):
            k_idx = torch.arange(start_idx, start_idx + k_len, device=dev)
            m = mask[:, :, q_start:q_start+T, start_idx:start_idx+k_len] if mask is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(q, k_blk[:, :, :k_len], q_idx, k_idx, m, True, is_causal_mask, can_debug and i==0, debug_dict, f"{prefix}_maxpass_kblock0", layer_idx)
                max_score = torch.maximum(max_score, s.amax(-1, keepdim=True))
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, k_len)
                start_idx = ((self.rank - (i+1)) % self.world_size) * self.block_size
        return max_score

    def _compute_sums_pass(self, q, k, v, mask, q_start, valid_len, max_score, exp_min, exp_max, is_causal_mask, can_debug, debug_dict, prefix, layer_idx):
        B, H, T, _ = q.shape
        Dv = v.shape[3]
        dev = q.device
        num = torch.zeros(B, H, T, Dv, device=dev, dtype=self._accum_dtype)
        den = torch.zeros(B, H, T, 1, device=dev, dtype=self._accum_dtype)
        num_comp = torch.zeros_like(num)
        den_comp = torch.zeros_like(den)
        q_idx = torch.arange(q_start, q_start + T, device=dev)
        k_full = _pad_to_block(k, self.block_size, dim=2)
        v_full = _pad_to_block(v, self.block_size, dim=2)
        k_blk, v_blk, k_len = k_full, v_full, k_full.shape[2]
        start_idx = ((self.rank) % self.world_size) * self.block_size
        total_min = total_max = 0
        for i in range(self.world_size):
            k_idx = torch.arange(start_idx, start_idx + k_len, device=dev)
            m = mask[:, :, q_start:q_start+T, start_idx:start_idx+k_len] if mask is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(q, k_blk[:, :, :k_len], q_idx, k_idx, m, True, is_causal_mask, can_debug and i==0, debug_dict, f"{prefix}_sumpass_kblock0", layer_idx)
                delta = s - max_score
                clamp = delta.clamp(min=exp_min, max=exp_max)
                e = torch.exp(clamp.masked_fill(torch.isneginf(clamp), exp_min))
                e = e.masked_fill(torch.isneginf(max_score.expand_as(s)), 0)
                contrib_num = torch.matmul(e, v_blk[:, :, :k_len])
                contrib_den = e.sum(-1, keepdim=True)
                y = contrib_num - num_comp; t = num + y; num_comp = (t - num) - y; num = t
                y = contrib_den - den_comp; t = den + y; den_comp = (t - den) - y; den = t
                if can_debug and i == 0:
                    # Populate the scores and probabilities for the first k-block
                    # `prefix` is the main debug key prefix (e.g., "ring_r0")
                    debug_dict[f"{prefix}_sdp_scores_kblock0"] = s.clone().cpu()
                    debug_dict[f"{prefix}_sdp_probs_kblock0"] = e.clone().cpu() # `e` is P_unnorm = exp(S-max(S))

                    total_min += (clamp == exp_min).sum().item()
                    total_max += (clamp == exp_max).sum().item()
                    # The 'stats' dictionary (max/min/mean of e) is not directly consumed by _compare_debug_data
                    # for the _sdp_probs_kblock0 key. If these stats are needed for other debug purposes,
                    # they should be stored under a different key.
                    # For now, we remove the population of the 'stats' variable as it's not used for the
                    # primary comparison keys and was the source of the AttributeError.

            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, k_len)
                v_blk, _ = self._ring_shift_tensor(v_blk, self.block_size, k_len)
                start_idx = ((self.rank - (i+1)) % self.world_size) * self.block_size
        if can_debug:
            debug_dict[f"{prefix}_clamped_min_total_count"] = torch.tensor(total_min).cpu()
            debug_dict[f"{prefix}_clamped_max_total_count"] = torch.tensor(total_max).cpu()
            # The keys _sdp_scores_kblock0 and _sdp_probs_kblock0 are now populated
            # inside the loop when i == 0.
            debug_dict[f"{prefix}_kahan_num_comp_norm"] = torch.linalg.norm(num_comp.float()).cpu()
            debug_dict[f"{prefix}_kahan_den_comp_norm"] = torch.linalg.norm(den_comp.float()).cpu()
        return num, den

    def _compute_attention_scores(
        self, q, k, q_idx, k_idx, mask=None, apply_mask=True, apply_causal=True,
        can_debug=False, debug_dict=None, prefix="", layer_idx=-1
    ):
        B, H, Tq, _ = q.shape
        Tk = k.shape[2]
        if Tq == 0:
            return torch.empty(B, H, 0, Tk, device=q.device, dtype=q.dtype)
        scores = torch.matmul(q / self.scale, k.transpose(-2, -1))
        if apply_mask and mask is not None:
            assert mask.shape == scores.shape
            if can_debug:
                debug_dict[f"{prefix}_mask_slice_sum"] = mask.sum().cpu()
            scores += mask.to(scores.dtype)
        if apply_causal:
            causal = k_idx[None, None, None, :] > q_idx[None, None, :, None]
            scores = scores.masked_fill(causal, float('-inf'))
            if can_debug:
                debug_dict[f"{prefix}_causal_mask_sum"] = causal.sum().cpu()
        if self.rank == 0 and Tq > 0 and can_debug:
            print(scores[0, 0, 0, :5].tolist(), flush=True)
        return scores

    def _ring_shift_tensor(self, tensor, pad_len, valid_len):
        rank, world = self.rank, self.world_size
        send, recv = (rank + 1) % world, (rank - 1) % world
        send_tensor = _pad_to_block(tensor[:, :, :valid_len], pad_len, -2).contiguous()
        recv_tensor = torch.empty_like(send_tensor)
        recv_len = torch.empty(1, dtype=torch.int32, device=send_tensor.device)
        ops = [
            dist.P2POp(dist.isend, torch.tensor([valid_len], dtype=torch.int32, device=send_tensor.device), peer=send),
            dist.P2POp(dist.irecv, recv_len, peer=recv),
            dist.P2POp(dist.isend, send_tensor, peer=send),
            dist.P2POp(dist.irecv, recv_tensor, peer=recv),
        ]
        for r in dist.batch_isend_irecv(ops): r.wait()
        rlen = recv_len.item()
        assert 0 <= rlen <= pad_len
        return recv_tensor[:, :, :rlen].contiguous(), rlen
