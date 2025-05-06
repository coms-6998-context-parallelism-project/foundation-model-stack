from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.distributed as dist
from torch.distributed import P2POp
import math

def _pad_to_block(t, target_len, dim=2):
    pad_len = target_len - t.shape[dim]
    if pad_len <= 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)

class RingAttentionHelper:
    def __init__(self, attn_module, strategy, llama_block, use_cache=False, debug_mode=False, minimal_debug_prints=False, ff=None, ff_norm=None):
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.strategy = strategy
        self.use_cache = use_cache
        self.llama_block = llama_block
        self.debug_mode = debug_mode
        self.minimal_debug_prints = minimal_debug_prints
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.head_dim = attn_module.emb_kq_per_head
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x_norm, strategy, mask=None, position_ids=None, past_key_value_state=None,
                is_causal_mask=False, rank=0, valid_len=0, residual=None):
        start_idx_global = self.rank * self.strategy.block_size
        B, T = x_norm.shape[:2]

        if position_ids is None:
            position_ids = torch.full((B, T), fill_value=-1, dtype=torch.long, device=x_norm.device)
            if valid_len > 0:
                valid_pos = torch.arange(start_idx_global, start_idx_global + valid_len, device=x_norm.device)
                position_ids[:, :valid_len] = valid_pos.unsqueeze(0)

        if self.debug_mode:
            print(f"[Rank {self.rank}] valid_len: {valid_len}, position_ids used: {position_ids[0, :valid_len].tolist() if valid_len > 0 else '[]'}")

        q_local, k_local, v_local = self.llama_block.compute_local_qkv_and_rope(
            self.attn,
            q=x_norm, k=x_norm, v=x_norm,
            position_ids=position_ids,
            use_cache=False,
            past_key_value_state=past_key_value_state,
            is_self=True
        )

        q_local = q_local[:, :, :valid_len, :]
        k_local = k_local[:, :, :valid_len, :]
        v_local = v_local[:, :, :valid_len, :]

        x_norm_local = x_norm[:, :valid_len, :]
        residual_local = residual[:, :valid_len, :] if residual is not None else None

        result = self.forward_full(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            mask_global=mask,
            x_block=residual_local,
            x_norm_block=x_norm_local,
            valid_len=valid_len,
            q_start_global=start_idx_global
        )

        if self.debug_mode:
            x, debug_info = result
            return x, None, debug_info
        else:
            return result, None, None

    def forward_full(self, q_local, k_local, v_local, mask_global, valid_len, x_block, x_norm_block, q_start_global):
        B, H, T_q_local, D_head = q_local.shape
        D_v = self.attn.emb_v_per_head

        debug_info = {} if self.debug_mode else None

        max_score = self._compute_max_score_pass(
            q_local, k_local, mask_global, q_start_global, T_q_local, debug_info
        )

        numerator, denominator = self._compute_sums_pass(
            q_local, k_local, v_local, mask_global, q_start_global, T_q_local, max_score, debug_info
        )

        attn_out_h = numerator / (denominator + 1e-10)
        attn_out = attn_out_h.to(q_local.dtype)

        if self.debug_mode and debug_info is not None:
            debug_info[f"attn_out_raw_r{self.rank}"] = attn_out.clone().detach().cpu()

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T_q_local, H * D_v)
        attn_out = self.attn.dense(attn_out)
        residual_1 = x_block + attn_out

        ff_ln_out = self.ff_norm(residual_1)
        ff_out = self.ff(ff_ln_out)
        x = ff_out + residual_1

        if x.dtype == torch.float16:
            x = torch.clamp(x, min=-5.0, max=5.0)

        if self.debug_mode and debug_info is not None:
            debug_info[f"block_output_r{self.rank}"] = x.clone().detach().cpu()
            return x, debug_info
        return x

    def _compute_max_score_pass(self, q_local, k_local, mask_global, q_start_global, valid_len_local, debug_info):
        B, H, T_q_local, _ = q_local.shape
        device = q_local.device
        dtype = torch.float32

        q_local_f32 = q_local.to(dtype)
        k_local_f32 = k_local.to(dtype)
        max_score = torch.full((B, H, T_q_local, 1), -float("inf"), device=device, dtype=dtype)
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local_f32
        current_k_len = current_k_block.shape[2]

        for i in range(self.world_size):
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None

            if current_k_len == 0:
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                continue

            scores = self._compute_attention_scores(q_local, current_k_block[:, :, :current_k_len, :],
                                                    q_indices_global, k_indices_global, current_mask,
                                                    apply_mask=True, keep_causal=True)

            if self.debug_mode and debug_info is not None:
                debug_info[f"scores_step{i}_r{self.rank}"] = scores.clone().detach().cpu()

            max_score = self._update_max_score(scores, max_score)

            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)

        return max_score

    def _compute_sums_pass(self, q_local, k_local, v_local, mask_global, q_start_global,
                           valid_len_local, max_score, debug_info=None):
        B, H, T_q_local, _ = q_local.shape
        D_v = self.attn.emb_v_per_head
        device = q_local.device
        dtype = torch.float32

        q_local_f32 = q_local.to(dtype)
        k_local_f32 = k_local.to(dtype)
        v_local_f32 = v_local.to(dtype)

        numerator = torch.zeros(B, H, T_q_local, D_v, device=device, dtype=dtype)
        denominator = torch.zeros(B, H, T_q_local, 1, device=device, dtype=dtype)
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local_f32
        current_v_block = v_local_f32
        current_k_len = current_k_block.shape[2]

        for i in range(self.world_size):
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None and current_k_len > 0 else None

            if current_k_len == 0:
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                    current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size)
                continue

            scores = self._compute_attention_scores(q_local, current_k_block[:, :, :current_k_len, :],
                                                    q_indices_global, k_indices_global, current_mask, apply_mask=True)
            numerator, denominator = self._update_totals(scores, current_v_block[:, :, :current_k_len, :], max_score, numerator, denominator)

            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size)

        return numerator, denominator

    def _compute_attention_scores(self, q, k, q_indices_global, k_indices_global, mask=None,
                                  apply_mask=True, keep_causal=True):
        scores = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32)) / self.scale
        scores = scores.clamp(min=-30.0, max=30.0)

        if apply_mask and mask is not None:
            scores = scores + mask.to(scores.dtype)
        if apply_mask and keep_causal:
            causal_mask = (k_indices_global[None, :] > q_indices_global[:, None]).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, -float("inf"))

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print(f"[Rank {self.rank}] NaN or Inf in scores! q shape: {q.shape}, k shape: {k.shape}")
        return scores

    def _update_max_score(self, scores, current_max):
        block_max = scores.masked_fill(scores == -torch.inf, torch.finfo(scores.dtype).min).amax(dim=-1, keepdim=True)
        return torch.maximum(current_max, block_max)

    def _update_totals(self, scores, v, max_score, numerator, denominator):
        stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0)
        exp_scores = torch.exp(stable_scores)
        numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v.to(numerator.dtype))
        denominator += exp_scores.sum(dim=-1, keepdim=True)
        return numerator, denominator

    def _ring_shift_tensor(self, tensor, pad_len):
        rank, world = self.rank, self.world_size
        send_rank = (rank + 1) % world
        recv_rank = (rank - 1 + world) % world
        valid_len = tensor.shape[-2]

        padded = _pad_to_block(tensor, pad_len, dim=-2).contiguous()
        assert padded.shape[-2] == pad_len, f"[Rank {self.rank}] Padded shape mismatch: got {padded.shape[-2]}, expected {pad_len}"
        device = tensor.device

        send_len = torch.tensor([valid_len], dtype=torch.int32, device=device)
        recv_len = torch.empty(1, dtype=torch.int32, device=device)
        tensor_recv = torch.empty_like(padded)

        ops = [
            P2POp(op=dist.isend, tensor=send_len, peer=send_rank),
            P2POp(op=dist.irecv, tensor=recv_len, peer=recv_rank),
            P2POp(op=dist.isend, tensor=padded, peer=send_rank),
            P2POp(op=dist.irecv, tensor=tensor_recv, peer=recv_rank),
        ]

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        recv_len = recv_len.item()
        assert recv_len <= pad_len, f"[Rank {self.rank}] Received more data ({recv_len}) than expected ({pad_len})"

        if self.debug_mode:
            print(f"[Rank {self.rank}] Ring shift recv_len: {recv_len}, shape: {tensor_recv.shape}")

        tensor_recv = tensor_recv[:, :, :recv_len, :].contiguous()
        return tensor_recv, recv_len