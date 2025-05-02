from typing import Tuple, Dict, Optional
import torch
import torch.distributed as dist
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
    def __init__(self, attn_module, layer_idx, strategy, llama_block, use_cache=False, debug_mode: bool = False, minimal_debug_prints: bool = False):
        self.attn = attn_module
        self.layer_idx = layer_idx
        self.strategy = strategy
        self.use_cache = use_cache
        self.debug_mode = debug_mode
        self.llama_block = llama_block
        self.minimal_debug_prints = minimal_debug_prints
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, x_norm, strategy, mask=None, position_ids=None, past_key_value_state=None,
            is_causal_mask=False, rank=0, minimal_debug_prints: bool = False, valid_len=0):

        if self.debug_mode:
            dist.barrier()

        if strategy.world_size > 1:
            x_norm_gathered = strategy.gather_tensor(x_norm, dim=1)
        else:
            x_norm_gathered = x_norm

        if self.debug_mode:
            dist.barrier()

        q_full, k_full, v_full = self.llama_block.compute_local_qkv_and_rope(
            self.attn, q=x_norm_gathered, k=x_norm_gathered, v=x_norm_gathered,
            position_ids=position_ids, use_cache=False, past_key_value_state=past_key_value_state, is_self=True
        )

        start_idx = self.rank * strategy.block_size
        real_T = valid_len

        # No padding for computation
        q_local = q_full[:, :, start_idx:start_idx + real_T, :]
        k_local = k_full[:, :, start_idx:start_idx + real_T, :]
        v_local = v_full[:, :, start_idx:start_idx + real_T, :]

        if self.debug_mode:
            debug_info = {
                f"q_local_r{self.rank}": q_local.detach().cpu(),
                f"k_local_r{self.rank}": k_local.detach().cpu(),
                f"v_local_r{self.rank}": v_local.detach().cpu(),
                f"x_norm_r{self.rank}": x_norm.detach().cpu(),
            }
        else:
            debug_info = None

        B, H, T_q, D = q_local.shape
        D_v = self.attn.emb_v_per_head
        scale = math.sqrt(D)

        max_score = torch.full((B, H, T_q, 1), -float("inf"), device=q_local.device, dtype=torch.float32)
        numerator = torch.zeros(B, H, T_q, D_v, device=q_local.device, dtype=torch.float32)
        denominator = torch.zeros(B, H, T_q, 1, device=q_local.device, dtype=torch.float32)

        q_indices = torch.arange(start_idx, start_idx + T_q, device=q_local.device)

        k = k_local
        v = v_local

        for i in range(self.world_size):
            k_start_idx = ((self.rank - i) % self.world_size) * strategy.block_size
            k_indices = torch.arange(k_start_idx, k_start_idx + k.shape[2], device=q_local.device)

            scores = torch.einsum("bhqd,bhkd->bhqk", q_local, k) / scale

            if self.debug_mode and debug_info is not None:
                debug_info[f"raw_scores_step{i}_r{self.rank}"] = scores.clone().detach().cpu()

            if mask is not None:
                scores += mask

            if is_causal_mask:
                causal_mask = (k_indices[None, :] > q_indices[:, None]).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask, -torch.inf)

            if self.debug_mode and debug_info is not None:
                debug_info[f"scores_step{i}_r{self.rank}"] = scores.clone().detach().cpu()

            block_max = scores.masked_fill(scores == -torch.inf, torch.finfo(scores.dtype).min).amax(dim=-1, keepdim=True)
            max_score = torch.maximum(max_score, block_max)

            stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0).to(torch.float32)
            exp_scores = torch.exp(stable_scores)
            exp_scores = torch.nan_to_num(exp_scores, nan=0.0, posinf=1e4, neginf=0.0)

            v_f32 = v.to(torch.float32)
            numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v_f32)
            denominator += exp_scores.sum(dim=-1, keepdim=True)

            if i < self.world_size - 1:
                k, k_valid_len = self._ring_shift_tensor(k, strategy.block_size)
                v, v_valid_len = self._ring_shift_tensor(v, strategy.block_size)
                k = k[:, :, :k_valid_len, :]
                v = v[:, :, :v_valid_len, :]

                if self.debug_mode and debug_info is not None:
                    debug_info[f"k_input_step{i+1}_r{self.rank}"] = k.clone().detach().cpu()
                    debug_info[f"v_input_step{i+1}_r{self.rank}"] = v.clone().detach().cpu()

        if self.debug_mode and debug_info is not None:
            debug_info[f"max_score_r{self.rank}"] = max_score.clone().detach().cpu()
            debug_info[f"numerator_r{self.rank}"] = numerator.clone().detach().cpu()
            debug_info[f"denominator_r{self.rank}"] = denominator.clone().detach().cpu()

        attn_out = (numerator / (denominator + 1e-10)).to(q_local.dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, H * D_v)
        attn_out = self.attn.dense(attn_out)

        if self.debug_mode and debug_info is not None:
            debug_info[f"attn_out_raw_r{self.rank}"] = attn_out.clone().detach().cpu()

        attn_out = attn_out[:, :real_T, :]
        x_residual = x_norm[:, :real_T, :]

        if self.llama_block.config.p_dropout != 0:
            attn_out = self.llama_block.dropout(attn_out)

        x = attn_out + x_residual

        if self.debug_mode and debug_info is not None:
            debug_info[f"attn_out_residual_r{self.rank}"] = x.clone().detach().cpu()

        residual = x
        ff_ln_out = self.llama_block.ff_ln(residual)

        if self.debug_mode and debug_info is not None:
            debug_info[f"ff_ln_out_r{self.rank}"] = ff_ln_out.clone().detach().cpu()

        ff_out_raw = self.llama_block.ff_sub_layer(ff_ln_out)

        if self.debug_mode and debug_info is not None:
            debug_info[f"ff_out_raw_r{self.rank}"] = ff_out_raw.clone().detach().cpu()

        if self.llama_block.config.p_dropout != 0:
            ff_out_raw = self.llama_block.dropout(ff_out_raw)

        x = ff_out_raw + residual

        if self.debug_mode and debug_info is not None:
            debug_info[f"block_output_r{self.rank}"] = x.clone().detach().cpu()

        return (x, None, debug_info) if self.debug_mode else (x, None)


    def _ring_shift_tensor(self, tensor: torch.Tensor, pad_len: int) -> Tuple[torch.Tensor, int]:
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1 + self.world_size) % self.world_size

        valid_len = tensor.shape[2]
        tensor_to_send = _pad_to_block(tensor, pad_len, dim=2).contiguous()

        send_len = torch.tensor([valid_len], dtype=torch.int32, device=tensor.device)
        recv_len = torch.empty(1, dtype=torch.int32, device=tensor.device)

        len_send_req = dist.isend(send_len, dst=send_rank)
        len_recv_req = dist.irecv(recv_len, src=recv_rank)
        len_send_req.wait()
        len_recv_req.wait()

        tensor_recv = torch.empty_like(tensor_to_send)
        tensor_send_req = dist.isend(tensor_to_send, dst=send_rank)
        tensor_recv_req = dist.irecv(tensor_recv, src=recv_rank)
        tensor_send_req.wait()
        tensor_recv_req.wait()

        return tensor_recv, recv_len.item()
