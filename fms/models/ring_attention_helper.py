from typing import Tuple, Dict, Optional
import torch
import torch.distributed as dist
import math

class RingAttentionHelper:
    def __init__(self, attn_module, layer_idx, strategy, llama_block, use_cache=False, debug_mode: bool = False):
        self.attn = attn_module
        self.layer_idx = layer_idx
        self.strategy = strategy
        self.use_cache = use_cache
        self.debug_mode = debug_mode
        self.llama_block = llama_block
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, x_norm, strategy, mask=None, position_ids=None, past_key_value_state=None, is_causal_mask=False, rank=0):
        # Log input shape before gather
        print(f"[Rank {self.rank}] x_norm shape before gather: {x_norm.shape}")
        dist.barrier()

        if strategy.world_size > 1:
            x_norm_gathered = strategy.gather_tensor(x_norm, dim=1)
            print(f"[Rank {self.rank}] x_norm_gathered shape: {x_norm_gathered.shape}")
        else:
            x_norm_gathered = x_norm

        dist.barrier()
        print(f"[Rank {self.rank}] Computing QKV...")
        q_full, k_full, v_full = self.llama_block.compute_local_qkv_and_rope(
            self.attn,
            q=x_norm_gathered,
            k=x_norm_gathered,
            v=x_norm_gathered,
            position_ids=position_ids,
            use_cache=False,
            past_key_value_state=past_key_value_state,
            is_self=True,
        )
        print(f"[Rank {self.rank}] QKV shapes - Q: {q_full.shape}, K: {k_full.shape}, V: {v_full.shape}")

        start_idx = self.rank * strategy.block_size
        end_idx = start_idx + strategy.block_size
        def _pad_to_block(t, target_len, dim=2):
            pad_len = target_len - t.shape[dim]
            if pad_len <= 0:
                return t
            pad_shape = list(t.shape)
            pad_shape[dim] = pad_len
            pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
            return torch.cat([t, pad], dim=dim)

        q = _pad_to_block(q_full[:, :, start_idx:end_idx, :], strategy.block_size)
        k = _pad_to_block(k_full[:, :, start_idx:end_idx, :], strategy.block_size)
        v = _pad_to_block(v_full[:, :, start_idx:end_idx, :], strategy.block_size)


        print(f"[Rank {self.rank}] Sliced QKV shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")

        if self.debug_mode:
            debug_info = {
                f"q_local_r{self.rank}": q.detach().cpu(),
                f"k_local_r{self.rank}": k.detach().cpu(),
                f"v_local_r{self.rank}": v.detach().cpu(),
            }
        else:
            debug_info = None

        B, H, T_q, D = q.shape
        D_v = self.attn.emb_v_per_head
        scale = math.sqrt(D)

        max_score = torch.full((B, H, T_q, 1), -float("inf"), device=q.device, dtype=q.dtype)
        numerator = torch.zeros(B, H, T_q, D_v, device=q.device, dtype=q.dtype)
        denominator = torch.zeros(B, H, T_q, 1, device=q.device, dtype=q.dtype)

        for i in range(self.world_size):
            print(f"[Rank {self.rank}] Ring step {i}")
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / scale

            if mask is not None:
                scores += mask

            if is_causal_mask:
                causal_mask = torch.tril(torch.ones(T_q, k.shape[2], device=q.device)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask == 0, -float("inf"))

            block_max = scores.amax(dim=-1, keepdim=True)
            max_score = torch.maximum(max_score, block_max)

            stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0)
            exp_scores = torch.exp(stable_scores)
            numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v)
            denominator += exp_scores.sum(dim=-1, keepdim=True)

            if i < self.world_size - 1:
                k, _ = self._ring_shift_tensor(k)
                v, _ = self._ring_shift_tensor(v)

        # Log intermediate values if debugging
        if self.debug_mode and debug_info is not None:
            debug_info[f"max_score_r{self.rank}"] = max_score.clone().detach().cpu()
            debug_info[f"numerator_r{self.rank}"] = numerator.clone().detach().cpu()
            debug_info[f"denominator_r{self.rank}"] = denominator.clone().detach().cpu()

        attn_out = numerator / (denominator + 1e-10)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, H * D_v)
        attn_out = self.attn.dense(attn_out)
        print(f"[Rank {self.rank}] Final attention output shape: {attn_out.shape}")

        return (attn_out, None, debug_info) if self.debug_mode else (attn_out, None)

    def _ring_shift_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1 + self.world_size) % self.world_size
        try:
            print(f"[Rank {self.rank}] Shifting tensor: send → {send_rank}, recv ← {recv_rank}, shape: {tensor.shape}")
            tensor_recv = torch.empty_like(tensor, memory_format=torch.contiguous_format)
            send_req = dist.isend(tensor.contiguous(), dst=send_rank)
            dist.recv(tensor_recv, src=recv_rank)
            send_req.wait()
            print(f"[Rank {self.rank}] Received tensor from {recv_rank}")
            return tensor_recv, recv_rank
        except Exception as e:
            print(f"[Rank {self.rank}] Error in ring shift: {e}")
            raise
