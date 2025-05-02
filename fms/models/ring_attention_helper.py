from typing import Tuple, Dict, Optional, Union
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

        # Store rank/world_size for convenience
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, x_norm, strategy, mask=None, position_ids=None, past_key_value_state=None, is_causal_mask=False, rank = 0 ):

        if strategy.world_size > 1:
            # Debug: Print local shape before gather
            print(f"[Rank {strategy.rank}] x_norm shape before gather: {x_norm.shape}")
            dist.barrier()  # Sync all ranks

            # Now gather
            x_norm_gathered = strategy.gather_tensor(x_norm, dim=1)
            
            # Debug: Print gathered shape
            print(f"[Rank {strategy.rank}] x_norm_gathered shape: {x_norm_gathered.shape}")
        else:
            x_norm_gathered = x_norm

                # if strategy.world_size > 1:
        # #     x_norm_gathered = strategy.gather_tensor(x_norm, dim=1)  # [B, T_total, D]
        # # else:
        # x_norm_gathered = x_norm

        # Compute QKV from full x_norm
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

        # # Let strategy decide which block this rank uses
        start_idx = self.rank * strategy.block_size
        end_idx = start_idx + strategy.block_size

        # # If total tokens < end_idx, slicing will be safe (padded if needed)
        q = q_full[:, :, start_idx:end_idx, :]
        k = k_full[:, :, start_idx:end_idx, :]
        v = v_full[:, :, start_idx:end_idx, :]

        # q, k, v = self.llama_block.compute_local_qkv_and_rope(
        #     self.attn,
        #     q=x_norm,
        #     k=x_norm,
        #     v=x_norm,
        #     position_ids=position_ids,
        #     use_cache=self.use_cache,
        #     past_key_value_state=past_key_value_state,
        #     is_self=True
        # )


        debug_info: Optional[Dict[str, torch.Tensor]] = {} if self.debug_mode else None
        if self.debug_mode:
            debug_info[f"q_local_r{self.rank}"] = q.clone().detach().cpu()
            debug_info[f"k_local_r{self.rank}"] = k.clone().detach().cpu()
            debug_info[f"v_local_r{self.rank}"] = v.clone().detach().cpu()

        q_local, k_local, v_local = q.clone(), k.clone(), v.clone()
        B, H, T_q, D = q.shape
        D_v = self.attn.emb_v_per_head
        scale = math.sqrt(D)

        max_score = torch.full((B, H, T_q, 1), -float("inf"), device=q.device, dtype=q.dtype)
        numerator = torch.zeros(B, H, T_q, D_v, device=q.device, dtype=q.dtype)
        denominator = torch.zeros(B, H, T_q, 1, device=q.device, dtype=q.dtype)
        for i in range(self.world_size):
            scores = torch.einsum("bhqd,bhkd->bhqk", q_local, k) / scale

            if mask is not None:
                scores += mask

            if is_causal_mask:
                causal_mask = torch.tril(torch.ones(T_q, k.shape[2], device=q.device)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask == 0, -float("inf"))

            block_max = scores.amax(dim=-1, keepdim=True)
            max_score = torch.maximum(max_score, block_max)

            stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0)
            exp_scores = torch.exp(stable_scores)
            num_update = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v)
            den_update = exp_scores.sum(dim=-1, keepdim=True)

            numerator += num_update
            denominator += den_update
            if i < self.world_size - 1:
                k, _ = self._ring_shift_tensor(k)
                v, _ = self._ring_shift_tensor(v)

        attn_out = numerator / (denominator + 1e-10)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, H * D_v)
        attn_out = self.attn.dense(attn_out)

        return (attn_out, None, debug_info) if self.debug_mode else (attn_out, None)

    def _ring_shift_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Ring-shift a tensor across ranks (send left, recv right) """
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1 + self.world_size) % self.world_size

        # Ensure the receive buffer is contiguous
        tensor_recv = torch.empty_like(tensor, memory_format=torch.contiguous_format)
        send_req = dist.isend(tensor.contiguous(), dst=send_rank) # Ensure tensor is contiguous
        dist.recv(tensor_recv, src=recv_rank)
        send_req.wait()
        return tensor_recv, recv_rank
