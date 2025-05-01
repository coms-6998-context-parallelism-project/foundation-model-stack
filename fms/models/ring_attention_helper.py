from typing import Tuple
import torch
import torch.distributed as dist
import math

class RingAttentionHelper:
    def __init__(self, attn_module, layer_idx, strategy, use_cache=False):
        self.attn = attn_module
        self.layer_idx = layer_idx
        self.strategy = strategy
        self.use_cache = use_cache

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, x_norm, mask=None, position_ids=None, past_key_value_state=None, is_causal_mask=False):
        # Step 1: Local QKV computation with RoPE
        # Use the unified method from the attention module
        q, k, v = self.attn.compute_qkv_and_rope(
            x_norm, x_norm, x_norm, # Pass x_norm as q, k, v inputs
            position_ids=position_ids,
            use_cache=self.use_cache,
            past_key_value_state=past_key_value_state,
        )  # Shapes: [B, H, T, D] each

        # Save original Q for local indexing
        q_local = q.clone()
        k_local = k.clone()
        v_local = v.clone()

        # Step 2: Init attention accumulator
        B, H, T_q, D = q.shape
        D_v = self.attn.emb_v_per_head
        scale = math.sqrt(D)

        max_score = torch.full((B, H, T_q, 1), -float("inf"), device=q.device, dtype=q.dtype)
        numerator = torch.zeros(B, H, T_q, D_v, device=q.device, dtype=q.dtype)
        denominator = torch.zeros(B, H, T_q, 1, device=q.device, dtype=q.dtype)

        for i in range(self.world_size):
            # Step 3: Compute attention
            scores = torch.einsum("bhqd,bhkd->bhqk", q_local, k) / scale

            if mask is not None:
                scores = scores + mask

            if is_causal_mask:
                causal_mask = torch.tril(torch.ones(T_q, k.shape[2], device=q.device)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask == 0, -float("inf"))

            block_max = scores.amax(dim=-1, keepdim=True)
            max_score = torch.maximum(max_score, block_max)

            # Softmax stabilization
            stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0)
            exp_scores = torch.exp(stable_scores)
            num_update = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v)
            den_update = exp_scores.sum(dim=-1, keepdim=True)

            numerator += num_update
            denominator += den_update

            # Step 4: Ring shift for k/v
            if i < self.world_size - 1:
                k, _ = self._ring_shift_tensor(k)
                v, _ = self._ring_shift_tensor(v)

        attn_out = numerator / (denominator + 1e-10)  # [B, H, T, Dv]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, H * D_v)
        attn_out = self.attn.dense(attn_out)

        return attn_out, None  # no caching for now

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
