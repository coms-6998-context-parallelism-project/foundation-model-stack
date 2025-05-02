from typing import Tuple, Dict, Optional
import torch
import torch.distributed as dist
import math

class RingAttentionHelper:
    def __init__(self, attn_module, layer_idx, strategy, llama_block, use_cache=False, debug_mode: bool = False, minimal_debug_prints: bool = False): # Add flag to init
        self.attn = attn_module
        self.layer_idx = layer_idx
        self.strategy = strategy
        self.use_cache = use_cache
        self.debug_mode = debug_mode
        self.llama_block = llama_block
        self.minimal_debug_prints = minimal_debug_prints # Store flag
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, x_norm, strategy, mask=None, position_ids=None, past_key_value_state=None, is_causal_mask=False, rank=0, minimal_debug_prints: bool = False):
        # Guard this block as well
        # if self.debug_mode and not self.minimal_debug_prints:
        #     if(self.layer_idx % 5 ==0):
        #         if(self.rank ==0):
        #             print(self.layer_idx, self.rank, x_norm.shape, end = "; ")
        #             dist.barrier()
        #         else:
        #             dist.barrier()
        #             print(self.layer_idx, self.rank, x_norm.shape)
        # if self.debug_mode:
        #     if not self.minimal_debug_prints: print(f"[Rank {self.rank}] x_norm shape before gather: {x_norm.shape}") # Guard this print
        # Guard barrier with debug_mode check
        if self.debug_mode: dist.barrier()

        if strategy.world_size > 1:
            x_norm_gathered = strategy.gather_tensor(x_norm, dim=1)
            # if self.debug_mode and not self.minimal_debug_prints: # Guard this print
            #     print(f"[Rank {self.rank}] x_norm_gathered shape: {x_norm_gathered.shape}")
        else:
            x_norm_gathered = x_norm

        # Consider removing barriers if not strictly needed for timing/debugging
        # Guard barrier with debug_mode check
        if self.debug_mode: dist.barrier()
        # if self.debug_mode and not self.minimal_debug_prints: # Guard this print
        #     print(f"[Rank {self.rank}] Computing QKV...")
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
        # if self.debug_mode and not self.minimal_debug_prints: # Guard this print
        #     print(f"[Rank {self.rank}] QKV shapes - Q: {q_full.shape}, K: {k_full.shape}, V: {v_full.shape}")

        # True token count for this rank
        real_T = x_norm.shape[1]

        start_idx = self.rank * strategy.block_size

        # Correct slicing of unpadded portion before padding
        q_local = q_full[:, :, start_idx:start_idx + real_T, :]
        k_local = k_full[:, :, start_idx:start_idx + real_T, :]
        v_local = v_full[:, :, start_idx:start_idx + real_T, :]

        def _pad_to_block(t, target_len, dim=2):
            pad_len = target_len - t.shape[dim]
            if pad_len <= 0:
                return t
            pad_shape = list(t.shape)
            pad_shape[dim] = pad_len
            pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
            return torch.cat([t, pad], dim=dim)

        q = _pad_to_block(q_local, strategy.block_size)
        k = _pad_to_block(k_local, strategy.block_size)
        v = _pad_to_block(v_local, strategy.block_size)

        # if self.debug_mode:
        #     if not self.minimal_debug_prints: print(f"[Rank {self.rank}] Sliced QKV shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")

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

        max_score = torch.full((B, H, T_q, 1), -float("inf"), device=q.device, dtype=torch.float32)
        numerator = torch.zeros(B, H, T_q, D_v, device=q.device, dtype=torch.float32)
        denominator = torch.zeros(B, H, T_q, 1, device=q.device, dtype=torch.float32)


        for i in range(self.world_size):
            # if self.debug_mode and not self.minimal_debug_prints:
            #     print(f"[Rank {self.rank}] Ring step {i}")
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / scale
            if self.debug_mode and debug_info is not None:
                debug_info[f"raw_scores_step{i}_r{self.rank}"] = scores.clone().detach().cpu()

            if mask is not None:
                scores += mask

            if is_causal_mask:
                q_pos_start = self.rank * strategy.block_size
                k_pos_start = ((self.rank - i) % self.world_size) * strategy.block_size
                q_pos = torch.arange(q_pos_start, q_pos_start + T_q, device=q.device).unsqueeze(-1)
                k_pos = torch.arange(k_pos_start, k_pos_start + k.shape[2], device=q.device).unsqueeze(0)
                causal_mask = (q_pos >= k_pos).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~causal_mask, -float("inf"))

            if self.debug_mode and debug_info is not None:
                debug_info[f"scores_step{i}_r{self.rank}"] = scores.clone().detach().cpu()

            block_max = scores.amax(dim=-1, keepdim=True)
            max_score = torch.maximum(max_score, block_max)

            stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0).to(torch.float32)
            exp_scores = torch.exp(stable_scores)  # Now in float32

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
        attn_out = (numerator / (denominator + 1e-10)).to(q.dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, H * D_v)
        attn_out = self.attn.dense(attn_out)
        attn_out = attn_out[:, :real_T, :]  # finally trim back to real token count

        # if self.debug_mode:
        #     if not self.minimal_debug_prints: print(f"[Rank {self.rank}] Final attention output shape: {attn_out.shape}")

        return (attn_out, None, debug_info) if self.debug_mode else (attn_out, None)


    def _ring_shift_tensor(self, tensor: torch.Tensor, pad_len: int) -> Tuple[torch.Tensor, int]:
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1 + self.world_size) % self.world_size

        valid_len = tensor.shape[2]
        pad_size = pad_len - valid_len

        if pad_size > 0:
            pad_tensor = torch.zeros(
                *tensor.shape[:2], pad_size, tensor.shape[-1],
                dtype=tensor.dtype, device=tensor.device
            )
            tensor_to_send = torch.cat([tensor, pad_tensor], dim=2)
        else:
            tensor_to_send = tensor

        tensor_to_send = tensor_to_send.contiguous()

        send_len = torch.tensor([valid_len], dtype=torch.int32, device=tensor.device)
        recv_len = torch.empty(1, dtype=torch.int32, device=tensor.device)

        # 🔄 Length exchange using async ops
        len_send_req = dist.isend(send_len, dst=send_rank)
        len_recv_req = dist.irecv(recv_len, src=recv_rank)
        len_send_req.wait()
        len_recv_req.wait()

        # 🔄 Tensor exchange using async ops
        tensor_recv = torch.empty_like(tensor_to_send)
        tensor_send_req = dist.isend(tensor_to_send, dst=send_rank)
        tensor_recv_req = dist.irecv(tensor_recv, src=recv_rank)
        tensor_send_req.wait()
        tensor_recv_req.wait()

        return tensor_recv, recv_len.item()
