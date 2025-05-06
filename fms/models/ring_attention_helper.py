from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.distributed as dist
import math

def _pad_to_block(t, target_len, dim=2):
    """Pads a tensor `t` to `target_len` along dimension `dim` with zeros."""
    pad_len = target_len - t.shape[dim]
    if pad_len <= 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)

class RingAttentionHelper:
    def __init__(self, attn_module, strategy, llama_block,use_cache=False,
             debug_mode: bool = False, minimal_debug_prints: bool = False,
             ff=None, ff_norm=None):  # <-- Add these
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.attn = attn_module
        self.strategy = strategy # Assuming strategy contains block_size
        self.use_cache = use_cache
        self.debug_mode = debug_mode
        self.llama_block = llama_block
        self.minimal_debug_prints = minimal_debug_prints
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.head_dim = attn_module.emb_kq_per_head
        self.scale = math.sqrt(self.head_dim)
        # # Ensure block_size is set
        # if not hasattr(self.strategy, 'block_size'):
        #      print("Warning: strategy object does not have 'block_size'. Using a default of 128.")
        #      self.strategy.block_size = 512

    def forward(self, x_norm, strategy, mask=None, position_ids=None, past_key_value_state=None,
            is_causal_mask=False, rank=0, minimal_debug_prints: bool = False, valid_len=0, 
            residual=None):
        """Main forward pass, delegates to forward_full after initial setup without global gather."""
        
        start_idx_global = self.rank * self.strategy.block_size
        B, T = x_norm.shape[:2]

        if position_ids is None:
            # Initialize to -1 by default
            position_ids = torch.full((B, T), fill_value=-1, dtype=torch.long, device=x_norm.device)
            
            if valid_len > 0:
                valid_pos = torch.arange(start_idx_global, start_idx_global + valid_len, device=x_norm.device)
                position_ids[:, :valid_len] = valid_pos.unsqueeze(0)  # Broadcast to all batches

        # Suggestion 2: Clip position_ids to valid_len BEFORE RoPE
        # Also, trim x_norm and residual to valid_len before passing to QKV computation
        position_ids_for_rope = position_ids[:, :valid_len]
        x_norm_for_rope = x_norm[:, :valid_len, :]
        residual_for_rope = residual[:, :valid_len, :] if residual is not None else None

        # Suggestion 1: Log the actual position_ids being used for RoPE
        if self.debug_mode:
            # Ensure rank is available for printing, self.rank should be set in __init__
            print(f"[Rank {self.rank}] position_ids used for RoPE: {position_ids_for_rope[0, :].tolist() if position_ids_for_rope.numel() > 0 else '[]'}")

        # Compute local QKV aligned to global rotary positions
        q_local, k_local, v_local = self.llama_block.compute_local_qkv_and_rope(
            self.attn,
            q=x_norm_for_rope, k=x_norm_for_rope, v=x_norm_for_rope, # Use trimmed inputs
            position_ids=position_ids_for_rope, # Use trimmed position_ids
            use_cache=False,
            past_key_value_state=past_key_value_state,
            is_self=True
        )

        # QKV are now naturally of valid_len due to trimmed inputs to compute_local_qkv_and_rope
        # x_norm_local and residual_local for forward_full should also be the trimmed versions
        x_norm_local = x_norm_for_rope
        residual_local = residual_for_rope

        # Forward full with locally computed Q/K/V
        result = self.forward_full(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            mask_global=mask,
            x_block=residual_local, # Pass trimmed residual
            x_norm_block=x_norm_local, # Pass trimmed norm
            valid_len=valid_len,
            q_start_global=start_idx_global
        )

        if self.debug_mode:
            x, debug_info = result
            return x, None, debug_info
        else:
            return result, None, None


    def forward_full(self, q_local: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor,
                     mask_global: Optional[torch.Tensor], # x_global: torch.Tensor, x_norm_global: torch.Tensor,
                     valid_len: int, x_block, x_norm_block, q_start_global) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict],]:
        """
        Performs the full ring attention forward pass using a two-pass approach.
        Uses torch.distributed for communication.
        """
        debug_info = {} if self.debug_mode else None

        B, H, T_q_local, D_head = q_local.shape
        D_v = self.attn.emb_v_per_head
        T_q_local = valid_len # Use the provided valid_len as the local sequence length
        T_block = self.strategy.block_size # Padded block size per rank

        # Determine the start and end indices for the local block on this rank
        start_idx_global = self.rank * self.strategy.block_size
        # Use T_q_local (valid_len) for slicing Q, K, V from the global QKV tensors
        end_idx_qkv = start_idx_global + T_q_local
        # Use T_block for slicing x_global and x_norm_global to match engine's block slicing
        # end_idx_block = start_idx_global + T_block # Not needed if we receive trimmed inputs

        if self.debug_mode and debug_info is not None:
            debug_info.update({
                f"q_local_r{self.rank}": q_local.detach().cpu(), # Shape: [B, H, valid_len, D]
                f"k_local_r{self.rank}": k_local.detach().cpu(), # Shape: [B, Hkv, valid_len, D]
                # Log the trimmed x_block (residual) and x_norm_block received
                f"v_local_r{self.rank}": v_local.detach().cpu(), # Shape: [B, Hkv, valid_len, Dv]
                f"x_norm_r{self.rank}": x_norm_block.detach().cpu(), # Shape: [B, valid_len, D_emb]
                f"x_block_r{self.rank}": x_block.detach().cpu(),
            })

        # --- Pass 1: Compute Max Scores ---
        max_score = self._compute_max_score_pass(
            q_local=q_local,
            k_local=k_local,
            mask_global=mask_global,
            q_start_global=start_idx_global,
            valid_len_local=T_q_local,
            debug_info=debug_info
        )

        if self.debug_mode and debug_info is not None:
             debug_info[f"max_score_r{self.rank}"] = max_score.clone().detach().cpu()


        # --- Pass 2: Compute Numerator and Denominator ---
        numerator, denominator = self._compute_sums_pass(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            mask_global=mask_global,
            q_start_global=start_idx_global,
            valid_len_local=T_q_local,
            max_score=max_score,
            debug_info=debug_info
        )

        if self.debug_mode and debug_info is not None:
            debug_info.update({
                f"numerator_r{self.rank}": numerator.clone().detach().cpu(),
                f"denominator_r{self.rank}": denominator.clone().detach().cpu()
            })

        # --- Final Attention Output Calculation ---
        attn_out_h = numerator / (denominator + 1e-10)
        attn_out = attn_out_h.to(q_local.dtype)
        B, H, T_q_local, D_v = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q_local, H * D_v)

        # Apply dense layer and dropout
        attn_out = self.attn.dense(attn_out)
        # if hasattr(self.llama_block, 'dropout') and self.llama_block.config.p_dropout != 0:
        #     attn_out = self.llama_block.dropout(attn_out)
        
        # Suggestion 1: Trim attn_out to valid_len (T_q_local) before residual connection
        attn_out = attn_out[:, :T_q_local, :]

        if self.debug_mode and debug_info is not None:
            # Suggestion 4: Log trimmed tensor
            debug_info[f"attn_out_raw_r{self.rank}"] = attn_out.clone().detach().cpu() 

        # Add residual connection for attention output (using trimmed tensors)
        residual_1 = x_block + attn_out
        if self.debug_mode and debug_info is not None:
            # Suggestion 4: Log trimmed tensor (already trimmed as inputs were)
            debug_info[f"attn_out_residual_r{self.rank}"] = residual_1.clone().detach().cpu() 

        # --- Feedforward Network ---
        # Suggestion 5: Apply FF norm and FF layers to unpadded (trimmed) residual_1
        # residual_1 is already trimmed to T_q_local (valid_len)
        ff_ln_out = self.ff_norm(residual_1) 
        if self.debug_mode and debug_info is not None:
            # Log unpadded tensor
            debug_info[f"ff_ln_out_r{self.rank}"] = ff_ln_out.clone().detach().cpu()

        ff_out_raw = self.ff(ff_ln_out) # ff_ln_out is also trimmed
        if self.debug_mode and debug_info is not None:
            # Log unpadded tensor
            debug_info[f"ff_out_raw_r{self.rank}"] = ff_out_raw.clone().detach().cpu()

        # if hasattr(self.llama_block, 'dropout') and self.llama_block.config.p_dropout != 0:
        #     ff_out_raw = self.llama_block.dropout(ff_out_raw)

        # Add residual (all tensors are already trimmed to T_q_local)
        x = ff_out_raw + residual_1 

        # Suggestion 1: Ensure final output x is explicitly trimmed (already done by operating on trimmed inputs)
        # x = x[:, :T_q_local, :] # This should be redundant now but safe to keep if unsure

        # Suggestion 2: Optional clamping for FP16 stability
        if x.dtype == torch.float16:
            x = torch.clamp(x, min=-5.0, max=5.0) # Example clamp range

        # Suggestion 7: Optional normalization for final output
        # if self.debug_mode: # Or a separate flag for this specific normalization
        #     x = x / (x.norm(dim=-1, keepdim=True) + 1e-5)

        if self.debug_mode and debug_info is not None:
            # Suggestion 4: Log trimmed tensor
            debug_info[f"block_output_r{self.rank}"] = x.clone().detach().cpu() 

        return (x, debug_info) if self.debug_mode else x

    def _compute_max_score_pass(self, q_local: torch.Tensor, k_local: torch.Tensor,
                                mask_global: Optional[torch.Tensor], q_start_global: int,
                                valid_len_local: int, debug_info: Optional[Dict]) -> torch.Tensor:
        """
        First pass: Computes maximum attention scores for numerical stability.
        """
        B, H, T_q_local, D_head = q_local.shape
        device = q_local.device
        dtype = torch.float32 # Use float32 for scores

        max_score = torch.full((B, H, T_q_local, 1), -float("inf"), device=device, dtype=dtype)

        # Global indices for the local query block
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local.clone() # Start with the local k block
        current_k_len = k_local.shape[2] # Valid length of the current k block

        for i in range(self.world_size):
            # Global start index of the current k block being processed
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)


            # Get the relevant slice of the global mask
            # Note: mask slicing uses global indices
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None
            if current_k_len == 0:
                # Need to shift tensors even if we skip computation
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                continue 
            # Compute raw scores for debugging before applying masks
            if self.debug_mode and debug_info is not None:
                 raw_scores = self._compute_attention_scores(
                    q=q_local, k=current_k_block[:, :, :current_k_len, :], # Use current_k_len for slicing
                    q_indices_global=q_indices_global, k_indices_global=k_indices_global,
                    mask=None, # No mask for raw scores
                    apply_mask=False, keep_causal=False # Do not apply any mask
                )
                 debug_info[f"raw_scores_step{i}_r{self.rank}"] = raw_scores.clone().detach().cpu()


            # Compute attention scores with masks
            scores = self._compute_attention_scores(
                q=q_local, k=current_k_block[:, :, :current_k_len, :], # Use current_k_len for slicing
                q_indices_global=q_indices_global, k_indices_global=k_indices_global,
                mask=current_mask,
                apply_mask=True # Apply both causal and padding masks
            )

            if self.debug_mode and debug_info is not None:
                debug_info[f"scores_step{i}_r{self.rank}"] = scores.clone().detach().cpu()

            # Update the maximum score
            max_score = self._update_max_score(scores, max_score)

            # Ring shift k block (if not the last iteration)
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)

                if self.debug_mode and debug_info is not None:
                    debug_info[f"k_input_step{i+1}_r{self.rank}"] = current_k_block[:, :, :current_k_len, :].clone().detach().cpu()


        return max_score


    def _compute_sums_pass(self, q_local: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor,
                           mask_global: Optional[torch.Tensor], q_start_global: int,
                           valid_len_local: int, max_score: torch.Tensor, debug_info: Optional[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Second pass: Computes numerator and denominator for the softmax.
        """
        B, H, T_q_local, D_head = q_local.shape
        D_v = self.attn.emb_v_per_head
        device = q_local.device
        dtype = torch.float32 # Use float32 for numerator and denominator

        numerator = torch.zeros(B, H, T_q_local, D_v, device=device, dtype=dtype)
        denominator = torch.zeros(B, H, T_q_local, 1, device=device, dtype=dtype)

        # Global indices for the local query block
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local.clone() # Start with the local k block
        current_v_block = v_local.clone() # Start with the local v block
        current_k_len = k_local.shape[2] # Valid length of the current k block

        for i in range(self.world_size):
            # Global start index of the current k block being processed
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)

            # Skip computation if the received K/V block has zero length
            if current_k_len == 0:
                # Need to shift tensors even if we skip computation
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                    current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size)
                continue # Skip to the next iteration

            # Get the relevant slice of the global mask
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None

            # Compute attention scores with masks
            scores = self._compute_attention_scores(
                q=q_local, k=current_k_block[:, :, :current_k_len, :], # Use current_k_len for slicing
                q_indices_global=q_indices_global, k_indices_global=k_indices_global,
                mask=current_mask,
                apply_mask=True
            )

            # Update numerator and denominator
            numerator, denominator = self._update_totals(scores, current_v_block[:, :, :current_k_len, :], max_score, numerator, denominator) # Use current_k_len for slicing v

            # Ring shift k and v blocks (if not the last iteration)
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size) # Assuming v has same length as k

                if self.debug_mode and debug_info is not None:
                     debug_info[f"k_input_step{i+1}_r{self.rank}"] = current_k_block[:, :, :current_k_len, :].clone().detach().cpu()
                     debug_info[f"v_input_step{i+1}_r{self.rank}"] = current_v_block[:, :, :current_k_len, :].clone().detach().cpu()


        return numerator, denominator

    def _compute_attention_scores(self, q: torch.Tensor, k: torch.Tensor, q_indices_global: torch.Tensor,
                                 k_indices_global: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                 apply_mask: bool = True, keep_causal: bool = True) -> torch.Tensor:
        """Computes attention scores, optionally applying padding and causal masks."""
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale

        if apply_mask and mask is not None:
            scores = scores + mask.to(scores.dtype)

        if apply_mask and keep_causal:
             causal_mask = (k_indices_global[None, :] > q_indices_global[:, None]).unsqueeze(0).unsqueeze(0)
             scores = scores.masked_fill(causal_mask, -torch.inf)

        return scores

    def _update_max_score(self, scores: torch.Tensor, current_max: torch.Tensor) -> torch.Tensor:
        """Updates max score handling -inf."""
        block_max = scores.masked_fill(scores == -torch.inf, torch.finfo(scores.dtype).min).amax(dim=-1, keepdim=True)
        return torch.maximum(current_max, block_max)

    def _update_totals(self, scores: torch.Tensor, v: torch.Tensor, max_score: torch.Tensor,
                      numerator: torch.Tensor, denominator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates numerator and denominator using stable exponentiation."""
        stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0)
        exp_scores = torch.exp(stable_scores)

        exp_scores = torch.nan_to_num(exp_scores, nan=0.0, posinf=torch.finfo(exp_scores.dtype).max, neginf=0.0)

        numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v.to(numerator.dtype))

        denominator += exp_scores.sum(dim=-1, keepdim=True)

        return numerator, denominator


    def _ring_shift_tensor(self, tensor: torch.Tensor, pad_len: int) -> Tuple[torch.Tensor, int]:
        """
        Ring‑shifts `tensor` along its last dimension by one rank:
        - CPU: non‑blocking isend/irecv.
        - GPU: all_gather of lengths + all_to_all of only the neighbor’s block.
        Returns (received_tensor cropped to true length, received_length).
        """
        rank, world = self.rank, self.world_size
        send_rank = (rank + 1) % world
        recv_rank = (rank - 1 + world) % world

        # 1) pad along last dim
        valid_len = tensor.shape[-1]
        padded = _pad_to_block(tensor, pad_len, dim=-1).contiguous()
        if self.debug_mode:
            print(f"[Rank {rank}][START] tensor.shape={tuple(tensor.shape)}")
        


        if not tensor.is_cuda:
            """
            Performs a ring shift of a tensor using torch.distributed.
            Pads to pad_len, sends, receives, and returns received tensor and its valid length.
            """
            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1 + self.world_size) % self.world_size

            valid_len = tensor.shape[2]
            padded = _pad_to_block(tensor, pad_len, dim=2).contiguous()

            send_len = torch.tensor([valid_len], dtype=torch.int32, device=tensor.device)
            recv_len = torch.empty(1, dtype=torch.int32, device=tensor.device)

            if self.debug_mode:
                print(f"[Rank {rank}][CPU SEND] len={valid_len}", flush=True)
                # Print stats and values of the tensor being sent
                padded_cpu = padded.detach().cpu().float()
                print(f"[Rank {rank}][CPU SEND] data[:5]={padded_cpu.flatten()[:5].tolist()}", flush=True)
                print(f"[Rank {rank}][CPU SEND] mean={padded_cpu.mean():.4f}, std={padded_cpu.std():.4f}", flush=True)

            reqs = [
                dist.isend(send_len, dst=send_rank),
                dist.irecv(recv_len, src=recv_rank)
            ]
            for req in reqs:
                req.wait()

            tensor_recv = torch.empty_like(padded)
            reqs = [
                dist.isend(padded, dst=send_rank),
                dist.irecv(tensor_recv, src=recv_rank)
            ]
            for req in reqs:
                req.wait()

            received_len_val = recv_len.item()
            if self.debug_mode:
                print(f"[Rank {rank}][CPU RECV] len={received_len_val}", flush=True)
                # Print stats and values of the tensor received
                recv_cpu = tensor_recv.detach().cpu().float()
                print(f"[Rank {rank}][CPU RECV] data[:5]={recv_cpu.flatten()[:5].tolist()}", flush=True)
                print(f"[Rank {rank}][CPU RECV] mean={recv_cpu.mean():.4f}, std={recv_cpu.std():.4f}", flush=True)
                print(f"[Rank {rank}][CPU] tensor_recv.shape={tuple(tensor_recv.shape)}, recv_len={recv_len}")
            recv_len = recv_len.item()

        else:
            # GPU: collective ring shift mirroring CPU behavior exactly

            # 1) determine true length, pad along block axis
            valid_len = tensor.size(-2)
            padded = _pad_to_block(tensor, pad_len, dim=-2).contiguous()

            # 2) exchange lengths collectively
            len_t = torch.tensor([valid_len], dtype=torch.int32, device=tensor.device)
            len_list = [torch.empty_like(len_t) for _ in range(world)]
            dist.all_gather(len_list, len_t)
            recv_len = int(len_list[recv_rank].item())

            if self.debug_mode:
                print(f"[Rank {rank}][GPU SEND] len={valid_len}", flush=True)
                # Print stats and values of the tensor being sent
                padded_cpu = padded.detach().cpu().float()
                print(f"[Rank {rank}][GPU SEND] data[:5]={padded_cpu.flatten()[:5].tolist()}", flush=True)
                print(f"[Rank {rank}][GPU SEND] mean={padded_cpu.mean():.4f}, std={padded_cpu.std():.4f}", flush=True)

            # 3) collective send/recv buffers (only neighbor slot has data)
            send_list = [torch.empty_like(padded) for _ in range(world)]
            recv_list = [torch.empty_like(padded) for _ in range(world)]
            send_list[send_rank].copy_(padded)

            # 4) collective exchange
            dist.all_to_all(recv_list, send_list)

            # 5) extract neighbor's block and return
            tensor_recv = recv_list[recv_rank]

            if self.debug_mode:
                print(f"[Rank {rank}][GPU RECV] len={recv_len}", flush=True)
                # Print stats and values of the tensor received
                recv_cpu = tensor_recv.detach().cpu().float()
                print(f"[Rank {rank}][GPU RECV] data[:5]={recv_cpu.flatten()[:5].tolist()}", flush=True)
                print(f"[Rank {rank}][GPU RECV] mean={recv_cpu.mean():.4f}, std={recv_cpu.std():.4f}", flush=True)
                print(f"[Rank {rank}][GPU] tensor_recv.shape={tuple(tensor_recv.shape)}, recv_len={recv_len}")

        dist.barrier()
        # exit(0)
        return tensor_recv, recv_len

    # Suggestion 3: Add Relative Tolerance-Based Comparison helper
    def _log_diff_stats(self, tag: str, a: torch.Tensor, b: torch.Tensor, tol=1e-4):
        if not self.debug_mode: # Check if debug_mode is enabled for this helper instance
            return
        # Ensure tensors are on CPU and float for comparison, and match shapes if necessary
        a_comp = a.detach().cpu().float()
        b_comp = b.detach().cpu().float()
        # Add more sophisticated shape handling if needed for comparison (e.g., trim to min length)
        diff = (a_comp - b_comp).abs()
        max_diff = diff.max().item()
        offending = (diff > tol).sum().item()
        total = diff.numel()
        print(f"[{tag} Rank {self.rank}] MaxDiff={max_diff:.6f}, Offending(>{tol:.1e})={offending}/{total} ({100*offending/total:.2f}%)")
