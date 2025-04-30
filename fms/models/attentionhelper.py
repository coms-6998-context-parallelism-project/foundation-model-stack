import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch.distributed as dist
import torch
import torch.nn as nn
from torch import Tensor, nn

from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit

logger = logging.getLogger(__name__)

# reread this file
# Note: This dataclass might become less necessary as we move away from threads,
# but we can keep it for now to structure the arguments passed within the rank.
@dataclass
class BlockData:
    engine_instance: 'RingAttentionEngine' 
    rank: int
    world_size: int # World size of the ring group
    q_local_start: int # Start index within the *local* shard for Q (always 0)
    q_local_end: int   # End index within the *local* shard for Q (local_seq_len)
    q_global_offset: int # Global start index of this rank's shard
    mask_global: Optional[Tensor]
    q_shard: Tensor # The local Q shard for this rank
    k_shard: Tensor # The local K shard for this rank (initial)
    v_shard: Tensor # The local V shard for this rank (initial)
    x_block: Tensor

# Ring Attention computation logic
class RingAttentionEngine:
    def __init__(self, strategy_block_size: int, attn: MultiHeadAttention, ff: GatedLinearUnit, ff_norm: nn.Module, is_causal: bool, group: dist.ProcessGroup):
        self.attn = attn
        self.ff = ff
        self.ff_norm = ff_norm
        self.is_causal = is_causal
        self.head_dim = attn.emb_kq_per_head
        self.scale = math.sqrt(self.head_dim)
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        self.next_rank = (self.rank + 1) % self.world_size
        self.strategy_block_size = strategy_block_size # Get the strategy's block size directly

    # main method
    def forward_full(self, q_shard: Tensor, k_shard: Tensor, v_shard: Tensor, mask_global: Optional[Tensor], x_shard: Tensor, q_global_offset: int, global_seq_len: int) -> Tensor:
        """
        Performs Ring Attention computation across the distributed group.
        Assumes q_shard, k_shard, v_shard, x_shard are the *local shards* for the current rank.
        q_global_offset indicates the starting position of this shard in the global sequence.
        global_seq_len is the total sequence length across all ranks.
        Returns the computed output shard for the current rank (size strategy_block_size).
        """
        # print(f"[RING DEBUG][rank{self.rank}] RingAttentionEngine.forward_full: ENTERED.", flush=True)
        # print(f"[rank{self.rank}] RingAttentionEngine.forward_full: START. q_shard shape: {q_shard.shape}, k_shard shape: {k_shard.shape}, v_shard shape: {v_shard.shape}, x_shard shape: {x_shard.shape}, q_global_offset: {q_global_offset}, global_seq_len: {global_seq_len}")
        T_q_local = q_shard.shape[2]
        if T_q_local == 0: return torch.empty_like(x_shard)

        # Determine the query block this rank is responsible for
        # The entire local shard is the query block for this rank
        q_local_start = 0 # Start index within the local shard (always 0)
        q_local_end = T_q_local

        # k_local and v_local are just the input k_shard and v_shard initially
        k_local = k_shard
        v_local = v_shard
        x_block = x_shard # Input X corresponding to the local Q shard

        # Package data for internal methods (optional, could pass args directly)
        block_data = BlockData(
            engine_instance=self, rank=self.rank, world_size=self.world_size,
            q_local_start=q_local_start, q_local_end=q_local_end,
            q_global_offset=q_global_offset,
            mask_global=mask_global,
            q_shard=q_shard, k_shard=k_local, v_shard=v_local, x_block=x_block
        )

        # --- Ring Computation ---
        # 1. Initialize max score, numerator, denominator
        initial_max_score, initial_num, initial_den = self.init_values(q_shard)

        # 2. First pass: Compute max score across all K blocks (shards)
        # print(f"[rank{self.rank}] Before compute_max_score", flush=True)
        final_max_score = self.compute_max_score(block_data, initial_max_score, global_seq_len)
        # print(f"[rank{self.rank}] After compute_max_score. Entering barrier...", flush=True)
        dist.barrier(self.group) # Sync after max score calculation
        # print(f"[rank{self.rank}] Exited barrier after compute_max_score.", flush=True)

        # 3. Second pass: Compute numerator and denominator sums
        final_num, final_den = self.compute_sums(block_data, final_max_score, initial_num, initial_den, global_seq_len)
        # print(f"[rank{self.rank}] After compute_sums. Entering barrier...", flush=True)
        dist.barrier(self.group) # Ensure all ranks have computed their sums
        # print(f"[RING DEBUG][rank{self.rank}] Exited barrier after compute_sums.", flush=True) # Removed for cleanup

        # 4. Compute local output block
        output_block = self.compute_block_output(x_block, final_num, final_den)

        # Return the locally computed block. Gathering happens outside.
        # print(f"[rank{self.rank}] RingAttentionEngine.forward_full: END. Output shape: {output_block.shape}")
        return output_block

    """ compute max scores for stability (first flash pass) """
    def compute_max_score(self, args: BlockData, initial_max_score: Tensor, global_seq_len: int) -> Tensor:
        engine = args.engine_instance

        device = args.q_shard.device
        # Global indices for the local Q shard
        q_global_indices = torch.arange(args.q_global_offset, args.q_global_offset + args.q_shard.shape[2], device=device)

        max_score = initial_max_score
        current_k_shard = args.k_shard
        current_k_rank = args.rank # Rank where the current K shard originated
        
        # Temporary buffer for receiving K, size is fixed to strategy_block_size
        recv_shape = list(args.k_shard.shape)
        recv_shape[2] = self.strategy_block_size
        recv_k = torch.empty(recv_shape, dtype=args.k_shard.dtype, device=device)

        # All shards have the same length now
        expected_recv_len = self.strategy_block_size

        for i in range(args.world_size):
            # print(f"[RING DEBUG][rank{args.rank}] compute_max_score: Iter {i}/{args.world_size}, processing K from rank {current_k_rank}", flush=True)
            # Calculate global indices and length for the *current* K shard
            # Length is always strategy_block_size, offset is based on rank * strategy_block_size
            current_k_len = self.strategy_block_size
            current_k_global_offset = current_k_rank * self.strategy_block_size
            # The global_seq_len passed in should be the padded length (world_size * block_size)
            # No need to adjust length for last rank due to global padding.
            current_k_global_indices = torch.arange(current_k_global_offset, current_k_global_offset + current_k_len, device=device)
            
            # Determine the length of the tensor we expect to receive in this iteration
            # The tensor we receive originated from the rank `(current_k_rank - 1 + world_size) % world_size`
            # However, it's simpler to think about which rank *sent* it to us: self.prev_rank
            # The length depends on which rank's data is currently at self.prev_rank
            incoming_rank_origin = (self.rank - 1 - i + args.world_size) % args.world_size
            # expected_recv_len is always strategy_block_size now

            # Slice the received buffer if necessary (only needed if shapes differ)
            # Use expected_recv_len (strategy_block_size) for slicing the *received* data *after* communication
            effective_recv_k = recv_k[:, :, :expected_recv_len, :] if i > 0 else current_k_shard

            # Compute attention with current K block
            # Mask needs to be sliced using global indices
            mask = args.mask_global[:, :, args.q_global_offset:args.q_global_offset+args.q_shard.shape[2], current_k_global_offset:current_k_global_offset+current_k_len] if args.mask_global is not None else None
            # print(f"[RING DEBUG][rank{args.rank}] compute_max_score: Before update_max_attn (Iter {i})", flush=True) # Add this
            max_score = engine.update_max_attn(args.q_shard, effective_recv_k, mask, q_global_indices, current_k_global_indices, max_score)
            # print(f"[RING DEBUG][rank{args.rank}] compute_max_score: After update_max_attn (Iter {i})", flush=True) # Add this

            # Send current_k_shard, receive into the full buffer
            # print(f"[RING DEBUG][rank{args.rank}] compute_max_score: Before send_recv_tensor (Iter {i})", flush=True)
            current_k_shard = self.send_recv_tensor(current_k_shard, recv_k, expected_recv_len)
            # print(f"[RING DEBUG][rank{args.rank}] compute_max_score: After send_recv_tensor (Iter {i})", flush=True)
            # Update the rank origin of the K shard we now hold
            current_k_rank = (current_k_rank - 1 + args.world_size) % args.world_size

        return max_score

    """ sum loop """
    def compute_sums(self, args: BlockData, final_max_score: Tensor, initial_num: Tensor, initial_den: Tensor, global_seq_len: int) -> Tuple[Tensor, Tensor]:

        engine = args.engine_instance
        device = args.q_shard.device
        # Global indices for the local Q shard
        q_global_indices = torch.arange(args.q_global_offset, args.q_global_offset + args.q_shard.shape[2], device=device)

        num, den = initial_num, initial_den
        current_k_shard = args.k_shard
        current_v_shard = args.v_shard
        current_kv_rank = args.rank # Rank where the current K/V shards originated

        # Temporary buffers for receiving K and V
        recv_k_shape = list(args.k_shard.shape); recv_k_shape[2] = self.strategy_block_size
        recv_v_shape = list(args.v_shard.shape); recv_v_shape[2] = self.strategy_block_size
        recv_k = torch.empty(recv_k_shape, dtype=args.k_shard.dtype, device=device)
        recv_v = torch.empty(recv_v_shape, dtype=args.v_shard.dtype, device=device)

        # All shards have the same length now
        expected_recv_len = self.strategy_block_size

        for i in range(args.world_size):
            # print(f"[RING DEBUG][rank{args.rank}] compute_sums: Iter {i}/{args.world_size}, processing K/V from rank {current_kv_rank}", flush=True)
            # Calculate global indices and length for the *current* K/V shards
            current_kv_len = (global_seq_len + args.world_size - 1) // args.world_size
            current_kv_global_offset = current_kv_rank * current_kv_len
            # Adjust length for the last rank
            # Length is always strategy_block_size, offset is based on rank * strategy_block_size
            current_kv_len = self.strategy_block_size
            current_kv_global_offset = current_kv_rank * self.strategy_block_size
            current_k_global_indices = torch.arange(current_kv_global_offset, current_kv_global_offset + current_kv_len, device=device)
            
            # Determine the length of the tensor we expect to receive in this iteration
            incoming_rank_origin = (self.rank - 1 - i + args.world_size) % args.world_size

            # Slice the received buffers if necessary
            # Use expected_recv_len for slicing the *received* data *after* communication
            effective_recv_k = recv_k[:, :, :expected_recv_len, :] if i > 0 else current_k_shard
            effective_recv_v = recv_v[:, :, :expected_recv_len, :] if i > 0 else current_v_shard

            # Compute attention with current K, V blocks
            # Mask needs to be sliced using global indices
            mask = args.mask_global[:, :, args.q_global_offset:args.q_global_offset+args.q_shard.shape[2], current_kv_global_offset:current_kv_global_offset+current_kv_len] if args.mask_global is not None else None

            # --- DETAILED DEBUGGING FOR RANK 0 HANG ---
            # if args.rank == 0 and i == 0:
            #     print(f"[rank0] compute_sums (Iter 0): BEFORE update_totals call.", flush=True)
            #     print(f"[rank0]   q_shard: {args.q_shard.shape}, {args.q_shard.device}, {args.q_shard.dtype}", flush=True)
            #     print(f"[rank0]   eff_k: {effective_recv_k.shape}, {effective_recv_k.device}, {effective_recv_k.dtype}", flush=True)
            #     print(f"[rank0]   eff_v: {effective_recv_v.shape}, {effective_recv_v.device}, {effective_recv_v.dtype}", flush=True)
            #     print(f"[rank0]   mask: {mask.shape if mask is not None else 'None'}", flush=True)
            #     print(f"[rank0]   q_indices: {q_global_indices.shape}", flush=True)
            #     print(f"[rank0]   k_indices: {current_k_global_indices.shape}", flush=True)
            #     print(f"[rank0]   final_max_score: {final_max_score.shape}, {final_max_score.device}, {final_max_score.dtype}, isnan={torch.isnan(final_max_score).any()}, isinf={torch.isinf(final_max_score).any()}", flush=True)
            #     print(f"[rank0]   num: {num.shape}, {num.device}, {num.dtype}, isnan={torch.isnan(num).any()}, isinf={torch.isinf(num).any()}", flush=True)
            #     print(f"[rank0]   den: {den.shape}, {den.device}, {den.dtype}, isnan={torch.isnan(den).any()}, isinf={torch.isinf(den).any()}", flush=True)
            # --- END DETAILED DEBUGGING ---
            # print(f"[RING DEBUG][rank{args.rank}] compute_sums: Before update_totals (Iter {i})", flush=True)
            num, den = engine.update_totals(args.q_shard, effective_recv_k, effective_recv_v, mask, q_global_indices, current_k_global_indices, final_max_score, num, den)
            # print(f"[RING DEBUG][rank{args.rank}] compute_sums: After update_totals (Iter {i})", flush=True)

            # Send current K, V shards, receive into buffers sliced for the *expected incoming length*
            recv_k_slice_for_irecv = recv_k[:, :, :expected_recv_len, :]
            recv_v_slice_for_irecv = recv_v[:, :, :expected_recv_len, :]
            # print(f"[RING DEBUG][rank{args.rank}] compute_sums: Before send_recv_kv (Iter {i})", flush=True)
            current_k_shard, current_v_shard = self.send_recv_kv(current_k_shard, current_v_shard, recv_k_slice_for_irecv, recv_v_slice_for_irecv)
            # print(f"[RING DEBUG][rank{args.rank}] compute_sums: After send_recv_kv (Iter {i})", flush=True)

            # Update the rank origin of the K/V shards we now hold
            current_kv_rank = (current_kv_rank - 1 + args.world_size) % args.world_size

        return num, den
    
    """ final output """
    def compute_block_output(self, x: Tensor, num: Tensor, den: Tensor) -> Tensor:

        B, q_len, E = x.shape; H, D_v = num.shape[1], num.shape[3]
        attn_out_h = num / (den + 10-10)
        attn_out = attn_out_h.transpose(1, 2).contiguous().view(B, q_len, H * D_v)
        attn_out = self.attn.dense(attn_out)
        residual_1 = x + attn_out
        ff_out = self.ff(self.ff_norm(residual_1))
        return residual_1 + ff_out
    
    """ Helper Functions"""

    def send_recv_tensor(self, send_tensor: Tensor, full_recv_buffer: Tensor, expected_recv_len: int) -> Tensor: # Definition expects 4 args (self + 3)
        """ Sends a tensor to the next rank and receives one from the previous rank. """
        # Use explicit blocking send/recv with different orders for even/odd ranks
        recv_buffer_slice = full_recv_buffer[:, :, :expected_recv_len, :].contiguous() # Ensure contiguous for recv
        # print(f"[RING COMM DEBUG][rank{self.rank}] Sending shape {send_tensor.shape} to rank {self.next_rank}, Receiving shape {recv_buffer_slice.shape} from rank {self.prev_rank}", flush=True)
        # print(f"[rank{self.rank}] RingAttentionEngine.send_recv_tensor: Sending shape {send_tensor.shape} to rank {self.next_rank}, receiving into buffer slice shape {recv_buffer_slice.shape} (expecting len {expected_recv_len}) from rank {self.prev_rank}", flush=True)

        send_tensor_c = send_tensor.contiguous() # Ensure contiguous before sending

        if self.rank % 2 == 0:
            # print(f"[rank{self.rank}] Sending to {self.next_rank}...", flush=True)
            dist.send(send_tensor_c, self.next_rank, group=self.group)
            # print(f"[rank{self.rank}] Sent to {self.next_rank}. Receiving from {self.prev_rank}...", flush=True)
            # Receive into the correctly sized slice
            dist.recv(recv_buffer_slice, self.prev_rank, group=self.group)
            # print(f"[rank{self.rank}] Received from {self.prev_rank}.", flush=True)
        else:
            # print(f"[rank{self.rank}] Receiving from {self.prev_rank}...", flush=True)
            # Receive into the correctly sized slice
            dist.recv(recv_buffer_slice, self.prev_rank, group=self.group)
            # print(f"[rank{self.rank}] Received from {self.prev_rank}. Sending to {self.next_rank}...", flush=True)
            dist.send(send_tensor_c, self.next_rank, group=self.group)
            # print(f"[rank{self.rank}] Sent to {self.next_rank}.", flush=True)

        # Return the relevant slice of the buffer
        return recv_buffer_slice


    def send_recv_kv(self, send_k: Tensor, send_v: Tensor, recv_k_buf: Tensor, recv_v_buf: Tensor) -> Tuple[Tensor, Tensor]:
        # Note: recv_k_buf and recv_v_buf are assumed to be pre-sliced correctly based on expected_recv_len in compute_sums
        # print(f"[rank{self.rank}] RingAttentionEngine.send_recv_kv: Sending K shape {send_k.shape}, V shape {send_v.shape} to rank {self.next_rank}, receiving into K shape {recv_k_buf.shape}, V shape {recv_v_buf.shape} from rank {self.prev_rank}", flush=True)
        # recv_k_buf and recv_v_buf MUST be correctly sized for the incoming tensors *before* calling this.
        """ Sends K, V to the next rank and receives K, V from the previous rank. """
        # Using explicit blocking send/recv with ordering like send_recv_tensor. recv_*_buf are assumed pre-sliced.

        send_k_c = send_k.contiguous()
        send_v_c = send_v.contiguous()
        recv_k_buf_c = recv_k_buf.contiguous() # Ensure contiguous for recv
        recv_v_buf_c = recv_v_buf.contiguous() # Ensure contiguous for recv

        if self.rank % 2 == 0:
            # print(f"[rank{self.rank}] Sending K/V to {self.next_rank}...", flush=True)
            dist.send(send_k_c, self.next_rank, group=self.group)
            dist.send(send_v_c, self.next_rank, group=self.group)
            # print(f"[rank{self.rank}] Sent K/V to {self.next_rank}. Receiving K/V from {self.prev_rank}...", flush=True)
            dist.recv(recv_k_buf_c, self.prev_rank, group=self.group)
            dist.recv(recv_v_buf_c, self.prev_rank, group=self.group)
            # print(f"[rank{self.rank}] Received K/V from {self.prev_rank}.", flush=True)
        else:
            # print(f"[rank{self.rank}] Receiving K/V from {self.prev_rank}...", flush=True)
            dist.recv(recv_k_buf_c, self.prev_rank, group=self.group)
            dist.recv(recv_v_buf_c, self.prev_rank, group=self.group)
            # print(f"[rank{self.rank}] Received K/V from {self.prev_rank}. Sending K/V to {self.next_rank}...", flush=True)
            dist.send(send_k_c, self.next_rank, group=self.group)
            dist.send(send_v_c, self.next_rank, group=self.group)
            # print(f"[rank{self.rank}] Sent K/V to {self.next_rank}.", flush=True)

        return recv_k_buf, recv_v_buf

    def raw_attention(self, q: Tensor, k: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor) -> Tensor:
        # q: [B, H, q_len, D]
        # k: [B, H, k_len, D]
        # q_indices: [q_len] (global indices)
        # k_indices: [k_len_origin] (global indices corresponding to the original shard k came from)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale

        # Create boolean masks (True means mask this position)
        final_mask = None
        q_len, k_len = q.shape[2], k.shape[2]

        # 1. Causal mask
        if self.is_causal:
            q_indices_dev = q_indices.to(scores.device)
            k_indices_dev = k_indices.to(scores.device)
            # Mask if key index is greater than query index (k > q)
            causal_mask_bool = k_indices_dev[None, :k_len] > q_indices_dev[:, None] # Shape [q_len, k_len]
            # Add batch and head dimensions for broadcasting
            final_mask = causal_mask_bool.unsqueeze(0).unsqueeze(0) # Shape [1, 1, q_len, k_len]

        # 2. Padding mask (additive mask converted to boolean)
        if mask is not None:
            # Input mask has shape [B, 1, q_len, k_len] or [B, H, q_len, k_len]
            padding_mask_bool = mask == -torch.inf
            if final_mask is None:
                final_mask = padding_mask_bool
            else:
                # OR combines the masks, broadcasting dimensions B and H
                final_mask = final_mask | padding_mask_bool

        # 3. Apply the combined mask if it exists
        if final_mask is not None:
            # print(f"[rank{self.rank}] raw_attention: scores shape {scores.shape}, final_mask shape {final_mask.shape}", flush=True) # DEBUG
            scores = scores.masked_fill(final_mask, -torch.inf)

        return scores

    def init_values(self, q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, H, q_len, D_head = q.shape; device, dtype = q.device, q.dtype
        D_v = self.attn.emb_v_per_head

        max_score = torch.full((B, H, q_len, 1), -torch.inf, dtype=dtype, device=device)
        numerator = torch.zeros(B, H, q_len, D_v, dtype=dtype, device=device)
        denominator = torch.zeros(B, H, q_len, 1, dtype=dtype, device=device)

        return max_score, numerator, denominator

    def update_max_attn(self, q: Tensor, k: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor, current_max_score: Tensor) -> Tensor:
        attn_scores = self.raw_attention(q, k, mask, q_indices, k_indices)
        block_max = attn_scores.masked_fill(attn_scores == -torch.inf, torch.finfo(attn_scores.dtype).min).amax(dim=-1, keepdim=True)
        return torch.maximum(current_max_score, block_max)

    def update_totals(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor, final_max_score: Tensor, current_num: Tensor, current_den: Tensor) -> Tuple[Tensor, Tensor]:
        # Store original dtype
        orig_dtype = q.dtype
        # Promote to fp32 for stable calculations
        q = q.float()
        k = k.float()
        v = v.float()
        final_max_score = final_max_score.float()
        current_num = current_num.float()
        current_den = current_den.float()

        # print(f"[rank{self.rank}] update_totals: Before raw_attention", flush=True)
        attn_scores = self.raw_attention(q, k, mask, q_indices, k_indices)

        # Check for NaNs/Infs, specifically positive infinity as -inf is expected from masking
        # It's okay if attn_scores has -inf from masking

        # print(f"[rank{self.rank}] update_totals: After raw_attention. Before stable_scores", flush=True)
        # Subtracting max score for stability. Ensure max_score isn't -inf where attn_scores is also -inf.
        # Replace -inf in final_max_score with a large negative number where attn_scores is also -inf to avoid inf - inf = nan
        safe_max_score = torch.where(final_max_score == -torch.inf, torch.finfo(final_max_score.dtype).min, final_max_score)
        stable_scores = attn_scores - safe_max_score
        # Replace potential NaN from (-inf - (-inf)) with -inf before exp
        stable_scores = torch.nan_to_num(stable_scores, nan=-torch.inf)

        stable_scores = torch.clamp(stable_scores, max=80.0, min = -80.0)

        exp_scores = torch.exp(stable_scores)
        # Check for NaNs/Infs after exponentiation (should ideally be only positive numbers or zero)
        if torch.isnan(exp_scores).any():
            print(f"[rank{self.rank}] WARNING: NaNs/Infs found in exp_scores!", flush=True)
        if (exp_scores == torch.inf).any():
            print(f"[rank{self.rank}] WARNING: Positive Infs found in exp_scores!", flush=True)

        # print(f"[rank{self.rank}] update_totals: After exp_scores. Before einsum", flush=True)
        num_update = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v)
        den_update = exp_scores.sum(dim=-1, keepdim=True)
        # print(f"[rank{self.rank}] update_totals: After einsum/sum", flush=True)

        # Check for NaNs/Infs in updates before adding
        if torch.isnan(num_update).any() or (num_update == torch.inf).any():
            print(f"[rank{self.rank}] WARNING: NaNs/Infs found in num_update!", flush=True)
        if torch.isnan(den_update).any() or (den_update == torch.inf).any():
            print(f"[rank{self.rank}] WARNING: NaNs/Infs found in den_update!", flush=True)
        # Cast back to original dtype before returning
        return (current_num + num_update).to(orig_dtype), (current_den + den_update).to(orig_dtype)
    

    # can you see me