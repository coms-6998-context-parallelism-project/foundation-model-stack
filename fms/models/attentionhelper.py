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

# fake ring attention, for now (no multi gpu setup quite yet)
class RingAttentionEngine:
    def __init__(self, block_size: int, attn: MultiHeadAttention, ff: GatedLinearUnit, ff_norm: nn.Module, is_causal: bool, group: dist.ProcessGroup):
        self.block_size = block_size
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

    # main method
    def forward_full(self, q_shard: Tensor, k_shard: Tensor, v_shard: Tensor, mask_global: Optional[Tensor], x_shard: Tensor, q_global_offset: int, global_seq_len: int) -> Tensor:
        """
        Performs Ring Attention computation across the distributed group.
        Assumes q_shard, k_shard, v_shard, x_shard are the *local shards* for the current rank.
        q_global_offset indicates the starting position of this shard in the global sequence.
        global_seq_len is the total sequence length across all ranks.
        Returns the computed output shard for the current rank.
        """
        T_q_local = q_shard.shape[2]
        if T_q_local == 0: return torch.empty_like(x_shard)

        # Determine the query block this rank is responsible for
        # The entire local shard is the query block for this rank
        q_local_start = 0
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
        final_max_score = self.compute_max_score(block_data, initial_max_score, global_seq_len)
        dist.barrier(self.group) # Ensure all ranks have computed their max score

        # 3. Second pass: Compute numerator and denominator sums
        final_num, final_den = self.compute_sums(block_data, final_max_score, initial_num, initial_den, global_seq_len)
        dist.barrier(self.group) # Ensure all ranks have computed their sums

        # 4. Compute local output block
        output_block = self.compute_block_output(x_block, final_num, final_den)

        # Return the locally computed block. Gathering happens outside.
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
        
        # Temporary buffer for receiving K
        # Determine max shard size for buffer allocation
        max_shard_len = (global_seq_len + args.world_size - 1) // args.world_size
        recv_shape = list(args.k_shard.shape)
        recv_shape[2] = max_shard_len
        recv_k = torch.empty(recv_shape, dtype=args.k_shard.dtype, device=device)

        for i in range(args.world_size):
            # Calculate global indices and length for the *current* K shard
            current_k_len = (global_seq_len + args.world_size - 1) // args.world_size
            current_k_global_offset = current_k_rank * current_k_len
            # Adjust length for the last rank
            current_k_len = min(current_k_len, global_seq_len - current_k_global_offset)
            current_k_global_indices = torch.arange(current_k_global_offset, current_k_global_offset + current_k_len, device=device)

            # Slice the received buffer if necessary (only needed if shapes differ)
            effective_recv_k = recv_k[:, :, :current_k_len, :] if i > 0 else current_k_shard

            # Compute attention with current K block
            # Mask needs to be sliced using global indices
            mask = args.mask_global[:, :, args.q_global_offset:args.q_global_offset+args.q_shard.shape[2], current_k_global_offset:current_k_global_offset+current_k_len] if args.mask_global is not None else None
            max_score = engine.update_max_attn(args.q_shard, effective_recv_k, mask, q_global_indices, current_k_global_indices, max_score)

            # Send current_k to next rank, receive k from previous rank
            # Use the buffer that can hold the largest possible shard
            recv_k_slice = recv_k[:, :, :current_k_shard.shape[2], :] # Slice buffer for send_recv
            self.send_recv_tensor(current_k_shard, recv_k_slice)
            current_k_shard = recv_k_slice # The received tensor is now in the potentially larger buffer

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
        # Determine max shard size for buffer allocation
        max_shard_len = (global_seq_len + args.world_size - 1) // args.world_size
        recv_k_shape = list(args.k_shard.shape); recv_k_shape[2] = max_shard_len
        recv_v_shape = list(args.v_shard.shape); recv_v_shape[2] = max_shard_len
        recv_k = torch.empty(recv_k_shape, dtype=args.k_shard.dtype, device=device)
        recv_v = torch.empty(recv_v_shape, dtype=args.v_shard.dtype, device=device)

        for i in range(args.world_size):
            # Calculate global indices and length for the *current* K/V shards
            current_kv_len = (global_seq_len + args.world_size - 1) // args.world_size
            current_kv_global_offset = current_kv_rank * current_kv_len
            # Adjust length for the last rank
            current_kv_len = min(current_kv_len, global_seq_len - current_kv_global_offset)
            current_k_global_indices = torch.arange(current_kv_global_offset, current_kv_global_offset + current_kv_len, device=device)

            # Slice the received buffers if necessary
            effective_recv_k = recv_k[:, :, :current_kv_len, :] if i > 0 else current_k_shard
            effective_recv_v = recv_v[:, :, :current_kv_len, :] if i > 0 else current_v_shard

            # Compute attention with current K, V blocks
            # Mask needs to be sliced using global indices
            mask = args.mask_global[:, :, args.q_global_offset:args.q_global_offset+args.q_shard.shape[2], current_kv_global_offset:current_kv_global_offset+current_kv_len] if args.mask_global is not None else None
            num, den = engine.update_totals(args.q_shard, effective_recv_k, effective_recv_v, mask, q_global_indices, current_k_global_indices, final_max_score, num, den)
            
            # Send current K, V to next rank, receive K, V from previous rank
            # Use slices of the potentially larger buffers for send/recv
            recv_k_slice = recv_k[:, :, :current_k_shard.shape[2], :]
            recv_v_slice = recv_v[:, :, :current_v_shard.shape[2], :]
            self.send_recv_kv(current_k_shard, current_v_shard, recv_k_slice, recv_v_slice)
            current_k_shard = recv_k_slice
            current_v_shard = recv_v_slice

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

    def send_recv_tensor(self, send_tensor: Tensor, recv_buffer: Tensor) -> Tensor:
        """ Sends a tensor to the next rank and receives one from the previous rank. """
        # Using blocking send_recv. recv_buffer might be larger than send_tensor.
        # Assumes recv_buffer is sliced correctly before calling. #TODO: Verify this assumption
        dist.send_recv(send_tensor, self.next_rank, recv_tensor=recv_buffer, source=self.prev_rank, group=self.group)
        return recv_buffer

    def send_recv_kv(self, send_k: Tensor, send_v: Tensor, recv_k_buf: Tensor, recv_v_buf: Tensor) -> Tuple[Tensor, Tensor]:
        """ Sends K, V to the next rank and receives K, V from the previous rank. """
        # This could be done with separate send/recv or combined ops if available/efficient.
        # Using separate send/recv for clarity now. Ensure order matches on both sides.
        ops = []
        ops.append(dist.P2POp(dist.isend, send_k, self.next_rank, group=self.group))
        ops.append(dist.P2POp(dist.isend, send_v, self.next_rank, group=self.group))
        ops.append(dist.P2POp(dist.irecv, recv_k_buf, self.prev_rank, group=self.group))
        ops.append(dist.P2POp(dist.irecv, recv_v_buf, self.prev_rank, group=self.group))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs: req.wait()
        return recv_k_buf, recv_v_buf

    def raw_attention(self, q: Tensor, k: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor) -> Tensor:
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale
        if mask is not None: scores = scores + mask
        if self.is_causal:
            q_indices_dev = q_indices.to(k_indices.device)
            causal_mask = (k_indices[None, :] > q_indices_dev[:, None]).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, -torch.inf)
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
        attn_scores = self.raw_attention(q, k, mask, q_indices, k_indices)
        stable_scores = (attn_scores - final_max_score).clamp(min=-10.0, max=10.0)
        exp_scores = torch.exp(stable_scores)
        num_update = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v)
        den_update = exp_scores.sum(dim=-1, keepdim=True)
        return current_num + num_update, current_den + den_update