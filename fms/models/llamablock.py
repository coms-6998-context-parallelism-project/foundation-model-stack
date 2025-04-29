import logging
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple



from fms.models.LlamaConfig import BlockData, LLaMAConfig
import torch.nn.functional as F
import torch
import torch.nn as nn

import torch.distributed as dist
from fms import models
# from fms.models.attentionhelper import RingAttentionEngine
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils.activation import str_to_activation

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
from torch import Tensor, nn

from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit



logger = logging.getLogger(__name__)


# params emb_dim heads layers lr
#  7B    4096    32    32     3.0E-04
# 13B    5120    40    40     3.0E-04
# 33B    6656    52    60     1.5.E-04
# 65B    8192    64    80     1.5.E-04


class LLaMABlock(nn.Module):
    def __init__(self, config: LLaMAConfig, rotary_emb: RotaryEmbedding):
        super(LLaMABlock, self).__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads
        self.block_size = 32
        self.scale = math.sqrt(self.config.emb_dim // self.config.nheads)


        self.ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=self.config.attn_bias,
            position_encoder=rotary_emb,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def compute_local_qkv_and_rope(self, attn_data, q, k=None, v=None, position_ids=None, use_cache=False, past_key_value_state=None, is_self=True):
        B, T, _ = q.shape

        q_out, k_out, v_out = attn_data.in_proj(q, k, v)

        # Reshape q, k, v to (B, n_heads, T, d_head)
        queries = q_out.view(B, T, attn_data.nheads, attn_data.emb_kq_per_head).contiguous()
        keys = k_out.view(B, T, attn_data.kvheads, attn_data.emb_kq_per_head).contiguous()
        values = v_out.view(B, T, attn_data.kvheads, attn_data.emb_v_per_head).contiguous()

        if attn_data.position_encoder is not None:
            if position_ids is None:
                position_ids = torch.arange(T, device=q.device).unsqueeze(0).expand(B, -1)
            queries, keys = attn_data.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        # Transpose to (B, n_heads, T, d_head)
        queries = queries.transpose(1, 2).contiguous()
        keys = keys.transpose(1, 2).contiguous()
        values = values.transpose(1, 2).contiguous()

        return queries, keys, values


    
    def forward(
        self,
        x,
        *,
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        is_causal_mask=False,
        attn_algorithm=None,
    ):
        self.is_causal = is_causal_mask and mask is None

        # === Distributed setup
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # === 1. LayerNorm
        x_ln = self.ln(x)

        # Save original x separately
        x_raw = x

        # if rank == 0:
        #     print(f"[Debug Rank {rank}] After LayerNorm: x_ln.shape = {x_ln.shape}, mean={x_ln.mean().item():.5f}, std={x_ln.std().item():.5f}")

        # === 2. Gather x_ln and x_raw across ranks if needed
        if world_size > 1:
            x_ln_list = [torch.empty_like(x_ln) for _ in range(world_size)]
            dist.all_gather(x_ln_list, x_ln)
            x_ln = torch.cat(x_ln_list, dim=1).contiguous()

            x_raw_list = [torch.empty_like(x_raw) for _ in range(world_size)]
            dist.all_gather(x_raw_list, x_raw)
            x_raw = torch.cat(x_raw_list, dim=1).contiguous()

        # === 3. Handle position_ids
        if position_ids is None:
            position_ids = torch.arange(
                0, x_ln.shape[1], dtype=torch.long, device=x_ln.device
            ).unsqueeze(0)  # (B, T)
        elif world_size > 1:
            pos_list = [torch.empty_like(position_ids) for _ in range(world_size)]
            dist.all_gather(pos_list, position_ids)
            position_ids = torch.cat(pos_list, dim=1).contiguous()

        # === 4. Build block-diagonal mask if needed
        if mask is None and self.is_causal:
            B, T_full, _ = x_ln.shape  # B=1 usually, T_full = total tokens
            H = self.attn.nheads
            T_local = T_full // world_size
            mask = torch.full((B, H, T_full, T_full), float('-inf'), device=x_ln.device, dtype=x_ln.dtype)

            for r in range(world_size):
                start = r * T_local
                end = (r + 1) * T_local
                mask[:, :, start:end, start:end] = 0  # Only allow local block attending

        elif mask is not None and world_size > 1:
            mask_list = [torch.empty_like(mask) for _ in range(world_size)]
            dist.all_gather(mask_list, mask)
            mask = torch.cat(mask_list, dim=3).contiguous()

        if world_size > 1:
            dist.barrier()

        # === 5. Compute queries, keys, values
        queries, keys, values = self.compute_local_qkv_and_rope(
            self.attn,
            q=x_ln,
            k=x_ln,
            v=x_ln,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value_state=past_key_value_state,
            is_self=True,
        )

        # === 6. Handle past key-value caching
        if use_cache and past_key_value_state is not None and past_key_value_state[0].numel() > 0:
            keys = torch.cat((past_key_value_state[0], keys), dim=2)
            values = torch.cat((past_key_value_state[1], values), dim=2)

        # === 7. Expand for GQA if needed
        expansion = self.attn.nheads // self.attn.kvheads
        keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else keys
        values_e = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else values

        self.kv_global = keys_e
        self.vv_global = values_e
        self.kv_total_len = keys_e.shape[2]

        # === 8. Prepare blocks
        x_global = x_raw
        q_global = queries
        mask_global = mask

        T_q = q_global.shape[2]
        q_starts = list(range(0, T_q, self.block_size))
        num_blocks = len(q_starts)

        result_buffer: Dict[int, Tensor] = {}

        for block_id, q_start in enumerate(q_starts):
            q_end = min(q_start + self.block_size, T_q)
            q_block = q_global[:, :, q_start:q_end, :]
            x_block = x_global[:, q_start:q_end, :]

            block_data = BlockData(
                engine_instance=self,
                block_id=block_id,
                num_blocks=num_blocks,
                q_start=q_start,
                q_end=q_end,
                mask_global=mask_global,
                block_queues=None,
                await_max=None,
                await_sums=None,
                result_buffer=result_buffer,
                q_block=q_block,
                k_local=None,
                v_local=None,
                x_block=x_block
            )

            self.block_worker(block_data)

        if num_blocks == 0:
            dummy_out = torch.empty_like(x_global)
            if world_size > 1:
                T_local = x.shape[1]
                start_idx = rank * T_local
                end_idx = (rank + 1) * T_local
                dummy_out = dummy_out[:, start_idx:end_idx, :].contiguous()
            return dummy_out

        ordered_results = [result_buffer[q_start] for q_start in q_starts]
        x_out = torch.cat(ordered_results, dim=1)

        if world_size > 1:
            T_local = x.shape[1]
            start_idx = rank * T_local
            end_idx = (rank + 1) * T_local
            x_out = x_out[:, start_idx:end_idx, :].contiguous()

        return (x_out, (keys, values)) if use_cache else x_out





    def block_worker(self, args: BlockData):
        initial_max_score, initial_num, initial_den = self.init_values(args.q_block)

        final_max_score = self.compute_max_score(args, initial_max_score)
        final_num, final_den = self.compute_sums(args, final_max_score, initial_num, initial_den)

        args.result_buffer[args.q_start] = self.compute_block_output(args.x_block, final_num, final_den)


    """ compute max scores for stability (first flash pass) """
    def compute_max_score(self, args: BlockData, initial_max_score: Tensor) -> Tensor:
        device = args.q_block.device
        q_indices = torch.arange(args.q_start, args.q_end, device=device)

        max_score = initial_max_score

        for block_idx in range(args.num_blocks):
            k_start = block_idx * self.block_size
            k_end = min(k_start + self.block_size, self.kv_total_len)

            current_k = self.kv_global[:, :, k_start:k_end, :]
            current_k_idx = torch.arange(k_start, k_end, device=device)

            mask = None
            if args.mask_global is not None:
                mask = args.mask_global[:, :, args.q_start:args.q_end, k_start:k_end]

            max_score = self.update_max_attn(args.q_block, current_k, mask, q_indices, current_k_idx, max_score)

        return max_score


    """ sum loop """
    def compute_sums(self, args: BlockData, final_max_score: Tensor, initial_num: Tensor, initial_den: Tensor) -> Tuple[Tensor, Tensor]:
        device = args.q_block.device
        q_indices = torch.arange(args.q_start, args.q_end, device=device)

        num, den = initial_num, initial_den

        for block_idx in range(args.num_blocks):
            k_start = block_idx * self.block_size
            k_end = min(k_start + self.block_size, self.kv_total_len)

            current_k = self.kv_global[:, :, k_start:k_end, :]
            current_v = self.vv_global[:, :, k_start:k_end, :]
            current_k_idx = torch.arange(k_start, k_end, device=device)

            mask = None
            if args.mask_global is not None:
                mask = args.mask_global[:, :, args.q_start:args.q_end, k_start:k_end]

            num, den = self.update_totals(args.q_block, current_k, current_v, mask, q_indices, current_k_idx, final_max_score, num, den)

        return num, den


    """ final output """
    def compute_block_output(self, x: Tensor, num: Tensor, den: Tensor) -> Tensor:
        B, q_len, E = x.shape
        H, D_v = num.shape[1], num.shape[3]
        attn_out_h = num / (den + 1e-6)
        attn_out = attn_out_h.transpose(1, 2).contiguous().view(B, q_len, H * D_v)
        attn_out = self.attn.dense(attn_out)
        residual_1 = x + attn_out
        ff_out = self.ff_sub_layer(self.ff_ln(residual_1))
        return residual_1 + ff_out


    """ Helper Functions """

    def raw_attention(self, q: Tensor, k: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor) -> Tensor:
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale
        if mask is not None:
            scores = scores + mask
        if self.is_causal:
            q_indices_dev = q_indices.to(k_indices.device)
            causal_mask = (k_indices[None, :] > q_indices_dev[:, None]).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, -torch.inf)
        return scores

    def init_values(self, q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, H, q_len, D_head = q.shape
        device, dtype = q.device, q.dtype
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




