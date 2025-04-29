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
        self.block_size = 64
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

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        T_local = x.shape[1]

        # Dynamically adjust block size if necessary
        if T_local > self.block_size:
            self.block_size = T_local


        # Local LayerNorm
        x_ln = self.ln(x)
        x_raw = x

        # Local Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                rank * T_local, (rank + 1) * T_local, device=x.device
            ).unsqueeze(0)

        # Local mask construction
        if mask is None and self.is_causal:
            B, T_local, _ = x_ln.shape
            H = self.attn.nheads
            mask = torch.full((B, H, T_local, T_local), float('-inf'), device=x.device, dtype=x_ln.dtype)
            mask[:, :, torch.arange(T_local), torch.arange(T_local)] = 0
        elif mask is not None:
            mask = mask[:, :, :, rank * T_local : (rank + 1) * T_local]

        # Local QKV computation
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

        if use_cache and past_key_value_state is not None and past_key_value_state[0].numel() > 0:
            keys = torch.cat((past_key_value_state[0], keys), dim=2)
            values = torch.cat((past_key_value_state[1], values), dim=2)

        # GQA expansion
        expansion = self.attn.nheads // self.attn.kvheads
        keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else keys
        values_e = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else values

        self.kv_local = keys_e   # Fix: call it local, not global
        self.vv_local = values_e
        self.kv_total_len = keys_e.shape[2]

        # Block-wise attention
        x_global = x_raw
        q_global = queries
        mask_global = mask

        T_q = q_global.shape[2]
        assert T_q <= self.block_size, "Local query size exceeds block size!"

        block_data = BlockData(
            engine_instance=self,
            block_id=0,
            num_blocks=1,
            q_start=0,
            q_end=T_q,
            mask_global=mask_global,
            block_queues=None,
            await_max=None,
            await_sums=None,
            result_buffer={},
            q_block=q_global,
            k_local=None,
            v_local=None,
            x_block=x_global,
        )

        self.block_worker(block_data)

        x_out = block_data.result_buffer[0]


        return (x_out, (keys, values)) if use_cache else x_out





    def block_worker(self, args: BlockData):
        initial_max_score, initial_num, initial_den = self.init_values(args.q_block)

        final_max_score = self.compute_max_score_ring(args, initial_max_score)
        final_num, final_den = self.compute_sums_ring(args, final_max_score, initial_num, initial_den)

        args.result_buffer[args.q_start] = self.compute_block_output(args.x_block, final_num, final_den)



    def compute_max_score_ring(self, args: BlockData, initial_max_score: Tensor) -> Tensor:
        device = args.q_block.device
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        current_k = self.kv_local

        max_score = initial_max_score

        T_local = current_k.shape[2]

        for step in range(world_size):
            global_k_start = step * T_local
            global_k_end = (step + 1) * T_local

            q_indices = torch.arange(args.q_start, args.q_end, device=device)
            k_indices = torch.arange(global_k_start, global_k_end, device=device)

            # No mask here
            max_score = self.update_max_attn(
                args.q_block,
                current_k,
                None,
                q_indices,
                k_indices,
                max_score,
            )

            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size

            if step < world_size - 1:
                current_k = self.ring_exchange(current_k, send_rank, recv_rank, tag=step)

        return max_score



    def compute_sums_ring(self, args: BlockData, final_max_score: Tensor, initial_num: Tensor, initial_den: Tensor) -> Tuple[Tensor, Tensor]:
        device = args.q_block.device
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        current_k = self.kv_local
        current_v = self.vv_local

        num, den = initial_num, initial_den

        T_local = current_k.shape[2]

        for step in range(world_size):
            global_k_start = step * T_local
            global_k_end = (step + 1) * T_local

            q_indices = torch.arange(args.q_start, args.q_end, device=device)
            k_indices = torch.arange(global_k_start, global_k_end, device=device)

            num, den = self.update_totals(
                args.q_block,
                current_k,
                current_v,
                None,
                q_indices,
                k_indices,
                final_max_score,
                num,
                den,
            )

            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size

            if step < world_size - 1:
                current_k = self.ring_exchange(current_k, send_rank, recv_rank, tag=step)
                current_v = self.ring_exchange(current_v, send_rank, recv_rank, tag=5000 + step)

        return num, den


    def ring_exchange(self, tensor: Tensor, send_rank: int, recv_rank: int, tag: int) -> Tensor:
        send_tensor = tensor.contiguous()
        recv_tensor = torch.empty_like(send_tensor)

        send_req = dist.isend(send_tensor, dst=send_rank, tag=tag)
        dist.recv(recv_tensor, src=recv_rank, tag=tag)
        send_req.wait()

        return recv_tensor




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

    def raw_attention(self, q: Tensor, k: Tensor, mask: Optional[Tensor], q_indices: Optional[Tensor], k_indices: Optional[Tensor]) -> Tensor:
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale

        if mask is not None:
            # (Rare) If user supplied a global mask (non-causal)
            scores = scores + mask

        if self.is_causal and (q_indices is not None) and (k_indices is not None):
            q_len = q.shape[2]
            k_len = k.shape[2]
            causal_mask = (k_indices.unsqueeze(0) > q_indices.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # (1,1,Q,K)
            scores = scores.masked_fill(causal_mask, float('-inf'))

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




