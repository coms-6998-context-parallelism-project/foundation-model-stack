import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy



def ring_forward(
    self,
    x,
    *,
    mask=None,
    position_ids=None,
    past_key_value_state=None,
    use_cache=False,
    is_causal_mask=False,
    attn_algorithm=None,
    distributed_strategy: Optional[DistributedStrategy] = None,
):

    residual = x 
    x_norm = self.ln(x)


    x = RingAttentionKernel.ring_attention(
        x_norm=x_norm,
        attn_module=self.attn,
        strategy=distributed_strategy,
        valid_len=distributed_strategy._local_valid_len,
        mask=mask, 
        position_ids=position_ids, # Global position_ids
        causal=is_causal_mask,
    )
    
    # use cache and dropout have not yet been implemented / tested
    x = x + residual

    # then we do FF and Add&Norm
    residual = x
    x = self.ff_ln(x)
    x = self.ff_sub_layer(x)
    x = x + residual

    if use_cache:
        return (x, None)
    else:
        return x

class RingAttentionKernel:

    @staticmethod
    def ring_attention(
        x_norm: Tensor,
        attn_module: MultiHeadAttention,
        strategy: RingAttentionStrategy,
        valid_len: int,
        *,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        causal: bool = False,
    ) -> Tensor:
        print(valid_len, end = ", ")
        batch_size, block_size, emb_dim = x_norm.shape 
        assert block_size == strategy.block_size
        current_rank_token_global_start_idx = strategy.rank * strategy.block_size

        q, k, v = RingAttentionKernel._compute_qkv_and_rope(
            attn_module, x_norm, position_ids
        )

        scale = attn_module.scale_factor or math.sqrt(attn_module.emb_kq_per_head)
        accum_dtype = torch.float32

        # Cast QKV and mask to accum_dtype once
        q_fp32 = q.to(accum_dtype)
        k_fp32 = k.to(accum_dtype)
        v_fp32 = v.to(accum_dtype)
        mask = mask.to(accum_dtype)

        # Pre-allocate reduction buffers
        nheads = q_fp32.shape[1]
        emb_v_per_head = v_fp32.shape[-1]
        max_score_buffer = torch.full((batch_size, nheads, block_size, 1),
                                   torch.finfo(accum_dtype).min,
                                   device=q_fp32.device, dtype=accum_dtype)
        numerator_buffer = torch.zeros((batch_size, nheads, block_size, emb_v_per_head), 
                                     device=q_fp32.device, dtype=accum_dtype)
        denominator_buffer = torch.zeros((batch_size, nheads, block_size, 1), 
                                       device=q_fp32.device, dtype=accum_dtype)

        # Pre-compute global indices for queries and a base for keys
        query_global_indices = torch.arange(current_rank_token_global_start_idx, 
                                            current_rank_token_global_start_idx + block_size, 
                                            device=q_fp32.device)

        base_key_indices_for_block = torch.arange(0, strategy.block_size, device=q_fp32.device)


        # main ring attention 
        out = RingAttentionKernel._compute_attention_ring(
            q_fp32, k_fp32, v_fp32, mask, strategy, 
            block_size, scale, causal,
            max_score_buffer, numerator_buffer, denominator_buffer,
            query_global_indices, base_key_indices_for_block
        )

        proj = out.transpose(1, 2).reshape(batch_size, strategy.block_size, -1)
        out = attn_module.dense(proj.to(x_norm.dtype)) 

        return out


    @staticmethod
    def _compute_qkv_and_rope(
        attn: MultiHeadAttention,
        x: Tensor,
        rope_position_ids: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        q_proj, k_proj, v_proj = attn.in_proj(x, None, None)
        nheads, kvheads = attn.nheads, attn.kvheads
        emb_kq_per_head, emb_v_per_head = attn.emb_kq_per_head, attn.emb_v_per_head

        # reshape & apply RoPE if needed
        q = q_proj.view(batch_size, seq_len, nheads, emb_kq_per_head)
        k = k_proj.view(batch_size, seq_len, kvheads, emb_kq_per_head)
        v = v_proj.view(batch_size, seq_len, kvheads, emb_v_per_head)
        if attn.position_encoder and seq_len:
            assert rope_position_ids is not None
            valid_rope_pos_mask = rope_position_ids.ne(-1)
            if valid_rope_pos_mask.any():
                rope_internal_max_seq_len = getattr(attn.position_encoder, "max_seq_len", 2048)
                clamped_rope_ids = rope_position_ids.clamp(0, rope_internal_max_seq_len - 1)
                q, k = attn.position_encoder.adjusted_qk(q, k, clamped_rope_ids)

        q, k, v = [x_tensor.permute(0, 2, 1, 3) for x_tensor in (q, k, v)]
        if nheads != kvheads:
            kv_to_q_head_ratio = nheads // kvheads
            k = k.repeat_interleave(kv_to_q_head_ratio, dim=1)
            v = v.repeat_interleave(kv_to_q_head_ratio, dim=1)
        return q, k, v
    
    @staticmethod
    def _compute_attention_ring(
        q_fp32: Tensor,
        k_fp32: Tensor,
        v_fp32: Tensor,
        mask: Optional[Tensor],
        strategy: RingAttentionStrategy,
        block_size: int,   # number of queries in q for this rank's block (num_queries_in_block)
        scale: float,
        causal: bool,
        # Pre-allocated buffers and indices
        max_score_buffer: Tensor,
        numerator_buffer: Tensor,
        denominator_buffer: Tensor,
        query_global_indices: Tensor,
        base_key_indices_for_block: Tensor
    ) -> Tensor:
        
        RingAttentionKernel._max_pass(
            q_fp32, k_fp32, mask,block_size, strategy, scale, causal, 
            max_score_buffer, query_global_indices, base_key_indices_for_block
        )

        RingAttentionKernel._sum_pass(
            q_fp32, k_fp32, v_fp32, mask, block_size, 
            max_score_buffer, strategy, scale, causal,
            numerator_buffer, denominator_buffer, query_global_indices, base_key_indices_for_block
        )
        return (numerator_buffer / (denominator_buffer + torch.finfo(denominator_buffer.dtype).eps)).to(q_fp32.dtype)
    

    @staticmethod
    def _attn_scores(
        Q: Tensor,
        K: Tensor,
        query_indices: Tensor,
        key_indices: Tensor, 
        scale: float,
        mask: Optional[Tensor],
        causal: bool,
    ) -> Tensor:
        batch_size, nheads, num_q, _ = Q.shape 
        num_k = K.shape[2]  
        if num_q == 0 or num_k == 0:
            return Q.new_empty((batch_size, nheads, num_q, num_k))

        scores = torch.matmul(Q / scale, K.transpose(-2, -1))
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        if causal:
            # build a [1,1,q_len,k_len] mask where key_pos > query_pos
            future_mask = (key_indices[None, :] > query_indices[:, None])
            future_mask = future_mask.unsqueeze(0).unsqueeze(0) 
            scores = scores.masked_fill(future_mask, float("-inf"))
        return scores
    
    @staticmethod
    def _max_pass(
        q: Tensor,
        k: Tensor, 
        mask: Optional[Tensor],
        block_size: int, 
        strategy: RingAttentionStrategy,
        scale: float,
        causal: bool,

        max_score_buffer: Tensor,
        query_global_indices: Tensor,
        base_key_indices_for_block: Tensor
    ) -> Tensor:

        q_fp32, k_fp32_current, mask_current = q, k, mask

        for i in range(strategy.world_size):
            source_rank = (strategy.rank - i) % strategy.world_size
            block_offset_for_source_rank = source_rank * strategy.block_size
            k_len_current_block = k_fp32_current.shape[2]
            
            if block_size and k_len_current_block:

                key_indices_slice = base_key_indices_for_block[:k_len_current_block]
                key_block_global_indices = key_indices_slice + block_offset_for_source_rank
                
                current_scores = RingAttentionKernel._attn_scores(q_fp32, k_fp32_current, query_global_indices, key_block_global_indices, scale, mask_current, causal)
                torch.maximum(max_score_buffer, current_scores.amax(dim=-1, keepdim=True), out=max_score_buffer)

            # no need for last round communication
            if i < strategy.world_size - 1:
                # ring attention communication -- shift kvs
                k_fp32_current, _ = strategy._ring_shift_tensor(k_fp32_current, k_len_current_block)
                mask_current, _ = strategy._ring_shift_tensor(mask_current, k_len_current_block)

    @staticmethod
    def _sum_pass(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        block_size: int, 
        max_score: Tensor,
        strategy: RingAttentionStrategy,
        scale: float,
        causal: bool,
        numerator_buffer: Tensor,
        denominator_buffer: Tensor,
        query_global_indices: Tensor,
        base_key_indices_for_block: Tensor
    ) -> Tuple[Tensor, Tensor]:

        q_fp32, k_fp32_current, v_fp32_current, mask_current = q, k, v, mask
        
        accum_dtype_val = q_fp32.dtype
        log_min_exp_threshold = math.log(torch.finfo(accum_dtype_val).tiny) + 1.0
        log_max_exp_threshold = math.log(torch.finfo(accum_dtype_val).max) - 1.0

        for i in range(strategy.world_size):
            source_rank = (strategy.rank - i) % strategy.world_size
            block_offset_for_source_rank = source_rank * strategy.block_size
            k_len_current_block = k_fp32_current.shape[2]

            if block_size and k_len_current_block:

                key_indices_slice = base_key_indices_for_block[:k_len_current_block]
                key_block_global_indices = key_indices_slice + block_offset_for_source_rank

                current_scores = RingAttentionKernel._attn_scores(q_fp32, k_fp32_current, query_global_indices, key_block_global_indices, scale, mask_current, causal)
                score_delta = torch.where(torch.isneginf(max_score), float("-inf"), current_scores - max_score)
                exp_scores = torch.exp(score_delta.clamp(min=log_min_exp_threshold, max=log_max_exp_threshold))
                exp_scores = exp_scores.masked_fill(torch.isneginf(max_score), 0.0)
                numerator_buffer.add_(torch.matmul(exp_scores, v_fp32_current.narrow(2, 0, k_len_current_block)))
                denominator_buffer.add_(exp_scores.sum(dim=-1, keepdim=True))
            
            # no need for last round communication
            if i < strategy.world_size - 1:
                # ring attention communication -- shift kvs

                # should ideally all happen in one pass together
                k_fp32_current, _ = strategy._ring_shift_tensor(k_fp32_current, k_len_current_block)
                v_fp32_current, _ = strategy._ring_shift_tensor(v_fp32_current, k_len_current_block)
                mask_current, _ = strategy._ring_shift_tensor(mask_current, k_len_current_block)
    
