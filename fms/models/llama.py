from typing import Any, List, Mapping, Optional, Tuple, Union # Keep necessary types

import torch
import torch.nn as nn
import torch.distributed as dist

# Keep only FMS components used *in these functions*
from fms.models.ring_attention_helper import RingAttentionHelper
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy
from fms.modules.attention import MultiHeadAttention
from fms.modules.positions import RotaryEmbedding


# These functions are assigned as methods to the LLaMABlock class.
# `self` refers to the LLaMABlock instance.

def compute_local_qkv_and_rope(
    self: nn.Module, # LLaMABlock
    attn_data: MultiHeadAttention, # self.attn
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    is_self: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Q, K, V tensors

    B, T, E = q.shape
    q_out, k_out, v_out = attn_data.in_proj(q, k, v)

    assert q_out.shape == (B, T, E)
    assert k_out.shape == (B, T, E)
    assert v_out.shape == (B, T, E)

    H_q = attn_data.nheads
    H_kv = attn_data.kvheads
    D_kq = attn_data.emb_kq_per_head
    D_v = attn_data.emb_v_per_head

    queries = q_out.view(B, T, H_q, D_kq)
    keys    = k_out.view(B, T, H_kv, D_kq)
    values  = v_out.view(B, T, H_kv, D_v)

    assert queries.shape == (B, T, H_q, D_kq)
    assert keys.shape == (B, T, H_kv, D_kq)
    assert values.shape == (B, T, H_kv, D_v)

    if attn_data.position_encoder is not None and T > 0:
        assert position_ids is not None
        expected_pos_shape = (B, T)
        if not (T == 0 and position_ids.shape == (B, 0)) and position_ids.shape != expected_pos_shape:
                raise AssertionError(f"Expected position_ids shape {expected_pos_shape}, got {position_ids.shape}")

        valid_mask = position_ids != -1
        if valid_mask.any():
            max_pos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
            position_ids_safe = position_ids.clone()
            position_ids_safe[valid_mask] = position_ids_safe[valid_mask].clamp(0, max_pos - 1)

            queries_rope, keys_rope = attn_data.position_encoder.adjusted_qk(
                queries, keys, position_ids_safe
            )

            if valid_mask.all():
                queries = queries_rope
                keys = keys_rope
            else:
                mask_qkv = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(queries)
                queries = torch.where(mask_qkv, queries_rope, queries)
                keys    = torch.where(mask_qkv.expand_as(keys), keys_rope, keys)


    return (
        queries.permute(0, 2, 1, 3),
        keys.permute(0, 2, 1, 3),
        values.permute(0, 2, 1, 3)
    )


# Assigned to LLaMABlock.forward for RingAttentionStrategy
def forward_ring(
    self: nn.Module, # LLaMABlock
    x: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False, # Passed from LLaMA model
    is_causal_mask: bool = False,
    attn_algorithm: Optional[str] = None,
    distributed_strategy: Optional[DistributedStrategy] = None, # Expected RingAttentionStrategy
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:

    # Warn if use_cache is true, as it's likely not supported by the helper
    if use_cache:
         # logger.warning(f"Rank {dist.get_rank()}: use_cache=True passed to RingAttention forward. Caching is likely not fully supported.")
         pass # Keeping warning minimal or removed

    # Dispatch to helper function assigned to the block
    x_out, cache_out = self._forward_ring_attention(
        x,
        mask=mask,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        use_cache=use_cache,
        is_causal_mask=is_causal_mask,
        strategy=distributed_strategy,
    )

    return (x_out, cache_out) if use_cache else x_out


# Assigned to LLaMABlock._forward_ring_attention
def _forward_ring_attention(
    self: nn.Module, # LLaMABlock
    x: torch.Tensor, # Sharded, padded input
    *,
    mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    use_cache: bool,
    is_causal_mask: bool,
    strategy: DistributedStrategy, # Expected RingAttentionStrategy
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Any]: # (output, cache, extra)

    assert isinstance(strategy, RingAttentionStrategy), f"Expected RingAttentionStrategy, got {type(strategy)}"

    residual = x
    x_norm_local = self.ln(x)

    correct_valid_len = strategy.get_local_valid_len() # Valid tokens on this rank

    # Lazy init RingAttentionHelper on first call with this strategy
    if self.ring_helper is None or self.ring_helper.strategy is not strategy: # Check if strategy changed
        self.ring_helper = RingAttentionHelper(
            attn_module=self.attn,
            strategy=strategy,
            llama_block=self,
            use_cache=use_cache,
            ff=self.ff_sub_layer,
            ff_norm=self.ff_ln,
        )

    # Call helper's core logic
    output, cache_from_helper, extra_output = self.ring_helper.forward(
        x_norm_local,
        mask=mask,
        strategy=strategy,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        valid_len=correct_valid_len,
        residual=residual,
    )

    return output, cache_from_helper, extra_output # Helper returns (x, None, None) for ring case

