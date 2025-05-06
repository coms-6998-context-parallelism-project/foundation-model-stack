from typing import Any, List, Mapping, Optional, Tuple, Union # Keep necessary types

import torch
import torch.nn as nn
import torch.distributed as dist

# Keep only FMS components used directly in these functions or their type hints
from fms.models.ring_attention_helper import RingAttentionHelper
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy # Need both for type hints


# Assigned to LLaMABlock.forward when RingAttentionStrategy is used
def forward_ring(
    self: nn.Module, # LLaMABlock instance
    x: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False, # Passed from LLaMA model
    is_causal_mask: bool = False,
    attn_algorithm: Optional[str] = None,
    distributed_strategy: Optional[DistributedStrategy] = None, # Expected RingAttentionStrategy
    # Add debug passthrough
    debug_label: str = "RingAttentionBlock",
    layer_idx: int = -1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:

    # Dispatch to helper function assigned to the block
    # Unpacks 3 values from _forward_ring_attention
    x_out, cache_out, _ = self._forward_ring_attention(
        x,
        mask=mask,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        use_cache=use_cache,
        is_causal_mask=is_causal_mask,
        strategy=distributed_strategy,
        debug_label=debug_label,
        layer_idx=layer_idx,
    )

    # Return based on the use_cache flag
    return (x_out, cache_out) if use_cache else x_out


# Assigned to LLaMABlock._forward_ring_attention
def _forward_ring_attention(
    self: nn.Module, # LLaMABlock instance
    x: torch.Tensor, # Sharded, padded input
    *,
    mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    use_cache: bool,
    is_causal_mask: bool,
    strategy: DistributedStrategy, # Expected RingAttentionStrategy
    # Add debug passthrough
    debug_label: str = "RingAttentionBlock", # Default, will be overridden
    layer_idx: int = -1, # Default, will be overridden
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Any]: # Returns (output, cache, extra)

    assert isinstance(strategy, RingAttentionStrategy), f"Expected RingAttentionStrategy, got {type(strategy)}"

    residual = x
    x_norm_local = self.ln(x)

    rank = strategy.rank # Get rank from strategy
    correct_valid_len = strategy.get_local_valid_len() # Valid tokens on this rank

    # Lazy init RingAttentionHelper on first call with this strategy
    if self.ring_helper is None or self.ring_helper.strategy is not strategy:
        self.ring_helper = RingAttentionHelper(
            attn_module=self.attn,
            strategy=strategy,
            llama_block=self,
            use_cache=use_cache,
            ff=self.ff_sub_layer,
            ff_norm=self.ff_ln,
        )

    # Debug print for input to RingAttentionHelper.forward (Rank 0's portion)
    if rank == 0 and layer_idx == 0:
        print(f"DEBUG ({debug_label}, Layer {layer_idx}, Rank {rank}): Input to RingHelper.forward (x_norm_local) norm = {torch.linalg.norm(x_norm_local.float()).item()}")

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
    
    # Debug print for output of RingAttentionHelper.forward (Rank 0's portion)
    # This 'output' is the block's final output for this rank, padded.
    if rank == 0 and layer_idx == 0:
        print(f"DEBUG ({debug_label}, Layer {layer_idx}, Rank {rank}): Output from RingHelper.forward (Block Output, Rank 0 portion) norm = {torch.linalg.norm(output.float()).item()}")

    return output, cache_from_helper, extra_output
