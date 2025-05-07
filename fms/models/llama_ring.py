from typing import Optional, Tuple # Keep necessary types

import torch
import torch.nn as nn

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
    is_causal_mask: bool = False,
    attn_algorithm: Optional[str] = None,
    distributed_strategy: Optional[DistributedStrategy] = None, # Expected RingAttentionStrategy
) -> torch.Tensor: # Returns output tensor
    
    # 'distributed_strategy' from the arguments is used as 'strategy' here.
    strategy = distributed_strategy
    assert isinstance(strategy, RingAttentionStrategy), f"Expected RingAttentionStrategy, got {type(strategy)}"

    # x is the sharded, padded input from the LLaMABlock.forward call.
    # This is the input to the block, used for the final residual connection by the helper.
    residual_input = x
    x_norm_local = self.ln(x) # Normalize the input

    rank = strategy.rank # Get rank from strategy
    local_valid_len = strategy.get_local_valid_len() # Valid tokens on this rank

    # Lazy init RingAttentionHelper on first call with this strategy
    # self refers to the LLaMABlock instance
    if self.ring_helper is None or self.ring_helper.strategy is not strategy:
        self.ring_helper = RingAttentionHelper(
            attn_module=self.attn,
            strategy=strategy,
            llama_block=self, # Pass the LLaMABlock instance
            ff=self.ff_sub_layer, # Pass the feed-forward sub-layer from LLaMABlock
            ff_norm=self.ff_ln, # Pass the feed-forward layer norm from LLaMABlock
        )

    # Call helper's core logic
    # The RingAttentionHelper's forward method handles the attention and subsequent MLP.
    # It expects the normalized input (x_norm_local) and the original pre-norm input (residual_input)
    # for its internal residual connections.
    output_tensor = self.ring_helper.forward(
        x_norm_local, # Input to attention and MLP (after normalization)
        mask=mask,
        strategy=strategy,
        position_ids=position_ids, # Pass global position_ids
        is_causal_mask=is_causal_mask,
        valid_len=local_valid_len,
        residual=residual_input, # Original input to the block for the helper's residual logic
    )

    return output_tensor
