from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn

from fms.models.ring_attention_helper import RingAttentionHelper
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy


def forward_ring(
    self: nn.Module,  # LLaMABlock instance
    x: torch.Tensor,  # Sharded, valid-len-only input
    *,
    mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    is_causal_mask: bool = False,
    attn_algorithm: Optional[str] = None,
    distributed_strategy: Optional[DistributedStrategy] = None,
    debug_label: str = "RingAttentionBlock",
    layer_idx: int = -1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:

    x_out, cache_out, debug_out = self._forward_ring_attention(
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

    # Optionally collect debug output from layer 0, rank 0
    first_block_debug_out_ring = None
    if isinstance(distributed_strategy, RingAttentionStrategy):
        if distributed_strategy.rank == 0 and layer_idx == 0:
            first_block_debug_out_ring = x_out.clone()

    return (x_out, cache_out, first_block_debug_out_ring) if use_cache else (x_out, first_block_debug_out_ring)


def _forward_ring_attention(
    self: nn.Module,  # LLaMABlock instance
    x: torch.Tensor,  # Sharded, valid-len-only input
    *,
    mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    use_cache: bool,
    is_causal_mask: bool,
    strategy: DistributedStrategy,
    debug_label: str = "RingAttentionBlock",
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Any]:

    assert isinstance(strategy, RingAttentionStrategy), f"Expected RingAttentionStrategy, got {type(strategy)}"

    residual = x
    x_norm_local = self.ln(x)

    rank = strategy.rank
    valid_len = strategy.get_local_valid_len()

    # Initialize RingAttentionHelper if needed
    if self.ring_helper is None or self.ring_helper.strategy is not strategy:
        self.ring_helper = RingAttentionHelper(
            attn_module=self.attn,
            strategy=strategy,
            llama_block=self,
            use_cache=use_cache,
            ff=self.ff_sub_layer,
            ff_norm=self.ff_ln,
        )

    # Debug: capture input to RingAttentionHelper on layer 0, rank 0
    if rank == 0 and layer_idx == 0:
        # print(f"[DEBUG] Input norm to RingAttentionHelper: {torch.linalg.norm(x_norm_local.float()).item()}")
        pass

    output, cache_out, debug_extra = self.ring_helper.forward(
        x_norm_local,
        mask=mask,
        strategy=strategy,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        valid_len=valid_len,
        residual=residual,
    )

    # Debug: capture output from RingAttentionHelper
    if rank == 0 and layer_idx == 0:
        # print(f"[DEBUG] Output norm from RingAttentionHelper: {torch.linalg.norm(output.float()).item()}")
        pass

    return output, cache_out, debug_extra
