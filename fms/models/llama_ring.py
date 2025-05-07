from typing import Any, List, Mapping, Optional, Tuple, Union # Keep necessary types
import sys  # For flushing stdout
import torch.nn.functional as F # For shadow pass
import math # For shadow pass

import torch
import torch.linalg # For norms in debug prints
import torch.nn as nn
import torch.distributed as dist

# Keep only FMS components used directly in these functions or their type hints
from fms.models.ring_attention_helper import RingAttentionHelper
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy # Need both for type hints

# ============================================================
# Placeholder for Debug Comparison Functions (from original script)
# You need to copy your actual implementations here.
# ============================================================
def _compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, layer_idx: int, tolerance: float = 1e-3, print_values: bool = False):
    """Placeholder helper to compare two tensors and print debug info."""
    print(f"[DEBUG L{layer_idx}] Comparing {name}...")
    sys.stdout.flush()
    if t1.shape != t2.shape:
        print(f"  MISMATCH SHAPE for {name}: Ring {t1.shape} vs NoRing {t2.shape}")
        sys.stdout.flush()
        return False

    # Ensure same dtype for comparison, cast t2 to t1's dtype if different
    if t1.dtype != t2.dtype:
        try:
            t2_casted = t2.to(t1.dtype)
        except Exception as e:
            print(f"  ERROR: Failed to cast NoRing tensor {name} from {t2.dtype} to {t1.dtype}: {e}")
            sys.stdout.flush()
            return False
    else:
        t2_casted = t2

    if not torch.allclose(t1, t2_casted, atol=tolerance, rtol=tolerance):
        diff = torch.abs(t1 - t2_casted)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  MISMATCH VAL for {name} (MaxDiff: {max_diff:.2e}, MeanDiff: {mean_diff:.2e}, RingNorm: {torch.linalg.norm(t1.float()).item():.2e}, NoRingNorm: {torch.linalg.norm(t2_casted.float()).item():.2e})")
        sys.stdout.flush()
        if print_values:
            print(f"    Ring tensor ({name}, shape {t1.shape}):\n{t1.flatten()[:5]}...\n{t1.flatten()[-5:]}")
            sys.stdout.flush()
            print(f"    NoRing tensor ({name}, shape {t2_casted.shape}):\n{t2_casted.flatten()[:5]}...\n{t2_casted.flatten()[-5:]}")
            sys.stdout.flush()
            # print(f"  Difference tensor ({name}):\n{diff.flatten()[:5]}...\n{diff.flatten()[-5:]}")
            # sys.stdout.flush()
        return False
    else:
        print(f"  MATCH for {name} (RingNorm: {torch.linalg.norm(t1.float()).item():.2e}, NoRingNorm: {torch.linalg.norm(t2_casted.float()).item():.2e})")
        sys.stdout.flush()
        return True

def _compare_debug_data(
    debug_no_ring: dict, 
    debug_ring_current_rank: dict, # Renamed from debug_ring_rank0
    strategy: RingAttentionStrategy, 
    layer_idx: int, 
    tolerance: float, 
    print_values: bool,
    current_rank_for_comparison: int, # New parameter
    current_rank_ring_key_prefix: str # New parameter (e.g., "ring_r0")
):
    """Placeholder comparison orchestrator."""
    print(f"\n--- Comparing Debug Data for Layer {layer_idx}, Rank {current_rank_for_comparison} ---")
    sys.stdout.flush()
    if not debug_no_ring or not debug_ring_current_rank:
        print("[DEBUG] One or both debug dictionaries are empty. Skipping comparison.")
        print(f"  NoRing dict populated: {bool(debug_no_ring)}")
        print(f"  Ring dict (Rank {current_rank_for_comparison}) populated: {bool(debug_ring_current_rank)}")
        sys.stdout.flush()
        return

    current_rank_valid_len = strategy.get_local_valid_len() # This is the valid_len for the current rank being compared
    original_total_len = strategy._original_seq_len if hasattr(strategy, '_original_seq_len') else -1
    print(f"[DEBUG L{layer_idx}] Rank {current_rank_for_comparison} Valid Len: {current_rank_valid_len}, Original Total Len: {original_total_len}")
    sys.stdout.flush()

    # Define mappings from no_ring keys (global) to ring keys (rank0 local)
    # Ensure these keys match what's populated in the dicts
    comparison_map = [
        # Format: (Display Name, NoRing Key Suffix, Ring Key Suffix, Global Seq Dim for Slicing NoRing)
        ("X_norm", "_x_norm_global", "_x_norm", 1),
        ("Q_local", "_q_global", "_attn_q_local", 2),
        ("K_local", "_k_global", "_attn_k_local", 2),
        ("V_local", "_v_global", "_attn_v_local", 2),
        ("Attn_Dense_Out", "_attn_dense_out_global", "_attn_out_raw", 1),
        ("Residual1", "_residual1_global", "_attn_out_residual", 1),
        ("FF_LN_Out", "_ff_ln_out_global", "_ff_ln_out", 1),
        ("FF_Out", "_ff_out_global", "_ff_out_raw", 1),
        ("Block_Output", "_block_output_global", "_block_output", 1),
    ]

    all_match = True
    for name, global_key_suffix, ring_key_suffix, global_seq_dim in comparison_map:
        # --- CHANGE IS HERE ---
        # Construct the full key expected in debug_no_ring
        full_global_key = f"noring{global_key_suffix}"

        # Check for the full key in debug_no_ring
        if full_global_key not in debug_no_ring:
            print(f"[DEBUG L{layer_idx}] Key {full_global_key} not found in debug_no_ring.")
            sys.stdout.flush()
            all_match = False
            continue
        # --- END CHANGE ---

        # Construct the full key for the current rank's ring debug dictionary
        full_ring_key_current_rank = f"{current_rank_ring_key_prefix}{ring_key_suffix}"
        if full_ring_key_current_rank not in debug_ring_current_rank:
            print(f"[DEBUG L{layer_idx}] Key {full_ring_key_current_rank} not found in current rank's ({current_rank_for_comparison}) ring_debug_dict.")
            sys.stdout.flush()
            all_match = False
            continue

        # --- CHANGE IS ALSO HERE ---
        # Retrieve the tensor using the full key
        no_ring_tensor_global = debug_no_ring[full_global_key].to(torch.float32)
        # --- END CHANGE ---
        ring_tensor_local_current_rank = debug_ring_current_rank[full_ring_key_current_rank].to(torch.float32)

        # Use current rank's valid length for slicing and comparison
        # current_rank_valid_len is already defined above
        
        if current_rank_valid_len == 0: # Check for the current rank being compared
            if ring_tensor_local_current_rank.numel() == 0:
                print(f"[DEBUG L{layer_idx}] MATCH for {name} (both empty for rank 0 with 0 valid_len)")
                sys.stdout.flush()
                continue
            else:
                print(f"[DEBUG L{layer_idx}] MISMATCH for {name}: Rank 0 has 0 valid_len, but ring tensor is not empty: shape {ring_tensor_local_r0.shape}")
                sys.stdout.flush()
                all_match = False
                continue

        slicers = [slice(None)] * no_ring_tensor_global.ndim
        if global_seq_dim >= no_ring_tensor_global.ndim:
             print(f"[DEBUG L{layer_idx}] Invalid global_seq_dim {global_seq_dim} for tensor {name} with shape {no_ring_tensor_global.shape}")
             sys.stdout.flush() # type: ignore
             all_match = False
             continue
        # Slice the sequence dimension
        # Calculate start offset for the current rank
        current_rank_start_offset = strategy.rank * strategy.block_size # Assuming block_size is uniform for sharding
        slicers[global_seq_dim] = slice(current_rank_start_offset, current_rank_start_offset + current_rank_valid_len)
        try:
            no_ring_tensor_r0_slice = no_ring_tensor_global[tuple(slicers)]
        except IndexError as e:
            print(f"[DEBUG L{layer_idx}] IndexError when slicing {name} ({full_global_key}) with shape {no_ring_tensor_global.shape} using slicers {slicers} (current_rank_valid_len: {current_rank_valid_len}). Error: {e}")
            sys.stdout.flush()
            all_match = False
            continue
        
        # Ensure the sliced standard tensor shape matches the local ring tensor shape
        # Ring tensor might have padding in non-sequence dims, standard slice won't. This comparison might need adjustment.
        # For now, assume shapes should match after slicing seq dim.
        if ring_tensor_local_current_rank.shape != no_ring_tensor_r0_slice.shape:
             print(f"[DEBUG L{layer_idx}] SHAPE MISMATCH after slicing for {name}: Ring {ring_tensor_local_current_rank.shape} vs Sliced NoRing {no_ring_tensor_r0_slice.shape}")
             sys.stdout.flush()
             # Optionally try slicing ring tensor if it has padding:
             # ring_slicers = [slice(None)] * ring_tensor_local_current_rank.ndim
             # ring_slicers[1] = slice(0, rank0_valid_len) # Assuming seq dim is 1 for ring tensors B,T,E ? Adjust if needed.
             # ring_tensor_local_r0_sliced = ring_tensor_local_current_rank[tuple(ring_slicers)]
             # if ring_tensor_local_r0_sliced.shape == no_ring_tensor_r0_slice.shape: ... proceed with comparison ...
             all_match = False
             continue # Skip comparison if shapes don't match after slicing standard tensor

        if not _compare_tensors(name, ring_tensor_local_current_rank, no_ring_tensor_r0_slice, layer_idx, tolerance, print_values):
            all_match = False
            # Potentially break early if one mismatch is found? Or collect all mismatches.

    print(f"--- Comparison Result for Layer {layer_idx}, Rank {current_rank_for_comparison}: {'ALL MATCH' if all_match else 'MISMATCHES FOUND'} ---")
    sys.stdout.flush()

 
def _perform_shadow_standard_attention_pass(
    self_block, # LLaMABlock instance from Rank 0
    x_norm_global: torch.Tensor,
    residual_global: torch.Tensor,
    mask_global: Optional[torch.Tensor], # This is the original global mask
    position_ids_global: Optional[torch.Tensor], # Global position ids
    is_causal_mask_global: bool, # From original call
    attn_algorithm: Optional[str]
) -> dict:
    """Performs a standard LLaMABlock forward pass for debugging comparison."""
    print(f"[DEBUG L_shadow {self_block.layer_idx}] Performing standard shadow pass...")
    sys.stdout.flush()
    
    debug_data = {}
    # self_block is an instance of LLaMABlock
    # self_block.eval() is called before this function, so dropout is disabled.
    # torch.no_grad() is also active.

    # Store inputs
    debug_data["noring_x_norm_global"] = x_norm_global.detach().clone().cpu()
    debug_data["noring_residual_global_input"] = residual_global.detach().clone().cpu()

    # 1. Attention part
    # MultiHeadAttention.forward populates its debug_dict. We'll use a temporary one.
    attn_debug_dict_shadow = {}
    attn_key_prefix_shadow = "noring_shadow"

    # self_block.attn is MultiHeadAttention
    # x_norm_global is the input to MHA (q, k, v for self-attention)
    attn_out_dense = self_block.attn(
        q=x_norm_global,
        k=x_norm_global, # For self-attention
        v=x_norm_global, # For self-attention
        mask=mask_global,
        position_ids=position_ids_global,
        attn_algorithm=attn_algorithm,
        past_key_value_state=None,  # No cache in shadow pass
        use_cache=False,            # No cache in shadow pass
        is_self=True,
        is_causal_mask=is_causal_mask_global,
        debug_dict=attn_debug_dict_shadow,
        debug_key_prefix=attn_key_prefix_shadow
    )
    # Dropout after attention is handled by self_block.eval() mode

    # Store Q, K, V after RoPE (if any) and before SDP
    if f"{attn_key_prefix_shadow}_q_final" in attn_debug_dict_shadow:
        debug_data["noring_q_global"] = attn_debug_dict_shadow[f"{attn_key_prefix_shadow}_q_final"]
    if f"{attn_key_prefix_shadow}_k_final" in attn_debug_dict_shadow:
        debug_data["noring_k_global"] = attn_debug_dict_shadow[f"{attn_key_prefix_shadow}_k_final"]
    if f"{attn_key_prefix_shadow}_v_final" in attn_debug_dict_shadow:
        debug_data["noring_v_global"] = attn_debug_dict_shadow[f"{attn_key_prefix_shadow}_v_final"]
    
    debug_data["noring_attn_dense_out_global"] = attn_out_dense.detach().clone().cpu()

    # 2. First Residual Connection
    # residual_global is the original input `x` to the block
    x_after_attn_residual = attn_out_dense + residual_global
    debug_data["noring_residual1_global"] = x_after_attn_residual.detach().clone().cpu()

    # 3. FeedForward LayerNorm
    ff_ln_out = self_block.ff_ln(x_after_attn_residual)
    debug_data["noring_ff_ln_out_global"] = ff_ln_out.detach().clone().cpu()

    # 4. FeedForward SubLayer (GatedLinearUnit)
    # GatedLinearUnit.forward now accepts debug_dict and debug_key_prefix
    ff_debug_dict_shadow = {} # Not strictly needed by comparison_map, but good practice
    ff_key_prefix_shadow = "noring_shadow_ff"
    ff_out_raw = self_block.ff_sub_layer(
        ff_ln_out,
        debug_dict=ff_debug_dict_shadow,
        debug_key_prefix=ff_key_prefix_shadow
    )
    # Dropout after FFN is handled by self_block.eval() mode
    debug_data["noring_ff_out_global"] = ff_out_raw.detach().clone().cpu()

    # 5. Second Residual Connection (Final Block Output)
    block_output_final = ff_out_raw + x_after_attn_residual
    debug_data["noring_block_output_global"] = block_output_final.detach().clone().cpu()
    
    print(f"[DEBUG L_shadow {self_block.layer_idx}] Standard shadow pass placeholder completed.")
    sys.stdout.flush()
    return debug_data

# ============================================================

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
    # New debug parameters from LLaMABlock.forward call
    debug_dict_populate: Optional[dict] = None,
    debug_key_prefix_populate: str = "",
    debug_print_values: bool = False,
    debug_tolerance: float = 1e-3,
    layer_idx: int = -1, # Passed from LLaMABlock
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
        attn_algorithm=attn_algorithm, # Pass it down
        # Pass through the debug parameters
        debug_dict_populate=debug_dict_populate,
        debug_key_prefix_populate=debug_key_prefix_populate,
        debug_print_values=debug_print_values,
        debug_tolerance=debug_tolerance,
        layer_idx=layer_idx,
    )

    first_block_debug_out_ring = None
    # We need rank here. The `self` is LLaMABlock, strategy is passed.
    rank_for_ring = distributed_strategy.rank if isinstance(distributed_strategy, RingAttentionStrategy) else 0

    # This debug collection seems specific to layer 0 output, might be separate from comparison logic
    if rank_for_ring == 0 and layer_idx == 0: # Collect for layer 0, rank 0 portion
        # x_out is the sharded, padded output of the block for this rank
        # print(f"DEBUG ({debug_label}, Layer {layer_idx}, Rank {rank_for_ring}, Tensor: LLaMABlock_final_output_RING_Rank0_portion): norm = {torch.linalg.norm(x_out.float()).item()}")
        first_block_debug_out_ring = x_out.clone() # Collect sharded output

    if use_cache:
        # When use_cache is True, return the main output, the cache, and optionally debug info
        return x_out, cache_out #, first_block_debug_out_ring # Decide if debug info is part of cache tuple
    else:
        # When use_cache is False, only return the main tensor output
        return x_out # first_block_debug_out_ring can be handled differently if needed


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
    attn_algorithm: Optional[str], # Add parameter to receive it
    strategy: DistributedStrategy, # Expected RingAttentionStrategy
    # Add debug passthrough
    layer_idx: int = -1, # Default, will be overridden
    # New debug parameters from forward_ring
    debug_dict_populate: Optional[dict] = None,
    debug_key_prefix_populate: str = "",
    debug_print_values: bool = False,
    debug_tolerance: float = 1e-3,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Any]: # Returns (output, cache, extra)

    assert isinstance(strategy, RingAttentionStrategy), f"Expected RingAttentionStrategy, got {type(strategy)}"

    residual = x
    x_norm_local = self.ln(x)

    rank = strategy.rank # Get rank from strategy
    correct_valid_len = strategy.get_local_valid_len() # Valid tokens on this rank

    # Construct a debug label if needed for print statements, using layer_idx
    local_debug_label = f"L{layer_idx}_R{rank}"

    # --- Debug Start ---
    # Condition for this rank to participate in debug prints/orchestration
    target_debug_ranks_for_orchestration = [0, 1] # Ranks that will print/compare
    is_this_rank_a_debug_orchestration_target = (
        self.config.debug_mode and 
        layer_idx == self.config.debug_target_layer and 
        rank in target_debug_ranks_for_orchestration
    )
    if is_this_rank_a_debug_orchestration_target: # Broader condition for initial prints
        print(f"--- Entering _forward_ring_attention for Layer {layer_idx}, Rank {rank} (Debug Target) ---")
        print(f"  Input x shape: {x.shape}, Pos IDs shape: {position_ids.shape if position_ids is not None else 'None'}")
        print(f"  Local valid len: {correct_valid_len}")
        sys.stdout.flush()

    # Lazy init RingAttentionHelper on first call with this strategy
    if not hasattr(self, 'ring_helper') or self.ring_helper is None or self.ring_helper.strategy is not strategy:
        self.ring_helper = RingAttentionHelper(
            attn_module=self.attn,
            strategy=strategy,
            llama_block=self,
            use_cache=use_cache,
            ff=self.ff_sub_layer,
            ff_norm=self.ff_ln,
        )

    # Debug print for input to RingAttentionHelper.forward (Rank 0's portion)
    # Use self.config from LLaMABlock to check debug_mode and debug_target_layer
    if is_this_rank_a_debug_orchestration_target:
        print(f"DEBUG ({local_debug_label}) Input norm to RingHelper: {torch.linalg.norm(x_norm_local.float()).item():.3f}")
        sys.stdout.flush()

    # Call helper's core logic
    output, cache_from_helper, extra_output = self.ring_helper.forward(
        x_norm_local,
        mask=mask,
        strategy=strategy,
        position_ids=position_ids, # Pass sharded position_ids (as received by _forward_ring_attention)
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        valid_len=correct_valid_len,
        residual=residual,
        # Pass the debug parameters
        debug_dict_populate=debug_dict_populate,
        debug_key_prefix_populate=debug_key_prefix_populate,
        layer_idx=layer_idx, # Pass layer_idx 
        debug_print_values=debug_print_values,
        debug_tolerance=debug_tolerance,
    )
    
    # Debug print for output of RingAttentionHelper.forward (Rank 0's portion)
    # This 'output' is the block's final output for this rank, padded.
    if is_this_rank_a_debug_orchestration_target:
        print(f"DEBUG ({local_debug_label}) Output norm from RingHelper: {torch.linalg.norm(output.float()).item():.3f}")
        if debug_dict_populate:
            # debug_dict_populate now contains keys like "ring_r{rank}_x_norm"
            print(f"DEBUG ({local_debug_label}) Keys populated by RingHelper for this rank: {list(debug_dict_populate.keys())}")
        else:
            print(f"DEBUG ({local_debug_label}) debug_dict_populate was not provided or populated.")
        sys.stdout.flush()

    # --- Standard Shadow Pass (Rank 0 Only) and Comparison (Ranks 0, 1) ---
    if self.config.debug_mode and layer_idx == self.config.debug_target_layer:
        print(f"DEBUG ({local_debug_label}) Starting shadow standard attention pass orchestration...")
        sys.stdout.flush()

        debug_info_no_ring_container = [None] # Container for broadcast

        # All ranks must participate in gathering the sharded inputs
        world_size = strategy.world_size
        x_gathered_shards = [torch.empty_like(x) for _ in range(world_size)]
        print(f"DEBUG ({local_debug_label}, Rank {rank}) BEFORE all_gather for x_gathered_shards.")
        sys.stdout.flush()
        dist.all_gather(x_gathered_shards, x, group=strategy.group)
        print(f"DEBUG ({local_debug_label}, Rank {rank}) AFTER all_gather for x_gathered_shards.")
        sys.stdout.flush()

        pos_ids_gathered_shards_list = None # Use a list to handle None case for broadcast
        if position_ids is not None:
            pos_ids_gathered_shards = [torch.empty_like(position_ids) for _ in range(world_size)]
            print(f"DEBUG ({local_debug_label}, Rank {rank}) BEFORE all_gather for pos_ids_gathered_shards.")
            sys.stdout.flush()
            dist.all_gather(pos_ids_gathered_shards, position_ids, group=strategy.group)
            print(f"DEBUG ({local_debug_label}, Rank {rank}) AFTER all_gather for pos_ids_gathered_shards.")
            sys.stdout.flush()
            pos_ids_gathered_shards_list = pos_ids_gathered_shards # Store the list of tensors

        if rank == 0: # Rank 0 performs the shadow pass
            orig_seq_len = strategy._original_seq_len if hasattr(strategy, '_original_seq_len') else -1
            x_global_reconstructed = torch.cat(x_gathered_shards, dim=1)
            if orig_seq_len > 0 and x_global_reconstructed.size(1) > orig_seq_len:
                 x_global_reconstructed = x_global_reconstructed[:, :orig_seq_len, :]
            print(f"DEBUG ({local_debug_label}, Rank 0) Reconstructed global x shape: {x_global_reconstructed.shape}")
            
            pos_ids_global_reconstructed = None
            if pos_ids_gathered_shards_list is not None: # Rank 0 uses the gathered list
                pos_ids_global_reconstructed = torch.cat(pos_ids_gathered_shards_list, dim=1)
                if orig_seq_len > 0 and pos_ids_global_reconstructed.size(1) > orig_seq_len:
                     pos_ids_global_reconstructed = pos_ids_global_reconstructed[:, :orig_seq_len]
                print(f"DEBUG ({local_debug_label}, Rank 0) Reconstructed global pos_ids shape: {pos_ids_global_reconstructed.shape}")
            else:
                print(f"DEBUG ({local_debug_label}, Rank 0) position_ids was None, so pos_ids_global_reconstructed is None for shadow pass.")
            sys.stdout.flush()

            self.eval() 
            with torch.no_grad():
                x_norm_global_reconstructed = self.ln(x_global_reconstructed)
                # _perform_shadow_standard_attention_pass populates a dict with keys like "noring_x_norm_global"
                # The prefix "noring" is hardcoded there.
                debug_info_no_ring_container[0] = _perform_shadow_standard_attention_pass(
                    self, 
                    x_norm_global_reconstructed,
                    x_global_reconstructed, 
                    mask, 
                    pos_ids_global_reconstructed,
                    is_causal_mask, 
                    attn_algorithm
                )
            self.train()
        
        # Broadcast the shadow pass data from rank 0 to all other ranks
        if strategy.world_size > 1:
            print(f"DEBUG ({local_debug_label}, Rank {rank}) BEFORE broadcast_object_list for debug_info_no_ring_container.")
            sys.stdout.flush()
            dist.broadcast_object_list(debug_info_no_ring_container, src=0, group=strategy.group)
            print(f"DEBUG ({local_debug_label}, Rank {rank}) AFTER broadcast_object_list for debug_info_no_ring_container.")
            sys.stdout.flush()
            
        debug_info_no_ring_for_comparison = debug_info_no_ring_container[0]

        # Comparison on ranks 0 and 1 (or other target ranks)
        if rank in target_debug_ranks_for_orchestration:
            if debug_info_no_ring_for_comparison and debug_dict_populate:
                print(f"DEBUG ({local_debug_label}) Comparing Standard vs Ring data for Rank {rank}...")
                sys.stdout.flush()
                # Construct the prefix for the current rank's ring data keys
                # This prefix was used when populating debug_dict_populate via RingAttentionHelper
                current_rank_ring_key_prefix = f"ring_r{rank}"
                
                # The _compare_debug_data function needs to be aware of this prefix
                # to correctly look up keys like "ring_r{rank}_x_norm"
                # We will modify _compare_debug_data to accept this prefix.
                # For now, let's assume _compare_debug_data can handle it or we adjust keys in comparison_map.
                # The comparison_map's ring_key should be the base suffix (e.g., "_x_norm")
                # and _compare_debug_data will prepend current_rank_ring_key_prefix.
                _compare_debug_data(
                    debug_info_no_ring_for_comparison, 
                    debug_dict_populate, # This is the current rank's populated dict
                    strategy, 
                    layer_idx, 
                    debug_tolerance, 
                    debug_print_values,
                    current_rank_for_comparison=rank, # Pass current rank
                    current_rank_ring_key_prefix=current_rank_ring_key_prefix # Pass the prefix
                )
            else:
                print(f"DEBUG ({local_debug_label}) Skipping comparison for Rank {rank}: NoRing data: {bool(debug_info_no_ring_for_comparison)}, Ring data: {bool(debug_dict_populate)}")
            sys.stdout.flush()
            print(f"--- Finished Debug Orchestration for Layer {layer_idx}, Rank {rank} ---")
            sys.stdout.flush()

    # Return the actual output from the ring attention path
    # extra_output from helper is likely None now, dict is populated directly.
    return output, cache_from_helper, None
