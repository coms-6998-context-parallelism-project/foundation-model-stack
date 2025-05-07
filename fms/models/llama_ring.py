from typing import Any, List, Mapping, Optional, Tuple, Union, Dict # Keep necessary types

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F # For softmax in shadow pass
import sys # For flushing stdout
import math # For scale factor in shadow pass

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

    first_block_debug_out_ring = None
    # We need rank here. The `self` is LLaMABlock, strategy is passed.
    rank_for_ring = distributed_strategy.rank if isinstance(distributed_strategy, RingAttentionStrategy) else 0

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


def _compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, layer_idx: int, tolerance: float = 1e-3, print_values: bool = False):
    """Helper to compare two tensors and print debug info."""
    if t1.shape != t2.shape:
        print(f"[DEBUG L{layer_idx}] MISMATCH SHAPE for {name}: Ring {t1.shape} vs NoRing {t2.shape}")
        return False
    sys.stdout.flush()
    
    # Ensure same dtype for comparison, cast t2 to t1's dtype if different
    if t1.dtype != t2.dtype:
        # print(f"[DEBUG L{layer_idx}] WARN: Dtype mismatch for {name}: Ring {t1.dtype} vs NoRing {t2.dtype}. Casting NoRing to Ring's dtype for comparison.")
        try:
            t2_casted = t2.to(t1.dtype)
        except Exception as e:
            print(f"[DEBUG L{layer_idx}] ERROR: Failed to cast NoRing tensor {name} from {t2.dtype} to {t1.dtype}: {e}")
            sys.stdout.flush()
            return False
        sys.stdout.flush()
    else:
        t2_casted = t2

    if not torch.allclose(t1, t2_casted, atol=tolerance, rtol=tolerance):
        diff = torch.abs(t1 - t2_casted)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[DEBUG L{layer_idx}] MISMATCH VAL for {name} (MaxDiff: {max_diff:.2e}, MeanDiff: {mean_diff:.2e}, RingNorm: {torch.linalg.norm(t1.float()).item():.2e}, NoRingNorm: {torch.linalg.norm(t2_casted.float()).item():.2e})")
        sys.stdout.flush()
        if print_values:
            print(f"  Ring tensor ({name}):\n{t1.flatten()[:5]}...\n{t1.flatten()[-5:]}")
            sys.stdout.flush()
            print(f"  NoRing tensor ({name}):\n{t2_casted.flatten()[:5]}...\n{t2_casted.flatten()[-5:]}")
            sys.stdout.flush()
            print(f"  Difference tensor ({name}):\n{diff.flatten()[:5]}...\n{diff.flatten()[-5:]}")
            sys.stdout.flush()
        return False
    else:
        # print(f"[DEBUG L{layer_idx}] MATCH for {name} (RingNorm: {torch.linalg.norm(t1.float()).item():.2e}, NoRingNorm: {torch.linalg.norm(t2_casted.float()).item():.2e})")
        # sys.stdout.flush() # Optional: flush after match print if uncommented
        return True

def _compare_debug_data(
    debug_no_ring: Dict[str, torch.Tensor], 
    debug_ring_rank0: Dict[str, torch.Tensor], 
    strategy: RingAttentionStrategy, 
    layer_idx: int
):
    print(f"\n--- Comparing Debug Data for Layer {layer_idx}, Rank 0 ---")
    sys.stdout.flush()
    if not debug_no_ring or not debug_ring_rank0:
        print("[DEBUG] One of the debug dictionaries is empty. Skipping comparison.")
        sys.stdout.flush()
        return

    # Determine Rank 0's valid slice from the global tensor
    # rank0_start = 0 # Rank is 0
    # rank0_valid_len = strategy.get_local_valid_len() # This is for rank 0
    # rank0_slice = slice(0, rank0_valid_len)

    # For ring attention, the local tensors (q_local, k_local, v_local, block_output etc.)
    # correspond to a specific shard of the global sequence.
    # The `valid_len` for rank 0 is `strategy.get_local_valid_len()`.
    # The global tensors from `debug_no_ring` need to be sliced to compare with rank 0's part.
    
    # Assuming _original_seq_len is available and correctly set by shard_input
    # For rank 0, the slice is always from the beginning of the sequence up to its valid length.
    rank0_valid_len = strategy.get_local_valid_len()
    if rank0_valid_len == 0 and strategy._original_seq_len > 0 : # If rank 0 has no valid tokens but there is a sequence
        print(f"[DEBUG L{layer_idx}] Rank 0 has 0 valid tokens. Comparison might be trivial or misleading.")
        sys.stdout.flush()

    # Define mappings from no_ring keys (global) to ring keys (rank0 local)
    # and the dimension along which to slice the global tensor (typically sequence dim)
    # (Tensor Name, Global Key, Ring Key, Sequence Dimension in Global Tensor)
    comparison_map = [
        ("X_norm_local", "noring_x_norm_global", "ring_x_norm_r0", 1), # (B, T, E) -> seq_dim=1
        ("Q_local", "noring_q_global", "ring_q_local_r0", 2),       # (B, H, T, D) -> seq_dim=2
        ("K_local", "noring_k_global", "ring_k_local_r0", 2),
        ("V_local", "noring_v_global", "ring_v_local_r0", 2),
        ("Attn_Dense_Out", "noring_attn_dense_out_global", "ring_attn_out_raw_r0", 1), # (B, T, E)
        ("Residual1", "noring_residual1_global", "ring_attn_out_residual_r0", 1),
        ("FF_LN_Out", "noring_ff_ln_out_global", "ring_ff_ln_out_r0", 1),
        ("FF_Out", "noring_ff_out_global", "ring_ff_out_raw_r0", 1),
        ("Block_Output", "noring_block_output_global", "ring_block_output_r0", 1),
    ]

    for name, global_key, ring_key, global_seq_dim in comparison_map:
        if global_key not in debug_no_ring:
            print(f"[DEBUG L{layer_idx}] Key {global_key} not found in debug_no_ring.")
            sys.stdout.flush()
            continue
        if ring_key not in debug_ring_rank0:
            print(f"[DEBUG L{layer_idx}] Key {ring_key} not found in debug_ring_rank0.")
            sys.stdout.flush()
            continue

        no_ring_tensor_global = debug_no_ring[global_key].to(torch.float32) # Promote for stable comparison
        ring_tensor_local_r0 = debug_ring_rank0[ring_key].to(torch.float32)

        if rank0_valid_len == 0: # If rank 0 has no valid tokens
            if ring_tensor_local_r0.numel() == 0: # And ring tensor is empty
                print(f"[DEBUG L{layer_idx}] MATCH for {name} (both empty for rank 0 with 0 valid_len)")
                sys.stdout.flush()
                continue
            else: # Ring tensor not empty, but should be
                print(f"[DEBUG L{layer_idx}] MISMATCH for {name}: Rank 0 has 0 valid_len, but ring tensor is not empty: shape {ring_tensor_local_r0.shape}")
                sys.stdout.flush()
                continue

        # Slice the global tensor to get rank 0's expected part
        # Construct slice object: [:, :, ..., slice(0, rank0_valid_len), ...]
        # The slice should apply to the sequence dimension `global_seq_dim`
        slicers = [slice(None)] * no_ring_tensor_global.ndim
        slicers[global_seq_dim] = slice(0, rank0_valid_len)
        no_ring_tensor_r0_slice = no_ring_tensor_global[tuple(slicers)]

        _compare_tensors(name, ring_tensor_local_r0, no_ring_tensor_r0_slice, layer_idx, print_values=True)
    
    print(f"--- End Comparison for Layer {layer_idx}, Rank 0 ---\n")
    sys.stdout.flush()


def _perform_shadow_standard_attention_pass(
    self_block, 
    x_norm_global: torch.Tensor,
    residual_global: torch.Tensor,
    mask_global: Optional[torch.Tensor], # This is the original global mask
    position_ids_global: Optional[torch.Tensor],
    # use_cache: bool, # For debug, typically False
    is_causal_mask_global: bool, # From original call
    attn_algorithm: Optional[str]
) -> Dict[str, torch.Tensor]:
    # This function simulates LLaMABlock.forward's standard path
    # It needs to be careful with device placement if x_norm_global etc. are on CPU
    # but self_block's parameters are on GPU. For now, assume inputs are on correct device.
    print(f"[DEBUG L_shadow {self_block.config.model_variant} L{self_block.layer_idx if hasattr(self_block, 'layer_idx') else 'Unknown'}] Entered _perform_shadow_standard_attention_pass. Input x_norm_global shape: {x_norm_global.shape}, device: {x_norm_global.device}")
    sys.stdout.flush()

    debug_data = {}
    config = self_block.config # LLaMAConfig
    current_device = x_norm_global.device # Assume inputs are on the target device

    # Store initial inputs
    debug_data["noring_x_norm_global"] = x_norm_global.detach().clone().cpu()
    debug_data["noring_residual_global_input"] = residual_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Stored initial inputs to debug_data.")
    sys.stdout.flush()

    # 1. Attention part (mimicking MultiHeadAttention internals)
    # 1a. Input projection
    q_proj, k_proj, v_proj = self_block.attn.in_proj(x_norm_global, x_norm_global, x_norm_global)
    print(f"[DEBUG L_shadow] Computed q_proj, k_proj, v_proj. q_proj shape: {q_proj.shape}")
    sys.stdout.flush()

    B, T, E = q_proj.shape # T is global sequence length here
    
    # 1b. Reshape for multi-head and apply RoPE
    q_reshaped = q_proj.view(B, T, self_block.attn.nheads, self_block.attn.emb_kq_per_head)
    k_reshaped = k_proj.view(B, T, self_block.attn.kvheads, self_block.attn.emb_kq_per_head)
    v_reshaped = v_proj.view(B, T, self_block.attn.kvheads, self_block.attn.emb_v_per_head)
    print(f"[DEBUG L_shadow] Reshaped QKV. q_reshaped shape: {q_reshaped.shape}")
    sys.stdout.flush()

    if self_block.attn.position_encoder and T > 0:
        # Ensure position_ids_global is correctly shaped (B, T) if not None
        q_rope, k_rope = self_block.attn.position_encoder.adjusted_qk(q_reshaped, k_reshaped, position_ids_global)
        print(f"[DEBUG L_shadow] Applied RoPE. q_rope shape: {q_rope.shape}")
        sys.stdout.flush()
    else:
        print(f"[DEBUG L_shadow] No RoPE applied (T=0 or no position_encoder).")
        sys.stdout.flush()
        q_rope, k_rope = q_reshaped, k_reshaped
    v_rope = v_reshaped # V is not typically rotated

    # Permute for matmul: (B, H, T, D)
    q_final = q_rope.permute(0, 2, 1, 3)
    k_final = k_rope.permute(0, 2, 1, 3)
    v_final = v_rope.permute(0, 2, 1, 3)

    debug_data["noring_q_global"] = q_final.detach().clone().cpu()
    debug_data["noring_k_global"] = k_final.detach().clone().cpu()
    debug_data["noring_v_global"] = v_final.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Stored q_final, k_final, v_final to debug_data.")
    sys.stdout.flush()

    # Expand K, V for GQA/MQA
    if self_block.attn.nheads != self_block.attn.kvheads:
        expansion_factor = self_block.attn.nheads // self_block.attn.kvheads
        k_final_expanded = k_final.unsqueeze(2).expand(-1, -1, expansion_factor, -1, -1).reshape(B, self_block.attn.nheads, T, self_block.attn.emb_kq_per_head)
        v_final_expanded = v_final.unsqueeze(2).expand(-1, -1, expansion_factor, -1, -1).reshape(B, self_block.attn.nheads, T, self_block.attn.emb_v_per_head)
        print(f"[DEBUG L_shadow] Expanded K, V for GQA/MQA.")
        sys.stdout.flush()
    else:
        k_final_expanded = k_final
        v_final_expanded = v_final
        print(f"[DEBUG L_shadow] No K, V expansion needed (MHA).")
        sys.stdout.flush()

    # 1c. Compute scores
    scale = self_block.attn.scale_factor or math.sqrt(self_block.attn.emb_kq_per_head)
    # scores_global: (B, H, T_q, T_k) where T_q = T_k = T (global_seq_len)
    scores_global = torch.matmul(q_final / scale, k_final_expanded.transpose(-2, -1))
    debug_data["noring_raw_scores_global"] = scores_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed raw scores. scores_global shape: {scores_global.shape}")
    sys.stdout.flush()

    # Apply masks
    # The mask_global passed to _forward_ring_attention is already the one LLaMA._helper prepares.
    # It's typically (B, 1, T_q_global, T_kv_global).
    # If is_causal_mask_global is True and mask_global is None, MHA would create a causal mask.
    # Here, we assume mask_global is either the correct combined mask or None (then is_causal_mask_global applies).
    
    effective_mask = mask_global
    if is_causal_mask_global:
        if effective_mask is None: # Create a causal mask if none provided
            # This is a simplified causal mask for self-attention.
            # LLaMA's MHA has more sophisticated mask handling for combined padding/causal.
            causal_m = torch.triu(torch.ones(T, T, device=current_device, dtype=torch.bool), diagonal=1)
            # Expand to (B, H, T, T)
            causal_m = causal_m.unsqueeze(0).unsqueeze(0).expand(B, self_block.attn.nheads, T, T)
            # Add this to scores_global (masked_fill with -inf)
            scores_global = scores_global.masked_fill(causal_m, float('-inf'))
        # If mask_global is already provided and is_causal_mask_global is true,
        # it implies mask_global should already be causal or combined.
        # For simplicity, we'll assume if mask_global is not None, it's the one to use.
        # The MHA module itself has logic for this:
        # if mask is not None: scores = scores + mask
        # if is_causal_mask and (mask is None or not getattr(mask, "is_causal", False)):
        #    scores = scores.masked_fill(causal_mask_matrix, float("-inf"))
        # We'll simplify here: if mask_global is present, use it. If not, and is_causal_mask_global, apply simple causal.

    if effective_mask is not None:
         scores_global = scores_global + effective_mask.to(scores_global.dtype)
    print(f"[DEBUG L_shadow] Applied mask to scores. scores_global shape after mask: {scores_global.shape}")
    sys.stdout.flush()

    debug_data["noring_masked_scores_global"] = scores_global.detach().clone().cpu()
    
    # 1d. Softmax
    attn_weights_global = F.softmax(scores_global, dim=-1)
    debug_data["noring_attn_weights_global"] = attn_weights_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed softmax (attn_weights_global).")
    sys.stdout.flush()

    # 1e. Weighted sum with V
    attn_output_global = torch.matmul(attn_weights_global, v_final_expanded) # (B, H, T, Dv)
    debug_data["noring_attn_output_global"] = attn_output_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed weighted sum with V (attn_output_global).")
    sys.stdout.flush()

    # 1f. Output projection
    attn_output_global_reshaped = attn_output_global.transpose(1, 2).contiguous().view(B, T, E)
    attn_dense_out_global = self_block.attn.dense(attn_output_global_reshaped)
    debug_data["noring_attn_dense_out_global"] = attn_dense_out_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed output projection (attn_dense_out_global).")
    sys.stdout.flush()

    if config.p_dropout != 0: # Apply dropout if configured
        attn_dense_out_global = self_block.dropout(attn_dense_out_global)
        print(f"[DEBUG L_shadow] Applied dropout to attention output.")
        sys.stdout.flush()
    
    # 1g. First residual connection
    residual1_global = residual_global + attn_dense_out_global
    debug_data["noring_residual1_global"] = residual1_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed first residual (residual1_global).")
    sys.stdout.flush()

    # 2. Feedforward part
    # 2a. LayerNorm
    ff_ln_out_global = self_block.ff_ln(residual1_global)
    debug_data["noring_ff_ln_out_global"] = ff_ln_out_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed FF LayerNorm output (ff_ln_out_global).")
    sys.stdout.flush()

    # 2b. MLP
    ff_out_global = self_block.ff_sub_layer(ff_ln_out_global)
    debug_data["noring_ff_out_global"] = ff_out_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed MLP output (ff_out_global).")
    sys.stdout.flush()

    if config.p_dropout != 0: # Apply dropout if configured
        ff_out_global = self_block.dropout(ff_out_global)
        print(f"[DEBUG L_shadow] Applied dropout to MLP output.")
        sys.stdout.flush()
    
    # 2c. Second residual connection
    block_output_global = residual1_global + ff_out_global
    debug_data["noring_block_output_global"] = block_output_global.detach().clone().cpu()
    print(f"[DEBUG L_shadow] Computed second residual (block_output_global).")
    sys.stdout.flush()

    print(f"[DEBUG L_shadow] Exiting _perform_shadow_standard_attention_pass successfully.")
    sys.stdout.flush()
    return debug_data


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
    # `self` here is the LLaMABlock instance.
    assert isinstance(strategy, RingAttentionStrategy), f"Expected RingAttentionStrategy, got {type(strategy)}"

    residual = x
    x_norm_local = self.ln(x)

    rank = strategy.rank # Get rank from strategy
    correct_valid_len = strategy.get_local_valid_len() # Valid tokens on this rank

    # Lazy init RingAttentionHelper on first call with this strategy
    # Also re-initialize if debug_mode has changed in the config
    current_debug_mode = self.config.debug_mode
    helper_needs_reinit = self.ring_helper is None or self.ring_helper.strategy is not strategy or \
                          (hasattr(self.ring_helper, 'debug_mode') and self.ring_helper.debug_mode != current_debug_mode)
    if helper_needs_reinit:
        self.ring_helper = RingAttentionHelper(
            attn_module=self.attn,
            strategy=strategy,
            llama_block=self,
            use_cache=use_cache,
            ff=self.ff_sub_layer,
            ff_norm=self.ff_ln,
        )
        # Explicitly set debug_mode on the helper if it's part of its __init__ or as an attribute
        self.ring_helper.debug_mode = current_debug_mode 

    # Debug print for input to RingAttentionHelper.forward (Rank 0's portion)
    if rank == 0 and layer_idx == 0 and self.config.debug_mode:
        print(f"DEBUG ({debug_label}, Layer {layer_idx}, Rank {rank}, Tensor: input_to_RingHelper_x_norm_local_Rank0): norm = {torch.linalg.norm(x_norm_local.float()).item()}")
        pass

    # Call helper's core logic
    output, cache_from_helper, extra_output = self.ring_helper.forward(
        x_norm_local,
        mask=mask,
        strategy=strategy,
        position_ids=position_ids, # Pass sharded position_ids
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        valid_len=correct_valid_len,
        residual=residual,
    )
    # output is the block's final output for this rank, padded.
    # cache_from_helper is None for now.
    # extra_output is debug_info_ring from the helper if debug_mode was on for helper.
    debug_info_ring_rank0 = extra_output if self.config.debug_mode and rank == 0 else None
    
    if rank == 0 and self.config.debug_mode and layer_idx == 0: # Limiting debug to layer 0 for now
        print(f"[DEBUG L{layer_idx}] Rank 0: Starting shadow standard attention pass.")
        sys.stdout.flush()

        # 1. Gather inputs
        # `x` is the sharded input to the block. `position_ids` is also sharded.
        # `mask` is global.
        world_size = strategy.world_size
        print(f"[DEBUG L{layer_idx}] Rank 0: Before all_gather for x_global_shards. World size: {world_size}, x.shape: {x.shape}, x.device: {x.device}, x dtype: {x.dtype}")
        sys.stdout.flush()

        # Gather `x` (original input to block)
        x_global_shards = [torch.empty_like(x, device=x.device) for _ in range(world_size)]
        dist.all_gather(x_global_shards, x, group=strategy.group)
        print(f"[DEBUG L{layer_idx}] Rank 0: After all_gather for x_global_shards.")
        sys.stdout.flush()

        # Concatenate along sequence dim (dim=1 for (B,T,E)) and slice to original length
        x_global_reconstructed = torch.cat(x_global_shards, dim=1)[:, :strategy._original_seq_len, :]
        print(f"[DEBUG L{layer_idx}] Rank 0: Reconstructed x_global. Shape: {x_global_reconstructed.shape}, Device: {x_global_reconstructed.device}, Dtype: {x_global_reconstructed.dtype}")
        sys.stdout.flush()

        # Gather `position_ids`
        print(f"[DEBUG L{layer_idx}] Rank 0: Before all_gather for pos_ids_global_shards. position_ids.shape: {position_ids.shape}, position_ids.device: {position_ids.device}, pos_ids dtype: {position_ids.dtype}")
        sys.stdout.flush()
        pos_ids_global_shards = [torch.empty_like(position_ids, device=position_ids.device) for _ in range(world_size)]
        dist.all_gather(pos_ids_global_shards, position_ids, group=strategy.group)
        print(f"[DEBUG L{layer_idx}] Rank 0: After all_gather for pos_ids_global_shards.")
        sys.stdout.flush()

        position_ids_global_reconstructed = torch.cat(pos_ids_global_shards, dim=1)[:, :strategy._original_seq_len]
        print(f"[DEBUG L{layer_idx}] Rank 0: Reconstructed position_ids_global. Shape: {position_ids_global_reconstructed.shape}, Device: {position_ids_global_reconstructed.device}, Dtype: {position_ids_global_reconstructed.dtype}")
        sys.stdout.flush()

        # Compute global norm and residual for the shadow pass
        # Ensure they are on the same device as the LLaMABlock's parameters (self.ln.weight.device)
        target_device = self.ln.weight.device
        print(f"[DEBUG L{layer_idx}] Rank 0: Target device for shadow pass: {target_device}.")
        sys.stdout.flush()

        x_global_reconstructed_dev = x_global_reconstructed.to(target_device)
        position_ids_global_reconstructed_dev = position_ids_global_reconstructed.to(target_device)
        mask_dev = mask.to(target_device) if mask is not None else None
        print(f"[DEBUG L{layer_idx}] Rank 0: Inputs moved to target device for shadow pass. x_global_reconstructed_dev.device: {x_global_reconstructed_dev.device}, mask_dev is {'None' if mask_dev is None else 'Present'}")
        sys.stdout.flush()

        x_norm_global_reconstructed = self.ln(x_global_reconstructed_dev)
        residual_global_reconstructed = x_global_reconstructed_dev # This is the input to the block
        print(f"[DEBUG L{layer_idx}] Rank 0: Computed x_norm_global (shape: {x_norm_global_reconstructed.shape}, device: {x_norm_global_reconstructed.device}) and residual_global (shape: {residual_global_reconstructed.shape}, device: {residual_global_reconstructed.device}) for shadow pass.")
        sys.stdout.flush()

        # 2. Perform shadow standard attention pass
        print(f"[DEBUG L{layer_idx}] Rank 0: Calling _perform_shadow_standard_attention_pass.")
        sys.stdout.flush()
        debug_info_no_ring = _perform_shadow_standard_attention_pass(
            self, # LLaMABlock instance
            x_norm_global_reconstructed,
            residual_global_reconstructed,
            mask_dev, # Global mask
            position_ids_global_reconstructed_dev, # Global position_ids
            is_causal_mask, # from original call
            attn_algorithm
        )
        print(f"[DEBUG L{layer_idx}] Rank 0: Returned from _perform_shadow_standard_attention_pass.")
        sys.stdout.flush()

        # 3. Compare
        if debug_info_no_ring and debug_info_ring_rank0:
            print(f"[DEBUG L{layer_idx}] Rank 0: Both debug_info_no_ring and debug_info_ring_rank0 are populated. Calling _compare_debug_data.")
            sys.stdout.flush()
            _compare_debug_data(debug_info_no_ring, debug_info_ring_rank0, strategy, layer_idx)
        else:
            print(f"[DEBUG L{layer_idx}] Skipping comparison: debug_info_no_ring is {type(debug_info_no_ring)}, debug_info_ring_rank0 is {type(debug_info_ring_rank0)}")
            sys.stdout.flush()

    # Return the actual output from the ring attention path
    return output, cache_from_helper, debug_info_ring_rank0 # Return rank0's ring debug info
