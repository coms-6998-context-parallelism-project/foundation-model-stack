import logging
import re
from dataclasses import dataclass
import time
from typing import Any, List, Mapping, Optional, Tuple

from fms.models.ring_attention_helper import RingAttentionHelper
from fms.models.threaded_ring_attention import ThreadedRingAttentionEngine
import torch
import torch.nn as nn
import torch.distributed as dist # Import dist for rank info

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    RingAttentionStrategy,
    TensorParallelStrategy,
)
from fms.modules.attention import MultiHeadAttention
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


# params emb_dim heads layers lr
#  7B    4096    32    32     3.0E-04
# 13B    5120    40    40     3.0E-04
# 33B    6656    52    60     1.5.E-04
# 65B    8192    64    80     1.5.E-04


@dataclass
class LLaMAConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    emb_dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 0
    nlayers: int = 32
    pad_id: int = -1
    hidden_grow_factor: float = 8 / 3
    multiple_of: int = 256
    activation_fn: str = "swish"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 4096
    ntk_scaling: bool = False
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = False
    rope_theta: float = 10_000.0
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True

    


class LLaMABlock(nn.Module):
    def __init__(self, config: LLaMAConfig, rotary_emb: RotaryEmbedding, layer_index: int):
        super(LLaMABlock, self).__init__()
        self.layer_index = layer_index
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

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

        queries = q_out.view(B, T, attn_data.nheads, attn_data.emb_kq_per_head)
        keys = k_out.view(B, T, attn_data.kvheads, attn_data.emb_kq_per_head)
        values = v_out.view(B, T, attn_data.kvheads, attn_data.emb_v_per_head)

        if attn_data.position_encoder is not None:
            if position_ids is None:
                position_ids = torch.arange(T, device=q.device).unsqueeze(0).expand(B, -1)
            queries, keys = attn_data.position_encoder.adjusted_qk(queries, keys, position_ids, past_key_value_state, use_cache)

        return queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
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
        distributed_strategy: Optional[DistributedStrategy] = None,
    ):
        
        print(self.layer_index, end = ", ")
        enable_debug_info = True  # Set to True to collect debug info
        minimal_debug_prints= True
        debug_info = {} # This dict will still be populated for comparisons
        x_original = x # Store the original input for debug comparison

        if isinstance(distributed_strategy, RingAttentionStrategy):
            # --- RING ATTENTION PATH ---
            rank = dist.get_rank() 
            output_ring = self._forward_ring_attention(
                x,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                strategy=distributed_strategy,
                enable_debug_info=enable_debug_info,
                rank = rank,
                            minimal_debug_prints = minimal_debug_prints,
            )

            if use_cache:
                x, cache, debug_ring = output_ring
            else:
                x, _, debug_ring = output_ring
                cache = None

            # No nested dict expected — flatten directly
            if enable_debug_info and debug_ring:
                for k, v in debug_ring.items(): # Keys already have rank suffix from _forward_ring_attention
                    debug_info[k] = v # Directly assign without adding another suffix

            # --- ENGINE PATH FOR COMPARISON ---
            if enable_debug_info:
                # Gather the ORIGINAL block input for the engine comparison path
                # Pad x_original to the block size before gathering, similar to how
                # RingAttentionStrategy pads tensors before communication.
                padded_x_original = x_original
                target_len = distributed_strategy.block_size
                current_len = x_original.shape[1]
                pad_len = target_len - current_len
                if pad_len > 0:
                    pad_shape = list(x_original.shape)
                    pad_shape[1] = pad_len
                    pad = torch.zeros(*pad_shape, dtype=x_original.dtype, device=x_original.device)
                    padded_x_original = torch.cat([x_original, pad], dim=1)
                x_gathered_original = distributed_strategy.gather_output(padded_x_original)
                rank = dist.get_rank() if dist.is_initialized() else 0
                output_engine_gathered = self._forward_engine_attention(
                    x_gathered_original, # Pass the gathered original input
                    mask=mask,
                    position_ids=position_ids,
                    past_key_value_state=past_key_value_state,
                    use_cache=False,
                    is_causal_mask=is_causal_mask,
                    enable_debug_info=True,
                    minimal_debug_prints=minimal_debug_prints, # Pass flag
                )
                # Log the gathered input that was actually passed (optional, but good for clarity)
                # Note: _forward_engine_attention already logs this internally as x_input_r{rank}
                # debug_info[f"engine_x_gathered_input_r{rank}"] = x_gathered_original.clone().detach().cpu()

                _, _, debug_engine = output_engine_gathered

                if debug_engine:
                    for k, v in debug_engine.items(): # Keys already have prefix and rank suffix from _forward_engine_attention
                        debug_info[k] = v # Directly assign

                # --- DEBUG DIFF ---
                diffs = self._diff_debug_dicts(
                    {k: v for k, v in debug_info.items() if k.startswith("ring")},
                    {k: v for k, v in debug_info.items() if k.startswith("engine")},
                    minimal_debug_prints=minimal_debug_prints, # Pass flag as argument
                )
                rank = dist.get_rank() if dist.is_initialized() else 0
                if not minimal_debug_prints: # Guard the diff print header
                    print(f"\n--- [Rank {rank}] Debug Info Diffs in LLaMABlock {self.layer_index} ---")
                for key, formatted_string in diffs.items():
                    print(formatted_string)
                # print(f"--- Exiting after debug diff in Rank {rank}, Layer {self.layer_index} ---") # Commented out to prevent early exit crash

                # if dist.is_initialized():
                #     dist.barrier() # Commented out barrier as well
                # time.sleep(3)

        else:
            # --- ENGINE-ONLY PATH ---
            output_engine = self._forward_engine_attention(
                x,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                enable_debug_info=enable_debug_info,
                minimal_debug_prints=minimal_debug_prints, # Pass flag
            )

            if use_cache:
                x, cache, debug_engine = output_engine
            else:
                x, _, debug_engine = output_engine
                cache = None

            if enable_debug_info and debug_engine:
                for k, v in debug_engine.items(): # Keys already have prefix and rank suffix from _forward_engine_attention
                    debug_info[k] = v # Directly assign

        return (x, cache) if use_cache else x



    def _diff_debug_dicts(self, d1, d2, minimal_debug_prints=False): # Add flag to signature
        diffs = {}


        def normalize(key, prefix, is_engine=False):
            # Strip prefix
            base = key[len(prefix):]
            # Engine keys use _r{block_id}, Ring keys use _r{rank}
            # For comparison, we treat block_id as equivalent to rank
            # No need to collapse repeated suffixes as that was fixed.
            return base

        # Build reverse maps
        ring_map = {}
        for k in d1:
             if k.startswith("ring_"):
                 ring_map[normalize(k, "ring_")] = k

        engine_map = {}
        for k in d2:
            if k.startswith("engine_"):
                 engine_map[normalize(k, "engine_", is_engine=True)] = k

        shared = ring_map.keys() & engine_map.keys()
        missing_ring = sorted(set(engine_map.keys()) - set(ring_map.keys()))
        missing_engine = sorted(set(ring_map.keys()) - set(engine_map.keys()))

        # Only print these if not minimal
        # if not minimal_debug_prints:
        #     print("engine keys: ", engine_map.keys())
        #     print("ring keys:   ", ring_map.keys())

        #     if missing_ring:
        #         print("Missing in Ring:")
        #         for k in missing_ring:
        #             print(f"  {engine_map[k]}")

        #     if missing_engine:
        #         print("Missing in Engine:")
        #         for k in missing_engine:
        #             print(f"  {ring_map[k]}")

        # --- Add Summary of Value Matches ---
        matching_keys = []
        non_matching_keys = []
        comparison_failed_keys = []

        for suffix in sorted(shared):
            rk, ek = ring_map[suffix], engine_map[suffix]
            val1, val2 = d1[rk], d2[ek]

            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                # Determine shorter tensor length for comparison
                n_compare = min(val1.numel(), val2.numel())
                if n_compare > 0:
                    v1_flat = val1.flatten()[:n_compare]
                    v2_flat = val2.flatten()[:n_compare]
                    try:
                        # Use torch.allclose on the shorter length with relative tolerance (1%)
                        # Added atol for stability near zero
                        # Ensure tensors are on the same device (CPU) and same dtype for comparison
                        v1_flat = v1_flat.cpu().float()
                        v2_flat = v2_flat.cpu().float()
                        if torch.allclose(v1_flat.float(), v2_flat.float(), rtol=0.01, atol=1e-5):
                            matching_keys.append(suffix)
                        else:
                            non_matching_keys.append(suffix)
                    except Exception as e:
                        comparison_failed_keys.append(f"{suffix} (Error: {e})")
                else:
                     # If tensors are empty or only one is, consider non-matching
                     non_matching_keys.append(suffix)
            else:
                # Non-tensor types or mismatched types are considered non-matching for this check
                non_matching_keys.append(suffix)

        # --- End Summary ---

        # Always print the summary
        summary_str = "\n--- Value Match Summary (First 5 Vals, 1% Tolerance) ---\n"
        summary_str += f"Matching Keys ({len(matching_keys)}): {sorted(matching_keys)}\n"
        summary_str += f"Non-Matching Keys ({len(non_matching_keys)}): {sorted(non_matching_keys)}\n"
        if comparison_failed_keys:
            summary_str += f"Comparison Failed Keys ({len(comparison_failed_keys)}): {sorted(comparison_failed_keys)}\n"
        summary_str += "--------------------------------------------------------\n"
        print(summary_str) # Print summary before detailed diffs

        # If minimal, print norm diffs for non-matching keys
        if minimal_debug_prints:
            print("--- Norm Diffs for Non-Matching Keys (First 5 Vals) ---")
            for suffix in sorted(non_matching_keys):
                if suffix not in shared: continue # Skip if key wasn't actually shared (e.g., comparison failed)
                rk, ek = ring_map[suffix], engine_map[suffix]
                val1, val2 = d1[rk], d2[ek]
                if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                    n_compare = min(val1.numel(), val2.numel())
                    if n_compare > 0:
                        # Calculate norm diff over the shorter length
                        v1_flat = val1.flatten()[:n_compare].cpu().float()
                        v2_flat = val2.flatten()[:n_compare].cpu().float()
                        norm_diff = torch.linalg.norm(v1_flat - v2_flat).item()
                        print(f"  {suffix}: {norm_diff:.6f}")
                    else:
                        # Handle case where one or both tensors are empty
                        print(f"  {suffix}: N/A (Empty Tensor)")
                else:
                    print(f"  {suffix}: N/A (Non-Tensor or Type Mismatch)")
            print("--------------------------------------------------------")
        # Otherwise, print the full diffs
        else:
            # Indent this whole block
             for suffix in sorted(shared):
                 rk, ek = ring_map[suffix], engine_map[suffix]
                 val1, val2 = d1[rk], d2[ek]

                 diff_lines = [f"\n--- Key Suffix: {suffix} ---",
                             f"                    {'Shape':<25} {'Dtype':<15} {'First 5 Vals'}"] # Revert header

                 if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                     # Print first 5 values
                     n_print = min(val1.numel(), val2.numel(), 5)
                     v1_flat = val1.flatten()[:n_print].cpu().float()
                     v2_flat = val2.flatten()[:n_print].cpu().float()
                     v1 = [f"{v:.4f}" for v in v1_flat.tolist()]
                     v2 = [f"{v:.4f}" for v in v2_flat.tolist()]
                     diff_lines.append(f"Ring:   {str(val1.shape):<25} {str(val1.dtype):<15} {v1}")
                     diff_lines.append(f"Engine: {str(val2.shape):<25} {str(val2.dtype):<15} {v2}")

                     # Add overall diff stats for non-matching tensors
                     if suffix in non_matching_keys:
                         n_compare = min(val1.numel(), val2.numel())
                         if n_compare > 0:
                             v1_flat_comp = val1.flatten()[:n_compare].cpu().float()
                             v2_flat_comp = val2.flatten()[:n_compare].cpu().float()
                             norm_diff = torch.linalg.norm(v1_flat_comp - v2_flat_comp).item()
                             abs_diff = torch.abs(v1_flat_comp - v2_flat_comp)
                             max_diff = torch.max(abs_diff).item()
                             max_diff_idx = torch.argmax(abs_diff).item()
                             # --- Add offending element count and top diffs ---
                             threshold = 1e-3
                             offending_indices = torch.where(abs_diff > threshold)[0]
                             num_offending = offending_indices.numel()
                             percentage_offending = (num_offending / n_compare) * 100 if n_compare > 0 else 0
                             # Find top 5 largest absolute differences
                             top_k_diffs, top_k_indices = torch.topk(abs_diff, k=min(5, n_compare))
                             top_diff_details = []
                             for i in range(top_k_indices.numel()):
                                 idx = top_k_indices[i].item()
                                 top_diff_details.append(f"idx {idx}: Ring={v1_flat_comp[idx]:.4f}, Engine={v2_flat_comp[idx]:.4f} (Diff={top_k_diffs[i]:.4f})")
                             diff_lines.append(f"  Stats: L2 Norm Diff={norm_diff:.6f}, Max Abs Diff={max_diff:.6f} @ index {max_diff_idx}, Offending (> {threshold:.1e}): {num_offending}/{n_compare} ({percentage_offending:.2f}%)")
                             diff_lines.append(f"  Top 5 Diffs: [{', '.join(top_diff_details)}]")

                 # ... (rest of the detailed diff logic for tuples, types, etc.) ...
                 elif isinstance(val1, tuple) and isinstance(val2, tuple):
                     # ... (existing tuple comparison logic) ...
                     pass # Placeholder if tuple logic was removed/commented
                 else:
                     diff_lines.append(f"  Mismatched types: Ring={type(val1)}, Engine={type(val2)}")

                 diffs[suffix] = "\n".join(diff_lines) # Store the detailed diff string

        return diffs




    def _gather_debug_tensors(self, data, strategy):
        """Recursively gathers tensors (if sharded) and returns CPU copy."""
        if isinstance(data, torch.Tensor):
            try:
                if isinstance(strategy, RingAttentionStrategy) and strategy.world_size > 1:
                    if data.ndim == 3:
                        dim = 1
                    elif data.ndim == 4:
                        dim = 2
                    else:
                        return data.detach().cpu()
                    return strategy.gather_tensor(data, dim=dim).detach().cpu()
            except Exception as e:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"[Rank {rank}] Warning: gather failed: {e}")
            return data.detach().cpu()

        elif isinstance(data, dict):
            return {k: self._gather_debug_tensors(v, strategy) for k, v in data.items()}

        elif isinstance(data, (list, tuple)):
            return type(data)(self._gather_debug_tensors(v, strategy) for v in data)

        return data  # Scalar or unknown type


    def _forward_ring_attention(
        self,
        x,
        *,
        mask,
        position_ids,
        past_key_value_state,
        use_cache,
        is_causal_mask,
        strategy,
        enable_debug_info: bool,
        minimal_debug_prints: bool, # Add flag
        rank=0,
    ):
        debug = {}
        
        if enable_debug_info:
            debug[f"ring_x_input_r{rank}"] = x.clone().detach().cpu()

        residual = x
        x_norm = self.ln(x)


        if enable_debug_info:
            debug[f"ring_x_norm_r{rank}"] = x_norm.clone().detach().cpu()

        ring_helper = RingAttentionHelper(
            attn_module=self.attn,
            layer_idx=self.layer_index,
            strategy=strategy,
            llama_block=self,
            use_cache=use_cache,
            debug_mode=enable_debug_info,
            minimal_debug_prints=minimal_debug_prints,
            ff=self.ff_sub_layer,              # Match engine
            ff_norm=self.ff_ln,                # Match engine
        )



        x, cache, debug_ring = ring_helper.forward(
            x_norm,
            mask=mask,
            strategy=strategy,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            is_causal_mask=is_causal_mask,
            rank=rank,
            minimal_debug_prints=minimal_debug_prints,
            valid_len=strategy._local_valid_len,
            residual=residual  # ✅ Pass the correct x_block source
        )


        if enable_debug_info and debug_ring:
            for k, v in debug_ring.items():
                debug[f"ring_{k}"] = v

        return x, cache, debug if enable_debug_info else (x, cache)







    def _forward_engine_attention(
        self,
        x,
        *,
        mask,
        position_ids,
        past_key_value_state,
        use_cache,
        is_causal_mask,
        enable_debug_info: bool,
        minimal_debug_prints: bool, # Add flag
    ):
        debug = {}
        # Log the input x received by this function
        if enable_debug_info:
            debug[f"x_input_r{dist.get_rank() if dist.is_initialized() else 0}"] = x.clone().detach().cpu() # Use simple key
        x_norm = self.ln(x) # This is the global x_norm
        if enable_debug_info:
            debug[f"x_norm_r{dist.get_rank() if dist.is_initialized() else 0}"] = x_norm.clone().detach().cpu() # Use simple key

        queries, keys, values = self.compute_local_qkv_and_rope(
            self.attn,
            q=x_norm,
            k=x_norm,
            v=x_norm,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value_state=past_key_value_state,
            is_self=True,
        )

        if use_cache and past_key_value_state and past_key_value_state[0].numel() > 0:
            keys = torch.cat((past_key_value_state[0], keys), dim=2)
            values = torch.cat((past_key_value_state[1], values), dim=2)

        expansion = self.attn.nheads // self.attn.kvheads
        keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else keys
        values_e = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else values

        engine = ThreadedRingAttentionEngine(
            block_size=32,
            attn=self.attn,
            ff=self.ff_sub_layer,
            ff_norm=self.ff_ln,
            is_causal=is_causal_mask and mask is None,
            debug_mode=enable_debug_info,
            minimal_debug_prints=minimal_debug_prints, # Pass flag
        )


        engine_output = engine.forward_full(
            q_global=queries,
            k_global=keys_e,
            v_global=values_e,
            mask_global=mask,
            x_global=x,
            x_norm_global=x_norm, # Pass global x_norm
        )


        if enable_debug_info:
            x, engine_debug_data = engine_output
            if engine_debug_data:
                # Add engine prefix to keys coming from the engine's debug buffer
                for k, v in engine_debug_data.items():
                    debug[f"engine_{k}"] = v

        else:
            x = engine_output


        cache = (keys, values) if use_cache else None
        return x, cache, debug



class LLaMA(nn.Module):
    def __init__(
        self,
        config: Optional[LLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        print(distributed_strategy)
        super(LLaMA, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = LLaMAConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        shared = WordEmbedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
            abs_pos=False,
            reversible=True,
            tie_weights=self.config.tie_heads,
            bias=False,
        )

        # TP does not work with tied weights
        if (
            not isinstance(self.distributed_strategy, TensorParallelStrategy)
            or not self.config.tie_heads
        ):
            self.shared = self.distributed_strategy.distribute_module(shared)
        else:
            logger.warning(
                "You're using TP on a model with tied weights between head and embedding. "
                "The tied weights won't be sharded, which can result in unexpected OOMs."
            )
            self.shared = shared

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            ntk_scaling=self.config.ntk_scaling,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_theta,
        )
        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = LLaMABlock(self.config, self.rot_emb, layer_index=i)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def get_config(self) -> LLaMAConfig:
        return self.config

    @classmethod
    def from_config(cls, config: LLaMAConfig) -> "LLaMA":
        return cls(config)

    def reset_parameters(self):
        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, WordEmbedding)
                or isinstance(m, GatedLinearUnit)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

    def validate_reset_parameters(self):
        # Verifies that the above self.reset_parameters() executed correctly.
        # This may not always be the case for distributed settings with sharded tensors,
        # such as FSDP or TP. Note that performing this check may require unsharding /
        # re-materializing the full model on a single rank to access the underlying tensors.
        tolerance = 1e-3

        def check_close(x):
            assert x.mean().abs() < tolerance
            assert x.std().sub(0.02).abs() < tolerance

        with torch.no_grad():
            for p in self.parameters():
                assert p.isnan().int().sum() == 0
                assert p.isinf().int().sum() == 0
            for m in self.modules():
                if isinstance(LayerNormParameterized):
                    if m.elementwise_scale:
                        assert m.weight.sum() == m.weight.numel()
                    if m.elementwise_shift:
                        assert m.bias.add(1).sum() == m.bias.numel()
                elif isinstance(WordEmbedding):
                    check_close(m.emb.weight)
                    check_close(m.head.weight)
                elif isinstance(GatedLinearUnit):
                    check_close(m.w1.weight)
                    check_close(m.w2.weight)
                    check_close(m.wg.weight)
                elif isinstance(MultiHeadAttention):
                    check_close(m.query.weight)
                    check_close(m.key.weight)
                    check_close(m.value.weight)
                    check_close(m.dense.weight)

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[Optional[torch.device], dict[int, torch.Tensor]],
        max_seq_len_cached: dict[Optional[torch.device], int],
    ):
        # remove meta tensors from cached_freqs
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device

        # if this model ties weights, they are tied here
        if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.shared.head.weight.device == torch.device("meta"):
                self.shared.head.weight = self.shared.emb.weight
            else:
                self.shared.emb.weight = self.shared.head.weight

        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def _helper(
        self,
        x_in,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        attn_algorithm=None,
        distributed_strategy: Optional[DistributedStrategy] = None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        qlen = x_in.size(1)
        klen = x_in.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_key_value_states[0] is not None:
            klen += past_key_value_states[0][0].size(-2)

        # if mask is none, we need to specify causal mask
        if mask is None:
            # we are caching and can assume all 1s in the mask
            if use_cache and klen != 1 and qlen == 1:
                # b x h x qlen x kvlen
                is_causal_mask = False
            else:
                is_causal_mask = True
        else:
            is_causal_mask = False

        x_in = self.shared(x_in)

        if isinstance(distributed_strategy, RingAttentionStrategy):
            x_in = self.distributed_strategy.shard_input(x_in)


        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
                distributed_strategy=distributed_strategy, # Pass strategy to the block
            )

            if use_cache:
                x_in, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x_in = output

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        # if isinstance(distributed_strategy, RingAttentionStrategy):
        #     dec_out = distributed_strategy.gather_output(dec_out)

        return dec_out, present_key_value_states

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        output, cache = self._helper(
            x,
            mask,
            position_ids,
            past_key_value_states,
            use_cache,
            attn_algorithm,
            # Pass the strategy from the main model instance
            distributed_strategy=self.distributed_strategy,
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.shared(output, reverse=True)

        if use_cache:
            return preds, cache
        else:
            return preds


# Register common LLaMA variants with the model registration API

# a micro llama model to use with a char-level tokenizer
_micro_char_config = LLaMAConfig(
    emb_dim=192, nheads=4, nlayers=5, max_expected_seq_len=1024, src_vocab_size=256
)

_7b_config = LLaMAConfig()
_13b_config = LLaMAConfig(emb_dim=5120, nheads=40, nlayers=40)
# todo: add 35B config

_70b_config = LLaMAConfig(
    emb_dim=8192,
    multiple_of=4096,
    nheads=64,
    kvheads=8,
    nlayers=80,
    hidden_grow_factor=(1.3 * 8 / 3),
)

_8b_llama3_config = LLaMAConfig(
    src_vocab_size=128256,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=32,
    hidden_grow_factor=3.5,
    multiple_of=1024,
    max_expected_seq_len=8192,
    rope_theta=500_000.0,
)

# Granite configs
_granite_7b_config = LLaMAConfig(
    src_vocab_size=32008,
)

_granite_3b_code_config = LLaMAConfig(
    src_vocab_size=49152,
    emb_dim=2560,
    pad_id=0,
    hidden_grow_factor=10240 / 2560,
    multiple_of=1,
    p_dropout=0.1,
    max_expected_seq_len=2048,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

_granite_8b_code_config = LLaMAConfig(
    src_vocab_size=49152,
    emb_dim=4096,
    kvheads=8,
    nlayers=36,
    pad_id=0,
    hidden_grow_factor=14336 / 4096,
    multiple_of=1,
    p_dropout=0.1,
    max_expected_seq_len=4096,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

_architecture_name = "llama"


def _llama_factory_factory(config):
    def factory(**kwargs):
        return LLaMA(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "micro", _llama_factory_factory(_micro_char_config)
)
# Backwards compat
models.register_model(_architecture_name, "7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "13b", _llama_factory_factory(_13b_config))
models.register_model(_architecture_name, "70b", _llama_factory_factory(_70b_config))

# LLama 2 family
models.register_model(_architecture_name, "2-7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "2-13b", _llama_factory_factory(_13b_config))
models.register_model(_architecture_name, "2-70b", _llama_factory_factory(_70b_config))

# LLama 3 family
models.register_model(
    _architecture_name, "3-8b", _llama_factory_factory((_8b_llama3_config))
)

# Granite family
models.register_model(
    _architecture_name, "granite-7b", _llama_factory_factory((_granite_7b_config))
)
models.register_model(
    _architecture_name,
    "granite.code-3b",
    _llama_factory_factory((_granite_3b_code_config)),
)
models.register_model(
    _architecture_name,
    "granite.code-8b",
    _llama_factory_factory((_granite_8b_code_config)),
)

# Create all the pieces to generate adapters for different checkpoints
serialization.register_adapter_step(
    "llama", "pre0.0.6_attn_unfused_to_fused", serialization._pre006_attn_adapter_step
)

serialization.register_adapter_step(
    "llama",
    "swiglu_unfused_to_fused",
    serialization._mlp_glu_unfused_to_fused_adapter_step,
)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[LLaMAConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd


serialization.register_adapter_step("llama", "weight_fusion", _weight_fusion)


def _hf_gptq_llama_check(
    input_sd: Mapping[str, Any], model_config: Optional[LLaMAConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF llama checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    "llama", "hf_gptq_fusion_check", _hf_gptq_llama_check
)


def _meta_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^tok_embeddings", "shared.emb"),
        (r"^norm", "dec_norm"),
        (r"^output", "shared.head"),
        (r"^layers", "layers"),
        (r"\.attention\.", ".attn."),
        (r"attn\.wq", "attn.in_proj.query"),
        (r"attn\.wk", "attn.in_proj.key"),
        (r"attn\.wv", "attn.in_proj.value"),
        (r"attn\.wo", "attn.dense"),
        (r"attention_norm", "ln"),
        (r"feed_forward\.w1", "ff_sub_layer.wg"),
        (r"feed_forward\.w2", "ff_sub_layer.w2"),
        (r"feed_forward\.w3", "ff_sub_layer.w1"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd


serialization.register_adapter_step("llama", "meta_to_fms_names", _meta_to_fms_names)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^lm_head.weight", "shared.head.weight"),
        (r"^model.embed_tokens.weight", "shared.emb.weight"),
        (r"^model.norm", "dec_norm"),
        (r"^model.layers", "layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step("llama", "hf_to_fms_names", _hf_to_fms_names)


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[LLaMAConfig] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
        linear_type = "torch_linear"
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models
        linear_type = "torch_linear"

    rope_params = _get_rope_params(linear_type)
    trans_required_pattern = re.compile(
        f"layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
    )
    for name, param in input_sd.items():
        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Llama models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.match(name)):
            temp = param
            if "gptq" in linear_type and temp.dim() == 2:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if "gptq" in linear_type and temp.dim() == 2:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step("llama", "hf_to_fms_rope", _hf_to_fms_rope)


serialization.register_adapter("llama", "meta", ["meta_to_fms_names", "weight_fusion"])
serialization.register_adapter(
    "llama",
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "hf_gptq_fusion_check", "weight_fusion"],
)
serialization.register_adapter(
    "llama",
    "fms.pre0.0.6",
    ["pre0.0.6_attn_unfused_to_fused", "swiglu_unfused_to_fused", "weight_fusion"],
)
