import logging
import re
from dataclasses import dataclass, field # Import field
from typing import Any, List, Mapping, Optional, Tuple, Union # Added Union
import sys # For flushing stdout, useful for debugging
from fms.models.ring_attention_helper import RingAttentionHelper
import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    RingAttentionStrategy,
    TensorParallelStrategy,
)

from fms.models.llama_ring import (
    forward_ring,
    _forward_ring_attention,
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
class LLaMAConfig(ModelConfig): # type: ignore
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
    # Debugging fields
    debug_mode: bool = field(default=True, metadata={"help": "Enable debug mode for tensor comparison"}) # Default to True for easier testing
    debug_target_layer: Optional[int] = field(default=0, metadata={"help": "Specific layer to debug (e.g., 0 for the first layer)"}) # Default to 0
    debug_print_values: bool = field(default=True, metadata={"help": "Print tensor values during debug comparison"})
    debug_tolerance: float = field(default=1e-9, metadata={"help": "Tolerance for tensor comparison in debug mode"})


class LLaMABlock(nn.Module):

    def __init__(self, config: LLaMAConfig, rotary_emb: RotaryEmbedding, layer_idx: int = -1): # Added layer_idx
        super(LLaMABlock, self).__init__()
        self.config = config
        self.layer_idx = layer_idx # Store layer index
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

        # Make _forward_ring_attention available as an instance method
        self._forward_ring_attention = _forward_ring_attention.__get__(self, LLaMABlock)

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

        self.ring_helper = None # Initialize to None, will be created on first use with the correct strategy

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal_mask: bool = False,
        attn_algorithm: Optional[str] = None,
        distributed_strategy: Optional[DistributedStrategy] = None,
        # New parameters for debug dict population (only used by forward_ring for now)
        debug_dict_ring: Optional[dict] = None, 
        debug_key_prefix_ring: str = "",
        # layer_idx_for_debug is passed from LLaMA._helper
        layer_idx_for_debug: int = -1
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:

        effective_strategy = distributed_strategy if distributed_strategy is not None else NoOpStrategy

        if isinstance(effective_strategy, RingAttentionStrategy):
            # print(torch.distributed.get_rank(), x.shape)
            return forward_ring(
                self, # LLaMABlock instance
                x,    # Expects sharded input
                mask=mask,
                position_ids=position_ids, # Expects sharded position_ids
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
                distributed_strategy=effective_strategy, # Pass the RingAttentionStrategy
                # Pass the ring-specific debug dict and prefix for population
                # These will be used by the _forward_ring_attention and RingAttentionHelper
                debug_dict_populate=debug_dict_ring, 
                debug_key_prefix_populate=debug_key_prefix_ring,
                debug_print_values=self.config.debug_print_values, # from block's config
                debug_tolerance=self.config.debug_tolerance,     # from block's config
                layer_idx=self.layer_idx # Use the block's own layer_idx
            )
        
        # if the cache is not empty, we need to get the kv cache for self and cross attention
        self_attn_past_key_value = past_key_value_state

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            mask=mask,
            position_ids=position_ids,
            attn_algorithm=attn_algorithm,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            is_self=True,
            is_causal_mask=is_causal_mask,
        )
        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # residual connection
        x = x + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x + residual

        if use_cache:
            return (x, cache)
        else:
            return x


class LLaMA(nn.Module):
    def __init__(
        self,
        config: Optional[LLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(LLaMA, self).__init__()
        # Ensure NoOpStrategy is instanced if passed as class
        if distributed_strategy == NoOpStrategy:
            distributed_strategy = NoOpStrategy

        if config is not None:
            self.config = config
        else:
            self.config = LLaMAConfig() # Create default if None
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
        # RoPE init should happen after model is moved to device, in post_init typically.
        # Initializing on default device if available, post_init will handle proper device.
        try:
            # Check if parameters exist and get a device
            # This might fail if called before model is fully initialized or moved to device
            if len(list(self.parameters())) > 0:
                 dev = next(self.parameters()).device
                 if dev != torch.device("meta"): # Don't compute for meta device
                    self.rot_emb.compute_freqs_cis(dev, self.config.max_expected_seq_len)
        except StopIteration: # No parameters yet
            pass # Will be handled in post_init


        layers = []
        for i in range(self.config.nlayers):
            # Pass layer_idx to LLaMABlock
            block: nn.Module = LLaMABlock(self.config, self.rot_emb, layer_idx=i)
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

        if self.config.p_dropout: # Use config.p_dropout consistently
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
                # Check if the tensor is on meta device
                if hasattr(cached_freqs[dev][alp], 'device') and cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device

        # if this model ties weights, they are tied here
        if self.config.tie_heads and hasattr(self.shared, 'head') and hasattr(self.shared, 'emb'):
            # Handle assignment of non-meta weights to meta parameters
            if self.shared.head.weight.device == torch.device("meta") and \
               self.shared.emb.weight.device != torch.device("meta"):
                self.shared.head.weight = self.shared.emb.weight # type: ignore
            elif self.shared.emb.weight.device == torch.device("meta") and \
                 self.shared.head.weight.device != torch.device("meta"):
                self.shared.emb.weight = self.shared.head.weight # type: ignore

        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )

        # init RoPE on the right device(s)
        devices = set()
        for param in self.parameters():
            devices.add(param.device)
        for buffer in self.buffers(): # type: ignore
            devices.add(buffer.device)
        
        for device in devices:
            if device != torch.device("meta"): # Don't compute for meta device
                self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def _helper(
        self,
        x_in: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        active_strategy = self.distributed_strategy

        original_seq_len = x_in.size(1) # Capture original sequence length
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))] # type: ignore

        qlen = x_in.size(1)
        klen = x_in.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_key_value_states[0] is not None:
            klen += past_key_value_states[0][0].size(-2)
        # if mask is none, we need to specify causal mask
        if mask is None:
            is_causal_mask = not (use_cache and klen != 1 and qlen == 1)
        else:
            is_causal_mask = False

        current_x = self.shared(x_in) # current_x is (B, T, E) and global

        present_key_value_states = []

        for i, layer_module in enumerate(self.layers): # Renamed `layer` to `layer_module`
            # Debug related setup
            current_rank = active_strategy.rank if hasattr(active_strategy, 'rank') else 0
            
            # Determine if the current layer on the current rank is a target for ring debug data population
            target_debug_ranks_for_ring_population = [0, 1] # Ranks that should populate their ring debug dict
            is_this_rank_a_ring_debug_target_for_population = (
                self.config.debug_mode and
                i == self.config.debug_target_layer and
                isinstance(active_strategy, RingAttentionStrategy) and
                current_rank in target_debug_ranks_for_ring_population
            )

            debug_data_ring_this_layer = {} if is_this_rank_a_ring_debug_target_for_population else None
            # The prefix will be used by RingAttentionHelper to create rank-specific keys
            debug_key_prefix_for_current_rank_ring = f"ring_r{current_rank}" if isinstance(active_strategy, RingAttentionStrategy) else ""

            # This part is simplified: it's now handled within _forward_ring_attention in llama_ring.py
            # LLaMA._helper will only pass down the debug_data_ring_this_layer for population.
            # The shadow pass and comparison logic is encapsulated in llama_ring.py.

            # --- Actual Path (Ring or Standard) Execution ---
            input_for_this_layer = current_x
            pos_ids_for_this_layer = position_ids

            if isinstance(active_strategy, RingAttentionStrategy):
                if hasattr(active_strategy, 'shard_input'):
                    input_for_this_layer = active_strategy.shard_input(current_x) # type: ignore
                else:
                    logger.warning(f"Rank {current_rank}: RingAttentionStrategy does not have shard_input method. Passing global input to layer {i}.")
                
                if position_ids is not None:
                    if hasattr(active_strategy, 'shard_input'): # Assuming shard_input can handle position_ids or a similar method exists
                        pos_ids_for_this_layer = active_strategy.shard_input(position_ids) # type: ignore
                    else:
                        logger.warning(f"Rank {current_rank}: RingAttentionStrategy does not have shard_input for position_ids. Passing global position_ids to layer {i}.")
            
            layer_output = layer_module(
                x=input_for_this_layer, 
                mask=mask,
                position_ids=pos_ids_for_this_layer,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
                distributed_strategy=active_strategy, # Pass model's actual strategy
                # Pass the dict for RingAttentionHelper to populate if it's a ring attention layer
                debug_dict_ring=debug_data_ring_this_layer if isinstance(active_strategy, RingAttentionStrategy) else None,
                debug_key_prefix_ring=debug_key_prefix_for_current_rank_ring,
                layer_idx_for_debug=i # Pass the current layer index
            )

            current_x_sharded_or_global: torch.Tensor
            if use_cache:
                current_x_sharded_or_global, present_key_value_state = layer_output # type: ignore
                present_key_value_states.append(present_key_value_state)
            else:
                current_x_sharded_or_global = layer_output # type: ignore

            if isinstance(active_strategy, RingAttentionStrategy):
                if hasattr(active_strategy, 'gather_output'):
                    current_x = active_strategy.gather_output(current_x_sharded_or_global) # type: ignore
                else:
                    logger.warning(f"Rank {current_rank}: RingAttentionStrategy does not have gather_output. Output for layer {i} might remain sharded.")
                    current_x = current_x_sharded_or_global # Fallback
            else:
                current_x = current_x_sharded_or_global
        
        dec_out = current_x # This should be global tensor now
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        if isinstance(active_strategy, RingAttentionStrategy):
            if dec_out.size(1) > original_seq_len:
                 dec_out = dec_out[:, :original_seq_len, :]

        return dec_out, present_key_value_states

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None, # type: ignore
        past_key_value_states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        output, cache = self._helper(
            x, # type: ignore
            mask,
            position_ids,
            past_key_value_states,
            use_cache,
            attn_algorithm,
            # distributed_strategy is taken from self.distributed_strategy in _helper
        )

        if only_last_token:
            if output.ndim == 3: # type: ignore
                output = output[:, -1, :] # type: ignore
            elif output.ndim == 2 and use_cache and only_last_token: 
                pass 
            else:
                logger.warning(f"Unexpected output shape for only_last_token: {output.shape}") # type: ignore
        preds = self.shared(output, reverse=True) # type: ignore

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
