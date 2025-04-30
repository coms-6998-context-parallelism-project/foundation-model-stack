import logging
import re
import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    RingAttentionStrategy,
    TensorParallelStrategy,
)

from fms.models.attentionhelper import RingAttentionEngine
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
    def __init__(self, config: LLaMAConfig, rotary_emb: RotaryEmbedding, ring_attention_group: Optional[dist.ProcessGroup] = None):
        super(LLaMABlock, self).__init__()
        self.config = config
        self.ring_attention_group = ring_attention_group # Store the group if provided
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
        global_seq_len = None,
    ):
        print(attn_algorithm)
        # If using ring attention, handle it separately
        if attn_algorithm == "ring": # This is the check you asked about
            print(f"[rank{self.ring_attention_group.rank() if self.ring_attention_group else 'N/A'}] LLaMABlock.forward: Using Ring Attention path.")
            if self.ring_attention_group is None:
                raise RuntimeError("Ring Attention algorithm requested but no ring_attention_group was provided to the LLaMABlock.")
            # TODO: KV caching with Ring Attention
            if use_cache:
                # TODO: Implement caching for Ring Attention if needed. It's complex due to the distributed nature.
                logger.warning("KV caching is not yet supported with Ring Attention.")
                use_cache = False # Force disable cache for now
            if global_seq_len is None:
                raise ValueError("global_seq_len must be provided when attn_algorithm='ring'")

            # Determine the global offset of this rank's shard
            # This assumes the input x is already sharded by the RingAttentionStrategy
            world_size = self.ring_attention_group.size()
            rank = self.ring_attention_group.rank()
            shard_len = (global_seq_len + world_size - 1) // world_size
            q_global_offset = rank * shard_len

            x_ln = self.ln(x)
            # Project Q, K, V without applying RoPE here
            B, T, _ = x_ln.shape
            q_proj, k_proj, v_proj = self.attn.in_proj(q=x_ln, k=x_ln, v=x_ln) # Pass x_ln for q, k, and v

            # Reshape for RingAttentionEngine (B, H, T, D)
            queries = q_proj.view(B, T, self.attn.nheads, self.attn.emb_kq_per_head).transpose(1, 2)
            keys = k_proj.view(B, T, self.attn.kvheads, self.attn.emb_kq_per_head).transpose(1, 2)
            values = v_proj.view(B, T, self.attn.kvheads, self.attn.emb_v_per_head).transpose(1, 2)

            # === Expand for GQA
            expansion = self.attn.nheads // self.attn.kvheads
            keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else keys

            engine_is_causal = is_causal_mask and mask is None

            engine = RingAttentionEngine(
                block_size=32, # block_size is less relevant now, internal chunking?
                attn=self.attn,
                ff=self.ff_sub_layer,
                ff_norm=self.ff_ln,
                is_causal=engine_is_causal,
                group=self.ring_attention_group # Pass the stored group
            )

            x_out = engine.forward_full(
                q_shard=queries, k_shard=keys_e, v_shard=values, # Pass original projected values
                mask_global=mask, # Mask is still global for now, engine slices it
                x_shard=x,
                q_global_offset=q_global_offset,
                global_seq_len=global_seq_len
            )
            # Ring attention engine now returns the local block, gathering happens outside
            return x_out # No cache returned for ring attention yet
        else:
            # Fallback to the original attention mechanism
            return self.forward_old(
                x,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
            )

       # will later change this to be configurable once we decide on the final architecture
    def forward_old(
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
        # if the cache is not empty, we need to get the kv cache for self and cross attention
        self_attn_past_key_value = past_key_value_state
        # if past_key_value_state is not None:
        #     self_attn_past_key_value = past_key_value_state[:2]
        # else:
        #     self_attn_past_key_value = None

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
            # Pass the group to the block if using RingAttentionStrategy
            ring_group = None
            if isinstance(self.distributed_strategy, RingAttentionStrategy):
                ring_group = self.distributed_strategy.group

            block: nn.Module = LLaMABlock(self.config, self.rot_emb, ring_attention_group=ring_group)
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
        # Store original length for potential unpadding later
        original_global_seq_len = klen
        # This will be the length used for computation (potentially padded)
        compute_global_seq_len = original_global_seq_len

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

        # --- Ring Attention Strategy: Shard Input ---
        is_ring_strategy = isinstance(self.distributed_strategy, RingAttentionStrategy)
        if is_ring_strategy:
            world_size = self.distributed_strategy.world_size
            # Pad sequence length to be divisible by world_size for fixed shards
            padding_needed = (world_size - (original_global_seq_len % world_size)) % world_size
            if padding_needed > 0:
                compute_global_seq_len = original_global_seq_len + padding_needed
                print(f"[rank{self.distributed_strategy.rank}] LLaMA._helper: Padding sequence from {original_global_seq_len} to {compute_global_seq_len} for Ring Attention.")
                # Pad x_in (assuming dim 1 is sequence length)
                # Use pad_id if available, otherwise 0. Be careful if 0 is a valid token.
                pad_value = self.config.pad_id if self.config.pad_id >= 0 else 0
                x_in = F.pad(x_in, (0, 0, 0, padding_needed), value=pad_value) # Pad last dim (seq length)

                # Adjust mask and position_ids
                if mask is not None:
                    # Assuming mask is [B, 1, qlen, klen] or similar, pad the klen dimension
                    mask = F.pad(mask, (0, padding_needed), value=-torch.inf) # Pad last dim (key length)
                    # If qlen also needs padding (e.g., mask is square), pad second-to-last dim
                    if mask.shape[-2] == original_global_seq_len:
                         mask = F.pad(mask, (0, 0, 0, padding_needed), value=-torch.inf) # Pad second-to-last dim (query length)
                if position_ids is not None:
                    # Pad position_ids like x_in
                    position_ids = F.pad(position_ids, (0, padding_needed), value=0) # Pad last dim

            x_in = self.distributed_strategy.shard_input(x_in, dim=1)

        x_in = self.shared(x_in)

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
                global_seq_len=compute_global_seq_len if is_ring_strategy else None, # Pass potentially padded length
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

        # --- Ring Attention Strategy: Gather Output ---
        if is_ring_strategy:
            # Gather along the sequence dimension
            dec_out = self.distributed_strategy.gather_output(dec_out, dim=1, global_len=compute_global_seq_len)
            # Truncate output back to original length if padding was added
            if compute_global_seq_len != original_global_seq_len:
                print(f"[rank{self.distributed_strategy.rank}] LLaMA._helper: Truncating output from {compute_global_seq_len} back to {original_global_seq_len}.")
                dec_out = dec_out[:, :original_global_seq_len, :]

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
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
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
