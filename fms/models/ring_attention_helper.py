from typing import List, Tuple, Dict, Optional, Union, Any
from fms.modules.attention import MultiHeadAttention
import torch
import torch.distributed as dist
from torch.distributed import P2POp
import math
import torch.nn.functional as F
import torch.nn as nn
from fms.distributed.strategy import RingAttentionStrategy
import sys
def compare_tensors(label, t1, t2, threshold=1e-6):
    if t1.dtype != t2.dtype:
        print(f"[{label}] WARNING: dtype mismatch {t1.dtype} vs {t2.dtype}. Casting both to float32 for comparison.")
        t1 = t1.float()
        t2 = t2.float()
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    count_over_thresh = (diff > threshold).sum().item()
    print(f"[{label}] shape={t1.shape}, dtype={t1.dtype}, max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}, diff>{threshold}: {count_over_thresh}")
    
    # Show indices and values of largest differences
    if count_over_thresh > 0:
        flat_diff = diff.view(-1)
        topk = torch.topk(flat_diff, k=min(5, flat_diff.numel()))
        indices = topk.indices
        print(f"  Top diffs:")
        for idx in indices:
            idx_unravel = list(torch.unravel_index(idx, t1.shape))
            val1 = t1[tuple(idx_unravel)].item()
            val2 = t2[tuple(idx_unravel)].item()
            delta = abs(val1 - val2)
            print(f"    idx={idx_unravel}, manual={val1:.4e}, sdpa={val2:.4e}, diff={delta:.4e}")

def _get_accum_dtype(dtype: torch.dtype) -> torch.dtype:
    return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32


def _get_clamp_bounds(accum_dtype: torch.dtype) -> Tuple[float, float]:
    info = torch.finfo(accum_dtype)
    margin = 2.0
    return math.log(info.tiny) + margin, math.log(info.max) - margin


def compute_local_qkv_and_rope(
    attn_data: MultiHeadAttention,
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    is_self: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, E = q.shape
    q_out, k_out, v_out = attn_data.in_proj(q, k, v)
    assert q_out.shape == k_out.shape == v_out.shape == (B, T, E)

    H_q, H_kv = attn_data.nheads, attn_data.kvheads
    D_kq, D_v = attn_data.emb_kq_per_head, attn_data.emb_v_per_head

    queries = q_out.view(B, T, H_q, D_kq)
    keys = k_out.view(B, T, H_kv, D_kq)
    values = v_out.view(B, T, H_kv, D_v)

    if attn_data.position_encoder and T > 0:
        assert position_ids is not None
        if position_ids.shape != (B, T) and not (T == 0 and position_ids.shape == (B, 0)):
            raise AssertionError(f"Expected shape {(B, T)}, got {position_ids.shape}")
        valid_mask = position_ids != -1
        if valid_mask.any():
            max_pos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
            position_ids_safe = position_ids.clone()
            position_ids_safe[valid_mask] = position_ids_safe[valid_mask].clamp(0, max_pos - 1)
            q_rope, k_rope = attn_data.position_encoder.adjusted_qk(queries, keys, position_ids_safe)
            if valid_mask.all():
                queries, keys = q_rope, k_rope
            else:
                mask = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(queries)
                queries = torch.where(mask, q_rope, queries)
                keys = torch.where(mask.expand_as(keys), k_rope, keys)

    return (
        queries.permute(0, 2, 1, 3),
        keys.permute(0, 2, 1, 3),
        values.permute(0, 2, 1, 3),
    )


def _pad_to_block(t: torch.Tensor, target_len: int, dim: int = 2) -> torch.Tensor:
    if t.size(dim) >= target_len:
        assert t.size(dim) == target_len, f"Shape {t.shape} incompatible with target {target_len}"
        return t
    pad = torch.zeros(
        *t.shape[:dim],
        target_len - t.size(dim),
        *t.shape[dim + 1 :],
        dtype=t.dtype,
        device=t.device,
    )
    return torch.cat([t, pad], dim=dim)


class RingAttentionHelper:
    def __init__(
        self,
        attn_module: nn.Module,
        strategy: Any,
        llama_block: nn.Module,
        use_cache: bool = False,
        ff: Optional[nn.Module] = None,
        ff_norm: Optional[nn.Module] = None,
    ):
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.strategy = strategy
        self.use_cache = use_cache
        self.llama_block = llama_block

        self.rank = strategy.rank
        self.world_size = strategy.world_size
        self.block_size = strategy.block_size

        self.head_dim = attn_module.emb_kq_per_head
        self.scale = attn_module.scale_factor or math.sqrt(self.head_dim)
        self._accum_dtype = torch.float32


    def forward(
        self,
        x_norm,
        strategy,
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        is_causal_mask=False,
        valid_len=0,
        residual=None,
        # New debug parameters
        debug_dict_populate: Optional[dict] = None,
        debug_key_prefix_populate: str = "",
        layer_idx: int = -1, 
        debug_print_values: bool = False, 
        debug_tolerance: float = 1e-3,
    ):
        assert isinstance(strategy, RingAttentionStrategy) and self.strategy is strategy
        B, T_padded, _ = x_norm.shape
        if self.world_size > 1:
            assert T_padded == self.block_size
        
        # Determine if we should populate the debug dictionary for this call
        # self.llama_block is available from __init__
        target_debug_ranks_for_population = [0, 1] # Ranks that should populate
        is_debug_target_layer_and_this_rank_should_populate = (
            self.llama_block.config.debug_mode and
            layer_idx == self.llama_block.config.debug_target_layer and
            self.rank in target_debug_ranks_for_population
        )
        can_populate_debug = is_debug_target_layer_and_this_rank_should_populate and debug_dict_populate is not None

        start = self.rank * self.block_size

        if position_ids is None:
            position_ids = torch.full((B, T_padded), -1, dtype=torch.long, device=x_norm.device)
            if valid_len > 0:
                # Ensure position_ids are correctly representing global positions for the local shard
                position_ids[:, :valid_len] = torch.arange(start, start + valid_len, device=x_norm.device)
        
        # Debug: Log and assert validity of position_ids
        if can_populate_debug and valid_len > 0 and position_ids is not None:
            local_pos_ids_valid_part = position_ids[:, :valid_len]
            if (local_pos_ids_valid_part == -1).all():
                print(f"WARNING HELPER (L{layer_idx} R{self.rank}): All position_ids for valid_len={valid_len} are -1. RoPE might be skipped or incorrect.")
                sys.stdout.flush()

        x_trim = x_norm[:, :valid_len, :]
        pos_trim = position_ids[:, :valid_len]
        res_trim = residual[:, :valid_len, :] if residual is not None else None

        if can_populate_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_x_norm"] = x_trim.detach().clone().cpu()
            if res_trim is not None: # Corresponds to the input residual to the block for this shard
                debug_dict_populate[f"{debug_key_prefix_populate}_residual_input"] = res_trim.detach().clone().cpu()

        q, k, v = compute_local_qkv_and_rope(self.attn, x_trim, x_trim, x_trim, pos_trim)

        if can_populate_debug:
            # q, k, v are (B, H, T_local, D_head)
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_q_local"] = q.detach().clone().cpu()
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_k_local"] = k.detach().clone().cpu()
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_v_local"] = v.detach().clone().cpu()

        if self.attn.nheads != self.attn.kvheads:
            e = self.attn.nheads // self.attn.kvheads
            k = k.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_kq_per_head)
            v = v.unsqueeze(2).expand(-1, -1, e, -1, -1).reshape(B, self.attn.nheads, valid_len, self.attn.emb_v_per_head)

        out_valid = self.forward_full(
            q, k, v, mask, 
            valid_len, # local valid length (T)
            res_trim, # residual input for this shard (x_block in forward_full)
            start,    # q_start_global,
            is_causal_mask, # Pass down the global causal flag
            # Pass debug parameters
            can_populate_debug=can_populate_debug,
            debug_dict_populate=debug_dict_populate,
            debug_key_prefix_populate=debug_key_prefix_populate,
            # layer_idx is implicitly available in self.llama_block.layer_idx if needed by forward_full
            # but RingAttentionHelper.forward gets it explicitly, so let's pass it if sub-methods need it.
            layer_idx=layer_idx # Pass layer_idx from the main forward call
        )
        # out_valid is the final output of the block for this shard, after all residuals and FFN
        if can_populate_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_block_output"] = out_valid.detach().clone().cpu()

        return (_pad_to_block(out_valid, self.block_size, dim=1), None, None) if self.world_size > 1 else (out_valid, None, None)

    def forward_full(self, q, k, v, mask, valid_len_local_T, x_block_residual_input, q_start_global,
                     is_causal_mask: bool, # Added is_causal_mask
                     can_populate_debug: bool = False,
                     debug_dict_populate: Optional[dict] = None,
                     debug_key_prefix_populate: str = "",
                     layer_idx: int = -1): # Added layer_idx
        B, H, T, D_kq = q.shape # T is the local query length for this rank
        _, _, _, D_v = v.shape   # D_v is the value head dimension

        # All ranks must participate in ring communication, even if T (local query length) is 0.
        # The max_score, num, and den computations will naturally result in tensors with a 0 query dimension if T=0.

        accum_dtype = self._accum_dtype
        exp_min, exp_max = _get_clamp_bounds(accum_dtype)
        
        # q_compute will have T as its query dimension length.
        # k_compute and v_compute sequence lengths are based on the K/V blocks being processed,
        # which may be non-zero even if local T is 0.
        q_compute = q.to(accum_dtype)
        k_compute = k.to(accum_dtype)
        v_compute = v.to(accum_dtype)

        if self.rank == 0 and can_populate_debug: # Check for Rank 0 if debugging
            # print(f"DEBUG HELPER forward_full (L{layer_idx} R{self.rank}): q_compute.shape={q_compute.shape}, "
            #       f"k_compute.shape={k_compute.shape}, v_compute.shape={v_compute.shape}, T (valid_len_local_T)={valid_len_local_T}")
            sys.stdout.flush()

        # max_score will have shape (B, H, T, 1)
        max_score = self._compute_max_score_pass(
            q_compute, k_compute, mask, q_start_global, valid_len_local_T,
            is_causal_mask, # Pass down
            can_populate_debug=can_populate_debug, # Pass down
            debug_dict_populate=debug_dict_populate, # Pass down
            debug_key_prefix_for_pass=debug_key_prefix_populate, # Pass down (renamed for clarity in sub-method)
            layer_idx=layer_idx # Pass down
        )
        # num will have shape (B, H, T, D_v) 
        # den will have shape (B, H, T, 1)
        num, denom = self._compute_sums_pass( # Add is_causal_mask here
            q_compute, k_compute, v_compute, mask, q_start_global, valid_len_local_T, max_score, exp_min, exp_max,
            is_causal_mask, # Pass down
            can_populate_debug=can_populate_debug, # Pass down
            debug_dict_populate=debug_dict_populate, # Pass down
            debug_key_prefix_populate=debug_key_prefix_populate, # Pass down
            layer_idx=layer_idx # Pass down
        )
        
        if T > 0:
            eps = torch.finfo(denom.dtype).eps
            attn_manual = (num / (denom + eps)).to(q.dtype)  # shape: (B, H, T, D_v)

            # Use SDPA
            attn_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal_mask)

            # --- Debug Comparison ---
            def log_stats(tensor, name):
                print(f"[{name}] shape={tensor.shape}, dtype={tensor.dtype}, "
                    f"min={tensor.min().item():.4e}, max={tensor.max().item():.4e}, "
                    f"mean={tensor.mean().item():.4e}, nan={torch.isnan(tensor).any().item()}, inf={torch.isinf(tensor).any().item()}",
                    flush=True)

            # log_stats(attn_manual, "ATTN_MANUAL")
            # log_stats(attn_sdpa, "ATTN_SDPA")

            compare_tensors("ATTN_COMPARE",attn_manual, attn_sdpa, threshold=1e-7)

            # diff = (attn_manual - attn_sdpa).abs()
            # print(f"[ATTN_DIFF] max={diff.max().item():.4e}, mean={diff.mean().item():.4e}, "
            #     f"diff>1e-3: {(diff > 1e-3).sum().item()}, diff>1e-2: {(diff > 1e-2).sum().item()}",
            #     flush=True)
            
            attn = attn_manual
            # Apply dropout if configured and in training mode (matches SDPA behavior)
            if self.attn.p_dropout > 0 and self.llama_block.training:
                attn = F.dropout(attn, p=self.attn.p_dropout, training=True)
            if can_populate_debug: # Capture pre-dense context (probs @ V)
                debug_dict_populate[f"{debug_key_prefix_populate}_context_raw"] = attn.detach().clone().cpu()

            attn_reshaped = attn.transpose(1, 2).contiguous().view(B, T, H * D_v)
            attn_out = self.attn.dense(attn_reshaped) # attn_out shape (B, T, E)
            if can_populate_debug: # Capture post-dense output
                debug_dict_populate[f"{debug_key_prefix_populate}_attn_out_dense"] = attn_out.detach().clone().cpu()
        else: # T == 0
            # For T=0, attn (pre-dense context) would be (B, H, 0, D_v)
            # self.attn.emb_v_per_head is D_v
            attn_empty_context = torch.empty(B, H, 0, self.attn.emb_v_per_head, device=q.device, dtype=q.dtype)
            if can_populate_debug:
                debug_dict_populate[f"{debug_key_prefix_populate}_context_raw"] = attn_empty_context.cpu()
            
            # attn_out is already correctly shaped as (B, 0, E)
            attn_out = torch.empty(B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype)
            if can_populate_debug: # Capture post-dense output (empty)
                debug_dict_populate[f"{debug_key_prefix_populate}_attn_out_dense"] = attn_out.detach().clone().cpu()

        # x_block_residual_input has shape (B, T, E). If T=0, it's (B, 0, E). This is the sharded input residual.
        x_after_attn_residual = x_block_residual_input + attn_out
        if can_populate_debug: # This capture is for the residual after attention
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_out_residual"] = x_after_attn_residual.detach().clone().cpu()

        # FeedForward part
        # self.ff_norm and self.ff are from the LLaMABlock instance
        if self.ff_norm is None or self.ff is None:
            raise ValueError("FeedForward (ff) and its LayerNorm (ff_norm) must be provided to RingAttentionHelper for a full block pass.")

        ff_ln_out = self.ff_norm(x_after_attn_residual)
        if can_populate_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_ff_ln_out"] = ff_ln_out.detach().clone().cpu()

        ff_out_raw = self.ff(ff_ln_out)
        if can_populate_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_ff_out_raw"] = ff_out_raw.detach().clone().cpu()

        # Second residual connection (output of the block for this shard)
        block_output_valid = ff_out_raw + x_after_attn_residual 
        # This final block_output_valid will be captured in the main forward method after padding.
        return block_output_valid

    def _compute_max_score_pass(self, q_compute, k_compute, mask_global, q_start_global, valid_len_local,
                                is_causal_mask: bool, # Added is_causal_mask
                                can_populate_debug: bool = False,
                                debug_dict_populate: Optional[dict] = None,
                                debug_key_prefix_for_pass: str = "", 
                                layer_idx: int = -1):
        B, H, T, _ = q_compute.shape
        dtype = q_compute.dtype
        dev = q_compute.device
        max_score = torch.full((B, H, T, 1), torch.finfo(dtype).min, device=dev, dtype=dtype)
        q_idx_global = torch.arange(q_start_global, q_start_global + T, device=dev) # Global indices for local queries
        k_blk, k_len = k_compute, k_compute.shape[2]
        
        # For mask logging in this pass
        # debug_key_prefix_for_pass is already specific (e.g., "ring_r0")
        current_k_block_start_global_idx = ((self.rank - 0 + self.world_size) % self.world_size) * self.block_size # Initial k_block start

        for i in range(self.world_size):
            # k_idx needs to reflect the global indices of the current k_blk
            k_idx_global = torch.arange(current_k_block_start_global_idx, current_k_block_start_global_idx + k_len, device=dev)
            m = mask_global[:, :, q_start_global:q_start_global + T, current_k_block_start_global_idx:current_k_block_start_global_idx + k_len] if mask_global is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(
                    q_compute, k_blk[:, :, :k_len], q_idx_global, k_idx_global, m, apply_implicit_causal_mask=is_causal_mask,
                    # Pass debug params for mask logging
                    can_populate_debug_mask=can_populate_debug and i==0, # Only for first k-block
                    debug_dict_populate_mask=debug_dict_populate,
                    debug_key_prefix_populate_mask=f"{debug_key_prefix_for_pass}_maxpass_kblock0" if i==0 else "",
                    layer_idx_mask=layer_idx
                )
                max_score = torch.maximum(max_score, s.amax(-1, keepdim=True))
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, k_len)
                # Update the start index for the next k_block
                current_k_block_start_global_idx = ((self.rank - (i + 1) + self.world_size) % self.world_size) * self.block_size
        return max_score

    def _compute_sums_pass(
        self, q_compute, k_compute, v_compute, mask_global, q_start_global, valid_len_local, max_score, exp_min, exp_max, is_causal_mask: bool, # Added is_causal_mask
        # Debug parameters
        can_populate_debug: bool = False, 
        debug_dict_populate: Optional[dict] = None,
        debug_key_prefix_populate: str = "", # This is the rank-specific prefix like "ring_r0"
        layer_idx: int = -1
    ):
        B, H, T, Dk = q_compute.shape # q_compute has Dk
        _, _, _, Dv = v_compute.shape # v_compute has Dv
        dev = q_compute.device
        dtype = self._accum_dtype
        num = torch.zeros(B, H, T, Dv, device=dev, dtype=dtype)
        den = torch.zeros(B, H, T, 1, device=dev, dtype=dtype)
        num_comp = torch.zeros_like(num)
        den_comp = torch.zeros_like(den)
        q_idx_global = torch.arange(q_start_global, q_start_global + T, device=dev) # Global indices for local queries
        k_blk, v_blk, k_len = k_compute, v_compute, k_compute.shape[2]
        current_k_block_start_global_idx = ((self.rank - 0 + self.world_size) % self.world_size) * self.block_size # Initial k_block start

        # Print k_len immediately after initialization, only for rank 0 if debugging this layer
        if self.rank == 0 and can_populate_debug: # Check layer_idx if it's available and relevant
            # print(f"DEBUG HELPER _compute_sums_pass INIT (L{layer_idx} R{self.rank}):\n"
            #       f"  k_compute.shape: {k_compute.shape}\n"
            #       f"  k_len initialized to: {k_len}")
            sys.stdout.flush()

        # For clamping diagnostics and softmax stats (already present, ensure they use passed debug params)
        total_clamped_min_count_this_rank = 0
        total_clamped_max_count_this_rank = 0
        softmax_stats_kblock0_this_rank = {}

        for i in range(self.world_size):
            # k_idx needs to reflect the global indices of the current k_blk
            k_idx_global = torch.arange(current_k_block_start_global_idx, current_k_block_start_global_idx + k_len, device=dev)
            m = mask_global[:, :, q_start_global:q_start_global + T, current_k_block_start_global_idx:current_k_block_start_global_idx + k_len] if mask_global is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(
                    q_compute, k_blk[:, :, :k_len], q_idx_global, k_idx_global, m, apply_implicit_causal_mask=is_causal_mask,
                    # Pass debug params for mask logging
                    can_populate_debug_mask=can_populate_debug and i==0, # Only for first k-block
                    debug_dict_populate_mask=debug_dict_populate,
                    debug_key_prefix_populate_mask=f"{debug_key_prefix_populate}_sumpass_kblock0" if i==0 else "",
                    layer_idx_mask=layer_idx
                )
                # Clamp s - max_score to a tighter range for softmax stability
                # Revert to using exp_min, exp_max derived from accum_dtype for clamping s - max_score
                s_minus_max = s - max_score
                s_minus_max_clamped = s_minus_max#.clamp(min=exp_min, max=exp_max)
                
                # If s_minus_max_clamped contains -inf (e.g., s was -inf and max_score was finite),
                # replace these -inf with exp_min before exponentiation.
                # This ensures exp() results in a tiny positive number rather than exactly 0.
                s_for_exp = s_minus_max_clamped.masked_fill(
                    torch.isneginf(s_minus_max_clamped), exp_min
                )
                e = torch.exp(s_for_exp)

                # If max_score was -inf (i.e., all original scores 's' in a row were -inf),
                # then s_minus_max would be NaN, s_for_exp would be NaN, and e would be NaN.
                # In such cases, 'e' (the contribution to softmax) should be 0.
                invalid_row_mask = torch.isneginf(max_score.expand_as(s))
                e = e.masked_fill(invalid_row_mask, 0.0)
                # Using torch.matmul for clarity and potential alignment with SDPA
                # e shape: (B, H, Tq, Tk_block)
                # v_blk[:, :, :k_len] shape: (B, H, Tk_block, Dv)
                # Result shape: (B, H, Tq, Dv)
                contrib_num = torch.matmul(e, v_blk[:, :, :k_len])

                contrib_den = e.sum(-1, keepdim=True)

                # Kahan summation
                y_num = contrib_num - num_comp
                t_num = num + y_num
                num_comp = (t_num - num) - y_num
                num = t_num

                y_den = contrib_den - den_comp
                t_den = den + y_den
                den_comp = (t_den - den) - y_den
                den = t_den

            if i < self.world_size - 1:
                valid = k_len
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, valid)
                v_blk, _ = self._ring_shift_tensor(v_blk, self.block_size, valid)
                # Update the start index for the next k_block
                current_k_block_start_global_idx = ((self.rank - (i + 1) + self.world_size) % self.world_size) * self.block_size
            
            # Diagnostic print BEFORE the main logging block for i=0
            if self.rank == 0 and i == 0: # Focus on Rank 0, first k-block
                # This k_len is the one for the current iteration i, 
                # which for i=0, is the one initialized from k_compute.shape[2]
                # at the start of _compute_sums_pass.
                # print(f"DEBUG HELPER _compute_sums_pass (L{layer_idx} R{self.rank} i={i}) "
                #       f"k_compute.shape_at_start_of_func={k_compute.shape}, "
                #       f"k_len_at_start_of_iter_i={k_len}, T={T}")
                sys.stdout.flush()

                # print(f"DEBUG HELPER PRE-CHECK (L{layer_idx} R{self.rank} i={i}):\n"
                #       f"  can_populate_debug: {can_populate_debug}\n"
                #       f"  debug_dict_populate is not None: {debug_dict_populate is not None}\n"
                #       f"  T (local_query_len): {T}\n"
                #       f"  k_len_in_pre_check_for_iter_i: {k_len}\n" 
                #       f"  Overall condition for logging: {can_populate_debug and i == 0 and debug_dict_populate is not None and T > 0 and k_len > 0}")
                sys.stdout.flush()
            
            # Log intermediate scores and probs for the first k-block if debugging
            if can_populate_debug and i == 0 and debug_dict_populate is not None:
                # k_len_current_block is k_len for this iteration i. For i=0, k_len_current_block is T.
                # T is valid_len_local.
                # The 's', 'e', 'contrib_den' variables are defined if k_len > 0 in the block above.
                # When i=0, k_len (from k_blk.shape[2]) is T.
                
                if T > 0: # If local query length T > 0, then k_len for the first block is also > 0.
                    # 's', 'e', 'contrib_den' must have been computed.
                    # Check for s_minus_max_clamped as well, as it's used for clamping diagnostics.
                    if 's' not in locals() or 'e' not in locals() or 'contrib_den' not in locals() or 's_minus_max_clamped' not in locals():
                        # This case should ideally not be hit if T > 0, as the k_len > 0 block should have run.
                        print(f"DEBUG HELPER WARNING (L{layer_idx} R{self.rank} i={i}): T > 0 but s, e, contrib_den, or s_minus_max_clamped not defined. Skipping population of sdp_scores/probs/stats for kblock0.")
                        sys.stdout.flush()
                    else:
                        # 1. Raw scores per block (s)
                        debug_dict_populate[f"{debug_key_prefix_populate}_sdp_scores_kblock0"] = s.detach().clone().cpu()
                        
                        # 2. "Probs" per block BEFORE summing into num/den
                        # s_minus_max_clamped is (s - max_score_global).clamp(exp_min, exp_max) for this block
                        # Use F.softmax for clarity on the debug probabilities for this block.
                        debug_dict_populate[f"{debug_key_prefix_populate}_sdp_probs_kblock0"]  = F.softmax(s_minus_max_clamped, dim=-1).detach().clone().cpu()
                        debug_dict_populate[f"{debug_key_prefix_populate}_contrib_den_kblock0"] = contrib_den.detach().clone().cpu()

                        if self.rank == 0: # More focused print
                            print(f"DEBUG HELPER (L{layer_idx} R{self.rank} i={i}) Populated _sdp_scores_kblock0 and _sdp_probs_kblock0.")
                            sys.stdout.flush()

                        # Clamping diagnostics for this block (i=0, T>0)
                        # These depend on s_minus_max_clamped and e from the if k_len > 0 block of this iteration.
                        clamped_min_count_block = (s_minus_max_clamped == exp_min).sum().item()
                        clamped_max_count_block = (s_minus_max_clamped == exp_max).sum().item()
                        total_clamped_min_count_this_rank += clamped_min_count_block
                        total_clamped_max_count_this_rank += clamped_max_count_block

                        # Softmax probabilities (e) stats for this block (i=0, T>0)
                        if e.numel() > 0:
                            e_flat_per_head = e.view(B, H, -1)
                            softmax_max_per_head = e_flat_per_head.max(dim=2)[0].mean(dim=0)
                            softmax_min_per_head = e_flat_per_head.min(dim=2)[0].mean(dim=0)
                            softmax_mean_per_head = e_flat_per_head.mean(dim=2).mean(dim=0)
                            softmax_stats_kblock0_this_rank = {
                                "max_per_head": softmax_max_per_head.detach().clone().cpu(),
                                "min_per_head": softmax_min_per_head.detach().clone().cpu(),
                                "mean_per_head": softmax_mean_per_head.detach().clone().cpu(),
                            }

                
                elif T == 0: # Handle case where local query length is 0 for the current rank
                    B_q, H_q, _, Dk_q = q_compute.shape # T_q is 0
                    # For i=0, k_len (length of the first k-block) is also 0.
                    # Scores shape (B, H, Tq=0, Tk_block=0)
                    # Probs shape (B, H, Tq=0, Tk_block=0)
                    empty_scores_kblock0 = torch.empty(B_q, H_q, 0, 0, device=q_compute.device, dtype=q_compute.dtype).cpu()
                    empty_probs_kblock0  = torch.empty(B_q, H_q, 0, 0, device=q_compute.device, dtype=q_compute.dtype).cpu()
                    
                    debug_dict_populate[f"{debug_key_prefix_populate}_sdp_scores_kblock0"] = empty_scores_kblock0
                    debug_dict_populate[f"{debug_key_prefix_populate}_sdp_probs_kblock0"]  = empty_probs_kblock0
                    
                    # contrib_den has shape (B, H, Tq=0, 1)
                    empty_contrib_den_kblock0 = torch.empty(B_q, H_q, 0, 1, device=den.device, dtype=den.dtype).cpu()
                    debug_dict_populate[f"{debug_key_prefix_populate}_contrib_den_kblock0"] = empty_contrib_den_kblock0

                    if self.rank == 0:
                        print(f"DEBUG HELPER (L{layer_idx} R{self.rank} i={i}) Populated EMPTY _sdp_scores_kblock0, _sdp_probs_kblock0, _contrib_den_kblock0 due to T=0.")
                        sys.stdout.flush()
                # If T == 0, s_minus_max_clamped and e are not defined for this block,
                # so clamping and softmax stats are not applicable / computed.

        # Populate overall debug info after the loop
        if can_populate_debug and debug_dict_populate is not None:
            debug_dict_populate[f"{debug_key_prefix_populate}_clamped_min_total_count"] = torch.tensor(total_clamped_min_count_this_rank, dtype=torch.long).cpu()
            debug_dict_populate[f"{debug_key_prefix_populate}_clamped_max_total_count"] = torch.tensor(total_clamped_max_count_this_rank, dtype=torch.long).cpu()
            if softmax_stats_kblock0_this_rank:
                 debug_dict_populate[f"{debug_key_prefix_populate}_softmax_stats_kblock0"] = softmax_stats_kblock0_this_rank
            debug_dict_populate[f"{debug_key_prefix_populate}_kahan_num_comp_norm"] = torch.linalg.norm(num_comp.float()).detach().clone().cpu()
            debug_dict_populate[f"{debug_key_prefix_populate}_kahan_den_comp_norm"] = torch.linalg.norm(den_comp.float()).detach().clone().cpu()

        return num, den

    def _compute_attention_scores(self, q, k, q_idx, k_idx, mask=None, apply_explicit_mask=True, apply_implicit_causal_mask=True,
                                  # Debug parameters for mask
                                  can_populate_debug_mask: bool = False,
                                  debug_dict_populate_mask: Optional[dict] = None,
                                  debug_key_prefix_populate_mask: str = "",
                                  layer_idx_mask: int = -1):
        B, H, Tq, D = q.shape
        Tk_block = k.shape[2] # Length of the current key block

        # Skip attention score computation if query length is zero (Tq == 0)
        if Tq == 0:
            # Scores shape would be (B, H, Tq=0, Tk_block)
            return torch.empty(B, H, 0, Tk_block, device=q.device, dtype=q.dtype)

        # Ensure scale is valid and non-zero
        assert self.scale > 0, f"Invalid attention scale: {self.scale}"

        scores = torch.matmul(q / self.scale, k.transpose(-2, -1))

        if apply_explicit_mask: # Control for the explicit `mask` (slice of global mask)
            if mask is not None:
                # Validate mask shape before applying
                assert mask.shape == scores.shape, f"Incompatible mask shape {mask.shape} vs scores shape {scores.shape}"
                if can_populate_debug_mask and debug_dict_populate_mask is not None and mask.numel() > 0:
                    debug_dict_populate_mask[f"{debug_key_prefix_populate_mask}_mask_slice_sum"] = mask.sum().detach().clone().cpu()
                    if mask.numel() < 1000:
                         debug_dict_populate_mask[f"{debug_key_prefix_populate_mask}_mask_slice_sample"] = mask.flatten()[:10].detach().clone().cpu()
                scores += mask.to(scores.dtype)
        
        if apply_implicit_causal_mask: # Control for the k_idx > q_idx based causal mask
                causal = k_idx[None, None, None, :] > q_idx[None, None, :, None]
                # Use float("-inf") for masking to align with SDPA's softmax nulling.
                scores = scores.masked_fill(causal, float("-inf"))
                if can_populate_debug_mask and debug_dict_populate_mask is not None:
                    debug_dict_populate_mask[f"{debug_key_prefix_populate_mask}_causal_mask_sum"] = causal.sum().cpu()
        
        if self.rank == 0 and Tq > 0 and can_populate_debug_mask: # Debug print for score slice
            print(f"DEBUG HELPER SCORES SLICE (L{layer_idx_mask} R{self.rank}):", scores[0, 0, 0, :5].tolist())
            sys.stdout.flush()
        
        # Removed clamp here: scores = scores.clamp(min=-80.0, max=80.0)
        # SDPA does not clamp scores at this stage. Clamping before exp in _compute_sums_pass is for stability.
        return scores

    def _ring_shift_tensor(self, tensor, pad_len, valid_len):
        rank, world = self.rank, self.world_size
        send, recv = (rank + 1) % world, (rank - 1 + world) % world
        device = tensor.device
        send_tensor = _pad_to_block(tensor[:, :, :valid_len], pad_len, -2).contiguous()
        recv_tensor = torch.empty_like(send_tensor)
        recv_len = torch.empty(1, dtype=torch.int32, device=device)

        ops = [
            P2POp(dist.isend, torch.tensor([valid_len], dtype=torch.int32, device=device), peer=send),
            P2POp(dist.irecv, recv_len, peer=recv),
            P2POp(dist.isend, send_tensor, peer=send),
            P2POp(dist.irecv, recv_tensor, peer=recv),
        ]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        r_len = recv_len.item()
        assert 0 <= r_len <= pad_len, f"Rank {rank}: invalid recv_len {r_len}"
        return recv_tensor[:, :, :r_len].contiguous(), r_len
