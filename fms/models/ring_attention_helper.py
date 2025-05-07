from typing import List, Tuple, Dict, Optional, Union, Any
from fms.modules.attention import MultiHeadAttention
import torch
import torch.distributed as dist
from torch.distributed import P2POp
import math
import torch.nn.functional as F
import torch.nn as nn
from fms.distributed.strategy import RingAttentionStrategy


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
                position_ids[:, :valid_len] = torch.arange(start, start + valid_len, device=x_norm.device)

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
            start,    # q_start_global
            # Pass debug parameters
            can_populate_debug=can_populate_debug,
            debug_dict_populate=debug_dict_populate,
            debug_key_prefix_populate=debug_key_prefix_populate
        )
        # out_valid is the final output of the block for this shard, after all residuals and FFN
        if can_populate_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_block_output"] = out_valid.detach().clone().cpu()

        return (_pad_to_block(out_valid, self.block_size, dim=1), None, None) if self.world_size > 1 else (out_valid, None, None)

    def forward_full(self, q, k, v, mask, valid_len_local_T, x_block_residual_input, q_start_global,
                     # Debug parameters
                     can_populate_debug: bool = False,
                     debug_dict_populate: Optional[dict] = None,
                     debug_key_prefix_populate: str = ""):
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

        # max_score will have shape (B, H, T, 1)
        max_score = self._compute_max_score_pass(q_compute, k_compute, mask, q_start_global, valid_len_local_T)
        # num will have shape (B, H, T, D_v) 
        # den will have shape (B, H, T, 1)
        num, denom = self._compute_sums_pass(
            q_compute, k_compute, v_compute, mask, q_start_global, valid_len_local_T, max_score, exp_min, exp_max
        )
        
        if T > 0:
            eps = torch.finfo(denom.dtype).eps
            attn = (num / (denom + eps)).to(q.dtype) # attn shape (B, H, T, D_v)

            # if self.attn.p_dropout and self.llama_block.training:
            #     attn = F.dropout(attn, p=self.attn.p_dropout, training=True)
            attn_reshaped = attn.transpose(1, 2).contiguous().view(B, T, H * D_v)
            attn_out = self.attn.dense(attn_reshaped) # attn_out shape (B, T, E)
        else: # T == 0
            attn_out = torch.empty(B, 0, self.attn.emb_dim, device=q.device, dtype=q.dtype)

        if can_populate_debug:
            debug_dict_populate[f"{debug_key_prefix_populate}_attn_out_raw"] = attn_out.detach().clone().cpu()

        # x_block_residual_input has shape (B, T, E). If T=0, it's (B, 0, E). This is the sharded input residual.
        x_after_attn_residual = x_block_residual_input + attn_out
        if can_populate_debug:
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
    def _compute_max_score_pass(self, q_compute, k_compute, mask_global, q_start_global, valid_len_local):
        B, H, T, _ = q_compute.shape
        dtype = q_compute.dtype
        dev = q_compute.device
        max_score = torch.full((B, H, T, 1), torch.finfo(dtype).min, device=dev, dtype=dtype)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=dev)
        k_blk, k_len = k_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = mask_global[:, :, q_start_global:q_start_global + T, k_start:k_start + k_len] if mask_global is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m)
                max_score = torch.maximum(max_score, s.amax(-1, keepdim=True))
            if i < self.world_size - 1:
                k_blk, k_len = self._ring_shift_tensor(k_blk, self.block_size, k_len)
        return max_score

    def _compute_sums_pass(
        self, q_compute, k_compute, v_compute, mask_global, q_start_global, valid_len_local, max_score, exp_min, exp_max
    ):
        B, H, T, Dv = q_compute.shape
        dev = q_compute.device
        dtype = self._accum_dtype
        num = torch.zeros(B, H, T, Dv, device=dev, dtype=dtype)
        den = torch.zeros(B, H, T, 1, device=dev, dtype=dtype)
        num_comp = torch.zeros_like(num)
        den_comp = torch.zeros_like(den)
        q_idx = torch.arange(q_start_global, q_start_global + T, device=dev)
        k_blk, v_blk, k_len = k_compute, v_compute, k_compute.shape[2]

        for i in range(self.world_size):
            k_start = ((self.rank - i + self.world_size) % self.world_size) * self.block_size
            k_idx = torch.arange(k_start, k_start + k_len, device=dev)
            m = mask_global[:, :, q_start_global:q_start_global + T, k_start:k_start + k_len] if mask_global is not None else None
            if k_len > 0:
                s = self._compute_attention_scores(q_compute, k_blk[:, :, :k_len], q_idx, k_idx, m)
                e = torch.exp((s - max_score).clamp(min=exp_min, max=exp_max))
                contrib_num = torch.einsum("bhqk,bhkd->bhqd", e, v_blk[:, :, :k_len])
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

        return num, den

    def _compute_attention_scores(self, q, k, q_idx, k_idx, mask=None, apply_mask=True, keep_causal=True):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        scores = torch.matmul(q / self.scale, k.transpose(-2, -1))

        if apply_mask:
            if mask is not None:
                scores += mask.to(scores.dtype)
            if keep_causal:
                causal = k_idx[None, None, None, :] > q_idx[None, None, :, None]
                scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
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
