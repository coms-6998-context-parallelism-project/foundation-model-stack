import math
from typing import MutableMapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fms import distributed # Import for rank_and_world in fallback


class PositionEncoder:
    """
    Provides the ability to insert position-encoding logic into MHA.
    """

    # Override to adjust the mask e.g. for Alibi
    def adjusted_mask(
        self,
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache=False,
    ) -> Optional[torch.Tensor]:
        return mask

    # Override to adjust q/k's e.g. for rotary embeddings
    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return q, k


class Alibi(PositionEncoder):
    """
    Attention Linear Bias layer for sequence models, as in https://arxiv.org/pdf/2108.12409.pdf.
    ...
    Args
    ----
    nheads : int
        Number of attention heads (and thus position bias matrices)
    max_scale : float
        Maximum scaling factor. Defaults to 0.5 as in paper.
    min_scale : float
        Minimum scaling factor. Defaults to 2^-8 as in paper.
    """

    def __init__(self, nheads, max_scale=0.5, min_scale=1 / (2**8)):
        super(Alibi, self).__init__()
        self.nheads = nheads
        start = math.log2(max_scale)
        end = math.log2(min_scale)
        self.scales = (
            2
            ** torch.arange(
                start, end + 1e-6 * math.sign(end - start), (end - start) / (nheads - 1)
            ).view(1, nheads, 1, 1),
        )

    def adjusted_mask(
        self,
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache=False,
    ) -> Optional[torch.Tensor]:
        qlen = q.size(1)
        klen = k.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_kv_state is not None and past_kv_state[0] is not None:
            klen += past_kv_state[0][0].size(-2)
            qlen += past_kv_state[0][1].size(-2)

        # Automatically allocates on chosen cuda
        device = self.scales.device
        q_pos = torch.arange(qlen, dtype=torch.long, device=device)
        k_pos = torch.arange(klen, dtype=torch.long, device=device)

        # rel_pos: qlen x klen
        rel_pos = k_pos[None, :] - q_pos[:, None]
        values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)

        bias = values * self.scales

        # we need to pick the k-length row of alibi maxtrix when caching is being used and not first iteration
        if use_cache and klen != 1 and qlen == 1:
            bias = bias[:, :, -1:, :]

        attn_mask = bias
        # We expect the shapes of mask and rel_pos_bias to be at least broadcastable
        if mask is not None:
            # Can't do in-place op in case broadcast makes attn_mask bigger
            attn_mask = attn_mask.masked_fill(mask == 0, float("-inf"))

        return attn_mask


class RotaryEmbedding(PositionEncoder):
    def __init__(
        self,
        dim: int,
        ratio: float = 10_000.0,
        max_seq_len=2048,
        ntk_scaling=False,
        partial_rope=1.0,
    ):
        """
        This implementation of Rotary Position Embeddings (RoPE) avoids
        complex numbers, and so can be used with torch.compile.

        https://arxiv.org/abs/2104.09864

        ...
        Args
        ----
        dim : int
            Per-head embedding dimension
        max_seq_len : int
            Maximum expected sequence length for the model, if exceeded the cached freqs will be recomputed
        ratio: int
            The ratio for the geometric progression to compute the rotation angles
        partial_rope: int
            fraction of head dimension to apply rope to
        """
        super(RotaryEmbedding, self).__init__()
        self.partial_rope = partial_rope
        self.dim = int(partial_rope * dim)
        self.ratio = ratio
        self.cached_freqs: MutableMapping[int, MutableMapping[int, torch.Tensor]] = {}
        self.max_seq_len_cached: MutableMapping[int, int] = {}
        self.ntk_scaling = ntk_scaling
        self.max_seq_len = max_seq_len

        # --- Prompt 17: Flag to bypass RoPE ---
        self.bypass_rope = False # Set to True to temporarily disable RoPE
        self.debug_ring = False # Add a debug flag, ideally set from config/args

    def _alpha(self, seq_len) -> int:
        if not self.ntk_scaling:
            return 1
        else:
            alpha = seq_len / self.max_seq_len
            alpha = math.ceil(alpha)
            # for some reason math.log2 didn't `torch.compile` but
            # `math.log` does
            alpha = math.log(alpha) / math.log(2)
            alpha = math.ceil(alpha)
            alpha = 2**alpha
            alpha = int(alpha)
            return alpha

    def compute_freqs_cis(self, device, max_seq_len=2048):
        # NTK scaling.
        # https://arxiv.org/abs/2306.15595
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        #
        # we'll store the freqs for each alpha value. This means that for
        # shorter sequences, we preserve the original scale.
        # To limit the number of multiples to store we'll maintain alphas for
        # `2**i` where i is the ratio of actual vs initial max seq len. (i.e. 2,
        # 4, 8, ... as needed)
        alpha = self._alpha(max_seq_len)
        dev_idx = device.index if device.type == 'cuda' else 0 # Handle CPU case

        if dev_idx not in self.cached_freqs:
            self.cached_freqs[dev_idx] = {}
        if dev_idx not in self.max_seq_len_cached:
            self.max_seq_len_cached[dev_idx] = 0

        # This condition can be combined with the model using Rotary calling this method
        # on model init when device is known to avoid a graph break (see llama.py)
        if self.ntk_scaling:
            max_seq_len = max(max_seq_len, self.max_seq_len * alpha)
        else:
            if self.max_seq_len_cached[dev_idx] > 0:
                return alpha
            max_seq_len = max(max_seq_len, self.max_seq_len)

        if (
            alpha in self.cached_freqs[dev_idx]
            and max_seq_len <= self.max_seq_len_cached[dev_idx]
        ):
            return alpha

        ratio = self.ratio
        dim = self.dim

        if self.ntk_scaling:
            ratio = ratio * alpha ** (dim / (dim - 2))

        freqs = 1.0 / (
            ratio
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )

        t = torch.arange(max_seq_len, device=device, dtype=freqs.dtype)
        freqs = torch.outer(t, freqs).float()
        self.max_seq_len_cached[dev_idx] = max_seq_len
        self.cached_freqs[dev_idx][alpha] = torch.stack(
            [
                torch.cos(freqs),
                -torch.sin(freqs),
                torch.sin(freqs),
                torch.cos(freqs),
            ],
            dim=2,
        ).view(*freqs.size(), 2, 2)

        return alpha

    def reshape_for_broadcast(self, x: torch.Tensor, cur_freqs):
        ndim = x.ndim
        assert 1 < ndim, ndim
        assert cur_freqs.size()[:2] == (
            x.size(2),
            x.size(-2),
        ), f"for {cur_freqs.size()} and {x.size()}"
        shape = [d if i == 2 or i >= ndim - 2 else 1 for i, d in enumerate(x.size())]
        return cur_freqs.view(*shape, 2)

    def adjusted_qk(
        self, q, k, position_ids=None, past_kv_state=None, use_cache=False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        --- Prompt 17: Check bypass flag ---
        """
        if self.bypass_rope: # Keep existing bypass
            return q, k
        """
        Applies rotary position embedding to query and key tensors.

        Expects input tensors q, k to have shape [batch_size, seq_len, nheads, head_dim].

        Args:
            q: Query tensor.
            k: Key tensor.
            position_ids: Positions of tokens in the sequence, shape [batch_size, seq_len].
                          MUST represent GLOBAL positions if used in Ring Attention context.
                          If None, attempts to calculate based on seq_len and cache.
            past_kv_state: Optional tuple containing past keys and values for caching.
            use_cache: Flag indicating if caching is enabled.

        Returns:
            Tuple of rotated query and key tensors.
        """
        # Debug: Print input arguments and shapes
        rank, _ = distributed.rank_and_world() # Use default group
        if self.debug_ring:
            print(f"[RANK {rank} RoPE] adjusted_qk: q={q.shape}, k={k.shape}, pos_ids provided={position_ids is not None}, use_cache={use_cache}", flush=True)
        # Input validation: Ensure expected shape [B, S, H, D]
        assert len(q.size()) == 4
        assert len(k.size()) == 4
        batch_size = q.size(0)
        # Allow different seq lengths for Q and K, common in ring attention cross-blocks
        # assert q.size(1) == k.size(1), f"Seq len mismatch: q={q.size(1)}, k={k.size(1)}"

        # Compute position_ids if not provided (Fallback for standard attention)
        seq_len = max(k.size(1), q.size(1))
        q_seq_len = q.size(1) # Store original q_seq_len for clarity
        if position_ids is None:
            if self.debug_ring:
                print(f"[RANK {rank} RoPE] position_ids is None, using fallback logic.", flush=True)
            # Fallback for standard attention (non-ring)
            # Check if we are in decoding phase (use_cache=True and past state exists)
            cache_len = -1 # Initialize cache_len
            is_decode_phase = use_cache and past_kv_state is not None and past_kv_state[0].numel() > 0
            if is_decode_phase:
                # Single token decoding: position is the cache length
                try:
                    # Assuming cache shape [B, H, cache_len, D]
                    cache_len = past_kv_state[0].size(2)
                    position_ids = torch.full((q.size(0), 1), cache_len, dtype=torch.long, device=q.device)
                    # --- Prompt 15: Check fallback decode position_ids ---
                    assert position_ids.shape == (batch_size, 1), f"Fallback decode position_ids shape mismatch: expected ({batch_size}, 1), got {position_ids.shape}"
                    assert position_ids[0, 0].item() == cache_len, f"Fallback decode position_ids value mismatch: expected {cache_len}, got {position_ids[0, 0].item()}"
                    if self.debug_ring:
                        print(f"[RANK {rank} RoPE] Fallback: Decode phase. cache_len={cache_len}. Calculated position_ids={position_ids.flatten().tolist()}", flush=True)
                except IndexError:
                     if self.debug_ring: print(f"[WARN RoPE rank{rank}] Fallback failed to get cache length. Shape: {past_kv_state[0].shape}", flush=True)
                     # Fallback further to just sequence length (likely incorrect for decode)
                     position_ids = torch.arange(0, q_seq_len, dtype=torch.long, device=q.device).unsqueeze(0)
                     # --- Prompt 15: Check fallback decode (error case) position_ids ---
                     # This case is less critical as it's likely wrong anyway, but check shape
                     assert position_ids.shape[1] == q_seq_len, f"Fallback decode (error) position_ids seq len mismatch: expected {q_seq_len}, got {position_ids.shape[1]}"
                     if self.debug_ring:
                         print(f"[WARN RoPE rank{rank}] Fallback: Decode phase (IndexError). Using arange({q_seq_len}). Calculated position_ids={position_ids.flatten().tolist()}", flush=True)
            else:
                # Prefill phase: positions are 0 to seq_len-1
                # Note: This simple arange doesn't account for left-padding.
                # The caller should ideally provide position_ids computed correctly for padding.
                position_ids = torch.arange(0, q_seq_len, dtype=torch.long, device=q.device).unsqueeze(0)
                # --- Prompt 15: Check fallback prefill position_ids ---
                assert position_ids.shape == (1, q_seq_len), f"Fallback prefill position_ids shape mismatch: expected (1, {q_seq_len}), got {position_ids.shape}"
                if q_seq_len > 0:
                    expected_vals = torch.arange(0, q_seq_len, device=q.device)
                    assert torch.equal(position_ids.squeeze(0), expected_vals), f"Fallback prefill position_ids value mismatch: expected {expected_vals.tolist()}, got {position_ids.squeeze(0).tolist()}"
                if self.debug_ring:
                    print(f"[RANK {rank} RoPE] Fallback: Prefill phase. Using arange({q_seq_len}). Calculated position_ids={position_ids.flatten().tolist()}", flush=True)

            # Expand batch dimension if needed (should match q batch size)
            if position_ids.size(0) != q.size(0):
                 position_ids = position_ids.expand(q.size(0), -1)
        else:
            if self.debug_ring: print(f"[RANK {rank} RoPE] Received position_ids (shape {position_ids.shape}): {position_ids.flatten().tolist()}", flush=True)
            # print(f"[DEBUG RoPE rank{rank}] Received position_ids (shape {position_ids.shape}): {position_ids.flatten().tolist()}", flush=True) # Removed

        if self.partial_rope != 1.0:
            q_rope = q[..., : self.dim]
            k_rope = k[..., : self.dim]
        else:
            q_rope = q # B S H D
            k_rope = k # B S H D

        # Cast the part to be rotated to float32 for calculation stability
        orig_dtype = q_rope.dtype # Store original dtype
        # Shape: [B, S, H, D_rope/2, 2]
        q_ = q_rope.float().view(*q_rope.shape[:-1], -1, 2)
        k_ = k_rope.float().view(*k_rope.shape[:-1], -1, 2)

        # Fetch cached frequencies
        # Ensure frequencies are computed up to the max position needed.
        max_start_pos = torch.max(position_ids[:, 0]) if position_ids.numel() > 0 else 0
        # Use max position ID to determine required cache length
        max_pos_id_needed = torch.max(position_ids).item() if position_ids.numel() > 0 else 0
        if self.debug_ring:
            print(f"[RANK {rank} RoPE] Max position ID needed: {max_pos_id_needed}", flush=True)
        alpha = self.compute_freqs_cis(q.device, max_pos_id_needed + 1)
        dev_idx = q.device.index if q.device.type == 'cuda' else 0 # Handle CPU case
        if self.debug_ring:
            print(f"[RANK {rank} RoPE] Max seq len cached for device {dev_idx}: {self.max_seq_len_cached.get(dev_idx, 0)}", flush=True)
        assert self.max_seq_len_cached.get(dev_idx, 0) > max_pos_id_needed, \
            f"RoPE cache size insufficient: max needed={max_pos_id_needed}, cache size={self.max_seq_len_cached.get(dev_idx, 0)}"
        # print(f"[DEBUG RoPE rank{rank}] Using alpha={alpha}. Cache size for alpha: {self.cached_freqs.get(q.device.index, {}).get(alpha, torch.empty(0)).shape[0]}", flush=True) # Less noisy

        # Directly index cache with the provided (global) position_ids
        try:
            freqs = self.cached_freqs[dev_idx][alpha][position_ids].float() # Shape [B, S_pos, D/2, 2, 2]
            # --- Prompt 16: Check frequency indexing shape ---
            assert freqs.size(0) == batch_size, f"Freqs batch size mismatch: expected {batch_size}, got {freqs.size(0)}"
            assert freqs.size(1) == position_ids.size(1), f"Freqs seq len mismatch: expected {position_ids.size(1)} (from position_ids), got {freqs.size(1)}"
            if self.debug_ring:
                print(f"[RANK {rank} RoPE] Successfully indexed freqs cache. freqs.shape={freqs.shape}", flush=True)
        except IndexError as e:
            print(f"[ERROR RoPE rank{rank}] Failed to index freqs cache. Max requested pos: {max_pos_id_needed}, Cache size: {self.cached_freqs[dev_idx][alpha].shape[0]}. Error: {e}", flush=True) # Keep critical error
            # Fallback or re-raise depending on desired behavior
            raise e

        freqs = freqs.float()

        # Apply rotation (simulated complex multiplication) in float32
        q_len = q.size(1)
        k_len = k.size(1)
        # Assume freqs has shape [B, S_pos, D/2, 2, 2] where S_pos >= max(q_len, k_len)
        # Slice directly based on Q and K lengths.
        # Note: This slicing assumes position_ids correspond 1:1 with the sequence dimension.
        # If position_ids are arbitrary (e.g., due to padding or complex caching), this might be wrong.
        freqs_q = freqs[:, :q_len, ...] # [B, q_len, D/2, 2, 2]
        freqs_k = freqs[:, :k_len, ...] # [B, k_len, D/2, 2, 2]
        # --- Prompt 16: Check sliced frequency shape ---
        assert freqs_q.size(1) == q_len, f"Sliced freqs_q seq len mismatch: expected {q_len}, got {freqs_q.size(1)}"
        assert freqs_k.size(1) == k_len, f"Sliced freqs_k seq len mismatch: expected {k_len}, got {freqs_k.size(1)}"
        if self.debug_ring:
            print(f"[RANK {rank} RoPE] Sliced freqs_q.shape={freqs_q.shape}, freqs_k.shape={freqs_k.shape}", flush=True)

        # --- Use Alternative RoPE Math ---
        # Original FMS math commented out:
        # freqs_q_reshaped = freqs_q.unsqueeze(2) # [B, q_len, 1, D/2, 2, 2]
        # freqs_k_reshaped = freqs_k.unsqueeze(2) # [B, k_len, 1, D/2, 2, 2]
        # q_rotated_float = (freqs_q_reshaped * q_.unsqueeze(-1)).sum(-1)
        # k_rotated_float = (freqs_k_reshaped * k_.unsqueeze(-1)).sum(-1)

        # Apply Alternative math (using FMS's freqs_q/freqs_k slicing)
        # Note: Alternative used unsqueeze(-2) and sum(5). Adjusting for FMS tensor shapes.
        q_rotated_float = (freqs_q.unsqueeze(2) * q_.unsqueeze(-2)).sum(5) # [B,S,H,D/2,2]
        k_rotated_float = (freqs_k.unsqueeze(2) * k_.unsqueeze(-2)).sum(5) # [B,S,H,D/2,2]

        # Flatten the last two dimensions: [B, S, H, D_rope]
        q_rotated_float = q_rotated_float.flatten(-2)
        k_rotated_float = k_rotated_float.flatten(-2)

        # Cast back to the original input type (e.g., fp16)
        q_out = q_rotated_float.to(orig_dtype)
        k_out = k_rotated_float.to(orig_dtype)

        if self.debug_ring:
            print(f"[RANK {rank} RoPE] adjusted_qk complete. q_out={q_out.shape}, k_out={k_out.shape}", flush=True)

        if self.partial_rope != 1.0:
            # Ensure the non-rotated part has the same type before concatenating
            q_out = torch.cat([q_out, q[..., self.dim :].type_as(q_out)], dim=-1)
            k_out = torch.cat([k_out, k[..., self.dim :].type_as(k_out)], dim=-1)
        return q_out, k_out
