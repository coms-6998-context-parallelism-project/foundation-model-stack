import math
from typing import MutableMapping, Optional, Tuple

import torch


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
        dev_idx = device.index

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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies rotary position embedding to query and key tensors.

        Expects input tensors q, k to have shape [batch_size, seq_len, nheads, head_dim].

        Args:
            q: Query tensor.
            k: Key tensor.
            position_ids: Positions of tokens in the sequence, shape [batch_size, seq_len].
            past_kv_state: Optional tuple containing past keys and values for caching.
            use_cache: Flag indicating if caching is enabled.

        Returns:
            Tuple of rotated query and key tensors.
        """
        # Input validation: Ensure expected shape [B, S, H, D]
        assert len(q.size()) == 4, f"Expected q rank 4, got {q.ndim}"
        assert len(k.size()) == 4, f"Expected k rank 4, got {k.ndim}"
        assert q.size(1) == k.size(1), f"Seq len mismatch: q={q.size(1)}, k={k.size(1)}"

        # seq_len is dimension 1
        seq_len = q.size(1)

        # Compute position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=q.device, dtype=torch.long
            ).unsqueeze(0).repeat(k.size(0), 1)
            # Adjust position_ids based on cache length if applicable
            if use_cache and past_kv_state is not None and past_kv_state[0].numel() > 0:
                # Assuming past_kv_state[0] (keys) has shape [B, H, cache_len, D]
                # Or potentially [B, cache_len, H, D] if stored differently
                # Let's assume cache length is dim 2 if shape is [B, H, cache_len, D]
                # Need to confirm cache shape if this causes issues.
                # If cache shape is [B, cache_len, H, D], use past_kv_state[0].size(1)
                try:
                    cache_len = past_kv_state[0].size(2) # Assuming [B, H, cache_len, D]
                    position_ids += cache_len
                except IndexError:
                     rank = dist.get_rank() if dist.is_initialized() else 0
                     print(f"[WARN][rank{rank}] RoPE: Could not determine cache length from past_kv_state[0] shape {past_kv_state[0].shape}", flush=True)


        # Handle partial RoPE application
        if self.partial_rope != 1.0:
            q_rope = q[..., : self.dim]
            k_rope = k[..., : self.dim]
        else:
            q_rope = q
            k_rope = k

        # Cast the part to be rotated to float32 for calculation stability
        # Shape: [B, S, H, D_rope/2, 2]
        q_ = q_rope.float().view(*q_rope.shape[:-1], -1, 2)
        k_ = k_rope.float().view(*k_rope.shape[:-1], -1, 2)

        # Fetch cached frequencies
        # Ensure frequencies are computed up to the max position needed
        max_start_pos = torch.max(position_ids[:, 0]) if position_ids.numel() > 0 else 0
        # Use a consistent length (e.g., max_expected_seq_len) for caching key 'alpha'
        # This assumes compute_freqs_cis caches based on this length.
        self.compute_freqs_cis(q.device, self.max_seq_len)

        try:
            # Assuming a single alpha key exists per device after post_init
            alpha_key = next(iter(self.cached_freqs[q.device.index]))
            # Fetch frequencies using the computed position_ids
            # freqs shape: [B, S, D_rope/2, 2, 2]
            freqs = self.cached_freqs[q.device.index][alpha_key][position_ids]
        except (KeyError, StopIteration, IndexError) as e:
             rank = dist.get_rank() if dist.is_initialized() else 0
             print(f"[ERROR][rank{rank}] RoPE: Failed to fetch freqs! device.index={q.device.index}. Error: {e}", flush=True)
             # Return original q, k to avoid crashing, although output will be wrong.
             return q, k

        # Cast freqs to float32 for calculation
        freqs = freqs.float()

        # Apply rotation (simulated complex multiplication) in float32
        # Reshape freqs for broadcasting: [B, S, 1, D_rope/2, 2, 2]
        # to multiply with q_/k_ reshaped to [B, S, H, D_rope/2, 2, 1]
        freqs_reshaped = freqs.unsqueeze(2) # Add Head dimension

        # Perform rotation: ([B,S,1,D/2,2,2] * [B,S,H,D/2,2,1]).sum(-1) -> [B,S,H,D/2,2]
        q_rotated_float = (freqs_reshaped * q_.unsqueeze(-1)).sum(-1)
        k_rotated_float = (freqs_reshaped * k_.unsqueeze(-1)).sum(-1)

        # Flatten the last two dimensions: [B, S, H, D_rope]
        q_rotated_float = q_rotated_float.flatten(-2)
        k_rotated_float = k_rotated_float.flatten(-2)

        # Cast back to the original input type (e.g., fp16)
        q_out = q_rotated_float.type_as(q_rope)
        k_out = k_rotated_float.type_as(k_rope)

        # Concatenate back the non-rotated part if using partial RoPE
        if self.partial_rope != 1.0:
            # Ensure the non-rotated part has the same type before concatenating
            q_out = torch.cat([q_out, q[..., self.dim :].type_as(q_out)], dim=-1)
            k_out = torch.cat([k_out, k[..., self.dim :].type_as(k_out)], dim=-1)

        return q_out, k_out
