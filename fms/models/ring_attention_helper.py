import torch
import torch.nn as nn

class RingAttentionHelper(nn.Module):
    def __init__(self, attn_module: nn.Module, norm_layer: nn.Module):
        super().__init__()
        self.attn = attn_module  # e.g. MultiHeadAttention
        self.ln = norm_layer     # LayerNorm for input

    def compute_local_qkv_and_rope(
        self,
        q,
        k=None,
        v=None,
        position_ids=None,
        use_cache=False,
        past_key_value_state=None,
        is_self=True,
    ):
        B, T, _ = q.shape
        q_out, k_out, v_out = self.attn.in_proj(q, k, v)

        queries = q_out.view(B, T, self.attn.nheads, self.attn.emb_kq_per_head)
        keys = k_out.view(B, T, self.attn.kvheads, self.attn.emb_kq_per_head)
        values = v_out.view(B, T, self.attn.kvheads, self.attn.emb_v_per_head)

        if self.attn.position_encoder is not None:
            if position_ids is None:
                position_ids = torch.arange(T, device=q.device).unsqueeze(0).expand(B, -1)
            queries, keys = self.attn.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        return queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

    def forward(
        self,
        x,
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        is_causal_mask=False,
    ):
        x_ln = self.ln(x)

        queries, keys, values = self.compute_local_qkv_and_rope(
            q=x_ln,
            k=x_ln,
            v=x_ln,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value_state=past_key_value_state,
            is_self=True,
        )

        if use_cache and past_key_value_state is not None and past_key_value_state[0].numel() > 0:
            keys = torch.cat((past_key_value_state[0], keys), dim=2)
            values = torch.cat((past_key_value_state[1], values), dim=2)

        expansion = self.attn.nheads // self.attn.kvheads
        if expansion != 1:
            keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values_e = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        else:
            keys_e = keys
            values_e = values

        # NOTE: attention score computation (q @ k.T), masking, softmax, and output projection would go here.
        # This is just up to the expansion and preparation stage.
        return queries, keys_e, values_e
