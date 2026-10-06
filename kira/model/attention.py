"""Causal attention with contiguous GQA groups and compact, rotated KV caches."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from kira.config import ModelConfig
from kira.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.num_attention_heads
        self.q_proj = nn.Linear(config.d_model, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x, cache=None, use_cache: bool = False):
        batch, length, _ = x.shape
        c = self.config
        q = self.q_proj(x).view(batch, length, c.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, c.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, c.num_kv_heads, self.head_dim).transpose(1, 2)
        offset = 0 if cache is None else cache[0].shape[-2]
        q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=-2), torch.cat((cache[1], v), dim=-2)
        new_cache = (k, v) if use_cache else None
        groups = c.num_attention_heads // c.num_kv_heads
        keys, values = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        # Explicit absolute-position mask also handles multi-token cached decoding.
        query_positions = torch.arange(offset, offset + length, device=x.device)
        key_positions = torch.arange(k.shape[-2], device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        attended = F.scaled_dot_product_attention(q, keys, values, attn_mask=allowed,
                                                  dropout_p=c.dropout if self.training else 0.0)
        output = attended.transpose(1, 2).contiguous().view(batch, length, c.d_model)
        return self.out_proj(output), new_cache
