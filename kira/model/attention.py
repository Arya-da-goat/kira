"""Causal Grouped Query Attention (GQA) with Rotary Position Embeddings and KV Cache."""
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
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_groups = self.num_attention_heads // self.num_kv_heads

        # Separate Q, K, V projections
        self.q_proj = nn.Linear(config.d_model, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Rotary Positional Embedding with optional context extension scaling
        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta,
            config.rope_scaling_type,
            config.rope_scaling_factor
        )

    def forward(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, length, _ = x.shape
        c = self.config

        # Project and reshape into heads
        q = self.q_proj(x).view(batch, length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Position offset for RoPE during autoregressive cached generation
        offset = 0 if cache is None else cache[0].shape[-2]
        q = self.rope(q, offset)
        k = self.rope(k, offset)

        # Update KV cache
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        new_cache = (k, v) if use_cache else None

        # Repeat KV heads for Grouped Query Attention (GQA)
        total_k_len = k.shape[-2]
        if self.num_groups > 1:
            keys = k.repeat_interleave(self.num_groups, dim=1)
            values = v.repeat_interleave(self.num_groups, dim=1)
        else:
            keys, values = k, v

        # Construct causal attention mask respecting KV cache offset
        query_positions = torch.arange(offset, offset + length, device=x.device)
        key_positions = torch.arange(total_k_len, device=x.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]

        # Use efficient scaled dot product attention
        attended = F.scaled_dot_product_attention(
            q, keys, values,
            attn_mask=causal_mask,
            dropout_p=c.dropout if self.training else 0.0
        )

        output = attended.transpose(1, 2).contiguous().view(batch, length, c.d_model)
        return self.out_proj(output), new_cache
