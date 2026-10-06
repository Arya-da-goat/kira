from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from kira.config import ModelConfig
from kira.model.attention import CausalSelfAttention
from kira.model.feed_forward import SwiGLU
from kira.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.intermediate_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cache=None, use_cache=False):
        attention, cache = self.attention(self.attention_norm(x), cache, use_cache)
        x = x + self.dropout(attention)
        return x + self.dropout(self.ffn(self.ffn_norm(x))), cache


class KiraTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, cache=None, use_cache: bool = False):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('input_ids must have shape [batch, nonempty sequence]')
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError('Cache must contain one entry per layer')
        x = self.embedding(input_ids)
        caches = []
        for index, layer in enumerate(self.layers):
            x, layer_cache = layer(x, None if cache is None else cache[index], use_cache)
            if use_cache:
                caches.append(layer_cache)
        return self.lm_head(self.norm(x)), caches if use_cache else None


def next_token_loss(logits, targets, reduction: str = 'mean'):
    """Targets are already shifted by the dataset; padded positions use -100."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           targets.reshape(-1), ignore_index=-100, reduction=reduction)
