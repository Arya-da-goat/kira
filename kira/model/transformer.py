"""Pre-normalized decoder-only Transformer with GQA, RoPE, SwiGLU / MoE."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from kira.config import ModelConfig
from kira.model.attention import CausalSelfAttention
from kira.model.feed_forward import SwiGLU, SparseMoE
from kira.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)

        if config.ffn_type == 'moe':
            self.ffn = SparseMoE(config)
        else:
            self.ffn = SwiGLU(config.d_model, config.intermediate_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cache=None, use_cache: bool = False):
        # Pre-LN Causal Attention
        normed_x = self.attention_norm(x)
        attention, new_cache = self.attention(normed_x, cache, use_cache)
        x = x + self.dropout(attention)

        # Pre-LN Feed-Forward (SwiGLU or Sparse MoE)
        normed_ffn = self.ffn_norm(x)
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.config.ffn_type == 'moe':
            ffn_out, aux_loss = self.ffn(normed_ffn)
        else:
            ffn_out = self.ffn(normed_ffn)

        x = x + self.dropout(ffn_out)
        return x, new_cache, aux_loss


class KiraTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.last_aux_loss = torch.tensor(0.0)

        self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, cache=None, use_cache: bool = False,
                return_aux_loss: bool = False):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('input_ids must have shape [batch, nonempty sequence]')
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f'Input sequence length {input_ids.shape[1]} exceeds maximum context {self.config.max_seq_len}')
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError('Cache must contain one entry per layer')

        x = self.embedding(input_ids)
        caches = []
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)

        for index, layer in enumerate(self.layers):
            layer_cache_input = None if cache is None else cache[index]
            x, layer_cache, layer_aux_loss = layer(x, layer_cache_input, use_cache)
            if use_cache:
                caches.append(layer_cache)
            total_aux_loss = total_aux_loss + layer_aux_loss

        self.last_aux_loss = total_aux_loss
        logits = self.lm_head(self.norm(x))

        cache_result = caches if use_cache else None
        if return_aux_loss:
            return logits, cache_result, total_aux_loss
        return logits, cache_result


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean',
                    aux_loss: torch.Tensor | None = None) -> torch.Tensor:
    """Targets are shifted tokens; non-target positions use -100."""
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=-100,
        reduction=reduction
    )
    if aux_loss is not None and aux_loss > 0:
        return ce_loss + aux_loss
    return ce_loss
