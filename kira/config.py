"""Validated, serializable model and training configuration."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ModelConfig:
    vocab_size: int = 384
    d_model: int = 64
    num_layers: int = 2
    num_attention_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 192
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name in ('vocab_size', 'd_model', 'num_layers', 'num_attention_heads',
                     'num_kv_heads', 'intermediate_size', 'max_seq_len'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_model % self.num_attention_heads:
            raise ValueError('d_model must be divisible by num_attention_heads')
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError('query heads must be divisible by KV heads')
        if (self.d_model // self.num_attention_heads) % 2:
            raise ValueError('RoPE requires an even head dimension')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError('rope_theta must be finite and positive')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive')


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_steps: int = 100
    max_epochs: int = 100
    grad_clip: float = 1.0
    precision: str = 'auto'
    eval_interval: int = 10
    save_interval: int = 50
    seed: int = 42
    validation_fraction: float = 0.2

    def __post_init__(self):
        for name in ('batch_size', 'gradient_accumulation_steps', 'max_steps',
                     'max_epochs', 'eval_interval', 'save_interval'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError('warmup_steps must be in [0, max_steps)')
        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must be between 0 and 1')
        if self.precision not in ('auto', 'fp32', 'fp16', 'bf16'):
            raise ValueError('unsupported precision')
        for name in ('learning_rate', 'grad_clip'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
