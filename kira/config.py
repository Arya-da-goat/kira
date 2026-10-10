"""Validated, serializable model and training configuration for Kira Frontier Architecture."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Literal, Optional


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
    rope_scaling_type: Optional[str] = None  # None, 'linear', 'ntk'
    rope_scaling_factor: float = 1.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    ffn_type: Literal['swiglu', 'moe'] = 'swiglu'
    num_experts: int = 4
    moe_top_k: int = 2
    moe_aux_loss_coeff: float = 0.01
    tie_word_embeddings: bool = True

    def __post_init__(self):
        for name in ('vocab_size', 'd_model', 'num_layers', 'num_attention_heads',
                     'num_kv_heads', 'intermediate_size', 'max_seq_len'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_model % self.num_attention_heads != 0:
            raise ValueError('d_model must be divisible by num_attention_heads')
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError('query heads must be divisible by KV heads')
        if (self.d_model // self.num_attention_heads) % 2 != 0:
            raise ValueError('RoPE requires an even head dimension')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError('rope_theta must be finite and positive')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive')
        if self.ffn_type not in ('swiglu', 'moe'):
            raise ValueError("ffn_type must be either 'swiglu' or 'moe'")
        if self.ffn_type == 'moe':
            if type(self.num_experts) is not int or self.num_experts < 2:
                raise ValueError('num_experts must be an integer >= 2')
            if type(self.moe_top_k) is not int or self.moe_top_k < 1 or self.moe_top_k > self.num_experts:
                raise ValueError('moe_top_k must be between 1 and num_experts')
            if not math.isfinite(self.moe_aux_loss_coeff) or self.moe_aux_loss_coeff < 0:
                raise ValueError('moe_aux_loss_coeff must be finite and non-negative')
        if self.rope_scaling_type is not None:
            if self.rope_scaling_type not in ('linear', 'ntk'):
                raise ValueError("rope_scaling_type must be None, 'linear', or 'ntk'")
            if not math.isfinite(self.rope_scaling_factor) or self.rope_scaling_factor <= 0:
                raise ValueError('rope_scaling_factor must be positive')

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_attention_heads

    def parameter_breakdown(self) -> dict[str, int]:
        head_dim = self.head_dim
        embedding_params = self.vocab_size * self.d_model
        
        # Per attention layer
        q_params = self.d_model * (self.num_attention_heads * head_dim)
        k_params = self.d_model * (self.num_kv_heads * head_dim)
        v_params = self.d_model * (self.num_kv_heads * head_dim)
        o_params = (self.num_attention_heads * head_dim) * self.d_model
        attn_norm_params = self.d_model
        per_layer_attn = q_params + k_params + v_params + o_params + attn_norm_params

        # Per FFN layer
        ffn_norm_params = self.d_model
        if self.ffn_type == 'swiglu':
            per_expert_swiglu = 3 * self.d_model * self.intermediate_size
            per_layer_ffn = per_expert_swiglu + ffn_norm_params
            router_params = 0
            expert_params = per_expert_swiglu * self.num_layers
        else:
            router_params = self.d_model * self.num_experts
            per_expert_swiglu = 3 * self.d_model * self.intermediate_size
            total_moe_ffn = router_params + (self.num_experts * per_expert_swiglu)
            per_layer_ffn = total_moe_ffn + ffn_norm_params
            expert_params = (self.num_experts * per_expert_swiglu) * self.num_layers

        total_layers = self.num_layers * (per_layer_attn + per_layer_ffn)
        final_norm_params = self.d_model
        lm_head_params = 0 if self.tie_word_embeddings else (self.d_model * self.vocab_size)
        total = embedding_params + total_layers + final_norm_params + lm_head_params

        return {
            'embedding': embedding_params,
            'attention_total': (per_layer_attn - attn_norm_params) * self.num_layers,
            'ffn_total': expert_params + (router_params * self.num_layers),
            'normalization': (2 * self.num_layers + 1) * self.d_model,
            'lm_head': lm_head_params,
            'tied_embeddings': self.tie_word_embeddings,
            'total_parameters': total,
        }

    def estimate_memory_bytes(self, batch_size: int = 1, seq_len: int = 128, precision: str = 'fp32') -> dict[str, int]:
        bytes_per_param = 4 if precision == 'fp32' else 2
        params = self.parameter_breakdown()['total_parameters']
        weights_bytes = params * bytes_per_param
        # KV cache per token: 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
        kv_cache_per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim * bytes_per_param
        kv_cache_bytes = batch_size * seq_len * kv_cache_per_token
        # Rough activation memory estimation: hidden activations per layer
        activations_per_token = self.num_layers * self.d_model * bytes_per_param * 10
        activation_bytes = batch_size * seq_len * activations_per_token
        return {
            'weights_bytes': weights_bytes,
            'kv_cache_bytes': kv_cache_bytes,
            'activation_bytes': activation_bytes,
            'total_estimated_bytes': weights_bytes + kv_cache_bytes + activation_bytes
        }


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.001
    min_learning_rate: float = 0.00001
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
    scheduler_type: str = 'cosine'  # 'cosine' or 'linear'

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
        if not math.isfinite(self.min_learning_rate) or self.min_learning_rate < 0:
            raise ValueError('min_learning_rate must be finite and nonnegative')
