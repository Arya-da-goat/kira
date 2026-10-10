"""Rotary Positional Embedding (RoPE) with optional context extension scaling."""
from __future__ import annotations

import math
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0,
                 scaling_type: str | None = None, scaling_factor: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

        # NTK-aware scaling adjusts base theta directly
        effective_theta = theta
        if scaling_type == 'ntk' and scaling_factor > 1.0:
            effective_theta = theta * (scaling_factor ** (head_dim / (head_dim - 2)))

        dim_indices = torch.arange(0, head_dim, 2).float()
        inv_freq = 1.0 / (effective_theta ** (dim_indices / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        length = x.shape[-2]
        if offset < 0 or offset + length > self.max_seq_len:
            raise ValueError(f'RoPE positions (offset {offset} + length {length}) exceed configured context {self.max_seq_len}')
        
        positions = torch.arange(offset, offset + length, device=x.device, dtype=torch.float32)
        if self.scaling_type == 'linear' and self.scaling_factor > 1.0:
            positions = positions / self.scaling_factor

        angles = torch.outer(positions, self.inv_freq.float())
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        
        # Apply rotation to adjacent pairs
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        return rotated
