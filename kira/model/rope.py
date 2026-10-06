import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.register_buffer('inv_freq', theta ** (-torch.arange(0, head_dim, 2).float() / head_dim), persistent=False)

    def forward(self, x, offset: int = 0):
        length = x.shape[-2]
        if offset < 0 or offset + length > self.max_seq_len:
            raise ValueError('RoPE positions exceed configured context')
        positions = torch.arange(offset, offset + length, device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.float())
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
