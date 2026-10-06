from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(d_model, intermediate_size, bias=False)
        self.up = nn.Linear(d_model, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
