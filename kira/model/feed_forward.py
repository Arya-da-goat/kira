"""Feed-forward layers: Dense SwiGLU and Sparse Mixture-of-Experts (MoE)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from kira.config import ModelConfig


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit feed-forward network."""
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.gate = nn.Linear(d_model, intermediate_size, bias=False)
        self.up = nn.Linear(d_model, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    """Genuinely sparse Mixture-of-Experts with top-k gating and auxiliary load-balancing loss."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        self.router = nn.Linear(config.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(config.d_model, config.intermediate_size)
            for _ in range(self.num_experts)
        ])
        self.last_metrics: dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)  # [N, d_model]
        num_tokens = flat_x.shape[0]

        # Learned router logits and normalized probabilities
        router_logits = self.router(flat_x)  # [N, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        # Normalize weights among selected top-k
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Genuinely sparse computation: dispatch only routed tokens to each expert
        output = torch.zeros_like(flat_x)
        dispatch_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)

        for expert_idx in range(self.num_experts):
            # Mask of tokens where expert_idx was selected in top-k
            token_mask = (topk_indices == expert_idx)  # [N, top_k]
            if not token_mask.any():
                continue

            # Row indices of tokens routed to this expert
            row_indices, k_positions = torch.where(token_mask)
            selected_tokens = flat_x[row_indices]
            weights = topk_weights[row_indices, k_positions].unsqueeze(-1)

            # Evaluate expert ONLY on dispatched tokens
            expert_out = self.experts[expert_idx](selected_tokens)
            output.index_add_(0, row_indices, expert_out * weights)

            dispatch_counts[expert_idx] = float(row_indices.numel())

        # Load-balancing auxiliary loss (GShard / Switch Transformer formulation)
        # P_e = fraction of router probability assigned to expert e
        # f_e = fraction of total top-k assignments dispatched to expert e
        total_assignments = num_tokens * self.top_k
        fraction_dispatched = dispatch_counts / max(total_assignments, 1)
        mean_router_prob = router_probs.mean(dim=0)
        aux_loss = self.aux_loss_coeff * self.num_experts * torch.sum(fraction_dispatched * mean_router_prob)

        # Record metrics for transparency
        with torch.no_grad():
            self.last_metrics = {
                f'expert_{i}_utilization': float(fraction_dispatched[i].item())
                for i in range(self.num_experts)
            }
            self.last_metrics['aux_loss'] = float(aux_loss.item())

        return output.view(batch_size, seq_len, d_model), aux_loss
