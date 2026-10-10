"""Direct Preference Optimization (DPO) for aligning Kira with pairwise preferences."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, label_pad_token_id: int = -100) -> torch.Tensor:
    """Calculates sequence log probabilities for tokens where labels != label_pad_token_id."""
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_mask = (shift_labels != label_pad_token_id)
    # Clamp shifted labels to 0 for cross_entropy gathering, masked out later
    gather_labels = shift_labels.clamp(min=0)
    per_token_logps = torch.gather(
        shift_logits.log_softmax(-1), dim=-1, index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)

    return (per_token_logps * loss_mask).sum(-1)


def dpo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             reference_chosen_logps: torch.Tensor,
             reference_rejected_logps: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates DPO loss and implicit rewards for chosen and rejected responses.
    
    L_DPO = -E[log(sigmoid(beta * (log(pi(yw)/ref(yw)) - log(pi(yl)/ref(yl)))))]
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class DPOTrainer:
    """Trainer for preference optimization on (prompt, chosen, rejected) triplets."""
    def __init__(self, model, reference_model, optimizer, beta: float = 0.1):
        self.model = model
        self.reference_model = reference_model
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.optimizer = optimizer
        self.beta = beta

    def train_step(self, chosen_ids: torch.Tensor, chosen_labels: torch.Tensor,
                   rejected_ids: torch.Tensor, rejected_labels: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # Policy forward pass
        policy_chosen_logits, _ = self.model(chosen_ids)
        policy_rejected_logits, _ = self.model(rejected_ids)

        policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels)

        # Reference model forward pass (no gradients)
        with torch.no_grad():
            ref_chosen_logits, _ = self.reference_model(chosen_ids)
            ref_rejected_logits, _ = self.reference_model(rejected_ids)
            ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
            ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)

        loss, chosen_r, rejected_r = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=self.beta
        )

        loss.backward()
        self.optimizer.step()

        return {
            'dpo_loss': float(loss.item()),
            'chosen_reward': float(chosen_r.item()),
            'rejected_reward': float(rejected_r.item()),
            'reward_margin': float((chosen_r - rejected_r).item())
        }
