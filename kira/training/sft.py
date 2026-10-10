"""Supervised Instruction Fine-Tuning (SFT) with assistant-only loss masking."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from kira.model.transformer import next_token_loss


class InstructionDataset(Dataset):
    """Dataset of multi-turn conversations tokenized with assistant-only loss masking."""
    def __init__(self, conversations: list[list[dict[str, str]]], tokenizer, max_seq_len: int = 128):
        self.samples = []
        for conv in conversations:
            input_ids, labels = tokenizer.encode_chat(conv, max_length=max_seq_len)
            if len(input_ids) < 2:
                continue
            # Shift for autoregressive prediction: x = tokens[:-1], y = labels[1:]
            x = torch.tensor(input_ids[:-1], dtype=torch.long)
            y = torch.tensor(labels[1:], dtype=torch.long)
            # Pad to max_seq_len - 1 if desired, or handle variable length
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sft_collate_fn(batch, pad_token_id: int = 0):
    max_len = max(x.shape[0] for x, _ in batch)
    padded_x = []
    padded_y = []
    for x, y in batch:
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            x_padded = torch.cat([x, torch.full((pad_size,), pad_token_id, dtype=torch.long)])
            y_padded = torch.cat([y, torch.full((pad_size,), -100, dtype=torch.long)])
        else:
            x_padded = x
            y_padded = y
        padded_x.append(x_padded)
        padded_y.append(y_padded)
    return torch.stack(padded_x), torch.stack(padded_y)


def compute_sft_loss(model, input_ids: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Calculates cross-entropy loss exclusively on unmasked assistant tokens."""
    logits, _ = model(input_ids)
    loss = next_token_loss(logits, targets)
    
    # Calculate token accuracy on supervised positions
    with torch.no_grad():
        valid_mask = (targets != -100)
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            preds = logits.argmax(dim=-1)
            correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
            acc = correct / num_valid
        else:
            acc = 0.0

    return loss, {'loss': float(loss.item()), 'token_accuracy': acc, 'supervised_tokens': num_valid}
