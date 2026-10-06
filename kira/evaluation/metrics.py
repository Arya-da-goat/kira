import math
import torch
from kira.hardware import autocast_context
from kira.model.transformer import next_token_loss


@torch.inference_mode()
def evaluate(model, loader, device, precision='fp32'):
    was_training = model.training
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    try:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast_context(device, precision):
                logits, _ = model(inputs)
                loss = next_token_loss(logits, targets, reduction='sum')
            mask = targets != -100
            loss_sum += loss.item()
            count += mask.sum().item()
            correct += ((logits.argmax(-1) == targets) & mask).sum().item()
    finally:
        model.train(was_training)
    if not count:
        raise ValueError('Evaluation contains no target tokens')
    mean = loss_sum / count
    return {'validation_loss': mean, 'perplexity': math.exp(mean) if mean < 709 else None,
            'token_accuracy': correct / count, 'validation_tokens': count}
