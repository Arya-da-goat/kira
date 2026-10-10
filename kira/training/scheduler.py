"""Learning rate scheduler with linear warmup and cosine decay to minimum learning rate."""
import math
from torch.optim.lr_scheduler import LambdaLR


def make_scheduler(optimizer, config):
    base_lr = config.learning_rate
    min_lr = getattr(config, 'min_learning_rate', 0.0)
    min_ratio = min_lr / max(base_lr, 1e-8)

    def factor(step):
        if step < config.warmup_steps:
            return (step + 1) / max(1, config.warmup_steps)
        total_decay_steps = max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, (step - config.warmup_steps) / total_decay_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Decay between 1.0 and min_ratio
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return LambdaLR(optimizer, factor)
