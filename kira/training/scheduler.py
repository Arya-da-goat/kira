import math
from torch.optim.lr_scheduler import LambdaLR


def make_scheduler(optimizer, config):
    def factor(step):
        if step < config.warmup_steps:
            return (step + 1) / max(1, config.warmup_steps)
        progress = min(1.0, (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, factor)
