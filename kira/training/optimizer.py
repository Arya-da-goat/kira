import torch


def make_optimizer(model, config):
    # Biases and normalization scales do not get weight decay.
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': config.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=config.learning_rate)
