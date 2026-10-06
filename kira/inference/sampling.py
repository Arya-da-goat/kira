"""Sampling transforms act on actual model logits only."""
import math
import torch


def sample_token(logits, history, temperature=0.8, top_k=40, top_p=0.95,
                 repetition_penalty=1.0, generator=None, forbidden=(0, 1, 3)):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if not 0 < top_p <= 1 or type(top_k) is not int or top_k < 0:
        raise ValueError('top_p must be in (0, 1] and top_k a nonnegative integer')
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError('repetition_penalty must be finite and positive')
    # CPU sampling provides a per-request RNG across CPU/CUDA/MPS backends.
    scores = logits.detach().float().cpu().clone()
    if not torch.isfinite(scores).all():
        raise ValueError('Model produced non-finite logits')
    if history:
        ids = torch.tensor(sorted(set(history)), dtype=torch.long)
        values = scores[ids]
        scores[ids] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    scores[list(forbidden)] = float('-inf')
    if temperature == 0:
        return scores.argmax().item()
    scores /= temperature
    if top_k:
        threshold = scores.topk(min(top_k, scores.numel())).values[-1]
        scores[scores < threshold] = float('-inf')
    if top_p < 1:
        sorted_scores, indices = scores.sort(descending=True)
        cumulative = sorted_scores.softmax(-1).cumsum(-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = float('-inf')
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).item()
