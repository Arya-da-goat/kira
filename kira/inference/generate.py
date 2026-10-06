from __future__ import annotations

import argparse
import json
import time
import torch
from kira.hardware import select_device, synchronize
from kira.inference.sampling import sample_token
from kira.training.checkpoint import load_model, training_status


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens=64, temperature=0.8,
             top_k=40, top_p=0.95, repetition_penalty=1.0, seed=42):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    prompt_ids = tokenizer.encode(prompt, bos=True)
    if len(prompt_ids) >= model.config.max_seq_len:
        raise ValueError(f'Prompt has {len(prompt_ids)} tokens; context limit is {model.config.max_seq_len}. Shorten the context.')
    allowed = min(max_new_tokens, model.config.max_seq_len - len(prompt_ids))
    device = next(model.parameters()).device
    model.eval()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    synchronize(device)
    started = time.perf_counter()
    logits, cache = model(torch.tensor([prompt_ids], device=device), use_cache=True)
    generated = []
    stop_reason = 'max_new_tokens' if allowed == max_new_tokens else 'context_limit'
    for index in range(allowed):
        token = sample_token(logits[0, -1], prompt_ids + generated, temperature, top_k,
                             top_p, repetition_penalty, generator)
        generated.append(token)
        if token == tokenizer.eos_id:
            stop_reason = 'eos'
            break
        if index + 1 < allowed:
            logits, cache = model(torch.tensor([[token]], device=device), cache=cache, use_cache=True)
    synchronize(device)
    seconds = time.perf_counter() - started
    return {'text': tokenizer.decode(generated), 'token_ids': generated,
            'prompt_tokens': len(prompt_ids), 'generated_tokens': len(generated),
            'seconds': seconds, 'tokens_per_second': len(generated) / seconds,
            'stop_reason': stop_reason}


def main():
    parser = argparse.ArgumentParser(description='Generate text from local Kira weights')
    parser.add_argument('--checkpoint', default='checkpoints/latest.pt')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, tokenizer, checkpoint = load_model(args.checkpoint, select_device(args.device))
    result = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
                      args.top_k, args.top_p, args.repetition_penalty, args.seed)
    print(json.dumps(dict(result, training_status=training_status(checkpoint['state']),
                          training_step=checkpoint['state']['step']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
