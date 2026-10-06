"""Real next-token pretraining, with deterministic document order and resumable updates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset
from kira.config import ModelConfig, TrainingConfig
from kira.hardware import select_device, select_precision, autocast_context, synchronize, describe
from kira.model.transformer import KiraTransformer, next_token_loss
from kira.tokenizer.tokenizer import Tokenizer
from kira.training.dataset import read_documents, split_documents, data_fingerprint, TokenDataset
from kira.training.optimizer import make_optimizer
from kira.training.scheduler import make_scheduler
from kira.training.checkpoint import load_model, save_checkpoint, restore_rng
from kira.evaluation.metrics import evaluate


def train(config_path='configs/tiny.json', data=None, tokenizer_path='checkpoints/tokenizer.json',
          output='checkpoints', resume=None, device_name='auto', stop_after=None):
    device = select_device(device_name)
    output = Path(output)
    if stop_after is not None and stop_after < 1:
        raise ValueError('stop_after must be a positive optimizer step')
    if resume:
        model, tokenizer, checkpoint = load_model(resume, device)
        config = checkpoint['config']
        tc = TrainingConfig(**config['training'])
        data = data or config['data']
        state = checkpoint['state']
    else:
        if (output / 'latest.pt').exists() or (output / 'metrics.jsonl').exists():
            raise ValueError('Output already has a run; use --resume or a fresh output directory')
        config = json.loads(Path(config_path).read_text())
        tc = TrainingConfig(**config['training'])
        data = data or 'data/example'
        random.seed(tc.seed)
        torch.manual_seed(tc.seed)
        tokenizer = Tokenizer.load(tokenizer_path)
        mc = ModelConfig(**dict(config['model'], vocab_size=tokenizer.vocab_size))
        model = KiraTransformer(mc).to(device)
        config = {'model': asdict(mc), 'training': asdict(tc), 'data': str(Path(data))}
        state = {'step': 0, 'epoch': 0, 'batch_in_epoch': 0, 'tokens_processed': 0,
                 'best_validation_loss': float('inf'), 'history': [], 'run_complete': False}
    precision = select_precision(device, tc.precision)
    train_docs, val_docs = split_documents(read_documents(data), tc.validation_fraction, tc.seed)
    fingerprint = data_fingerprint(train_docs, val_docs)
    if resume:
        if fingerprint != state['data_fingerprint']:
            raise ValueError('Dataset or split changed since checkpoint; refusing inconsistent resume')
    else:
        provenance_path = Path(str(tokenizer_path) + '.provenance.json')
        if not provenance_path.exists():
            raise ValueError('Tokenizer split provenance is missing; run kira.tokenizer.train first')
        provenance = json.loads(provenance_path.read_text())
        if provenance['data_fingerprint'] != fingerprint or provenance['tokenizer_fingerprint'] != tokenizer.fingerprint:
            raise ValueError('Tokenizer/data split mismatch; retrain tokenizer on this training split')
    state['data_fingerprint'] = fingerprint
    state['train_documents'], state['validation_documents'] = len(train_docs), len(val_docs)
    state['hardware'] = describe(device, precision)
    train_data = TokenDataset(train_docs, tokenizer, model.config.max_seq_len)
    val_loader = DataLoader(TokenDataset(val_docs, tokenizer, model.config.max_seq_len),
                            batch_size=tc.batch_size, generator=torch.Generator().manual_seed(tc.seed))
    optimizer = make_optimizer(model, tc)
    scheduler = make_scheduler(optimizer, tc)
    scaler = torch.amp.GradScaler('cuda', enabled=(precision == 'fp16'))
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'])
    output.mkdir(parents=True, exist_ok=True)
    # A checkpoint is the source of truth when resuming after a log-only update.
    log_path = output / 'metrics.jsonl'
    log_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in state['history']))
    tokenizer.save(output / 'tokenizer.json')
    print(json.dumps(dict(state['hardware'], parameters=sum(p.numel() for p in model.parameters()),
                          initial_state='resumed' if resume else 'random initialization'), indent=2))
    optimizer.zero_grad(set_to_none=True)
    limit = min(tc.max_steps, stop_after) if stop_after else tc.max_steps
    if state['step'] >= limit or state['epoch'] >= tc.max_epochs:
        return state
    for epoch in range(state['epoch'], tc.max_epochs):
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(tc.seed + epoch)).tolist()
        loader = DataLoader(Subset(train_data, order), batch_size=tc.batch_size,
                            generator=torch.Generator().manual_seed(tc.seed + epoch))
        cursor = state['batch_in_epoch'] if epoch == state['epoch'] else 0
        iterator = iter(loader)
        for _ in range(cursor):
            next(iterator)
        while cursor < len(loader):
            group = [next(iterator) for _ in range(min(tc.gradient_accumulation_steps, len(loader) - cursor))]
            target_count = sum((targets != -100).sum().item() for _, targets in group)
            model.train()
            synchronize(device)
            started = time.perf_counter()
            loss_sum = 0.0
            for inputs, targets in group:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast_context(device, precision):
                    logits, _ = model(inputs)
                    total_loss = next_token_loss(logits, targets, reduction='sum')
                    loss = total_loss / target_count
                loss_sum += total_loss.detach().item()
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            if not torch.isfinite(grad_norm):
                if precision == 'fp16':
                    scaler.step(optimizer)  # GradScaler skips this non-finite update.
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError('FP16 gradients overflowed; resume with a supported BF16/FP32 config')
                raise FloatingPointError('Non-finite gradients; checkpoint has not been advanced')
            learning_rate = optimizer.param_groups[0]['lr']
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            seconds = time.perf_counter() - started
            cursor += len(group)
            state['step'] += 1
            state['tokens_processed'] += target_count
            state['epoch'] = epoch + (cursor == len(loader))
            state['batch_in_epoch'] = 0 if cursor == len(loader) else cursor
            finished = state['step'] >= tc.max_steps or state['epoch'] >= tc.max_epochs
            state['run_complete'] = finished
            row = {'step': state['step'], 'epoch': epoch, 'train_loss': loss_sum / target_count,
                   'learning_rate': learning_rate, 'gradient_norm': grad_norm.item(),
                   'tokens_processed': state['tokens_processed'], 'step_tokens': target_count,
                   'tokens_per_second': target_count / seconds, 'seconds': seconds}
            should_stop = state['step'] >= limit or finished
            improved = False
            if state['step'] % tc.eval_interval == 0 or should_stop:
                row.update(evaluate(model, val_loader, device, precision))
                improved = row['validation_loss'] < state['best_validation_loss']
                state['best_validation_loss'] = min(state['best_validation_loss'], row['validation_loss'])
            state['history'].append(row)
            with log_path.open('a') as log:
                log.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row, allow_nan=False), flush=True)
            if improved:
                save_checkpoint(output / 'best.pt', model, tokenizer, optimizer, scheduler, scaler, config, state)
            if state['step'] % tc.save_interval == 0:
                save_checkpoint(output / f"step-{state['step']:08d}.pt", model, tokenizer, optimizer, scheduler, scaler, config, state)
            if improved or state['step'] % tc.save_interval == 0 or should_stop:
                save_checkpoint(output / 'latest.pt', model, tokenizer, optimizer, scheduler, scaler, config, state)
            if should_stop:
                return state
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/tiny.json')
    parser.add_argument('--data')
    parser.add_argument('--tokenizer', default='checkpoints/tokenizer.json')
    parser.add_argument('--output', default=None)
    parser.add_argument('--resume')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--stop-after', type=int, help='Stop cleanly at this optimizer step without changing the schedule')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    train(args.config, args.data, args.tokenizer, args.output or (str(Path(args.resume).parent) if args.resume else 'checkpoints'),
          args.resume, args.device, args.stop_after)


if __name__ == '__main__':
    main()
