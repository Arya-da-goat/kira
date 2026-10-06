"""Atomic checkpoints loaded with PyTorch's restricted weights-only loader."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import random
import tempfile
import torch
from kira.config import ModelConfig
from kira.model.transformer import KiraTransformer
from kira.tokenizer.tokenizer import Tokenizer


def rng_state():
    state = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'].cpu())


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, scaler, config, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format_version': 1, 'model_config': asdict(model.config),
               'model': model.state_dict(), 'tokenizer': tokenizer.to_dict(),
               'tokenizer_fingerprint': tokenizer.fingerprint,
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'scaler': scaler.state_dict(), 'config': config, 'state': state, 'rng': rng_state()}
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    return checkpoint


def load_model(path, device='cpu'):
    checkpoint = read_checkpoint(path)
    tokenizer = Tokenizer.from_dict(checkpoint['tokenizer'])
    if tokenizer.fingerprint != checkpoint['tokenizer_fingerprint']:
        raise ValueError('Checkpoint tokenizer fingerprint mismatch')
    config = ModelConfig(**checkpoint['model_config'])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError('Model/tokenizer vocabulary mismatch')
    model = KiraTransformer(config)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), tokenizer, checkpoint


def training_status(state):
    if state['step'] == 0:
        return 'untrained'
    return 'training run finished; quality unverified' if state.get('run_complete') else 'partially trained'
