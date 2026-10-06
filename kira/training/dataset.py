"""Document-level deduplication/split before tokenizer fitting or chunking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

import torch
from torch.utils.data import Dataset


def clean_text(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '').strip()


def read_documents(path: str | Path) -> list[str]:
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.suffix.lower() in ('.txt', '.json', '.jsonl')) if path.is_dir() else [path]
    documents = []
    for file in files:
        raw = file.read_text(encoding='utf-8')
        if file.suffix.lower() == '.txt':
            records = re.split(r'\n\s*\n', raw.replace('\r\n', '\n'))
        elif file.suffix.lower() == '.jsonl':
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        elif file.suffix.lower() == '.json':
            value = json.loads(raw)
            records = value if isinstance(value, list) else [value]
        else:
            raise ValueError(f'Unsupported dataset format: {file.suffix}')
        for record in records:
            text = record.get('text') if isinstance(record, dict) else record
            if not isinstance(text, str):
                raise ValueError(f'{file}: records must be strings or objects with a text string')
            text = clean_text(text)
            if text:
                documents.append(text)
    # Exact cleaned duplicates must never straddle the validation boundary.
    documents = list(dict.fromkeys(documents))
    if not documents:
        raise ValueError('Dataset contains no text documents')
    return documents


def split_documents(documents: list[str], validation_fraction: float, seed: int):
    documents = list(dict.fromkeys(documents))
    if len(documents) < 2:
        raise ValueError('At least two distinct documents are required for a held-out split')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be in (0, 1)')
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    count = min(len(documents) - 1, max(1, round(len(documents) * validation_fraction)))
    validation = [documents[i] for i in indices[:count]]
    train = [documents[i] for i in indices[count:]]
    return train, validation


def data_fingerprint(train: list[str], validation: list[str]) -> str:
    return hashlib.sha256(json.dumps([train, validation], ensure_ascii=False).encode()).hexdigest()


class TokenDataset(Dataset):
    def __init__(self, documents, tokenizer, sequence_length: int):
        if sequence_length < 1:
            raise ValueError('sequence_length must be positive')
        self.samples = []
        for text in documents:
            tokens = tokenizer.encode(text, bos=True, eos=True)
            for start in range(0, len(tokens) - 1, sequence_length):
                chunk = tokens[start:start + sequence_length + 1]
                inputs, labels = chunk[:-1], chunk[1:]
                padding = sequence_length - len(inputs)
                self.samples.append((torch.tensor(inputs + [tokenizer.pad_id] * padding),
                                     torch.tensor(labels + [-100] * padding)))
        if not self.samples:
            raise ValueError('No training samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
