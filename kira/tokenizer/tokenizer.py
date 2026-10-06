"""Deterministic byte-level BPE. No downloaded vocabulary or model required."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable


class Tokenizer:
    SPECIAL = ('<PAD>', '<BOS>', '<EOS>', '<UNK>')
    pad_id, bos_id, eos_id, unk_id = range(4)

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.pieces = [b''] * 4 + [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if not 4 <= left < len(self.pieces) or not 4 <= right < len(self.pieces):
                raise ValueError('Invalid BPE merge reference')
            self.pieces.append(self.pieces[left] + self.pieces[right])

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], replacement: int) -> list[int]:
        result, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                result.append(replacement)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    @classmethod
    def train(cls, documents: Iterable[str], vocab_size: int = 384) -> Tokenizer:
        if vocab_size < 260:
            raise ValueError('Byte BPE needs at least 260 vocabulary entries')
        sequences = [[byte + 4 for byte in text.encode('utf-8')] for text in documents]
        if not any(sequences):
            raise ValueError('Cannot train a tokenizer on empty text')
        merges = []
        while 260 + len(merges) < vocab_size:
            counts = Counter(pair for ids in sequences for pair in zip(ids, ids[1:]))
            if not counts:
                break
            pair, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            token_id = 260 + len(merges)
            sequences = [cls._merge(ids, pair, token_id) for ids in sequences]
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        # Literal special-token spellings in user text remain ordinary UTF-8 text.
        ids = [byte + 4 for byte in text.encode('utf-8')]
        for index, pair in enumerate(self.merges):
            ids = self._merge(ids, pair, 260 + index)
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chunks = []
        for token in ids:
            if not 0 <= token < self.vocab_size:
                token = self.unk_id
            if token < 4:
                if not skip_special or token == self.unk_id:
                    chunks.append(self.SPECIAL[token].encode())
            else:
                chunks.append(self.pieces[token])
        # A sampled sequence can end halfway through a UTF-8 character.
        return b''.join(chunks).decode('utf-8', errors='replace')

    def to_dict(self) -> dict:
        return {'format': 'kira-byte-bpe-v1', 'special_tokens': list(self.SPECIAL),
                'merges': [list(pair) for pair in self.merges],
                'vocab_hex': [piece.hex() for piece in self.pieces[4:]]}

    @classmethod
    def from_dict(cls, data: dict) -> Tokenizer:
        if data.get('format') != 'kira-byte-bpe-v1' or data.get('special_tokens') != list(cls.SPECIAL):
            raise ValueError('Unsupported tokenizer format or special tokens')
        tokenizer = cls(data['merges'])
        if tokenizer.to_dict() != data:
            raise ValueError('Tokenizer vocabulary does not match its merges')
        return tokenizer

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
