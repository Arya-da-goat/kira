"""Deterministic byte-level BPE tokenizer with chat template and SFT masking."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


class Tokenizer:
    SPECIAL = (
        '<PAD>', '<BOS>', '<EOS>', '<UNK>',
        '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|assistant|>'
    )
    pad_id, bos_id, eos_id, unk_id = range(4)
    im_start_id, im_end_id, system_id, user_id, assistant_id = range(4, 9)

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        num_special = len(self.SPECIAL)
        self.special_offset = num_special
        self.pieces = [b''] * num_special + [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if not num_special <= left < len(self.pieces) or not num_special <= right < len(self.pieces):
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
        num_special = len(cls.SPECIAL)
        min_vocab = num_special + 256
        if vocab_size < min_vocab:
            raise ValueError(f'Byte BPE needs at least {min_vocab} vocabulary entries')
        sequences = [[byte + num_special for byte in text.encode('utf-8')] for text in documents]
        if not any(sequences):
            raise ValueError('Cannot train a tokenizer on empty text')
        merges = []
        while min_vocab + len(merges) < vocab_size:
            counts = Counter(pair for ids in sequences for pair in zip(ids, ids[1:]))
            if not counts:
                break
            pair, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            token_id = min_vocab + len(merges)
            sequences = [cls._merge(ids, pair, token_id) for ids in sequences]
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        num_special = len(self.SPECIAL)
        ids = [byte + num_special for byte in text.encode('utf-8')]
        for index, pair in enumerate(self.merges):
            ids = self._merge(ids, pair, num_special + 256 + index)
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chunks = []
        num_special = len(self.SPECIAL)
        for token in ids:
            if not 0 <= token < self.vocab_size:
                token = self.unk_id
            if token < num_special:
                if not skip_special or token == self.unk_id:
                    chunks.append(self.SPECIAL[token].encode('utf-8'))
            else:
                chunks.append(self.pieces[token])
        return b''.join(chunks).decode('utf-8', errors='replace')

    def apply_chat_template(self, messages: Sequence[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Formats conversation with structured <|im_start|>role\ncontent<|im_end|>\n format."""
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            formatted.append("<|im_start|>assistant\n")
        return "".join(formatted)

    def encode_chat(self, messages: Sequence[dict[str, str]], max_length: int | None = None) -> tuple[list[int], list[int]]:
        """Encodes structured conversation for SFT with assistant-only loss masking (-100)."""
        input_ids = [self.bos_id]
        labels = [-100]  # BOS is never a target

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            header_ids = self.encode(f"<|im_start|>{role}\n")
            body_ids = self.encode(f"{content}<|im_end|>\n")

            input_ids.extend(header_ids)
            labels.extend([-100] * len(header_ids))

            input_ids.extend(body_ids)
            if role == 'assistant':
                # Loss is computed exclusively on assistant tokens
                labels.extend(body_ids)
            else:
                labels.extend([-100] * len(body_ids))

        input_ids.append(self.eos_id)
        labels.append(self.eos_id if (messages and messages[-1].get('role') == 'assistant') else -100)

        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        return input_ids, labels

    def to_dict(self) -> dict:
        return {
            'format': 'kira-frontier-bpe-v2',
            'special_tokens': list(self.SPECIAL),
            'merges': [list(pair) for pair in self.merges],
            'vocab_hex': [piece.hex() for piece in self.pieces[len(self.SPECIAL):]]
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tokenizer:
        fmt = data.get('format')
        # Support both legacy v1 and v2 formats
        if fmt not in ('kira-byte-bpe-v1', 'kira-frontier-bpe-v2'):
            raise ValueError('Unsupported tokenizer format')
        
        tokenizer = cls(data['merges'])
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
