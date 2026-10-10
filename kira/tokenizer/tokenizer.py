"""Deterministic byte-level BPE tokenizer with atomic special tokens and chat templates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


class Tokenizer:
    DEFAULT_SPECIAL = (
        '<PAD>', '<BOS>', '<EOS>', '<UNK>',
        '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|assistant|>'
    )
    LEGACY_SPECIAL = ('<PAD>', '<BOS>', '<EOS>', '<UNK>')

    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 special_tokens: Sequence[str] | None = None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.special_tokens = tuple(special_tokens if special_tokens is not None else self.DEFAULT_SPECIAL)
        self.special_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_special = {idx: tok for idx, tok in enumerate(self.special_tokens)}

        num_special = len(self.special_tokens)
        self.special_offset = num_special
        self.pieces = [b''] * num_special + [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if not num_special <= left < len(self.pieces) or not num_special <= right < len(self.pieces):
                raise ValueError('Invalid BPE merge reference')
            self.pieces.append(self.pieces[left] + self.pieces[right])

        # Precompile regex for atomic special token recognition
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(tok) for tok in sorted_tokens]
            self._special_regex = re.compile('(' + '|'.join(escaped) + ')')
        else:
            self._special_regex = None

    @property
    def pad_id(self) -> int:
        return self.special_to_id.get('<PAD>', 0)

    @property
    def bos_id(self) -> int:
        return self.special_to_id.get('<BOS>', 1)

    @property
    def eos_id(self) -> int:
        return self.special_to_id.get('<EOS>', 2)

    @property
    def unk_id(self) -> int:
        return self.special_to_id.get('<UNK>', 3)

    @property
    def im_start_id(self) -> int | None:
        return self.special_to_id.get('<|im_start|>')

    @property
    def im_end_id(self) -> int | None:
        return self.special_to_id.get('<|im_end|>')

    @property
    def system_id(self) -> int | None:
        return self.special_to_id.get('<|system|>')

    @property
    def user_id(self) -> int | None:
        return self.special_to_id.get('<|user|>')

    @property
    def assistant_id(self) -> int | None:
        return self.special_to_id.get('<|assistant|>')

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
    def train(cls, documents: Iterable[str], vocab_size: int = 384,
              special_tokens: Sequence[str] | None = None) -> Tokenizer:
        specials = tuple(special_tokens if special_tokens is not None else cls.DEFAULT_SPECIAL)
        num_special = len(specials)
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
        return cls(merges, special_tokens=specials)

    def _encode_ordinary_chunk(self, text: str) -> list[int]:
        num_special = self.special_offset
        ids = [byte + num_special for byte in text.encode('utf-8')]
        for index, pair in enumerate(self.merges):
            ids = self._merge(ids, pair, num_special + 256 + index)
        return ids

    def encode(self, text: str, bos: bool = False, eos: bool = False,
               allowed_special: bool | set[str] | Sequence[str] | str = False) -> list[int]:
        """Encodes text to token IDs.
        
        If allowed_special is True or 'all', all registered special tokens are recognized atomically.
        If allowed_special is a set/sequence of tokens, only those tokens are recognized atomically.
        If allowed_special is False, special-token-looking text remains ordinary UTF-8 byte tokens.
        """
        ids = []
        if bos:
            ids.append(self.bos_id)

        if not text:
            if eos:
                ids.append(self.eos_id)
            return ids

        # Determine which special tokens are allowed to be matched atomically
        if allowed_special is True or allowed_special == 'all':
            active_specials = set(self.special_tokens)
        elif isinstance(allowed_special, (set, list, tuple)):
            active_specials = set(allowed_special).intersection(self.special_tokens)
        else:
            active_specials = set()

        if active_specials and self._special_regex is not None:
            # Build regex for currently active special tokens
            escaped = [re.escape(tok) for tok in sorted(active_specials, key=len, reverse=True)]
            regex = re.compile('(' + '|'.join(escaped) + ')')
            parts = regex.split(text)
            for part in parts:
                if not part:
                    continue
                if part in active_specials:
                    ids.append(self.special_to_id[part])
                else:
                    ids.extend(self._encode_ordinary_chunk(part))
        else:
            ids.extend(self._encode_ordinary_chunk(text))

        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chunks = []
        num_special = self.special_offset
        for token in ids:
            if not 0 <= token < self.vocab_size:
                token = self.unk_id
            if token < num_special:
                if not skip_special or token == self.unk_id:
                    chunks.append(self.special_tokens[token].encode('utf-8'))
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
        """Encodes structured conversation for SFT with atomic special tokens and assistant-only loss masking (-100)."""
        if self.im_start_id is None or self.im_end_id is None:
            raise ValueError('Chat template requires a tokenizer with <|im_start|> and <|im_end|> special tokens.')

        im_start = self.im_start_id
        im_end = self.im_end_id

        token_seq = []
        label_seq = []

        # Start of sequence
        token_seq.append(self.bos_id)
        label_seq.append(-100)

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Turn header: <|im_start|> + role + \n (atomic special token + role text)
            header_ids = [im_start] + self.encode(f"{role}\n", allowed_special=False)
            token_seq.extend(header_ids)
            label_seq.extend([-100] * len(header_ids))

            # Turn content (user/system text encoded with allowed_special=False to prevent prompt injection)
            content_ids = self.encode(content, allowed_special=False)
            token_seq.extend(content_ids)

            # End marker: <|im_end|> + \n
            end_ids = [im_end] + self.encode("\n", allowed_special=False)
            token_seq.extend(end_ids)

            if role == 'assistant':
                # Supervised targets: loss is computed exclusively on assistant content and its end token
                label_seq.extend(content_ids)
                label_seq.extend(end_ids)
            else:
                label_seq.extend([-100] * len(content_ids))
                label_seq.extend([-100] * len(end_ids))

        # Handle max_length truncation while ensuring assistant targets are preserved
        if max_length is not None and len(token_seq) > max_length:
            has_assistant_target = any(lbl != -100 for lbl in label_seq)
            if has_assistant_target:
                # Check if right-truncation preserves any assistant targets
                right_truncated_labels = label_seq[:max_length]
                if any(lbl != -100 for lbl in right_truncated_labels):
                    token_seq = token_seq[:max_length]
                    label_seq = right_truncated_labels
                else:
                    # Slide window towards the end to preserve assistant targets
                    start_idx = len(token_seq) - max_length
                    token_seq = token_seq[start_idx:]
                    label_seq = label_seq[start_idx:]
            else:
                token_seq = token_seq[:max_length]
                label_seq = label_seq[:max_length]

        return token_seq, label_seq

    def to_dict(self) -> dict:
        is_v2 = ('<|im_start|>' in self.special_tokens)
        fmt = 'kira-frontier-bpe-v2' if is_v2 else 'kira-byte-bpe-v1'
        return {
            'format': fmt,
            'special_tokens': list(self.special_tokens),
            'merges': [list(pair) for pair in self.merges],
            'vocab_hex': [piece.hex() for piece in self.pieces[self.special_offset:]]
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tokenizer:
        fmt = data.get('format')
        if fmt not in ('kira-byte-bpe-v1', 'kira-frontier-bpe-v2'):
            raise ValueError(f'Unsupported tokenizer format: {fmt}')

        special_tokens = data.get('special_tokens')
        if fmt == 'kira-byte-bpe-v1' and not special_tokens:
            special_tokens = cls.LEGACY_SPECIAL

        tokenizer = cls(data['merges'], special_tokens=special_tokens)
        saved_vocab = data.get('vocab_hex')
        if saved_vocab is not None:
            reconstructed_vocab = [piece.hex() for piece in tokenizer.pieces[tokenizer.special_offset:]]
            if reconstructed_vocab != saved_vocab:
                raise ValueError('Tokenizer vocabulary does not match its merges or special tokens')

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
