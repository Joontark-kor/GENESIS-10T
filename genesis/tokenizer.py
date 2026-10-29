"""Tokenizers behind one small interface.

* ``ByteTokenizer`` – training-free UTF-8 bytes (257 ids). Reversible, but a
  Hangul syllable costs 3 tokens, so it wastes context on Korean.
* ``BPETokenizer`` – byte-level BPE trained with Hugging Face ``tokenizers``
  (NFC normalizer, single-digit splitting, reserved chat tokens). Lossless on
  any UTF-8 input, so there is no <unk>.

Use ``load_tokenizer("byte")`` or ``load_tokenizer("tokenizers/ko32k.json")``.
Every tokenizer can report ``token_byte_lengths()`` so evaluation can compute
bits-per-byte, the only loss number comparable across tokenizers.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

EOS_TOKEN = "<|endoftext|>"
SPECIAL_TOKENS = [
    EOS_TOKEN, "<|pad|>", "<|im_start|>", "<|im_end|>",
    "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>", "<|reserved_3|>",
]


class Tokenizer(Protocol):
    vocab_size: int
    eos_id: int

    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...
    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]: ...
    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str: ...
    def token_byte_lengths(self) -> np.ndarray: ...
    def describe(self) -> dict: ...


def token_dtype(vocab_size: int) -> np.dtype:
    """Smallest unsigned dtype that can store every id of the vocabulary."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size {vocab_size} does not fit in uint32")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ByteTokenizer:
    vocab_size = 257
    eos_id = 256

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        tokens = list(unicodedata.normalize("NFC", text).encode("utf-8"))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        return [self.encode(text, add_eos=add_eos) for text in texts]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        payload = []
        for token in ids:
            token = int(token)
            if token == self.eos_id:
                if skip_special:
                    continue
                raise ValueError("cannot render EOS as bytes; use skip_special=True")
            if not 0 <= token < 256:
                raise ValueError(f"invalid byte token: {token}")
            payload.append(token)
        return bytes(payload).decode("utf-8", errors="replace")

    def token_byte_lengths(self) -> np.ndarray:
        lengths = np.ones(self.vocab_size, dtype=np.int64)
        lengths[self.eos_id] = 0
        return lengths

    def describe(self) -> dict:
        return {"type": "byte", "spec": "byte", "vocab_size": self.vocab_size,
                "eos_id": self.eos_id, "sha256": None}


class BPETokenizer:
    """Wrapper around a ``tokenizers`` JSON file produced by ``train_bpe``."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as HFTokenizer

        self.path = Path(path)
        self._tok = HFTokenizer.from_file(str(self.path))
        eos = self._tok.token_to_id(EOS_TOKEN)
        if eos is None:
            raise ValueError(f"{path} has no {EOS_TOKEN} token")
        self.eos_id = eos
        self.vocab_size = self._tok.get_vocab_size(with_added_tokens=True)
        self.sha256 = file_sha256(self.path)
        self._special_ids = {self._tok.token_to_id(t) for t in SPECIAL_TOKENS} - {None}

    def token_id(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise KeyError(f"{token!r} is not in the tokenizer")
        return token_id

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        encoded = self._tok.encode_batch(texts, add_special_tokens=False)
        out = [e.ids for e in encoded]
        if add_eos:
            for ids in out:
                ids.append(self.eos_id)
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def token_byte_lengths(self) -> np.ndarray:
        # Byte-level BPE maps each raw byte to exactly one printable character,
        # so a token's string length equals the number of bytes it covers.
        lengths = np.zeros(self.vocab_size, dtype=np.int64)
        for token_id in range(self.vocab_size):
            if token_id in self._special_ids:
                continue
            token = self._tok.id_to_token(token_id)
            lengths[token_id] = len(token) if token is not None else 0
        return lengths

    def describe(self) -> dict:
        return {"type": "bpe", "spec": str(self.path), "vocab_size": self.vocab_size,
                "eos_id": self.eos_id, "sha256": self.sha256}


def load_tokenizer(spec: str | Path, expected_sha256: str | None = None) -> Tokenizer:
    """``"byte"`` or a path to a BPE JSON. Optionally verifies the file hash."""
    if str(spec) == "byte":
        return ByteTokenizer()
    tokenizer = BPETokenizer(spec)
    if expected_sha256 and tokenizer.sha256 != expected_sha256:
        raise ValueError(f"tokenizer {spec} hash {tokenizer.sha256[:12]} does not match "
                         f"expected {expected_sha256[:12]}; data and model would disagree")
    return tokenizer


def train_bpe(texts: Iterator[str], vocab_size: int, output: str | Path,
              min_frequency: int = 2, show_progress: bool = True) -> BPETokenizer:
    """Train a byte-level BPE with NFC normalization and per-digit splitting."""
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

    if vocab_size < 256 + len(SPECIAL_TOKENS) + 1:
        raise ValueError("vocab_size must exceed the 256-byte alphabet plus special tokens")
    tok = HFTokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output))
    return BPETokenizer(output)
