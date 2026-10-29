"""Exact and near-duplicate detection (MinHash + LSH), streaming, first-seen wins.

Near-duplicates matter twice: they waste compute, and when one copy lands in
training and another in validation/test they silently inflate scores. Running
dedup *before* the split removes that leak.

Shingles are character 5-grams after whitespace collapsing, which works for
Korean without a morphological analyzer.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

_SPACE = re.compile(r"\s+")
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def canonical(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def exact_key(text: str) -> str:
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def _shingle_hashes(text: str, k: int) -> np.ndarray:
    text = canonical(text)
    if len(text) <= k:
        grams = {text}
    else:
        grams = {text[i:i + k] for i in range(len(text) - k + 1)}
    values = [int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
              for g in grams]
    return np.fromiter(values, dtype=np.uint64, count=len(values))


class MinHashDeduper:
    """Keeps the first document of each near-duplicate cluster.

    ``bands * rows == num_perm``. The LSH S-curve threshold is roughly
    ``(1/bands) ** (1/rows)``; 16 bands x 8 rows gives ~0.71. Candidates are
    then confirmed with the estimated Jaccard similarity against ``threshold``.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, bands: int = 16,
                 shingle: int = 5, seed: int = 1):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.bands, self.rows = bands, num_perm // bands
        self.shingle = shingle
        rng = np.random.default_rng(seed)
        # Multiply-shift hashing in uint64 arithmetic (wraps mod 2**64).
        self._a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | np.uint64(1)
        self._b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
        self._tables: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]
        self._signatures: list[np.ndarray] = []

    def signature(self, text: str) -> np.ndarray:
        hashes = _shingle_hashes(text, self.shingle)
        with np.errstate(over="ignore"):
            mixed = (hashes[:, None] * self._a[None, :] + self._b[None, :]) & _MASK64
        return (mixed >> np.uint64(32)).min(axis=0).astype(np.uint32)

    def is_duplicate(self, text: str) -> bool:
        """Check ``text``; if new, remember it. Returns True for near-duplicates."""
        sig = self.signature(text)
        keys = [sig[b * self.rows:(b + 1) * self.rows].tobytes() for b in range(self.bands)]
        candidates: set[int] = set()
        for table, key in zip(self._tables, keys):
            candidates.update(table.get(key, ()))
        for idx in candidates:
            if float(np.mean(self._signatures[idx] == sig)) >= self.threshold:
                return True
        idx = len(self._signatures)
        self._signatures.append(sig)
        for table, key in zip(self._tables, keys):
            table[key].append(idx)
        return False

    def __len__(self) -> int:
        return len(self._signatures)
