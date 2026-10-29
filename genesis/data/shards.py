"""Token shard files and the reader used by training and evaluation.

Layout of a tokenized dataset directory::

    manifest.json            # tokenizer, dtype, per-split shard list, provenance
    train_00000.bin ...      # flat little-endian token ids, documents end with EOS
    val_00000.bin
    test_00000.bin

Older directories with ``train.bin``/``val.bin`` (uint16, byte tokenizer) are
still readable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
FORMAT_VERSION = 2


class ShardWriter:
    def __init__(self, out_dir: Path, split: str, dtype: np.dtype,
                 max_tokens_per_shard: int = 100_000_000):
        self.out_dir, self.split, self.dtype = Path(out_dir), split, np.dtype(dtype)
        self.max_tokens = max_tokens_per_shard
        self.shards: list[dict] = []
        self._handle = None
        self._current = 0
        self.total_tokens = 0
        self.documents = 0

    def _open_next(self) -> None:
        self.close()
        name = f"{self.split}_{len(self.shards):05d}.bin"
        self._handle = open(self.out_dir / name, "wb")
        self.shards.append({"file": name, "tokens": 0})
        self._current = 0

    def write(self, ids: list[int] | np.ndarray) -> None:
        array = np.asarray(ids, dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() >= np.iinfo(self.dtype).max + 1):
            raise ValueError(f"token id out of range for {self.dtype}")
        array = array.astype(self.dtype)
        while array.size:
            if self._handle is None or self._current >= self.max_tokens:
                self._open_next()
            room = self.max_tokens - self._current
            chunk, array = array[:room], array[room:]
            chunk.tofile(self._handle)
            self._current += chunk.size
            self.shards[-1]["tokens"] += int(chunk.size)
            self.total_tokens += int(chunk.size)
        self.documents += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def summary(self) -> dict:
        return {"documents": self.documents, "tokens": self.total_tokens, "shards": self.shards}


def read_manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run genesis.prepare or genesis.data.tokenize")
    return json.loads(path.read_text(encoding="utf-8"))


class TokenSplit:
    """Memory-mapped view over all shards of one split."""

    def __init__(self, data_dir: str | Path, split: str):
        data_dir = Path(data_dir)
        manifest = read_manifest(data_dir)
        self.manifest = manifest
        if manifest.get("format_version", 1) >= 2:
            info = manifest["splits"].get(split)
            if info is None or info["tokens"] == 0:
                raise ValueError(f"split {split!r} is empty in {data_dir}")
            dtype = np.dtype(manifest["dtype"])
            self.arrays = [np.memmap(data_dir / s["file"], dtype=dtype, mode="r")
                           for s in info["shards"] if s["tokens"] > 0]
        else:  # legacy single-file layout
            legacy = {"train": "train.bin", "val": "val.bin", "test": "test.bin"}[split]
            self.arrays = [np.memmap(data_dir / legacy, dtype=np.uint16, mode="r")]
        self.lengths = np.array([len(a) for a in self.arrays], dtype=np.int64)
        self.total = int(self.lengths.sum())

    def sample(self, rng: np.random.Generator, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Random windows of ``seq_len + 1`` tokens; shards weighted by usable length."""
        usable = np.maximum(self.lengths - seq_len - 1, 0)
        if usable.sum() == 0:
            raise ValueError(f"need > {seq_len + 1} tokens in a shard, have {self.lengths.tolist()}")
        shard_ids = rng.choice(len(self.arrays), size=batch_size, p=usable / usable.sum())
        x = np.empty((batch_size, seq_len), dtype=np.int64)
        y = np.empty((batch_size, seq_len), dtype=np.int64)
        for row, shard in enumerate(shard_ids):
            start = int(rng.integers(0, usable[shard] + 1))
            window = np.asarray(self.arrays[shard][start:start + seq_len + 1], dtype=np.int64)
            x[row], y[row] = window[:-1], window[1:]
        return x, y

    def windows(self, seq_len: int, max_tokens: int | None = None):
        """Deterministic, non-overlapping windows for evaluation."""
        budget = max_tokens if max_tokens is not None else self.total
        for array in self.arrays:
            if 2 <= len(array) <= seq_len:  # tiny eval shard: one shorter window
                window = np.asarray(array, dtype=np.int64)
                yield window[:-1], window[1:]
                budget -= len(window) - 1
                continue
            for start in range(0, len(array) - seq_len, seq_len):
                if budget <= 0:
                    return
                window = np.asarray(array[start:start + seq_len + 1], dtype=np.int64)
                if len(window) < seq_len + 1:
                    break
                yield window[:-1], window[1:]
                budget -= seq_len
