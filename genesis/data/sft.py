"""Instruction-tuning (SFT) data: conversations -> token ids + loss masks.

    python -m genesis.data.sft --input data/raw/sft/*.jsonl --tokenizer tokenizers/ko32k-v1.json \
        --output data/sft/v1 --source my-korean-instructions --license CC-BY-4.0 \
        --rights-basis open-license --max-length 2049

Accepted records (one JSON object per line):
* ``{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}``
* ``{"instruction": "...", "input": "...", "output": "..."}`` (Alpaca style)

Steps: validate -> optional PII masking -> exact dedup -> drop examples longer
than ``--max-length`` tokens (set it to the model's context_length + 1) ->
content-hash train/val split -> ``{split}_ids.bin``, ``{split}_mask.bin``,
``{split}_offsets.npy`` + ``manifest.json``. Only assistant tokens are learned.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import __version__
from ..chat import encode_conversation, validate
from ..text import normalize_text
from ..tokenizer import load_tokenizer, token_dtype
from .clean import assign_split
from .pii import mask_pii
from .sources import RIGHTS_BASES


def to_messages(record: dict) -> list[dict]:
    if "messages" in record:
        return [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
    if "instruction" in record and "output" in record:
        prompt = record["instruction"].strip()
        if record.get("input"):
            prompt += "\n\n" + str(record["input"]).strip()
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(record["output"])}]
    raise ValueError("record needs 'messages' or 'instruction'/'output'")


def build(inputs: list[Path], tokenizer_spec: str, output: Path, source: str, license_id: str,
          rights_basis: str, commercial_use: str = "unknown", max_length: int = 2049,
          val_fraction: float = 0.02, mask: bool = True) -> dict:
    if not source.strip() or not license_id.strip() or license_id.strip().lower() == "unknown":
        raise ValueError("source and a concrete license are required")
    if rights_basis not in RIGHTS_BASES:
        raise ValueError(f"rights_basis must be one of {sorted(RIGHTS_BASES)}")
    tokenizer = load_tokenizer(tokenizer_spec)
    if not hasattr(tokenizer, "token_id"):
        raise ValueError("SFT needs a BPE tokenizer with the reserved chat tokens")
    dtype = token_dtype(tokenizer.vocab_size)
    output.mkdir(parents=True, exist_ok=True)
    buffers = {s: {"ids": [], "mask": [], "offsets": [0], "loss_tokens": 0} for s in ("train", "val")}
    stats: Counter = Counter()
    pii: Counter = Counter()
    seen: set[str] = set()
    for path in inputs:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["seen"] += 1
                try:
                    messages = to_messages(json.loads(line))
                    for message in messages:
                        message["content"] = normalize_text(message["content"])
                        if mask:
                            message["content"], found = mask_pii(message["content"])
                            pii.update({k: v for k, v in found.items() if v})
                    validate(messages)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    stats["invalid"] += 1
                    continue
                key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()
                if key in seen:
                    stats["exact_duplicate"] += 1
                    continue
                seen.add(key)
                ids, loss_mask = encode_conversation(tokenizer, messages)
                if len(ids) > max_length:
                    stats["too_long"] += 1
                    continue
                split = "val" if assign_split(key, val_fraction, 0.0) == "val" else "train"
                buf = buffers[split]
                buf["ids"].extend(ids)
                buf["mask"].extend(loss_mask)
                buf["offsets"].append(len(buf["ids"]))
                buf["loss_tokens"] += sum(loss_mask)
                stats["kept"] += 1
    if not buffers["train"]["offsets"][1:]:
        raise ValueError("no usable training conversations")
    train, val = buffers["train"], buffers["val"]
    if len(val["offsets"]) == 1 and len(train["offsets"]) > 10:
        # tiny sets can hash nothing into val; move the last train example so validation exists
        start, end = train["offsets"][-2], train["offsets"][-1]
        val["ids"], val["mask"] = train["ids"][start:end], train["mask"][start:end]
        val["offsets"].append(end - start)
        val["loss_tokens"] = sum(val["mask"])
        train["loss_tokens"] -= val["loss_tokens"]
        del train["ids"][start:], train["mask"][start:]
        train["offsets"].pop()
        stats["moved_to_val"] = 1
    splits = {}
    for split, buf in buffers.items():
        files = {"ids": f"{split}_ids.bin", "mask": f"{split}_mask.bin", "offsets": f"{split}_offsets.npy"}
        np.asarray(buf["ids"], dtype=dtype).tofile(output / files["ids"])
        np.asarray(buf["mask"], dtype=np.uint8).tofile(output / files["mask"])
        np.save(output / files["offsets"], np.asarray(buf["offsets"], dtype=np.int64))
        splits[split] = {"examples": len(buf["offsets"]) - 1, "tokens": len(buf["ids"]),
                         "loss_tokens": buf["loss_tokens"], **files}
    manifest = {
        "format_version": 2, "stage": "sft", "genesis_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer.describe(), "dtype": dtype.name, "eos_id": tokenizer.eos_id,
        "template": "chatml-reserved-tokens (genesis.chat)", "max_length": max_length,
        "pii_masking": mask, "statistics": dict(stats), "pii_masked": dict(pii), "splits": splits,
        "sources": [{"name": source, "license": license_id, "rights_basis": rights_basis,
                     "commercial_use": commercial_use, "files": [str(p) for p in inputs]}],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class SFTSplit:
    """Variable-length conversations; targets outside assistant turns are -100."""

    def __init__(self, data_dir: str | Path, split: str):
        data_dir = Path(data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        info = manifest["splits"].get(split)
        if info is None or info["examples"] == 0:
            raise ValueError(f"SFT split {split!r} is empty in {data_dir}")
        self.manifest = manifest
        self.max_length = manifest["max_length"]
        self.ids = np.memmap(data_dir / info["ids"], dtype=np.dtype(manifest["dtype"]), mode="r")
        self.mask = np.memmap(data_dir / info["mask"], dtype=np.uint8, mode="r")
        self.offsets = np.load(data_dir / info["offsets"])
        self.examples = len(self.offsets) - 1
        self.total = int(self.offsets[-1])

    def example(self, index: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        end = min(end, start + seq_len + 1)
        ids = np.asarray(self.ids[start:end], dtype=np.int64)
        learn = np.asarray(self.mask[start:end], dtype=bool)
        y = ids[1:].copy()
        y[~learn[1:]] = -100
        return ids[:-1], y

    @staticmethod
    def pad(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        length = max(len(x) for x, _ in pairs)
        x = np.zeros((len(pairs), length), dtype=np.int64)
        y = np.full((len(pairs), length), -100, dtype=np.int64)
        for row, (a, b) in enumerate(pairs):
            x[row, :len(a)] = a
            y[row, :len(b)] = b
        return x, y

    def sample(self, rng: np.random.Generator, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        indices = rng.integers(0, self.examples, size=batch_size)
        return self.pad([self.example(int(i), seq_len) for i in indices])

    def windows(self, seq_len: int, max_tokens: int | None = None):
        budget = max_tokens if max_tokens is not None else self.total
        for index in range(self.examples):
            if budget <= 0:
                return
            x, y = self.example(index, seq_len)
            if (y != -100).any():
                yield x, y
                budget -= len(x)


def open_split(data_dir: str | Path, split: str):
    """TokenSplit for pretraining data, SFTSplit for instruction data."""
    from .shards import TokenSplit, read_manifest
    if read_manifest(Path(data_dir)).get("stage") == "sft":
        return SFTSplit(data_dir, split)
    return TokenSplit(data_dir, split)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="JSONL files or globs")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--rights-basis", required=True, choices=sorted(RIGHTS_BASES))
    p.add_argument("--commercial-use", choices=("yes", "no", "unknown"), default="unknown")
    p.add_argument("--max-length", type=int, default=2049, help="model context_length + 1")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--no-pii-mask", action="store_true")
    args = p.parse_args()
    inputs = sorted({Path(f) for pattern in args.input for f in glob.glob(pattern)})
    if not inputs:
        raise FileNotFoundError("no input files matched")
    manifest = build(inputs, args.tokenizer, args.output, args.source, args.license, args.rights_basis,
                     args.commercial_use, args.max_length, args.val_fraction, not args.no_pii_mask)
    print(json.dumps({k: manifest[k] for k in ("statistics", "splits", "pii_masked")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
