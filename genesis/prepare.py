"""Prepare text that you own or are licensed to train on.

Input: .txt files from --input recursively.
Output: train.bin/val.bin uint16 token streams and provenance manifest.
No unauthorized scraping, silent license assumptions, or personal-data harvesting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .tokenizer import ByteTokenizer


def clean_text(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").strip()


def prepare(input_dir: Path, output_dir: Path, source: str, license_id: str,
            val_fraction: float = 0.1, seed: int = 42) -> dict:
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    if not source.strip() or not license_id.strip():
        raise ValueError("source and license provenance are required")
    files = sorted(input_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files under {input_dir}")
    seen = set()
    documents = []
    for path in files:
        text = clean_text(path.read_text(encoding="utf-8"))
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            documents.append({"path": str(path.relative_to(input_dir)), "text": text, "sha256": digest})
    if not documents:
        raise ValueError("empty dataset after cleaning and exact deduplication")
    tokenizer = ByteTokenizer()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(documents))
    val_count = max(1, round(len(documents) * val_fraction)) if len(documents) > 1 else 0
    val_indices = set(order[:val_count].tolist())
    train_tokens, val_tokens = [], []
    for i, doc in enumerate(documents):
        target = val_tokens if i in val_indices else train_tokens
        target.extend(tokenizer.encode(doc["text"], add_eos=True))
    # A single-document starter corpus cannot have a document-held-out split.
    # Divide its token stream, but disclose the weaker evaluation methodology.
    split_strategy = "document-held-out"
    if not val_tokens:
        cut = int(len(train_tokens) * (1 - val_fraction))
        val_tokens = train_tokens[cut:]
        train_tokens = train_tokens[:cut]
        split_strategy = "token-held-out (single document; weaker evaluation)"
    if not train_tokens or not val_tokens:
        raise ValueError("too little data to form train and validation sets")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.asarray(train_tokens, dtype=np.uint16).tofile(output_dir / "train.bin")
    np.asarray(val_tokens, dtype=np.uint16).tofile(output_dir / "val.bin")
    metadata = {
        "source": source, "license": license_id,
        "tokenizer": "UTF-8 byte tokenizer v0, eos=256", "seed": seed,
        "split_strategy": split_strategy,
        "train_tokens": len(train_tokens), "val_tokens": len(val_tokens),
        "unique_documents": len(documents),
        "documents": [{"path": d["path"], "sha256": d["sha256"]} for d in documents],
        "limitations": ["exact-document dedup only", "no automatic personal-data removal", "not production-grade data governance"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--source", required=True, help="Where the text came from")
    p.add_argument("--license", required=True, help="Documented license or your own rights")
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.source, args.license, args.val_fraction), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
