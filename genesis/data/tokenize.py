"""Stage C: cleaned text -> token shards + manifest consumed by training.

    python -m genesis.data.tokenize --input data/clean/v1 \
        --tokenizer tokenizers/ko32k-v1.json --output data/tokens/ko32k-v1

Each document is encoded and terminated with EOS. The manifest pins the
tokenizer hash so training refuses to mix data and tokenizers by accident.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .. import __version__
from ..tokenizer import file_sha256, load_tokenizer, token_dtype
from .clean import iter_clean
from .shards import FORMAT_VERSION, SPLITS, ShardWriter


def _encode_split(records, tokenizer, writer: ShardWriter, batch_docs: int) -> None:
    batch: list[str] = []
    for record in records:
        batch.append(record["text"])
        if len(batch) >= batch_docs:
            for ids in tokenizer.encode_batch(batch, add_eos=True):
                writer.write(ids)
            batch = []
    if batch:
        for ids in tokenizer.encode_batch(batch, add_eos=True):
            writer.write(ids)


def _carve_eval_from_train(output: Path, summaries: dict, dtype: np.dtype,
                           missing: list[str], fraction: float) -> str:
    """Tiny corpora may hash every document into train; borrow its tail tokens."""
    train_info = summaries["train"]
    if len(train_info["shards"]) != 1:
        raise ValueError("eval split empty on a multi-shard corpus; add documents or raise the fractions")
    train_path = output / train_info["shards"][0]["file"]
    tokens = np.fromfile(train_path, dtype=dtype)
    take = max(1, int(len(tokens) * fraction))
    for split in missing:
        tokens, tail = tokens[:-take], tokens[-take:]
        name = f"{split}_00000.bin"
        tail.tofile(output / name)
        summaries[split] = {"documents": 0, "tokens": int(tail.size),
                            "shards": [{"file": name, "tokens": int(tail.size)}]}
    tokens.tofile(train_path)
    train_info["tokens"] = int(tokens.size)
    train_info["shards"][0]["tokens"] = int(tokens.size)
    return (f"token-held-out tail of train for {', '.join(missing)} "
            "(too few documents; weaker evaluation, do not report as benchmark)")


def tokenize_dataset(clean_dir: Path, tokenizer_spec: str, output: Path,
                     max_tokens_per_shard: int = 100_000_000, batch_docs: int = 1000,
                     fallback_fraction: float = 0.1) -> dict:
    clean_dir, output = Path(clean_dir), Path(output)
    clean_manifest_path = clean_dir / "clean_manifest.json"
    clean_manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(tokenizer_spec)
    dtype = token_dtype(tokenizer.vocab_size)
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.bin"):
        stale.unlink()
    summaries = {}
    for split in SPLITS:
        writer = ShardWriter(output, split, dtype, max_tokens_per_shard)
        try:
            _encode_split(iter_clean(clean_dir, split), tokenizer, writer, batch_docs)
        finally:
            writer.close()
        summaries[split] = writer.summary()
    if summaries["train"]["tokens"] == 0:
        raise ValueError("train split is empty after cleaning")
    missing = [s for s in ("val", "test") if summaries[s]["tokens"] == 0]
    strategy = "document-held-out (content-hash buckets)"
    if missing:
        strategy = _carve_eval_from_train(output, summaries, dtype, missing, fallback_fraction)

    manifest = {
        "format_version": FORMAT_VERSION,
        "stage": "tokenize",
        "genesis_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer.describe(),
        "dtype": dtype.name,
        "eos_id": tokenizer.eos_id,
        "split_strategy": strategy,
        "splits": summaries,
        "clean_manifest": {"path": str(clean_manifest_path), "sha256": file_sha256(clean_manifest_path)},
        "sources": [{k: s.get(k) for k in ("name", "license", "rights_basis", "commercial_use")}
                    for s in clean_manifest["sources"]],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="output directory of genesis.data.clean")
    p.add_argument("--tokenizer", default="byte", help='"byte" or a BPE JSON path')
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens-per-shard", type=int, default=100_000_000)
    args = p.parse_args()
    manifest = tokenize_dataset(args.input, args.tokenizer, args.output, args.max_tokens_per_shard)
    print(json.dumps({k: manifest[k] for k in ("tokenizer", "dtype", "split_strategy", "splits")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
