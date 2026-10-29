"""Stage B: train a Korean-first byte-level BPE on the *train* split only.

    python -m genesis.train_tokenizer --input data/clean/v1 \
        --vocab-size 32768 --output tokenizers/ko32k-v1.json

Training only on the train split keeps validation/test text unseen. A sidecar
``<output>.meta.json`` records the corpus manifest hash and settings, so the
tokenizer itself has provenance. Pick the vocabulary size from the report:
a larger vocabulary shortens Korean sequences but costs embedding parameters,
which matters for small models (32k is a sensible start under ~1B params).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .data.clean import iter_clean
from .tokenizer import file_sha256, train_bpe
from .tokenizer_report import report, to_markdown


def corpus(clean_dir: Path, max_docs: int | None, max_chars: int):
    for index, record in enumerate(iter_clean(clean_dir, "train")):
        if max_docs is not None and index >= max_docs:
            return
        yield record["text"][:max_chars]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="output directory of genesis.data.clean")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-docs", type=int, default=None, help="cap documents (e.g. 2_000_000)")
    p.add_argument("--max-chars-per-doc", type=int, default=20_000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    tokenizer = train_bpe(corpus(args.input, args.max_docs, args.max_chars_per_doc),
                          args.vocab_size, args.output, args.min_frequency)
    manifest_path = args.input / "clean_manifest.json"
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer.describe(),
        "requested_vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "trained_on": {"clean_dir": str(args.input), "split": "train",
                       "clean_manifest_sha256": file_sha256(manifest_path),
                       "max_docs": args.max_docs, "max_chars_per_doc": args.max_chars_per_doc},
    }
    held_out = [r["text"] for _, r in zip(range(2000), iter_clean(args.input, "val"))]
    if held_out:
        meta["held_out_report"] = report(held_out, ["byte", str(args.output)])
        print(to_markdown(meta["held_out_report"]))
    Path(str(args.output) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"saved {args.output} (vocab={tokenizer.vocab_size}, sha256={tokenizer.sha256[:12]})")


if __name__ == "__main__":
    main()
