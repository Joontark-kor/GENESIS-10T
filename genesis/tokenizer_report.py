"""Compare tokenizers on held-out Korean text (Gate 1 evidence).

    python -m genesis.tokenizer_report --input data/clean/v1 --split val \
        --tokenizers byte tokenizers/ko32k-v1.json

Reports, per tokenizer:
* tokens per character and per Hangul syllable (lower = more text per context)
* bytes per token (higher = more compression)
* exact round-trip rate after NFC normalization (must be 100%)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import unicodedata

from .text import is_hangul_syllable
from .tokenizer import load_tokenizer


def iter_texts(path: Path, split: str, max_docs: int):
    if path.is_dir() and (path / "clean_manifest.json").exists():
        from .data.clean import iter_clean
        source = (r["text"] for r in iter_clean(path, split))
    elif path.is_dir():
        source = (p.read_text(encoding="utf-8") for p in sorted(path.rglob("*.txt")))
    else:
        source = iter([path.read_text(encoding="utf-8")])
    for index, text in enumerate(source):
        if index >= max_docs:
            break
        yield unicodedata.normalize("NFC", text)


def report(texts: list[str], specs: list[str]) -> list[dict]:
    chars = sum(len(t) for t in texts)
    syllables = sum(1 for t in texts for ch in t if is_hangul_syllable(ch))
    raw_bytes = sum(len(t.encode("utf-8")) for t in texts)
    rows = []
    for spec in specs:
        tokenizer = load_tokenizer(spec)
        encoded = tokenizer.encode_batch(texts)
        tokens = sum(len(ids) for ids in encoded)
        exact = sum(tokenizer.decode(ids) == text for ids, text in zip(encoded, texts))
        rows.append({
            "tokenizer": spec, "vocab_size": tokenizer.vocab_size, "documents": len(texts),
            "tokens": tokens,
            "tokens_per_char": round(tokens / chars, 4) if chars else None,
            "tokens_per_hangul_syllable": round(tokens / syllables, 4) if syllables else None,
            "bytes_per_token": round(raw_bytes / tokens, 3) if tokens else None,
            "roundtrip_exact": f"{exact}/{len(texts)}",
        })
    return rows


def to_markdown(rows: list[dict]) -> str:
    header = ["tokenizer", "vocab_size", "tokens_per_char", "tokens_per_hangul_syllable",
              "bytes_per_token", "roundtrip_exact"]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(str(row[h]) for h in header) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="clean dir, txt dir, or one txt file")
    p.add_argument("--split", default="val")
    p.add_argument("--tokenizers", nargs="+", default=["byte"])
    p.add_argument("--max-docs", type=int, default=5000)
    p.add_argument("--json", type=Path, help="also write the rows here")
    args = p.parse_args()
    texts = list(iter_texts(args.input, args.split, args.max_docs))
    if not texts:
        raise ValueError("no text found")
    rows = report(texts, args.tokenizers)
    print(to_markdown(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
