"""Benchmark contamination check: do benchmark items appear in the training text?

    python -m genesis.contamination --train data/clean/v1 \
        --benchmark data/bench/kobest_boolq.jsonl --fields question paragraph \
        --report reports/contamination_kobest_boolq.json \
        --blocklist reports/contaminated_docs.txt

An item counts as contaminated when any of its word n-grams (default 8,
after NFC/lowercase/punctuation stripping) occurs in a training document.
Items shorter than n words are checked as a whole normalized string.

``--blocklist`` writes the sha256 of every matching training document; pass it
to ``genesis.data.clean --exclude-sha256`` to rebuild the corpus without them.
Report contamination numbers next to any benchmark score you publish.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def _hash(gram: str) -> bytes:
    return hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()


def ngrams(words: list[str], n: int):
    for i in range(len(words) - n + 1):
        yield " ".join(words[i:i + n])


def item_text(record: dict, fields: list[str]) -> str:
    parts = []
    for name in fields:
        value = record.get(name)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    return "\n".join(parts)


class ContaminationIndex:
    def __init__(self, items: list[str], n: int = 8):
        self.n = n
        self.items = items
        self.gram_to_items: dict[bytes, set[int]] = {}
        self.short_items: dict[str, int] = {}
        for idx, text in enumerate(items):
            words = normalize_for_match(text).split()
            if not words:
                continue
            if len(words) < n:
                self.short_items[" ".join(words)] = idx
                continue
            for gram in ngrams(words, n):
                self.gram_to_items.setdefault(_hash(gram), set()).add(idx)

    def match(self, document: str) -> set[int]:
        normalized = normalize_for_match(document)
        words = normalized.split()
        hits: set[int] = set()
        for gram in ngrams(words, self.n):
            found = self.gram_to_items.get(_hash(gram))
            if found:
                hits |= found
        if self.short_items:
            padded = f" {normalized} "
            hits |= {idx for text, idx in self.short_items.items() if f" {text} " in padded}
        return hits


def iter_training_docs(path: Path, splits: list[str]):
    if (path / "clean_manifest.json").exists():
        from .data.clean import iter_clean
        for split in splits:
            for record in iter_clean(path, split):
                yield record["sha256"], record["text"]
    else:
        for file in sorted(path.rglob("*.txt")):
            text = file.read_text(encoding="utf-8")
            yield hashlib.sha256(text.encode("utf-8")).hexdigest(), text


def check(train: Path, items: list[str], n: int = 8, splits: list[str] | None = None,
          max_examples: int = 20) -> tuple[dict, set[str]]:
    index = ContaminationIndex(items, n)
    contaminated: dict[int, list[str]] = {}
    docs_scanned = 0
    bad_docs: set[str] = set()
    for sha, text in iter_training_docs(train, splits or ["train"]):
        docs_scanned += 1
        for idx in index.match(text):
            contaminated.setdefault(idx, []).append(sha)
            bad_docs.add(sha)
    report = {
        "items": len(items), "ngram_n": n, "documents_scanned": docs_scanned,
        "contaminated_items": len(contaminated),
        "contamination_rate": len(contaminated) / len(items) if items else 0.0,
        "matching_training_documents": len(bad_docs),
        "examples": [{"item_index": idx, "item_preview": items[idx][:120], "doc_sha256": shas[:3]}
                     for idx, shas in sorted(contaminated.items())[:max_examples]],
    }
    return report, bad_docs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, required=True, help="clean dir or folder of .txt")
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--benchmark", type=Path, required=True, help="JSONL with one item per line")
    p.add_argument("--fields", nargs="+", required=True, help="fields that make up the item text")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--report", type=Path)
    p.add_argument("--blocklist", type=Path)
    args = p.parse_args()
    items = [item_text(json.loads(line), args.fields)
             for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    report, bad_docs = check(args.train, items, args.n, args.splits)
    report["benchmark"] = str(args.benchmark)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.blocklist:
        args.blocklist.parent.mkdir(parents=True, exist_ok=True)
        existing = set(args.blocklist.read_text().split()) if args.blocklist.exists() else set()
        args.blocklist.write_text("\n".join(sorted(existing | bad_docs)) + "\n")


if __name__ == "__main__":
    main()
