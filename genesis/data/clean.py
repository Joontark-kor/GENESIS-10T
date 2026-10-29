"""Stage A: sources -> normalized, filtered, PII-masked, deduplicated, split text.

    python -m genesis.data.clean --sources configs/sources.json --output data/clean/v1

Output::

    data/clean/v1/train/part-00000.jsonl.gz   {"id","source","sha256","text"}
    data/clean/v1/val/...  data/clean/v1/test/...
    data/clean/v1/clean_manifest.json         settings + per-source statistics
    data/clean/v1/DATA_CARD.md                human-readable summary

Splits are assigned from each document's content hash, so the assignment is
stable when sources are added and needs no global shuffle (streaming friendly).
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from .. import __version__
from ..text import normalize_text
from .dedup import MinHashDeduper, exact_key
from .filters import QualityConfig, quality_reason
from .pii import mask_pii
from .sources import SourceSpec, iter_documents, load_registry


def assign_split(sha256: str, val_fraction: float, test_fraction: float) -> str:
    bucket = int(sha256[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


class _PartWriter:
    def __init__(self, directory: Path, docs_per_part: int):
        self.directory, self.docs_per_part = directory, docs_per_part
        directory.mkdir(parents=True, exist_ok=True)
        self.parts: list[str] = []
        self._handle = None
        self._in_part = 0
        self.documents = 0
        self.chars = 0

    def write(self, record: dict) -> None:
        if self._handle is None or self._in_part >= self.docs_per_part:
            self.close()
            name = f"part-{len(self.parts):05d}.jsonl.gz"
            self._handle = gzip.open(self.directory / name, "wt", encoding="utf-8")
            self.parts.append(name)
            self._in_part = 0
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._in_part += 1
        self.documents += 1
        self.chars += len(record["text"])

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def iter_clean(clean_dir: str | Path, split: str):
    """Yield records of one split from a cleaned directory."""
    for part in sorted((Path(clean_dir) / split).glob("part-*.jsonl.gz")):
        with gzip.open(part, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def clean(sources: list[SourceSpec], output: Path, *, base_dir: Path | None = None,
          quality: QualityConfig | None = None, val_fraction: float = 0.01,
          test_fraction: float = 0.01, near_dedup: bool = True, mask: bool = True,
          dedup_threshold: float = 0.8, docs_per_part: int = 50_000,
          samples_per_reason: int = 3, exclude_sha256: set[str] | None = None) -> dict:
    if not 0 <= val_fraction < 0.5 or not 0 <= test_fraction < 0.5:
        raise ValueError("val/test fractions must be in [0, 0.5)")
    quality = quality or QualityConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    writers = {split: _PartWriter(output / split, docs_per_part) for split in ("train", "val", "test")}
    exact_seen: set[str] = set()
    deduper = MinHashDeduper(threshold=dedup_threshold) if near_dedup else None
    per_source: dict[str, dict] = {}
    removed_samples: dict[str, list[dict]] = {}

    try:
        for spec in sources:
            rules = quality.with_overrides(spec.filters)
            stats = {"seen": 0, "kept": 0, "removed": Counter(), "pii_masked": Counter(),
                     "kept_chars": 0, "splits": Counter()}
            for doc in iter_documents(spec, base_dir):
                stats["seen"] += 1
                text = normalize_text(doc["text"])
                reason = "empty" if not text else quality_reason(text, spec.language, rules)
                if reason is None:
                    key = exact_key(text)
                    if key in exact_seen:
                        reason = "exact_duplicate"
                    else:
                        exact_seen.add(key)
                        if deduper is not None and deduper.is_duplicate(text):
                            reason = "near_duplicate"
                if reason is not None:
                    stats["removed"][reason] += 1
                    bucket = removed_samples.setdefault(reason, [])
                    if len(bucket) < samples_per_reason:
                        bucket.append({"source": spec.name, "id": doc["doc_id"], "preview": text[:160]})
                    continue
                if mask:
                    text, found = mask_pii(text)
                    stats["pii_masked"].update({k: v for k, v in found.items() if v})
                sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if exclude_sha256 and sha in exclude_sha256:  # e.g. benchmark-contaminated docs
                    stats["removed"]["blocklisted"] += 1
                    continue
                split = assign_split(sha, val_fraction, test_fraction)
                writers[split].write({"id": doc["doc_id"], "source": spec.name, "sha256": sha, "text": text})
                stats["kept"] += 1
                stats["kept_chars"] += len(text)
                stats["splits"][split] += 1
            per_source[spec.name] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    finally:
        for writer in writers.values():
            writer.close()

    manifest = {
        "stage": "clean",
        "genesis_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {"quality": asdict(quality), "val_fraction": val_fraction,
                     "test_fraction": test_fraction, "near_dedup": near_dedup,
                     "dedup_threshold": dedup_threshold, "pii_masking": mask,
                     "blocklisted_sha256": len(exclude_sha256 or ()),
                     "normalization": "NFC + newline/control/zero-width cleanup",
                     "split_method": "content-hash buckets after global dedup"},
        "sources": [{**spec.provenance(), "filters": spec.filters} for spec in sources],
        "statistics": per_source,
        "splits": {name: {"documents": w.documents, "chars": w.chars, "parts": w.parts}
                   for name, w in writers.items()},
        "removed_examples": removed_samples,
        "limitations": [
            "license fields are asserted by the registry author, not verified",
            "regex PII masking misses names, addresses, account numbers",
            "heuristic quality filters; review removed_examples before scaling",
        ],
    }
    (output / "clean_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    (output / "DATA_CARD.md").write_text(render_data_card(manifest), encoding="utf-8")
    return manifest


def render_data_card(manifest: dict) -> str:
    lines = [f"# Data card ({manifest['created_at'][:10]})", "",
             "Generated by `genesis.data.clean`. Edit the registry, not this file.", "",
             "## Sources", "",
             "| Source | License | Rights basis | Commercial use | Seen | Kept | Kept chars |",
             "|---|---|---|---|---:|---:|---:|"]
    for src in manifest["sources"]:
        stat = manifest["statistics"].get(src["name"], {})
        lines.append(f"| {src['name']} | {src['license']} | {src['rights_basis']} | "
                     f"{src['commercial_use']} | {stat.get('seen', 0):,} | {stat.get('kept', 0):,} | "
                     f"{stat.get('kept_chars', 0):,} |")
    lines += ["", "## Removed documents by reason", "", "| Source | Reason | Count |", "|---|---|---:|"]
    for name, stat in manifest["statistics"].items():
        for reason, count in sorted(stat["removed"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {reason} | {count:,} |")
    lines += ["", "## PII placeholders inserted", "", "| Source | Kind | Count |", "|---|---|---:|"]
    for name, stat in manifest["statistics"].items():
        for kind, count in stat["pii_masked"].items():
            lines.append(f"| {name} | {kind} | {count:,} |")
    lines += ["", "## Splits", "", "| Split | Documents | Characters |", "|---|---:|---:|"]
    for split, info in manifest["splits"].items():
        lines.append(f"| {split} | {info['documents']:,} | {info['chars']:,} |")
    lines += ["", "## Attribution", ""]
    for src in manifest["sources"]:
        if src.get("attribution") or src.get("license_url"):
            lines.append(f"- **{src['name']}**: {src.get('attribution') or ''} {src.get('license_url') or ''}".rstrip())
    lines += ["", "## Known limitations", ""] + [f"- {item}" for item in manifest["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", type=Path, required=True, help="registry JSON (see configs/sources.example.json)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--test-fraction", type=float, default=0.01)
    p.add_argument("--dedup-threshold", type=float, default=0.8)
    p.add_argument("--no-near-dedup", action="store_true")
    p.add_argument("--no-pii-mask", action="store_true")
    p.add_argument("--exclude-sha256", type=Path, help="file of document hashes to drop (genesis.contamination)")
    args = p.parse_args()
    exclude = set(args.exclude_sha256.read_text().split()) if args.exclude_sha256 else None
    manifest = clean(load_registry(args.sources), args.output, val_fraction=args.val_fraction,
                     test_fraction=args.test_fraction, near_dedup=not args.no_near_dedup,
                     mask=not args.no_pii_mask, dedup_threshold=args.dedup_threshold,
                     exclude_sha256=exclude)
    print(json.dumps({"splits": manifest["splits"], "statistics": manifest["statistics"]},
                     ensure_ascii=False, indent=2))
    print(f"data card: {args.output / 'DATA_CARD.md'}")


if __name__ == "__main__":
    main()
