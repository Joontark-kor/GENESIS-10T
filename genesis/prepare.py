"""One-shot preparation of a folder of .txt files you own or are licensed to use.

    python -m genesis.prepare --input data/raw --output data/processed \
        --source self-authored-demo --license self-authored

Runs the same pipeline as the large-scale path (clean -> tokenize) with a
single txt source and permissive quality filters, so the tutorial corpus and a
real corpus produce the same manifest format. For real corpora use a source
registry with ``genesis.data.clean`` and ``genesis.data.tokenize`` instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.clean import clean
from .data.filters import QualityConfig
from .data.sources import SourceSpec
from .data.tokenize import tokenize_dataset


def prepare(input_dir: Path, output_dir: Path, source: str, license_id: str,
            val_fraction: float = 0.1, test_fraction: float = 0.1,
            tokenizer: str = "byte", doc_split: str = "file",
            rights_basis: str = "self-authored", language: str = "any") -> dict:
    if not source.strip() or not license_id.strip():
        raise ValueError("source and license provenance are required")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not any(input_dir.rglob("*.txt")):
        raise FileNotFoundError(f"no .txt files under {input_dir}")
    spec = SourceSpec(name=source, type="txt", path=str(input_dir), license=license_id,
                      rights_basis=rights_basis, language=language, doc_split=doc_split)
    clean_dir = output_dir / "_clean"
    clean_manifest = clean([spec], clean_dir, quality=QualityConfig.permissive(),
                           val_fraction=val_fraction, test_fraction=test_fraction)
    if sum(s["documents"] for s in clean_manifest["splits"].values()) == 0:
        raise ValueError("empty dataset after cleaning and deduplication")
    manifest = tokenize_dataset(clean_dir, tokenizer, output_dir, fallback_fraction=val_fraction)
    manifest["unique_documents"] = sum(s["documents"] for s in clean_manifest["splits"].values())
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--source", required=True, help="Where the text came from")
    p.add_argument("--license", required=True, help="Documented license or your own rights")
    p.add_argument("--rights-basis", default="self-authored")
    p.add_argument("--tokenizer", default="byte", help='"byte" or a BPE JSON path')
    p.add_argument("--doc-split", choices=("file", "paragraph", "line"), default="file")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    args = p.parse_args()
    manifest = prepare(args.input, args.output, args.source, args.license, args.val_fraction,
                       args.test_fraction, args.tokenizer, args.doc_split, args.rights_basis)
    print(json.dumps({k: manifest[k] for k in ("tokenizer", "dtype", "split_strategy", "splits")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
