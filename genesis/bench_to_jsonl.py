"""Download Korean benchmark items as JSONL for ``genesis.contamination``.

    pip install datasets
    python -m genesis.bench_to_jsonl --benchmarks kobest haerae click kmmlu --output data/bench

Writes ``data/bench/<benchmark>.jsonl`` with one item per line:
``{"benchmark", "config", "split", "index", "text"}`` where ``text`` joins the
fields a model sees (passage, question, choices). Then:

    python -m genesis.contamination --train data/clean/v1 \
        --benchmark data/bench/kobest.jsonl --fields text --report reports/contam_kobest.json

Dataset ids and fields follow the lm-evaluation-harness task definitions
(lm_eval 0.4.x). Check each dataset's license on its Hugging Face page before
redistributing anything derived from it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

KMMLU_SUBJECTS = (
    "Accounting,Agricultural-Sciences,Aviation-Engineering-and-Maintenance,Biology,Chemical-Engineering,"
    "Chemistry,Civil-Engineering,Computer-Science,Construction,Criminal-Law,Ecology,Economics,Education,"
    "Electrical-Engineering,Electronics-Engineering,Energy-Management,Environmental-Science,Fashion,"
    "Food-Processing,Gas-Technology-and-Engineering,Geomatics,Health,Industrial-Engineer,"
    "Information-Technology,Interior-Architecture-and-Design,Korean-History,Law,"
    "Machine-Design-and-Manufacturing,Management,Maritime-Engineering,Marketing,Materials-Engineering,Math,"
    "Mechanical-Engineering,Nondestructive-Testing,Patent,Political-Science-and-Sociology,Psychology,"
    "Public-Safety,Railway-and-Automotive-Engineering,Real-Estate,Refrigerating-Machinery,Social-Welfare,"
    "Taxation,Telecommunications-and-Wireless-Technology").split(",")

# benchmark -> list of (dataset path, config name or None, splits, fields)
PRESETS: dict[str, list[tuple[str, str | None, tuple[str, ...], tuple[str, ...]]]] = {
    "kobest": [
        ("skt/kobest_v1", "boolq", ("validation", "test"), ("paragraph", "question")),
        ("skt/kobest_v1", "copa", ("validation", "test"), ("premise", "alternative_1", "alternative_2")),
        ("skt/kobest_v1", "hellaswag", ("validation", "test"),
         ("context", "ending_1", "ending_2", "ending_3", "ending_4")),
        ("skt/kobest_v1", "sentineg", ("validation", "test"), ("sentence",)),
        ("skt/kobest_v1", "wic", ("validation", "test"), ("context_1", "context_2")),
    ],
    "haerae": [("HAERAE-HUB/HAE_RAE_BENCH", name, ("test",), ("query",))
               for name in ("general_knowledge", "history", "loan_words", "rare_words", "standard_nomenclature")],
    "click": [("EunsuKim/CLIcK", None, ("train",), ("paragraph", "question", "choices"))],
    "kmmlu": [("HAERAE-HUB/KMMLU", subject, ("test",), ("question", "A", "B", "C", "D"))
              for subject in KMMLU_SUBJECTS],
}


def item_text(record: dict, fields: Iterable[str]) -> str:
    parts = []
    for name in fields:
        value = record.get(name)
        if isinstance(value, list):
            parts.extend(str(v) for v in value if v is not None)
        elif value not in (None, ""):
            parts.append(str(value))
    return "\n".join(parts)


def _hf_loader(path: str, name: str | None, split: str):
    from datasets import load_dataset
    return load_dataset(path, name, split=split)


def export_benchmark(benchmark: str, output_dir: Path,
                     loader: Callable[[str, str | None, str], Iterable[dict]] = _hf_loader) -> dict:
    if benchmark not in PRESETS:
        raise ValueError(f"unknown benchmark {benchmark!r}; choose from {sorted(PRESETS)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{benchmark}.jsonl"
    counts: dict[str, int] = {}
    with open(path, "w", encoding="utf-8") as handle:
        for dataset, config, splits, fields in PRESETS[benchmark]:
            for split in splits:
                key = f"{config or 'default'}/{split}"
                counts[key] = 0
                for index, record in enumerate(loader(dataset, config, split)):
                    text = item_text(record, fields)
                    if not text.strip():
                        continue
                    handle.write(json.dumps({"benchmark": benchmark, "dataset": dataset, "config": config,
                                             "split": split, "index": index, "text": text},
                                            ensure_ascii=False) + "\n")
                    counts[key] += 1
    return {"benchmark": benchmark, "file": str(path), "items": sum(counts.values()), "by_split": counts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", nargs="+", default=["kobest", "haerae", "click"], choices=sorted(PRESETS))
    p.add_argument("--output", type=Path, default=Path("data/bench"))
    args = p.parse_args()
    for name in args.benchmarks:
        summary = export_benchmark(name, args.output)
        print(f"{name}: {summary['items']:,} items -> {summary['file']}")


if __name__ == "__main__":
    main()
