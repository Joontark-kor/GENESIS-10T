"""Turn rows of public benchmarks into eval items (the data itself is never stored in this repository).

Used on Colab, where the rows are downloaded at run time: KMMLU (HAERAE-HUB/KMMLU, CC-BY-ND-4.0, test split)
and a Korean GSM8K translation (kuotient/gsm8k-ko, license not stated on its page; test split, read only).
Neither is redistributed or modified on disk beyond the eval file written under ``data/`` (not committed).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KMMLU_SUBJECTS = ("Accounting", "Taxation", "Economics")
_SYSTEM_MC = ("당신은 한국어 문제를 푸는 보조입니다. 보기 A, B, C, D 중 하나를 골라 마지막 줄에 "
              "'정답: 알파벳' 형식으로 답하세요.")
_SYSTEM_MATH = ("당신은 수학 문장제를 푸는 보조입니다. 마지막 줄에 '정답: 값' 형식으로, 단위 없이 숫자만 쓰세요.")


def kmmlu_item(row: dict, subject: str, index: int) -> dict:
    options = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
    return {"id": f"kmmlu-{subject.lower()}-{index}", "kind": subject.lower(), "lang": "ko", "scorer": "choice",
            "messages": [{"role": "system", "content": _SYSTEM_MC},
                         {"role": "user", "content": f"{row['question']}\n{options}"}],
            "gold": "ABCD"[int(row["answer"]) - 1], "unit": "", "tolerance": 0, "reference": "정답: " + "ABCD"[int(row["answer"]) - 1]}


def gsm8k_gold(answer: str) -> float | None:
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", answer)
    return float(match.group(1).replace(",", "")) if match else None


def gsm8k_item(row: dict, index: int) -> dict | None:
    gold = gsm8k_gold(row["answer"])
    if gold is None:
        return None
    return {"id": f"gsm8k-ko-{index}", "kind": "gsm8k_ko", "lang": "ko",
            "messages": [{"role": "system", "content": _SYSTEM_MATH}, {"role": "user", "content": row["question"]}],
            "gold": gold, "unit": "", "tolerance": 0.01, "reference": f"정답: {gold:g}"}


def write(items: list[dict], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_test.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    manifest = {"stage": "external-benchmark", "items": len(items), "note": "rows downloaded at run time; not for redistribution"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("kmmlu", "gsm8k-ko"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=300, help="items per subject (kmmlu) or in total (gsm8k-ko)")
    args = p.parse_args()
    from datasets import load_dataset  # downloaded at run time only

    items: list[dict] = []
    if args.kind == "kmmlu":
        for subject in KMMLU_SUBJECTS:
            rows = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
            items += [kmmlu_item(row, subject, i) for i, row in enumerate(rows) if i < args.limit]
    else:
        rows = load_dataset("kuotient/gsm8k-ko", split="test")
        items = [item for i, row in enumerate(rows) if i < args.limit and (item := gsm8k_item(row, i))]
    print(json.dumps(write(items, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
