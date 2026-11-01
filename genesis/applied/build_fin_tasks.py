"""Financial-statement tasks with computed answers, from OpenDART data.

    # 1) collect normalized statements (needs DART_API_KEY; cached, resumable)
    python -m genesis.applied.build_fin_tasks fetch --years 2023 2024 2025 \
        --companies 400 --out data/applied/fin_v1

    # 2) turn them into training conversations and evaluation items
    python -m genesis.applied.build_fin_tasks build --out data/applied/fin_v1 --langs ko

Every answer is computed from the numbers shown in the question, so accuracy
is checked by arithmetic, not by judgement. Companies (not rows) are split
into train/val/test, so a test company never appears in training.

Task kinds: ``extract`` (read one line item) and the ratios
``operating_margin``, ``net_margin``, ``debt_ratio``, ``roe``,
``equity_ratio``, ``revenue_growth`` (percent, two decimals).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..data.clean import assign_split
from ..tokenizer import file_sha256
from .dart import DartClient, QuotaExceeded

METRICS = ("revenue", "operating_income", "net_income", "total_assets", "total_liabilities", "total_equity")

LABELS = {
    "ko": {"revenue": "매출액", "operating_income": "영업이익", "net_income": "당기순이익",
           "total_assets": "자산총계", "total_liabilities": "부채총계", "total_equity": "자본총계"},
    "ja": {"revenue": "売上高", "operating_income": "営業利益", "net_income": "当期純利益",
           "total_assets": "総資産", "total_liabilities": "負債合計", "total_equity": "純資産合計"},
}
UNITS = {("ko", "KRW"): "백만원", ("ko", "JPY"): "백만엔", ("ja", "KRW"): "百万ウォン", ("ja", "JPY"): "百万円"}
SYSTEM = {
    "ko": "당신은 기업 재무제표를 정확히 읽고 계산하는 분석 보조입니다. 주어진 표의 숫자만 사용하고, "
          "투자 권유는 하지 않습니다. 마지막 줄에 '정답: 값' 형식으로 답하세요.",
    "ja": "あなたは企業の財務諸表を正確に読み取り計算する分析アシスタントです。与えられた表の数値だけを使い、"
          "投資の勧誘はしません。最後の行に「答え: 値」の形式で答えてください。",
}
ANSWER_TAG = {"ko": "정답", "ja": "答え"}

# kind -> (numerator(s), denominator, ko name, ja name, ko formula, ja formula)
RATIOS = {
    "operating_margin": ("operating_income", "revenue", "영업이익률", "営業利益率",
                         "영업이익 ÷ 매출액 × 100", "営業利益 ÷ 売上高 × 100"),
    "net_margin": ("net_income", "revenue", "순이익률", "純利益率",
                   "당기순이익 ÷ 매출액 × 100", "当期純利益 ÷ 売上高 × 100"),
    "debt_ratio": ("total_liabilities", "total_equity", "부채비율", "負債比率",
                   "부채총계 ÷ 자본총계 × 100", "負債合計 ÷ 純資産合計 × 100"),
    "roe": ("net_income", "total_equity", "ROE(기말 자본 기준)", "ROE（期末純資産ベース）",
            "당기순이익 ÷ 기말 자본총계 × 100", "当期純利益 ÷ 期末純資産合計 × 100"),
    "equity_ratio": ("total_equity", "total_assets", "자기자본비율", "自己資本比率",
                     "자본총계 ÷ 자산총계 × 100", "純資産合計 ÷ 総資産 × 100"),
}


# ------------------------------------------------------------------ fetch

def sample_companies(companies: list[dict], n: int, seed: str = "fin_v1") -> list[dict]:
    """Deterministic pseudo-random sample, stable when the company list grows."""
    keyed = sorted(companies, key=lambda c: hashlib.sha256(f"{seed}:{c['corp_code']}".encode()).hexdigest())
    return keyed[:n]


def fetch(out: Path, years: list[int], n_companies: int | None, corp_codes: list[str] | None,
          client: DartClient | None = None) -> dict:
    client = client or DartClient(cache_dir=out.parent / "cache" / "dart")
    companies = client.corp_codes(listed_only=True)
    if corp_codes:
        wanted = set(corp_codes)
        companies = [c for c in companies if c["corp_code"] in wanted]
    elif n_companies:
        companies = sample_companies(companies, n_companies)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["corp_code"], record["year"]))
    stats = Counter()
    stopped = None
    with open(path, "a", encoding="utf-8") as handle:
        for company in companies:
            for year in years:
                if (company["corp_code"], year) in done:
                    stats["cached"] += 1
                    continue
                try:
                    record = client.company_year(company, year)
                except QuotaExceeded as exc:
                    stopped = str(exc)
                    break
                if record is None:
                    stats["no_data"] += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                stats["written"] += 1
            if stopped:
                break
    return {"companies": len(companies), "years": years, **stats, "api_calls": client.calls,
            "stopped": stopped, "records": str(path)}


# ------------------------------------------------------------------ build

def to_millions(value: int) -> int:
    return int(round(value / 1_000_000))


def table(record: dict, lang: str, unit: str) -> str:
    year = record["year"]
    labels = LABELS[lang]
    current = record["metrics"].get("current", {})
    prior = record["metrics"].get("prior", {})
    if lang == "ko":
        basis = "연결" if record.get("fs_div") == "CFS" else "별도"
        head = f"{record['corp_name']} {year}년 {basis}재무제표 요약 (단위: {unit})"
        cols = f"| 항목 | {year}년 | {year - 1}년 |"
    else:
        basis = "連結" if record.get("fs_div") == "CFS" else "単体"
        head = f"{record['corp_name']} {year}年度 {basis}財務諸表の要約（単位: {unit}）"
        cols = f"| 項目 | {year}年度 | {year - 1}年度 |"
    lines = [head, "", cols, "|---|---:|---:|"]
    for metric in METRICS:
        if metric not in current and metric not in prior:
            continue
        cell = lambda d: f"{to_millions(d[metric]):,}" if metric in d else "-"  # noqa: E731
        lines.append(f"| {labels[metric]} | {cell(current)} | {cell(prior)} |")
    return "\n".join(lines)


def _pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2)


def make_tasks(record: dict, lang: str = "ko") -> list[dict]:
    """All tasks one statement supports, computed from the rounded numbers shown."""
    currency = record.get("currency") if record.get("currency") in {"KRW", "JPY"} else "KRW"
    unit = UNITS[(lang, currency)]
    context = table(record, lang, unit)
    name, year, tag = record["corp_name"], record["year"], ANSWER_TAG[lang]
    shown = {k: to_millions(v) for k, v in record["metrics"].get("current", {}).items()}
    shown_prior = {k: to_millions(v) for k, v in record["metrics"].get("prior", {}).items()}
    tasks = []

    def add(kind: str, question: str, reasoning: str, gold: float, unit_label: str, tolerance: float):
        answer_value = f"{gold:,.2f}%" if unit_label == "%" else f"{int(gold):,}{unit_label}"
        tasks.append({
            "kind": kind, "lang": lang, "question": f"{context}\n\n{question}",
            "answer": f"{reasoning}\n{tag}: {answer_value}", "gold": gold, "unit": unit_label,
            "tolerance": tolerance,
        })

    for kind, (num, den, ko_name, ja_name, ko_formula, ja_formula) in RATIOS.items():
        if num not in shown or den not in shown or shown[den] <= 0:
            continue
        value = _pct(shown[num], shown[den])
        if lang == "ko":
            q = f"질문: {name}의 {year}년 {ko_name}({ko_formula})을 소수점 둘째 자리까지 계산하세요."
            r = f"{ko_name} = {ko_formula}\n= {shown[num]:,} ÷ {shown[den]:,} × 100\n= {value:.2f}%"
        else:
            q = f"質問: {name}の{year}年度の{ja_name}（{ja_formula}）を小数点第2位まで計算してください。"
            r = f"{ja_name} = {ja_formula}\n= {shown[num]:,} ÷ {shown[den]:,} × 100\n= {value:.2f}%"
        add(kind, q, r, value, "%", 0.01)

    if "revenue" in shown and shown_prior.get("revenue", 0) > 0:
        now, before = shown["revenue"], shown_prior["revenue"]
        value = _pct(now - before, before)
        label = LABELS[lang]["revenue"]
        if lang == "ko":
            q = f"질문: {name}의 {year}년 {label} 증가율(전년 대비, %)을 소수점 둘째 자리까지 계산하세요."
            r = f"{label} 증가율 = ({year}년 - {year - 1}년) ÷ {year - 1}년 × 100\n= ({now:,} - {before:,}) ÷ {before:,} × 100\n= {value:.2f}%"
        else:
            q = f"質問: {name}の{year}年度の{label}成長率（前年度比、%）を小数点第2位まで計算してください。"
            r = f"{label}成長率 = ({year}年度 - {year - 1}年度) ÷ {year - 1}年度 × 100\n= ({now:,} - {before:,}) ÷ {before:,} × 100\n= {value:.2f}%"
        add("revenue_growth", q, r, value, "%", 0.01)

    available = [m for m in METRICS if m in shown]
    seed = int(hashlib.sha256(f"{record['corp_code']}:{year}".encode()).hexdigest()[:8], 16)
    for offset in range(min(2, len(available))):
        metric = available[(seed + offset * 3) % len(available)]
        if any(t["kind"] == "extract" and t.get("metric") == metric for t in tasks):
            continue
        label, value = LABELS[lang][metric], shown[metric]
        if lang == "ko":
            q = f"질문: {name}의 {year}년 {label}은(는) 얼마입니까? 단위는 {unit}입니다."
            r = f"표에서 {year}년 {label}은(는) {value:,}{unit}입니다."
        else:
            q = f"質問: {name}の{year}年度の{label}はいくらですか。単位は{unit}です。"
            r = f"表によると{year}年度の{label}は{value:,}{unit}です。"
        add("extract", q, r, float(value), unit, 0.5)
        tasks[-1]["metric"] = metric
    for i, task in enumerate(tasks):
        task["id"] = f"{record['corp_code']}-{year}-{lang}-{task['kind']}-{i}"
        task.update({"corp_code": record["corp_code"], "corp_name": name, "year": year})
    return tasks


def company_split(corp_code: str, val_fraction: float = 0.1, test_fraction: float = 0.1) -> str:
    return assign_split(hashlib.sha256(f"company:{corp_code}".encode()).hexdigest(), val_fraction, test_fraction)


def build(out: Path, langs: list[str], val_fraction: float = 0.1, test_fraction: float = 0.1) -> dict:
    records_path = out / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no records in {records_path}; run the fetch step first")
    files = {name: open(out / name, "w", encoding="utf-8")
             for name in ("sft_train.jsonl", "sft_val.jsonl", "eval_val.jsonl", "eval_test.jsonl")}
    counts: Counter = Counter()
    companies: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    try:
        for record in records:
            split = company_split(record["corp_code"], val_fraction, test_fraction)
            companies[split].add(record["corp_code"])
            for lang in langs:
                for task in make_tasks(record, lang):
                    prompt = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": task["question"]}]
                    meta = {k: task[k] for k in ("id", "kind", "lang", "corp_code", "year")}
                    if split in ("train", "val"):
                        files[f"sft_{split}.jsonl"].write(json.dumps(
                            {"messages": prompt + [{"role": "assistant", "content": task["answer"]}], "meta": meta},
                            ensure_ascii=False) + "\n")
                    if split in ("val", "test"):
                        files[f"eval_{split}.jsonl"].write(json.dumps(
                            {**meta, "messages": prompt, "gold": task["gold"], "unit": task["unit"],
                             "tolerance": task["tolerance"], "reference": task["answer"]},
                            ensure_ascii=False) + "\n")
                    counts[f"{split}:{task['kind']}"] += 1
                    counts[split] += 1
    finally:
        for handle in files.values():
            handle.close()
    manifest = {
        "stage": "applied-fin-tasks", "created_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records), "records_sha256": file_sha256(records_path), "langs": langs,
        "split_by": "company (corp_code hash)", "companies": {k: len(v) for k, v in companies.items()},
        "tasks": dict(sorted(counts.items())),
        "source": {"name": "OpenDART fnlttSinglAcntAll", "url": "https://opendart.fss.or.kr",
                   "license": "OpenDART 이용약관", "rights_basis": "contract",
                   "notes": "Public disclosures via the official API; check the terms before redistributing data."},
        "units": "amounts shown in millions of the reporting currency; answers computed from the shown values",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download statements from OpenDART (needs DART_API_KEY)")
    f.add_argument("--out", type=Path, default=Path("data/applied/fin_v1"))
    f.add_argument("--years", type=int, nargs="+", required=True)
    f.add_argument("--companies", type=int, default=300, help="sample size of listed companies")
    f.add_argument("--corp-codes", nargs="*", help="explicit corp_codes instead of a sample")
    b = sub.add_parser("build", help="records.jsonl -> SFT and evaluation files")
    b.add_argument("--out", type=Path, default=Path("data/applied/fin_v1"))
    b.add_argument("--langs", nargs="+", default=["ko"], choices=sorted(LABELS))
    args = p.parse_args()
    if args.command == "fetch":
        print(json.dumps(fetch(args.out, args.years, args.companies, args.corp_codes), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build(args.out, args.langs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
