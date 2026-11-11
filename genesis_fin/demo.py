"""Terminal demo (4-a): ask a calculation question about a listed company's filing and get the value with its
formula, its source and a check against the filing's own numbers.

    # offline, from records already collected with ``genesis_fin.build_tasks fetch``
    python -m genesis_fin.demo --records data/applied/fin_v1/records.jsonl \
        --endpoint http://localhost:8000/v1 --model Qwen/Qwen3-4B "삼성전자 2024년 부채비율은?"

    # live: DART_API_KEY set, the statement is fetched (and cached) first
    python -m genesis_fin.demo --endpoint http://localhost:8000/v1 --model fin "삼성전자 2024년 영업이익률은?"

Scope (docs/12, decision 2026-11-11): polarbear is called only for structured calculation questions whose values
are given. The demo recognizes the six ratios of ``genesis_fin.build_tasks`` and says so for anything else instead
of sending it to the model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable

from .build_tasks import SYSTEM, make_tasks

KEYWORDS = (  # longest first, so "자기자본비율" is not read as "자본"
    ("영업이익률", "operating_margin"), ("순이익률", "net_margin"), ("당기순이익률", "net_margin"),
    ("자기자본비율", "equity_ratio"), ("부채비율", "debt_ratio"), ("roe", "roe"),
    ("매출성장률", "revenue_growth"), ("매출증가율", "revenue_growth"), ("매출액증가율", "revenue_growth"),
    ("매출성장", "revenue_growth"), ("매출증가", "revenue_growth"),
)
SUPPORTED = "영업이익률, 순이익률, 부채비율, 자기자본비율, ROE, 매출 증가율"
DART_VIEW = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo="


@dataclass
class DemoAnswer:
    ok: bool                        # a value was produced and checked
    message: str                    # what to print when ok is False
    corp_name: str = ""
    year: int = 0
    kind: str = ""
    value: str | None = None        # the calculated value as text
    formula: str | None = None
    source: str = ""
    check: str = ""                 # "pass" / "fail" / "no value"
    expected: float | None = None
    warnings: list[str] = field(default_factory=list)


def parse_question(text: str) -> tuple[str | None, int | None, str]:
    """(metric kind, year, text without spaces) from a Korean question."""
    squashed = re.sub(r"\s+", "", text).lower()
    kind = next((k for word, k in KEYWORDS if word in squashed), None)
    year = re.search(r"(20\d\d)년", squashed)
    return kind, int(year.group(1)) if year else None, squashed


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_in_records(records: list[dict], squashed: str, year: int | None) -> dict | None:
    names = sorted({r["corp_name"] for r in records}, key=len, reverse=True)
    name = next((n for n in names if re.sub(r"\s+", "", n).lower() in squashed), None)
    if name is None:
        return None
    mine = [r for r in records if r["corp_name"] == name]
    if year is None:
        return max(mine, key=lambda r: r["year"])
    return next((r for r in mine if r["year"] == year), None)


def source_line(record: dict) -> str:
    basis = "연결" if record.get("fs_div") == "CFS" else "별도"
    rcept = record.get("rcept_no", "")
    link = f", {DART_VIEW}{rcept}" if rcept else ""
    return f"OpenDART {record['year']}년 사업보고서 {basis}재무제표 (접수번호 {rcept or '없음'}{link})"


def answer(question: str, find: Callable[[str, int | None], dict | None], ask: Callable[[list[dict]], object],
           tolerance: float = 0.01) -> DemoAnswer:
    kind, year, squashed = parse_question(question)
    if kind is None:
        return DemoAnswer(False, f"계산 질문으로 읽지 못했습니다. 지금은 {SUPPORTED}을 물을 수 있습니다. "
                                 "(계산이 아닌 질문은 이 데모의 범위 밖입니다.)")
    record = find(squashed, year)
    if record is None:
        when = f"{year}년 " if year else ""
        return DemoAnswer(False, f"{when}공시를 찾지 못했습니다. 회사 이름과 연도를 확인하거나 수집된 기록을 늘려 주세요.")
    tasks = [t for t in make_tasks(record, "ko") if t["kind"] == kind]
    if not tasks:
        return DemoAnswer(False, f"{record['corp_name']} {record['year']}년 공시에 {kind} 계산에 필요한 항목이 없습니다.")
    task = tasks[0]
    result = ask([{"role": "system", "content": SYSTEM["ko"]}, {"role": "user", "content": task["question"]}])
    out = DemoAnswer(True, "", record["corp_name"], record["year"], kind, result.answer, result.formula,
                     source_line(record), expected=task["gold"], warnings=list(result.warnings))
    if result.answer is None:
        out.check = "no value"
        return out
    try:
        got = float(str(result.answer).replace(",", "").rstrip("%"))
    except ValueError:
        out.check = "no value"
        return out
    out.check = "pass" if abs(got - task["gold"]) <= tolerance else "fail"
    return out


def render(a: DemoAnswer) -> str:
    if not a.ok:
        return a.message
    lines = [f"{a.corp_name} {a.year}년 {a.kind}", f"출처: {a.source}"]
    lines.append(f"식: {a.formula}" if a.formula else "식: (모델이 식을 쓰지 않았습니다)")
    lines.append(f"답: {a.value if a.value is not None else '(계산된 값 없음)'}")
    if a.check == "pass":
        lines.append(f"검산: 통과 (표의 숫자로 다시 계산한 값 {a.expected:.2f}와 일치)")
    elif a.check == "fail":
        lines.append(f"검산: 불일치 — 표의 숫자로 다시 계산하면 {a.expected:.2f}입니다. 이 답은 쓰지 마세요.")
    else:
        lines.append("검산: 할 수 없음 (계산된 값이 없습니다)")
    lines += [f"확인할 점: {w}" for w in a.warnings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--records", default=None, help="records.jsonl from genesis_fin.build_tasks fetch (offline mode)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--cache-dir", default="data/applied/cache/dart")
    args = p.parse_args(argv)

    from genesis.serving.calc import CalcClient
    from genesis.serving.endpoint import EndpointClient

    if args.records:
        records = load_records(Path(args.records))
        find = lambda squashed, year: find_in_records(records, squashed, year)  # noqa: E731
    else:
        from .dart import DartClient
        client = DartClient(cache_dir=args.cache_dir)  # reads DART_API_KEY
        companies = client.corp_codes(listed_only=True)

        def find(squashed: str, year: int | None):
            names = sorted(companies, key=lambda c: len(c["corp_name"]), reverse=True)
            company = next((c for c in names if re.sub(r"\s+", "", c["corp_name"]).lower() in squashed), None)
            if company is None:
                return None
            if year is None:
                return None  # live mode needs the year: do not guess which filing is meant
            return client.company_year(company, year)

    calc = CalcClient(EndpointClient(args.endpoint, api_key=args.api_key or os.environ.get("GENESIS_API_KEY")),
                      args.model, template_kwargs={"enable_thinking": False})
    print(render(answer(args.question, find, calc.ask)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
