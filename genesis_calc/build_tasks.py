"""Everyday numeric word problems with computed answers (a second domain next to finance).

    python -m genesis_calc.build_tasks --out data/applied/calc_v1 --per-kind 60 --seed 1

Purpose: test whether the gorilla pipeline (model writes the formula, a calculator computes)
helps outside finance. Every answer is computed from the numbers in the question; each
question also carries one or two distractor numbers the formula must not use.
Eval only (no training split): a second-domain check, not a second adapter.

Kinds: percent_change, discount, tax_inclusive, average, simple_interest, speed_time, unit_convert.

``--set hard`` (data/applied/calc_hard_v1) is a third check with formulas that need powers, roots and
several steps (compound interest, CAGR, BMI, electric power, fuel cost, molarity). A 4B model that
does these in its head should fail more often, so this is where the calculator should matter most.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM = ("당신은 일상 수치 문제를 정확히 계산하는 보조입니다. 질문에 필요한 숫자만 사용하세요. "
          "마지막 줄에 '정답: 값' 형식으로 답하세요.")
KINDS = ("percent_change", "discount", "tax_inclusive", "average", "simple_interest", "speed_time", "unit_convert")


HARD_KINDS = ("compound_interest", "cagr", "bmi", "electric_power", "fuel_cost", "molarity")


def _round(value: float) -> float:
    return round(value, 2)


def make(kind: str, rng: random.Random) -> tuple[str, float, float]:
    """(question, gold, tolerance)."""
    if kind == "percent_change":
        a, b = rng.randint(200, 900) * 10, rng.randint(200, 900) * 10
        year = rng.randint(2019, 2024)
        q = f"{year}년 월 구독자가 {a:,}명이었고 다음 해에는 {b:,}명이 되었다. 증감률은 몇 %인가? (소수 둘째 자리)"
        return q, _round((b - a) / a * 100), 0.01
    if kind == "discount":
        price, pct, coupon = rng.randint(20, 300) * 1000, rng.choice([10, 15, 20, 25, 30, 35]), rng.randint(1, 9) * 1000
        q = (f"정가 {price:,}원짜리 상품이 {pct}% 할인 중이다. 배송비는 {coupon:,}원이다. "
             f"할인된 상품 가격(배송비 제외)은 몇 원인가?")
        return q, _round(price * (100 - pct) / 100), 0.5
    if kind == "tax_inclusive":
        net, rate = rng.randint(10, 900) * 1000, rng.choice([5, 8, 10])
        q = f"공급가액이 {net:,}원이고 부가세율이 {rate}%일 때, 부가세를 포함한 합계는 몇 원인가? (사업자 번호 끝자리: {rng.randint(1000, 9999)})"
        return q, _round(net * (100 + rate) / 100), 0.5
    if kind == "average":
        xs = [rng.randint(40, 100) for _ in range(rng.randint(4, 6))]
        q = f"한 학생의 {len(xs)}개 과목 점수가 {', '.join(map(str, xs))}점이다. 학번은 {rng.randint(1000, 9999)}이다. 평균 점수는 몇 점인가? (소수 둘째 자리)"
        return q, _round(sum(xs) / len(xs)), 0.01
    if kind == "simple_interest":
        p, r, y = rng.randint(10, 500) * 10000, rng.choice([2, 3, 4, 5, 6]), rng.randint(1, 5)
        q = f"원금 {p:,}원을 연 {r}% 단리로 {y}년 맡겼다. 통장 개설 수수료는 {rng.randint(1, 5) * 1000:,}원이다. 만기 이자는 몇 원인가?"
        return q, _round(p * r / 100 * y), 0.5
    if kind == "speed_time":
        v, t = rng.choice([40, 60, 72, 80, 90, 100]), rng.choice([1.5, 2, 2.5, 3, 4])
        q = f"시속 {v}km로 {t}시간 달렸다. 출발 시각은 오전 {rng.randint(6, 11)}시였다. 이동 거리는 몇 km인가?"
        return q, _round(v * t), 0.01
    if kind == "unit_convert":
        km, price = rng.randint(2, 90), rng.randint(2, 9) * 100
        q = f"{km}km는 몇 m인가? (참고: 이 지역 버스 요금은 {price}원이다)"
        return q, float(km * 1000), 0.5
    raise ValueError(kind)


def make_hard(kind: str, rng: random.Random) -> tuple[str, float, float]:
    """(question, gold, tolerance) for the harder set: powers, roots and multi-step formulas."""
    if kind == "compound_interest":
        p, r, y = rng.randint(5, 90) * 100, rng.choice([2.5, 3, 3.5, 4, 4.5, 5]), rng.randint(2, 6)
        q = (f"{p:,}만원을 연 {r}% 복리(1년에 한 번)로 {y}년 맡겼다. 통장 개설 수수료는 {rng.randint(1, 5)}만원이다. "
             f"만기 원리금 합계는 몇 만원인가? (소수 둘째 자리)")
        return q, _round(p * (1 + r / 100) ** y), 0.01
    if kind == "cagr":
        a, y = rng.randint(50, 400) * 10, rng.randint(2, 6)
        b = round(a * rng.uniform(1.2, 3.0))
        q = (f"매출이 {a:,}억원에서 {y}년 뒤 {b:,}억원이 되었다. 사업장은 {rng.randint(3, 40)}곳이다. "
             f"연평균 성장률(CAGR)은 몇 %인가? (소수 둘째 자리)")
        return q, _round(((b / a) ** (1 / y) - 1) * 100), 0.01
    if kind == "bmi":
        kg, cm = rng.randint(45, 110), rng.randint(150, 195)
        q = f"나이 {rng.randint(20, 60)}세, 몸무게 {kg}kg, 키 {cm}cm인 사람의 BMI(체중 kg / 키 m의 제곱)는? (소수 둘째 자리)"
        return q, _round(kg / (cm / 100) ** 2), 0.01
    if kind == "electric_power":
        v, r = rng.choice([5, 12, 24, 110, 220]), rng.choice([4, 8, 10, 22, 47, 100])
        q = f"전압 {v}V를 저항 {r}Ω에 걸었다. 전선 길이는 {rng.randint(2, 30)}m이다. 소비 전력(전압의 제곱 / 저항)은 몇 W인가? (소수 둘째 자리)"
        return q, _round(v ** 2 / r), 0.01
    if kind == "fuel_cost":
        km, eff, price = rng.randint(120, 640), rng.choice([9.5, 11, 12.5, 14, 16.5]), rng.randint(14, 19) * 100
        q = (f"{km}km를 연비 {eff}km/L인 차로 갔다. 휘발유는 L당 {price:,}원이고 동승자는 {rng.randint(1, 4)}명이다. "
             f"연료비는 몇 원인가? (소수 둘째 자리)")
        return q, _round(km / eff * price), 0.5
    if kind == "molarity":
        mass, molar, ml = rng.randint(2, 60), rng.choice([40.0, 58.44, 74.55, 98.08, 180.16]), rng.choice([100, 250, 500, 1000])
        q = (f"분자량 {molar}인 물질 {mass}g을 물에 녹여 {ml}mL 용액을 만들었다. 실험실 온도는 {rng.randint(18, 28)}도이다. "
             f"몰농도(mol/L)는? (소수 둘째 자리)")
        return q, _round(mass / molar / (ml / 1000)), 0.01
    raise ValueError(kind)


def build(out: Path, per_kind: int, seed: int, kinds_set: str = "basic") -> dict:
    kinds, maker = (HARD_KINDS, make_hard) if kinds_set == "hard" else (KINDS, make)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    items, seen = [], set()
    for kind in kinds:
        n = 0
        while n < per_kind:
            q, gold, tol = maker(kind, rng)
            if q in seen:
                continue
            seen.add(q)
            items.append({"id": f"calc-{kind}-{n}", "kind": kind, "lang": "ko",
                          "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                          "gold": gold, "unit": "", "tolerance": tol, "reference": f"정답: {gold:g}"})
            n += 1
    path = out / "eval_test.jsonl"
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    manifest = {"stage": "applied-calc-tasks", "seed": seed, "per_kind": per_kind, "items": len(items),
                "set": kinds_set, "kinds": list(kinds), "note": "synthetic, generated by code; no external data"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/applied/calc_v1"))
    p.add_argument("--per-kind", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--set", dest="kinds_set", choices=("basic", "hard"), default="basic")
    args = p.parse_args()
    print(json.dumps(build(args.out, args.per_kind, args.seed, args.kinds_set), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
