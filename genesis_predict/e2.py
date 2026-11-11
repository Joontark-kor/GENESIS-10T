"""E2: does the event engine (ETS) help the model forecast, and does the model add anything to the engine?

Items: the frozen 53 KOSIS series (E1), origins 2025-01 and 2025-04, horizons 1 and 3 -> 212 questions.
Each question shows the series name, unit and the last 24 months, and asks for the value of a target month
as ``예측: 중앙값, 하한(10%), 상한(90%)``. Three conditions (greedy, thinking off):

* ``llm``     - the model alone;
* ``llm_ets`` - the same prompt plus the ETS forecast (median and 80% interval) as a reference;
* ``ets``     - the engine alone (no model).

Pre-registered (docs/13): ``llm_ets`` vs ``llm`` - per-series MASE improvement >= 5% with a series-bootstrap
95% CI above 0. Reported, not gated: ``llm_ets`` vs ``ets`` (does the model add value to the engine?) and
80% interval coverage. Greedy decoding is deterministic, so uncertainty comes from the series bootstrap.

    python -m genesis_predict.e2 --csv series.csv --model Qwen/Qwen3-4B --out runs/e2
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics
from .e1 import load_csv
from .event import StatsForecastEngine

REPORT_KIND = "genesis_predict.e2/v1"
ORIGINS = ["2025-01", "2025-04"]
HORIZONS = [1, 3]
PRED_RE = re.compile(r"예측\s*[:：]\s*([-+]?\d[\d,]*\.?\d*)\s*,\s*([-+]?\d[\d,]*\.?\d*)\s*,\s*([-+]?\d[\d,]*\.?\d*)")


def _month_add(ym: str, k: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7]) + k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y}-{m:02d}"


def build_items(series: dict[str, list[tuple[str, float]]], names: dict[str, dict] | None = None) -> list[dict]:
    names = names or {}
    ets = StatsForecastEngine("ets")
    items = []
    for sid, pts in series.items():
        dates = [d for d, _ in pts]
        vals = np.array([v for _, v in pts])
        for origin in ORIGINS:
            if origin not in dates:
                continue
            cut = dates.index(origin) + 1
            hist = vals[:cut]
            fc = ets.forecast(hist, max(HORIZONS), 12)
            for h in HORIZONS:
                if cut + h - 1 >= len(vals):
                    continue
                lo, hi = fc.interval(h - 1)
                items.append({"id": f"{sid}@{origin}+{h}", "series": sid, "origin": origin, "h": h,
                              "target": _month_add(origin, h), "name": names.get(sid, {}).get("name", sid),
                              "unit": names.get(sid, {}).get("unit", ""),
                              "recent": [(dates[i], float(vals[i])) for i in range(max(0, cut - 24), cut)],
                              "history": hist.tolist(), "actual": float(vals[cut + h - 1]),
                              "ets": {"median": float(fc.median[h - 1]), "p10": lo, "p90": hi}})
    return items


def prompt(item: dict, with_ets: bool) -> str:
    rows = "\n".join(f"{d}: {v:g}" for d, v in item["recent"])
    ref = ""
    if with_ets:
        e = item["ets"]
        ref = (f"\n참고: 통계 모델(ETS)의 {item['target']} 예측은 중앙값 {e['median']:.4g}, 80% 구간 "
               f"[{e['p10']:.4g}, {e['p90']:.4g}]이다. 참고하되 판단은 스스로 하라.")
    return (f"다음은 '{item['name']}'(단위: {item['unit']})의 최근 월별 값이다.\n{rows}{ref}\n"
            f"{item['target']}의 값을 예측하라. 마지막 줄은 반드시 '예측: 중앙값, 하한, 상한' 형식으로, "
            f"하한·상한은 실제 값이 그 사이에 들 확률이 80%가 되도록 숫자만 써라.")


def parse(text: str):
    m = PRED_RE.findall(text or "")
    if not m:
        return None
    try:
        med, lo, hi = (float(x.replace(",", "")) for x in m[-1])
    except ValueError:
        return None
    lo, hi = min(lo, hi), max(lo, hi)
    return {"median": med, "p10": lo, "p90": hi}


def score(items: list[dict], llm_out: list[str], llm_ets_out: list[str]) -> dict:
    """Pure function of texts: unit-tested without a GPU. A missing/unparsable answer falls back to the
    last observed value (seasonal-naive-free 'persistence'), and is counted."""
    rows = []
    for it, a, b in zip(items, llm_out, llm_ets_out):
        hist = np.asarray(it["history"])
        last = float(hist[-1])
        for cond, pred in (("llm", parse(a)), ("llm_ets", parse(b)), ("ets", it["ets"])):
            parsed = pred is not None
            pred = pred or {"median": last, "p10": last, "p90": last}
            rows.append({"id": it["id"], "series": it["series"], "h": it["h"], "cond": cond, "parsed": parsed,
                         "median": pred["median"], "actual": it["actual"],
                         "mase": metrics.mase([it["actual"]], [pred["median"]], hist, 12),
                         "covered80": bool(pred["p10"] <= it["actual"] <= pred["p90"])})
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    summary = {c: {"n": len(rs), "parsed": float(np.mean([r["parsed"] for r in rs])),
                   "mase": float(np.mean([r["mase"] for r in rs])),
                   "coverage80": float(np.mean([r["covered80"] for r in rs]))} for c, rs in by.items()}

    def per_series(c):
        acc = defaultdict(list)
        for r in by[c]:
            acc[r["series"]].append(r["mase"])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    def gain(new, ref):
        a, b = per_series(new), per_series(ref)
        g = [1 - a[s] / b[s] for s in sorted(set(a) & set(b)) if b[s] > 0]
        lo, hi = metrics.bootstrap_mean_ci(g) if len(g) > 1 else (float("nan"), float("nan"))
        return {"series": len(g), "mase_gain": float(np.mean(g)) if g else None, "ci95": [lo, hi]}

    comp = {"llm_ets_vs_llm": gain("llm_ets", "llm"), "llm_ets_vs_ets": gain("llm_ets", "ets"), "ets_vs_llm": gain("ets", "llm")}
    g = comp["llm_ets_vs_llm"]
    passed = bool(g["mase_gain"] is not None and g["mase_gain"] >= 0.05 and g["ci95"][0] > 0)
    return {"kind": REPORT_KIND, "summary": summary, "comparisons": comp, "pass_engine_helps_llm": passed, "rows": rows}


def main(argv=None) -> None:
    from .e3 import Generator

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", required=True)
    p.add_argument("--names", default=None, help="series_list.json from genesis_predict.kosis (names and units)")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    names = {}
    if a.names:
        names = {s["id"]: s for s in json.load(open(a.names, encoding="utf-8"))["series"]}
    items = build_items(load_csv(a.csv), names)[: a.limit]
    print(f"items: {len(items)}", flush=True)
    gen = Generator(a.model, max_new_tokens=256)
    t0 = time.time()
    llm = gen([prompt(it, False) for it in items], a.batch_size)
    llm_ets = gen([prompt(it, True) for it in items], a.batch_size)
    rep = score(items, llm, llm_ets)
    rep.update({"model": a.model, "hardware": gen.hardware, "seconds": round(time.time() - t0, 1)})
    for it, x, y in zip(items, llm, llm_ets):
        it["llm_tail"], it["llm_ets_tail"] = x[-200:], y[-200:]
    (a.out / "e2_items.json").write_text(json.dumps(items, ensure_ascii=False, default=float), encoding="utf-8")
    (a.out / "e2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("E2", json.dumps({k: rep[k] for k in ("summary", "comparisons", "pass_engine_helps_llm")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
