"""E1: does a time-series engine beat simple baselines? (no LLM, CPU)

Input: a long-format CSV ``series_id,date,value`` (monthly). For each series, each forecast origin
and each horizon, every engine forecasts from the history up to the origin; we score MASE (point),
CRPS (distribution) and 80% interval coverage. The pass rule is fixed in advance (docs/13):

* mean per-series MASE improvement vs ETS >= 5%, and its series-level bootstrap 95% CI excludes 0;
* 80% interval coverage between 70% and 90%.

    python -m genesis_predict.e1 --csv series.csv --engines seasonal_naive,ets,arima,chronos2 \
        --origins 2024-01,2024-04,2024-07,2024-10,2025-01,2025-04 --horizons 1,3,6 --out e1.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np

from . import metrics
from .event import ENGINES

REPORT_KIND = "genesis_predict.e1/v1"


def load_csv(path: str) -> dict[str, list[tuple[str, float]]]:
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series[row["series_id"]].append((row["date"][:7], float(row["value"])))
    return {k: sorted(v) for k, v in series.items()}


def run(series: dict[str, list[tuple[str, float]]], engines: list[str], origins: list[str],
        horizons: list[int], season: int = 12, min_history: int = 36) -> dict:
    maxh = max(horizons)
    rows, timing = [], defaultdict(float)
    built = {name: ENGINES[name]() for name in engines}
    for sid, points in series.items():
        dates = [d for d, _ in points]
        values = np.array([v for _, v in points])
        for origin in origins:
            if origin not in dates:
                continue
            cut = dates.index(origin) + 1          # history includes the origin month
            if cut < min_history or cut + maxh > len(values):
                continue
            hist, future = values[:cut], values[cut:cut + maxh]
            for name, eng in built.items():
                t0 = time.perf_counter()
                fc = eng.forecast(hist, maxh, season)
                timing[name] += time.perf_counter() - t0
                for h in horizons:
                    s = h - 1
                    qs = {q: float(v[s]) for q, v in fc.quantiles.items()}
                    lo, hi = fc.interval(s, 0.1, 0.9)
                    rows.append({"series": sid, "origin": origin, "h": h, "engine": name,
                                 "actual": float(future[s]), "median": float(fc.median[s]),
                                 "mase": metrics.mase([future[s]], [fc.median[s]], hist, season),
                                 "crps": metrics.crps_from_quantiles(float(future[s]), qs),
                                 "covered80": bool(metrics.covered(float(future[s]), lo, hi))})
    return summarize(rows, engines, dict(timing))


def summarize(rows: list[dict], engines: list[str], timing: dict, reference: str = "ets") -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["engine"]].append(r)
    summary = {}
    for name in engines:
        rs = by.get(name, [])
        if not rs:
            continue
        summary[name] = {"forecasts": len(rs),
                         "mase": float(np.mean([r["mase"] for r in rs])),
                         "crps": float(np.mean([r["crps"] for r in rs])),
                         "coverage80": float(np.mean([r["covered80"] for r in rs])),
                         "seconds": round(timing.get(name, 0.0), 2)}
    verdicts = {}
    if reference in by:
        ref = _per_series(by[reference], "mase")
        for name in engines:
            if name == reference or name not in by:
                continue
            cur = _per_series(by[name], "mase")
            common = sorted(set(ref) & set(cur))
            gains = [1 - cur[s] / ref[s] for s in common if ref[s] > 0]
            lo, hi = metrics.bootstrap_mean_ci(gains) if len(gains) > 1 else (float("nan"), float("nan"))
            cov = summary[name]["coverage80"]
            verdicts[name] = {"series": len(gains), "mase_gain_vs_ets": float(np.mean(gains)) if gains else None,
                              "ci95": [lo, hi],
                              "pass": bool(gains and np.mean(gains) >= 0.05 and lo > 0 and 0.70 <= cov <= 0.90)}
    return {"kind": REPORT_KIND, "summary": summary, "verdict_vs_ets": verdicts, "rows": rows}


def _per_series(rows: list[dict], key: str) -> dict[str, float]:
    acc = defaultdict(list)
    for r in rows:
        acc[r["series"]].append(r[key])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", required=True)
    p.add_argument("--engines", default="seasonal_naive,ets,arima")
    p.add_argument("--origins", required=True)
    p.add_argument("--horizons", default="1,3,6")
    p.add_argument("--season", type=int, default=12)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    report = run(load_csv(a.csv), a.engines.split(","), a.origins.split(","),
                 [int(h) for h in a.horizons.split(",")], a.season)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps({"summary": report["summary"], "verdict_vs_ets": report["verdict_vs_ets"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
