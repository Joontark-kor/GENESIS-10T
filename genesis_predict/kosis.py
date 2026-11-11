"""KOSIS OpenAPI -> long CSV for E1, with a selection rule fixed before any forecast is made.

Tables are fetched with every classification and item (``ALL``), one year per call to stay under
the 40,000-cell limit. Series are then *chosen without looking at their values*: a series is
eligible when it is monthly, complete over the whole window and not constant; from each table at
most ``per_table`` eligible series are taken in the order of a hash of their id. That keeps the
choice reproducible and free of cherry-picking.

    python -m genesis_predict.kosis --tables genesis_predict/kosis_tables.json \
        --start 201501 --end 202510 --out series.csv --list series_list.json
(the key is read from the KOSIS_API_KEY environment variable; it is never written to disk)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict

URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


def _months(start: str, end: str) -> list[str]:
    y, m = int(start[:4]), int(start[4:])
    out = []
    while f"{y}{m:02d}" <= end:
        out.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_table(api_key: str, org_id: str, tbl_id: str, start: str, end: str, levels: int = 1,
                opener=urllib.request.urlopen, pause: float = 0.3) -> list[dict]:
    """All cells of one table: the whole window in one call, year by year if KOSIS refuses that
    (e.g. the 40,000-cell limit). Connection drops are retried. Raises RuntimeError on a KOSIS error."""

    def get(a: str, b: str):
        params = {"method": "getList", "apiKey": api_key, "orgId": org_id, "tblId": tbl_id,
                  "itmId": "ALL", "prdSe": "M", "startPrdDe": a, "endPrdDe": b,
                  "format": "json", "jsonVD": "Y"}
        for i in range(1, levels + 1):
            params[f"objL{i}"] = "ALL"
        for attempt in range(4):
            try:
                with opener(URL + "?" + urllib.parse.urlencode(params), timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except (OSError, json.JSONDecodeError) as exc:  # reset, timeout, truncated body
                if attempt == 3:
                    raise RuntimeError(f"{org_id}/{tbl_id} {a}-{b}: {type(exc).__name__}") from None
                time.sleep(pause + 2 ** attempt)
        time.sleep(pause)
        if isinstance(data, dict):
            if str(data.get("err")) == "30":       # no data in this window
                return []
            raise RuntimeError(f"{org_id}/{tbl_id} {a}-{b}: {data.get('err')} {data.get('errMsg')}")
        return data

    try:
        return get(start, end)
    except RuntimeError as exc:
        if " 21 " in str(exc):                    # wrong variables: chunking will not help
            raise
    rows = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        rows.extend(get(max(start, f"{year}01"), min(end, f"{year}12")))
    return rows


def to_series(rows: list[dict]) -> dict[str, dict]:
    """KOSIS cells -> {series_id: {"name", "unit", "points": {YYYYMM: value}}}."""
    out: dict[str, dict] = {}
    for r in rows:
        cls = [r.get(f"C{i}") for i in range(1, 9) if r.get(f"C{i}")]
        sid = "|".join([r["ORG_ID"], r["TBL_ID"], *cls, r["ITM_ID"]])
        names = [r.get(f"C{i}_NM") for i in range(1, 9) if r.get(f"C{i}_NM")]
        s = out.setdefault(sid, {"name": " / ".join([r.get("TBL_NM", ""), *names, r.get("ITM_NM", "")]),
                                 "unit": r.get("UNIT_NM", ""), "points": {}})
        try:
            s["points"][r["PRD_DE"]] = float(r["DT"])
        except (TypeError, ValueError):
            pass  # '-', 'x' and blanks are missing values
    return out


def select(series: dict[str, dict], start: str, end: str, per_table: int) -> list[str]:
    """Eligible = complete over [start, end] and not constant; then hash order, capped per table."""
    months = _months(start, end)
    by_table = defaultdict(list)
    for sid, s in series.items():
        vals = [s["points"].get(m) for m in months]
        if any(v is None for v in vals) or max(vals) == min(vals):
            continue
        table = "|".join(sid.split("|")[:2])
        by_table[table].append(sid)
    chosen = []
    for table in sorted(by_table):
        ids = sorted(by_table[table], key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        chosen.extend(ids[:per_table])
    return chosen


def write_csv(series: dict[str, dict], ids: list[str], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series_id", "date", "value"])
        for sid in ids:
            for m, v in sorted(series[sid]["points"].items()):
                w.writerow([sid, f"{m[:4]}-{m[4:]}", v])


def probe(api_key: str, org_id: str, tbl_id: str) -> None:
    """Print timing/shape of a few period ranges for one table (diagnostics only; key masked)."""
    for a, b in [("202401", "202401"), ("202401", "202412"), ("201501", "201501"), ("201501", "201512"), ("201501", "202510")]:
        params = {"method": "getList", "apiKey": api_key, "orgId": org_id, "tblId": tbl_id, "itmId": "ALL",
                  "objL1": "ALL", "prdSe": "M", "startPrdDe": a, "endPrdDe": b, "format": "json", "jsonVD": "Y"}
        t0 = time.time()
        try:
            with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(params), timeout=30) as r:
                body = r.read().decode("utf-8")
            data = json.loads(body)
            info = (f"rows={len(data)} first={data[0].get('PRD_DE')} last={data[-1].get('PRD_DE')}"
                    if isinstance(data, list) else f"err={data}")
        except Exception as exc:
            info = f"exception={type(exc).__name__}: {str(exc)[:80]}"
        print(f"{tbl_id} {a}-{b} {time.time() - t0:.1f}s {info}".replace(api_key, "***"), flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tables", help="JSON list of {org, tbl, levels, note}")
    p.add_argument("--probe", help="diagnose one table: ORG/TBL")
    p.add_argument("--start", default="201501")
    p.add_argument("--end", default="202510")
    p.add_argument("--per-table", type=int, default=12)
    p.add_argument("--target", type=int, default=60)
    p.add_argument("--out")
    p.add_argument("--list")
    a = p.parse_args(argv)
    key = os.environ["KOSIS_API_KEY"]
    if a.probe:
        org, tbl = a.probe.split("/")
        probe(key, org, tbl)
        return
    tables = json.load(open(a.tables, encoding="utf-8"))
    series, log = {}, []
    for t in tables:
        got, errs = None, []
        for levels in ([t["levels"]] if "levels" in t else [1, 2, 3]):
            try:
                got = to_series(fetch_table(key, t["org"], t["tbl"], a.start, a.end, levels))
                log.append({"table": t["tbl"], "note": t.get("note", ""), "levels": levels, "series": len(got)})
                print(f"ok   {t['tbl']} levels={levels} series={len(got)}", flush=True)
                break
            except Exception as exc:  # wrong level count, unknown table, network
                errs.append(f"L{levels}: " + str(exc).replace(key, "***"))
        if got is None:
            print(f"fail {t['tbl']}: {' | '.join(e[:90] for e in errs)}", flush=True)
            log.append({"table": t["tbl"], "note": t.get("note", ""), "error": errs})
            continue
        series.update(got)
    ids = select(series, a.start, a.end, a.per_table)[: a.target]
    write_csv(series, ids, a.out)
    listing = {"rule": f"complete {a.start}-{a.end}, non-constant, sha256 order, <= {a.per_table} per table, first {a.target}",
               "tables": log, "series": [{"id": i, "name": series[i]["name"], "unit": series[i]["unit"]} for i in ids]}
    json.dump(listing, open(a.list, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"tables": log, "chosen": len(ids)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
