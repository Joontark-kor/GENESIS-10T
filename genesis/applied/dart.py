"""OpenDART client for listed-company financial statements.

Get a free key at https://opendart.fss.or.kr (인증키 신청) and keep it out of
code: ``export DART_API_KEY=...``. Every response is cached on disk, so a
re-run costs no quota. OpenDART allows roughly 20,000 requests a day per key;
status 020 means the quota is spent and collection must stop until tomorrow.

Used endpoints:
* ``corpCode.xml`` – ZIP with every company's 8-digit corp_code.
* ``fnlttSinglAcntAll.json`` – full financial statements of one company
  (``reprt_code`` 11011 annual, 11012 half-year, 11013 Q1, 11014 Q3;
  ``fs_div`` CFS consolidated, OFS separate). Data from fiscal 2015 on.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import time
from typing import Callable
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

BASE_URL = "https://opendart.fss.or.kr/api"
REPORT_CODES = {"annual": "11011", "half": "11012", "q1": "11013", "q3": "11014"}

# metric -> (statement kinds, standard XBRL account ids, Korean account names as fallback)
ACCOUNTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "revenue": (("IS", "CIS"), ("ifrs-full_Revenue",), ("매출액", "수익(매출액)", "영업수익", "매출")),
    "operating_income": (("IS", "CIS"), ("dart_OperatingIncomeLoss",), ("영업이익", "영업이익(손실)")),
    "net_income": (("IS", "CIS"), ("ifrs-full_ProfitLoss",), ("당기순이익", "당기순이익(손실)", "분기순이익", "반기순이익")),
    "total_assets": (("BS",), ("ifrs-full_Assets",), ("자산총계",)),
    "total_liabilities": (("BS",), ("ifrs-full_Liabilities",), ("부채총계",)),
    "total_equity": (("BS",), ("ifrs-full_Equity",), ("자본총계",)),
}


class DartError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(f"OpenDART status {status}: {message}")
        self.status = status


class QuotaExceeded(DartError):
    """Status 020: the daily request limit is spent. Stop and resume tomorrow."""


def parse_amount(value) -> int | None:
    """'1,234' -> 1234, '-56' -> -56, '' / '-' / None -> None."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _default_fetch(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "genesis-applied/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class DartClient:
    def __init__(self, api_key: str | None = None, cache_dir: str | Path = "data/applied/cache/dart",
                 min_interval: float = 0.15, fetch: Callable[[str], bytes] | None = None,
                 max_retries: int = 3):
        self.api_key = api_key or os.environ.get("DART_API_KEY", "")
        if not self.api_key:
            raise ValueError("set DART_API_KEY (OpenDART 인증키) or pass api_key")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.fetch = fetch or _default_fetch
        self.max_retries = max_retries
        self._last_call = 0.0
        self.calls = 0

    # ------------------------------------------------------------------ http
    def _get(self, endpoint: str, params: dict) -> bytes:
        query = urllib.parse.urlencode({"crtfc_key": self.api_key, **params})
        url = f"{BASE_URL}/{endpoint}?{query}"
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                self.calls += 1
                return self.fetch(url)
            except OSError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------- companies
    def corp_codes(self, listed_only: bool = True, refresh: bool = False) -> list[dict]:
        cache = self.cache_dir / "corp_codes.json"
        if cache.exists() and not refresh:
            companies = json.loads(cache.read_text(encoding="utf-8"))
        else:
            payload = self._get("corpCode.xml", {})
            if not payload.startswith(b"PK"):  # an error comes back as JSON/XML, not a ZIP
                raise DartError("corpCode", payload[:200].decode("utf-8", errors="replace"))
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                xml_bytes = archive.read(archive.namelist()[0])
            companies = []
            for node in ET.fromstring(xml_bytes).iter("list"):
                companies.append({field: (node.findtext(field) or "").strip()
                                  for field in ("corp_code", "corp_name", "stock_code", "modify_date")})
            cache.write_text(json.dumps(companies, ensure_ascii=False), encoding="utf-8")
        if listed_only:
            companies = [c for c in companies if c.get("stock_code")]
        return companies

    # ------------------------------------------------------------ statements
    def statements(self, corp_code: str, year: int, report: str = "annual",
                   fs_div: str = "CFS") -> list[dict]:
        """Raw statement rows; [] when DART has none (status 013)."""
        reprt_code = REPORT_CODES.get(report, report)
        cache = self.cache_dir / "statements" / f"{corp_code}_{year}_{reprt_code}_{fs_div}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        payload = json.loads(self._get("fnlttSinglAcntAll.json", {
            "corp_code": corp_code, "bsns_year": str(year), "reprt_code": reprt_code, "fs_div": fs_div}))
        status, message = payload.get("status"), payload.get("message", "")
        if status == "013":
            rows = []
        elif status == "020":
            raise QuotaExceeded(status, message)
        elif status != "000":
            raise DartError(status, message)
        else:
            rows = payload.get("list", [])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return rows

    def company_year(self, company: dict, year: int, report: str = "annual") -> dict | None:
        """Normalized metrics for one company and fiscal year (consolidated, else separate)."""
        for fs_div in ("CFS", "OFS"):
            rows = self.statements(company["corp_code"], year, report, fs_div)
            if rows:
                record = normalize(rows)
                if record["metrics"]:
                    record.update({"corp_code": company["corp_code"], "corp_name": company["corp_name"],
                                   "stock_code": company.get("stock_code", ""), "fs_div": fs_div,
                                   "report": report, "year": year, "source": "OpenDART fnlttSinglAcntAll"})
                    return record
        return None


def _pick(rows: list[dict], kinds, ids, names) -> dict | None:
    candidates = [r for r in rows if r.get("sj_div") in kinds]
    for account_id in ids:
        for row in candidates:
            if row.get("account_id") == account_id:
                return row
    for name in names:
        for row in candidates:
            if (row.get("account_nm") or "").replace(" ", "") == name:
                return row
    return None


def normalize(rows: list[dict]) -> dict:
    """Pick the standard lines of the current and prior period from DART rows.

    Returns ``{"metrics": {"current": {...}, "prior": {...}}, "currency", "rcept_no"}``
    with amounts in the statement's currency unit (KRW for most filers).
    """
    current: dict[str, int] = {}
    prior: dict[str, int] = {}
    for metric, (kinds, ids, names) in ACCOUNTS.items():
        row = _pick(rows, kinds, ids, names)
        if row is None:
            continue
        this_year = parse_amount(row.get("thstrm_amount"))
        last_year = parse_amount(row.get("frmtrm_amount"))
        if this_year is not None:
            current[metric] = this_year
        if last_year is not None:
            prior[metric] = last_year
    metrics = {}
    if current:
        metrics["current"] = current
    if prior:
        metrics["prior"] = prior
    first = rows[0] if rows else {}
    return {"metrics": metrics, "currency": first.get("currency") or "KRW",
            "rcept_no": first.get("rcept_no", "")}
