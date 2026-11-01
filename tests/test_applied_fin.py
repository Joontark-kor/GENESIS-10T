import io
import json
from pathlib import Path
import zipfile

import pytest

from genesis.applied.build_fin_tasks import build, company_split, fetch, make_tasks
from genesis.applied.dart import DartClient, QuotaExceeded, normalize, parse_amount


def row(sj, account_id, name, now, before):
    return {"sj_div": sj, "account_id": account_id, "account_nm": name, "thstrm_amount": now,
            "frmtrm_amount": before, "currency": "KRW", "rcept_no": "20260310000123"}


SAMSUNG_LIKE = [
    row("BS", "ifrs-full_Assets", "자산총계", "455,905,980,000,000", "448,424,507,000,000"),
    row("BS", "ifrs-full_Liabilities", "부채총계", "92,228,115,000,000", "92,417,780,000,000"),
    row("BS", "ifrs-full_Equity", "자본총계", "363,677,865,000,000", "356,006,727,000,000"),
    row("CIS", "ifrs-full_Revenue", "매출액", "258,935,494,000,000", "302,231,360,000,000"),
    row("CIS", "-", "영업이익", "6,566,976,000,000", "43,376,630,000,000"),
    row("CIS", "ifrs-full_ProfitLoss", "당기순이익", "15,487,100,000,000", "55,654,077,000,000"),
    row("CIS", "ifrs-full_Revenue", "매출액", "1", "1"),  # duplicates must not win
]


def test_parse_amount():
    assert parse_amount("1,234") == 1234 and parse_amount("-56") == -56
    assert parse_amount("") is None and parse_amount("-") is None and parse_amount(None) is None


def test_normalize_uses_ids_then_names():
    record = normalize(SAMSUNG_LIKE)
    cur, pri = record["metrics"]["current"], record["metrics"]["prior"]
    assert cur["revenue"] == 258_935_494_000_000
    assert cur["operating_income"] == 6_566_976_000_000  # matched by Korean name (account_id '-')
    assert pri["net_income"] == 55_654_077_000_000
    assert record["currency"] == "KRW"


def _record(code="00126380", name="가나전자", year=2025, rows=SAMSUNG_LIKE):
    rec = normalize(rows)
    rec.update({"corp_code": code, "corp_name": name, "year": year, "fs_div": "CFS"})
    return rec


def test_tasks_have_correct_arithmetic():
    tasks = {t["kind"]: t for t in make_tasks(_record())}
    assert tasks["operating_margin"]["gold"] == round(6566976 / 258935494 * 100, 2) == 2.54
    assert tasks["debt_ratio"]["gold"] == round(92228115 / 363677865 * 100, 2)
    assert tasks["revenue_growth"]["gold"] == round((258935494 - 302231360) / 302231360 * 100, 2)
    assert tasks["operating_margin"]["answer"].endswith("정답: 2.54%")
    assert "| 매출액 | 258,935,494 | 302,231,360 |" in tasks["roe"]["question"]
    assert any(t["kind"] == "extract" for t in make_tasks(_record()))
    ja = {t["kind"]: t for t in make_tasks(_record(), "ja")}
    assert ja["operating_margin"]["answer"].endswith("答え: 2.54%") and "百万ウォン" in ja["roe"]["question"]


def test_ratio_skipped_when_denominator_not_positive():
    rows = [r for r in SAMSUNG_LIKE if r["account_nm"] != "자본총계"]
    rows.append(row("BS", "ifrs-full_Equity", "자본총계", "-5,000,000,000", "1,000,000,000"))
    kinds = {t["kind"] for t in make_tasks(_record(rows=rows))}
    assert "debt_ratio" not in kinds and "roe" not in kinds and "operating_margin" in kinds


class FakeDart:
    def __init__(self, quota_after=None):
        self.quota_after, self.seen = quota_after, 0

    def __call__(self, url):
        if "corpCode.xml" in url:
            xml = "<result>" + "".join(
                f"<list><corp_code>{i:08d}</corp_code><corp_name>회사{i}</corp_name>"
                f"<stock_code>{'%06d' % i if i % 2 else ''}</stock_code><modify_date>20260101</modify_date></list>"
                for i in range(1, 41)) + "</result>"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("CORPCODE.xml", xml)
            return buf.getvalue()
        self.seen += 1
        if self.quota_after is not None and self.seen > self.quota_after:
            return json.dumps({"status": "020", "message": "사용한도 초과"}).encode()
        if "fs_div=CFS" in url and "00000003" in url:
            return json.dumps({"status": "013", "message": "조회된 데이타가 없습니다."}).encode()
        return json.dumps({"status": "000", "message": "정상", "list": SAMSUNG_LIKE}).encode()


def test_fetch_build_end_to_end(tmp_path: Path):
    out = tmp_path / "fin"
    client = DartClient(api_key="x" * 40, cache_dir=tmp_path / "cache", min_interval=0, fetch=FakeDart())
    summary = fetch(out, [2024, 2025], n_companies=None, corp_codes=None, client=client)
    assert summary["written"] == 40  # 20 listed companies x 2 years
    records = [json.loads(l) for l in (out / "records.jsonl").read_text().splitlines()]
    assert {r["fs_div"] for r in records if r["corp_code"] == "00000003"} == {"OFS"}  # CFS empty -> OFS
    calls = client.calls
    fetch(out, [2024, 2025], None, None, client=client)  # resumable: nothing refetched
    assert client.calls == calls

    manifest = build(out, ["ko", "ja"])
    assert manifest["records"] == 40 and manifest["tasks"]["train"] > 0
    train = {json.loads(l)["meta"]["corp_code"] for l in (out / "sft_train.jsonl").read_text().splitlines()}
    test = {json.loads(l)["corp_code"] for l in (out / "eval_test.jsonl").read_text().splitlines()}
    assert not (train & test)
    item = json.loads((out / "eval_val.jsonl").read_text().splitlines()[0]) if (out / "eval_val.jsonl").stat().st_size else None
    if item:
        assert item["messages"][-1]["role"] == "user" and "gold" in item


def test_quota_stops_and_keeps_progress(tmp_path: Path):
    client = DartClient(api_key="x" * 40, cache_dir=tmp_path / "cache", min_interval=0, fetch=FakeDart(quota_after=5))
    summary = fetch(tmp_path / "fin", [2025], None, None, client=client)
    assert summary["stopped"] and summary["written"] == 4  # company 3 needs two calls (CFS empty, then OFS)
    with pytest.raises(QuotaExceeded):
        client.statements("99999999", 2025)


def test_company_split_is_stable():
    assert company_split("00126380") == company_split("00126380")
    splits = [company_split(f"{i:08d}") for i in range(2000)]
    assert 0.05 < splits.count("test") / 2000 < 0.15


def test_spacs_are_excluded(tmp_path: Path):
    out = tmp_path / "fin"
    out.mkdir()
    records = [_record(code="00000001", name="가나전자"), _record(code="00000002", name="하나금융제30호스팩"),
               _record(code="00000003", name="미래에셋비전기업인수목적2호")]
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    manifest = build(out, ["ko"])
    assert manifest["excluded"] == {"spac_records": 2}
    assert sum(manifest["companies"].values()) == 1
