"""Research track: prediction engines (CPU only, no downloads)."""
import json
import math

import numpy as np
import pytest

from genesis_predict import metrics
from genesis_predict.event import SeasonalNaive, propagate, _quantiles_to_forecast
from genesis_predict import e1, sealed


def _seasonal(n=72, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n)


# ---------- metrics ----------

def test_crps_samples_matches_point_mass():
    assert metrics.crps_from_samples(3.0, [3.0] * 50) == pytest.approx(0.0)
    assert metrics.crps_from_samples(5.0, [3.0] * 50) == pytest.approx(2.0)


def test_crps_quantiles_close_to_samples_for_normal():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, 20000)
    qs = {q: float(np.quantile(s, q)) for q in np.linspace(0.05, 0.95, 19)}
    assert metrics.crps_from_quantiles(0.5, qs) == pytest.approx(metrics.crps_from_samples(0.5, s), rel=0.1)


def test_mase_and_brier():
    hist = np.arange(30, dtype=float)
    assert metrics.mase([31], [30], hist, season=1) == pytest.approx(1.0)
    assert metrics.brier(0.8, True) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        metrics.brier(1.2, True)


# ---------- event engines ----------

def test_seasonal_naive_repeats_last_season():
    y = _seasonal()
    fc = SeasonalNaive().forecast(y, 6)
    assert np.allclose(fc.median, y[-12:-6], atol=1.0)
    lo, hi = fc.interval(0)
    assert lo <= fc.median[0] <= hi


def test_quantiles_to_samples_preserves_median():
    qs = {0.1: np.array([8.0]), 0.5: np.array([10.0]), 0.9: np.array([12.0])}
    fc = _quantiles_to_forecast("t", qs, 4000, 0)
    assert float(np.median(fc.samples[:, 0])) == pytest.approx(10.0, abs=0.2)


def test_propagate_formula_over_samples():
    fc = _quantiles_to_forecast("t", {0.1: np.array([90.0]), 0.5: np.array([100.0]), 0.9: np.array([110.0])}, 2000, 0)
    out = propagate(lambda cpi, base: (cpi - base) / base * 100, {"cpi": fc}, 0, {"base": 100.0})
    assert out["median"] == pytest.approx(0.0, abs=0.5)
    assert out["p10"] < 0 < out["p90"]


def test_e1_runs_and_judges_against_ets_reference():
    pytest.importorskip("statsforecast")
    months = [f"{2019 + i // 12}-{i % 12 + 1:02d}" for i in range(84)]
    series = {f"s{k}": list(zip(months, _seasonal(84, k))) for k in range(3)}
    rep = e1.run(series, ["seasonal_naive", "ets"], ["2023-01", "2023-06"], [1, 3])
    assert rep["summary"]["ets"]["forecasts"] == 3 * 2 * 2
    assert "seasonal_naive" in rep["verdict_vs_ets"]
    assert rep["kind"] == e1.REPORT_KIND
    json.dumps(rep)  # the report must be serializable without a custom encoder


# ---------- physics engine ----------
pint = pytest.importorskip("pint")
from genesis_predict import physics  # noqa: E402


def test_projectile_range_matches_textbook():
    spec = {"template": "projectile", "params": {"v0": "20 m/s", "angle": "45 deg"}, "ask": "range", "unit": "m"}
    status, q = physics.solve(spec)
    assert status == "ok" and q.magnitude == pytest.approx(400 / 9.81, rel=1e-6)


def test_projectile_closed_form_agrees_with_ode():
    scipy = pytest.importorskip("scipy")
    from scipy.integrate import solve_ivp

    v0, ang, g, h0 = 15.0, math.radians(35), 9.81, 2.0

    def hit(t, s):
        return s[1]
    hit.terminal, hit.direction = True, -1
    sol = solve_ivp(lambda t, s: [s[2], s[3], 0, -g], (0, 10), [0, h0, v0 * math.cos(ang), v0 * math.sin(ang)],
                    events=hit, rtol=1e-10, atol=1e-10)
    spec = {"template": "projectile", "params": {"v0": "15 m/s", "angle": "35 deg", "h0": "2 m"}, "ask": "range", "unit": "m"}
    assert physics.solve(spec)[1].magnitude == pytest.approx(sol.y_events[0][0][0], rel=1e-6)


def test_cooling_closed_form_agrees_with_ode():
    from scipy.integrate import solve_ivp

    sol = solve_ivp(lambda t, T: -0.1 * (T - 20), (0, 10), [90], rtol=1e-10, atol=1e-10)
    spec = {"template": "cooling", "params": {"T0": "90 degC", "Tenv": "20 degC", "k": "0.1 1/min", "t": "10 min"},
            "ask": "temperature", "unit": "degC"}
    assert physics.solve(spec)[1].magnitude == pytest.approx(sol.y[0][-1], rel=1e-6)


def test_units_are_checked_and_converted():
    ok = {"template": "pendulum", "params": {"L": "100 cm"}, "ask": "period", "unit": "s"}
    assert physics.solve(ok)[1].magnitude == pytest.approx(2 * math.pi * math.sqrt(1 / 9.81))
    wrong_dim = {"template": "pendulum", "params": {"L": "2 s"}, "ask": "period"}
    assert physics.solve(wrong_dim)[0] == "bad_spec"
    wrong_unit = {"template": "pendulum", "params": {"L": "1 m"}, "ask": "period", "unit": "m"}
    assert physics.solve(wrong_unit)[0] == "bad_spec"


def test_verify_outcomes():
    spec = {"template": "ohm", "params": {"V": "12 V", "R": "4 ohm"}, "ask": "power", "unit": "W"}
    assert physics.verify(spec, 36)["status"] == "match"
    assert physics.verify(spec, 3)["status"] == "mismatch"
    assert physics.verify({"template": "fluid_jet", "ask": "x"}, 1)["status"] == "unsupported"
    assert physics.verify(None, 1)["status"] == "bad_spec"


def test_extract_spec_from_reply():
    reply = '풀이 생략\n시나리오: {"template": "free_fall", "params": {"h": "20 m"}, "ask": "time", "unit": "s"}\n정답: 2.02'
    spec = physics.extract_spec(reply)
    assert spec["template"] == "free_fall"
    assert physics.verify(spec, 2.02)["status"] == "match"


def test_unknown_g_is_used_when_given():
    spec = {"template": "free_fall", "params": {"h": "20 m", "g": "3.7 m/s^2"}, "ask": "time", "unit": "s"}
    assert physics.solve(spec)[1].magnitude == pytest.approx(math.sqrt(40 / 3.7))


# ---------- sealed sets ----------

def test_sealed_set_opens_once_and_detects_edits(tmp_path):
    data = tmp_path / "s1.jsonl"
    data.write_text(json.dumps({"q": "a"}) + "\n" + json.dumps({"q": "b"}) + "\n", encoding="utf-8")
    m = sealed.seal(data, "s1", "e1/v1", tmp_path / "manifests")
    assert m["items"] == 2
    mpath = tmp_path / "manifests" / "s1.sealed.json"
    assert len(sealed.open_sealed(data, mpath, "test")) == 2
    with pytest.raises(PermissionError):
        sealed.open_sealed(data, mpath, "again")
    data.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        sealed.open_sealed(data, mpath, "edited")


# ---------- KOSIS (offline, fake responses) ----------

def test_kosis_fetch_select_and_csv(tmp_path):
    import io
    from genesis_predict import kosis

    def cell(c1, itm, m, v):
        return {"ORG_ID": "101", "TBL_ID": "T1", "TBL_NM": "표", "C1": c1, "C1_NM": c1, "ITM_ID": itm,
                "ITM_NM": "값", "UNIT_NM": "", "PRD_DE": m, "DT": v}

    def opener(url, timeout=60):
        q = dict(x.split("=", 1) for x in url.split("?", 1)[1].split("&"))
        rows = []
        for m in kosis._months(q["startPrdDe"], q["endPrdDe"]):
            rows.append(cell("A", "T", m, str(int(m[-2:]) + int(m[:4]))))   # complete, varying
            rows.append(cell("B", "T", m, "5"))                              # constant -> dropped
            rows.append(cell("C", "T", m, "-" if m == "202402" else "1.5"))  # gap -> dropped
        return io.BytesIO(json.dumps(rows).encode())

    rows = kosis.fetch_table("k", "101", "T1", "202311", "202403", opener=opener, pause=0)
    assert len(rows) == 3 * 5
    series = kosis.to_series(rows)
    assert kosis.select(series, "202311", "202403", 10) == ["101|T1|A|T"]
    out = tmp_path / "s.csv"
    kosis.write_csv(series, ["101|T1|A|T"], out)
    loaded = e1.load_csv(str(out))
    assert [d for d, _ in loaded["101|T1|A|T"]] == ["2023-11", "2023-12", "2024-01", "2024-02", "2024-03"]


def test_kosis_error_is_raised_without_key():
    import io
    from genesis_predict import kosis

    def opener(url, timeout=60):
        return io.BytesIO(json.dumps({"err": "20", "errMsg": "필수요청변수값 누락"}).encode())
    with pytest.raises(RuntimeError) as exc:
        kosis.fetch_table("secret", "101", "T1", "202401", "202402", opener=opener, pause=0)
    assert "secret" not in str(exc.value)


def test_kosis_retries_connection_reset():
    import io
    from genesis_predict import kosis
    calls = {"n": 0}

    def opener(url, timeout=60):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError(104, "reset")
        return io.BytesIO(json.dumps([{"ORG_ID": "1", "TBL_ID": "T", "ITM_ID": "I", "PRD_DE": "202401", "DT": "1"}]).encode())
    assert len(kosis.fetch_table("k", "1", "T", "202401", "202401", opener=opener, pause=0)) == 1
    assert calls["n"] == 2


# ---------- physics dev set ----------

def test_physics_set_shape_and_josa():
    from genesis_predict import physics_set as ps
    items = ps.make(seed=3, per_template=4, unsupported=5)
    assert len(items) == 10 * 4 + 5
    assert ps.josa("시간") == "시간은" and ps.josa("거리") == "거리는" and ps.josa("최고 높이(지면 기준)") == "최고 높이(지면 기준)는"
    for it in items:
        if it["status"] == "supported":
            assert sorted(c["error"] is None for c in it["claims"]) == [False, True]
            assert "는 몇" in it["question"] or "은 몇" in it["question"] or "인가" in it["question"]
            assert "시간는" not in it["question"] and "전류은" not in it["question"]


def test_physics_p0_engine_agrees_with_independent_gold():
    pytest.importorskip("scipy")
    from genesis_predict import physics_set as ps
    rep = ps.p0(ps.make(seed=7, per_template=6, unsupported=0))
    assert rep["engine_vs_gold_agree"] == 1.0 and rep["false_block_rate"] == 0.0 and rep["detection_rate"] == 1.0
    assert rep["pass"]


def test_physics_gold_is_independent_of_engine():
    import inspect
    from genesis_predict import physics_set as ps
    src = inspect.getsource(ps)
    gold_part = src[src.index("# ---- gold solvers"):src.index("# ---- planted errors")]
    assert "physics." not in gold_part and "pint" not in gold_part


def test_physics_score_model():
    from genesis_predict import physics_set as ps
    items = ps.make(seed=4, per_template=2, unsupported=2)
    perfect = {it["id"]: ({"spec": it["spec"], "answer": it["gold"]} if it["status"] == "supported" else {"unsupported": True})
               for it in items}
    s = ps.score_model(items, perfect)
    assert s["accuracy"] == 1.0 and s["spec_accuracy"] == 1.0 and s["unsupported_recall"] == 1.0 and s["engine_harm"] == 0
    # a model that writes the right spec but miscalculates: the engine would rescue every item
    sloppy = {k: ({"spec": v["spec"], "answer": v["answer"] * 2} if "spec" in v else v) for k, v in perfect.items()}
    s2 = ps.score_model(items, sloppy)
    assert s2["accuracy"] == 0.0 and s2["engine_rescue"] == 1.0
    wrong_spec = dict(items[0]["spec"], params=dict(items[0]["spec"]["params"]))
    k0 = next(iter(wrong_spec["params"]))
    wrong_spec["params"][k0] = "999 m"
    assert not ps.spec_equal(wrong_spec, items[0]["spec"])


def test_e3_score_without_gpu():
    from genesis_predict import e3, physics_set as ps
    items = ps.make(seed=5, per_template=1, unsupported=1)
    sup = [it for it in items if it["status"] == "supported"]
    direct = [f"풀이 생략\n정답: {it['gold'] * 2}" for it in sup] + ["모름"]
    spec = [f"시나리오: {json.dumps(it['spec'], ensure_ascii=False)}\n정답: {it['gold'] * 2}" for it in sup] + ["시나리오: 미지원"]
    rep = e3.score(sup + [items[-1]], direct, spec)["summary"]
    assert rep["direct"]["accuracy"] == 0.0 and rep["spec_model"]["accuracy"] == 0.0
    assert rep["spec_engine"]["accuracy"] == 1.0 and rep["spec_model"]["engine_rescue"] == 1.0
    assert rep["detection"] == 1.0 and rep["false_block"] == 0.0
    assert rep["spec_engine"]["unsupported_recall"] == 1.0


def test_e3_parse_and_normalize():
    from genesis_predict import e3
    assert e3.parse_answer("... 정답: 1,234.5") == 1234.5 and e3.parse_answer("정답: -3.2e2") == -320.0
    assert e3.parse_answer("답 없음") is None
    s = e3.normalize_spec({"template": "cooling", "params": {"T0": "90 °C", "k": "0.1 1/min"}, "unit": "°C"})
    assert s["params"]["T0"] == "90 degC" and s["unit"] == "degC"
    s2 = e3.normalize_spec({"template": "rc_charge", "params": {"C": "10 μF", "R": "4.7 kΩ"}})
    from genesis_predict import physics
    assert physics.Q(s2["params"]["C"]).to("F").magnitude == pytest.approx(1e-5)
    assert physics.Q(s2["params"]["R"]).to("ohm").magnitude == pytest.approx(4700)


def test_e2_build_score_without_gpu():
    pytest.importorskip("statsforecast")
    from genesis_predict import e2
    months = [f"{2019 + i // 12}-{i % 12 + 1:02d}" for i in range(84)]
    series = {f"s{k}": list(zip(months, _seasonal(84, k))) for k in range(3)}
    items = e2.build_items(series)
    assert len(items) == 3 * 2 * 2 and items[0]["target"] in ("2025-02", "2025-04")
    assert "예측: 중앙값, 하한, 상한" in e2.prompt(items[0], True) and "ETS" in e2.prompt(items[0], True)
    assert "ETS" not in e2.prompt(items[0], False)
    good = [f"예측: {it['actual']}, {it['actual'] - 1}, {it['actual'] + 1}" for it in items]
    junk = ["모르겠다"] * len(items)
    rep = e2.score(items, junk, good)
    assert rep["summary"]["llm_ets"]["mase"] == pytest.approx(0.0) and rep["summary"]["llm"]["parsed"] == 0.0
    assert rep["comparisons"]["llm_ets_vs_llm"]["mase_gain"] == pytest.approx(1.0)
    assert rep["pass_engine_helps_llm"] is True
    json.dumps(rep)
    assert e2.parse("... 예측: 1,234.5, 1,200, 1,300") == {"median": 1234.5, "p10": 1200.0, "p90": 1300.0}


def test_e3_extract_spec_lenient():
    from genesis_predict import e3
    a = e3.extract_spec("풀이...\n시나리오: ```json\n{\"template\": \"ohm\", \"params\": {\"V\": \"12 V\", \"R\": \"4 ohm\"}, \"ask\": \"power\", \"unit\": \"W\"}\n```\n정답: 36")
    assert a["template"] == "ohm"
    b = e3.extract_spec("시나리오: {'template': 'ohm', 'params': {'V': '12 V', 'R': '4 ohm'}, 'ask': 'current', 'unit': 'A'}")
    assert b["ask"] == "current"
    assert e3.extract_spec("시나리오: {\"template\": \"ohm\", \"params\": {\"V\": ") is None


def test_e3_normalize_template_signature_and_v2_prompt():
    from genesis_predict import e3, physics_set as ps
    s = e3.normalize_spec({"template": "incline(acceleration, normal_force; angle, mu, g, m)", "params": {"angle": 25}})
    assert s["template"] == "incline"
    assert '"template": "ohm"' in ps.PROMPT_V3 and ps.PROMPT_V3.endswith("문제: ") and "g에 그 값을" in ps.PROMPT_V3
