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
