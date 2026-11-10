"""Event (time-series) engines behind one small interface.

Every engine returns a :class:`Forecast` with quantiles *and* samples, so a polarbear formula can be
evaluated over the distribution (see :func:`propagate`). Heavy libraries load lazily:
``statsforecast`` for ETS/AutoARIMA, ``chronos`` (chronos-forecasting) for Chronos-2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class Forecast:
    engine: str
    horizon: int
    quantiles: dict[float, np.ndarray]          # level -> array of length horizon
    samples: np.ndarray                         # shape (n_samples, horizon)
    meta: dict = field(default_factory=dict)

    @property
    def median(self) -> np.ndarray:
        return self.quantiles[0.5]

    def interval(self, step: int, low: float = 0.1, high: float = 0.9) -> tuple[float, float]:
        return float(self.quantiles[low][step]), float(self.quantiles[high][step])


class Engine(Protocol):
    name: str

    def forecast(self, history, horizon: int, season: int = 12) -> Forecast: ...


def _from_samples(name: str, samples: np.ndarray, meta: dict | None = None) -> Forecast:
    qs = {q: np.quantile(samples, q, axis=0) for q in QUANTILES}
    return Forecast(name, samples.shape[1], qs, samples, meta or {})


def _residual_samples(point: np.ndarray, residuals: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Bootstrap residuals into sample paths; spread grows like sqrt(step)."""
    rng = np.random.default_rng(seed)
    res = residuals[np.isfinite(residuals)]
    if len(res) == 0:
        res = np.zeros(1)
    res = res - np.median(res)   # spread only: the engine's point forecast stays the median
    steps = np.sqrt(np.arange(1, len(point) + 1))
    draws = rng.choice(res, size=(n, len(point)), replace=True) * steps
    return point[None, :] + draws


@dataclass
class SeasonalNaive:
    """Repeat the last season; uncertainty from in-sample seasonal differences."""
    n_samples: int = 200
    seed: int = 0
    name: str = "seasonal_naive"

    def forecast(self, history, horizon: int, season: int = 12) -> Forecast:
        y = np.asarray(history, dtype=float)
        m = season if len(y) > season else 1
        point = np.array([y[len(y) - m + (h % m)] for h in range(horizon)])
        residuals = y[m:] - y[:-m]
        return _from_samples(self.name, _residual_samples(point, residuals, self.n_samples, self.seed))


@dataclass
class StatsForecastEngine:
    """ETS or AutoARIMA via statsforecast (prediction intervals -> quantiles -> samples)."""
    model: str = "ets"          # "ets" | "arima"
    n_samples: int = 200
    seed: int = 0

    @property
    def name(self) -> str:
        return f"sf_{self.model}"

    def forecast(self, history, horizon: int, season: int = 12) -> Forecast:
        from statsforecast.models import AutoARIMA, AutoETS  # lazy

        y = np.asarray(history, dtype=float)
        m = season if len(y) >= 2 * season else 1
        model = AutoETS(season_length=m) if self.model == "ets" else AutoARIMA(season_length=m)
        levels = [20, 40, 60, 80]
        out = model.forecast(y=y, h=horizon, level=levels)
        qs = {0.5: np.asarray(out["mean"], dtype=float)}
        for lv in levels:
            lo, hi = (100 - lv) / 200, 1 - (100 - lv) / 200
            qs[round(lo, 2)] = np.asarray(out[f"lo-{lv}"], dtype=float)
            qs[round(hi, 2)] = np.asarray(out[f"hi-{lv}"], dtype=float)
        return _quantiles_to_forecast(self.name, qs, self.n_samples, self.seed)


@dataclass
class ChronosEngine:
    """Chronos-2 (open weights). Needs ``pip install chronos-forecasting`` and a weight download."""
    model_id: str = "amazon/chronos-2"
    device: str = "cpu"
    n_samples: int = 200
    seed: int = 0
    _pipe: object = None
    name: str = "chronos2"

    def _pipeline(self):
        if self._pipe is None:
            from chronos import BaseChronosPipeline  # lazy; weight download happens here

            self._pipe = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device)
        return self._pipe

    def forecast(self, history, horizon: int, season: int = 12) -> Forecast:
        import torch  # lazy

        pipe = self._pipeline()
        ctx = torch.tensor(np.asarray(history, dtype=np.float32))
        q, _mean = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(QUANTILES))
        arr = q[0].detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q[0])
        arr = arr.reshape(horizon, len(QUANTILES))
        qs = {lv: arr[:, i] for i, lv in enumerate(QUANTILES)}
        return _quantiles_to_forecast(self.name, qs, self.n_samples, self.seed, meta={"model_id": self.model_id})


def _quantiles_to_forecast(name, qs: dict[float, np.ndarray], n: int, seed: int, meta=None) -> Forecast:
    """Inverse-CDF sampling by linear interpolation between quantile levels (tails clamped)."""
    levels = np.array(sorted(qs))
    values = np.stack([qs[lv] for lv in levels])          # (levels, horizon)
    values = np.maximum.accumulate(values, axis=0)        # keep quantiles monotone
    rng = np.random.default_rng(seed)
    u = rng.uniform(levels[0], levels[-1], size=(n, values.shape[1]))
    samples = np.empty_like(u)
    for h in range(values.shape[1]):
        samples[:, h] = np.interp(u[:, h], levels, values[:, h])
    full = _from_samples(name, samples, meta)
    for lv in levels:                                     # report the engine's own quantiles where it has them
        full.quantiles[float(lv)] = values[list(levels).index(lv)]
    return full


def propagate(formula: Callable[..., float], forecasts: dict[str, Forecast], step: int,
              known: dict[str, float] | None = None) -> dict:
    """Evaluate a formula over joint samples (independent engines, same sample index)."""
    known = known or {}
    n = min(f.samples.shape[0] for f in forecasts.values())
    outs = np.array([formula(**known, **{k: f.samples[i, step] for k, f in forecasts.items()}) for i in range(n)])
    return {"median": float(np.median(outs)), "p10": float(np.quantile(outs, 0.1)),
            "p90": float(np.quantile(outs, 0.9)), "samples": outs}


ENGINES: dict[str, Callable[[], Engine]] = {
    "seasonal_naive": SeasonalNaive,
    "ets": lambda: StatsForecastEngine("ets"),
    "arima": lambda: StatsForecastEngine("arima"),
    "chronos2": ChronosEngine,
}
