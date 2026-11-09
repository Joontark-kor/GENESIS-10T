"""Scores for probabilistic forecasts. Pure numpy, no model needed."""
from __future__ import annotations

import numpy as np


def mase(actual, forecast, history, season: int = 12) -> float:
    """Mean absolute error scaled by the in-sample seasonal-naive error of ``history``."""
    actual, forecast, history = (np.asarray(x, dtype=float) for x in (actual, forecast, history))
    m = season if len(history) > season else 1
    scale = np.mean(np.abs(history[m:] - history[:-m]))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("history is constant or too short for MASE")
    return float(np.mean(np.abs(actual - forecast)) / scale)


def quantile_loss(actual: float, quantiles: dict[float, float]) -> float:
    """Average pinball loss over the given quantile levels."""
    losses = []
    for q, value in quantiles.items():
        diff = actual - value
        losses.append(max(q * diff, (q - 1) * diff))
    return float(np.mean(losses))


def crps_from_quantiles(actual: float, quantiles: dict[float, float]) -> float:
    """CRPS approximated as 2 x mean pinball loss over evenly spread quantile levels."""
    return 2.0 * quantile_loss(actual, quantiles)


def crps_from_samples(actual: float, samples) -> float:
    """Sample (energy-form) CRPS: E|X - y| - 0.5 E|X - X'|."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    term1 = np.mean(np.abs(x - actual))
    # E|X - X'| for sorted samples in O(n)
    weights = 2 * np.arange(1, n + 1) - n - 1
    term2 = 2.0 * np.sum(weights * x) / (n * n)
    return float(term1 - 0.5 * term2)


def covered(actual: float, low: float, high: float) -> bool:
    return low <= actual <= high


def brier(prob: float, outcome: bool) -> float:
    if not 0.0 <= prob <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return (prob - float(outcome)) ** 2


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean (resampling units, e.g. series)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
