"""Evaluation helpers shared by labs and automated checks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import t


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    terms = np.divide(2.0 * np.abs(predicted - actual), denominator, out=np.zeros_like(denominator), where=denominator != 0)
    return float(100.0 * np.mean(terms))


def mean_ci95(values: Iterable[float]) -> tuple[float, float, float]:
    """Return a two-sided Student-t interval for an iid sample mean.

    Exact coverage requires iid normal observations; small, skewed samples
    make the approximation uncertain. Independent episodes are sampling units,
    not their dependent time steps. This is not a prediction interval or
    evidence that the sampled episodes cover every operational regime.
    """

    sample = np.asarray(list(values), dtype=float)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError("At least two scalar values are required to estimate a mean interval")
    if not np.isfinite(sample).all():
        raise ValueError("All values must be finite")
    mean = float(sample.mean())
    multiplier = float(t.ppf(0.975, df=sample.size - 1))
    half_width = multiplier * float(sample.std(ddof=1)) / float(np.sqrt(sample.size))
    return mean, mean - half_width, mean + half_width
