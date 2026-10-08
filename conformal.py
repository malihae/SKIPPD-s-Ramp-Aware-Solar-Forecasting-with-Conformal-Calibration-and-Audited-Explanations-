"""Split conformal prediction utilities (absolute-residual score)."""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of calibration scores.

    Returns +inf when there are too few calibration points for the requested level
    (ceil((n + 1)(1 - alpha)) > n), in which case no finite interval is valid.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of observations inside [lo, hi]."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


class AffineCorrector:
    """Least-squares map y ~ a * mu + b, fitted on a handful of target-system measurements.

    A cheap stand-in for a transfer step: it absorbs gain/offset differences between
    instruments before the conformal step measures what error is left.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, mu: np.ndarray, y: np.ndarray) -> "AffineCorrector":
        mu, y = np.asarray(mu, dtype=float), np.asarray(y, dtype=float)
        A = np.column_stack([mu, np.ones_like(mu)])
        (self.a, self.b), *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, mu: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(mu, dtype=float) + self.b
