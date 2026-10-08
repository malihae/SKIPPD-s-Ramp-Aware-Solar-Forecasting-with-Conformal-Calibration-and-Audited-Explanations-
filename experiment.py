"""Transfer-calibration experiment.

Pipeline for each random seed
  1. Autonomous exploration on source system A (GP uncertainty sampling) -> model.
  2. Split-conformal calibration on fresh A measurements -> interval half-width q_A.
  3. Evaluate on A (in-distribution) and on target system B (shifted inputs, distorted,
     noisier) with three ways of building intervals:
        naive            reuse q_A on B                       (no target data)
        recalibrated     recompute the conformal quantile on n_B target measurements
        affine+conformal fit y ~ a*mu + b on half of n_B, conformal on the other half
  4. Report marginal coverage, coverage in the high-noise region (x1 > 0.75) and width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import AffineCorrector, conformal_quantile, coverage, mean_width
from .explore import explore
from .systems import SYSTEM_A, SYSTEM_B, sample_inputs_A, sample_inputs_B

HI_NOISE_X1 = 0.75


def _row(seed, method, n_target, y, lo, hi, X):
    hi_mask = X[:, 0] > HI_NOISE_X1
    return {
        "seed": seed,
        "method": method,
        "n_target": n_target,
        "coverage": coverage(y, lo, hi),
        "coverage_high_noise": coverage(y[hi_mask], lo[hi_mask], hi[hi_mask]),
        "width": mean_width(lo, hi),
    }


def run_seed(seed: int, alpha: float = 0.1, budget: int = 60, n_init: int = 15,
             n_calib_A: int = 200, n_test: int = 3000, n_target_grid=(20, 40, 80, 160)) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # 1. exploration on A
    _, _, gp = explore(SYSTEM_A, budget, n_init, rng)

    # 2. conformal calibration on A
    Xc = sample_inputs_A(n_calib_A, rng)
    yc = SYSTEM_A.measure(Xc, rng)
    q_A = conformal_quantile(np.abs(yc - gp.predict(Xc)), alpha)

    # 3a. in-distribution test on A
    Xa = sample_inputs_A(n_test, rng)
    ya = SYSTEM_A.measure(Xa, rng)
    mu_a = gp.predict(Xa)
    rows.append(_row(seed, "A in-distribution", 0, ya, mu_a - q_A, mu_a + q_A, Xa))

    # 3b. transfer to B
    Xb = sample_inputs_B(n_test, rng)
    yb = SYSTEM_B.measure(Xb, rng)
    mu_b = gp.predict(Xb)
    rows.append(_row(seed, "B naive (reuse A quantile)", 0, yb, mu_b - q_A, mu_b + q_A, Xb))

    for n_t in n_target_grid:
        Xt = sample_inputs_B(n_t, rng)
        yt = SYSTEM_B.measure(Xt, rng)
        mu_t = gp.predict(Xt)

        q_B = conformal_quantile(np.abs(yt - mu_t), alpha)
        rows.append(_row(seed, "B recalibrated", n_t, yb, mu_b - q_B, mu_b + q_B, Xb))

        h = n_t // 2
        corr = AffineCorrector().fit(mu_t[:h], yt[:h])
        q_aff = conformal_quantile(np.abs(yt[h:] - corr.predict(mu_t[h:])), alpha)
        centre = corr.predict(mu_b)
        rows.append(_row(seed, "B affine + conformal", n_t, yb, centre - q_aff, centre + q_aff, Xb))
    return rows


def run_experiment(n_seeds: int = 30, **kwargs) -> pd.DataFrame:
    rows: list[dict] = []
    for s in range(n_seeds):
        rows.extend(run_seed(s, **kwargs))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "n_target"])
    out = g.agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        coverage_high_noise_mean=("coverage_high_noise", "mean"),
        width_mean=("width", "mean"),
        width_std=("width", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return out
