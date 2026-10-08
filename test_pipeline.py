import numpy as np

from tcdemo.experiment import run_seed
from tcdemo.explore import explore
from tcdemo.systems import SYSTEM_A, SYSTEM_B, landscape, sample_inputs_A, sample_inputs_B


def test_measurements_are_reproducible_for_a_fixed_seed():
    X = sample_inputs_A(10, np.random.default_rng(0))
    y1 = SYSTEM_A.measure(X, np.random.default_rng(5))
    y2 = SYSTEM_A.measure(X, np.random.default_rng(5))
    assert np.array_equal(y1, y2)


def test_system_b_is_distorted_and_noisier_than_a():
    rng = np.random.default_rng(0)
    X = np.full((20000, 2), 0.8)
    resid_a = SYSTEM_A.measure(X, rng) - landscape(X)
    resid_b = SYSTEM_B.measure(X, rng) - (SYSTEM_B.gain * landscape(X) + SYSTEM_B.offset)
    assert resid_b.std() > resid_a.std()
    assert abs(SYSTEM_B.measure(X, rng).mean() - landscape(X).mean()) > 0.1


def test_target_inputs_are_shifted_towards_high_x1():
    rng = np.random.default_rng(0)
    assert sample_inputs_B(5000, rng)[:, 0].mean() > sample_inputs_A(5000, rng)[:, 0].mean() + 0.1


def test_explore_respects_budget():
    X, y, gp = explore(SYSTEM_A, budget=20, n_init=10, rng=np.random.default_rng(0))
    assert X.shape == (20, 2) and y.shape == (20,)
    assert np.isfinite(gp.predict(X)).all()


def test_transfer_degrades_coverage_and_recalibration_restores_it():
    """Averaged over a few seeds: A is near nominal, naive transfer undercovers, recalibration fixes it."""
    rows = []
    for s in range(4):
        rows += run_seed(s, budget=40, n_calib_A=150, n_test=2000, n_target_grid=(80,))
    import pandas as pd
    df = pd.DataFrame(rows).groupby("method").coverage.mean()
    assert df["A in-distribution"] > 0.85
    assert df["B naive (reuse A quantile)"] < 0.80
    assert df["B recalibrated"] > 0.85
