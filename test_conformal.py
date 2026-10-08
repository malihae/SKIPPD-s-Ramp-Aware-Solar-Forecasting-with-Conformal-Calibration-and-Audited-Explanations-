import numpy as np

from tcdemo.conformal import AffineCorrector, conformal_quantile, coverage, mean_width


def test_quantile_returns_inf_when_too_few_points():
    # n=5, alpha=0.1 -> k = ceil(6 * 0.9) = 6 > 5
    assert conformal_quantile(np.arange(5.0), 0.1) == float("inf")
    assert conformal_quantile(np.array([]), 0.1) == float("inf")


def test_quantile_picks_expected_order_statistic():
    scores = np.arange(1.0, 10.0)  # n=9, alpha=0.2 -> k = ceil(10*0.8) = 8
    assert conformal_quantile(scores, 0.2) == 8.0


def test_split_conformal_is_valid_under_exchangeability():
    """With iid data the marginal coverage must be >= 1 - alpha (up to Monte-Carlo error)."""
    rng = np.random.default_rng(0)
    alpha, n_cal, n_test, trials = 0.1, 50, 4000, 200
    covs = []
    for _ in range(trials):
        cal = np.abs(rng.normal(size=n_cal))
        q = conformal_quantile(cal, alpha)
        test = rng.normal(size=n_test)
        covs.append(coverage(test, -q, q))
    assert np.mean(covs) >= 1 - alpha - 0.01


def test_coverage_and_width():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo, hi = y - 0.5, y + 0.5
    assert coverage(y, lo, hi) == 1.0
    assert coverage(y + 1.0, lo, hi) == 0.0
    assert mean_width(lo, hi) == 1.0


def test_affine_corrector_recovers_gain_and_offset():
    rng = np.random.default_rng(1)
    mu = rng.uniform(-1, 1, 100)
    y = 1.3 * mu + 0.4
    c = AffineCorrector().fit(mu, y)
    assert abs(c.a - 1.3) < 1e-8 and abs(c.b - 0.4) < 1e-8
    assert np.allclose(c.predict(mu), y)
