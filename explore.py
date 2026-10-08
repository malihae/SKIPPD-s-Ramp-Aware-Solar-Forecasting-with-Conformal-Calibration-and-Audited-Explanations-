"""A minimal autonomous-exploration loop: Gaussian-process uncertainty sampling."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .systems import System, sample_inputs_A


def fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 0) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF([0.3, 0.3], (0.05, 2.0)) + WhiteKernel(1e-2, (1e-5, 1e0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def explore(system: System, budget: int, n_init: int, rng: np.random.Generator, pool_size: int = 1500):
    """Run `budget` experiments on `system`, choosing each new point where the GP is least certain.

    Returns the collected inputs, measurements and the final fitted GP.
    """
    if budget < n_init:
        raise ValueError("budget must be >= n_init")
    X = sample_inputs_A(n_init, rng)
    y = system.measure(X, rng)
    for _ in range(budget - n_init):
        gp = fit_gp(X, y)
        pool = sample_inputs_A(pool_size, rng)
        _, std = gp.predict(pool, return_std=True)
        x_next = pool[np.argmax(std)][None, :]
        X = np.vstack([X, x_next])
        y = np.append(y, system.measure(x_next, rng))
    return X, y, fit_gp(X, y)
