"""Two synthetic 'experimental systems' that measure the same underlying property landscape.

The landscape f(x) is a smooth, multimodal function of two normalised process parameters
x = (x1, x2) in [0, 1]^2 (think: composition and annealing temperature). Each system
measures a distorted, noisy version of f:

    y = gain * f(x) + offset + noise(x),   noise(x) ~ N(0, (noise_base + noise_slope * x1)^2)

System A is the "source" instrument (low noise, no distortion). System B is the "target"
instrument (different gain/offset, larger and input-dependent noise) and is also sampled
from a shifted input region. Everything here is synthetic: it is a controlled test bench
for calibration under transfer, not a model of any real instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def landscape(X: np.ndarray) -> np.ndarray:
    """Noise-free property landscape on [0, 1]^2."""
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    ridge = np.sin(3 * np.pi * x1) * np.cos(2 * np.pi * x2)
    peak = 0.5 * np.exp(-((x1 - 0.7) ** 2 + (x2 - 0.3) ** 2) / 0.05)
    trend = 0.3 * x1
    return ridge + peak + trend


@dataclass(frozen=True)
class System:
    name: str
    gain: float = 1.0
    offset: float = 0.0
    noise_base: float = 0.05
    noise_slope: float = 0.0

    def noise_std(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.noise_base + self.noise_slope * X[:, 0]

    def measure(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        X = np.atleast_2d(X)
        clean = self.gain * landscape(X) + self.offset
        return clean + rng.normal(size=len(X)) * self.noise_std(X)


SYSTEM_A = System("A (source)", gain=1.0, offset=0.0, noise_base=0.05, noise_slope=0.0)
SYSTEM_B = System("B (target)", gain=1.15, offset=0.20, noise_base=0.08, noise_slope=0.20)


def sample_inputs_A(n: int, rng: np.random.Generator) -> np.ndarray:
    """Source inputs: uniform over the whole parameter square."""
    return rng.uniform(size=(n, 2))


def sample_inputs_B(n: int, rng: np.random.Generator) -> np.ndarray:
    """Target inputs: covariate shift, with x1 concentrated towards high values."""
    x1 = rng.beta(3.0, 1.5, size=n)
    x2 = rng.uniform(size=n)
    return np.column_stack([x1, x2])
