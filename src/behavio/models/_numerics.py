"""Deterministic finite-difference helpers shared by the optimizer-backed models.

``behavio.models.baselines`` and ``behavio.models.rl`` each carry a private Hessian
helper of their own. Those copies are deliberately left alone: their fits are pinned by
committed benchmarks, and re-pointing them at a shared implementation would be a
behaviour-affecting change dressed up as a cleanup. New models use this module.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

RELATIVE_STEP = 1e-4


def _step(value: float, relative_step: float) -> float:
    return relative_step * max(1.0, abs(value))


def finite_difference_gradient(
    objective: Callable[[NDArray[np.float64]], float],
    values: NDArray[np.float64],
    *,
    relative_step: float = RELATIVE_STEP,
) -> NDArray[np.float64]:
    """Return the central-difference gradient of a scalar objective."""

    gradient = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        step = _step(float(values[index]), relative_step)
        left = np.array(values, dtype=np.float64)
        right = np.array(values, dtype=np.float64)
        left[index] -= step
        right[index] += step
        gradient[index] = (objective(right) - objective(left)) / (2.0 * step)
    return gradient


def finite_difference_hessian(
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    values: NDArray[np.float64],
    *,
    relative_step: float = RELATIVE_STEP,
) -> NDArray[np.float64]:
    """Return a symmetrized central-difference Hessian built from a gradient callable."""

    size = len(values)
    hessian = np.empty((size, size), dtype=np.float64)
    for column in range(size):
        step = _step(float(values[column]), relative_step)
        left = np.array(values, dtype=np.float64)
        right = np.array(values, dtype=np.float64)
        left[column] -= step
        right[column] += step
        hessian[:, column] = (gradient(right) - gradient(left)) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def covariance_from_hessian(hessian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the pseudo-inverse covariance of a symmetric observed-information matrix."""

    return np.linalg.pinv(hessian, hermitian=True)
