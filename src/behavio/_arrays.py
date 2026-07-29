"""Shared array coercion and time-vector validation for observed behaviour.

``behavio.pose``, ``behavio.ethograms``, ``behavio.covariates`` and
``behavio.sync`` all coerce caller-supplied sequences into read-only float
arrays and enforce the same strictly increasing, finite time contract. These
helpers are private and exist only so those four modules agree on it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly_float(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


def _validate_time(time_s: NDArray[np.float64]) -> None:
    if len(time_s) == 0:
        raise ValueError("time_s must contain at least one sample")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must contain only finite values")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing")
