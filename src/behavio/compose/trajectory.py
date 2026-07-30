"""The value type every time-varying model reports its coefficients through."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array

__all__ = ["CoefficientTrajectory"]


@dataclass(frozen=True, slots=True)
class CoefficientTrajectory:
    """Named coefficient values evaluated along an explicit temporal clock."""

    clock: str
    times: NDArray[np.float64]
    coefficient_names: tuple[str, ...]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = protected_array(self.times, dtype=np.float64)
        values = protected_array(self.values, dtype=np.float64)
        names = tuple(self.coefficient_names)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("trajectory times must be a finite one-dimensional array")
        if values.shape != (len(times), len(names)):
            raise ValueError("trajectory values must have one column per coefficient")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory values must be finite")
        if not names or len(set(names)) != len(names):
            raise ValueError("coefficient names must be non-empty and unique")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "values", values)
