"""Explicit response-time columns and units."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from behavio.models.base import _protected_array
from behavio.study import REQUIRED_COLUMNS, Study


class ResponseTimeUnit(StrEnum):
    """Supported physical units for observed response times."""

    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"

    @property
    def seconds_per_unit(self) -> float:
        return 1.0 if self is ResponseTimeUnit.SECONDS else 0.001


@dataclass(frozen=True, slots=True)
class ResponseTimeSpec:
    """The column and physical unit of one response-time observation."""

    column: str = "response_time"
    unit: ResponseTimeUnit = ResponseTimeUnit.SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("response-time column must be a non-empty string")
        if self.column in REQUIRED_COLUMNS:
            raise ValueError("response-time column cannot replace a required Study column")
        object.__setattr__(self, "unit", ResponseTimeUnit(self.unit))

    def read(self, study: Study) -> ResponseTimes:
        """Validate and convert one study column to seconds."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        if self.column not in study.columns:
            raise ValueError(f"study is missing response-time column {self.column!r}")
        raw_values = np.asarray(study[self.column])
        try:
            raw = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("response times must be finite positive numeric values") from None
        if raw.ndim != 1 or not np.all(np.isfinite(raw)) or np.any(raw <= 0):
            raise ValueError("response times must be finite positive numeric values")
        return ResponseTimes(
            column=self.column,
            unit=self.unit,
            raw=raw,
            seconds=raw * self.unit.seconds_per_unit,
        )


@dataclass(frozen=True, slots=True)
class ResponseTimes:
    """Validated response times in their source unit and canonical seconds."""

    column: str
    unit: ResponseTimeUnit
    raw: NDArray[np.float64]
    seconds: NDArray[np.float64]

    def __post_init__(self) -> None:
        unit = ResponseTimeUnit(self.unit)
        raw = _protected_array(self.raw, dtype=np.float64)
        seconds = _protected_array(self.seconds, dtype=np.float64)
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("response-time column must be a non-empty string")
        if raw.ndim != 1 or seconds.shape != raw.shape:
            raise ValueError("response-time arrays must be one-dimensional and aligned")
        if not np.all(np.isfinite(raw)) or np.any(raw <= 0):
            raise ValueError("raw response times must be finite and positive")
        if not np.all(np.isfinite(seconds)) or np.any(seconds <= 0):
            raise ValueError("response times in seconds must be finite and positive")
        if not np.allclose(seconds, raw * unit.seconds_per_unit, rtol=1e-12, atol=0.0):
            raise ValueError("raw and second-valued response times do not match their unit")
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "seconds", seconds)

    def from_seconds(self, seconds: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert canonical seconds back to the declared source unit."""

        values = np.asarray(seconds, dtype=np.float64) / self.unit.seconds_per_unit
        return _protected_array(values, dtype=np.float64)
