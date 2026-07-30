"""Explicit temporal coordinates for longitudinal studies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from behavio.trials import REQUIRED_COLUMNS, Study


class ClockKind(StrEnum):
    """Scientific meaning of a temporal coordinate."""

    SESSION_ORDER = "session_order"
    CUMULATIVE_TRIAL = "cumulative_trial"
    ELAPSED_TIME = "elapsed_time"
    TASK_PHASE = "task_phase"
    LANDMARK_RELATIVE = "landmark_relative"
    CUSTOM = "custom"


class ClockScope(StrEnum):
    """Population level at which clock values are comparable."""

    SUBJECT = "subject"
    GLOBAL = "global"


class ClockValidationError(ValueError):
    """Raised when a clock is missing, malformed, or temporally inconsistent."""


@dataclass(frozen=True, slots=True)
class ClockSpec:
    """Metadata and validation rules for one explicit Study clock."""

    column: str
    kind: ClockKind
    scope: ClockScope = ClockScope.SUBJECT
    unit: str | None = None
    numeric: bool = True
    monotonic_within_subject: bool = True
    allow_missing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("clock column must be a non-empty string")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValueError("clock unit must be None or a non-empty string")
        object.__setattr__(self, "kind", ClockKind(self.kind))
        object.__setattr__(self, "scope", ClockScope(self.scope))

    def validate(self, study: Study) -> ClockSpec:
        """Validate this clock against a study and return the specification."""

        if self.column not in study.columns:
            raise ClockValidationError(f"study is missing clock column {self.column!r}")
        values = study[self.column]
        if self.numeric:
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                raise ClockValidationError(
                    f"clock column {self.column!r} must be numeric"
                ) from None
            if not self.allow_missing and not np.all(np.isfinite(numeric)):
                raise ClockValidationError(
                    f"clock column {self.column!r} must contain finite values"
                )
            if self.allow_missing and np.any(np.isinf(numeric)):
                raise ClockValidationError(
                    f"clock column {self.column!r} may contain NaN but not infinity"
                )
            if self.monotonic_within_subject:
                _validate_subject_monotonic(study, numeric, self.column)
        elif not self.allow_missing:
            for row, value in enumerate(values):
                if _is_missing(value):
                    raise ClockValidationError(
                        f"clock column {self.column!r} is missing at row {row}"
                    )
        return self


@dataclass(frozen=True, slots=True)
class ClockedStudy:
    """A study paired with the clock specification just added or selected."""

    study: Study
    clock: ClockSpec

    def __post_init__(self) -> None:
        self.clock.validate(self.study)


def session_order_clock() -> ClockSpec:
    """Return the canonical within-subject session-order clock."""

    return ClockSpec(
        column="session_order",
        kind=ClockKind.SESSION_ORDER,
        scope=ClockScope.SUBJECT,
        unit="session",
    )


def with_cumulative_trial_clock(
    study: Study,
    *,
    output: str = "cumulative_trial",
    start: int = 0,
) -> ClockedStudy:
    """Add observed-trial count within subject without changing source row order."""

    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a non-negative integer")
    _validate_output_column(study, output)
    values = np.empty(len(study), dtype=np.int64)
    counts: dict[Any, int] = {}
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        subject = _scalar(study["subject"][index])
        value = counts.get(subject, start)
        values[index] = value
        counts[subject] = value + 1
    clock = ClockSpec(
        column=output,
        kind=ClockKind.CUMULATIVE_TRIAL,
        scope=ClockScope.SUBJECT,
        unit="observed_trial",
    )
    return ClockedStudy(_with_column(study, output, values), clock)


def with_elapsed_time_clock(
    study: Study,
    *,
    source: str,
    output: str = "elapsed_time",
    source_kind: Literal["datetime", "numeric"] = "datetime",
    unit: Literal["seconds", "minutes", "hours", "days"] = "days",
) -> ClockedStudy:
    """Add elapsed time from each subject's first chronological observation.

    Numeric sources are assumed to already use ``unit``. Datetime sources are converted
    from NumPy-compatible timestamps to the requested unit.
    """

    if source not in study.columns:
        raise ClockValidationError(f"study is missing elapsed-time source column {source!r}")
    _validate_output_column(study, output)
    if source_kind not in ("datetime", "numeric"):
        raise ValueError("source_kind must be 'datetime' or 'numeric'")
    divisors = {
        "seconds": 1e9,
        "minutes": 60e9,
        "hours": 3_600e9,
        "days": 86_400e9,
    }
    if unit not in divisors:
        raise ValueError("unit must be seconds, minutes, hours, or days")

    source_values = study[source]
    if source_kind == "datetime":
        try:
            datetimes = np.asarray(source_values).astype("datetime64[ns]")
        except (TypeError, ValueError):
            raise ClockValidationError(
                f"source column {source!r} must contain datetime-compatible values"
            ) from None
        if np.any(np.isnat(datetimes)):
            raise ClockValidationError(f"source column {source!r} must not contain NaT")
        raw = datetimes.astype(np.int64).astype(np.float64) / divisors[unit]
    else:
        try:
            raw = np.asarray(source_values, dtype=np.float64)
        except (TypeError, ValueError):
            raise ClockValidationError(
                f"source column {source!r} must contain numeric values"
            ) from None
        if not np.all(np.isfinite(raw)):
            raise ClockValidationError(f"source column {source!r} must contain finite values")

    _validate_subject_monotonic(study, raw, source)
    elapsed = np.empty(len(study), dtype=np.float64)
    origins: dict[Any, float] = {}
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        subject = _scalar(study["subject"][index])
        origin = origins.setdefault(subject, float(raw[index]))
        elapsed[index] = raw[index] - origin
    clock = ClockSpec(
        column=output,
        kind=ClockKind.ELAPSED_TIME,
        scope=ClockScope.SUBJECT,
        unit=unit,
    )
    return ClockedStudy(_with_column(study, output, elapsed), clock)


def _with_column(study: Study, name: str, values: Sequence[Any] | NDArray[Any]) -> Study:
    _validate_output_column(study, name)
    columns = {column: study[column] for column in study.columns}
    columns[name] = values
    return Study(columns)


def _validate_output_column(study: Study, output: str) -> None:
    if not isinstance(output, str) or not output:
        raise ValueError("output column must be a non-empty string")
    if output in REQUIRED_COLUMNS:
        raise ValueError("a clock cannot replace a required Study column")
    if output in study.columns:
        raise ValueError(f"output column {output!r} already exists")


def _validate_subject_monotonic(study: Study, values: NDArray[np.float64], column: str) -> None:
    previous: dict[Any, float] = {}
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        value = float(values[index])
        if np.isnan(value):
            continue
        subject = _scalar(study["subject"][index])
        if subject in previous and value < previous[subject]:
            raise ClockValidationError(
                f"clock column {column!r} decreases within subject {subject!r}"
            )
        previous[subject] = value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
