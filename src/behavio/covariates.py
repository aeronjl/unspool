"""One timestamped behavioural covariate with an explicit validity mask.

A covariate is any named scalar observed over time alongside behaviour -
confidence-gated speed, pupil area, a state probability. Values and validity
stay separate so that missingness survives alignment instead of being filled in
silently.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from behavio._arrays import _identity, _readonly_float, _validate_time


@dataclass(frozen=True)
class BehaviorCovariate:
    """One timestamped external covariate with an explicit validity mask.

    ``subject`` and ``session`` accept any hashable identifier, matching
    :class:`behavio.study.Study`, so an integer subject id survives the join
    onto a trial-level study without a lossy ``str()``.
    """

    subject: Hashable
    session: Hashable
    name: str
    time_s: NDArray[np.float64]
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    unit: str
    source: str
    clock_id: str
    source_version: str | None = None
    source_artifact: str | None = None
    clock_synchronization_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        time_s = _readonly_float(self.time_s, name="time_s")
        values = _readonly_float(self.values, name="values")
        valid = np.array(self.valid, dtype=bool, copy=True)
        if valid.ndim != 1:
            raise ValueError("valid must be one-dimensional")
        if len(time_s) != len(values) or len(time_s) != len(valid):
            raise ValueError("time_s, values, and valid must have equal length")
        _validate_time(time_s)
        if not self.name.strip() or not self.unit.strip():
            raise ValueError("covariate name and unit must be non-empty")
        if not self.source.strip() or not self.clock_id.strip():
            raise ValueError("covariate source and clock_id must be non-empty")
        synchronization_ids = tuple(str(value) for value in self.clock_synchronization_ids)
        if any(not value.strip() for value in synchronization_ids):
            raise ValueError("clock synchronization IDs must be non-empty")
        valid &= np.isfinite(values)
        valid.setflags(write=False)
        object.__setattr__(self, "subject", _identity(self.subject, name="subject"))
        object.__setattr__(self, "session", _identity(self.session, name="session"))
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "clock_synchronization_ids", synchronization_ids)

    def align_to(
        self,
        target_time_s: ArrayLike,
        *,
        target_clock_id: str,
        max_gap_s: float,
    ) -> NDArray[np.float64]:
        """Return aligned values; use :meth:`aligned_to` to retain the mask."""

        return self.aligned_to(
            target_time_s,
            target_clock_id=target_clock_id,
            max_gap_s=max_gap_s,
        ).values

    def aligned_to(
        self,
        target_time_s: ArrayLike,
        *,
        target_clock_id: str,
        max_gap_s: float,
    ) -> BehaviorCovariate:
        """Interpolate within valid runs and retain the aligned validity mask."""

        if target_clock_id != self.clock_id:
            raise ValueError("clock mismatch; synchronize externally and declare a shared clock_id")
        target = _readonly_float(target_time_s, name="target_time_s")
        _validate_time(target)
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and positive")

        aligned = np.full(len(target), np.nan, dtype=float)
        indices = np.flatnonzero(self.valid)
        if not len(indices):
            return self._aligned_covariate(target, aligned)

        split_after = np.flatnonzero(
            (np.diff(indices) != 1) | (np.diff(self.time_s[indices]) > max_gap_s)
        )
        runs = np.split(indices, split_after + 1)
        for run in runs:
            run_time = self.time_s[run]
            run_values = self.values[run]
            if len(run) == 1:
                matches = np.isclose(target, run_time[0], rtol=0.0, atol=1e-12)
                aligned[matches] = run_values[0]
                continue
            inside = (target >= run_time[0]) & (target <= run_time[-1])
            aligned[inside] = np.interp(target[inside], run_time, run_values)
        return self._aligned_covariate(target, aligned)

    def _aligned_covariate(
        self,
        target_time_s: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> BehaviorCovariate:
        return BehaviorCovariate(
            subject=self.subject,
            session=self.session,
            name=self.name,
            time_s=target_time_s,
            values=values,
            valid=np.isfinite(values),
            unit=self.unit,
            source=self.source,
            clock_id=self.clock_id,
            source_version=self.source_version,
            source_artifact=self.source_artifact,
            clock_synchronization_ids=self.clock_synchronization_ids,
        )
