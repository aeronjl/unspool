"""Matched-pulse synchronisation between two physical hardware clocks.

Two devices that timestamp the same experiment rarely agree. Given pulses
observed on both, :func:`fit_device_clock_sync` fits the affine
offset-and-drift mapping ``target = intercept + scale * source``, refuses it
against prospective residual, drift, pulse-count and span thresholds, and
retains every pulse residual as evidence. Applying the accepted mapping to a
pose, covariate or ethogram rewrites its ``clock_id`` and appends the
synchronisation ID to its lineage.

A *device* clock is a physical hardware clock measured in seconds. It is a different
object from the learning-time clocks in :mod:`behavio.time.clocks`, which name the
temporal coordinate a trial sits on; every name here carries the ``DeviceClock`` prefix
so the two never have to be told apart from context.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from behavio.observed._arrays import _readonly_float, _validate_time
from behavio.observed.covariates import BehaviorCovariate
from behavio.observed.ethograms import BehaviorAnnotations, BehaviorInterval
from behavio.observed.pose import PoseTrajectory


@dataclass(frozen=True)
class DeviceClockPulses:
    """Explicit one-to-one synchronization pulses observed on two clocks."""

    source_clock_id: str
    target_clock_id: str
    source_time_s: tuple[float, ...]
    target_time_s: tuple[float, ...]
    match_labels: tuple[str, ...] = ()

    @classmethod
    def from_arrays(
        cls,
        *,
        source_clock_id: str,
        target_clock_id: str,
        source_time_s: ArrayLike,
        target_time_s: ArrayLike,
        match_labels: Sequence[str] | None = None,
    ) -> DeviceClockPulses:
        """Create declared pulse pairs without attempting automatic matching."""

        source = _readonly_float(source_time_s, name="source_time_s")
        target = _readonly_float(target_time_s, name="target_time_s")
        return cls(
            source_clock_id=str(source_clock_id),
            target_clock_id=str(target_clock_id),
            source_time_s=tuple(float(value) for value in source),
            target_time_s=tuple(float(value) for value in target),
            match_labels=tuple(str(value) for value in (match_labels or ())),
        )

    def __post_init__(self) -> None:
        if not self.source_clock_id.strip() or not self.target_clock_id.strip():
            raise ValueError("clock IDs must be non-empty")
        if self.source_clock_id == self.target_clock_id:
            raise ValueError("source and target clock IDs must differ")
        source = tuple(float(value) for value in self.source_time_s)
        target = tuple(float(value) for value in self.target_time_s)
        if len(source) != len(target):
            raise ValueError("source and target pulse counts must match")
        if any(not np.isfinite(value) for value in (*source, *target)):
            raise ValueError("pulse times must be finite")
        if any(right <= left for left, right in pairwise(source)):
            raise ValueError("source pulse times must be strictly increasing")
        if any(right <= left for left, right in pairwise(target)):
            raise ValueError("target pulse times must be strictly increasing")
        labels = tuple(str(value) for value in self.match_labels)
        if labels and len(labels) != len(source):
            raise ValueError("match_labels must be empty or match the pulse count")
        if any(not value.strip() for value in labels):
            raise ValueError("match labels must be non-empty")
        if labels and len(labels) != len(set(labels)):
            raise ValueError("match labels must be unique")
        object.__setattr__(self, "source_time_s", source)
        object.__setattr__(self, "target_time_s", target)
        object.__setattr__(self, "match_labels", labels)


@dataclass(frozen=True)
class DeviceClockSyncSpec:
    """Prospective acceptance thresholds for an affine clock mapping."""

    maximum_absolute_residual_s: float
    maximum_drift_ppm: float
    minimum_matches: int = 3
    minimum_source_span_s: float = 1.0
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.maximum_absolute_residual_s)
            or self.maximum_absolute_residual_s <= 0
        ):
            raise ValueError("maximum_absolute_residual_s must be finite and positive")
        if not np.isfinite(self.maximum_drift_ppm) or self.maximum_drift_ppm < 0:
            raise ValueError("maximum_drift_ppm must be finite and non-negative")
        if self.minimum_matches < 3:
            raise ValueError("minimum_matches must be at least three")
        if not np.isfinite(self.minimum_source_span_s) or self.minimum_source_span_s <= 0:
            raise ValueError("minimum_source_span_s must be finite and positive")


@dataclass(frozen=True)
class DeviceClockSync:
    """Accepted affine mapping and complete pulse-level diagnostic evidence."""

    source_clock_id: str
    target_clock_id: str
    intercept_s: float
    scale: float
    drift_ppm: float
    source_start_s: float
    source_stop_s: float
    source_span_s: float
    matched_pulses: int
    source_pulse_time_s: tuple[float, ...]
    target_pulse_time_s: tuple[float, ...]
    match_labels: tuple[str, ...]
    fitted_target_time_s: tuple[float, ...]
    residual_s: tuple[float, ...]
    root_mean_square_residual_s: float
    median_absolute_residual_s: float
    maximum_absolute_residual_s: float
    allowed_maximum_absolute_residual_s: float
    allowed_maximum_drift_ppm: float
    minimum_matches: int
    minimum_source_span_s: float
    synchronization_id: str
    method: Literal["affine_matched_pulses"] = "affine_matched_pulses"
    artifact_type: Literal["clock_synchronization"] = "clock_synchronization"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.source_clock_id.strip() or not self.target_clock_id.strip():
            raise ValueError("clock IDs must be non-empty")
        if self.source_clock_id == self.target_clock_id:
            raise ValueError("source and target clock IDs must differ")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("clock synchronization scale must be finite and positive")
        scalars = (
            self.intercept_s,
            self.drift_ppm,
            self.source_start_s,
            self.source_stop_s,
            self.source_span_s,
            self.root_mean_square_residual_s,
            self.median_absolute_residual_s,
            self.maximum_absolute_residual_s,
            self.allowed_maximum_absolute_residual_s,
            self.allowed_maximum_drift_ppm,
            self.minimum_source_span_s,
        )
        if any(not np.isfinite(value) for value in scalars):
            raise ValueError("clock synchronization diagnostics must be finite")
        if self.matched_pulses < 3 or self.minimum_matches < 3:
            raise ValueError("clock synchronization requires at least three pulses")
        evidence_lengths = {
            len(self.source_pulse_time_s),
            len(self.target_pulse_time_s),
            len(self.fitted_target_time_s),
            len(self.residual_s),
        }
        if evidence_lengths != {self.matched_pulses}:
            raise ValueError("clock pulse evidence must match matched_pulses")
        if self.match_labels and len(self.match_labels) != self.matched_pulses:
            raise ValueError("clock match labels must match matched_pulses")
        if self.source_stop_s <= self.source_start_s or self.source_span_s <= 0:
            raise ValueError("clock synchronization source span must be positive")
        if self.maximum_absolute_residual_s < 0:
            raise ValueError("clock synchronization residual must be non-negative")
        if self.maximum_absolute_residual_s > self.allowed_maximum_absolute_residual_s + 1e-12:
            raise ValueError("clock synchronization exceeds its residual threshold")
        if abs(self.drift_ppm) > self.allowed_maximum_drift_ppm + 1e-12:
            raise ValueError("clock synchronization exceeds its drift threshold")
        if not self.synchronization_id.strip():
            raise ValueError("synchronization_id must be non-empty")

    def to_json(self) -> str:
        """Serialize the accepted mapping and all matched-pulse evidence."""

        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def transform_time(
        self,
        source_time_s: ArrayLike,
        *,
        maximum_extrapolation_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """Map source times, refusing extrapolation beyond a declared allowance."""

        source = _readonly_float(source_time_s, name="source_time_s")
        if not len(source):
            return source
        _validate_time(source)
        self._validate_extrapolation(source, maximum_extrapolation_s)
        return _readonly_float(
            self.intercept_s + self.scale * source,
            name="target_time_s",
        )

    def synchronize_covariate(
        self,
        covariate: BehaviorCovariate,
        *,
        maximum_extrapolation_s: float = 0.0,
    ) -> BehaviorCovariate:
        """Transform covariate timestamps and append synchronization provenance."""

        self._require_source_clock(covariate.clock_id)
        return BehaviorCovariate(
            subject=covariate.subject,
            session=covariate.session,
            name=covariate.name,
            time_s=self.transform_time(
                covariate.time_s,
                maximum_extrapolation_s=maximum_extrapolation_s,
            ),
            values=covariate.values,
            valid=covariate.valid,
            unit=covariate.unit,
            source=covariate.source,
            clock_id=self.target_clock_id,
            source_version=covariate.source_version,
            source_artifact=covariate.source_artifact,
            clock_synchronization_ids=(
                *covariate.clock_synchronization_ids,
                self.synchronization_id,
            ),
        )

    def synchronize_pose(
        self,
        pose: PoseTrajectory,
        *,
        maximum_extrapolation_s: float = 0.0,
    ) -> PoseTrajectory:
        """Transform pose timestamps without changing coordinates or confidence."""

        self._require_source_clock(pose.clock_id)
        return PoseTrajectory(
            subject=pose.subject,
            session=pose.session,
            keypoint=pose.keypoint,
            time_s=self.transform_time(
                pose.time_s,
                maximum_extrapolation_s=maximum_extrapolation_s,
            ),
            x=pose.x,
            y=pose.y,
            confidence=pose.confidence,
            coordinate_unit=pose.coordinate_unit,
            source=pose.source,
            clock_id=self.target_clock_id,
            z=pose.z,
            reference_frame=pose.reference_frame,
            confidence_definition=pose.confidence_definition,
            individual=pose.individual,
            source_version=pose.source_version,
            source_artifact=pose.source_artifact,
            clock_synchronization_ids=(
                *pose.clock_synchronization_ids,
                self.synchronization_id,
            ),
        )

    def synchronize_annotations(
        self,
        annotations: BehaviorAnnotations,
        *,
        maximum_extrapolation_s: float = 0.0,
    ) -> BehaviorAnnotations:
        """Transform point and interval times while retaining their semantics."""

        self._require_source_clock(annotations.clock_id)
        points = {
            label: tuple(
                float(value)
                for value in self.transform_time(
                    times,
                    maximum_extrapolation_s=maximum_extrapolation_s,
                )
            )
            for label, times in annotations.point_events.items()
        }
        intervals = tuple(
            BehaviorInterval(
                label=item.label,
                start_s=float(
                    self.transform_time(
                        [item.start_s],
                        maximum_extrapolation_s=maximum_extrapolation_s,
                    )[0]
                ),
                stop_s=float(
                    self.transform_time(
                        [item.stop_s],
                        maximum_extrapolation_s=maximum_extrapolation_s,
                    )[0]
                ),
                confidence=item.confidence,
            )
            for item in annotations.intervals
        )
        return BehaviorAnnotations(
            subject=annotations.subject,
            session=annotations.session,
            point_events=points,
            intervals=intervals,
            source=annotations.source,
            clock_id=self.target_clock_id,
            source_version=annotations.source_version,
            source_artifact=annotations.source_artifact,
            clock_synchronization_ids=(
                *annotations.clock_synchronization_ids,
                self.synchronization_id,
            ),
        )

    def _require_source_clock(self, clock_id: str) -> None:
        if clock_id != self.source_clock_id:
            raise ValueError(
                f"clock synchronization expects {self.source_clock_id!r}, received {clock_id!r}"
            )

    def _validate_extrapolation(
        self,
        source_time_s: NDArray[np.float64],
        maximum_extrapolation_s: float,
    ) -> None:
        if not np.isfinite(maximum_extrapolation_s) or maximum_extrapolation_s < 0:
            raise ValueError("maximum_extrapolation_s must be finite and non-negative")
        before = max(0.0, self.source_start_s - float(source_time_s[0]))
        after = max(0.0, float(source_time_s[-1]) - self.source_stop_s)
        required = max(before, after)
        if required > maximum_extrapolation_s + 1e-12:
            raise ValueError(
                f"clock transform requires {required:g}s extrapolation; allowed "
                f"maximum is {maximum_extrapolation_s:g}s"
            )


def fit_device_clock_sync(
    matches: DeviceClockPulses,
    spec: DeviceClockSyncSpec,
) -> DeviceClockSync:
    """Fit and validate ``target_time = intercept + scale * source_time``."""

    source = np.asarray(matches.source_time_s, dtype=float)
    target = np.asarray(matches.target_time_s, dtype=float)
    if len(source) < spec.minimum_matches:
        raise ValueError(
            f"clock synchronization has {len(source)} matches; minimum is {spec.minimum_matches}"
        )
    source_span = float(source[-1] - source[0])
    if source_span < spec.minimum_source_span_s:
        raise ValueError(
            f"clock synchronization spans {source_span:g}s; minimum is "
            f"{spec.minimum_source_span_s:g}s"
        )
    centered_source = source - np.mean(source)
    centered_target = target - np.mean(target)
    denominator = float(centered_source @ centered_source)
    scale = float(centered_source @ centered_target / denominator)
    intercept = float(np.mean(target) - scale * np.mean(source))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("clock synchronization must produce a positive finite scale")
    fitted = intercept + scale * source
    residual = target - fitted
    drift_ppm = (scale - 1.0) * 1e6
    maximum_residual = float(np.max(np.abs(residual)))
    if abs(drift_ppm) > spec.maximum_drift_ppm:
        raise ValueError(
            f"clock drift is {drift_ppm:g} ppm; allowed magnitude is {spec.maximum_drift_ppm:g} ppm"
        )
    if maximum_residual > spec.maximum_absolute_residual_s:
        raise ValueError(
            f"clock maximum absolute residual is {maximum_residual:g}s; allowed "
            f"maximum is {spec.maximum_absolute_residual_s:g}s"
        )
    evidence = {
        "source_clock_id": matches.source_clock_id,
        "target_clock_id": matches.target_clock_id,
        "source_time_s": matches.source_time_s,
        "target_time_s": matches.target_time_s,
        "match_labels": matches.match_labels,
        "spec": asdict(spec),
        "intercept_s": intercept,
        "scale": scale,
    }
    digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return DeviceClockSync(
        source_clock_id=matches.source_clock_id,
        target_clock_id=matches.target_clock_id,
        intercept_s=intercept,
        scale=scale,
        drift_ppm=drift_ppm,
        source_start_s=float(source[0]),
        source_stop_s=float(source[-1]),
        source_span_s=source_span,
        matched_pulses=len(source),
        source_pulse_time_s=matches.source_time_s,
        target_pulse_time_s=matches.target_time_s,
        match_labels=matches.match_labels,
        fitted_target_time_s=tuple(float(value) for value in fitted),
        residual_s=tuple(float(value) for value in residual),
        root_mean_square_residual_s=float(np.sqrt(np.mean(residual**2))),
        median_absolute_residual_s=float(np.median(np.abs(residual))),
        maximum_absolute_residual_s=maximum_residual,
        allowed_maximum_absolute_residual_s=spec.maximum_absolute_residual_s,
        allowed_maximum_drift_ppm=spec.maximum_drift_ppm,
        minimum_matches=spec.minimum_matches,
        minimum_source_span_s=spec.minimum_source_span_s,
        synchronization_id=f"clock-sync-{digest}",
    )
