"""Fold-fitted temporal transforms with inspectable learned state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from unspool.clocks import (
    ClockedStudy,
    ClockKind,
    ClockScope,
    ClockSpec,
    _scalar,
    _validate_output_column,
    _with_column,
)
from unspool.study import Study
from unspool.validation import ValidationFold


class LandmarkNotFoundError(ValueError):
    """Raised when a training fold does not contain the requested landmark."""


@dataclass(frozen=True, slots=True)
class TransformProvenance:
    """Training-only state retained by a fitted temporal transform."""

    transform_signature: str
    n_fit_trials: int
    fit_subjects: tuple[Any, ...]
    learned_values: Mapping[Any, float | None]

    def __post_init__(self) -> None:
        if self.n_fit_trials < 1:
            raise ValueError("n_fit_trials must be positive")
        object.__setattr__(self, "fit_subjects", tuple(self.fit_subjects))
        object.__setattr__(self, "learned_values", MappingProxyType(dict(self.learned_values)))


@runtime_checkable
class FittedStudyTransform(Protocol):
    """A transform whose learned state is fixed and inspectable."""

    @property
    def signature(self) -> str: ...

    @property
    def output_clock(self) -> ClockSpec: ...

    @property
    def provenance(self) -> TransformProvenance: ...

    def transform(self, study: Study) -> ClockedStudy: ...


@runtime_checkable
class StudyTransform(Protocol):
    """A transform that learns state from one training Study."""

    @property
    def signature(self) -> str: ...

    def fit(self, study: Study) -> FittedStudyTransform: ...


@dataclass(frozen=True, slots=True)
class ThresholdLandmarkClock:
    """Learn the first sustained rolling-metric threshold within each subject."""

    clock: ClockSpec
    metric: str
    output: str = "landmark_time"
    threshold: float = 0.8
    window: int = 20
    consecutive: int = 1
    direction: Literal["above", "below"] = "above"
    on_missing: Literal["error", "nan"] = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be a non-empty Study column name")
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("output must be a non-empty Study column name")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        _require_positive_integer(self.window, "window")
        _require_positive_integer(self.consecutive, "consecutive")
        if self.direction not in ("above", "below"):
            raise ValueError("direction must be 'above' or 'below'")
        if self.on_missing not in ("error", "nan"):
            raise ValueError("on_missing must be 'error' or 'nan'")
        if not self.clock.numeric:
            raise ValueError("threshold landmarks require a numeric source clock")

    @property
    def signature(self) -> str:
        return (
            f"threshold-landmark[clock={self.clock.column};metric={self.metric};"
            f"threshold={self.threshold};window={self.window};"
            f"consecutive={self.consecutive};direction={self.direction};"
            f"on_missing={self.on_missing};output={self.output}]"
        )

    def fit(self, study: Study) -> FittedThresholdLandmarkClock:
        """Estimate one detection-time landmark per subject from training rows only."""

        self.clock.validate(study)
        _validate_output_column(study, self.output)
        if self.metric not in study.columns:
            raise ValueError(f"study is missing landmark metric column {self.metric!r}")
        try:
            metric = np.asarray(study[self.metric], dtype=np.float64)
            clock_values = np.asarray(study[self.clock.column], dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("landmark metric and source clock must be numeric") from None
        if not np.all(np.isfinite(metric)) or not np.all(np.isfinite(clock_values)):
            raise ValueError("landmark metric and source clock must be finite in training data")

        subject_rows = _subject_rows(study)
        landmarks: dict[Any, float | None] = {}
        missing: list[Any] = []
        for subject, rows in subject_rows.items():
            qualifying = 0
            landmark: float | None = None
            for position in range(self.window - 1, len(rows)):
                window_rows = rows[position - self.window + 1 : position + 1]
                rolling_value = float(np.mean(metric[list(window_rows)]))
                meets = (
                    rolling_value >= self.threshold
                    if self.direction == "above"
                    else rolling_value <= self.threshold
                )
                qualifying = qualifying + 1 if meets else 0
                if qualifying >= self.consecutive:
                    landmark = float(clock_values[rows[position]])
                    break
            landmarks[subject] = landmark
            if landmark is None:
                missing.append(subject)

        if missing and self.on_missing == "error":
            raise LandmarkNotFoundError(
                f"landmark not found in training data for subjects: {missing!r}"
            )
        return FittedThresholdLandmarkClock(
            source_clock=self.clock,
            output_clock=ClockSpec(
                column=self.output,
                kind=ClockKind.LANDMARK_RELATIVE,
                scope=ClockScope.SUBJECT,
                unit=self.clock.unit,
                allow_missing=self.on_missing == "nan",
            ),
            transform_signature=self.signature,
            landmarks=landmarks,
            n_fit_trials=len(study),
            fit_subjects=study.subjects,
        )


@dataclass(frozen=True, slots=True)
class FittedThresholdLandmarkClock:
    """A threshold landmark learned from one training fold."""

    source_clock: ClockSpec
    output_clock: ClockSpec
    transform_signature: str
    landmarks: Mapping[Any, float | None]
    n_fit_trials: int
    fit_subjects: tuple[Any, ...]

    def __post_init__(self) -> None:
        landmarks = MappingProxyType(dict(self.landmarks))
        if set(landmarks) != set(self.fit_subjects):
            raise ValueError("landmarks must contain every fitted subject exactly once")
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "fit_subjects", tuple(self.fit_subjects))

    @property
    def signature(self) -> str:
        return self.transform_signature

    @property
    def provenance(self) -> TransformProvenance:
        return TransformProvenance(
            transform_signature=self.signature,
            n_fit_trials=self.n_fit_trials,
            fit_subjects=self.fit_subjects,
            learned_values=self.landmarks,
        )

    def transform(self, study: Study) -> ClockedStudy:
        """Subtract learned subject landmarks without re-estimating them."""

        self.source_clock.validate(study)
        _validate_output_column(study, self.output_clock.column)
        clock_values = np.asarray(study[self.source_clock.column], dtype=np.float64)
        relative = np.empty(len(study), dtype=np.float64)
        for row in range(len(study)):
            subject = _scalar(study["subject"][row])
            if subject not in self.landmarks:
                raise ValueError(
                    f"subject {subject!r} was not present when the landmark was fitted"
                )
            landmark = self.landmarks[subject]
            relative[row] = np.nan if landmark is None else float(clock_values[row]) - landmark
        transformed = _with_column(study, self.output_clock.column, relative)
        return ClockedStudy(transformed, self.output_clock)


@dataclass(frozen=True, slots=True)
class FoldTransformResult:
    """Training and test studies transformed by training-only fitted state."""

    split: ValidationFold
    fitted_transform: FittedStudyTransform
    training: ClockedStudy
    testing: ClockedStudy


def fit_transform_split(
    transform: StudyTransform,
    study: Study,
    split: ValidationFold,
    *,
    require_prospective: bool = True,
) -> FoldTransformResult:
    """Fit on training rows and apply the frozen transform to both sides of one split."""

    if require_prospective and not split.prospective:
        raise ValueError(
            f"split scheme {split.scheme!r} is not prospective; "
            "set require_prospective=False only for an intentional interpolation analysis"
        )
    training = study.take(split.train_indices)
    testing = study.take(split.test_indices)
    fitted = transform.fit(training)
    return FoldTransformResult(
        split=split,
        fitted_transform=fitted,
        training=fitted.transform(training),
        testing=fitted.transform(testing),
    )


def fit_transform_splits(
    transform: StudyTransform,
    study: Study,
    splits: Iterable[ValidationFold],
    *,
    require_prospective: bool = True,
) -> tuple[FoldTransformResult, ...]:
    """Apply ``fit_transform_split`` independently at every supplied origin."""

    return tuple(
        fit_transform_split(
            transform,
            study,
            split,
            require_prospective=require_prospective,
        )
        for split in splits
    )


def _subject_rows(study: Study) -> dict[Any, tuple[int, ...]]:
    rows: dict[Any, list[int]] = {}
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        subject = _scalar(study["subject"][index])
        rows.setdefault(subject, []).append(index)
    return {subject: tuple(indices) for subject, indices in rows.items()}


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
