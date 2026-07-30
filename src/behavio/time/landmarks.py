"""Landmark clocks: learning-time coordinates re-origined at a *learned* event.

A landmark clock is still a clock in the sense of :mod:`behavio.time.clocks` -- it names the
temporal coordinate a trial sits on -- but its origin is not declared by the analyst. It is
detected from the training rows of a fold (a performance threshold crossing, say) and then
frozen, so the test rows are re-origined by state the model has never seen the answer to.
Every clock here therefore arrives as a :class:`~behavio.contracts.transform.StudyTransform`
and is driven by :mod:`behavio.time.transforms`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from behavio.contracts.transform import (
    TransformProvenance,
)
from behavio.time.clocks import (
    ClockedStudy,
    ClockKind,
    ClockScope,
    ClockSpec,
    _scalar,
    _validate_output_column,
    _with_column,
)
from behavio.trials import Study


class LandmarkNotFoundError(ValueError):
    """Raised when a training fold does not contain the requested landmark."""


@dataclass(frozen=True, slots=True)
class LandmarkUncertaintyEstimate:
    """Point estimate and bootstrap draws for one subject's learning landmark."""

    point: float | None
    samples: np.ndarray
    interval_level: float

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float64, copy=True)
        if samples.ndim != 1 or not len(samples):
            raise ValueError("landmark samples must be a non-empty vector")
        if np.any(np.isinf(samples)):
            raise ValueError("landmark samples may contain NaN but not infinity")
        if self.point is not None and not np.isfinite(self.point):
            raise ValueError("point landmark must be finite or None")
        if not np.isfinite(self.interval_level) or not 0 < self.interval_level < 1:
            raise ValueError("interval_level must lie strictly between zero and one")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def n_resolved(self) -> int:
        """Return the number of resamples in which the landmark was detected."""

        return int(np.sum(np.isfinite(self.samples)))

    @property
    def resolution_rate(self) -> float:
        """Return the fraction of bootstrap resamples with a detected landmark."""

        return self.n_resolved / len(self.samples)

    @property
    def median(self) -> float | None:
        """Return the median resolved landmark, or None if no draw resolved."""

        resolved = self.samples[np.isfinite(self.samples)]
        return None if not len(resolved) else float(np.median(resolved))

    @property
    def interval(self) -> tuple[float, float] | None:
        """Return the equal-tailed interval among resolved draws."""

        resolved = self.samples[np.isfinite(self.samples)]
        if not len(resolved):
            return None
        tail = (1.0 - self.interval_level) / 2.0
        lower, upper = np.quantile(resolved, [tail, 1.0 - tail])
        return float(lower), float(upper)


@dataclass(frozen=True, slots=True)
class ThresholdLandmarkUncertainty:
    """Training-only plug-in bootstrap evidence for threshold landmarks."""

    transform_signature: str
    n_fit_trials: int
    fit_subjects: tuple[Any, ...]
    n_resamples: int
    seed: int
    smoothing_window: int
    interval_level: float
    estimates: Mapping[Any, LandmarkUncertaintyEstimate]

    def __post_init__(self) -> None:
        subjects = tuple(self.fit_subjects)
        estimates = MappingProxyType(dict(self.estimates))
        if self.n_fit_trials < 1:
            raise ValueError("n_fit_trials must be positive")
        _require_positive_integer(self.n_resamples, "n_resamples")
        _require_non_negative_integer(self.seed, "seed")
        _require_positive_integer(self.smoothing_window, "smoothing_window")
        if not np.isfinite(self.interval_level) or not 0 < self.interval_level < 1:
            raise ValueError("interval_level must lie strictly between zero and one")
        if set(estimates) != set(subjects):
            raise ValueError("uncertainty estimates must contain every fitted subject")
        if any(len(estimate.samples) != self.n_resamples for estimate in estimates.values()):
            raise ValueError("every subject must retain exactly n_resamples landmark draws")
        object.__setattr__(self, "fit_subjects", subjects)
        object.__setattr__(self, "estimates", estimates)

    @property
    def signature(self) -> str:
        """Return the complete reproducible uncertainty-procedure signature."""

        return (
            f"bernoulli-plugin-bootstrap[transform={self.transform_signature};"
            f"resamples={self.n_resamples};seed={self.seed};"
            f"smoothing_window={self.smoothing_window};interval={self.interval_level}]"
        )


@dataclass(frozen=True, slots=True)
class LandmarkClockSamples:
    """Frozen landmark-relative clock draws in source row order."""

    values: np.ndarray
    clock: ClockSpec

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 2 or not all(values.shape):
            raise ValueError("landmark clock samples must be a non-empty draw-by-trial matrix")
        if np.any(np.isinf(values)):
            raise ValueError("landmark clock samples may contain NaN but not infinity")
        if self.clock.kind is not ClockKind.LANDMARK_RELATIVE or not self.clock.numeric:
            raise ValueError("landmark clock samples require a numeric landmark-relative clock")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def n_resamples(self) -> int:
        return self.values.shape[0]

    @property
    def n_trials(self) -> int:
        return self.values.shape[1]


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
            indices = np.asarray(rows, dtype=np.intp)
            landmark = _detect_threshold_landmark(
                metric[indices],
                clock_values[indices],
                threshold=self.threshold,
                window=self.window,
                consecutive=self.consecutive,
                direction=self.direction,
            )
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

    def fit_with_uncertainty(
        self,
        study: Study,
        *,
        n_resamples: int = 500,
        seed: int = 0,
        smoothing_window: int | None = None,
        interval_level: float = 0.95,
    ) -> FittedThresholdLandmarkClock:
        """Fit point landmarks and a training-only plug-in Bernoulli bootstrap.

        The observed binary metric is causally smoothed within subject using Jeffreys
        regularization. Bernoulli sequences are then drawn at the original chronological
        positions and passed through the unchanged landmark rule. This preserves the
        estimated non-stationary trajectory but does not model residual serial dependence.
        """

        _require_positive_integer(n_resamples, "n_resamples")
        _require_non_negative_integer(seed, "seed")
        selected_window = self.window if smoothing_window is None else smoothing_window
        _require_positive_integer(selected_window, "smoothing_window")
        if not np.isfinite(interval_level) or not 0 < interval_level < 1:
            raise ValueError("interval_level must lie strictly between zero and one")
        fitted = self.fit(study)
        metric = np.asarray(study[self.metric], dtype=np.float64)
        clock_values = np.asarray(study[self.clock.column], dtype=np.float64)
        if np.any((metric != 0.0) & (metric != 1.0)):
            raise ValueError("landmark uncertainty currently requires a binary metric")

        subject_rows = _subject_rows(study)
        child_sequences = np.random.SeedSequence(seed).spawn(len(fitted.fit_subjects))
        estimates: dict[Any, LandmarkUncertaintyEstimate] = {}
        for subject, child_sequence in zip(
            fitted.fit_subjects,
            child_sequences,
            strict=True,
        ):
            rows = np.asarray(subject_rows[subject], dtype=np.intp)
            subject_metric = metric[rows]
            subject_clock = clock_values[rows]
            rates = _causal_bernoulli_rates(subject_metric, selected_window)
            generator = np.random.default_rng(child_sequence)
            samples = np.full(n_resamples, np.nan, dtype=np.float64)
            for draw in range(n_resamples):
                simulated_metric = generator.binomial(1, rates).astype(np.float64)
                landmark = _detect_threshold_landmark(
                    simulated_metric,
                    subject_clock,
                    threshold=self.threshold,
                    window=self.window,
                    consecutive=self.consecutive,
                    direction=self.direction,
                )
                if landmark is not None:
                    samples[draw] = landmark
            estimates[subject] = LandmarkUncertaintyEstimate(
                point=fitted.landmarks[subject],
                samples=samples,
                interval_level=interval_level,
            )

        uncertainty = ThresholdLandmarkUncertainty(
            transform_signature=self.signature,
            n_fit_trials=len(study),
            fit_subjects=fitted.fit_subjects,
            n_resamples=n_resamples,
            seed=seed,
            smoothing_window=selected_window,
            interval_level=interval_level,
            estimates=estimates,
        )
        return replace(fitted, uncertainty=uncertainty)


@dataclass(frozen=True, slots=True)
class BootstrapThresholdLandmarkClock:
    """Configure uncertainty-aware fitting for the generic fold-transform helpers."""

    landmark: ThresholdLandmarkClock
    n_resamples: int = 500
    seed: int = 0
    smoothing_window: int | None = None
    interval_level: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.landmark, ThresholdLandmarkClock):
            raise TypeError("landmark must be a ThresholdLandmarkClock")
        _require_positive_integer(self.n_resamples, "n_resamples")
        _require_non_negative_integer(self.seed, "seed")
        if self.smoothing_window is not None:
            _require_positive_integer(self.smoothing_window, "smoothing_window")
        if not np.isfinite(self.interval_level) or not 0 < self.interval_level < 1:
            raise ValueError("interval_level must lie strictly between zero and one")

    @property
    def signature(self) -> str:
        return self.landmark.signature

    def fit(self, study: Study) -> FittedThresholdLandmarkClock:
        """Fit uncertainty using only the Study supplied by the fold helper."""

        return self.landmark.fit_with_uncertainty(
            study,
            n_resamples=self.n_resamples,
            seed=self.seed,
            smoothing_window=self.smoothing_window,
            interval_level=self.interval_level,
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
    uncertainty: ThresholdLandmarkUncertainty | None = None

    def __post_init__(self) -> None:
        landmarks = MappingProxyType(dict(self.landmarks))
        if set(landmarks) != set(self.fit_subjects):
            raise ValueError("landmarks must contain every fitted subject exactly once")
        if self.uncertainty is not None:
            if self.uncertainty.transform_signature != self.transform_signature:
                raise ValueError("uncertainty and point transform signatures must match")
            if self.uncertainty.n_fit_trials != self.n_fit_trials:
                raise ValueError("uncertainty and point transform fit sizes must match")
            if self.uncertainty.fit_subjects != tuple(self.fit_subjects):
                raise ValueError("uncertainty and point transform subjects must match")
            for subject, landmark in landmarks.items():
                if self.uncertainty.estimates[subject].point != landmark:
                    raise ValueError("uncertainty and point landmarks must match")
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

    def transform_samples(self, study: Study) -> LandmarkClockSamples:
        """Apply frozen bootstrap landmark draws without re-reading outcome metrics."""

        if self.uncertainty is None:
            raise ValueError("fit_with_uncertainty must be used before transforming samples")
        self.source_clock.validate(study)
        _validate_output_column(study, self.output_clock.column)
        clock_values = np.asarray(study[self.source_clock.column], dtype=np.float64)
        values = np.empty((self.uncertainty.n_resamples, len(study)), dtype=np.float64)
        for row in range(len(study)):
            subject = _scalar(study["subject"][row])
            if subject not in self.uncertainty.estimates:
                raise ValueError(
                    f"subject {subject!r} was not present when the landmark was fitted"
                )
            samples = self.uncertainty.estimates[subject].samples
            values[:, row] = float(clock_values[row]) - samples
        return LandmarkClockSamples(
            values=values,
            clock=replace(self.output_clock, allow_missing=True),
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


def _require_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _detect_threshold_landmark(
    metric: np.ndarray,
    clock_values: np.ndarray,
    *,
    threshold: float,
    window: int,
    consecutive: int,
    direction: Literal["above", "below"],
) -> float | None:
    qualifying = 0
    for position in range(window - 1, len(metric)):
        rolling_value = float(np.mean(metric[position - window + 1 : position + 1]))
        meets = rolling_value >= threshold if direction == "above" else rolling_value <= threshold
        qualifying = qualifying + 1 if meets else 0
        if qualifying >= consecutive:
            return float(clock_values[position])
    return None


def _causal_bernoulli_rates(metric: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(metric, dtype=np.float64)))
    rates = np.empty(len(metric), dtype=np.float64)
    for position in range(len(metric)):
        start = max(0, position - window + 1)
        count = position - start + 1
        successes = cumulative[position + 1] - cumulative[start]
        rates[position] = (successes + 0.5) / (count + 1.0)
    return rates
