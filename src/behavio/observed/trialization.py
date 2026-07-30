"""The bridge from continuously observed behaviour to trial-level ``Study`` columns.

:mod:`behavio.observed.pose`, :mod:`behavio.observed.ethograms` and
:mod:`behavio.observed.covariates` hold behaviour as it was observed: float64
seconds on a named physical ``clock_id``.
:class:`behavio.trials.Study` holds behaviour as it was designed: one row per
trial, keyed by ``subject``/``session``/``trial``/``session_order``, with no time
column at all. Nothing related the two, so a covariate and a study could not be
joined even in principle. This module is that join.

The missing coordinate is trial timing, and it is *declared*, never guessed.
:class:`TrialTiming` is a first-class artifact: it names the clock its onsets are
measured on, carries the same ``subject``/``session``/``trial`` keys a ``Study``
uses, and is checked against the observed signal's ``clock_id`` before anything
is reduced. This follows :class:`behavio.time.clocks.ClockSpec`, which makes a
longitudinal time coordinate an explicit object rather than a convention, and
:meth:`behavio.observed.covariates.BehaviorCovariate.aligned_to`, which refuses a clock
mismatch instead of assuming one. Two clocks are related only by
:class:`behavio.observed.device_clocks.DeviceClockSync`; :meth:`TrialTiming.synchronized_to`
routes through it so that trial onsets acquire the same synchronisation lineage
as the pose and annotations they will be reduced against.

A reduction never returns a bare number. :class:`TrialReduction` carries the
value, the fraction of the trial window actually covered by valid observation,
and a :class:`TrialCoverageStatus` per trial, so a window that ran past the end
of the recording, straddled a gap, or contained no valid sample is visible in
the study rather than indistinguishable from a confident measurement.

Leakage discipline: every reducer shipped here reads only the samples inside one
trial's own window, so it is fold-independent by construction and safe to apply
before splitting. Anything that must *learn* from data - a threshold, a
normalisation, a baseline - is not a reducer. Produce the raw per-trial column
here, then fit the learned part as a :class:`behavio.contracts.transform.StudyTransform`
inside a training fold with :func:`behavio.time.transforms.fit_transform_split`.
Reducers declare this with ``fold_independent``; a reducer that declares
``False`` is refused rather than silently applied to a whole study.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from behavio.observed._arrays import _identity, _readonly_float, _validate_time
from behavio.observed.covariates import BehaviorCovariate
from behavio.observed.device_clocks import DeviceClockSync
from behavio.observed.ethograms import BehaviorAnnotations
from behavio.trials import REQUIRED_COLUMNS, Study

_COVERAGE_TOLERANCE = 1e-9


class TrializationError(ValueError):
    """Raised when observed behaviour cannot be reduced onto declared trials."""


class TrialCoverageStatus(StrEnum):
    """How much declared trial window a reduction actually observed."""

    OK = "ok"
    PARTIAL_COVERAGE = "partial_coverage"
    BELOW_MINIMUM_COVERAGE = "below_minimum_coverage"
    NO_VALID_SAMPLES = "no_valid_samples"
    OUTSIDE_OBSERVED_SPAN = "outside_observed_span"
    NOT_REDUCED = "not_reduced"


@dataclass(frozen=True)
class TrialTiming:
    """Declared trial onsets, and optional offsets, on one named physical clock.

    One instance describes one ``subject``/``session``. ``trial`` uses the same
    numbering as the :class:`behavio.trials.Study` rows the reduction will be
    attached to; that is the join key, which is why no time column is added to
    ``Study``. Onsets and trial numbers must both increase strictly, so trial
    order in time and trial order in the study can never disagree silently.
    """

    subject: Hashable
    session: Hashable
    trial: tuple[int, ...]
    onset_s: NDArray[np.float64]
    clock_id: str
    offset_s: NDArray[np.float64] | None = None
    source: str = "declared"
    clock_synchronization_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        onset = _readonly_float(self.onset_s, name="onset_s")
        _validate_time(onset)
        offset = None
        if self.offset_s is not None:
            offset = _readonly_float(self.offset_s, name="offset_s")
        trial = tuple(_trial_number(value, position) for position, value in enumerate(self.trial))
        if len(trial) != len(onset):
            raise TrializationError("trial and onset_s must have equal length")
        if any(later <= earlier for earlier, later in pairwise(trial)):
            raise TrializationError("trial numbers must be strictly increasing")
        if offset is not None:
            if len(offset) != len(onset):
                raise TrializationError("offset_s must have the same length as onset_s")
            if not np.all(np.isfinite(offset)):
                raise TrializationError("offset_s must contain only finite values")
            if np.any(offset <= onset):
                raise TrializationError("each offset_s must be greater than its onset_s")
        if not str(self.clock_id).strip():
            raise TrializationError("trial timing clock_id must be non-empty")
        if not str(self.source).strip():
            raise TrializationError("trial timing source must be non-empty")
        synchronization_ids = tuple(str(value) for value in self.clock_synchronization_ids)
        if any(not value.strip() for value in synchronization_ids):
            raise TrializationError("clock synchronization IDs must be non-empty")
        object.__setattr__(self, "subject", _identity(self.subject, name="subject"))
        object.__setattr__(self, "session", _identity(self.session, name="session"))
        object.__setattr__(self, "trial", trial)
        object.__setattr__(self, "onset_s", onset)
        object.__setattr__(self, "offset_s", offset)
        object.__setattr__(self, "clock_id", str(self.clock_id))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "clock_synchronization_ids", synchronization_ids)

    @classmethod
    def from_arrays(
        cls,
        *,
        subject: Hashable,
        session: Hashable,
        onset_s: ArrayLike,
        clock_id: str,
        trial: Sequence[int] | ArrayLike | None = None,
        offset_s: ArrayLike | None = None,
        source: str = "declared",
        clock_synchronization_ids: Sequence[str] = (),
    ) -> TrialTiming:
        """Declare trial timing, numbering trials ``0..n-1`` when ``trial`` is omitted."""

        onsets = _readonly_float(onset_s, name="onset_s")
        numbers = tuple(range(len(onsets))) if trial is None else tuple(int(v) for v in trial)  # type: ignore[call-overload]
        return cls(
            subject=subject,
            session=session,
            trial=numbers,
            onset_s=onsets,
            clock_id=str(clock_id),
            offset_s=None if offset_s is None else _readonly_float(offset_s, name="offset_s"),
            source=source,
            clock_synchronization_ids=tuple(str(value) for value in clock_synchronization_ids),
        )

    def __len__(self) -> int:
        return len(self.trial)

    @property
    def n_trials(self) -> int:
        """Number of declared trials."""

        return len(self.trial)

    def anchor_time_s(self, anchor: Literal["onset", "offset"]) -> NDArray[np.float64]:
        """Return the declared anchor times, refusing an undeclared offset anchor."""

        if anchor == "onset":
            return self.onset_s
        if anchor != "offset":
            raise TrializationError("anchor must be 'onset' or 'offset'")
        if self.offset_s is None:
            raise TrializationError(
                "this trial timing declares no offset_s; an offset-locked window "
                "requires offsets rather than an assumed trial duration"
            )
        return self.offset_s

    def windows(self, window: TrialWindow) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return per-trial window bounds in this timing's own clock."""

        start = self.anchor_time_s(window.start_anchor) + window.start_offset_s
        stop = self.anchor_time_s(window.stop_anchor) + window.stop_offset_s
        if np.any(stop <= start):
            raise TrializationError(
                "every trial window must have positive duration; a mixed-anchor "
                "window collapsed on at least one trial"
            )
        return start, stop

    def synchronized_to(
        self,
        synchronization: DeviceClockSync,
        *,
        maximum_extrapolation_s: float = 0.0,
    ) -> TrialTiming:
        """Map declared onsets onto the synchronisation's target clock.

        Trial timing is a physical measurement like any other, so it moves
        between clocks the same way a pose or a covariate does: through an
        accepted :class:`behavio.observed.device_clocks.DeviceClockSync`, retaining its
        identifier in the returned lineage.
        """

        if synchronization.source_clock_id != self.clock_id:
            raise TrializationError(
                f"clock synchronization expects {synchronization.source_clock_id!r}, "
                f"trial timing is on {self.clock_id!r}"
            )
        return TrialTiming(
            subject=self.subject,
            session=self.session,
            trial=self.trial,
            onset_s=synchronization.transform_time(
                self.onset_s,
                maximum_extrapolation_s=maximum_extrapolation_s,
            ),
            clock_id=synchronization.target_clock_id,
            offset_s=(
                None
                if self.offset_s is None
                else synchronization.transform_time(
                    self.offset_s,
                    maximum_extrapolation_s=maximum_extrapolation_s,
                )
            ),
            source=self.source,
            clock_synchronization_ids=(
                *self.clock_synchronization_ids,
                synchronization.synchronization_id,
            ),
        )


def trial_timing_from_events(
    annotations: BehaviorAnnotations,
    *,
    onset_label: str,
    offset_label: str | None = None,
    include_points: bool = True,
    edge: Literal["onset", "offset"] = "onset",
    first_trial: int = 0,
) -> TrialTiming:
    """Declare trial timing from one named event stream in an ethogram.

    The label is supplied by the caller; nothing here decides which of a
    session's events delimits a trial. Offsets are optional and, when requested,
    are paired to onsets one-to-one in time order, so a missing or extra offset
    is an error rather than a silently shortened session.
    """

    times = annotations.event_times(edge=edge, include_points=include_points)
    if onset_label not in times:
        raise TrializationError(
            f"annotations have no {onset_label!r} events; observed {sorted(times)}"
        )
    onsets = np.asarray(times[onset_label], dtype=float)
    if not len(onsets):
        raise TrializationError(f"annotations have no {onset_label!r} events")
    offsets: NDArray[np.float64] | None = None
    if offset_label is not None:
        if offset_label not in times:
            raise TrializationError(
                f"annotations have no {offset_label!r} events; observed {sorted(times)}"
            )
        offsets = np.asarray(times[offset_label], dtype=float)
        if len(offsets) != len(onsets):
            raise TrializationError(
                f"{onset_label!r} has {len(onsets)} events and {offset_label!r} has "
                f"{len(offsets)}; pair them explicitly rather than truncating"
            )
    if isinstance(first_trial, bool) or not isinstance(first_trial, int) or first_trial < 0:
        raise TrializationError("first_trial must be a non-negative integer")
    return TrialTiming(
        subject=annotations.subject,
        session=annotations.session,
        trial=tuple(range(first_trial, first_trial + len(onsets))),
        onset_s=onsets,
        clock_id=annotations.clock_id,
        offset_s=offsets,
        source=f"{annotations.source}:{onset_label}",
        clock_synchronization_ids=annotations.clock_synchronization_ids,
    )


@dataclass(frozen=True)
class TrialWindow:
    """A trial-relative window, anchored to a declared onset or offset.

    ``start_anchor`` and ``stop_anchor`` allow onset-locked, offset-locked and
    mixed windows, so a window can span a whole trial (``onset`` to ``offset``)
    or a fixed epoch around either edge. Whether a window may run into the next
    trial is a scientific decision, so ``on_next_trial_overlap`` makes it a
    declaration; overlap is recorded per trial either way.
    """

    start_offset_s: float
    stop_offset_s: float
    start_anchor: Literal["onset", "offset"] = "onset"
    stop_anchor: Literal["onset", "offset"] = "onset"
    on_next_trial_overlap: Literal["allow", "reject"] = "reject"

    def __post_init__(self) -> None:
        for value, name in (
            (self.start_offset_s, "start_offset_s"),
            (self.stop_offset_s, "stop_offset_s"),
        ):
            if not np.isfinite(value):
                raise TrializationError(f"{name} must be finite")
        for anchor, name in (
            (self.start_anchor, "start_anchor"),
            (self.stop_anchor, "stop_anchor"),
        ):
            if anchor not in {"onset", "offset"}:
                raise TrializationError(f"{name} must be 'onset' or 'offset'")
        if self.on_next_trial_overlap not in {"allow", "reject"}:
            raise TrializationError("on_next_trial_overlap must be 'allow' or 'reject'")
        if self.start_anchor == self.stop_anchor and self.stop_offset_s <= self.start_offset_s:
            raise TrializationError("stop_offset_s must exceed start_offset_s on a shared anchor")
        object.__setattr__(self, "start_offset_s", float(self.start_offset_s))
        object.__setattr__(self, "stop_offset_s", float(self.stop_offset_s))

    @property
    def description(self) -> str:
        """Return a reproducible description of the declared window."""

        return (
            f"[{self.start_anchor}{self.start_offset_s:+g}s, "
            f"{self.stop_anchor}{self.stop_offset_s:+g}s]"
        )


@runtime_checkable
class TrialCovariateReducer(Protocol):
    """Reduce the valid covariate samples inside one trial window to one number.

    Implementations receive only the samples the mask already accepted, already
    restricted to the window, so a reducer never has to re-derive validity.
    ``fold_independent`` must be ``True`` unless the reduction learns something
    from data, in which case it belongs in a fold-fitted ``StudyTransform``.
    """

    @property
    def name(self) -> str:
        """Short reducer name used to build a default column name."""

    @property
    def fold_independent(self) -> bool:
        """Whether this reduction reads only one trial's own window."""

    def unit(self, source_unit: str) -> str:
        """Return the unit of the reduced value given the covariate's unit."""

    def reduce(
        self,
        time_s: NDArray[np.float64],
        values: NDArray[np.float64],
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        """Return the reduced value for one trial."""


@runtime_checkable
class TrialAnnotationReducer(Protocol):
    """Reduce an ethogram to one number inside one trial window."""

    @property
    def name(self) -> str:
        """Short reducer name used to build a default column name."""

    @property
    def unit(self) -> str:
        """Unit of the reduced value."""

    @property
    def fold_independent(self) -> bool:
        """Whether this reduction reads only one trial's own window."""

    def reduce(
        self,
        annotations: BehaviorAnnotations,
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        """Return the reduced value for one trial."""


@dataclass(frozen=True)
class _SampleStatistic:
    """Shared behaviour for the summary statistics of in-window valid samples."""

    @property
    def fold_independent(self) -> bool:
        return True

    def unit(self, source_unit: str) -> str:
        return source_unit


@dataclass(frozen=True)
class MeanValue(_SampleStatistic):
    """Unweighted mean of the valid covariate samples inside the window."""

    @property
    def name(self) -> str:
        return "mean"

    def reduce(
        self,
        time_s: NDArray[np.float64],
        values: NDArray[np.float64],
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        return float(np.mean(values))


@dataclass(frozen=True)
class MedianValue(_SampleStatistic):
    """Median of the valid covariate samples inside the window."""

    @property
    def name(self) -> str:
        return "median"

    def reduce(
        self,
        time_s: NDArray[np.float64],
        values: NDArray[np.float64],
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        return float(np.median(values))


@dataclass(frozen=True)
class MinimumValue(_SampleStatistic):
    """Smallest valid covariate sample inside the window."""

    @property
    def name(self) -> str:
        return "minimum"

    def reduce(
        self,
        time_s: NDArray[np.float64],
        values: NDArray[np.float64],
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        return float(np.min(values))


@dataclass(frozen=True)
class MaximumValue(_SampleStatistic):
    """Largest valid covariate sample inside the window."""

    @property
    def name(self) -> str:
        return "maximum"

    def reduce(
        self,
        time_s: NDArray[np.float64],
        values: NDArray[np.float64],
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        return float(np.max(values))


@dataclass(frozen=True)
class FractionOfTimeInState:
    """Proportion of the trial window spent inside bouts of one label.

    Overlapping same-label bouts are unioned rather than summed, so the result
    can never exceed one.
    """

    label: str

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise TrializationError("state label must be non-empty")

    @property
    def name(self) -> str:
        return f"{self.label}_fraction_of_time"

    @property
    def unit(self) -> str:
        return "proportion"

    @property
    def fold_independent(self) -> bool:
        return True

    def reduce(
        self,
        annotations: BehaviorAnnotations,
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        spans = [
            (max(interval.start_s, window_start_s), min(interval.stop_s, window_stop_s))
            for interval in annotations.intervals
            if interval.label == self.label
        ]
        occupied = _union_measure([span for span in spans if span[1] > span[0]])
        return occupied / (window_stop_s - window_start_s)


@dataclass(frozen=True)
class EventCount:
    """Number of occurrences of one label inside the half-open trial window.

    The window is half-open in ``[start, stop)`` so that back-to-back windows
    cannot count the same event twice.
    """

    label: str
    include_points: bool = True
    edge: Literal["onset", "offset"] = "onset"

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise TrializationError("event label must be non-empty")
        if self.edge not in {"onset", "offset"}:
            raise TrializationError("edge must be 'onset' or 'offset'")

    @property
    def name(self) -> str:
        return f"{self.label}_count"

    @property
    def unit(self) -> str:
        return "count"

    @property
    def fold_independent(self) -> bool:
        return True

    def reduce(
        self,
        annotations: BehaviorAnnotations,
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        times = _label_times(annotations, self.label, self.edge, self.include_points)
        inside = (times >= window_start_s) & (times < window_stop_s)
        return float(np.count_nonzero(inside))


@dataclass(frozen=True)
class FirstOccurrenceLatency:
    """Latency from the window's start anchor to the first occurrence of a label.

    Returns ``NaN`` when the label does not occur in the window. That ``NaN`` is
    a censored observation, not missing data: the window was observed and the
    behaviour did not happen. ``TrialReduction.status`` distinguishes it from a
    window that was never observed at all.
    """

    label: str
    include_points: bool = True
    edge: Literal["onset", "offset"] = "onset"

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise TrializationError("event label must be non-empty")
        if self.edge not in {"onset", "offset"}:
            raise TrializationError("edge must be 'onset' or 'offset'")

    @property
    def name(self) -> str:
        return f"{self.label}_latency"

    @property
    def unit(self) -> str:
        return "s"

    @property
    def fold_independent(self) -> bool:
        return True

    def reduce(
        self,
        annotations: BehaviorAnnotations,
        *,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
    ) -> float:
        times = _label_times(annotations, self.label, self.edge, self.include_points)
        inside = times[(times >= window_start_s) & (times < window_stop_s)]
        return float(inside[0] - anchor_s) if len(inside) else float("nan")


@dataclass(frozen=True)
class TrialReduction:
    """One reduced per-trial column with its coverage and per-trial status.

    ``values`` is ``NaN`` wherever ``status`` says the window was not observed
    well enough to summarise, so a partially covered trial can never be mistaken
    for a confident measurement.
    """

    name: str
    unit: str
    subject: Hashable
    session: Hashable
    trial: tuple[int, ...]
    values: NDArray[np.float64]
    coverage: NDArray[np.float64]
    status: tuple[TrialCoverageStatus, ...]
    valid_sample_count: NDArray[np.int64]
    overlaps_next_trial: NDArray[np.bool_]
    window: TrialWindow
    reducer: str
    clock_id: str
    minimum_coverage: float
    source: str
    clock_synchronization_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise TrializationError("reduction name must be non-empty")
        lengths = {
            len(self.trial),
            len(self.values),
            len(self.coverage),
            len(self.status),
            len(self.valid_sample_count),
            len(self.overlaps_next_trial),
        }
        if len(lengths) != 1:
            raise TrializationError("reduction arrays must have equal length")
        values = np.array(self.values, dtype=float, copy=True)
        coverage = np.array(self.coverage, dtype=float, copy=True)
        counts = np.array(self.valid_sample_count, dtype=np.int64, copy=True)
        overlaps = np.array(self.overlaps_next_trial, dtype=bool, copy=True)
        for array in (values, coverage, counts, overlaps):
            array.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "valid_sample_count", counts)
        object.__setattr__(self, "overlaps_next_trial", overlaps)
        object.__setattr__(self, "status", tuple(TrialCoverageStatus(v) for v in self.status))

    @property
    def n_reduced(self) -> int:
        """Number of trials that produced a usable value."""

        return int(np.count_nonzero(np.isfinite(self.values)))

    @property
    def column_names(self) -> tuple[str, str, str]:
        """Value, coverage and status column names this reduction contributes."""

        return (self.name, f"{self.name}_coverage", f"{self.name}_status")


def reduce_covariate_to_trials(
    covariate: BehaviorCovariate,
    *,
    timing: TrialTiming,
    window: TrialWindow,
    reducer: TrialCovariateReducer,
    max_gap_s: float,
    minimum_coverage: float,
    name: str | None = None,
) -> TrialReduction:
    """Reduce a continuous covariate to one value per declared trial.

    ``max_gap_s`` has the same meaning as in
    :meth:`behavio.observed.covariates.BehaviorCovariate.aligned_to`: consecutive valid
    samples further apart than this do not span the time between them, so the
    intervening window time counts as unobserved. ``minimum_coverage`` is the
    fraction of the window that must be observed before a value is reported at
    all; below it the trial returns ``NaN`` with an explanatory status.
    """

    _require_fold_independent(reducer)
    _require_matching_stream(covariate.subject, covariate.session, covariate.clock_id, timing)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise TrializationError("max_gap_s must be finite and positive")
    coverage_floor = _validate_minimum_coverage(minimum_coverage)

    start_s, stop_s = timing.windows(window)
    anchors = timing.anchor_time_s(window.start_anchor)
    overlaps = _next_trial_overlap(timing, stop_s, window)
    observed = (float(covariate.time_s[0]), float(covariate.time_s[-1]))
    valid_time = covariate.time_s[covariate.valid]
    valid_values = covariate.values[covariate.valid]

    values = np.full(len(timing), np.nan, dtype=float)
    coverage = np.zeros(len(timing), dtype=float)
    counts = np.zeros(len(timing), dtype=np.int64)
    status: list[TrialCoverageStatus] = []
    for index in range(len(timing)):
        start = float(start_s[index])
        stop = float(stop_s[index])
        coverage[index] = _sampled_coverage(
            covariate.time_s,
            covariate.valid,
            start=start,
            stop=stop,
            max_gap_s=max_gap_s,
        )
        inside = (valid_time >= start) & (valid_time <= stop)
        counts[index] = int(np.count_nonzero(inside))
        if min(stop, observed[1]) <= max(start, observed[0]):
            status.append(TrialCoverageStatus.OUTSIDE_OBSERVED_SPAN)
            continue
        if not counts[index]:
            status.append(TrialCoverageStatus.NO_VALID_SAMPLES)
            continue
        if coverage[index] + _COVERAGE_TOLERANCE < coverage_floor:
            status.append(TrialCoverageStatus.BELOW_MINIMUM_COVERAGE)
            continue
        values[index] = reducer.reduce(
            valid_time[inside],
            valid_values[inside],
            window_start_s=start,
            window_stop_s=stop,
            anchor_s=float(anchors[index]),
        )
        status.append(_covered_status(coverage[index]))

    return TrialReduction(
        name=name or f"{covariate.name}_{reducer.name}",
        unit=reducer.unit(covariate.unit),
        subject=timing.subject,
        session=timing.session,
        trial=timing.trial,
        values=values,
        coverage=coverage,
        status=tuple(status),
        valid_sample_count=counts,
        overlaps_next_trial=overlaps,
        window=window,
        reducer=type(reducer).__name__,
        clock_id=timing.clock_id,
        minimum_coverage=coverage_floor,
        source=covariate.source,
        clock_synchronization_ids=covariate.clock_synchronization_ids,
    )


def reduce_annotations_to_trials(
    annotations: BehaviorAnnotations,
    *,
    timing: TrialTiming,
    window: TrialWindow,
    reducer: TrialAnnotationReducer,
    observed_span_s: tuple[float, float],
    minimum_coverage: float,
    name: str | None = None,
) -> TrialReduction:
    """Reduce an ethogram to one value per declared trial.

    An ethogram carries no sampling grid, so the span the annotator actually
    scored cannot be inferred from it and must be declared as
    ``observed_span_s``. Coverage is the fraction of each window inside that
    span. Within a covered window the absence of a bout is a real zero rather
    than missingness, which is why a covered trial always reports a value.
    """

    _require_fold_independent(reducer)
    _require_matching_stream(annotations.subject, annotations.session, annotations.clock_id, timing)
    span_start, span_stop = (float(observed_span_s[0]), float(observed_span_s[1]))
    if not np.isfinite(span_start) or not np.isfinite(span_stop) or span_stop <= span_start:
        raise TrializationError("observed_span_s must be a finite, increasing (start, stop)")
    coverage_floor = _validate_minimum_coverage(minimum_coverage)

    start_s, stop_s = timing.windows(window)
    anchors = timing.anchor_time_s(window.start_anchor)
    overlaps = _next_trial_overlap(timing, stop_s, window)

    values = np.full(len(timing), np.nan, dtype=float)
    coverage = np.zeros(len(timing), dtype=float)
    counts = np.zeros(len(timing), dtype=np.int64)
    status: list[TrialCoverageStatus] = []
    for index in range(len(timing)):
        start = float(start_s[index])
        stop = float(stop_s[index])
        overlap = min(stop, span_stop) - max(start, span_start)
        coverage[index] = max(overlap, 0.0) / (stop - start)
        if overlap <= 0:
            status.append(TrialCoverageStatus.OUTSIDE_OBSERVED_SPAN)
            continue
        if coverage[index] + _COVERAGE_TOLERANCE < coverage_floor:
            status.append(TrialCoverageStatus.BELOW_MINIMUM_COVERAGE)
            continue
        values[index] = reducer.reduce(
            annotations,
            window_start_s=start,
            window_stop_s=stop,
            anchor_s=float(anchors[index]),
        )
        counts[index] = 1
        status.append(_covered_status(coverage[index]))

    return TrialReduction(
        name=name or reducer.name,
        unit=reducer.unit,
        subject=timing.subject,
        session=timing.session,
        trial=timing.trial,
        values=values,
        coverage=coverage,
        status=tuple(status),
        valid_sample_count=counts,
        overlaps_next_trial=overlaps,
        window=window,
        reducer=type(reducer).__name__,
        clock_id=timing.clock_id,
        minimum_coverage=coverage_floor,
        source=annotations.source,
        clock_synchronization_ids=annotations.clock_synchronization_ids,
    )


def attach_trial_columns(
    study: Study,
    reductions: Iterable[TrialReduction],
    *,
    on_missing: Literal["error", "nan"] = "error",
    include_coverage: bool = True,
    include_status: bool = True,
) -> Study:
    """Join reductions onto a study by ``subject``/``session``/``trial``.

    Source row order is preserved: values are written to the row position each
    key already occupies, never sorted into the reduction's order. Coverage and
    status travel with the value by default, so a downstream model cannot read
    the number without being able to see how much of the window supported it.
    """

    if on_missing not in {"error", "nan"}:
        raise TrializationError("on_missing must be 'error' or 'nan'")
    grouped = _group_reductions(reductions)
    if not grouped:
        raise TrializationError("attach_trial_columns requires at least one reduction")

    columns: dict[str, Any] = {column: study[column] for column in study.columns}
    for column_name, group in grouped.items():
        for candidate in _emitted_names(column_name, include_coverage, include_status):
            if candidate in REQUIRED_COLUMNS:
                raise TrializationError("a reduction cannot replace a required Study column")
            if candidate in columns:
                raise TrializationError(f"study already has a column named {candidate!r}")

        lookup: dict[tuple[Any, Any, int], tuple[float, float, str]] = {}
        for reduction in group:
            for position, trial in enumerate(reduction.trial):
                key = (reduction.subject, reduction.session, int(trial))
                if key in lookup:
                    raise TrializationError(
                        f"reductions named {column_name!r} cover {key!r} more than once"
                    )
                lookup[key] = (
                    float(reduction.values[position]),
                    float(reduction.coverage[position]),
                    str(reduction.status[position]),
                )

        values = np.full(len(study), np.nan, dtype=float)
        coverage = np.full(len(study), np.nan, dtype=float)
        status = [str(TrialCoverageStatus.NOT_REDUCED)] * len(study)
        missing: list[tuple[Any, Any, int]] = []
        for row in range(len(study)):
            key = (
                _study_key(study["subject"][row]),
                _study_key(study["session"][row]),
                int(study["trial"][row]),
            )
            record = lookup.get(key)
            if record is None:
                missing.append(key)
                continue
            values[row], coverage[row], status[row] = record
        if missing and on_missing == "error":
            raise TrializationError(
                f"reduction {column_name!r} does not cover {len(missing)} study rows, "
                f"starting at {missing[0]!r}; pass on_missing='nan' to keep them empty"
            )
        columns[column_name] = values
        if include_coverage:
            columns[f"{column_name}_coverage"] = coverage
        if include_status:
            columns[f"{column_name}_status"] = np.asarray(status, dtype=object)
    return Study(columns)


def _emitted_names(name: str, include_coverage: bool, include_status: bool) -> tuple[str, ...]:
    emitted = [name]
    if include_coverage:
        emitted.append(f"{name}_coverage")
    if include_status:
        emitted.append(f"{name}_status")
    return tuple(emitted)


def _group_reductions(
    reductions: Iterable[TrialReduction],
) -> dict[str, list[TrialReduction]]:
    grouped: dict[str, list[TrialReduction]] = {}
    for reduction in reductions:
        existing = grouped.setdefault(reduction.name, [])
        if existing and (existing[0].unit, existing[0].reducer) != (
            reduction.unit,
            reduction.reducer,
        ):
            raise TrializationError(
                f"reductions named {reduction.name!r} disagree on unit or reducer; "
                "one column cannot hold two different quantities"
            )
        existing.append(reduction)
    return grouped


def _covered_status(coverage: float) -> TrialCoverageStatus:
    if coverage >= 1.0 - _COVERAGE_TOLERANCE:
        return TrialCoverageStatus.OK
    return TrialCoverageStatus.PARTIAL_COVERAGE


def _validate_minimum_coverage(minimum_coverage: float) -> float:
    if not np.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
        raise TrializationError("minimum_coverage must lie between zero and one")
    return float(minimum_coverage)


def _require_fold_independent(reducer: Any) -> None:
    if not getattr(reducer, "fold_independent", False):
        raise TrializationError(
            f"{type(reducer).__name__} declares fold_independent=False; a reduction that "
            "learns from data must be fitted inside a training fold. Reduce the raw "
            "per-trial quantity here, then fit the learned step as a StudyTransform "
            "with behavio.time.transforms.fit_transform_split."
        )


def _require_matching_stream(
    subject: Hashable,
    session: Hashable,
    clock_id: str,
    timing: TrialTiming,
) -> None:
    if clock_id != timing.clock_id:
        raise TrializationError(
            f"observed behaviour is on clock {clock_id!r} and trial timing is on "
            f"{timing.clock_id!r}; relate them with fit_device_clock_sync() and "
            "TrialTiming.synchronized_to() rather than renaming a clock"
        )
    observed = (_identity(subject, name="subject"), _identity(session, name="session"))
    declared = (timing.subject, timing.session)
    if observed != declared:
        raise TrializationError(
            f"observed behaviour is {observed!r} and trial timing is {declared!r}"
        )


def _next_trial_overlap(
    timing: TrialTiming,
    stop_s: NDArray[np.float64],
    window: TrialWindow,
) -> NDArray[np.bool_]:
    overlaps = np.zeros(len(timing), dtype=bool)
    if len(timing) > 1:
        overlaps[:-1] = stop_s[:-1] > timing.onset_s[1:]
    if window.on_next_trial_overlap == "reject" and np.any(overlaps):
        first = int(np.flatnonzero(overlaps)[0])
        raise TrializationError(
            f"window {window.description} on trial {timing.trial[first]} runs past the "
            f"next trial's onset; declare on_next_trial_overlap='allow' to accept "
            "windows that share time between trials"
        )
    overlaps.setflags(write=False)
    return overlaps


def _sampled_coverage(
    time_s: NDArray[np.float64],
    valid: NDArray[np.bool_],
    *,
    start: float,
    stop: float,
    max_gap_s: float,
) -> float:
    if len(time_s) < 2:
        return 0.0
    spans_valid = valid[:-1] & valid[1:]
    gaps = np.diff(time_s)
    usable = spans_valid & (gaps <= max_gap_s)
    if not np.any(usable):
        return 0.0
    lower = np.maximum(time_s[:-1][usable], start)
    upper = np.minimum(time_s[1:][usable], stop)
    covered = float(np.sum(np.maximum(upper - lower, 0.0)))
    return min(covered / (stop - start), 1.0)


def _union_measure(spans: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    current_start: float | None = None
    current_stop = 0.0
    for start, stop in sorted(spans):
        if current_start is None:
            current_start, current_stop = start, stop
            continue
        if start > current_stop:
            total += current_stop - current_start
            current_start, current_stop = start, stop
        else:
            current_stop = max(current_stop, stop)
    if current_start is not None:
        total += current_stop - current_start
    return total


def _label_times(
    annotations: BehaviorAnnotations,
    label: str,
    edge: Literal["onset", "offset"],
    include_points: bool,
) -> NDArray[np.float64]:
    times = annotations.event_times(edge=edge, include_points=include_points)
    return np.asarray(times.get(label, ()), dtype=float)


def _trial_number(value: Any, position: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TrializationError(f"trial must contain integers; entry {position} is {value!r}")
    if value < 0:
        raise TrializationError(f"trial must be non-negative; entry {position} is {value!r}")
    return int(value)


def _study_key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
