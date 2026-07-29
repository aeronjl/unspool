"""Point events and labelled behavioural intervals, and the readers for them.

An ethogram is a catalogue of what an animal did and when. :class:`BehaviorInterval`
is one labelled bout with physical start and stop; :class:`BehaviorAnnotations`
collects those bouts alongside instantaneous point events. The readers for
Keypoint-MoSeq and BORIS live here because annotations are what they produce.

The module is named ``ethograms`` rather than ``annotations`` so that it does
not read as ``from __future__ import annotations`` at a glance.
"""

from __future__ import annotations

import csv
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from behavio._arrays import _identity, _readonly_float, _validate_time
from behavio.covariates import BehaviorCovariate


@dataclass(frozen=True)
class BehaviorInterval:
    """One externally identified behavioral state or bout."""

    label: str
    start_s: float
    stop_s: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("interval label must be non-empty")
        if not np.isfinite(self.start_s) or not np.isfinite(self.stop_s):
            raise ValueError("interval bounds must be finite")
        if self.stop_s <= self.start_s:
            raise ValueError("interval stop_s must be greater than start_s")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("interval confidence must lie between zero and one")


@dataclass(frozen=True)
class IntervalEncodingInputs:
    """Aligned interval edges, durations and physical bounds for encoding models."""

    events: Mapping[str, tuple[float, ...]]
    event_values: Mapping[str, Mapping[str, tuple[float, ...]]]
    intervals: Mapping[str, tuple[tuple[float, float], ...]]
    edge: Literal["onset", "offset"]
    schema_version: str = "1"


@dataclass(frozen=True)
class BehaviorAnnotations:
    """Point events and intervals discovered by an external behavior tool.

    ``subject`` and ``session`` accept any hashable identifier, matching
    :class:`behavio.study.Study`.
    """

    subject: Hashable
    session: Hashable
    point_events: Mapping[str, tuple[float, ...]]
    intervals: tuple[BehaviorInterval, ...]
    source: str
    clock_id: str
    source_version: str | None = None
    source_artifact: str | None = None
    clock_synchronization_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.clock_id.strip():
            raise ValueError("annotation source and clock_id must be non-empty")
        synchronization_ids = tuple(str(value) for value in self.clock_synchronization_ids)
        if any(not value.strip() for value in synchronization_ids):
            raise ValueError("clock synchronization IDs must be non-empty")
        normalized: dict[str, tuple[float, ...]] = {}
        for label, times in self.point_events.items():
            if not label.strip():
                raise ValueError("point-event labels must be non-empty")
            values = tuple(float(value) for value in times)
            if any(not np.isfinite(value) for value in values):
                raise ValueError("point-event times must be finite")
            normalized[label] = tuple(sorted(values))
        object.__setattr__(self, "subject", _identity(self.subject, name="subject"))
        object.__setattr__(self, "session", _identity(self.session, name="session"))
        object.__setattr__(self, "point_events", normalized)
        object.__setattr__(
            self,
            "intervals",
            tuple(sorted(self.intervals, key=lambda item: (item.start_s, item.stop_s))),
        )
        object.__setattr__(self, "clock_synchronization_ids", synchronization_ids)

    def event_times(
        self,
        *,
        edge: Literal["onset", "offset"] = "onset",
        include_points: bool = True,
    ) -> Mapping[str, tuple[float, ...]]:
        """Return event-kernel-ready times without discarding interval identity."""

        collected = {label: list(times) for label, times in self.point_events.items()}
        if not include_points:
            collected = {}
        attribute = "start_s" if edge == "onset" else "stop_s"
        for interval in self.intervals:
            collected.setdefault(interval.label, []).append(getattr(interval, attribute))
        return {label: tuple(sorted(times)) for label, times in sorted(collected.items())}

    def normalized_progress(
        self,
        time_s: ArrayLike,
        *,
        label: str,
    ) -> BehaviorCovariate:
        """Map samples inside non-overlapping bouts to progress in ``[0, 1]``."""

        time = _readonly_float(time_s, name="time_s")
        _validate_time(time)
        selected = [interval for interval in self.intervals if interval.label == label]
        values = np.full(len(time), np.nan, dtype=float)
        valid = np.zeros(len(time), dtype=bool)
        for interval in selected:
            inside = (time >= interval.start_s) & (time <= interval.stop_s)
            if np.any(valid & inside):
                raise ValueError(f"overlapping {label!r} intervals make progress ambiguous")
            values[inside] = (time[inside] - interval.start_s) / (
                interval.stop_s - interval.start_s
            )
            valid[inside] = True
        return BehaviorCovariate(
            subject=self.subject,
            session=self.session,
            name=f"{label}_progress",
            time_s=time,
            values=values,
            valid=valid,
            unit="proportion",
            source=self.source,
            clock_id=self.clock_id,
            source_version=self.source_version,
            source_artifact=self.source_artifact,
            clock_synchronization_ids=self.clock_synchronization_ids,
        )

    def interval_encoding_inputs(
        self,
        *,
        edge: Literal["onset", "offset"] = "onset",
    ) -> IntervalEncodingInputs:
        """Return aligned interval edges, duration values and physical bounds."""

        if edge not in {"onset", "offset"}:
            raise ValueError("interval encoding edge must be 'onset' or 'offset'")
        labels = sorted({interval.label for interval in self.intervals})
        events: dict[str, tuple[float, ...]] = {}
        event_values: dict[str, Mapping[str, tuple[float, ...]]] = {}
        intervals: dict[str, tuple[tuple[float, float], ...]] = {}
        for label in labels:
            selected = [item for item in self.intervals if item.label == label]
            by_edge = sorted(
                selected,
                key=(
                    (lambda item: (item.start_s, item.stop_s))
                    if edge == "onset"
                    else (lambda item: (item.stop_s, item.start_s))
                ),
            )
            events[label] = tuple(
                item.start_s if edge == "onset" else item.stop_s for item in by_edge
            )
            event_values[label] = {
                "duration_s": tuple(item.stop_s - item.start_s for item in by_edge)
            }
            intervals[label] = tuple(
                (item.start_s, item.stop_s)
                for item in sorted(
                    selected,
                    key=lambda item: (item.start_s, item.stop_s),
                )
            )
        return IntervalEncodingInputs(events, event_values, intervals, edge)


def annotations_from_moseq(
    syllable: Sequence[int],
    *,
    subject: str,
    session: str,
    fps: float,
    labels: Mapping[int, str] | None = None,
    clock_id: str = "video",
    source_version: str | None = None,
    source_artifact: str | None = None,
) -> BehaviorAnnotations:
    """Run-length encode a Keypoint-MoSeq syllable sequence as bouts."""

    states = np.asarray(syllable)
    if states.ndim != 1 or len(states) == 0:
        raise ValueError("syllable must be a non-empty one-dimensional sequence")
    if states.dtype.kind not in "iu":
        raise TypeError("syllable must contain integer state labels")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    changes = np.flatnonzero(np.diff(states) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(states)]))
    intervals = tuple(
        BehaviorInterval(
            label=(labels or {}).get(int(states[start]), f"syllable_{int(states[start])}"),
            start_s=float(start / fps),
            stop_s=float(stop / fps),
        )
        for start, stop in zip(starts, stops, strict=True)
    )
    return BehaviorAnnotations(
        subject=subject,
        session=session,
        point_events={},
        intervals=intervals,
        source="keypoint-moseq",
        clock_id=clock_id,
        source_version=source_version,
        source_artifact=source_artifact,
    )


def annotations_from_moseq_results_h5(
    path: str | Path,
    *,
    recording: str,
    subject: str,
    session: str,
    fps: float,
    labels: Mapping[int, str] | None = None,
    clock_id: str = "video",
    source_version: str | None = None,
) -> BehaviorAnnotations:
    """Read a recording's syllable sequence from Keypoint-MoSeq results HDF5."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Keypoint-MoSeq file input requires the 'behavior' optional dependency"
        ) from error
    source = Path(path)
    with h5py.File(source, "r") as file:
        dataset = f"{recording}/syllable"
        if dataset not in file:
            raise ValueError(f"Keypoint-MoSeq results have no syllable dataset {dataset!r}")
        syllable = file[dataset][:]
    return annotations_from_moseq(
        syllable,
        subject=subject,
        session=session,
        fps=fps,
        labels=labels,
        clock_id=clock_id,
        source_version=source_version,
        source_artifact=str(source),
    )


def annotations_from_boris(
    columns: Mapping[str, Sequence[Any]],
    *,
    subject: str,
    session: str,
    behavior_column: str,
    type_column: str,
    start_column: str,
    stop_column: str,
    clock_id: str = "video",
    source_version: str | None = None,
    source_artifact: str | None = None,
) -> BehaviorAnnotations:
    """Convert an aggregated BORIS export after explicit column selection."""

    required = (behavior_column, type_column, start_column, stop_column)
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"BORIS export is missing columns: {missing}")
    lengths = {len(columns[name]) for name in required}
    if len(lengths) != 1:
        raise ValueError("selected BORIS columns must have equal length")
    points: dict[str, list[float]] = {}
    intervals: list[BehaviorInterval] = []
    for index in range(next(iter(lengths))):
        label = str(columns[behavior_column][index]).strip()
        kind = str(columns[type_column][index]).strip().upper()
        start = float(columns[start_column][index])
        if kind == "POINT":
            points.setdefault(label, []).append(start)
        elif kind == "STATE":
            stop = float(columns[stop_column][index])
            intervals.append(BehaviorInterval(label, start, stop))
        else:
            raise ValueError(f"BORIS row {index} has unsupported behavior type {kind!r}")
    return BehaviorAnnotations(
        subject=subject,
        session=session,
        point_events={label: tuple(times) for label, times in points.items()},
        intervals=tuple(intervals),
        source="boris",
        clock_id=clock_id,
        source_version=source_version,
        source_artifact=source_artifact,
    )


def annotations_from_boris_aggregated_file(
    path: str | Path,
    *,
    subject: str,
    session: str,
    source_subject: str | None = None,
    source_observation: str | None = None,
    clock_id: str = "video",
    source_version: str | None = None,
) -> BehaviorAnnotations:
    """Read a BORIS aggregated-event CSV or TSV without collapsing event type.

    BORIS aggregated exports already contain one row per complete POINT or STATE
    event. Selection of an observation and focal subject is explicit whenever a
    file contains more than one of either.
    """

    source = Path(path)
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))
    if not rows:
        raise ValueError("BORIS aggregated export is empty")
    header = [value.strip() for value in rows[0]]
    required = {
        "Observation id",
        "Subject",
        "Behavior",
        "Behavior type",
        "Start (s)",
        "Stop (s)",
    }
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"BORIS aggregated export is missing columns: {missing}")
    records = [
        dict(zip(header, row, strict=False))
        for row in rows[1:]
        if any(value.strip() for value in row)
    ]

    observations = sorted({record["Observation id"].strip() for record in records})
    observations = [value for value in observations if value]
    selected_observation = source_observation
    if selected_observation is None:
        if len(observations) != 1:
            raise ValueError(
                "BORIS aggregated export contains multiple observations; declare source_observation"
            )
        selected_observation = observations[0]
    if selected_observation not in observations:
        raise ValueError(f"BORIS source_observation {selected_observation!r} was not found")
    observation_records = [
        record for record in records if record["Observation id"].strip() == selected_observation
    ]

    subjects = sorted({record["Subject"].strip() for record in observation_records})
    subjects = [value for value in subjects if value]
    selected_subject = source_subject
    if selected_subject is None:
        if len(subjects) != 1:
            raise ValueError(
                "BORIS aggregated export contains multiple subjects; declare source_subject"
            )
        selected_subject = subjects[0]
    if selected_subject not in subjects:
        raise ValueError(f"BORIS source_subject {selected_subject!r} was not found")
    selected = [
        record for record in observation_records if record["Subject"].strip() == selected_subject
    ]
    return annotations_from_boris(
        {
            "behavior": [record["Behavior"] for record in selected],
            "type": [record["Behavior type"] for record in selected],
            "start": [record["Start (s)"] for record in selected],
            "stop": [record["Stop (s)"] for record in selected],
        },
        subject=subject,
        session=session,
        behavior_column="behavior",
        type_column="type",
        start_column="start",
        stop_column="stop",
        clock_id=clock_id,
        source_version=source_version,
        source_artifact=str(source),
    )


def annotations_from_boris_tabular_file(
    path: str | Path,
    *,
    subject: str,
    session: str,
    source_subject: str | None = None,
    clock_id: str = "video",
    source_version: str | None = None,
) -> BehaviorAnnotations:
    """Read BORIS tabular CSV, pairing START/STOP state-event rows."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if {"Time", "Subject", "Behavior", "Status"} <= set(row)
        ),
        None,
    )
    if header_index is None:
        raise ValueError("BORIS tabular CSV has no Time/Subject/Behavior/Status header")
    header = rows[header_index]
    records = [
        dict(zip(header, row, strict=False))
        for row in rows[header_index + 1 :]
        if any(value.strip() for value in row)
    ]
    observed_subjects = sorted({record.get("Subject", "").strip() for record in records})
    observed_subjects = [value for value in observed_subjects if value]
    selected_subject = source_subject
    if selected_subject is None:
        if len(observed_subjects) != 1:
            raise ValueError("BORIS tabular CSV contains multiple subjects; declare source_subject")
        selected_subject = observed_subjects[0]
    if selected_subject not in observed_subjects:
        raise ValueError(f"BORIS source_subject {selected_subject!r} was not found")

    points: dict[str, list[float]] = {}
    intervals: list[BehaviorInterval] = []
    open_states: dict[str, float] = {}
    for record in records:
        if record.get("Subject", "").strip() != selected_subject:
            continue
        label = record.get("Behavior", "").strip()
        status = record.get("Status", "").strip().upper()
        time = float(record.get("Time", "nan"))
        if not label or not np.isfinite(time):
            raise ValueError("BORIS event rows require finite time and behavior")
        if status == "START":
            if label in open_states:
                raise ValueError(f"BORIS state {label!r} has consecutive START rows")
            open_states[label] = time
        elif status == "STOP":
            try:
                start = open_states.pop(label)
            except KeyError:
                raise ValueError(f"BORIS state {label!r} has STOP without START") from None
            intervals.append(BehaviorInterval(label, start, time))
        elif status in {"", "POINT"}:
            points.setdefault(label, []).append(time)
        else:
            raise ValueError(f"unsupported BORIS tabular status {status!r}")
    if open_states:
        raise ValueError(f"BORIS states have START without STOP: {sorted(open_states)}")
    return BehaviorAnnotations(
        subject=subject,
        session=session,
        point_events={label: tuple(times) for label, times in points.items()},
        intervals=tuple(intervals),
        source="boris",
        clock_id=clock_id,
        source_version=source_version,
        source_artifact=str(source),
    )
