"""Deterministic, auditable policies for externally discovered behaviour intervals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, TypeAlias

import numpy as np

from behavio.observed.ethograms import BehaviorAnnotations, BehaviorInterval


def _nonempty(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _labels(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(_nonempty(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class FilterIntervals:
    """Keep intervals that satisfy declared label, duration, and confidence rules."""

    operation_id: str
    include_labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = ()
    minimum_duration_s: float | None = None
    maximum_duration_s: float | None = None
    minimum_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _nonempty(self.operation_id, "operation_id"))
        object.__setattr__(self, "include_labels", _labels(self.include_labels, "include_labels"))
        object.__setattr__(self, "exclude_labels", _labels(self.exclude_labels, "exclude_labels"))
        if set(self.include_labels) & set(self.exclude_labels):
            raise ValueError("include_labels and exclude_labels must not overlap")
        for value, name in (
            (self.minimum_duration_s, "minimum_duration_s"),
            (self.maximum_duration_s, "maximum_duration_s"),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.minimum_duration_s is not None
            and self.maximum_duration_s is not None
            and self.minimum_duration_s > self.maximum_duration_s
        ):
            raise ValueError("minimum_duration_s must not exceed maximum_duration_s")
        if self.minimum_confidence is not None and not (
            np.isfinite(self.minimum_confidence) and 0 <= self.minimum_confidence <= 1
        ):
            raise ValueError("minimum_confidence must lie between zero and one")


@dataclass(frozen=True)
class MergeIntervals:
    """Merge same-label intervals separated by at most ``maximum_gap_s``."""

    operation_id: str
    maximum_gap_s: float
    labels: tuple[str, ...] = ()
    confidence: Literal["minimum", "maximum", "mean", "drop"] = "minimum"

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _nonempty(self.operation_id, "operation_id"))
        object.__setattr__(self, "labels", _labels(self.labels, "labels"))
        if not np.isfinite(self.maximum_gap_s) or self.maximum_gap_s < 0:
            raise ValueError("maximum_gap_s must be finite and non-negative")
        if self.confidence not in {"minimum", "maximum", "mean", "drop"}:
            raise ValueError("unsupported merge confidence rule")


@dataclass(frozen=True)
class SplitIntervals:
    """Split intervals at declared timestamps and/or a maximum segment duration."""

    operation_id: str
    labels: tuple[str, ...] = ()
    maximum_duration_s: float | None = None
    cut_points_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _nonempty(self.operation_id, "operation_id"))
        object.__setattr__(self, "labels", _labels(self.labels, "labels"))
        if self.maximum_duration_s is not None and (
            not np.isfinite(self.maximum_duration_s) or self.maximum_duration_s <= 0
        ):
            raise ValueError("maximum_duration_s must be finite and positive")
        cuts = tuple(float(value) for value in self.cut_points_s)
        if any(not np.isfinite(value) for value in cuts):
            raise ValueError("cut_points_s must be finite")
        object.__setattr__(self, "cut_points_s", tuple(sorted(set(cuts))))
        if self.maximum_duration_s is None and not cuts:
            raise ValueError("split operation requires maximum_duration_s or cut_points_s")


@dataclass(frozen=True)
class ContextualizeIntervals:
    """Relabel target intervals from overlap with a named external context source."""

    operation_id: str
    context_source: str
    labels: tuple[str, ...] = ()
    context_labels: tuple[str, ...] = ()
    minimum_overlap_fraction: float = 0.0
    multiple_matches: Literal["duplicate", "first", "reject"] = "duplicate"
    keep_unmatched: bool = True
    label_template: str = "{label}@{context}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _nonempty(self.operation_id, "operation_id"))
        object.__setattr__(self, "context_source", _nonempty(self.context_source, "context_source"))
        object.__setattr__(self, "labels", _labels(self.labels, "labels"))
        object.__setattr__(self, "context_labels", _labels(self.context_labels, "context_labels"))
        if not np.isfinite(self.minimum_overlap_fraction) or not (
            0 <= self.minimum_overlap_fraction <= 1
        ):
            raise ValueError("minimum_overlap_fraction must lie between zero and one")
        if self.multiple_matches not in {"duplicate", "first", "reject"}:
            raise ValueError("unsupported multiple_matches rule")
        try:
            rendered = self.label_template.format(
                label="target", context="context", source="source"
            )
        except (KeyError, ValueError) as error:
            raise ValueError("label_template has unsupported placeholders") from error
        if not rendered.strip():
            raise ValueError("label_template must render a non-empty label")


@dataclass(frozen=True)
class ResolveIntervalOverlaps:
    """Reject overlaps or retain higher-priority interval portions explicitly."""

    operation_id: str
    scope: Literal["same_label", "all"] = "same_label"
    strategy: Literal["reject", "priority"] = "reject"
    label_priority: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _nonempty(self.operation_id, "operation_id"))
        object.__setattr__(self, "label_priority", _labels(self.label_priority, "label_priority"))
        if self.scope not in {"same_label", "all"}:
            raise ValueError("unsupported overlap scope")
        if self.strategy not in {"reject", "priority"}:
            raise ValueError("unsupported overlap strategy")
        if self.strategy == "reject" and self.label_priority:
            raise ValueError("label_priority is only valid for priority resolution")


IntervalOperation: TypeAlias = (
    FilterIntervals
    | MergeIntervals
    | SplitIntervals
    | ContextualizeIntervals
    | ResolveIntervalOverlaps
)


@dataclass(frozen=True)
class IntervalPolicy:
    """An ordered, reproducible sequence of interval transformations."""

    operations: tuple[IntervalOperation, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("unsupported interval-policy schema_version")
        identifiers = tuple(operation.operation_id for operation in self.operations)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("interval-policy operation IDs must be unique")


@dataclass(frozen=True)
class IntervalSnapshot:
    """One immutable interval identity recorded in the transformation ledger."""

    interval_id: str
    label: str
    start_s: float
    stop_s: float
    confidence: float | None
    source_interval_ids: tuple[str, ...]


@dataclass(frozen=True)
class IntervalPolicyLedgerEntry:
    """Inputs and outputs for one auditable policy decision."""

    operation_id: str
    operation_type: str
    action: Literal["kept", "removed", "merged", "split", "relabelled", "trimmed"]
    reason: str
    inputs: tuple[IntervalSnapshot, ...]
    outputs: tuple[IntervalSnapshot, ...]
    context: tuple[IntervalSnapshot, ...] = ()


@dataclass(frozen=True)
class IntervalPolicyContext:
    """A named external annotation source bound into policy evidence."""

    name: str
    annotations: BehaviorAnnotations

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "context name"))


@dataclass(frozen=True)
class IntervalPolicyResult:
    """Transformed annotations and complete provenance for every policy operation."""

    annotations: BehaviorAnnotations
    policy: IntervalPolicy
    ledger: tuple[IntervalPolicyLedgerEntry, ...]
    contexts: tuple[IntervalPolicyContext, ...]
    input_interval_count: int
    output_interval_count: int
    evidence_fingerprint: str
    schema_version: str = "1"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evidence artifact."""

        return _result_payload(self, include_fingerprint=True)


@dataclass(frozen=True)
class _TrackedInterval:
    interval_id: str
    interval: BehaviorInterval
    source_interval_ids: tuple[str, ...]


def apply_interval_policy(
    annotations: BehaviorAnnotations,
    policy: IntervalPolicy,
    *,
    context_sources: Mapping[str, BehaviorAnnotations] | None = None,
) -> IntervalPolicyResult:
    """Apply ordered interval rules without changing point events or clock identity."""

    contexts = dict(context_sources or {})
    referenced_contexts: dict[str, BehaviorAnnotations] = {}
    tracked = [
        _TrackedInterval(f"input:{index:06d}", interval, (f"input:{index:06d}",))
        for index, interval in enumerate(annotations.intervals)
    ]
    ledger: list[IntervalPolicyLedgerEntry] = []
    for operation in policy.operations:
        if isinstance(operation, FilterIntervals):
            tracked, entries = _apply_filter(tracked, operation)
        elif isinstance(operation, MergeIntervals):
            tracked, entries = _apply_merge(tracked, operation)
        elif isinstance(operation, SplitIntervals):
            tracked, entries = _apply_split(tracked, operation)
        elif isinstance(operation, ContextualizeIntervals):
            context = contexts.get(operation.context_source)
            if context is None:
                raise ValueError(
                    f"missing context source {operation.context_source!r} for "
                    f"operation {operation.operation_id!r}"
                )
            _validate_context_compatibility(annotations, context, operation.context_source)
            referenced_contexts[operation.context_source] = context
            tracked, entries = _apply_context(tracked, operation, context)
        else:
            tracked, entries = _apply_overlaps(tracked, operation)
        ledger.extend(entries)
        tracked.sort(key=_tracked_sort_key)

    output = BehaviorAnnotations(
        subject=annotations.subject,
        session=annotations.session,
        point_events=annotations.point_events,
        intervals=tuple(item.interval for item in tracked),
        source=annotations.source,
        clock_id=annotations.clock_id,
        source_version=annotations.source_version,
        source_artifact=annotations.source_artifact,
        clock_synchronization_ids=annotations.clock_synchronization_ids,
    )
    provisional = IntervalPolicyResult(
        output,
        policy,
        tuple(ledger),
        tuple(
            IntervalPolicyContext(name, context)
            for name, context in sorted(referenced_contexts.items())
        ),
        len(annotations.intervals),
        len(output.intervals),
        "",
    )
    payload = _result_payload(provisional, include_fingerprint=False)
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return IntervalPolicyResult(
        output,
        policy,
        tuple(ledger),
        tuple(
            IntervalPolicyContext(name, context)
            for name, context in sorted(referenced_contexts.items())
        ),
        len(annotations.intervals),
        len(output.intervals),
        fingerprint,
    )


def _apply_filter(
    tracked: list[_TrackedInterval], operation: FilterIntervals
) -> tuple[list[_TrackedInterval], list[IntervalPolicyLedgerEntry]]:
    kept: list[_TrackedInterval] = []
    ledger: list[IntervalPolicyLedgerEntry] = []
    for item in tracked:
        interval = item.interval
        duration = interval.stop_s - interval.start_s
        reason: str | None = None
        if operation.include_labels and interval.label not in operation.include_labels:
            reason = "label not included"
        elif interval.label in operation.exclude_labels:
            reason = "label excluded"
        elif operation.minimum_duration_s is not None and duration < operation.minimum_duration_s:
            reason = "duration below minimum"
        elif operation.maximum_duration_s is not None and duration > operation.maximum_duration_s:
            reason = "duration above maximum"
        elif operation.minimum_confidence is not None and (
            interval.confidence is None or interval.confidence < operation.minimum_confidence
        ):
            reason = "confidence missing or below minimum"
        if reason is None:
            kept.append(item)
            ledger.append(_entry(operation, "kept", "all filter rules passed", (item,), (item,)))
        else:
            ledger.append(_entry(operation, "removed", reason, (item,), ()))
    return kept, ledger


def _apply_merge(
    tracked: list[_TrackedInterval], operation: MergeIntervals
) -> tuple[list[_TrackedInterval], list[IntervalPolicyLedgerEntry]]:
    selected_labels = (
        set(operation.labels) if operation.labels else {item.interval.label for item in tracked}
    )
    output = [item for item in tracked if item.interval.label not in selected_labels]
    ledger = [
        _entry(operation, "kept", "label outside merge scope", (item,), (item,)) for item in output
    ]
    output_index = 0
    for label in sorted(selected_labels):
        items = sorted(
            (item for item in tracked if item.interval.label == label),
            key=_tracked_sort_key,
        )
        groups: list[list[_TrackedInterval]] = []
        for item in items:
            if (
                not groups
                or item.interval.start_s - max(member.interval.stop_s for member in groups[-1])
                > operation.maximum_gap_s
            ):
                groups.append([item])
            else:
                groups[-1].append(item)
        for group in groups:
            if len(group) == 1:
                item = group[0]
                output.append(item)
                ledger.append(_entry(operation, "kept", "no eligible neighbour", (item,), (item,)))
                continue
            confidence = _merge_confidence(
                tuple(item.interval.confidence for item in group), operation.confidence
            )
            merged = _TrackedInterval(
                f"{operation.operation_id}:{output_index:06d}",
                BehaviorInterval(
                    label,
                    min(item.interval.start_s for item in group),
                    max(item.interval.stop_s for item in group),
                    confidence,
                ),
                tuple(
                    dict.fromkeys(source for item in group for source in item.source_interval_ids)
                ),
            )
            output_index += 1
            output.append(merged)
            ledger.append(
                _entry(
                    operation,
                    "merged",
                    f"same label within {operation.maximum_gap_s:g} s",
                    tuple(group),
                    (merged,),
                )
            )
    return output, ledger


def _apply_split(
    tracked: list[_TrackedInterval], operation: SplitIntervals
) -> tuple[list[_TrackedInterval], list[IntervalPolicyLedgerEntry]]:
    output: list[_TrackedInterval] = []
    ledger: list[IntervalPolicyLedgerEntry] = []
    output_index = 0
    for item in tracked:
        interval = item.interval
        if operation.labels and interval.label not in operation.labels:
            output.append(item)
            ledger.append(_entry(operation, "kept", "label outside split scope", (item,), (item,)))
            continue
        cuts = [
            value for value in operation.cut_points_s if interval.start_s < value < interval.stop_s
        ]
        if operation.maximum_duration_s is not None:
            boundary = interval.start_s + operation.maximum_duration_s
            while boundary < interval.stop_s:
                cuts.append(boundary)
                boundary += operation.maximum_duration_s
        cuts = sorted(set(cuts))
        if not cuts:
            output.append(item)
            ledger.append(
                _entry(
                    operation,
                    "kept",
                    "no split boundary inside interval",
                    (item,),
                    (item,),
                )
            )
            continue
        boundaries = (interval.start_s, *cuts, interval.stop_s)
        pieces = tuple(
            _TrackedInterval(
                f"{operation.operation_id}:{output_index + index:06d}",
                BehaviorInterval(
                    interval.label,
                    start,
                    stop,
                    interval.confidence,
                ),
                item.source_interval_ids,
            )
            for index, (start, stop) in enumerate(pairwise(boundaries))
        )
        output_index += len(pieces)
        output.extend(pieces)
        ledger.append(_entry(operation, "split", "declared split boundary", (item,), pieces))
    return output, ledger


def _apply_context(
    tracked: list[_TrackedInterval],
    operation: ContextualizeIntervals,
    context: BehaviorAnnotations,
) -> tuple[list[_TrackedInterval], list[IntervalPolicyLedgerEntry]]:
    output: list[_TrackedInterval] = []
    ledger: list[IntervalPolicyLedgerEntry] = []
    output_index = 0
    allowed_contexts = set(operation.context_labels)
    priority = {label: index for index, label in enumerate(operation.context_labels)}
    context_intervals = [
        (index, interval)
        for index, interval in enumerate(context.intervals)
        if not allowed_contexts or interval.label in allowed_contexts
    ]
    for item in tracked:
        interval = item.interval
        if operation.labels and interval.label not in operation.labels:
            output.append(item)
            ledger.append(
                _entry(operation, "kept", "label outside context scope", (item,), (item,))
            )
            continue
        matches = []
        duration = interval.stop_s - interval.start_s
        for context_index, candidate in context_intervals:
            overlap = min(interval.stop_s, candidate.stop_s) - max(
                interval.start_s, candidate.start_s
            )
            fraction = max(overlap, 0.0) / duration
            if overlap > 0 and fraction >= operation.minimum_overlap_fraction:
                matches.append((context_index, candidate))
        unique_labels = sorted(
            {candidate.label for _, candidate in matches},
            key=lambda label: (priority.get(label, len(priority)), label),
        )
        if len(unique_labels) > 1 and operation.multiple_matches == "reject":
            raise ValueError(
                f"interval {item.interval_id!r} matches multiple contexts: "
                f"{', '.join(unique_labels)}"
            )
        if operation.multiple_matches == "first":
            unique_labels = unique_labels[:1]
        if not unique_labels:
            if operation.keep_unmatched:
                output.append(item)
                ledger.append(
                    _entry(
                        operation,
                        "kept",
                        "no qualifying context overlap",
                        (item,),
                        (item,),
                    )
                )
            else:
                ledger.append(
                    _entry(
                        operation,
                        "removed",
                        "no qualifying context overlap",
                        (item,),
                        (),
                    )
                )
            continue
        relabelled = tuple(
            _TrackedInterval(
                f"{operation.operation_id}:{output_index + index:06d}",
                BehaviorInterval(
                    operation.label_template.format(
                        label=interval.label,
                        context=context_label,
                        source=operation.context_source,
                    ),
                    interval.start_s,
                    interval.stop_s,
                    interval.confidence,
                ),
                item.source_interval_ids,
            )
            for index, context_label in enumerate(unique_labels)
        )
        output_index += len(relabelled)
        output.extend(relabelled)
        ledger.append(
            _entry(
                operation,
                "relabelled",
                f"overlap with context source {operation.context_source!r}",
                (item,),
                relabelled,
                context=tuple(
                    _TrackedInterval(
                        f"context:{operation.context_source}:{context_index:06d}",
                        candidate,
                        (f"context:{operation.context_source}:{context_index:06d}",),
                    )
                    for context_index, candidate in matches
                ),
            )
        )
    return output, ledger


def _apply_overlaps(
    tracked: list[_TrackedInterval], operation: ResolveIntervalOverlaps
) -> tuple[list[_TrackedInterval], list[IntervalPolicyLedgerEntry]]:
    if operation.strategy == "reject":
        ordered = sorted(tracked, key=_tracked_sort_key)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right.interval.start_s >= left.interval.stop_s:
                    break
                if _in_overlap_scope(left, right, operation.scope):
                    raise ValueError(
                        f"overlap policy {operation.operation_id!r} rejected "
                        f"{left.interval_id!r} and {right.interval_id!r}"
                    )
        return ordered, [
            _entry(operation, "kept", "no overlap in declared scope", (item,), (item,))
            for item in ordered
        ]

    rank = {label: index for index, label in enumerate(operation.label_priority)}
    ordered = sorted(
        tracked,
        key=lambda item: (
            rank.get(item.interval.label, len(rank)),
            item.interval.start_s,
            item.interval.stop_s,
            item.interval.label,
            item.interval_id,
        ),
    )
    accepted: list[_TrackedInterval] = []
    ledger: list[IntervalPolicyLedgerEntry] = []
    output_index = 0
    for item in ordered:
        pieces = [(item.interval.start_s, item.interval.stop_s)]
        for blocker in accepted:
            if not _in_overlap_scope(item, blocker, operation.scope):
                continue
            pieces = [
                remainder
                for piece in pieces
                for remainder in _subtract_interval(piece, blocker.interval)
            ]
            if not pieces:
                break
        if pieces == [(item.interval.start_s, item.interval.stop_s)]:
            accepted.append(item)
            ledger.append(_entry(operation, "kept", "no higher-priority overlap", (item,), (item,)))
            continue
        replacements = tuple(
            _TrackedInterval(
                f"{operation.operation_id}:{output_index + index:06d}",
                BehaviorInterval(
                    item.interval.label,
                    start,
                    stop,
                    item.interval.confidence,
                ),
                item.source_interval_ids,
            )
            for index, (start, stop) in enumerate(pieces)
            if stop > start
        )
        output_index += len(replacements)
        accepted.extend(replacements)
        action: Literal["removed", "trimmed"] = "trimmed" if replacements else "removed"
        ledger.append(
            _entry(
                operation,
                action,
                "higher-priority overlap removed from interval",
                (item,),
                replacements,
            )
        )
    return accepted, ledger


def _subtract_interval(
    piece: tuple[float, float], blocker: BehaviorInterval
) -> list[tuple[float, float]]:
    start, stop = piece
    overlap_start = max(start, blocker.start_s)
    overlap_stop = min(stop, blocker.stop_s)
    if overlap_stop <= overlap_start:
        return [piece]
    result = []
    if start < overlap_start:
        result.append((start, overlap_start))
    if overlap_stop < stop:
        result.append((overlap_stop, stop))
    return result


def _in_overlap_scope(
    left: _TrackedInterval,
    right: _TrackedInterval,
    scope: Literal["same_label", "all"],
) -> bool:
    return scope == "all" or left.interval.label == right.interval.label


def _merge_confidence(
    values: tuple[float | None, ...],
    rule: Literal["minimum", "maximum", "mean", "drop"],
) -> float | None:
    if rule == "drop" or any(value is None for value in values):
        return None
    numeric = tuple(float(value) for value in values if value is not None)
    if rule == "minimum":
        return min(numeric)
    if rule == "maximum":
        return max(numeric)
    return float(np.mean(numeric))


def _validate_context_compatibility(
    target: BehaviorAnnotations,
    context: BehaviorAnnotations,
    context_source: str,
) -> None:
    if (target.subject, target.session) != (context.subject, context.session):
        raise ValueError(f"context source {context_source!r} has a different subject/session")
    if target.clock_id != context.clock_id:
        raise ValueError(f"context source {context_source!r} uses a different clock_id")


def _tracked_sort_key(item: _TrackedInterval) -> tuple[float, float, str, str]:
    return (
        item.interval.start_s,
        item.interval.stop_s,
        item.interval.label,
        item.interval_id,
    )


def _snapshot(item: _TrackedInterval) -> IntervalSnapshot:
    interval = item.interval
    return IntervalSnapshot(
        item.interval_id,
        interval.label,
        interval.start_s,
        interval.stop_s,
        interval.confidence,
        item.source_interval_ids,
    )


def _entry(
    operation: IntervalOperation,
    action: Literal["kept", "removed", "merged", "split", "relabelled", "trimmed"],
    reason: str,
    inputs: tuple[_TrackedInterval, ...],
    outputs: tuple[_TrackedInterval, ...],
    *,
    context: tuple[_TrackedInterval, ...] = (),
) -> IntervalPolicyLedgerEntry:
    return IntervalPolicyLedgerEntry(
        operation.operation_id,
        type(operation).__name__,
        action,
        reason,
        tuple(_snapshot(item) for item in inputs),
        tuple(_snapshot(item) for item in outputs),
        tuple(_snapshot(item) for item in context),
    )


def _operation_payload(operation: IntervalOperation) -> dict[str, object]:
    payload = dict(vars(operation))
    payload["type"] = type(operation).__name__
    return payload


def _snapshot_payload(snapshot: IntervalSnapshot) -> dict[str, object]:
    return {
        "interval_id": snapshot.interval_id,
        "label": snapshot.label,
        "start_s": snapshot.start_s,
        "stop_s": snapshot.stop_s,
        "confidence": snapshot.confidence,
        "source_interval_ids": list(snapshot.source_interval_ids),
    }


def _result_payload(
    result: IntervalPolicyResult, *, include_fingerprint: bool
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": result.schema_version,
        "annotations": _annotations_payload(result.annotations),
        "policy": {
            "schema_version": result.policy.schema_version,
            "operations": [_operation_payload(operation) for operation in result.policy.operations],
        },
        "ledger": [
            {
                "operation_id": entry.operation_id,
                "operation_type": entry.operation_type,
                "action": entry.action,
                "reason": entry.reason,
                "inputs": [_snapshot_payload(value) for value in entry.inputs],
                "outputs": [_snapshot_payload(value) for value in entry.outputs],
                "context": [_snapshot_payload(value) for value in entry.context],
            }
            for entry in result.ledger
        ],
        "contexts": [
            {
                "name": context.name,
                "annotations": _annotations_payload(context.annotations),
            }
            for context in result.contexts
        ],
        "input_interval_count": result.input_interval_count,
        "output_interval_count": result.output_interval_count,
    }
    if include_fingerprint:
        payload["evidence_fingerprint"] = result.evidence_fingerprint
    return payload


def _annotations_payload(annotations: BehaviorAnnotations) -> dict[str, object]:
    return {
        "subject": annotations.subject,
        "session": annotations.session,
        "point_events": {label: list(times) for label, times in annotations.point_events.items()},
        "intervals": [
            {
                "label": interval.label,
                "start_s": interval.start_s,
                "stop_s": interval.stop_s,
                "confidence": interval.confidence,
            }
            for interval in annotations.intervals
        ],
        "source": annotations.source,
        "clock_id": annotations.clock_id,
        "source_version": annotations.source_version,
        "source_artifact": annotations.source_artifact,
        "clock_synchronization_ids": list(annotations.clock_synchronization_ids),
    }
