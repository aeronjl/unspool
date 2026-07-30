import json

import pytest

from behavio.observed.ethograms import BehaviorAnnotations, BehaviorInterval
from behavio.observed.interval_policy import (
    ContextualizeIntervals,
    FilterIntervals,
    IntervalPolicy,
    MergeIntervals,
    ResolveIntervalOverlaps,
    SplitIntervals,
    apply_interval_policy,
)


def _annotations(*intervals: BehaviorInterval) -> BehaviorAnnotations:
    return BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"cue": (0.5,)},
        intervals=intervals,
        source="keypoint-moseq",
        clock_id="photometry",
        source_version="0.6",
        source_artifact="results.h5",
        clock_synchronization_ids=("sync-1",),
    )


def test_ordered_filter_and_merge_have_explicitly_different_results() -> None:
    annotations = _annotations(
        BehaviorInterval("groom", 0.0, 0.4, 0.9),
        BehaviorInterval("groom", 0.5, 0.9, 0.8),
    )

    filter_then_merge = apply_interval_policy(
        annotations,
        IntervalPolicy(
            (
                FilterIntervals("minimum-duration", minimum_duration_s=0.5),
                MergeIntervals("short-gaps", maximum_gap_s=0.2),
            )
        ),
    )
    merge_then_filter = apply_interval_policy(
        annotations,
        IntervalPolicy(
            (
                MergeIntervals("short-gaps", maximum_gap_s=0.2),
                FilterIntervals("minimum-duration", minimum_duration_s=0.5),
            )
        ),
    )

    assert filter_then_merge.annotations.intervals == ()
    assert merge_then_filter.annotations.intervals == (BehaviorInterval("groom", 0.0, 0.9, 0.8),)
    assert [entry.action for entry in merge_then_filter.ledger] == [
        "merged",
        "kept",
    ]
    assert merge_then_filter.ledger[0].outputs[0].source_interval_ids == (
        "input:000000",
        "input:000001",
    )


def test_filter_requires_observed_confidence_when_a_threshold_is_declared() -> None:
    result = apply_interval_policy(
        _annotations(
            BehaviorInterval("rear", 0.0, 1.0),
            BehaviorInterval("rear", 2.0, 3.0, 0.75),
        ),
        IntervalPolicy((FilterIntervals("confidence", minimum_confidence=0.7),)),
    )

    assert result.annotations.intervals == (BehaviorInterval("rear", 2.0, 3.0, 0.75),)
    assert result.ledger[0].action == "removed"
    assert "missing" in result.ledger[0].reason


def test_split_uses_physical_cut_points_and_maximum_duration() -> None:
    result = apply_interval_policy(
        _annotations(BehaviorInterval("run", 0.0, 5.0, 0.9)),
        IntervalPolicy(
            (
                SplitIntervals(
                    "epochs",
                    maximum_duration_s=2.0,
                    cut_points_s=(1.0, 4.5),
                ),
            )
        ),
    )

    assert [(interval.start_s, interval.stop_s) for interval in result.annotations.intervals] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (4.0, 4.5),
        (4.5, 5.0),
    ]
    assert result.ledger[0].action == "split"
    assert all(
        output.source_interval_ids == ("input:000000",) for output in result.ledger[0].outputs
    )


def test_contextualize_duplicates_named_contexts_without_discovering_them() -> None:
    target = _annotations(BehaviorInterval("approach", 1.0, 3.0, 0.8))
    context = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={},
        intervals=(
            BehaviorInterval("laser-off", 0.0, 2.0),
            BehaviorInterval("laser-on", 2.0, 4.0),
        ),
        source="boris",
        clock_id="photometry",
    )
    result = apply_interval_policy(
        target,
        IntervalPolicy(
            (
                ContextualizeIntervals(
                    "laser-context",
                    context_source="stimulation-state",
                    minimum_overlap_fraction=0.5,
                ),
            )
        ),
        context_sources={"stimulation-state": context},
    )

    assert [interval.label for interval in result.annotations.intervals] == [
        "approach@laser-off",
        "approach@laser-on",
    ]
    assert result.annotations.point_events == target.point_events
    assert result.ledger[0].action == "relabelled"
    assert [snapshot.label for snapshot in result.ledger[0].context] == [
        "laser-off",
        "laser-on",
    ]
    assert result.contexts[0].name == "stimulation-state"
    assert result.to_dict()["contexts"][0]["annotations"]["source"] == "boris"
    encoding = result.annotations.interval_encoding_inputs()
    assert encoding.event_values["approach@laser-on"]["duration_s"] == (2.0,)

    changed_context = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={},
        intervals=context.intervals,
        source="boris",
        clock_id="photometry",
        source_artifact="revised.csv",
    )
    changed = apply_interval_policy(
        target,
        result.policy,
        context_sources={"stimulation-state": changed_context},
    )
    assert changed.evidence_fingerprint != result.evidence_fingerprint


def test_contextualize_requires_matching_clock_and_session() -> None:
    target = _annotations(BehaviorInterval("approach", 1.0, 3.0))
    context = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={},
        intervals=(BehaviorInterval("light", 0.0, 4.0),),
        source="boris",
        clock_id="video",
    )
    with pytest.raises(ValueError, match="different clock_id"):
        apply_interval_policy(
            target,
            IntervalPolicy((ContextualizeIntervals("context", context_source="epochs"),)),
            context_sources={"epochs": context},
        )


def test_priority_overlap_resolution_trims_and_splits_lower_priority_bouts() -> None:
    result = apply_interval_policy(
        _annotations(
            BehaviorInterval("locomotion", 0.0, 10.0),
            BehaviorInterval("groom", 2.0, 4.0),
            BehaviorInterval("rear", 6.0, 8.0),
        ),
        IntervalPolicy(
            (
                ResolveIntervalOverlaps(
                    "exclusive-state",
                    scope="all",
                    strategy="priority",
                    label_priority=("groom", "rear", "locomotion"),
                ),
            )
        ),
    )

    assert [
        (interval.label, interval.start_s, interval.stop_s)
        for interval in result.annotations.intervals
    ] == [
        ("locomotion", 0.0, 2.0),
        ("groom", 2.0, 4.0),
        ("locomotion", 4.0, 6.0),
        ("rear", 6.0, 8.0),
        ("locomotion", 8.0, 10.0),
    ]
    assert [entry.action for entry in result.ledger] == [
        "kept",
        "kept",
        "trimmed",
    ]
    assert len(result.ledger[-1].outputs) == 3


def test_overlap_rejection_is_scoped_and_names_conflicting_ids() -> None:
    annotations = _annotations(
        BehaviorInterval("groom", 0.0, 2.0),
        BehaviorInterval("groom", 1.0, 3.0),
    )
    with pytest.raises(ValueError, match=r"input:000000.*input:000001"):
        apply_interval_policy(
            annotations,
            IntervalPolicy((ResolveIntervalOverlaps("no-overlap"),)),
        )


def test_evidence_artifact_is_stable_and_json_serializable() -> None:
    annotations = _annotations(BehaviorInterval("rear", 0.0, 1.0, 0.9))
    policy = IntervalPolicy((FilterIntervals("keep-rear", include_labels=("rear",)),))

    first = apply_interval_policy(annotations, policy)
    second = apply_interval_policy(annotations, policy)

    assert first.evidence_fingerprint == second.evidence_fingerprint
    assert len(first.evidence_fingerprint) == 64
    payload = first.to_dict()
    assert payload["evidence_fingerprint"] == first.evidence_fingerprint
    assert json.loads(json.dumps(payload))["output_interval_count"] == 1


def test_operation_specs_reject_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        FilterIntervals("filter", minimum_duration_s=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        MergeIntervals("merge", maximum_gap_s=-1.0)
    with pytest.raises(ValueError, match="requires"):
        SplitIntervals("split")
    with pytest.raises(ValueError, match="between zero and one"):
        ContextualizeIntervals("context", context_source="epochs", minimum_overlap_fraction=1.1)
