"""Reducing observed behaviour onto declared trials and into Study columns."""

from __future__ import annotations

import numpy as np
import pytest

from behavio.observed.covariates import BehaviorCovariate
from behavio.observed.device_clocks import (
    DeviceClockPulses,
    DeviceClockSyncSpec,
    fit_device_clock_sync,
)
from behavio.observed.ethograms import BehaviorAnnotations, BehaviorInterval
from behavio.observed.pose import PoseTrajectory
from behavio.observed.trialization import (
    EventCount,
    FirstOccurrenceLatency,
    FractionOfTimeInState,
    MaximumValue,
    MeanValue,
    MedianValue,
    MinimumValue,
    TrialCoverageStatus,
    TrializationError,
    TrialReduction,
    TrialTiming,
    TrialWindow,
    attach_trial_columns,
    reduce_annotations_to_trials,
    reduce_covariate_to_trials,
    trial_timing_from_events,
)
from behavio.trials import Study


def _covariate(
    *,
    time_s: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray | None = None,
    subject: object = "mouse-1",
    session: object = "day-1",
    clock_id: str = "video",
    name: str = "speed",
) -> BehaviorCovariate:
    return BehaviorCovariate(
        subject=subject,
        session=session,
        name=name,
        time_s=time_s,
        values=values,
        valid=np.ones(len(time_s), dtype=bool) if valid is None else valid,
        unit="cm/s",
        source="pose",
        clock_id=clock_id,
    )


def _timing(
    *,
    onsets: list[float],
    subject: object = "mouse-1",
    session: object = "day-1",
    clock_id: str = "video",
    offsets: list[float] | None = None,
) -> TrialTiming:
    return TrialTiming.from_arrays(
        subject=subject,
        session=session,
        onset_s=onsets,
        offset_s=offsets,
        clock_id=clock_id,
    )


def _study(*, subject: object = "mouse-1", session: object = "day-1", n_trials: int = 2) -> Study:
    return Study.from_columns(
        {
            "subject": [subject] * n_trials,
            "session": [session] * n_trials,
            "trial": list(range(n_trials)),
            "session_order": [0] * n_trials,
        }
    )


def test_covariate_mean_reduces_to_a_known_per_trial_value() -> None:
    """A hand-computable window mean must land in the study unchanged."""

    covariate = _covariate(
        time_s=np.arange(0.0, 4.01, 0.5),
        values=np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]),
    )
    timing = _timing(onsets=[0.0, 2.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=1.0)

    reduction = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=window,
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=1.0,
    )

    # trial 0 covers t=0.0, 0.5, 1.0 -> (1 + 1 + 2) / 3.
    # trial 1 covers t=2.0, 2.5, 3.0 -> (3 + 3 + 4) / 3.
    assert reduction.name == "speed_mean"
    assert reduction.unit == "cm/s"
    assert reduction.values == pytest.approx([4 / 3, 10 / 3])
    assert reduction.coverage == pytest.approx([1.0, 1.0])
    assert reduction.status == (TrialCoverageStatus.OK, TrialCoverageStatus.OK)
    assert reduction.valid_sample_count.tolist() == [3, 3]


def test_covariate_summary_reducers_agree_with_numpy_on_the_same_window() -> None:
    covariate = _covariate(
        time_s=np.array([0.0, 0.25, 0.5, 0.75]),
        values=np.array([4.0, 1.0, 9.0, 2.0]),
    )
    timing = _timing(onsets=[0.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=0.75)

    reduced = {
        reducer.name: reduce_covariate_to_trials(
            covariate,
            timing=timing,
            window=window,
            reducer=reducer,
            max_gap_s=0.3,
            minimum_coverage=0.0,
        ).values[0]
        for reducer in (MeanValue(), MedianValue(), MinimumValue(), MaximumValue())
    }

    assert reduced["mean"] == pytest.approx(4.0)
    assert reduced["median"] == pytest.approx(3.0)
    assert reduced["minimum"] == pytest.approx(1.0)
    assert reduced["maximum"] == pytest.approx(9.0)


def test_window_with_no_samples_is_reported_rather_than_summarised() -> None:
    """An empty window must not silently produce a number."""

    covariate = _covariate(
        time_s=np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2]),
        values=np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0]),
    )
    timing = _timing(onsets=[0.0, 2.0, 5.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=0.25)

    reduction = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=window,
        reducer=MeanValue(),
        max_gap_s=0.15,
        minimum_coverage=0.0,
    )

    assert reduction.status[1] is TrialCoverageStatus.NO_VALID_SAMPLES
    assert np.isnan(reduction.values[1])
    assert reduction.valid_sample_count[1] == 0
    assert reduction.coverage[1] == pytest.approx(0.0)
    assert reduction.n_reduced == 2


def test_window_past_the_end_of_the_recording_is_outside_the_observed_span() -> None:
    covariate = _covariate(
        time_s=np.array([0.0, 0.5, 1.0]),
        values=np.array([2.0, 2.0, 2.0]),
    )
    timing = _timing(onsets=[0.0, 3.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=1.0)

    reduction = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=window,
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=0.0,
    )

    assert reduction.status[1] is TrialCoverageStatus.OUTSIDE_OBSERVED_SPAN
    assert np.isnan(reduction.values[1])


def test_window_straddling_a_gap_surfaces_partial_coverage() -> None:
    """A window half of which is an unsampled gap must report half coverage."""

    covariate = _covariate(
        time_s=np.array([0.0, 0.5, 1.5, 2.0]),
        values=np.array([1.0, 1.0, 3.0, 3.0]),
    )
    timing = _timing(onsets=[0.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=2.0)

    reduction = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=window,
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=0.0,
    )

    # 0.0-0.5 and 1.5-2.0 span the window; the 1.0 s gap exceeds max_gap_s.
    assert reduction.coverage[0] == pytest.approx(0.5)
    assert reduction.status[0] is TrialCoverageStatus.PARTIAL_COVERAGE
    assert reduction.values[0] == pytest.approx(2.0)


def test_partial_validity_below_the_declared_minimum_withholds_the_value() -> None:
    valid = np.array([True, True, False, False, False, False, False, False, True])
    covariate = _covariate(
        time_s=np.arange(0.0, 4.01, 0.5),
        values=np.full(9, 2.0),
        valid=valid,
    )
    timing = _timing(onsets=[0.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=4.0)

    permissive = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=window,
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=0.0,
    )
    strict = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=window,
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=0.5,
    )

    assert permissive.coverage[0] == pytest.approx(0.125)
    assert permissive.status[0] is TrialCoverageStatus.PARTIAL_COVERAGE
    assert permissive.values[0] == pytest.approx(2.0)
    assert strict.status[0] is TrialCoverageStatus.BELOW_MINIMUM_COVERAGE
    assert np.isnan(strict.values[0])


def test_mismatched_clocks_are_refused_rather_than_combined() -> None:
    covariate = _covariate(
        time_s=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 1.0, 1.0]),
        clock_id="video",
    )
    timing = _timing(onsets=[0.0, 1.0], clock_id="acquisition")

    with pytest.raises(TrializationError, match="clock"):
        reduce_covariate_to_trials(
            covariate,
            timing=timing,
            window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
            reducer=MeanValue(),
            max_gap_s=1.5,
            minimum_coverage=0.0,
        )


def test_mismatched_subject_or_session_is_refused() -> None:
    covariate = _covariate(
        time_s=np.array([0.0, 1.0]),
        values=np.array([1.0, 1.0]),
        session="day-2",
    )
    timing = _timing(onsets=[0.0])

    with pytest.raises(TrializationError, match="trial timing is"):
        reduce_covariate_to_trials(
            covariate,
            timing=timing,
            window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
            reducer=MeanValue(),
            max_gap_s=1.5,
            minimum_coverage=0.0,
        )


def test_trial_timing_moves_between_clocks_only_through_a_synchronization() -> None:
    synchronization = fit_device_clock_sync(
        DeviceClockPulses.from_arrays(
            source_clock_id="video",
            target_clock_id="acquisition",
            source_time_s=[0.0, 1.0, 2.0],
            target_time_s=[0.5, 1.5, 2.5],
        ),
        DeviceClockSyncSpec(
            maximum_absolute_residual_s=1e-9,
            maximum_drift_ppm=10.0,
            minimum_source_span_s=1.0,
        ),
    )
    timing = _timing(onsets=[0.0, 1.0], clock_id="video")

    moved = timing.synchronized_to(synchronization)

    assert moved.clock_id == "acquisition"
    assert moved.onset_s == pytest.approx([0.5, 1.5])
    assert moved.trial == timing.trial
    assert moved.clock_synchronization_ids == (synchronization.synchronization_id,)
    with pytest.raises(TrializationError, match="clock synchronization expects"):
        moved.synchronized_to(synchronization)


def test_a_non_string_subject_id_round_trips_from_pose_to_a_study_column() -> None:
    """An integer subject id must join without a lossy str() anywhere on the path."""

    pose = PoseTrajectory(
        subject=np.int64(7),
        session=3,
        keypoint="nose",
        time_s=np.arange(0.0, 2.01, 0.5),
        x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        y=np.zeros(5),
        confidence=np.ones(5),
        coordinate_unit="cm",
        source="deeplabcut",
        clock_id="video",
    )
    speed = pose.speed(minimum_confidence=0.5)
    timing = _timing(onsets=[0.5, 1.5], subject=7, session=3)
    study = _study(subject=7, session=3)

    assert pose.subject == 7
    assert isinstance(pose.subject, int)
    reduction = reduce_covariate_to_trials(
        speed,
        timing=timing,
        window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=1.0,
    )
    joined = attach_trial_columns(study, [reduction])

    assert joined["nose_speed_mean"] == pytest.approx([2.0, 2.0])
    assert joined["nose_speed_mean_status"].tolist() == ["ok", "ok"]


def test_annotation_reducers_produce_fraction_count_and_latency() -> None:
    annotations = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"lick": (0.4, 0.7, 1.6)},
        intervals=(
            BehaviorInterval("approach", 0.25, 0.75),
            BehaviorInterval("approach", 1.5, 1.6),
        ),
        source="boris",
        clock_id="video",
    )
    timing = _timing(onsets=[0.0, 1.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=1.0)

    fraction = reduce_annotations_to_trials(
        annotations,
        timing=timing,
        window=window,
        reducer=FractionOfTimeInState("approach"),
        observed_span_s=(0.0, 2.0),
        minimum_coverage=1.0,
    )
    count = reduce_annotations_to_trials(
        annotations,
        timing=timing,
        window=window,
        reducer=EventCount("lick", include_points=True),
        observed_span_s=(0.0, 2.0),
        minimum_coverage=1.0,
    )
    latency = reduce_annotations_to_trials(
        annotations,
        timing=timing,
        window=window,
        reducer=FirstOccurrenceLatency("approach", include_points=False),
        observed_span_s=(0.0, 2.0),
        minimum_coverage=1.0,
    )

    assert fraction.name == "approach_fraction_of_time"
    assert fraction.values == pytest.approx([0.5, 0.1])
    assert count.values == pytest.approx([2.0, 1.0])
    assert latency.values == pytest.approx([0.25, 0.5])
    assert all(status is TrialCoverageStatus.OK for status in fraction.status)


def test_annotation_latency_without_an_occurrence_is_censored_not_unobserved() -> None:
    annotations = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"lick": (0.4,)},
        intervals=(),
        source="boris",
        clock_id="video",
    )
    timing = _timing(onsets=[0.0, 1.0])

    latency = reduce_annotations_to_trials(
        annotations,
        timing=timing,
        window=TrialWindow(start_offset_s=0.0, stop_offset_s=1.0),
        reducer=FirstOccurrenceLatency("lick"),
        observed_span_s=(0.0, 2.0),
        minimum_coverage=1.0,
    )

    assert latency.values[0] == pytest.approx(0.4)
    assert np.isnan(latency.values[1])
    assert latency.status[1] is TrialCoverageStatus.OK


def test_annotation_window_outside_the_declared_observed_span_is_visible() -> None:
    annotations = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"lick": (0.4,)},
        intervals=(),
        source="boris",
        clock_id="video",
    )
    timing = _timing(onsets=[0.0, 5.0])

    reduction = reduce_annotations_to_trials(
        annotations,
        timing=timing,
        window=TrialWindow(start_offset_s=0.0, stop_offset_s=1.0),
        reducer=EventCount("lick"),
        observed_span_s=(0.0, 2.0),
        minimum_coverage=0.0,
    )

    assert reduction.status == (
        TrialCoverageStatus.OK,
        TrialCoverageStatus.OUTSIDE_OBSERVED_SPAN,
    )
    assert np.isnan(reduction.values[1])


def test_offset_locked_windows_require_declared_offsets() -> None:
    timing = _timing(onsets=[0.0, 2.0])
    window = TrialWindow(
        start_offset_s=-0.5,
        stop_offset_s=0.0,
        start_anchor="offset",
        stop_anchor="offset",
    )

    with pytest.raises(TrializationError, match="declares no offset_s"):
        timing.windows(window)

    with_offsets = _timing(onsets=[0.0, 2.0], offsets=[1.0, 3.0])
    start, stop = with_offsets.windows(window)

    assert start == pytest.approx([0.5, 2.5])
    assert stop == pytest.approx([1.0, 3.0])


def test_whole_trial_windows_span_onset_to_offset() -> None:
    timing = _timing(onsets=[0.0, 2.0], offsets=[1.5, 3.0])
    window = TrialWindow(
        start_offset_s=0.0,
        stop_offset_s=0.0,
        start_anchor="onset",
        stop_anchor="offset",
    )

    start, stop = timing.windows(window)

    assert start == pytest.approx([0.0, 2.0])
    assert stop == pytest.approx([1.5, 3.0])


def test_a_window_running_into_the_next_trial_is_refused_unless_declared() -> None:
    covariate = _covariate(
        time_s=np.arange(0.0, 4.01, 0.5),
        values=np.full(9, 1.0),
    )
    timing = _timing(onsets=[0.0, 1.0])
    overlapping = TrialWindow(start_offset_s=0.0, stop_offset_s=2.0)

    with pytest.raises(TrializationError, match="next trial's onset"):
        reduce_covariate_to_trials(
            covariate,
            timing=timing,
            window=overlapping,
            reducer=MeanValue(),
            max_gap_s=0.6,
            minimum_coverage=0.0,
        )

    allowed = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=TrialWindow(
            start_offset_s=0.0,
            stop_offset_s=2.0,
            on_next_trial_overlap="allow",
        ),
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=0.0,
    )

    assert allowed.overlaps_next_trial.tolist() == [True, False]


def test_a_learned_reducer_is_refused_and_points_at_fold_fitting() -> None:
    class ThresholdCrossingFraction:
        """A reducer whose threshold would have to be learned from the study."""

        name = "threshold_crossing"
        fold_independent = False

        def unit(self, source_unit: str) -> str:
            return "proportion"

        def reduce(self, time_s, values, **_: object) -> float:
            return float(np.mean(values > np.mean(values)))

    covariate = _covariate(time_s=np.array([0.0, 1.0]), values=np.array([1.0, 2.0]))

    with pytest.raises(TrializationError, match="fit_transform_split"):
        reduce_covariate_to_trials(
            covariate,
            timing=_timing(onsets=[0.0]),
            window=TrialWindow(start_offset_s=0.0, stop_offset_s=1.0),
            reducer=ThresholdCrossingFraction(),  # type: ignore[arg-type]
            max_gap_s=1.5,
            minimum_coverage=0.0,
        )


def test_attaching_preserves_source_row_order_and_carries_coverage() -> None:
    covariate = _covariate(
        time_s=np.arange(0.0, 3.01, 0.5),
        values=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    timing = _timing(onsets=[0.0, 1.0, 2.0])
    study = Study.from_columns(
        {
            "subject": ["mouse-1"] * 3,
            "session": ["day-1"] * 3,
            "trial": [2, 0, 1],
            "session_order": [0, 0, 0],
        }
    )

    reduction = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=1.0,
    )
    joined = attach_trial_columns(study, [reduction])

    assert joined["trial"].tolist() == [2, 0, 1]
    assert joined["speed_mean"] == pytest.approx([4.5, 0.5, 2.5])
    assert joined["speed_mean_coverage"] == pytest.approx([1.0, 1.0, 1.0])
    assert joined.columns[-3:] == ("speed_mean", "speed_mean_coverage", "speed_mean_status")


def test_attaching_refuses_uncovered_rows_unless_they_are_declared_empty() -> None:
    covariate = _covariate(time_s=np.arange(0.0, 2.01, 0.5), values=np.full(5, 1.0))
    timing = _timing(onsets=[0.0])
    study = _study(n_trials=2)
    reduction = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=1.0,
    )

    with pytest.raises(TrializationError, match="does not cover"):
        attach_trial_columns(study, [reduction])

    joined = attach_trial_columns(study, [reduction], on_missing="nan")

    assert np.isnan(joined["speed_mean"][1])
    assert joined["speed_mean_status"].tolist() == ["ok", "not_reduced"]


def test_attaching_refuses_to_overwrite_an_existing_column() -> None:
    covariate = _covariate(time_s=np.arange(0.0, 2.01, 0.5), values=np.full(5, 1.0))
    reduction = reduce_covariate_to_trials(
        covariate,
        timing=_timing(onsets=[0.0, 1.0]),
        window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=1.0,
        name="subject",
    )

    with pytest.raises(TrializationError, match="required Study column"):
        attach_trial_columns(_study(), [reduction])


def test_attaching_refuses_a_collision_with_a_column_it_just_added() -> None:
    covariate = _covariate(time_s=np.arange(0.0, 2.01, 0.5), values=np.full(5, 1.0))
    timing = _timing(onsets=[0.0, 1.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=0.5)
    shared = {"max_gap_s": 0.6, "minimum_coverage": 0.0}
    value = reduce_covariate_to_trials(
        covariate, timing=timing, window=window, reducer=MeanValue(), name="speed", **shared
    )
    collides = reduce_covariate_to_trials(
        covariate,
        timing=timing,
        window=window,
        reducer=MeanValue(),
        name="speed_coverage",
        **shared,
    )

    with pytest.raises(TrializationError, match="already has a column named"):
        attach_trial_columns(_study(), [value, collides])


def test_reductions_sharing_a_name_must_measure_the_same_quantity() -> None:
    covariate = _covariate(time_s=np.arange(0.0, 2.01, 0.5), values=np.full(5, 1.0))
    timing = _timing(onsets=[0.0, 1.0])
    window = TrialWindow(start_offset_s=0.0, stop_offset_s=0.5)
    shared = {"max_gap_s": 0.6, "minimum_coverage": 0.0, "name": "summary"}
    mean = reduce_covariate_to_trials(
        covariate, timing=timing, window=window, reducer=MeanValue(), **shared
    )
    maximum = reduce_covariate_to_trials(
        covariate, timing=timing, window=window, reducer=MaximumValue(), **shared
    )

    with pytest.raises(TrializationError, match="disagree on unit or reducer"):
        attach_trial_columns(_study(), [mean, maximum])


def test_two_sessions_reduce_into_one_column_without_crossing_boundaries() -> None:
    study = Study.from_columns(
        {
            "subject": ["mouse-1"] * 4,
            "session": ["day-1", "day-1", "day-2", "day-2"],
            "trial": [0, 1, 0, 1],
            "session_order": [0, 0, 1, 1],
        }
    )
    reductions = [
        reduce_covariate_to_trials(
            _covariate(
                time_s=np.arange(0.0, 2.01, 0.5),
                values=np.full(5, value),
                session=session,
            ),
            timing=_timing(onsets=[0.0, 1.0], session=session),
            window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
            reducer=MeanValue(),
            max_gap_s=0.6,
            minimum_coverage=1.0,
        )
        for session, value in (("day-1", 2.0), ("day-2", 5.0))
    ]

    joined = attach_trial_columns(study, reductions)

    assert joined["speed_mean"] == pytest.approx([2.0, 2.0, 5.0, 5.0])


def test_trial_timing_is_declared_from_a_named_event_stream() -> None:
    annotations = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"cue": (0.5, 2.5, 4.5), "reward": (1.0, 3.0, 5.0)},
        intervals=(),
        source="boris",
        clock_id="video",
    )

    timing = trial_timing_from_events(
        annotations,
        onset_label="cue",
        offset_label="reward",
    )

    assert timing.trial == (0, 1, 2)
    assert timing.onset_s == pytest.approx([0.5, 2.5, 4.5])
    assert timing.offset_s == pytest.approx([1.0, 3.0, 5.0])
    assert timing.clock_id == "video"
    with pytest.raises(TrializationError, match="no 'missing' events"):
        trial_timing_from_events(annotations, onset_label="missing")


def test_unpaired_offsets_are_refused_rather_than_truncated() -> None:
    annotations = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"cue": (0.5, 2.5, 4.5), "reward": (1.0, 3.0)},
        intervals=(),
        source="boris",
        clock_id="video",
    )

    with pytest.raises(TrializationError, match="pair them explicitly"):
        trial_timing_from_events(annotations, onset_label="cue", offset_label="reward")


def test_trial_timing_refuses_disordered_trial_numbers() -> None:
    with pytest.raises(TrializationError, match="strictly increasing"):
        TrialTiming(
            subject="mouse-1",
            session="day-1",
            trial=(1, 0),
            onset_s=np.array([0.0, 1.0]),
            clock_id="video",
        )


def test_trial_timing_refuses_disordered_onsets() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TrialTiming.from_arrays(
            subject="mouse-1",
            session="day-1",
            onset_s=[1.0, 0.0],
            clock_id="video",
        )


def test_reduction_reports_its_emitted_column_names() -> None:
    covariate = _covariate(time_s=np.arange(0.0, 2.01, 0.5), values=np.full(5, 1.0))
    reduction = reduce_covariate_to_trials(
        covariate,
        timing=_timing(onsets=[0.0, 1.0]),
        window=TrialWindow(start_offset_s=0.0, stop_offset_s=0.5),
        reducer=MeanValue(),
        max_gap_s=0.6,
        minimum_coverage=1.0,
    )

    assert isinstance(reduction, TrialReduction)
    assert reduction.column_names == (
        "speed_mean",
        "speed_mean_coverage",
        "speed_mean_status",
    )
    assert reduction.window.description == "[onset+0s, onset+0.5s]"
