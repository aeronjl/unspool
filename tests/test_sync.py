"""Matched-pulse clock synchronisation and its effect on the three carried types."""

from __future__ import annotations

import json

import numpy as np
import pytest

from behavio.covariates import BehaviorCovariate
from behavio.ethograms import BehaviorAnnotations, BehaviorInterval
from behavio.pose import pose_from_sleap
from behavio.sync import (
    ClockPulseMatches,
    ClockSynchronizationSpec,
    fit_clock_synchronization,
)


def _clock_synchronization():
    source = np.asarray([0.0, 20.0, 40.0, 60.0])
    target = 0.35 + 1.00012 * source + np.asarray([0.0, 0.0002, -0.0001, 0.0])
    matches = ClockPulseMatches.from_arrays(
        source_clock_id="video",
        target_clock_id="photometry",
        source_time_s=source,
        target_time_s=target,
        match_labels=("pulse-0", "pulse-1", "pulse-2", "pulse-3"),
    )
    spec = ClockSynchronizationSpec(
        maximum_absolute_residual_s=0.001,
        maximum_drift_ppm=200.0,
        minimum_source_span_s=30.0,
    )
    return fit_clock_synchronization(matches, spec)


def test_clock_synchronization_recovers_drift_and_retains_evidence() -> None:
    synchronization = _clock_synchronization()
    payload = json.loads(synchronization.to_json())

    assert synchronization.intercept_s == pytest.approx(0.35004, abs=1e-4)
    assert synchronization.scale == pytest.approx(1.000119, abs=1e-5)
    assert synchronization.drift_ppm == pytest.approx(119.0, abs=10.0)
    assert synchronization.matched_pulses == 4
    assert synchronization.maximum_absolute_residual_s < 0.001
    assert payload["artifact_type"] == "clock_synchronization"
    assert payload["schema_version"] == "1"
    assert len(payload["residual_s"]) == 4
    assert payload["match_labels"] == [
        "pulse-0",
        "pulse-1",
        "pulse-2",
        "pulse-3",
    ]
    assert synchronization.synchronization_id.startswith("clock-sync-")
    assert synchronization.synchronization_id == _clock_synchronization().synchronization_id

    transformed = synchronization.transform_time([5.0, 55.0])
    assert transformed == pytest.approx(
        synchronization.intercept_s + synchronization.scale * np.asarray([5.0, 55.0])
    )
    assert not transformed.flags.writeable
    with pytest.raises(ValueError, match="requires 1s extrapolation"):
        synchronization.transform_time([61.0])
    allowed = synchronization.transform_time([61.0], maximum_extrapolation_s=1.0)
    assert allowed[0] == pytest.approx(synchronization.intercept_s + synchronization.scale * 61.0)


def test_clock_synchronization_rejects_bad_matches_and_failed_thresholds() -> None:
    with pytest.raises(ValueError, match="pulse counts must match"):
        ClockPulseMatches.from_arrays(
            source_clock_id="video",
            target_clock_id="photometry",
            source_time_s=[0.0, 1.0],
            target_time_s=[0.0],
        )
    with pytest.raises(ValueError, match="target pulse times must be strictly"):
        ClockPulseMatches.from_arrays(
            source_clock_id="video",
            target_clock_id="photometry",
            source_time_s=[0.0, 1.0, 2.0],
            target_time_s=[0.0, 2.0, 1.0],
        )

    too_few = ClockPulseMatches.from_arrays(
        source_clock_id="video",
        target_clock_id="photometry",
        source_time_s=[0.0, 20.0],
        target_time_s=[0.0, 20.0],
    )
    spec = ClockSynchronizationSpec(0.01, 200.0)
    with pytest.raises(ValueError, match="has 2 matches"):
        fit_clock_synchronization(too_few, spec)

    excessive_drift = ClockPulseMatches.from_arrays(
        source_clock_id="video",
        target_clock_id="photometry",
        source_time_s=[0.0, 10.0, 20.0, 30.0],
        target_time_s=[0.0, 10.01, 20.02, 30.03],
    )
    with pytest.raises(ValueError, match="clock drift"):
        fit_clock_synchronization(excessive_drift, spec)

    nonlinear = ClockPulseMatches.from_arrays(
        source_clock_id="video",
        target_clock_id="photometry",
        source_time_s=[0.0, 10.0, 20.0, 30.0],
        target_time_s=[0.0, 10.0, 20.1, 30.0],
    )
    residual_spec = ClockSynchronizationSpec(0.01, 5_000.0)
    with pytest.raises(ValueError, match="maximum absolute residual"):
        fit_clock_synchronization(nonlinear, residual_spec)


def test_synchronization_composes_pose_covariates_and_annotations() -> None:
    synchronization = _clock_synchronization()
    pose = pose_from_sleap(
        np.zeros((2, 1, 1, 2)),
        subject="mouse-1",
        session="day-1",
        node_names=["nose"],
        node="nose",
        dims=("frame", "track", "node", "xy"),
        time_s=[10.0, 20.0],
        clock_id="video",
    )
    synchronized_pose = synchronization.synchronize_pose(pose)
    speed = synchronized_pose.speed(minimum_confidence=0.0)
    assert synchronized_pose.clock_id == "photometry"
    assert synchronized_pose.clock_synchronization_ids == (synchronization.synchronization_id,)
    assert speed.clock_synchronization_ids == synchronized_pose.clock_synchronization_ids

    covariate = BehaviorCovariate(
        subject="mouse-1",
        session="day-1",
        name="speed",
        time_s=np.asarray([10.0, 20.0]),
        values=np.asarray([1.0, 2.0]),
        valid=np.asarray([True, False]),
        unit="cm/s",
        source="pose",
        clock_id="video",
    )
    synchronized_covariate = synchronization.synchronize_covariate(covariate)
    assert synchronized_covariate.clock_id == "photometry"
    assert synchronized_covariate.values.tolist() == [1.0, 2.0]
    assert synchronized_covariate.valid.tolist() == [True, False]
    assert synchronized_covariate.clock_synchronization_ids == (synchronization.synchronization_id,)

    annotations = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={"cue": (10.0,)},
        intervals=(BehaviorInterval("rear", 20.0, 30.0),),
        source="boris",
        clock_id="video",
    )
    synchronized_annotations = synchronization.synchronize_annotations(annotations)
    assert synchronized_annotations.clock_id == "photometry"
    assert synchronized_annotations.point_events["cue"] == pytest.approx(
        (synchronization.intercept_s + synchronization.scale * 10.0,)
    )
    assert synchronized_annotations.intervals[0].start_s == pytest.approx(
        synchronization.intercept_s + synchronization.scale * 20.0
    )
    assert synchronized_annotations.clock_synchronization_ids == (
        synchronization.synchronization_id,
    )
    with pytest.raises(ValueError, match="expects 'video'"):
        synchronization.synchronize_covariate(synchronized_covariate)
