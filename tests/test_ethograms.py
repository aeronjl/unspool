"""Behavioural intervals, point events, and the MoSeq and BORIS readers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from behavio.observed.ethograms import (
    BehaviorAnnotations,
    BehaviorInterval,
    annotations_from_boris,
    annotations_from_boris_tabular_file,
    annotations_from_moseq,
    annotations_from_moseq_results_h5,
)

FIXTURES = Path(__file__).parent / "fixtures" / "readers"


def test_moseq_bouts_preserve_duration_and_expose_edges() -> None:
    annotations = annotations_from_moseq(
        [2, 2, 2, 5, 5, 2],
        subject="mouse-1",
        session="day-1",
        fps=2.0,
        labels={2: "rear", 5: "groom"},
    )

    assert [item.label for item in annotations.intervals] == [
        "rear",
        "groom",
        "rear",
    ]
    assert annotations.intervals[0].start_s == 0.0
    assert annotations.intervals[0].stop_s == 1.5
    assert annotations.event_times()["groom"] == (1.5,)
    assert annotations.event_times(edge="offset")["groom"] == (2.5,)
    inputs = annotations.interval_encoding_inputs()
    assert inputs.events["rear"] == (0.0, 2.5)
    assert inputs.event_values["rear"]["duration_s"] == (1.5, 0.5)
    assert inputs.intervals["rear"] == ((0.0, 1.5), (2.5, 3.0))
    assert inputs.events["groom"] == (1.5,)
    assert inputs.event_values["groom"]["duration_s"] == (1.0,)
    assert inputs.intervals["groom"] == ((1.5, 2.5),)


def test_moseq_documented_results_h5_shape(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as file:
        recording = file.create_group("recording-01")
        recording.create_dataset("syllable", data=[3, 3, 1, 1, 1, 3])
        recording.create_dataset("latent_state", data=np.zeros((6, 2)))
        recording.create_dataset("centroid", data=np.zeros((6, 2)))
        recording.create_dataset("heading", data=np.zeros(6))

    annotations = annotations_from_moseq_results_h5(
        path,
        recording="recording-01",
        subject="mouse-1",
        session="day-1",
        fps=30.0,
        labels={1: "rear", 3: "pause"},
        source_version="schema-generated",
    )

    assert [item.label for item in annotations.intervals] == [
        "pause",
        "rear",
        "pause",
    ]
    assert annotations.source_artifact == str(path)


def test_boris_points_states_and_normalized_progress_are_encoding_ready() -> None:
    annotations = annotations_from_boris(
        {
            "Behavior": ["cue", "approach"],
            "Type": ["POINT", "STATE"],
            "Start": [0.5, 1.0],
            "Stop": [np.nan, 3.0],
        },
        subject="mouse-1",
        session="day-1",
        behavior_column="Behavior",
        type_column="Type",
        start_column="Start",
        stop_column="Stop",
    )
    progress = annotations.normalized_progress([0.0, 1.0, 2.0, 3.0, 4.0], label="approach")

    assert annotations.point_events == {"cue": (0.5,)}
    assert progress.values[1:4].tolist() == [0.0, 0.5, 1.0]
    assert progress.valid.tolist() == [False, True, True, True, False]
    assert annotations.event_times()["approach"] == (1.0,)
    assert annotations.event_times()["cue"] == (0.5,)


def test_progress_rejects_overlapping_bouts() -> None:
    annotations = BehaviorAnnotations(
        subject="mouse-1",
        session="day-1",
        point_events={},
        intervals=(
            BehaviorInterval("groom", 0.0, 2.0),
            BehaviorInterval("groom", 1.0, 3.0),
        ),
        source="manual",
        clock_id="video",
    )
    with pytest.raises(ValueError, match="overlapping"):
        annotations.normalized_progress([0.0, 1.0, 2.0, 3.0], label="groom")


def test_official_boris_tabular_fixture_pairs_state_rows() -> None:
    annotations = annotations_from_boris_tabular_file(
        FIXTURES / "boris-test-export-events-tabular.csv",
        subject="canonical-mouse-1",
        session="observation-1",
        source_subject="subject1",
        source_version="upstream-fixture",
    )

    assert annotations.point_events == {}
    assert [(item.start_s, item.stop_s) for item in annotations.intervals] == [
        (3.3, 7.75),
        (9.9, 16.2),
        (18.35, 24.475),
    ]
    assert {item.label for item in annotations.intervals} == {"s"}


def test_boris_tabular_fixture_requires_subject_selection() -> None:
    with pytest.raises(ValueError, match="multiple subjects"):
        annotations_from_boris_tabular_file(
            FIXTURES / "boris-test-export-events-tabular.csv",
            subject="mouse-1",
            session="observation-1",
        )


def test_boris_tabular_file_preserves_point_rows(tmp_path: Path) -> None:
    path = tmp_path / "point.csv"
    path.write_text(
        "Observation id,demo,,,,\n"
        "Time,Subject,Behavior,Status\n"
        "1.25,mouse-1,cue,POINT\n"
        "2.50,mouse-1,reward,\n"
    )

    annotations = annotations_from_boris_tabular_file(
        path,
        subject="mouse-1",
        session="day-1",
    )

    assert annotations.point_events == {"cue": (1.25,), "reward": (2.5,)}
