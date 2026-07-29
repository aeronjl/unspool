"""Checksum-pinned reader fixtures from released DeepLabCut, SLEAP, MoSeq and BORIS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from behavio.ethograms import (
    annotations_from_boris_aggregated_file,
    annotations_from_moseq_results_h5,
)
from behavio.pose import (
    pose_from_deeplabcut_file,
    pose_from_sleap_analysis_h5,
)

FIXTURES = Path(__file__).parent / "fixtures" / "readers"


def test_current_interoperability_fixture_checksums() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for fixture in manifest["fixtures"]:
        path = FIXTURES / fixture["file"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"]


def test_current_deeplabcut_single_animal_writer_fixture() -> None:
    pose = pose_from_deeplabcut_file(
        FIXTURES / "deeplabcut-3.0.0-single.h5",
        subject="mouse-1",
        session="day-1",
        keypoint="nose",
        scorer="DLC_Resnet50_current",
        fps=30.0,
        source_version="3.0.0",
    )

    assert pose.x.tolist() == [10.0, 11.0, 12.0]
    assert pose.y.tolist() == [20.0, 20.5, 21.0]
    assert pose.confidence.tolist() == [0.99, 0.95, 0.4]


def test_current_deeplabcut_multi_animal_writer_fixture_requires_identity() -> None:
    path = FIXTURES / "deeplabcut-3.0.0-multi.h5"
    with pytest.raises(ValueError, match="missing or ambiguous"):
        pose_from_deeplabcut_file(
            path,
            subject="mouse-a",
            session="day-1",
            keypoint="nose",
            scorer="DLC_Resnet50_current",
            fps=30.0,
        )

    pose = pose_from_deeplabcut_file(
        path,
        subject="mouse-a",
        session="day-1",
        keypoint="nose",
        scorer="DLC_Resnet50_current",
        individual="mouse-a",
        fps=30.0,
        source_version="3.0.0",
    )
    assert pose.x.tolist() == [10.0, 11.0, 12.0]
    assert pose.confidence.tolist() == [0.99, 0.96, 0.91]
    assert pose.individual == "mouse-a"


def test_current_keypoint_moseq_writer_fixture_preserves_bouts() -> None:
    annotations = annotations_from_moseq_results_h5(
        FIXTURES / "keypoint-moseq-0.6.8-results.h5",
        recording="recording-01",
        subject="mouse-1",
        session="day-1",
        fps=30.0,
        labels={1: "rear", 3: "pause"},
        source_version="0.6.8",
    )

    assert [item.label for item in annotations.intervals] == [
        "pause",
        "rear",
        "pause",
    ]
    assert [(item.start_s, item.stop_s) for item in annotations.intervals] == [
        pytest.approx((0.0, 2 / 30)),
        pytest.approx((2 / 30, 5 / 30)),
        pytest.approx((5 / 30, 6 / 30)),
    ]


def test_current_sleap_standard_writer_fixture_uses_stored_dimensions() -> None:
    pose = pose_from_sleap_analysis_h5(
        FIXTURES / "sleap-io-0.9.2-standard.analysis.h5",
        subject="fly-pair",
        session="clip-1",
        node="head",
        track=0,
        fps=30.0,
        source_version="sleap-io-0.9.2",
    )

    assert len(pose.x) == 1100
    assert pose.x[:3].tolist() == [201.0, 201.0, 201.0]
    assert pose.confidence[:3].tolist() == pytest.approx(
        [0.8258838057518005, 0.8114952445030212, 0.7909944653511047]
    )


def test_current_boris_aggregated_fixture_preserves_points_and_states() -> None:
    annotations = annotations_from_boris_aggregated_file(
        FIXTURES / "boris-9.13.0-aggregated.tsv",
        subject="canonical-mouse",
        session="observation-2",
        source_subject="No focal subject",
        source_version="9.13.0",
    )

    assert annotations.point_events["p"] == (
        32.825,
        34.15,
        34.925,
        299.715,
        301.34,
        303.24,
    )
    assert [(item.start_s, item.stop_s) for item in annotations.intervals[:3]] == [
        (1.8, 8.125),
        (10.25, 23.35),
        (26.775, 31.475),
    ]
    assert {item.label for item in annotations.intervals} == {"s"}


def test_boris_aggregated_fixture_requires_subject_selection() -> None:
    with pytest.raises(ValueError, match="multiple subjects"):
        annotations_from_boris_aggregated_file(
            FIXTURES / "boris-9.13.0-aggregated.tsv",
            subject="canonical-mouse",
            session="observation-2",
        )


def test_upstream_fixture_checksums_match_manifest() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for item in manifest["fixtures"]:
        digest = hashlib.sha256((FIXTURES / item["file"]).read_bytes()).hexdigest()
        assert digest == item["sha256"]
