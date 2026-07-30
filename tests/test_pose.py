"""Pose trajectories and the DeepLabCut, SLEAP, and movement readers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from behavio.observed.pose import (
    pose_from_deeplabcut,
    pose_from_deeplabcut_file,
    pose_from_movement,
    pose_from_sleap,
    pose_from_sleap_analysis_h5,
)

FIXTURES = Path(__file__).parent / "fixtures" / "readers"


def _movement_poses(
    *,
    position: np.ndarray,
    confidence: np.ndarray | None = None,
    keypoints: list[str],
    individuals: list[str],
    space: list[str] | None = None,
    fps: float | None = 1.0,
    source_software: str = "DeepLabCut",
    dimension_names: tuple[str, str] = ("keypoint", "individual"),
) -> xr.Dataset:
    keypoint_dim, individual_dim = dimension_names
    dims = ("time", "space", keypoint_dim, individual_dim)
    coords: dict[str, object] = {
        "space": space if space is not None else ["x", "y"],
        keypoint_dim: keypoints,
        individual_dim: individuals,
    }
    attrs: dict[str, object] = {"ds_type": "poses", "source_software": source_software}
    if fps is not None:
        coords["time"] = np.arange(position.shape[0], dtype=float) / fps
        attrs["fps"] = fps
        attrs["time_unit"] = "seconds"
    else:
        coords["time"] = np.arange(position.shape[0], dtype=float)
        attrs["time_unit"] = "frames"
    data_vars: dict[str, object] = {"position": (dims, position.astype(np.float32))}
    if confidence is not None:
        data_vars["confidence"] = (
            ("time", keypoint_dim, individual_dim),
            confidence.astype(np.float32),
        )
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def _single_keypoint_movement_poses(
    x: list[float],
    y: list[float],
    likelihood: list[float] | None,
    *,
    fps: float | None = 1.0,
    **kwargs: object,
) -> xr.Dataset:
    position = np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    position = position.T.reshape(len(x), 2, 1, 1)
    confidence = (
        np.asarray(likelihood, dtype=float).reshape(len(x), 1, 1)
        if likelihood is not None
        else None
    )
    return _movement_poses(
        position=position,
        confidence=confidence,
        keypoints=["nose"],
        individuals=["individual_0"],
        fps=fps,
        **kwargs,  # type: ignore[arg-type]
    )


class _Frame:
    def __init__(self, columns: dict[tuple[str, ...], list[float]]) -> None:
        self._columns = columns
        self.columns = tuple(columns)

    def __getitem__(self, name: tuple[str, ...]) -> list[float]:
        return self._columns[name]


def test_deeplabcut_pose_speed_and_gap_safe_alignment() -> None:
    frame = _Frame(
        {
            ("network", "nose", "x"): [0.0, 1.0, 2.0, 30.0, 31.0],
            ("network", "nose", "y"): [0.0, 0.0, 0.0, 0.0, 0.0],
            ("network", "nose", "likelihood"): [0.99, 0.99, 0.2, 0.99, 0.99],
        }
    )
    pose = pose_from_deeplabcut(
        frame,
        subject="mouse-1",
        session="day-1",
        keypoint="nose",
        fps=1.0,
        scorer="network",
    )
    speed = pose.speed(minimum_confidence=0.9)
    aligned_covariate = speed.aligned_to(
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        target_clock_id="video",
        max_gap_s=1.1,
    )
    aligned = aligned_covariate.values

    assert speed.unit == "px/s"
    assert speed.valid.tolist() == [False, True, False, False, True]
    assert aligned_covariate.valid.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert aligned[0] == pytest.approx(1.0)
    assert np.isnan(aligned[1:6]).all()
    assert aligned[6] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="clock mismatch"):
        speed.align_to([1.0], target_clock_id="photometry", max_gap_s=1.1)


def test_deeplabcut_requires_explicit_identity_when_columns_are_ambiguous() -> None:
    columns: dict[tuple[str, ...], list[float]] = {}
    for scorer in ("network-a", "network-b"):
        columns[(scorer, "nose", "x")] = [0.0, 1.0]
        columns[(scorer, "nose", "y")] = [0.0, 1.0]
        columns[(scorer, "nose", "likelihood")] = [0.9, 0.9]
    with pytest.raises(ValueError, match="missing or ambiguous"):
        pose_from_deeplabcut(
            _Frame(columns),
            subject="mouse-1",
            session="day-1",
            keypoint="nose",
            fps=30.0,
        )


@pytest.mark.parametrize("suffix", [".csv", ".h5"])
def test_deeplabcut_documented_file_shapes(tmp_path: Path, suffix: str) -> None:
    columns = pd.MultiIndex.from_product([["network"], ["nose"], ["x", "y", "likelihood"]])
    frame = pd.DataFrame(
        [[0.0, 2.0, 0.99], [1.0, 3.0, 0.95], [2.0, 4.0, 0.91]],
        columns=columns,
    )
    path = tmp_path / f"predictions{suffix}"
    if suffix == ".csv":
        frame.to_csv(path)
    else:
        frame.to_hdf(path, key="df_with_missing")

    pose = pose_from_deeplabcut_file(
        path,
        subject="mouse-1",
        session="day-1",
        keypoint="nose",
        scorer="network",
        fps=30.0,
        source_version="schema-generated",
    )

    assert pose.x.tolist() == [0.0, 1.0, 2.0]
    assert pose.confidence.tolist() == [0.99, 0.95, 0.91]
    assert pose.source_artifact == str(path)


def test_sleap_matlab_and_standard_axes_are_explicit() -> None:
    standard = np.zeros((4, 1, 2, 2), dtype=float)
    standard[:, 0, 1, 0] = [0.0, 1.0, 2.0, 3.0]
    standard[:, 0, 1, 1] = [4.0, 5.0, 6.0, 7.0]
    scores = np.ones((4, 1, 2), dtype=float)
    pose = pose_from_sleap(
        standard,
        subject="mouse-1",
        session="day-1",
        node_names=["tail", "nose"],
        node="nose",
        dims=("frame", "track", "node", "xy"),
        point_scores=scores,
        fps=20.0,
    )
    matlab = np.transpose(standard, (1, 3, 2, 0))
    matlab_scores = np.transpose(scores, (1, 2, 0))
    same_pose = pose_from_sleap(
        matlab,
        subject="mouse-1",
        session="day-1",
        node_names=["tail", "nose"],
        node="nose",
        dims=("track", "xy", "node", "frame"),
        point_scores=matlab_scores,
        fps=20.0,
    )

    assert pose.x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert same_pose.x.tolist() == pose.x.tolist()
    assert same_pose.y.tolist() == pose.y.tolist()


def test_official_sleap_analysis_h5_fixture() -> None:
    pose = pose_from_sleap_analysis_h5(
        FIXTURES / "sleap-small-robot.analysis.h5",
        subject="robot-1",
        session="three-frames",
        node="front",
        dims=("track", "xy", "node", "frame"),
        fps=25.0,
        source_version="legacy-format-v1",
    )

    assert pose.keypoint == "front"
    assert pose.x.tolist() == pytest.approx([382.74652904, 384.70135498, 382.74652904])
    assert pose.y.tolist() == pytest.approx([243.01542777, 244.34580994, 243.01542777])
    assert pose.confidence[1] > 1.0
    assert pose.source_artifact is not None


def test_legacy_sleap_file_requires_explicit_axes() -> None:
    with pytest.raises(ValueError, match="declare dims"):
        pose_from_sleap_analysis_h5(
            FIXTURES / "sleap-small-robot.analysis.h5",
            subject="robot-1",
            session="three-frames",
            node="front",
            fps=25.0,
        )


def test_sleap_file_uses_json_dimension_metadata(tmp_path: Path) -> None:
    path = tmp_path / "current.analysis.h5"
    with h5py.File(path, "w") as file:
        tracks = file.create_dataset("tracks", data=np.zeros((3, 1, 1, 2)))
        tracks.attrs["dims"] = '["frame", "track", "node", "xy"]'
        scores = file.create_dataset("point_scores", data=np.ones((3, 1, 1)))
        scores.attrs["dims"] = '["frame", "track", "node"]'
        file.create_dataset("node_names", data=[b"nose"])

    pose = pose_from_sleap_analysis_h5(
        path,
        subject="mouse-1",
        session="day-1",
        node="nose",
        fps=30.0,
    )

    assert pose.x.tolist() == [0.0, 0.0, 0.0]
    assert pose.confidence.tolist() == [1.0, 1.0, 1.0]


def test_movement_dataset_matches_first_party_reader_values_and_mask() -> None:
    x = [0.0, 1.0, 2.0, 30.0, 31.0]
    y = [0.0, 0.0, 0.0, 0.0, 0.0]
    likelihood = [0.99, 0.99, 0.2, 0.99, 0.99]
    native = pose_from_deeplabcut(
        _Frame(
            {
                ("network", "nose", "x"): x,
                ("network", "nose", "y"): y,
                ("network", "nose", "likelihood"): likelihood,
            }
        ),
        subject="mouse-1",
        session="day-1",
        keypoint="nose",
        fps=1.0,
        scorer="network",
    )
    bridged = pose_from_movement(
        _single_keypoint_movement_poses(x, y, likelihood),
        subject="mouse-1",
        session="day-1",
    )

    assert bridged.source == native.source == "deeplabcut"
    assert bridged.keypoint == native.keypoint == "nose"
    assert bridged.time_s.tolist() == native.time_s.tolist()
    assert bridged.x.tolist() == pytest.approx(native.x.tolist())
    assert bridged.y.tolist() == pytest.approx(native.y.tolist())
    assert bridged.confidence.tolist() == pytest.approx(native.confidence.tolist())

    bridged_speed = bridged.speed(minimum_confidence=0.9)
    native_speed = native.speed(minimum_confidence=0.9)
    assert bridged_speed.valid.tolist() == native_speed.valid.tolist()
    assert bridged_speed.valid.tolist() == [False, True, False, False, True]
    assert bridged_speed.values[1:].tolist() == pytest.approx(native_speed.values[1:].tolist())
    assert np.isnan(bridged_speed.values[0])


def test_movement_speed_never_reports_a_confidence_gated_sample() -> None:
    speed = pose_from_movement(
        _single_keypoint_movement_poses(
            [0.0, 1.0, 2.0, 30.0, 31.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.99, 0.99, 0.2, 0.99, 0.99],
        ),
        subject="mouse-1",
        session="day-1",
    ).speed(minimum_confidence=0.9)

    assert speed.values[3] == pytest.approx(28.0)
    assert not speed.valid[3]
    assert speed.valid[1]
    assert speed.values[1] == pytest.approx(1.0)
    assert 14.5 not in speed.values[speed.valid].tolist()


def test_movement_dataset_requires_explicit_identity_when_ambiguous() -> None:
    position = np.zeros((3, 2, 2, 2), dtype=float)
    dataset = _movement_poses(
        position=position,
        confidence=np.ones((3, 2, 2), dtype=float),
        keypoints=["nose", "tail_base"],
        individuals=["mouse-a", "mouse-b"],
    )

    with pytest.raises(ValueError, match="keypoint is missing or ambiguous"):
        pose_from_movement(dataset, subject="mouse-1", session="day-1")
    with pytest.raises(ValueError, match="individual is missing or ambiguous"):
        pose_from_movement(dataset, subject="mouse-1", session="day-1", keypoint="nose")
    with pytest.raises(ValueError, match="no keypoint 'snout'"):
        pose_from_movement(
            dataset,
            subject="mouse-1",
            session="day-1",
            keypoint="snout",
            individual="mouse-a",
        )

    pose = pose_from_movement(
        dataset,
        subject="mouse-1",
        session="day-1",
        keypoint="tail_base",
        individual="mouse-b",
    )
    assert pose.keypoint == "tail_base"
    assert pose.individual == "mouse-b"


def test_movement_dataset_refuses_frame_indexed_time_without_declaration() -> None:
    dataset = _single_keypoint_movement_poses(
        [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], fps=None
    )

    with pytest.raises(ValueError, match="not expressed in seconds"):
        pose_from_movement(dataset, subject="mouse-1", session="day-1")

    pose = pose_from_movement(dataset, subject="mouse-1", session="day-1", fps=25.0)
    assert pose.time_s.tolist() == pytest.approx([0.0, 0.04, 0.08])


def test_movement_dataset_carries_float32_precision_into_float64() -> None:
    value = 382.74652904163474
    pose = pose_from_movement(
        _single_keypoint_movement_poses([value], [0.0], [1.0]),
        subject="mouse-1",
        session="day-1",
    )

    assert pose.x.dtype == np.float64
    assert pose.x[0] != value
    assert pose.x[0] == pytest.approx(value)
    assert pose.x[0] == float(np.float32(value))


def test_movement_dataset_reads_three_dimensional_and_plural_dimensions() -> None:
    position = np.zeros((3, 3, 1, 1), dtype=float)
    position[:, 2, 0, 0] = [0.0, 3.0, 6.0]
    dataset = _movement_poses(
        position=position,
        confidence=np.ones((3, 1, 1), dtype=float),
        keypoints=["nose"],
        individuals=["id_0"],
        space=["x", "y", "z"],
        source_software="SLEAP",
        dimension_names=("keypoints", "individuals"),
    )

    pose = pose_from_movement(dataset, subject="mouse-1", session="day-1")

    assert pose.source == "sleap"
    assert pose.z is not None
    assert pose.z.tolist() == pytest.approx([0.0, 3.0, 6.0])
    assert pose.speed(minimum_confidence=0.5).values[1:].tolist() == pytest.approx([3.0, 3.0])


def test_movement_dataset_without_confidence_is_missing_not_certain() -> None:
    pose = pose_from_movement(
        _single_keypoint_movement_poses([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], None),
        subject="mouse-1",
        session="day-1",
    )

    assert np.isnan(pose.confidence).all()
    assert not pose.speed(minimum_confidence=0.0).valid.any()


def test_movement_bboxes_dataset_is_refused() -> None:
    dataset = _single_keypoint_movement_poses([0.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    dataset.attrs["ds_type"] = "bboxes"

    with pytest.raises(ValueError, match="must be a poses dataset"):
        pose_from_movement(dataset, subject="mouse-1", session="day-1")
