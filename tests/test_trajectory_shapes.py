from __future__ import annotations

import json

import numpy as np
import pytest

from behavio import (
    TrajectoryPanel,
    audit_trajectory_replication,
    compare_trajectory_shapes,
)


def replicated_panel() -> TrajectoryPanel:
    reference = np.asarray([-1.0, 0.0, 1.0])
    level_shift = reference + 4.0
    different_shape = np.asarray([-1.0, 1.0, -1.0])
    curves: list[np.ndarray] = []
    subjects: list[str] = []
    groups: list[str] = []
    offsets = (-0.15, -0.05, 0.05, 0.15)
    for group, mean in (
        ("reference", reference),
        ("level", level_shift),
        ("shape", different_shape),
    ):
        for subject_index, offset in enumerate(offsets):
            curves.append(mean + offset)
            subjects.append(f"{group}-{subject_index}")
            groups.append(group)
    return TrajectoryPanel(
        grid=np.asarray([0.0, 1.0, 2.0]),
        values=np.stack(curves),
        subjects=tuple(subjects),
        groups=tuple(groups),
        clock_name="landmark_relative_session",
        parameter_name="stimulus_weight",
    )


def test_shape_comparison_separates_level_from_scale_free_shape() -> None:
    report = compare_trajectory_shapes(
        replicated_panel(), bootstrap_resamples=200, bootstrap_seed=31
    )

    level = report.comparison_for("reference", "level")
    shape = report.comparison_for("reference", "shape")
    assert level.raw_distance.estimate == pytest.approx(4.0)
    assert level.level_difference.estimate == pytest.approx(-4.0)
    assert level.centered_distance.estimate == pytest.approx(0.0, abs=1e-14)
    assert level.amplitude_difference.estimate == pytest.approx(0.0, abs=1e-14)
    assert level.shape_distance is not None
    assert level.shape_distance.estimate == pytest.approx(0.0, abs=1e-14)
    assert shape.centered_distance.estimate > 0.5
    assert shape.shape_distance is not None
    assert shape.shape_distance.estimate > 0.5

    reversed_level = report.comparison_for("level", "reference")
    assert reversed_level.raw_distance == level.raw_distance
    assert reversed_level.level_difference.estimate == pytest.approx(4.0)
    assert reversed_level.amplitude_difference.estimate == pytest.approx(
        -level.amplitude_difference.estimate
    )


def test_report_retains_alignment_replication_and_bootstrap_scope() -> None:
    panel = replicated_panel()
    report = compare_trajectory_shapes(panel, bootstrap_resamples=100, bootstrap_seed=9)

    assert report.replication_audit.inferentially_ready
    assert dict(report.replication_audit.group_sizes) == {
        "reference": 4,
        "level": 4,
        "shape": 4,
    }
    assert not report.grid.flags.writeable
    assert all(not summary.mean_values.flags.writeable for summary in report.group_summaries)
    payload = report.to_dict()
    assert payload["alignment"] == "fixed common grid; no interpolation or time warping"
    assert payload["bootstrap"]["unit"] == "subject within fixed group"
    assert payload["bootstrap"]["scope"].startswith("uncertainty in fixed group means")
    json.dumps(payload, allow_nan=False)


def test_singleton_labs_are_audited_and_rejected_for_comparison() -> None:
    panel = TrajectoryPanel(
        grid=np.asarray([0.0, 1.0, 2.0]),
        values=np.asarray([[0.0, 1.0, 2.0], [0.2, 1.2, 2.2]]),
        subjects=("mouse-a", "mouse-b"),
        groups=("lab-a", "lab-b"),
        clock_name="session_order",
        parameter_name="accuracy",
    )

    audit = audit_trajectory_replication(panel)
    assert not audit.inferentially_ready
    assert audit.singleton_groups == ("lab-a", "lab-b")
    assert audit.under_replicated_groups == ("lab-a", "lab-b")
    with pytest.raises(ValueError, match="at least 2 subjects per group"):
        compare_trajectory_shapes(panel, bootstrap_resamples=10)


def test_flat_group_means_leave_scale_free_shape_unresolved() -> None:
    panel = TrajectoryPanel(
        grid=np.asarray([0.0, 1.0, 2.0]),
        values=np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        ),
        subjects=("a-1", "a-2", "b-1", "b-2"),
        groups=("a", "a", "b", "b"),
        clock_name="session_order",
        parameter_name="flat_parameter",
    )

    comparison = compare_trajectory_shapes(panel, bootstrap_resamples=20).comparison_for("a", "b")
    assert comparison.raw_distance.estimate == pytest.approx(1.0)
    assert comparison.centered_distance.estimate == pytest.approx(0.0)
    assert comparison.shape_distance is None
    assert comparison.effective_shape_resamples == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid": [0.0, 0.0, 1.0]}, "strictly increasing"),
        ({"values": [[0.0, 1.0], [1.0, 2.0]]}, "aligned trajectory"),
        ({"subjects": ("same", "same")}, "unique"),
        ({"groups": ("one", "one")}, "at least two groups"),
    ],
)
def test_panel_rejects_ambiguous_alignment_and_identity(kwargs: dict, message: str) -> None:
    arguments = {
        "grid": [0.0, 1.0, 2.0],
        "values": [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
        "subjects": ("a", "b"),
        "groups": ("left", "right"),
        "clock_name": "session_order",
        "parameter_name": "accuracy",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TrajectoryPanel(**arguments)


def test_comparison_rejects_invalid_uncertainty_controls() -> None:
    panel = replicated_panel()
    with pytest.raises(ValueError, match="positive integer"):
        compare_trajectory_shapes(panel, bootstrap_resamples=0)
    with pytest.raises(ValueError, match="strictly between"):
        compare_trajectory_shapes(panel, bootstrap_resamples=10, confidence_level=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        compare_trajectory_shapes(panel, bootstrap_resamples=10, flat_tolerance=0.0)
