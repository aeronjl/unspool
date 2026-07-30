"""Recover level, amplitude, and scale-free shape differences across replicated labs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from behavio import Study
from behavio.compare import (
    ParameterTrajectoryPanel,
    audit_trajectory_replication,
    compare_trajectory_shapes,
)
from behavio.evaluate import leave_one_lab_out_splits
from benchmarks.provenance import render

GROUPS = ("reference", "level_shift", "amplitude_shift", "shape_change")
SAME_SHAPE_PAIRS = (
    ("reference", "level_shift"),
    ("reference", "amplitude_shift"),
    ("level_shift", "amplitude_shift"),
)
CHANGED_SHAPE_PAIRS = (
    ("reference", "shape_change"),
    ("level_shift", "shape_change"),
    ("amplitude_shift", "shape_change"),
)


def experiment(
    *, seed: int, subjects_per_lab: int = 10, bootstrap_resamples: int = 300
) -> dict[str, Any]:
    """Run one matched four-lab component-recovery experiment."""

    if subjects_per_lab < 2:
        raise ValueError("subjects_per_lab must be at least two")
    grid = np.linspace(0.0, 1.0, 9)
    base = 2 * grid - 1
    truths = {
        "reference": base,
        "level_shift": base + 2.0,
        "amplitude_shift": 2.0 * base,
        "shape_change": np.sin(2 * np.pi * grid),
    }
    generator = np.random.default_rng(seed)
    subjects: list[str] = []
    groups: list[str] = []
    curves: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for group in GROUPS:
        truth = truths[group]
        truth_level = float(np.trapezoid(truth, grid))
        centered_truth = truth - truth_level
        for subject_index in range(subjects_per_lab):
            subject_intercept = generator.normal(0.0, 0.15)
            subject_amplitude = generator.normal(1.0, 0.08)
            point_noise = generator.normal(0.0, 0.06, size=grid.size)
            curves.append(
                truth_level + subject_intercept + subject_amplitude * centered_truth + point_noise
            )
            subjects.append(f"{group}-{subject_index:02d}")
            groups.append(group)
    panel = ParameterTrajectoryPanel(
        grid=grid,
        values=np.stack(curves),
        subjects=tuple(subjects),
        groups=tuple(groups),
        clock_name="normalized_learning_progress",
        parameter_name="synthetic_strategy_weight",
    )
    report = compare_trajectory_shapes(
        panel,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=seed + 1,
    )
    metrics = {
        _pair_name(left, right): {
            "raw_distance": report.comparison_for(left, right).raw_distance.estimate,
            "absolute_level_difference": abs(
                report.comparison_for(left, right).level_difference.estimate
            ),
            "centered_distance": report.comparison_for(left, right).centered_distance.estimate,
            "absolute_amplitude_difference": abs(
                report.comparison_for(left, right).amplitude_difference.estimate
            ),
            "shape_distance": _shape_estimate(report, left, right),
        }
        for left_index, left in enumerate(GROUPS)
        for right in GROUPS[left_index + 1 :]
    }
    same_shape_max = max(metrics[_pair_name(*pair)]["shape_distance"] for pair in SAME_SHAPE_PAIRS)
    changed_shape_min = min(
        metrics[_pair_name(*pair)]["shape_distance"] for pair in CHANGED_SHAPE_PAIRS
    )
    reference_level = report.comparison_for("reference", "level_shift")
    reference_amplitude = report.comparison_for("reference", "amplitude_shift")
    return {
        "seed": seed,
        "metrics": metrics,
        "component_structure_recovered": same_shape_max < changed_shape_min,
        "level_shift_is_largest_reference_level_contrast": abs(
            reference_level.level_difference.estimate
        )
        > max(
            abs(report.comparison_for("reference", "amplitude_shift").level_difference.estimate),
            abs(report.comparison_for("reference", "shape_change").level_difference.estimate),
        ),
        "amplitude_shift_is_largest_reference_amplitude_contrast": abs(
            reference_amplitude.amplitude_difference.estimate
        )
        > max(
            abs(report.comparison_for("reference", "level_shift").amplitude_difference.estimate),
            abs(report.comparison_for("reference", "shape_change").amplitude_difference.estimate),
        ),
        "level_interval_excludes_zero": not (
            reference_level.level_difference.lower <= 0 <= reference_level.level_difference.upper
        ),
        "amplitude_interval_excludes_zero": not (
            reference_amplitude.amplitude_difference.lower
            <= 0
            <= reference_amplitude.amplitude_difference.upper
        ),
    }


def run(
    *, repetitions: int = 20, seed: int = 8_119, bootstrap_resamples: int = 300
) -> dict[str, Any]:
    """Aggregate matched repetitions and retain the singleton-lab boundary."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    runs = [
        experiment(
            seed=seed + repeat * 10,
            bootstrap_resamples=bootstrap_resamples,
        )
        for repeat in range(repetitions)
    ]
    pair_names = tuple(runs[0]["metrics"])
    mean_metrics = {
        pair: {
            metric: float(np.mean([result["metrics"][pair][metric] for result in runs]))
            for metric in runs[0]["metrics"][pair]
        }
        for pair in pair_names
    }
    recovery_rates = {
        name: float(np.mean([result[name] for result in runs]))
        for name in (
            "component_structure_recovered",
            "level_shift_is_largest_reference_level_contrast",
            "amplitude_shift_is_largest_reference_amplitude_contrast",
            "level_interval_excludes_zero",
            "amplitude_interval_excludes_zero",
        )
    }
    singleton_panel = ParameterTrajectoryPanel(
        grid=np.asarray([0.0, 1.0]),
        values=np.zeros((9, 2)),
        subjects=tuple(f"mouse-{index}" for index in range(9)),
        groups=tuple(f"lab-{index}" for index in range(9)),
        clock_name="aligned_session",
        parameter_name="design_audit_only",
    )
    singleton_audit = audit_trajectory_replication(singleton_panel)
    holdout = _complete_lab_holdout_contract()
    contract = {
        **{name: rate == 1.0 for name, rate in recovery_rates.items()},
        "singleton_labs_rejected": not singleton_audit.inferentially_ready
        and len(singleton_audit.singleton_groups) == 9,
        "complete_lab_holdout": holdout["all_rows_tested_once"]
        and holdout["complete_subject_trajectories"]
        and holdout["held_out_subjects_per_lab"] == [10, 10, 10, 10],
    }
    return {
        "benchmark": "cross-lab trajectory component recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "labs": len(GROUPS),
            "subjects_per_lab": 10,
            "common_grid_positions": 9,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_unit": "subject within fixed lab",
        },
        "generating_components": {
            "reference": "linear rise",
            "level_shift": "same rise plus a constant",
            "amplitude_shift": "same centered rise at twice the amplitude",
            "shape_change": "one complete sinusoidal cycle",
        },
        "mean_pairwise_metrics": mean_metrics,
        "recovery_rates": recovery_rates,
        "complete_lab_holdout": holdout,
        "singleton_lab_audit": singleton_audit.to_dict(),
        "contract": contract,
        "contract_passed": all(contract.values()),
    }


def _shape_estimate(report: Any, left: str, right: str) -> float:
    interval = report.comparison_for(left, right).shape_distance
    if interval is None:
        raise RuntimeError("non-flat benchmark curves must have resolved shape")
    return interval.estimate


def _complete_lab_holdout_contract() -> dict[str, Any]:
    subjects = tuple(f"{group}-{index:02d}" for group in GROUPS for index in range(10))
    subject_groups = tuple(group for group in GROUPS for _ in range(10))
    grid_positions = range(9)
    study = Study(
        {
            "subject": [subject for subject in subjects for _grid_position in grid_positions],
            "session": [
                f"{subject}-session-{grid_position}"
                for subject in subjects
                for grid_position in grid_positions
            ],
            "trial": [0] * (len(subjects) * len(grid_positions)),
            "session_order": list(grid_positions) * len(subjects),
            "lab": [group for group in subject_groups for _grid_position in grid_positions],
        }
    )
    splits = leave_one_lab_out_splits(study)
    test_rows = np.concatenate([split.test_indices for split in splits])
    complete_subjects = True
    for split in splits:
        for subject in split.test_subjects:
            rows = split.test_indices[study["subject"][split.test_indices] == subject]
            if len(rows) != len(grid_positions):
                complete_subjects = False
    return {
        "folds": len(splits),
        "held_out_subjects_per_lab": sorted(len(split.test_subjects) for split in splits),
        "all_rows_tested_once": bool(np.array_equal(np.sort(test_rows), np.arange(len(study)))),
        "complete_subject_trajectories": complete_subjects,
    }


def _pair_name(left: str, right: str) -> str:
    order = {group: index for index, group in enumerate(GROUPS)}
    return f"{left}__{right}" if order[left] < order[right] else f"{right}__{left}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8_119)
    parser.add_argument("--bootstrap-resamples", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        repetitions=args.repetitions,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    rendered = render(result, allow_nan=True)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
