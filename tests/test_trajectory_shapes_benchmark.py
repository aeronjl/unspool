from __future__ import annotations

import json
from pathlib import Path

from benchmarks.trajectory_shapes.benchmark import experiment

RESULT_PATH = Path("benchmarks/trajectory_shapes/result.json")


def test_component_recovery_experiment_is_reproducible() -> None:
    first = experiment(seed=71, subjects_per_lab=6, bootstrap_resamples=50)
    second = experiment(seed=71, subjects_per_lab=6, bootstrap_resamples=50)

    assert first == second
    assert first["component_structure_recovered"]
    assert first["level_shift_is_largest_reference_level_contrast"]
    assert first["amplitude_shift_is_largest_reference_amplitude_contrast"]


def test_committed_result_retains_recovery_and_identifiability_boundary() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["repetitions"] == 20
    assert result["contract_passed"]
    assert all(result["contract"].values())
    assert all(rate == 1.0 for rate in result["recovery_rates"].values())
    holdout = result["complete_lab_holdout"]
    assert holdout["folds"] == 4
    assert holdout["held_out_subjects_per_lab"] == [10, 10, 10, 10]
    assert holdout["all_rows_tested_once"]
    assert holdout["complete_subject_trajectories"]
    audit = result["singleton_lab_audit"]
    assert not audit["inferentially_ready"]
    assert len(audit["singleton_groups"]) == 9
