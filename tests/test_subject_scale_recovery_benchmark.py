import json
from pathlib import Path

import pytest

from benchmarks.subject_scale_recovery.benchmark import experiment

RESULT_PATH = Path("benchmarks/subject_scale_recovery/result.json")


def test_subject_scale_experiment_is_reproducible() -> None:
    first = experiment(n_subjects=8, true_scale=0.5, seed=71)
    second = experiment(n_subjects=8, true_scale=0.5, seed=71)

    assert first == second
    assert first["fit_converged"]
    assert first["scale_estimate"] == pytest.approx(0.5, abs=0.25)
    assert first["estimated_prospective_log_loss"] > 0
    assert first["estimated_brier_score"] > 0


def test_committed_scale_recovery_result_retains_calibration_evidence() -> None:
    result = json.loads(RESULT_PATH.read_text())

    assert result["repetitions"] == 20
    assert set(result["regimes"]) == {
        f"subjects={n_subjects},scale={scale}"
        for n_subjects in (8, 24)
        for scale in (0.1, 0.5, 1.0)
    }
    assert all(result["more_subjects_reduce_rmse"].values())
    for regime in result["regimes"].values():
        assert regime["convergence_rate"] == 1.0
        assert 0.0 <= regime["coverage_95"] <= 1.0
        assert 0.0 <= regime["boundary_rate"] <= 1.0
