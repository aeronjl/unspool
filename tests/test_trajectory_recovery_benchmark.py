import json
from pathlib import Path

import pytest

from benchmarks.trajectory_recovery.benchmark import EXPECTED_WINNER, METHODS, experiment

pytestmark = pytest.mark.benchmark

RESULT_PATH = Path("benchmarks/trajectory_recovery/result.json")


def test_trajectory_experiment_is_reproducible_and_matched() -> None:
    first = experiment(regime="individual_drift", seed=91)
    second = experiment(regime="individual_drift", seed=91)

    assert first == second
    assert tuple(first) == METHODS
    for metrics in first.values():
        assert metrics["subject_trajectory_rmse"] > 0
        assert metrics["prospective_log_loss"] > 0
        assert metrics["all_fits_converged"]


def test_committed_result_recovers_each_generating_structure() -> None:
    result = json.loads(RESULT_PATH.read_text())

    assert result["repetitions"] == 20
    assert result["all_expected_winners_recovered"]
    for regime, expected in EXPECTED_WINNER.items():
        summary = result["regimes"][regime]
        assert summary["expected_winner"] == expected
        assert summary["trajectory_rmse_winner"] == expected
        assert summary["prospective_log_loss_winner"] == expected
        assert all(metrics["all_fits_converged"] for metrics in summary["methods"].values())
