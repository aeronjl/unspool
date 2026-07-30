import json
from pathlib import Path

import pytest

from benchmarks.hierarchical_smooth_ddm.benchmark import EXPECTED_WINNER, METHODS, experiment

pytestmark = pytest.mark.benchmark


def test_hierarchical_smooth_ddm_experiment_is_reproducible_and_matched() -> None:
    first = experiment(regime="individual_change", seed=91)
    second = experiment(regime="individual_change", seed=91)

    assert first == second
    assert tuple(first["methods"]) == METHODS
    for metrics in first["methods"].values():
        assert metrics["subject_trajectory_rmse"] > 0
        assert metrics["all_fits_converged"]
        assert all(audit["status"] != "fail" for audit in metrics["audits"])


def test_pinned_hierarchical_smooth_ddm_result_recovers_each_structure() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "hierarchical_smooth_ddm" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["repetitions"] == 20
    assert result["all_expected_winners_recovered"]
    for regime, expected in EXPECTED_WINNER.items():
        summary = result["regimes"][regime]
        assert summary["expected_winner"] == expected
        assert summary["trajectory_rmse_winner"] == expected
        assert summary["future_log_loss_winner"] == expected
        assert summary["n_complete_eligible"] == 20
