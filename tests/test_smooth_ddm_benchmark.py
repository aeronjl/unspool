import json
import math
from pathlib import Path

import pytest

from benchmarks.smooth_ddm.benchmark import EXPECTED_WINNER, METHODS, experiment

pytestmark = pytest.mark.benchmark


def test_smooth_ddm_experiment_is_reproducible_and_matched() -> None:
    first = experiment(regime="changing", seed=91)
    second = experiment(regime="changing", seed=91)

    assert first == second
    assert tuple(first["methods"]) == METHODS
    for metrics in first["methods"].values():
        assert metrics["training_trajectory_rmse"] > 0
        assert math.isfinite(metrics["future_log_loss"])
        assert metrics["converged"]


def test_pinned_smooth_ddm_result_separates_stationarity_from_change() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "smooth_ddm" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["repetitions"] == 20
    assert result["all_expected_winners_recovered"]
    for regime, expected in EXPECTED_WINNER.items():
        summary = result["regimes"][regime]
        assert summary["expected_winner"] == expected
        assert summary["trajectory_rmse_winner"] == expected
        assert summary["future_log_loss_winner"] == expected
        assert summary["n_paired_eligible"] == 20
        assert all(metrics["all_fits_converged"] for metrics in summary["methods"].values())
