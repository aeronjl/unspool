import json
from pathlib import Path

from benchmarks.landmark_uncertainty.benchmark import experiment

RESULT_PATH = Path("benchmarks/landmark_uncertainty/result.json")


def test_landmark_uncertainty_experiment_is_reproducible() -> None:
    first = experiment(regime="decisive", seed=43, n_resamples=50)
    second = experiment(regime="decisive", seed=43, n_resamples=50)

    assert first == second
    assert first["point_resolved"]
    assert first["bootstrap_resolution_rate"] > 0.9
    assert first["point_matches_transform"]
    assert first["final_clock_resolution_rate"] == first["bootstrap_resolution_rate"]


def test_committed_landmark_result_retains_the_resolution_boundary() -> None:
    result = json.loads(RESULT_PATH.read_text())
    decisive = result["regimes"]["decisive"]
    marginal = result["regimes"]["marginal"]

    assert result["repetitions"] == 30
    assert result["n_resamples_per_fit"] == 200
    assert result["contract_passed"]
    assert all(result["contract"].values())
    assert decisive["point_resolution_rate"] > marginal["point_resolution_rate"]
    assert decisive["mean_bootstrap_resolution_rate"] > marginal["mean_bootstrap_resolution_rate"]
    assert (
        decisive["mean_interval_width_when_resolved"]
        < marginal["mean_interval_width_when_resolved"]
    )
