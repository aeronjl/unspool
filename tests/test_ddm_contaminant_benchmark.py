import json
from pathlib import Path

from benchmarks.ddm_contaminants.benchmark import SHARED_PARAMETERS, run


def test_pinned_contaminant_benchmark_retains_paired_recovery_and_forecasts() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "ddm_contaminants" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    summary = result["summary"]

    assert result["scored_columns"] == ["choice", "response_time"]
    assert result["repeats"] == 20
    assert summary["n_paired_eligible"] == 20
    assert summary["contaminant_probability_rmse"] < 0.02
    assert summary["future_mean_log_loss"]["robust_better_runs"] == 20
    assert summary["future_mean_log_loss"]["naive_minus_robust"] > 0
    assert all(
        summary["parameter_rmse"][parameter]["robust_better"] for parameter in SHARED_PARAMETERS
    )
    assert summary["mean_responsibility"]["true_contaminants"] > 0.5
    assert summary["mean_responsibility"]["ordinary_trials"] < 0.05
    assert all(row["robust"]["audit"]["status"] == "pass" for row in result["runs"])


def test_contaminant_benchmark_is_reproducible_at_small_scale() -> None:
    first = run(repeats=2, seed=501)
    second = run(repeats=2, seed=501)

    assert first == second
    assert first["summary"]["n_paired_eligible"] == 2
