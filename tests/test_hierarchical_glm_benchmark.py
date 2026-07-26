import json
from pathlib import Path

from benchmarks.hierarchical_glm.benchmark import METHODS, run

RESULT_PATH = Path("benchmarks/hierarchical_glm/result.json")


def test_hierarchical_benchmark_is_reproducible_and_matched() -> None:
    first = run(repetitions=1, seed=100)
    second = run(repetitions=1, seed=100)

    assert first == second
    assert first["scope"]["subject_scale"].startswith("fixed")
    assert set(first["regimes"]) == {"0.1", "0.5", "1.0"}
    for regime in first["regimes"].values():
        assert tuple(regime["methods"]) == METHODS
        for metrics in regime["methods"].values():
            assert metrics["mean_subject_coefficient_rmse"] > 0
            assert metrics["mean_prospective_log_loss"] > 0
            assert metrics["all_fits_converged"]


def test_committed_result_supports_its_bounded_claim() -> None:
    result = json.loads(RESULT_PATH.read_text())

    assert result["repetitions"] == 20
    assert result["partial_pooling_wins"] == {
        "mean_subject_coefficient_rmse": 3,
        "mean_prospective_log_loss": 3,
    }
    for regime in result["regimes"].values():
        assert all(metrics["all_fits_converged"] for metrics in regime["methods"].values())
