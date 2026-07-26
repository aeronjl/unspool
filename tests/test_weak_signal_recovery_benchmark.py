import json
from pathlib import Path

from benchmarks.weak_signal_recovery.benchmark import run

RESULT_PATH = Path("benchmarks/weak_signal_recovery/result.json")


def test_weak_signal_benchmark_exposes_boundary_ambiguity() -> None:
    result = run(repeats=1)

    assert result["contract_passed"]
    assert result["candidate_labels"] == ["static", "smooth", "hmm", "q-learning"]
    assert result["n_trials"] == 300
    assert len(result["scenario_matrix"]["scenario_names"]) == 8
    assert all(sum(row) == 1 for row in result["scenario_matrix"]["counts"])
    assert (
        result["signal_levels"]["strong"]["accuracy"] > result["signal_levels"]["weak"]["accuracy"]
    )
    json.dumps(result)


def test_committed_weak_signal_result_retains_repeated_evidence() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["contract_passed"]
    assert result["repeats"] == 10
    assert result["signal_levels"]["strong"]["correct"] == 28
    assert result["signal_levels"]["weak"]["correct"] == 13
    assert result["audit_failure_rate"] == 0.0
    assert len(result["run_seeds"]) == 80
