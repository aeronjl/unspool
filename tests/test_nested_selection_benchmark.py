import json
from pathlib import Path

import pytest

from benchmarks.nested_selection.benchmark import CANDIDATES, EXPECTED, REGIMES, experiment

pytestmark = pytest.mark.benchmark

RESULT_PATH = Path("benchmarks/nested_selection/result.json")


def test_nested_selection_experiment_is_reproducible_and_audited() -> None:
    first = experiment(regime="shared_drift", seed=103)
    second = experiment(regime="shared_drift", seed=103)

    assert first == second
    assert first["expected_model"] == "shared_smooth"
    assert len(first["outer_fold_selections"]) == 2
    assert first["inner_fold_counts"] == [2, 3]
    assert first["all_selected_fits_passed_audit"]
    assert set(first["fixed_candidate_outer_log_loss"]) == set(CANDIDATES)


def test_committed_nested_selection_result_retains_resolution_limits() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["repetitions"] == 20
    assert result["selection_contract"]["outer_test_outcomes_available_during_selection"] is False
    assert result["selection_contract"]["candidate_order"] == list(CANDIDATES)
    assert set(result["regimes"]) == set(REGIMES)
    assert result["all_regimes_majority_recovered"]
    assert result["regimes"]["stationary"]["expected_model"] == EXPECTED["stationary"]
    assert result["regimes"]["stationary"]["expected_selection_rate"] == pytest.approx(0.925)
    assert result["regimes"]["stationary"]["majority_recovery_rate"] == pytest.approx(0.85)
    assert result["regimes"]["shared_drift"]["expected_model"] == EXPECTED["shared_drift"]
    assert result["regimes"]["shared_drift"]["expected_selection_rate"] == pytest.approx(1.0)
    assert result["regimes"]["shared_drift"]["majority_recovery_rate"] == pytest.approx(1.0)
    assert all(regime["all_selected_fits_passed_audit"] for regime in result["regimes"].values())
