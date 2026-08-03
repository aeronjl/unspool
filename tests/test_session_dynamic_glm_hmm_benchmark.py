import json
from pathlib import Path

import pytest

from benchmarks.session_dynamic_glm_hmm.benchmark import (
    COMPETITORS,
    REGIMES,
    experiment,
)

pytestmark = pytest.mark.benchmark

RESULT_PATH = Path("benchmarks/session_dynamic_glm_hmm/result.json")


def test_experiment_is_reproducible_and_keeps_selection_inside_training() -> None:
    first = experiment(regime="session_dynamic", seed=411)
    second = experiment(regime="session_dynamic", seed=411)

    assert first == second
    assert first["outer_train_session_orders"] == [0, 1, 2, 3, 4, 5]
    assert first["outer_test_session_orders"] == [6]
    assert first["inner_fold_count"] == 2
    assert set(first["prospective_log_loss"]) == {"selected_session_dynamic", *COMPETITORS}
    assert first["selected_partial_stage_converged"]
    assert first["selected_full_stage_converged"]


def test_committed_result_retains_the_negative_prospective_result_and_recovery() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["repetitions"] == 4
    assert not result["contract_passed"]
    assert set(result["regimes"]) == set(REGIMES)
    stationary = result["regimes"]["stationary"]
    assert stationary["prospective_win_counts"]["stationary_glm_hmm"] == 3
    assert not stationary["all_selected_partial_stages_converged"]
    dynamic = result["regimes"]["session_dynamic"]
    assert dynamic["mean_score_winner"] == "stationary_glm_hmm"
    assert dynamic["prospective_win_counts"]["selected_session_dynamic"] >= 1
    assert dynamic["selection_counts"]["K=2;sigma=0.35;alpha=30"] == 4
    assert dynamic["all_selected_partial_stages_converged"]
    assert dynamic["all_selected_full_stages_converged"]
    assert dynamic["mean_decoded_state_accuracy"] >= 0.7
    assert dynamic["n_aligned_path_recoveries"] == 4
