import json
from pathlib import Path

import pytest

from benchmarks.state_alignment.benchmark import experiment

pytestmark = pytest.mark.benchmark

RESULT_PATH = Path("benchmarks/state_alignment/result.json")


def test_state_alignment_experiment_is_reproducible_and_permutation_invariant() -> None:
    first = experiment(regime="clear", seed=37)
    second = experiment(regime="clear", seed=37)

    assert first == second
    assert first["fit_converged"]
    assert first["permutation_invariant"]
    assert first["decoded_accuracy"] > 0.8
    assert first["alignment_ambiguous"] is False


def test_committed_alignment_result_retains_the_identifiability_boundary() -> None:
    result = json.loads(RESULT_PATH.read_text())
    clear = result["regimes"]["clear"]
    overlapping = result["regimes"]["overlapping"]

    assert result["repetitions"] == 20
    assert result["contract_passed"]
    assert all(result["contract"].values())
    assert clear["convergence_rate"] == 1.0
    assert overlapping["convergence_rate"] == 1.0
    assert clear["permutation_invariance_rate"] == 1.0
    assert overlapping["permutation_invariance_rate"] == 1.0
    assert clear["mean_decoded_accuracy"] > overlapping["mean_decoded_accuracy"]
    assert clear["mean_score_gap"] > overlapping["mean_score_gap"]
    assert overlapping["alignment_ambiguity_rate"] > clear["alignment_ambiguity_rate"]
