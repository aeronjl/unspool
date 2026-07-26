from benchmarks.recovery_grid.benchmark import run


def test_four_family_recovery_grid_satisfies_its_bounded_contract() -> None:
    result = run()

    assert result["contract_passed"]
    assert result["candidate_labels"] == ["static", "smooth", "hmm", "q-learning"]
    sparse, dense = result["designs"]
    assert sparse["n_trials"] == 150
    assert dense["n_trials"] == 300
    assert sparse["design_seed"] != dense["design_seed"]
    assert sparse["n_folds"] == [2, 2, 2, 2]
    assert dense["n_folds"] == [2, 2, 2, 2]
    assert sparse["overall_accuracy"] == 0.5
    assert dense["overall_accuracy"] == 1.0
    assert sparse["audit_failure_rate"] == 0.0
    assert dense["audit_failure_rate"] == 0.0
