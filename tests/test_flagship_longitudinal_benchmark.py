from __future__ import annotations

import json
from pathlib import Path

import pytest

from behavio import Study
from benchmarks.flagship_longitudinal.benchmark import (
    KNOTS,
    METHODS,
    analyze_panel,
    build_cell_panel,
    build_ibl_panel,
)

RESULT_PATH = Path("benchmarks/flagship_longitudinal/result.json")


def _source_study(*, ibl: bool) -> Study:
    subjects = ("mouse-a", "mouse-b")
    sessions = range(7)
    trials = range(12)
    columns: dict[str, list[object]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "session_order": [],
        "choice": [],
    }
    if ibl:
        columns.update(signed_stimulus=[], lab=[], phase=[], task_protocol=[])
    else:
        columns.update(paper_session_order=[], stimulus_side=[])
    for subject_index, subject in enumerate(subjects):
        for session in sessions:
            for trial in trials:
                stimulus = (-1, 0, 1)[trial % 3]
                choice = int(stimulus + (trial % 2) > 0)
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-session-{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                if ibl:
                    columns["choice"].append(0 if trial == 0 else -1 if choice else 1)
                    columns["signed_stimulus"].append(float(stimulus))
                    columns["lab"].append(f"lab-{subject_index}")
                    columns["phase"].append("early" if session < 3 else "late_training")
                    columns["task_protocol"].append("trainingChoiceWorld")
                else:
                    columns["choice"].append(choice)
                    columns["paper_session_order"].append(session)
                    columns["stimulus_side"].append(stimulus)
    return Study(columns)


def test_cell_panel_selects_disjoint_early_and_late_sessions() -> None:
    panel = build_cell_panel(_source_study(ibl=False))

    assert len(panel) == 2 * 6 * 12
    assert set(panel["session_order"]) == set(range(6))
    assert set(panel["source_session_order"]) == {0, 1, 2, 4, 5, 6}
    assert set(panel["phase"]) == {"early", "late"}
    assert set(panel["stimulus"]) == {-1.0, 0.0, 1.0}


def test_ibl_panel_recodes_choice_and_drops_invalid_trials() -> None:
    panel = build_ibl_panel(_source_study(ibl=True))

    assert len(panel) == 2 * 6 * 11
    assert set(panel["choice"]) == {0, 1}
    assert set(panel["session_order"]) == set(range(6))
    assert set(panel["source_session_order"]) == {0, 1, 2, 4, 5, 6}
    assert "source_phase" in panel.columns
    assert "lab" in panel.columns


def test_flagship_analysis_is_prospective_subject_balanced_and_audited() -> None:
    panel = build_cell_panel(_source_study(ibl=False))

    result = analyze_panel(panel, bootstrap_seed=41, bootstrap_resamples=100)

    assert result["panel"]["n_subjects"] == 2
    assert result["panel"]["n_training_trials"] == 2 * 5 * 12
    assert result["panel"]["n_test_trials"] == 2 * 12
    assert result["winner_by_subject_balanced_log_loss"] in METHODS
    assert tuple(result["methods"]) == METHODS
    assert len(result["pairwise_subject_log_loss_differences"]) == 6
    for metrics in result["methods"].values():
        assert metrics["subject_balanced_log_loss"] > 0
        assert len(metrics["subject_log_loss"]) == 2
        assert metrics["fit_audit"]["status"] in {"pass", "warning"}
        assert metrics["stimulus_trajectory"]["knots"] == list(KNOTS)
        assert len(metrics["stimulus_trajectory"]["subject_values"]) == 2
        lower, upper = metrics["subject_bootstrap_log_loss_95_interval"]
        assert lower <= upper


def test_committed_flagship_result_retains_both_dataset_contracts() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["analysis_contract"]["model_order"] == list(METHODS)
    assert result["analysis_contract"]["knots"] == list(KNOTS)
    assert set(result["datasets"]) == {"cell2025", "ibl2021"}
    expected = {
        "cell2025": (30, 33_814, "static_partial_pooling", 0.5411518211140368),
        "ibl2021": (9, 28_097, "shared_smooth_drift", 0.6310282675624881),
    }
    assert result["sources"]["cell2025"]["member_sha256"] == (
        "94a6d541bfde731f769e02a68dbc652ab5b73dbc1ec13b8b7c8100d181b8048a"
    )
    assert result["sources"]["ibl2021"]["manifest_sha256"] == (
        "63ac8b40b35ea21bc036cbdb4819f2dc04b448c803f804361dc9e40768aa32a0"
    )
    for name, dataset in result["datasets"].items():
        n_subjects, n_trials, winner, winning_loss = expected[name]
        assert dataset["panel"]["n_subjects"] == n_subjects
        assert dataset["panel"]["n_trials"] == n_trials
        assert dataset["panel"]["n_sessions"] == 6 * dataset["panel"]["n_subjects"]
        assert dataset["winner_by_subject_balanced_log_loss"] == winner
        assert dataset["methods"][winner]["subject_balanced_log_loss"] == pytest.approx(
            winning_loss, rel=1e-9
        )
        assert set(dataset["methods"]) == set(METHODS)
        assert all(
            method["fit_audit"]["status"] != "fail" for method in dataset["methods"].values()
        )
