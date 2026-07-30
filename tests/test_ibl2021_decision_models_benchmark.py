from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.ibl2021_decision_models.benchmark import selected_manifest_rows
from benchmarks.ibl2021_replicated.manifest import EXPECTED_MANIFEST_SHA256

pytestmark = pytest.mark.benchmark

RESULT_PATH = Path("benchmarks/ibl2021_decision_models/result.json")


def test_subject_selection_is_outcome_blind_and_complete() -> None:
    subject, rows = selected_manifest_rows()

    assert subject == "CSHL045"
    assert len(rows) == 6
    assert [row["window_position"] for row in rows] == list(range(6))
    forbidden = {"choice", "feedbackType", "response_time", "accuracy"}
    assert not forbidden.intersection(set().union(*(set(row) for row in rows)))


def test_committed_decision_model_result_retains_prospective_boundaries() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["contract_passed"]
    assert all(result["contract"].values())
    assert result["source"]["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert result["source"]["subject"] == "CSHL045"
    assert result["eligibility"]["source_rows_after_cap"] == {
        str(position): 150 for position in range(6)
    }
    assert result["eligibility"]["response_time_validity_seconds"] == [0.05, 3.0]
    assert result["ddm"]["train_positions"] == [3, 4]
    assert result["ddm"]["test_position"] == 5
    assert result["ddm"]["scored_columns"] == ["choice", "response_time"]
    assert result["glm_hmm"]["selection"]["inner_train_positions"] == [0, 1, 2, 3]
    assert result["glm_hmm"]["selection"]["selection_position"] == 4
    assert result["glm_hmm"]["outer_test_position"] == 5
    assert result["glm_hmm"]["scored_columns"] == ["choice"]
    assert all(
        payload["fit_audit"]["status"] != "fail"
        for payload in (
            result["ddm"]["naive"],
            result["ddm"]["robust"],
            result["glm_hmm"]["static_glm"],
            result["glm_hmm"]["selected_glm_hmm"],
        )
    )


def test_committed_decision_model_evidence_is_row_aligned() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    ddm = result["ddm"]
    hmm = result["glm_hmm"]

    assert ddm["n_test"] == 111
    assert all(
        len(values) == ddm["n_test"]
        for values in ddm["heldout"].values()
        if isinstance(values, list) and values and not isinstance(values[0], dict)
    )
    assert hmm["n_outer_test"] == 150
    assert len(hmm["heldout"]["filtered_state_probability"]) == hmm["n_outer_test"]
    assert all(
        len(probability) == hmm["selection"]["selected_states"]
        for probability in hmm["heldout"]["filtered_state_probability"]
    )
