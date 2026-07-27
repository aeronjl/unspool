from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ibl2021_replicated.manifest import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SUBJECTS_PER_LAB,
    load_manifest,
    sources_from_manifest,
)

RESULT_PATH = Path("benchmarks/ibl2021_replicated/result.json")


def test_replicated_manifest_is_exact_replicated_and_outcome_blind() -> None:
    manifest = load_manifest()
    rows = manifest["sessions"]

    assert manifest["sessions_sha256"] == EXPECTED_MANIFEST_SHA256
    assert manifest["subjects_per_lab"] == EXPECTED_SUBJECTS_PER_LAB
    assert manifest["n_subjects"] == 78
    assert manifest["n_labs"] == 9
    assert manifest["n_sessions"] == 468
    assert manifest["total_file_size"] == 21_465_187
    forbidden_outcomes = {"choice", "feedbackType", "reward", "accuracy"}
    manifest_fields = set().union(*(set(row) for row in rows))
    assert not forbidden_outcomes.intersection(manifest_fields)
    for subject in {row["subject"] for row in rows}:
        subject_rows = [row for row in rows if row["subject"] == subject]
        assert [row["window_position"] for row in subject_rows] == list(range(6))
        assert [row["phase"] for row in subject_rows] == ["early"] * 3 + ["late_training"] * 3


def test_manifest_builds_checksum_pinned_one_sources_without_recasting_choices() -> None:
    sources = sources_from_manifest()

    assert len(sources) == 468
    assert len({source.dataset_id for source in sources}) == 468
    assert all(
        source.columns == ("contrastLeft", "contrastRight", "feedbackType", "choice")
        for source in sources
    )
    assert all(source.column_map["choice"] == "source_choice" for source in sources)
    assert all(source.column_map["feedbackType"] == "source_feedback" for source in sources)


def test_committed_replicated_result_pins_public_data_and_design_boundaries() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["contract_passed"]
    assert all(result["contract"].values())
    assert result["selection"]["outcome_blind"] is True
    assert result["selection"]["subjects_per_lab"] == EXPECTED_SUBJECTS_PER_LAB
    assert result["study"] == {
        "choice_trials": 259_651,
        "easy_trials": 134_454,
        "source_choice_counts": {"-1": 128_470, "0": 1_182, "1": 131_181},
        "source_datasets": 468,
        "trials": 260_833,
    }
    assert result["overall"]["subjects_with_positive_change"] == 78
    assert result["overall"]["mean_accuracy_change"] == 0.4228135684945791
    assert result["population_validation"]["leave_one_lab_out_folds"] == 9
    assert result["population_validation"]["held_out_subjects"] == EXPECTED_SUBJECTS_PER_LAB
    assert result["clock"]["name"] == "endpoint_window_position"
    assert "not uniform elapsed training time" in result["clock"]["interpretation"]
