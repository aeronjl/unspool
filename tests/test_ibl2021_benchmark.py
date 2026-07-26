from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.ibl2021.benchmark import (
    build_study,
    calculate_session_metrics,
    population_validation_summary,
)
from benchmarks.ibl2021.fetch_data import EXPECTED_MANIFEST_SHA256, load_manifest
from benchmarks.ibl2021.selection import manifest_digest, select_learning_panel

RESULT_PATH = Path("benchmarks/ibl2021/result.json")


def _subject_sessions(lab: str, subject: str, n_training: int, n_biased: int = 2) -> list[dict]:
    rows = []
    for order in range(n_training + n_biased):
        biased = order >= n_training
        rows.append(
            {
                "id": f"{lab}-{subject}-{order:02d}",
                "lab": lab,
                "subject": subject,
                "start_time": f"2026-01-{order + 1:02d}T10:00:00",
                "number": 1,
                "task_protocol": "_iblrig_tasks_"
                + ("biasedChoiceWorld" if biased else "trainingChoiceWorld"),
            }
        )
    return rows


def test_selection_is_disjoint_and_balanced_by_lab() -> None:
    sessions = [
        *_subject_sessions("lab-a", "short", 6),
        *_subject_sessions("lab-a", "long", 8),
        *_subject_sessions("lab-b", "only", 7),
        *_subject_sessions("lab-c", "ineligible", 5),
    ]
    available = {row["id"] for row in sessions}

    selected = select_learning_panel(sessions, available)

    assert {row["lab"] for row in selected} == {"lab-a", "lab-b"}
    assert {row["subject"] for row in selected if row["lab"] == "lab-a"} == {"long"}
    assert len(selected) == 12
    for subject in ("long", "only"):
        rows = [row for row in selected if row["subject"] == subject]
        early = {row["session"] for row in rows if row["phase"] == "early"}
        late = {row["session"] for row in rows if row["phase"] == "late_training"}
        assert len(early) == len(late) == 3
        assert early.isdisjoint(late)
        assert max(row["session_order"] for row in rows if row["phase"] == "early") < min(
            row["session_order"] for row in rows if row["phase"] == "late_training"
        )


def test_selection_ignores_sessions_without_trial_tables() -> None:
    sessions = _subject_sessions("lab-a", "mouse", 7)
    available = {row["id"] for row in sessions}
    available.remove("lab-a-mouse-03")

    selected = select_learning_panel(sessions, available)

    assert len(selected) == 6
    assert {row["n_training_sessions_before_transition"] for row in selected} == {6}


def test_manifest_digest_is_canonical_for_mapping_key_order() -> None:
    first = [{"session": "one", "md5": "abc"}]
    second = [{"md5": "abc", "session": "one"}]

    assert manifest_digest(first) == manifest_digest(second)


def test_committed_manifest_covers_nine_labs_with_disjoint_phases() -> None:
    manifest = load_manifest()
    rows = manifest["sessions"]

    assert manifest["sessions_sha256"] == EXPECTED_MANIFEST_SHA256
    assert len(rows) == 54
    assert len({row["lab"] for row in rows}) == 9
    assert len({row["subject"] for row in rows}) == 9
    for subject in {row["subject"] for row in rows}:
        subject_rows = [row for row in rows if row["subject"] == subject]
        assert [row["phase"] for row in subject_rows].count("early") == 3
        assert [row["phase"] for row in subject_rows].count("late_training") == 3
        assert len({row["session"] for row in subject_rows}) == 6


def test_committed_result_retains_population_holdout_coverage() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["contract_passed"]
    assert result["n_leave_subject_out_folds"] == 9
    assert result["n_leave_lab_out_folds"] == 9
    assert result["n_matching_subject_lab_partitions"] == 9
    assert result["n_lab_holdouts_with_multiple_subjects"] == 0


def _trials(rewards: list[int]) -> dict[str, Any]:
    n_trials = len(rewards)
    return {
        "choice": np.ones(n_trials, dtype=int),
        "feedbackType": np.where(rewards, 1, -1),
        "contrastRight": np.ones(n_trials),
        "contrastLeft": np.full(n_trials, np.nan),
        "response_times": np.arange(n_trials, dtype=float) + 1.5,
        "stimOn_times": np.arange(n_trials, dtype=float) + 1.0,
    }


def test_ibl_adapter_preserves_session_order_and_easy_accuracy() -> None:
    session_trials = [
        (
            {
                "subject": "mouse",
                "session": "late",
                "session_order": 8,
                "lab": "lab",
                "phase": "late_training",
                "task_protocol": "trainingChoiceWorld",
            },
            _trials([1, 1, 1, 0]),
        ),
        (
            {
                "subject": "mouse",
                "session": "early",
                "session_order": 0,
                "lab": "lab",
                "phase": "early",
                "task_protocol": "trainingChoiceWorld",
            },
            _trials([1, 0, 0, 0]),
        ),
    ]

    study = build_study(session_trials)
    metrics = calculate_session_metrics(study)

    assert len(study) == 8
    assert list(study["session"][:4]) == ["late"] * 4
    assert list(study["session"][study.chronological_indices()[:4]]) == ["early"] * 4
    assert [(row.session, row.easy_accuracy) for row in metrics] == [
        ("early", 0.25),
        ("late", 0.75),
    ]


def test_population_validation_summary_distinguishes_subject_and_lab_partitions() -> None:
    session_trials = [
        (
            {
                "subject": subject,
                "session": f"{subject}-session",
                "session_order": 0,
                "lab": lab,
                "phase": "early",
                "task_protocol": "trainingChoiceWorld",
            },
            _trials([1, 0]),
        )
        for subject, lab in (("a", "north"), ("b", "north"), ("c", "south"))
    ]

    summary = population_validation_summary(build_study(session_trials))

    assert summary == {
        "n_leave_subject_out_folds": 3,
        "n_leave_lab_out_folds": 2,
        "n_matching_subject_lab_partitions": 1,
        "n_lab_holdouts_with_multiple_subjects": 1,
    }
