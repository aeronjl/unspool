from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from behavio import BernoulliHistoryGLM, Study, cohort_forward_session_splits, nested_select_model
from behavio.compose import hierarchical, smooth
from behavio.evaluate import leave_one_lab_out_session_forecast_splits
from benchmarks.ibl2021_nested_selection.benchmark import (
    CANDIDATES,
    _candidates,
    fit_nested_procedures,
)

pytestmark = pytest.mark.benchmark

RESULT_PATH = Path("benchmarks/ibl2021_nested_selection/result.json")


def _panel() -> Study:
    subjects = tuple(f"mouse-{index}" for index in range(6))
    labs = {subject: f"lab-{index // 2}" for index, subject in enumerate(subjects)}
    n_sessions = 6
    n_trials = 16
    design = Study(
        {
            "subject": [
                subject
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "session": [
                f"{subject}-{session}"
                for subject in subjects
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions * len(subjects),
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "lab": [
                labs[subject]
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "stimulus": np.random.default_rng(61).normal(
                size=len(subjects) * n_sessions * n_trials
            ),
        }
    )
    generator = hierarchical(
        smooth(
            BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02),
            over="session_order",
            knots=(0.0, 2.0, 5.0),
            smoothness=3.0,
        ),
        over="subject",
        scale=0.4,
    )
    return generator.simulate(
        design,
        generator.model.parameters_from_paths(
            {
                "intercept": [-0.3, -0.1, 0.2],
                "stimulus": [0.4, 1.0, 1.8],
                "choice_lag_1": [0.2, 0.2, 0.2],
            }
        ),
        seed=62,
    )


def test_nested_procedures_match_inner_validation_to_each_outer_estimand() -> None:
    within, transfer = fit_nested_procedures(
        _panel(),
        bootstrap_resamples=20,
        inner_bootstrap_resamples=10,
        bootstrap_seed=63,
    )

    assert within.candidate_names == CANDIDATES
    assert len(within.folds) == 1
    assert len(within.folds[0].inner_report.splits) == 2
    assert transfer.candidate_names == CANDIDATES
    assert len(transfer.folds) == 3
    assert all(len(fold.inner_report.splits) == 2 for fold in transfer.folds)
    assert all(
        fold.outer_split.held_out_group
        not in {split.held_out_group for split in fold.inner_report.splits}
        for fold in transfer.folds
    )
    assert within.audit_status.value in {"pass", "warning"}
    assert transfer.audit_status.value in {"pass", "warning"}


def test_within_selection_cannot_see_outer_position_five_choices() -> None:
    panel = _panel()
    outer = cohort_forward_session_splits(panel, min_train_sessions=5)

    def inner(training: Study):
        return cohort_forward_session_splits(training, min_train_sessions=3)

    first = nested_select_model(
        _candidates(),
        panel,
        outer,
        inner,
        bootstrap_resamples=20,
        inner_bootstrap_resamples=10,
        bootstrap_seed=64,
    )
    changed = _flip_choices(panel, outer[0].test_indices)
    second = nested_select_model(
        _candidates(),
        changed,
        outer,
        inner,
        bootstrap_resamples=20,
        inner_bootstrap_resamples=10,
        bootstrap_seed=64,
    )

    assert first.folds[0].selected_model == second.folds[0].selected_model
    assert first.folds[0].inner_report.to_dict() == second.folds[0].inner_report.to_dict()
    assert first.unit_balanced_log_loss != second.unit_balanced_log_loss


def test_lab_selection_cannot_see_outer_lab_or_position_five_choices() -> None:
    panel = _panel()
    outer = leave_one_lab_out_session_forecast_splits(panel, train_session_count=5)[:1]

    def inner(training: Study):
        return leave_one_lab_out_session_forecast_splits(training, train_session_count=4)

    first = nested_select_model(
        _candidates(),
        panel,
        outer,
        inner,
        bootstrap_resamples=20,
        inner_bootstrap_resamples=10,
        bootstrap_seed=65,
    )
    changed = _flip_choices(panel, outer[0].test_indices)
    second = nested_select_model(
        _candidates(),
        changed,
        outer,
        inner,
        bootstrap_resamples=20,
        inner_bootstrap_resamples=10,
        bootstrap_seed=65,
    )

    assert first.folds[0].selected_model == second.folds[0].selected_model
    assert first.folds[0].inner_report.to_dict() == second.folds[0].inner_report.to_dict()
    assert first.unit_balanced_log_loss != second.unit_balanced_log_loss


def test_committed_result_pins_training_only_selection_and_outer_scores() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["source"]["fixed_comparison_sha256"] == (
        "3d2fc25771369d60955752aef4fcd731b4936985c399f5d08a6cc2f9008daf7c"
    )
    assert result["panel"]["subjects"] == 78
    assert result["panel"]["labs"] == 9
    assert result["panel"]["trials"] == 46_152
    assert result["contract_passed"]
    assert all(result["contract"].values())

    within = result["within_subject_future_session"]
    transfer = result["held_out_lab_future_session"]
    assert within["selection_counts"] == {
        "static": 0,
        "drift_smoothness_1": 0,
        "drift_smoothness_3": 0,
        "drift_smoothness_9": 1,
    }
    assert transfer["selection_counts"] == {
        "static": 0,
        "drift_smoothness_1": 0,
        "drift_smoothness_3": 0,
        "drift_smoothness_9": 9,
    }
    assert within["subject_balanced_log_loss"] == pytest.approx(0.5472312174454272)
    assert transfer["subject_balanced_log_loss"] == pytest.approx(0.5971344039731026)
    assert all(
        candidate["audit_status"] == "pass"
        for report in (within, transfer)
        for fold in report["folds"]
        for candidate in fold["inner_candidates"].values()
    )
    assert all(
        fold["selected_outer_fit_audit"]["status"] == "pass"
        for report in (within, transfer)
        for fold in report["folds"]
    )

    within_fixed = within["comparison_with_fixed_candidates"]
    assert within_fixed["fixed_drift_smoothness_3"]["estimate"] == pytest.approx(
        -0.007679897821400701
    )
    assert within_fixed["fixed_drift_smoothness_3"]["interval_95"][1] < 0
    transfer_fixed = transfer["comparison_with_fixed_candidates"]
    assert transfer_fixed["fixed_drift_smoothness_3"]["estimate"] == pytest.approx(
        -0.0077734688112012255
    )
    assert transfer_fixed["fixed_drift_smoothness_3"]["interval_95"][1] < 0
    assert (
        transfer_fixed["fixed_static"]["interval_95"][0]
        < 0
        < transfer_fixed["fixed_static"]["interval_95"][1]
    )


def _flip_choices(study: Study, positions: np.ndarray) -> Study:
    columns = {name: study[name] for name in study.columns}
    choices = np.array(study["choice"], copy=True)
    choices[positions] = 1 - choices[positions]
    columns["choice"] = choices
    return Study(columns)
