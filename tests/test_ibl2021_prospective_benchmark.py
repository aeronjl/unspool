from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.ibl2021_prospective.benchmark import (
    METHODS,
    analyze_panel,
    build_panel,
)
from unspool import Study

RESULT_PATH = Path("benchmarks/ibl2021_prospective/result.json")


def _source_study() -> Study:
    columns: dict[str, list[object]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "session_order": [],
        "lab": [],
        "phase": [],
        "window_position": [],
        "source_choice": [],
        "contrastLeft": [],
        "contrastRight": [],
        "source_ibl_dataset_id": [],
    }
    for subject_index, subject in enumerate(("a", "b", "c", "d")):
        lab = "north" if subject_index < 2 else "south"
        for session in range(6):
            for trial in range(14):
                stimulus = (-1.0, -0.5, 0.5, 1.0)[trial % 4]
                rightward = trial % 5 < session
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session + subject_index)
                columns["lab"].append(lab)
                columns["phase"].append("early" if session < 3 else "late_training")
                columns["window_position"].append(session)
                columns["source_choice"].append(0 if trial == 1 else -1 if rightward else 1)
                columns["contrastLeft"].append(-stimulus if stimulus < 0 else np.nan)
                columns["contrastRight"].append(stimulus if stimulus > 0 else np.nan)
                columns["source_ibl_dataset_id"].append(f"dataset-{subject}-{session}")
    return Study(columns)


def test_panel_caps_source_rows_before_choice_filter_and_maps_semantics() -> None:
    panel = build_panel(_source_study(), trials_per_session=10)

    assert len(panel) == 4 * 6 * 9
    assert set(panel["session_order"]) == set(range(6))
    assert set(panel["source_choice"]) == {-1, 1}
    assert set(panel["choice"]) == {0, 1}
    assert np.array_equal(panel["choice"], (panel["source_choice"] < 0).astype(np.int8))
    assert set(panel["stimulus"]) == {-1.0, -0.5, 0.5, 1.0}


def test_panel_membership_and_features_do_not_depend_on_choice_direction() -> None:
    source = _source_study()
    changed = {name: source[name] for name in source.columns}
    changed_choice = np.array(source["source_choice"], copy=True)
    changed_choice[changed_choice != 0] *= -1
    changed["source_choice"] = changed_choice

    first = build_panel(source, trials_per_session=10)
    second = build_panel(Study(changed), trials_per_session=10)

    for name in ("subject", "session", "trial", "session_order", "lab", "stimulus"):
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(first["choice"], 1 - second["choice"])


def test_analysis_combines_future_session_and_unseen_lab_forecasts() -> None:
    panel = build_panel(_source_study(), trials_per_session=10)

    result = analyze_panel(panel, bootstrap_resamples=20, bootstrap_seed=51)

    assert result["contract_passed"]
    assert all(result["contract"].values())
    within = result["within_subject_future_session"]
    transfer = result["held_out_lab_future_session"]
    assert within["model_order"] == list(METHODS)
    assert len(within["folds"]) == 1
    assert transfer["model_order"] == list(METHODS)
    assert len(transfer["folds"]) == 2
    assert transfer["lab_balanced_subject_log_loss"]["labs"] == ["north", "south"]


def test_committed_result_pins_both_prospective_estimands() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["source"]["manifest_sha256"] == (
        "c0a45addbc14b936f6b3aaac0c06b6a4d7108725d82dcc3df8f1501f4b1aec0b"
    )
    assert result["panel"]["subjects"] == 78
    assert result["panel"]["labs"] == 9
    assert result["panel"]["sessions"] == 468
    assert result["panel"]["trials"] == 46_152
    assert result["contract_passed"]
    assert all(result["contract"].values())

    within = result["within_subject_future_session"]
    transfer = result["held_out_lab_future_session"]
    assert within["winner_by_unit_balanced_log_loss"] == "hierarchical_smooth_drift"
    assert transfer["winner_by_unit_balanced_log_loss"] == "hierarchical_smooth_drift"
    assert len(within["folds"]) == 1
    assert len(transfer["folds"]) == 9
    assert {fold["held_out_group"] for fold in transfer["folds"]} == set(
        result["panel"]["subjects_per_lab"]
    )
    assert all(fold["train_session_orders"] == list(range(5)) for fold in transfer["folds"])
    assert all(fold["test_session_orders"] == [5] for fold in transfer["folds"])
    assert all(
        model["audit_status"] == "pass"
        for report in (within, transfer)
        for model in report["models"].values()
    )

    within_difference = within["pairwise_log_loss_differences"][
        "static_partial_pooling_minus_hierarchical_smooth_drift"
    ]["left_minus_right"]
    assert within_difference["estimate"] == pytest.approx(0.0850967639126085)
    assert within_difference["lower"] > 0
    transfer_difference = transfer["pairwise_log_loss_differences"][
        "static_partial_pooling_minus_hierarchical_smooth_drift"
    ]["left_minus_right"]
    assert transfer_difference["estimate"] == pytest.approx(0.023552863885507314)
    assert transfer_difference["lower"] < 0 < transfer_difference["upper"]
    lab_difference = transfer["lab_balanced_subject_log_loss"]["static_minus_drifting"]
    assert lab_difference["estimate"] == pytest.approx(0.009682940423179779)
    assert lab_difference["interval_95"][0] < 0 < lab_difference["interval_95"][1]
