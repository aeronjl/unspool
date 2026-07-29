import json
from pathlib import Path

import numpy as np

from behavio import Study
from benchmarks.cell2025_flagship.benchmark import (
    MODEL_ORDER,
    _models,
    _published_artifact_summary,
    _recovery_scenarios,
    _simulate_reward_history_world,
    _structural_models,
    build_forecast_panel,
    panel_manifest,
    summarize_response_times,
)

ROOT = Path(__file__).parents[1]


def synthetic_source_study() -> Study:
    columns: dict[str, list[object]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "source_trial": [],
        "session_order": [],
        "paper_session_order": [],
        "choice": [],
        "reward": [],
        "stimulus_side": [],
        "signed_contrast": [],
        "left_contrast": [],
        "right_contrast": [],
        "response_time": [],
    }
    for subject in ("a", "b", "c", "d"):
        source_order = 0
        for paper_day in range(1, 14):
            session_trials = (40, 40) if subject == "a" and paper_day == 1 else (80,)
            for source_part, n_trials in enumerate(session_trials):
                session = f"{subject}-day-{paper_day}-part-{source_part}"
                for trial in range(n_trials):
                    combined_trial = source_part * 40 + trial
                    side = -1 if combined_trial < 30 else 0 if combined_trial < 50 else 1
                    contrast = 0.5 * side
                    choice = int((combined_trial + paper_day + ord(subject)) % 3 != 0)
                    columns["subject"].append(subject)
                    columns["session"].append(session)
                    columns["trial"].append(trial)
                    columns["source_trial"].append(trial)
                    columns["session_order"].append(source_order)
                    columns["paper_session_order"].append(paper_day)
                    columns["choice"].append(choice)
                    columns["reward"].append(int(choice == int(side >= 0)))
                    columns["stimulus_side"].append(side)
                    columns["signed_contrast"].append(contrast)
                    columns["left_contrast"].append(min(contrast, 0.0))
                    columns["right_contrast"].append(max(contrast, 0.0))
                    columns["response_time"].append(0.2 + trial / 1_000)
                source_order += 1
    return Study(columns)


def test_flagship_panel_merges_paper_days_but_preserves_source_provenance() -> None:
    panel = build_forecast_panel(synthetic_source_study())
    manifest = panel_manifest(panel)

    assert manifest["n_trials"] == 4 * 13 * 80
    assert manifest["n_subjects"] == 4
    assert manifest["n_source_sessions"] == 53
    assert manifest["n_derived_paper_day_sessions"] == 52
    assert manifest["aligned_session_orders"] == list(range(13))
    assert set(panel["phase"]) == {"context", "forecast"}
    assert np.all(np.isfinite(panel["early_bias"]))

    first_day = (panel["subject"] == "a") & (panel["paper_session_order"] == 1)
    assert len(set(panel["source_session"][first_day])) == 2
    assert panel["trial"][first_day].tolist() == list(range(80))
    assert len(set(panel["session"][first_day])) == 1


def test_flagship_candidates_match_the_frozen_contract() -> None:
    models = _models()

    assert tuple(models) == MODEL_ORDER
    assert all(model.choice_lags == 0 for model in models.values())
    assert models["pooled_psychometric"].covariates == (
        "left_contrast",
        "right_contrast",
    )
    assert models["early_bias_forecast"].covariates[-3:] == (
        "early_bias_forecast_phase",
        "early_bias_forecast_left_contrast",
        "early_bias_forecast_right_contrast",
    )

    structural = _structural_models()
    scenarios = _recovery_scenarios(structural)
    assert tuple(structural) == (
        "pooled_psychometric",
        "static_partial_pooling",
        "shared_smooth_trajectory",
        "hierarchical_smooth_trajectory",
    )
    assert tuple(scenario.truth_label for scenario in scenarios) == tuple(structural)


def test_published_behaviour_artifacts_are_pinned_and_response_times_are_summarized() -> None:
    artifacts = _published_artifact_summary()
    clustering = artifacts["trajectory_clustering"]
    assert clustering["released_membership_validation"]["exact_semantic_membership_match"]
    assert sum(len(values) for values in clustering["memberships"].values()) == 30
    assert (
        artifacts["q_value_comparison"]["aggregate"]["innate_and_reward"]["best_bic_animal_count"]
        == 13
    )

    source = synthetic_source_study()
    labels = {str(subject): "balanced" for subject in source.subjects}
    summary = summarize_response_times(source, cluster_labels=labels)
    assert summary["n_subjects"] == 4
    assert summary["clusters"]["balanced"]["n_subjects"] == 4
    assert summary["paired_change"]["p_value"] >= 0


def test_reward_history_competing_world_recomputes_outcome_derived_features() -> None:
    panel = build_forecast_panel(synthetic_source_study())

    first = _simulate_reward_history_world(panel, seed=42)
    repeated = _simulate_reward_history_world(panel, seed=42)
    alternative = _simulate_reward_history_world(panel, seed=43)

    assert np.array_equal(first["choice"], repeated["choice"])
    assert not np.array_equal(first["choice"], alternative["choice"])
    assert set(first["reward"]) <= {0, 1}
    assert np.all(np.isfinite(first["early_bias"]))
    for subject in first.subjects:
        values = first["early_bias"][first["subject"] == subject]
        assert np.all(values == values[0])


def test_committed_flagship_evidence_matches_the_frozen_contract() -> None:
    path = ROOT / "benchmarks" / "cell2025_flagship" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    reproduction = result["published_reproduction"]
    assert reproduction["contract_passed"]
    assert reproduction["n_subjects"] == 30
    assert reproduction["n_trials"] == 192_238
    assert np.isclose(reproduction["early_bias_late_slope_r"], 0.6947896564480975)

    forecast = result["historical_cohort_forecast"]
    assert forecast["model_order"] == list(MODEL_ORDER)
    assert forecast["panel"]["n_trials"] == 73_042
    assert forecast["winner_by_unit_balanced_log_loss"] == "early_bias_forecast"
    assert all(model["audit_status"] == "pass" for model in forecast["models"].values())
    pooled_difference = forecast["pairwise_log_loss_differences"][
        "pooled_psychometric_minus_early_bias_forecast"
    ]["left_minus_right"]
    assert pooled_difference["lower"] > 0

    recovery = result["exact_design_model_recovery"]
    assert recovery["resolution_rate"] == 1
    assert np.isclose(recovery["overall_accuracy"], 11 / 12)
    assert recovery["audit_warning_rate"] == 0
    assert recovery["audit_failure_rate"] == 0

    feature = result["early_bias_feature_recovery"]
    assert feature["repeats_per_world"] == 12
    assert feature["summary"]["context_predicts_late_asymmetry"]["selection_counts"] == {
        "early_bias_forecast": 12,
        "late_phase_psychometric": 0,
    }
    assert feature["summary"]["null_no_subject_signal"]["selection_counts"] == {
        "early_bias_forecast": 2,
        "late_phase_psychometric": 10,
    }
    reward_counts = feature["summary"]["reward_history_without_stable_strategy"]["selection_counts"]
    assert sum(reward_counts.values()) == 12
