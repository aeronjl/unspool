"""Reproduce and prospectively test Cell 2025 behavioural trajectories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import expit

from benchmarks.cell2025.benchmark import (
    calculate_session_metrics,
    load_study,
)
from benchmarks.cell2025.benchmark import (
    run as run_bounded_reproduction,
)
from benchmarks.cell2025.fetch_data import (
    DEFAULT_DESTINATION,
    FIGSHARE_ARTICLE_DOI,
    MEMBER_SHA256,
    sha256,
)
from unspool import (
    BernoulliHistoryGLM,
    HierarchicalBernoulliHistoryGLM,
    HierarchicalSmoothBernoulliHistoryGLM,
    HierarchicalSmoothGLMFitResult,
    ModelRecoveryReport,
    ModelRecoveryScenario,
    SmoothBernoulliHistoryGLM,
    Study,
    compare_models,
    historical_cohort_forecast_splits,
    run_model_recovery,
)

CONTEXT_PAPER_DAYS = tuple(range(1, 9))
FORECAST_HORIZON = 5
N_FOLDS = 6
KNOTS = (0.0, 3.0, 7.0, 9.0, 12.0)
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_250_525
MODEL_RECOVERY_SEED = 20_250_526
PARAMETER_RECOVERY_SEED = 20_250_527
MODEL_RECOVERY_REPEATS = 3
PARAMETER_RECOVERY_REPEATS = 3
EARLY_BIAS_RECOVERY_REPEATS = 12
EARLY_BIAS_RECOVERY_SEED = 20_250_528
TRAJECTORY_CLUSTER_ARTIFACT = Path(__file__).with_name("trajectory_clusters.json")
Q_VALUE_SUMMARY_ARTIFACT = Path(__file__).with_name("released_q_value_summary.json")
MODEL_ORDER = (
    "pooled_psychometric",
    "late_phase_psychometric",
    "early_bias_forecast",
    "static_partial_pooling",
    "shared_smooth_trajectory",
    "hierarchical_smooth_trajectory",
)


def build_forecast_panel(study: Study) -> Study:
    """Build the frozen first-eight/final-five paper-day panel.

    Source sessions sharing a published paper-day number are combined into one derived
    modeling session. Their identifiers, chronological orders, and trial numbers remain
    explicit source columns; derived trial indices are reassigned only to satisfy the
    canonical uniqueness contract.
    """

    required = {
        "choice",
        "reward",
        "stimulus_side",
        "signed_contrast",
        "left_contrast",
        "right_contrast",
        "response_time",
        "paper_session_order",
        "source_trial",
    }
    missing = required - set(study.columns)
    if missing:
        raise ValueError(f"Cell study is missing required columns: {sorted(missing)}")

    eligible = {
        _scalar(study["subject"][row])
        for row in range(len(study))
        if int(study["paper_session_order"][row]) < 3
        and not _excluded_source_session(study["session"][row])
    }
    days_by_subject: dict[Any, set[int]] = defaultdict(set)
    for row in range(len(study)):
        subject = _scalar(study["subject"][row])
        if subject in eligible and not _excluded_source_session(study["session"][row]):
            days_by_subject[subject].add(int(study["paper_session_order"][row]))

    selected_rank: dict[tuple[Any, int], int] = {}
    forecast_days: dict[Any, tuple[int, ...]] = {}
    for subject in sorted(eligible, key=str):
        observed = tuple(sorted(days_by_subject[subject]))
        missing_context = sorted(set(CONTEXT_PAPER_DAYS) - set(observed))
        if missing_context:
            raise ValueError(f"subject {subject!r} lacks context days {missing_context}")
        final_days = observed[-FORECAST_HORIZON:]
        selected = (*CONTEXT_PAPER_DAYS, *final_days)
        if len(set(selected)) != len(selected):
            raise ValueError(f"subject {subject!r} lacks disjoint context and forecast sessions")
        forecast_days[subject] = final_days
        selected_rank.update(
            {(subject, paper_day): rank for rank, paper_day in enumerate(selected)}
        )
    if not selected_rank:
        raise ValueError("panel selection produced no eligible subjects")

    early_bias = _early_bias_by_subject(study, eligible)
    columns: dict[str, list[Any]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "session_order": [],
        "paper_session_order": [],
        "source_session": [],
        "source_session_order": [],
        "source_trial": [],
        "phase": [],
        "choice": [],
        "reward": [],
        "stimulus_side": [],
        "signed_contrast": [],
        "left_contrast": [],
        "right_contrast": [],
        "response_time": [],
        "early_bias": [],
        "early_bias_left_contrast": [],
        "early_bias_right_contrast": [],
        "forecast_phase": [],
        "forecast_phase_left_contrast": [],
        "forecast_phase_right_contrast": [],
        "early_bias_forecast_phase": [],
        "early_bias_forecast_left_contrast": [],
        "early_bias_forecast_right_contrast": [],
    }
    next_trial: dict[tuple[Any, int], int] = defaultdict(int)
    for raw_row in study.chronological_indices():
        row = int(raw_row)
        subject = _scalar(study["subject"][row])
        paper_day = int(study["paper_session_order"][row])
        key = (subject, paper_day)
        if key not in selected_rank or _excluded_source_session(study["session"][row]):
            continue
        rank = selected_rank[key]
        trial = next_trial[key]
        next_trial[key] += 1
        left = float(study["left_contrast"][row])
        right = float(study["right_contrast"][row])
        bias = early_bias[subject]
        columns["subject"].append(subject)
        columns["session"].append(f"{subject}:paper-day:{paper_day}")
        columns["trial"].append(trial)
        columns["session_order"].append(rank)
        columns["paper_session_order"].append(paper_day)
        columns["source_session"].append(_scalar(study["session"][row]))
        columns["source_session_order"].append(int(study["session_order"][row]))
        columns["source_trial"].append(int(study["source_trial"][row]))
        columns["phase"].append("context" if rank < len(CONTEXT_PAPER_DAYS) else "forecast")
        columns["choice"].append(int(study["choice"][row]))
        columns["reward"].append(int(study["reward"][row]))
        columns["stimulus_side"].append(int(study["stimulus_side"][row]))
        columns["signed_contrast"].append(float(study["signed_contrast"][row]))
        columns["left_contrast"].append(left)
        columns["right_contrast"].append(right)
        columns["response_time"].append(float(study["response_time"][row]))
        columns["early_bias"].append(bias)
        columns["early_bias_left_contrast"].append(bias * left)
        columns["early_bias_right_contrast"].append(bias * right)
        forecast_phase = int(rank >= len(CONTEXT_PAPER_DAYS))
        columns["forecast_phase"].append(forecast_phase)
        columns["forecast_phase_left_contrast"].append(forecast_phase * left)
        columns["forecast_phase_right_contrast"].append(forecast_phase * right)
        columns["early_bias_forecast_phase"].append(bias * forecast_phase)
        columns["early_bias_forecast_left_contrast"].append(bias * forecast_phase * left)
        columns["early_bias_forecast_right_contrast"].append(bias * forecast_phase * right)

    panel = Study(columns)
    expected_orders = set(range(len(CONTEXT_PAPER_DAYS) + FORECAST_HORIZON))
    for subject in panel.subjects:
        mask = panel["subject"] == subject
        observed_orders = set(panel["session_order"][mask])
        if observed_orders != expected_orders:
            raise ValueError(f"subject {subject!r} lacks retained trials in a panel session")
        observed_forecast_days = tuple(
            sorted(set(panel["paper_session_order"][mask & (panel["phase"] == "forecast")]))
        )
        if observed_forecast_days != forecast_days[_scalar(subject)]:
            raise ValueError(f"subject {subject!r} forecast-day mapping changed")
    return panel


def analyze_forecast(
    panel: Study,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fit the frozen candidates under historical-cohort prospective folds."""

    splits = historical_cohort_forecast_splits(
        panel,
        context_session_count=len(CONTEXT_PAPER_DAYS),
        horizon=FORECAST_HORIZON,
        n_folds=N_FOLDS,
    )
    report = compare_models(
        _models(),
        panel,
        splits,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    payload = report.to_dict()
    payload["panel"] = panel_manifest(panel)
    return payload


def analyze_model_recovery(
    panel: Study,
    *,
    repeats: int = MODEL_RECOVERY_REPEATS,
    seed: int = MODEL_RECOVERY_SEED,
) -> dict[str, Any]:
    """Run structural model recovery under the exact flagship validation design."""

    candidates = _structural_models()
    report = run_model_recovery(
        panel,
        _recovery_scenarios(candidates),
        candidates,
        repeats=repeats,
        seed=seed,
        tie_tolerance=1e-6,
        splitter=lambda study: historical_cohort_forecast_splits(
            study,
            context_session_count=len(CONTEXT_PAPER_DAYS),
            horizon=FORECAST_HORIZON,
            n_folds=N_FOLDS,
        ),
        splitter_name="historical-cohort-session-forecast",
        aggregation_column="subject",
    )
    return _model_recovery_payload(report)


def analyze_hierarchical_path_recovery(
    panel: Study,
    *,
    repeats: int = PARAMETER_RECOVERY_REPEATS,
    seed: int = PARAMETER_RECOVERY_SEED,
) -> dict[str, Any]:
    """Recover population and realized animal paths for the flagship winner family."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    model = _structural_models()["hierarchical_smooth_trajectory"]
    if not isinstance(model, HierarchicalSmoothBernoulliHistoryGLM):
        raise TypeError("hierarchical recovery candidate changed model family")
    parameters = _hierarchical_drift_parameters(model)
    child_sequences = np.random.SeedSequence(seed).spawn(repeats)
    runs: list[dict[str, Any]] = []
    for sequence in child_sequences:
        child_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        simulation = model.simulate_with_effects(panel, parameters, seed=child_seed)
        fit = model.fit(simulation.study)
        if not isinstance(fit, HierarchicalSmoothGLMFitResult):
            raise TypeError("hierarchical smooth recovery lost path estimates")
        population_error = fit.population_knot_values - simulation.population_knot_values
        subject_error = fit.subject_knot_values - simulation.subject_knot_values
        runs.append(
            {
                "seed": child_seed,
                "converged": fit.diagnostics.converged,
                "fit_audit": fit.audit().to_dict(),
                "population_path_rmse": float(np.sqrt(np.mean(population_error**2))),
                "subject_path_rmse": float(np.sqrt(np.mean(subject_error**2))),
                "population_path_rmse_by_coefficient": {
                    name: float(np.sqrt(np.mean(population_error[index] ** 2)))
                    for index, name in enumerate(model.coefficient_names)
                },
                "subject_path_rmse_by_coefficient": {
                    name: float(np.sqrt(np.mean(subject_error[:, index, :] ** 2)))
                    for index, name in enumerate(model.coefficient_names)
                },
                "subject_path_correlation_by_coefficient": {
                    name: _safe_correlation(
                        simulation.subject_knot_values[:, index, :],
                        fit.subject_knot_values[:, index, :],
                    )
                    for index, name in enumerate(model.coefficient_names)
                },
            }
        )
    return {
        "model_signature": model.signature,
        "design": panel_manifest(panel),
        "root_seed": seed,
        "repeats": repeats,
        "generating_population_paths": {
            name: [
                float(parameters[f"{name}[session_order={_format_knot(knot)}]"]) for knot in KNOTS
            ]
            for name in model.coefficient_names
        },
        "runs": runs,
        "summary": {
            "convergence_rate": float(np.mean([run["converged"] for run in runs])),
            "population_path_rmse_mean": float(
                np.mean([run["population_path_rmse"] for run in runs])
            ),
            "subject_path_rmse_mean": float(np.mean([run["subject_path_rmse"] for run in runs])),
        },
    }


def analyze_early_bias_recovery(
    panel: Study,
    *,
    repeats: int = EARLY_BIAS_RECOVERY_REPEATS,
    seed: int = EARLY_BIAS_RECOVERY_SEED,
) -> dict[str, Any]:
    """Audit the outcome-derived early-bias feature under null and predictive worlds."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    models = _models()
    candidates = {name: models[name] for name in ("late_phase_psychometric", "early_bias_forecast")}
    splits = historical_cohort_forecast_splits(
        panel,
        context_session_count=len(CONTEXT_PAPER_DAYS),
        horizon=FORECAST_HORIZON,
        n_folds=N_FOLDS,
    )
    worlds = (
        "null_no_subject_signal",
        "context_predicts_late_asymmetry",
        "reward_history_without_stable_strategy",
    )
    child_sequences = np.random.SeedSequence(seed).spawn(len(worlds) * repeats)
    run_index = 0
    runs: list[dict[str, Any]] = []
    for world in worlds:
        for _ in range(repeats):
            child_seed = int(child_sequences[run_index].generate_state(1, dtype=np.uint64)[0])
            simulated = (
                _simulate_reward_history_world(panel, seed=child_seed)
                if world == "reward_history_without_stable_strategy"
                else _simulate_early_bias_world(
                    panel,
                    seed=child_seed,
                    predictive=world == "context_predicts_late_asymmetry",
                )
            )
            report = compare_models(
                candidates,
                simulated,
                splits,
                bootstrap_resamples=1,
                bootstrap_seed=child_seed,
            )
            pooled = report.result_for("late_phase_psychometric")
            early = report.result_for("early_bias_forecast")
            difference = pooled.unit_balanced_log_loss - early.unit_balanced_log_loss
            runs.append(
                {
                    "world": world,
                    "seed": child_seed,
                    "selected": report.winner,
                    "phase_control_minus_early_bias_log_loss": difference,
                    "phase_control_log_loss": pooled.unit_balanced_log_loss,
                    "early_bias_log_loss": early.unit_balanced_log_loss,
                    "audit_status": {
                        "late_phase_psychometric": pooled.audit_status.value,
                        "early_bias_forecast": early.audit_status.value,
                    },
                }
            )
            run_index += 1
    summary = {}
    for world in worlds:
        selected = [run for run in runs if run["world"] == world]
        differences = np.asarray(
            [run["phase_control_minus_early_bias_log_loss"] for run in selected],
            dtype=np.float64,
        )
        summary[world] = {
            "expected_winner": (
                "early_bias_forecast"
                if world == "context_predicts_late_asymmetry"
                else "late_phase_psychometric"
            ),
            "selection_counts": {
                name: sum(run["selected"] == name for run in selected) for name in candidates
            },
            "mean_phase_control_minus_early_bias_log_loss": float(np.mean(differences)),
            "minimum_phase_control_minus_early_bias_log_loss": float(np.min(differences)),
            "maximum_phase_control_minus_early_bias_log_loss": float(np.max(differences)),
        }
    return {
        "analysis": "exact-design recovery for the outcome-derived early-bias feature",
        "validation_scheme": "historical-cohort-session-forecast",
        "aggregation_column": "subject",
        "feature_recomputed_from_simulated_days": list(range(4, 9)),
        "root_seed": seed,
        "repeats_per_world": repeats,
        "summary": summary,
        "runs": runs,
    }


def summarize_response_times(
    study: Study,
    *,
    cluster_labels: Mapping[str, str],
) -> dict[str, Any]:
    """Summarize the paper's session-level chronometric quantities."""

    eligible = {
        _scalar(study["subject"][row])
        for row in range(len(study))
        if int(study["paper_session_order"][row]) < 3
        and not _excluded_source_session(study["session"][row])
    }
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in range(len(study)):
        subject = str(_scalar(study["subject"][row]))
        if subject in eligible and not _excluded_source_session(study["session"][row]):
            grouped[(subject, int(study["paper_session_order"][row]))].append(row)

    sessions_by_subject: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for (subject, paper_day), rows in sorted(grouped.items()):
        positions = np.asarray(rows, dtype=np.intp)
        side = np.asarray(study["stimulus_side"][positions], dtype=np.int8)
        response_time = np.asarray(study["response_time"][positions], dtype=np.float64)
        medians = {value: float(np.median(response_time[side == value])) for value in (-1, 0, 1)}
        sessions_by_subject[subject].append(
            {
                "paper_day": paper_day,
                "n_trials": len(rows),
                "mean_response_time": float(np.mean(response_time)),
                "zero_response_time": medians[0],
                "left_chronometric_slope": medians[0] - medians[-1],
                "right_chronometric_slope": medians[0] - medians[1],
            }
        )

    animals: dict[str, Any] = {}
    first_response_times: list[float] = []
    late_response_times: list[float] = []
    for subject in sorted(sessions_by_subject):
        rows = sessions_by_subject[subject]
        final = rows[-5:]
        first_response_time = float(rows[0]["mean_response_time"])
        late_response_time = float(np.mean([row["mean_response_time"] for row in final]))
        first_response_times.append(first_response_time)
        late_response_times.append(late_response_time)
        animals[subject] = {
            "cluster": cluster_labels[subject],
            "first_session_mean_response_time": first_response_time,
            "final_five_mean_response_time": late_response_time,
            "final_five_left_chronometric_slope": float(
                np.mean([row["left_chronometric_slope"] for row in final])
            ),
            "final_five_right_chronometric_slope": float(
                np.mean([row["right_chronometric_slope"] for row in final])
            ),
        }
    first_array = np.asarray(first_response_times, dtype=np.float64)
    late_array = np.asarray(late_response_times, dtype=np.float64)
    paired = stats.ttest_rel(first_array, late_array)
    clusters = {
        cluster: {
            "n_subjects": sum(row["cluster"] == cluster for row in animals.values()),
            "final_five_left_chronometric_slope_mean": _mean_or_none(
                [
                    float(row["final_five_left_chronometric_slope"])
                    for row in animals.values()
                    if row["cluster"] == cluster
                ]
            ),
            "final_five_right_chronometric_slope_mean": _mean_or_none(
                [
                    float(row["final_five_right_chronometric_slope"])
                    for row in animals.values()
                    if row["cluster"] == cluster
                ]
            ),
        }
        for cluster in ("left", "balanced", "right")
    }
    return {
        "analysis": "independent session-level response-time and chronometric summary",
        "claim_boundary": "descriptive reproduction; no prospective response-time model",
        "definitions": {
            "response_time": "choice completion minus stimulus onset",
            "chronometric_slope": (
                "median zero-contrast response time minus median stimulus-side response time"
            ),
            "late": "final five paper sessions",
        },
        "n_subjects": len(animals),
        "first_session_mean_response_time": float(np.mean(first_array)),
        "final_five_mean_response_time": float(np.mean(late_array)),
        "paired_change": {
            "first_minus_final_five": float(np.mean(first_array - late_array)),
            "t_statistic": float(paired.statistic),
            "p_value": float(paired.pvalue),
        },
        "clusters": clusters,
        "animals": animals,
    }


def panel_manifest(panel: Study) -> dict[str, Any]:
    """Return compact experimental-unit and provenance counts for the derived panel."""

    source_sessions = {
        (_scalar(subject), _scalar(session))
        for subject, session in zip(panel["subject"], panel["source_session"], strict=True)
    }
    derived_sessions = {
        (_scalar(subject), _scalar(session))
        for subject, session in zip(panel["subject"], panel["session"], strict=True)
    }
    return {
        "n_trials": len(panel),
        "n_subjects": len(panel.subjects),
        "n_source_sessions": len(source_sessions),
        "n_derived_paper_day_sessions": len(derived_sessions),
        "context_paper_days": list(CONTEXT_PAPER_DAYS),
        "forecast_horizon_sessions": FORECAST_HORIZON,
        "aligned_session_orders": sorted(set(int(value) for value in panel["session_order"])),
        "subjects": [str(subject) for subject in panel.subjects],
        "early_bias_by_subject": {
            str(subject): float(panel["early_bias"][np.flatnonzero(panel["subject"] == subject)[0]])
            for subject in panel.subjects
        },
    }


def run(
    path: Path,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    model_recovery_repeats: int = MODEL_RECOVERY_REPEATS,
    parameter_recovery_repeats: int = PARAMETER_RECOVERY_REPEATS,
    early_bias_recovery_repeats: int = EARLY_BIAS_RECOVERY_REPEATS,
) -> dict[str, Any]:
    """Execute the bounded reproduction and frozen prospective forecast."""

    observed_sha256 = sha256(path)
    if observed_sha256 != MEMBER_SHA256:
        raise ValueError(
            f"input checksum mismatch: observed {observed_sha256}, expected {MEMBER_SHA256}"
        )
    study = load_study(path)
    panel = build_forecast_panel(study)
    published_artifacts = _published_artifact_summary()
    return {
        "benchmark": "Cell 2025 behavioural flagship",
        "source": {
            "doi": FIGSHARE_ARTICLE_DOI,
            "member_sha256": observed_sha256,
        },
        "claim_layers": {
            "published_reproduction": (
                "retrospective reproduction; not evidence of prospective generalization"
            ),
            "historical_cohort_forecast": (
                "completed reference animals plus forecast-animal days 1-8 predict its final five"
            ),
        },
        "published_reproduction": asdict(run_bounded_reproduction(path)),
        "released_trajectory_clustering": published_artifacts["trajectory_clustering"],
        "released_q_value_comparison": published_artifacts["q_value_comparison"],
        "response_time_summary": summarize_response_times(
            study,
            cluster_labels=published_artifacts["trajectory_clustering"][
                "semantic_label_by_subject"
            ],
        ),
        "historical_cohort_forecast": analyze_forecast(
            panel,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "exact_design_model_recovery": analyze_model_recovery(
            panel,
            repeats=model_recovery_repeats,
        ),
        "hierarchical_path_parameter_recovery": analyze_hierarchical_path_recovery(
            panel,
            repeats=parameter_recovery_repeats,
        ),
        "early_bias_feature_recovery": analyze_early_bias_recovery(
            panel,
            repeats=early_bias_recovery_repeats,
        ),
    }


def _models() -> Mapping[str, Any]:
    psychometric = ("left_contrast", "right_contrast")
    late_phase = (
        *psychometric,
        "forecast_phase",
        "forecast_phase_left_contrast",
        "forecast_phase_right_contrast",
    )
    common = {"choice_lags": 0, "l2": 0.02}
    return {
        "pooled_psychometric": BernoulliHistoryGLM(covariates=psychometric, **common),
        "late_phase_psychometric": BernoulliHistoryGLM(covariates=late_phase, **common),
        "early_bias_forecast": BernoulliHistoryGLM(
            covariates=(
                *late_phase,
                "early_bias",
                "early_bias_forecast_phase",
                "early_bias_forecast_left_contrast",
                "early_bias_forecast_right_contrast",
            ),
            **common,
        ),
        "static_partial_pooling": HierarchicalBernoulliHistoryGLM(
            covariates=psychometric,
            subject_scale=0.4,
            **common,
        ),
        "shared_smooth_trajectory": SmoothBernoulliHistoryGLM(
            covariates=psychometric,
            knots=KNOTS,
            smoothness=3.0,
            shared_trajectory=True,
            **common,
        ),
        "hierarchical_smooth_trajectory": HierarchicalSmoothBernoulliHistoryGLM(
            covariates=psychometric,
            knots=KNOTS,
            smoothness=3.0,
            subject_scale=0.4,
            subject_smoothness=3.0,
            **common,
        ),
    }


def _published_artifact_summary() -> dict[str, Any]:
    with TRAJECTORY_CLUSTER_ARTIFACT.open(encoding="utf-8") as handle:
        clusters = json.load(handle)
    if clusters["source_member_sha256"] != MEMBER_SHA256:
        raise ValueError("trajectory artifact does not match the benchmark source")
    validation = clusters.get("released_membership_validation")
    if not validation or not validation["exact_semantic_membership_match"]:
        raise ValueError("trajectory artifact lacks exact released-membership validation")

    with Q_VALUE_SUMMARY_ARTIFACT.open(encoding="utf-8") as handle:
        q_value = json.load(handle)
    expected_q_digest = "ba69393ca8ceb8932c77958ba66f27d1c14089684adbb0fd32a38f0e27daee5e"
    if q_value["source_sha256"] != expected_q_digest:
        raise ValueError("Q-value summary does not match the released pickle")
    return {
        "trajectory_clustering": {
            "interpretation": clusters["interpretation"],
            "analysis_doi": clusters["analysis_doi"],
            "environment": clusters["environment"],
            "contract": clusters["contract"],
            "memberships": clusters["memberships"],
            "semantic_label_by_subject": clusters["semantic_label_by_subject"],
            "released_membership_validation": validation,
            "full_artifact": str(
                TRAJECTORY_CLUSTER_ARTIFACT.relative_to(Path(__file__).parents[2])
            ),
        },
        "q_value_comparison": {
            "interpretation": q_value["interpretation"],
            "analysis_doi": q_value["analysis_doi"],
            "source_member": q_value["source_member"],
            "source_sha256": q_value["source_sha256"],
            "contract": q_value["contract"],
            "aggregate": q_value["aggregate"],
            "full_artifact": str(Q_VALUE_SUMMARY_ARTIFACT.relative_to(Path(__file__).parents[2])),
        },
    }


def _structural_models() -> Mapping[str, Any]:
    models = _models()
    return {
        name: models[name]
        for name in (
            "pooled_psychometric",
            "static_partial_pooling",
            "shared_smooth_trajectory",
            "hierarchical_smooth_trajectory",
        )
    }


def _recovery_scenarios(
    candidates: Mapping[str, Any],
) -> tuple[ModelRecoveryScenario, ...]:
    pooled = candidates["pooled_psychometric"]
    static = candidates["static_partial_pooling"]
    shared = candidates["shared_smooth_trajectory"]
    hierarchical = candidates["hierarchical_smooth_trajectory"]
    stationary = {"intercept": 0.0, "left_contrast": 4.0, "right_contrast": 4.0}
    shared_parameters = shared.parameters_from_paths(
        {
            "intercept": (-0.2, -0.1, 0.0, 0.1, 0.1),
            "left_contrast": (0.5, 1.5, 3.0, 4.0, 5.0),
            "right_contrast": (0.5, 1.5, 3.0, 4.0, 5.0),
        }
    )
    return (
        ModelRecoveryScenario(
            name="stationary_complete_pooling",
            truth_label="pooled_psychometric",
            generator=pooled,
            parameters=stationary,
        ),
        ModelRecoveryScenario(
            name="stationary_individual_heterogeneity",
            truth_label="static_partial_pooling",
            generator=static,
            parameters=stationary,
        ),
        ModelRecoveryScenario(
            name="shared_population_drift",
            truth_label="shared_smooth_trajectory",
            generator=shared,
            parameters=shared_parameters,
        ),
        ModelRecoveryScenario(
            name="individual_smooth_drift",
            truth_label="hierarchical_smooth_trajectory",
            generator=hierarchical,
            parameters=_hierarchical_drift_parameters(hierarchical),
        ),
    )


def _hierarchical_drift_parameters(
    model: HierarchicalSmoothBernoulliHistoryGLM,
) -> Mapping[str, float]:
    return model.parameters_from_paths(
        {
            "intercept": (-0.2, -0.1, 0.0, 0.1, 0.1),
            "left_contrast": (0.5, 1.5, 3.0, 4.0, 5.0),
            "right_contrast": (0.5, 1.5, 3.0, 4.0, 5.0),
        }
    )


def _model_recovery_payload(report: ModelRecoveryReport) -> dict[str, Any]:
    confusion = report.confusion_matrix()
    scenario_confusion = report.scenario_confusion_matrix()
    return {
        "candidate_order": list(report.candidate_labels),
        "candidate_signatures": list(report.candidate_signatures),
        "scored_columns": list(report.scored_columns),
        "validation_scheme": report.validation_scheme,
        "aggregation_column": report.aggregation_column,
        "n_trials": report.n_trials,
        "n_subjects": report.n_subjects,
        "repeats": report.repeats,
        "root_seed": report.root_seed,
        "tie_tolerance": report.tie_tolerance,
        "resolution_rate": report.resolution_rate,
        "overall_accuracy": report.overall_accuracy,
        "resolved_accuracy": report.resolved_accuracy,
        "audit_warning_rate": report.audit_warning_rate,
        "audit_failure_rate": report.audit_failure_rate,
        "confusion_matrix": {
            "truth_labels": list(confusion.truth_labels),
            "selected_labels": list(confusion.selected_labels),
            "counts": confusion.counts.tolist(),
            "rates": confusion.rates.tolist(),
        },
        "scenario_confusion_matrix": {
            "scenario_names": list(scenario_confusion.scenario_names),
            "truth_labels": list(scenario_confusion.truth_labels),
            "selected_labels": list(scenario_confusion.selected_labels),
            "counts": scenario_confusion.counts.tolist(),
            "rates": scenario_confusion.rates.tolist(),
        },
        "runs": [
            {
                "scenario": report.scenario_names[index],
                "truth": report.truth_labels[index],
                "selected": report.selected_labels[index],
                "seed": int(report.seeds[index]),
                "n_folds": int(report.n_folds[index]),
                "mean_log_probability": {
                    label: float(report.mean_log_probabilities[index, column])
                    for column, label in enumerate(report.candidate_labels)
                },
                "converged": {
                    label: bool(report.converged[index, column])
                    for column, label in enumerate(report.candidate_labels)
                },
                "audit_status": {
                    label: report.audit_statuses[index][column].value
                    for column, label in enumerate(report.candidate_labels)
                },
                "audit_issue_codes": {
                    label: list(report.audit_issue_codes[index][column])
                    for column, label in enumerate(report.candidate_labels)
                },
                "failure_messages": {
                    label: report.failure_messages[index][column]
                    for column, label in enumerate(report.candidate_labels)
                },
            }
            for index in range(report.n_runs)
        ],
    }


def _early_bias_by_subject(study: Study, eligible: set[Any]) -> dict[Any, float]:
    rows_by_subject: dict[Any, list[float]] = defaultdict(list)
    for row in calculate_session_metrics(study):
        if row.subject in eligible and 3 < row.session_order <= 8:
            rows_by_subject[row.subject].append(row.zero_bias)
    result: dict[Any, float] = {}
    for subject in eligible:
        values = np.asarray(rows_by_subject[subject], dtype=np.float64)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError(f"subject {subject!r} lacks five finite early-bias sessions")
        result[subject] = float(np.mean(values))
    return result


def _simulate_early_bias_world(
    panel: Study,
    *,
    seed: int,
    predictive: bool,
) -> Study:
    generator = np.random.default_rng(seed)
    latent = {
        _scalar(subject): (
            float(np.clip(generator.normal(0.0, 0.35), -0.65, 0.65)) if predictive else 0.0
        )
        for subject in panel.subjects
    }
    choices = np.empty(len(panel), dtype=np.int8)
    for row in panel.chronological_indices():
        index = int(row)
        subject_strategy = latent[_scalar(panel["subject"][index])]
        order = int(panel["session_order"][index])
        base_slope = 0.75 + 4.25 * order / 12.0
        intercept = 0.0
        left_slope = base_slope
        right_slope = base_slope
        if 3 <= order <= 7:
            intercept = 3.0 * subject_strategy
        elif order >= 8:
            intercept = -subject_strategy
            left_slope -= 4.0 * subject_strategy
            right_slope += 4.0 * subject_strategy
        linear_predictor = intercept
        linear_predictor += left_slope * float(panel["left_contrast"][index])
        linear_predictor += right_slope * float(panel["right_contrast"][index])
        choices[index] = generator.binomial(1, expit(linear_predictor))
    columns = {name: panel[name] for name in panel.columns}
    columns["choice"] = choices
    simulated = Study(columns)
    return _recompute_early_bias_features(simulated)


def _simulate_reward_history_world(panel: Study, *, seed: int) -> Study:
    """Generate symmetric reinforcement history without a stable animal strategy.

    Every animal shares the same learning rate and inverse temperature. Random early
    rewards can create transient action-value asymmetry, but no animal-specific latent
    preference or trajectory is generated. Values persist across the retained aligned
    panel, matching the forecast design rather than silently reading excluded sessions.
    """

    generator = np.random.default_rng(seed)
    action_values = {
        _scalar(subject): np.asarray([0.5, 0.5], dtype=np.float64) for subject in panel.subjects
    }
    choices = np.empty(len(panel), dtype=np.int8)
    rewards = np.empty(len(panel), dtype=np.int8)
    learning_rate = 0.2
    inverse_temperature = 2.0
    for row in panel.chronological_indices():
        index = int(row)
        subject = _scalar(panel["subject"][index])
        values = action_values[subject]
        order = int(panel["session_order"][index])
        evidence_slope = 0.75 + 4.25 * order / 12.0
        linear_predictor = evidence_slope * float(panel["signed_contrast"][index])
        linear_predictor += inverse_temperature * float(values[1] - values[0])
        choice = int(generator.binomial(1, expit(linear_predictor)))
        side = int(panel["stimulus_side"][index])
        reward = int(generator.binomial(1, 0.5)) if side == 0 else int(choice == int(side > 0))
        values[choice] += learning_rate * (reward - values[choice])
        choices[index] = choice
        rewards[index] = reward
    columns = {name: panel[name] for name in panel.columns}
    columns["choice"] = choices
    columns["reward"] = rewards
    return _recompute_early_bias_features(Study(columns))


def _recompute_early_bias_features(study: Study) -> Study:
    bias_by_subject: dict[Any, float] = {}
    for subject in study.subjects:
        daily_biases = []
        for paper_day in range(4, 9):
            mask = (study["subject"] == subject) & (study["paper_session_order"] == paper_day)
            mask &= study["stimulus_side"] == 0
            if not np.any(mask):
                raise ValueError(
                    f"subject {subject!r} lacks zero-contrast trials on day {paper_day}"
                )
            daily_biases.append(float(np.mean(study["choice"][mask])) - 0.5)
        bias_by_subject[_scalar(subject)] = float(np.mean(daily_biases))
    early_bias = np.asarray(
        [bias_by_subject[_scalar(subject)] for subject in study["subject"]],
        dtype=np.float64,
    )
    columns = {name: study[name] for name in study.columns}
    columns["early_bias"] = early_bias
    columns["early_bias_left_contrast"] = early_bias * np.asarray(
        study["left_contrast"], dtype=np.float64
    )
    columns["early_bias_right_contrast"] = early_bias * np.asarray(
        study["right_contrast"], dtype=np.float64
    )
    forecast_phase = np.asarray(study["forecast_phase"], dtype=np.float64)
    columns["early_bias_forecast_phase"] = early_bias * forecast_phase
    columns["early_bias_forecast_left_contrast"] = (
        early_bias * forecast_phase * np.asarray(study["left_contrast"], dtype=np.float64)
    )
    columns["early_bias_forecast_right_contrast"] = (
        early_bias * forecast_phase * np.asarray(study["right_contrast"], dtype=np.float64)
    )
    return Study(columns)


def _excluded_source_session(value: Any) -> bool:
    session = str(_scalar(value))
    return "ALK" in session or "MMM" in session


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _safe_correlation(truth: np.ndarray, estimate: np.ndarray) -> float | None:
    truth_flat = np.asarray(truth, dtype=np.float64).ravel()
    estimate_flat = np.asarray(estimate, dtype=np.float64).ravel()
    if np.ptp(truth_flat) <= 1e-12 or np.ptp(estimate_flat) <= 1e-12:
        return None
    return float(np.corrcoef(truth_flat, estimate_flat)[0, 1])


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _format_knot(value: float) -> str:
    return np.format_float_positional(value, trim="-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--model-recovery-repeats", type=int, default=MODEL_RECOVERY_REPEATS)
    parser.add_argument(
        "--parameter-recovery-repeats",
        type=int,
        default=PARAMETER_RECOVERY_REPEATS,
    )
    parser.add_argument(
        "--early-bias-recovery-repeats",
        type=int,
        default=EARLY_BIAS_RECOVERY_REPEATS,
    )
    arguments = parser.parse_args()
    result = run(
        arguments.data.resolve(),
        bootstrap_resamples=arguments.bootstrap_resamples,
        model_recovery_repeats=arguments.model_recovery_repeats,
        parameter_recovery_repeats=arguments.parameter_recovery_repeats,
        early_bias_recovery_repeats=arguments.early_bias_recovery_repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
