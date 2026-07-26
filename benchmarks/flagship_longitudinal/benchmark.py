"""Forecast held-out sessions in matched Cell 2025 and IBL 2021 panels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cell2025.benchmark import load_study as load_cell_study
from benchmarks.cell2025.fetch_data import DEFAULT_DESTINATION as DEFAULT_CELL_DATA
from benchmarks.cell2025.fetch_data import FIGSHARE_ARTICLE_DOI, MEMBER_SHA256, sha256
from benchmarks.ibl2021.benchmark import load_study as load_ibl_study
from benchmarks.ibl2021.fetch_data import DEFAULT_DESTINATION as DEFAULT_IBL_DATA
from benchmarks.ibl2021.fetch_data import EXPECTED_MANIFEST_SHA256
from unspool import (
    BernoulliHistoryGLM,
    FitResult,
    HierarchicalBernoulliHistoryGLM,
    HierarchicalGLMFitResult,
    HierarchicalSmoothBernoulliHistoryGLM,
    HierarchicalSmoothGLMFitResult,
    SmoothBernoulliHistoryGLM,
    Study,
    audit_fit,
    cohort_forward_session_splits,
    evaluate_splits,
)

KNOTS = (0.0, 2.0, 5.0)
METHODS = (
    "complete_pooling",
    "static_partial_pooling",
    "shared_smooth_drift",
    "hierarchical_smooth_trajectories",
)
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_250_525


def build_cell_panel(study: Study) -> Study:
    """Select and align the first three and final three eligible Cell sessions."""

    required = {"paper_session_order", "choice", "stimulus_side"}
    missing = required - set(study.columns)
    if missing:
        raise ValueError(f"Cell study is missing required columns: {sorted(missing)}")
    eligible = {
        _scalar(study["subject"][row])
        for row in range(len(study))
        if int(study["paper_session_order"][row]) < 3
        and "ALK" not in str(study["session"][row])
        and "MMM" not in str(study["session"][row])
    }
    return _build_six_session_panel(
        study,
        eligible_subjects=eligible,
        choice=study["choice"],
        stimulus=study["stimulus_side"],
        exclude_session=lambda session: "ALK" in str(session) or "MMM" in str(session),
        retained_columns=(),
    )


def build_ibl_panel(study: Study) -> Study:
    """Align the pinned IBL early/late panel and remove invalid-choice trials."""

    required = {"choice", "signed_stimulus", "lab", "phase", "task_protocol"}
    missing = required - set(study.columns)
    if missing:
        raise ValueError(f"IBL study is missing required columns: {sorted(missing)}")
    raw_choice = np.asarray(study["choice"], dtype=float)
    valid_choice = np.isfinite(raw_choice) & (raw_choice != 0)
    # IBL ALF encodes rightward choices as -1; align both published panels to
    # Cell's binary convention in which 1 denotes a rightward choice.
    choice = (raw_choice < 0).astype(np.int8)
    return _build_six_session_panel(
        study,
        eligible_subjects=set(study.subjects),
        choice=choice,
        stimulus=study["signed_stimulus"],
        include_rows=valid_choice,
        retained_columns=("lab", "phase", "task_protocol"),
    )


def analyze_panel(
    panel: Study,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Fit four fixed structural accounts and forecast each subject's sixth session."""

    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    splits = cohort_forward_session_splits(panel, min_train_sessions=5, horizon=1)
    if len(splits) != 1:
        raise ValueError("a balanced six-session panel must produce exactly one flagship fold")
    split = splits[0]
    models = _models()
    subject_losses: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    method_payloads: dict[str, Any] = {}
    for method, model in models.items():
        evaluation = evaluate_splits(model, panel, (split,))[0]
        test = panel.take(split.test_indices)
        losses = -evaluation.pointwise_log_probability
        squared_errors = (evaluation.prediction.probability - test["choice"]) ** 2
        per_subject_loss = _subject_means(test, losses, split.subjects)
        per_subject_brier = _subject_means(test, squared_errors, split.subjects)
        subject_losses[method] = per_subject_loss
        method_payloads[method] = {
            "subject_balanced_log_loss": float(np.mean(per_subject_loss)),
            "pooled_trial_log_loss": float(np.mean(losses)),
            "subject_balanced_brier_score": float(np.mean(per_subject_brier)),
            "pooled_trial_brier_score": float(np.mean(squared_errors)),
            "subject_log_loss": {
                str(subject): float(value)
                for subject, value in zip(split.subjects, per_subject_loss, strict=True)
            },
            "fit_audit": audit_fit(evaluation.fit).to_dict(),
            "stimulus_trajectory": _stimulus_trajectory(method, evaluation.fit, split.subjects),
        }

    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(
        0, len(split.subjects), size=(bootstrap_resamples, len(split.subjects))
    )
    reference = subject_losses["complete_pooling"]
    for method in METHODS:
        values = subject_losses[method]
        bootstrap_loss = np.mean(values[draws], axis=1)
        improvement = reference - values
        bootstrap_improvement = np.mean(improvement[draws], axis=1)
        method_payloads[method]["subject_bootstrap_log_loss_95_interval"] = _interval(
            bootstrap_loss
        )
        method_payloads[method]["log_loss_improvement_over_complete_pooling"] = {
            "estimate": float(np.mean(improvement)),
            "interval_95": _interval(bootstrap_improvement),
            "bootstrap_probability_positive": float(np.mean(bootstrap_improvement > 0)),
        }

    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1 :]:
            difference = subject_losses[left] - subject_losses[right]
            bootstrap_difference = np.mean(difference[draws], axis=1)
            pairwise[f"{left}_minus_{right}"] = {
                "estimate": float(np.mean(difference)),
                "interval_95": _interval(bootstrap_difference),
                "bootstrap_probability_positive": float(np.mean(bootstrap_difference > 0)),
            }

    test_subjects = panel.take(split.test_indices)["subject"]
    return {
        "panel": {
            "n_trials": len(panel),
            "n_subjects": len(panel.subjects),
            "n_sessions": 6 * len(panel.subjects),
            "n_training_trials": len(split.train_indices),
            "n_test_trials": len(split.test_indices),
            "test_trials_by_subject": {
                str(subject): int(np.count_nonzero(test_subjects == subject))
                for subject in split.subjects
            },
            "subjects": [str(subject) for subject in split.subjects],
        },
        "winner_by_subject_balanced_log_loss": min(
            METHODS, key=lambda method: method_payloads[method]["subject_balanced_log_loss"]
        ),
        "pairwise_subject_log_loss_differences": pairwise,
        "methods": method_payloads,
    }


def run(cell_path: Path, ibl_directory: Path) -> dict[str, Any]:
    """Load checksum-pinned sources and execute the matched two-dataset analysis."""

    cell_digest = sha256(cell_path)
    if cell_digest != MEMBER_SHA256:
        raise ValueError(
            f"Cell input checksum mismatch: observed {cell_digest}, expected {MEMBER_SHA256}"
        )
    cell_panel = build_cell_panel(load_cell_study(cell_path))
    ibl_panel = build_ibl_panel(load_ibl_study(ibl_directory))
    return {
        "benchmark": "matched prospective longitudinal model comparison",
        "analysis_contract": {
            "panel": "first three and final three eligible sessions per subject",
            "forecast": "session rank 5 from ranks 0 through 4",
            "outcome": "binary choice",
            "covariates": ["stimulus", "one-session-reset choice lag"],
            "knots": list(KNOTS),
            "hyperparameters": {
                "l2": 0.02,
                "smoothness": 3.0,
                "subject_scale": 0.4,
                "subject_smoothness": 3.0,
            },
            "primary_metric": "mean subject-level prospective log loss",
            "uncertainty": "paired nonparametric subject bootstrap",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "model_order": list(METHODS),
        },
        "sources": {
            "cell2025": {
                "doi": FIGSHARE_ARTICLE_DOI,
                "member_sha256": cell_digest,
            },
            "ibl2021": {
                "release": "2021_Q1_IBL_et_al_Behaviour",
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            },
        },
        "datasets": {
            "cell2025": analyze_panel(cell_panel, bootstrap_seed=BOOTSTRAP_SEED),
            "ibl2021": analyze_panel(ibl_panel, bootstrap_seed=BOOTSTRAP_SEED + 1),
        },
    }


def _build_six_session_panel(
    study: Study,
    *,
    eligible_subjects: set[Any],
    choice: Any,
    stimulus: Any,
    include_rows: Any | None = None,
    exclude_session: Any | None = None,
    retained_columns: tuple[str, ...],
) -> Study:
    selected_rank: dict[tuple[Any, int], int] = {}
    for subject in study.subjects:
        key = _scalar(subject)
        if key not in eligible_subjects:
            continue
        sessions: dict[int, Any] = {}
        for row in study.chronological_indices():
            position = int(row)
            if _scalar(study["subject"][position]) != key:
                continue
            session = _scalar(study["session"][position])
            if exclude_session is not None and exclude_session(session):
                continue
            sessions[int(study["session_order"][position])] = session
        ordered = sorted(sessions)
        if len(ordered) < 6:
            raise ValueError(f"subject {subject!r} has fewer than six eligible sessions")
        selected = (*ordered[:3], *ordered[-3:])
        if len(set(selected)) != 6:
            raise ValueError(f"subject {subject!r} does not have six disjoint panel sessions")
        selected_rank.update(
            {(key, source_order): rank for rank, source_order in enumerate(selected)}
        )
    if not selected_rank:
        raise ValueError("panel selection produced no eligible subjects")

    row_mask = np.ones(len(study), dtype=np.bool_)
    if include_rows is not None:
        row_mask &= np.asarray(include_rows, dtype=np.bool_)
    positions = np.asarray(
        [
            row
            for row in range(len(study))
            if row_mask[row]
            and (_scalar(study["subject"][row]), int(study["session_order"][row])) in selected_rank
        ],
        dtype=np.intp,
    )
    source_orders = np.asarray(study["session_order"][positions], dtype=np.int64)
    subjects = study["subject"][positions]
    columns: dict[str, Any] = {
        "subject": subjects,
        "session": study["session"][positions],
        "trial": study["trial"][positions],
        "session_order": [
            selected_rank[(_scalar(subject), int(order))]
            for subject, order in zip(subjects, source_orders, strict=True)
        ],
        "source_session_order": source_orders,
        "phase": [
            "early" if selected_rank[(_scalar(subject), int(order))] < 3 else "late"
            for subject, order in zip(subjects, source_orders, strict=True)
        ],
        "choice": np.asarray(choice)[positions].astype(np.int8),
        "stimulus": np.asarray(stimulus, dtype=float)[positions],
    }
    for name in retained_columns:
        if name == "phase":
            columns["source_phase"] = study[name][positions]
        else:
            columns[name] = study[name][positions]
    panel = Study(columns)
    for subject in panel.subjects:
        orders = set(panel["session_order"][panel["subject"] == subject])
        if orders != set(range(6)):
            raise ValueError(f"subject {subject!r} lacks retained trials in a panel session")
    return panel


def _models() -> Mapping[str, Any]:
    common = {"covariates": ("stimulus",), "choice_lags": 1, "l2": 0.02}
    return {
        "complete_pooling": BernoulliHistoryGLM(**common),
        "static_partial_pooling": HierarchicalBernoulliHistoryGLM(**common, subject_scale=0.4),
        "shared_smooth_drift": SmoothBernoulliHistoryGLM(
            **common, knots=KNOTS, smoothness=3.0, shared_trajectory=True
        ),
        "hierarchical_smooth_trajectories": HierarchicalSmoothBernoulliHistoryGLM(
            **common,
            knots=KNOTS,
            smoothness=3.0,
            subject_scale=0.4,
            subject_smoothness=3.0,
        ),
    }


def _subject_means(study: Study, values: Any, subjects: tuple[Any, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray(
        [float(np.mean(array[study["subject"] == subject])) for subject in subjects],
        dtype=float,
    )


def _interval(values: Any) -> list[float]:
    lower, upper = np.quantile(np.asarray(values, dtype=float), (0.025, 0.975))
    return [float(lower), float(upper)]


def _stimulus_trajectory(method: str, fit: FitResult, subjects: tuple[Any, ...]) -> dict[str, Any]:
    if method == "complete_pooling":
        stimulus = float(fit.estimates[fit.parameter_names.index("stimulus")])
        population = np.repeat(stimulus, len(KNOTS))
        subject_paths = np.broadcast_to(population, (len(subjects), len(KNOTS)))
    elif method == "static_partial_pooling":
        if not isinstance(fit, HierarchicalGLMFitResult):
            raise TypeError("static partial-pooling fit lost hierarchical estimates")
        index = fit.parameter_names.index("stimulus")
        population = np.repeat(fit.estimates[index], len(KNOTS))
        subject_paths = np.repeat(fit.subject_coefficients[:, index, None], len(KNOTS), axis=1)
    elif method == "shared_smooth_drift":
        coefficient_names = ("intercept", "stimulus", "choice_lag_1")
        index = coefficient_names.index("stimulus")
        population = fit.estimates.reshape(len(coefficient_names), len(KNOTS))[index]
        subject_paths = np.broadcast_to(population, (len(subjects), len(KNOTS)))
    else:
        if not isinstance(fit, HierarchicalSmoothGLMFitResult):
            raise TypeError("hierarchical smooth fit lost subject trajectories")
        index = fit.coefficient_names.index("stimulus")
        population = fit.population_knot_values[index]
        subject_paths = fit.subject_knot_values[:, index, :]
    changes = subject_paths[:, -1] - subject_paths[:, 0]
    return {
        "clock": "aligned_session_rank",
        "knots": list(KNOTS),
        "population_values": [float(value) for value in population],
        "subject_values": {
            str(subject): [float(value) for value in path]
            for subject, path in zip(subjects, subject_paths, strict=True)
        },
        "subject_change_summary": {
            "mean": float(np.mean(changes)),
            "standard_deviation": float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0,
            "minimum": float(np.min(changes)),
            "maximum": float(np.max(changes)),
        },
    }


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-data", type=Path, default=DEFAULT_CELL_DATA)
    parser.add_argument("--ibl-data", type=Path, default=DEFAULT_IBL_DATA)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(arguments.cell_data.resolve(), arguments.ibl_data.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
