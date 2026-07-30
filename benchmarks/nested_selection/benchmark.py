"""Recover stationary versus shared-drift structure with nested prospective selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from behavio import (
    BernoulliHistoryGLM,
    Study,
    cohort_forward_session_splits,
    compare_models,
    nested_select_model,
)
from behavio.compose import SmoothModel, smooth
from benchmarks.provenance import render

KNOTS = (0.0, 3.0, 6.0)
REGIMES = ("stationary", "shared_drift")
CANDIDATES = ("static", "shared_smooth")
EXPECTED = {"stationary": "static", "shared_drift": "shared_smooth"}


def build_design(
    *,
    seed: int,
    n_subjects: int = 10,
    n_sessions: int = 7,
    trials_per_session: int = 40,
) -> Study:
    """Construct one balanced multi-subject design with a fixed session clock."""

    generator = np.random.default_rng(seed)
    subjects = tuple(f"mouse-{index:02d}" for index in range(n_subjects))
    n_rows = n_subjects * n_sessions * trials_per_session
    return Study(
        {
            "subject": [
                subject
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "session": [
                f"session-{session}"
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_subjects * n_sessions,
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=n_rows),
        }
    )


def experiment(*, regime: str, seed: int) -> dict[str, Any]:
    """Run one nested selection procedure and an outer-fold descriptive comparison."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    generator = _generator_model()
    paths = {
        "intercept": [-0.2, -0.2, -0.2],
        "stimulus": [1.1, 1.1, 1.1],
        "choice_lag_1": [0.25, 0.25, 0.25],
    }
    if regime == "shared_drift":
        paths = {
            "intercept": [-0.8, 0.0, 0.8],
            "stimulus": [0.2, 1.2, 2.4],
            "choice_lag_1": [0.25, 0.25, 0.25],
        }
    simulation = generator.simulate(
        build_design(seed=seed),
        generator.parameters_from_paths(paths),
        seed=seed + 1,
    )
    outer_splits = cohort_forward_session_splits(simulation, min_train_sessions=5)

    def inner_splitter(training: Study):
        return cohort_forward_session_splits(training, min_train_sessions=3)

    candidates = _candidates()
    nested = nested_select_model(
        candidates,
        simulation,
        outer_splits,
        inner_splitter,
        bootstrap_resamples=200,
        inner_bootstrap_resamples=100,
        bootstrap_seed=seed + 2,
    )
    outer = compare_models(
        candidates,
        simulation,
        outer_splits,
        bootstrap_resamples=200,
        bootstrap_seed=seed + 3,
    )
    expected = EXPECTED[regime]
    selections = tuple(fold.selected_model for fold in nested.folds)
    return {
        "expected_model": expected,
        "outer_fold_selections": list(selections),
        "n_expected_selections": selections.count(expected),
        "majority_recovered": selections.count(expected) > len(selections) / 2,
        "nested_outer_log_loss": nested.unit_balanced_log_loss,
        "fixed_candidate_outer_log_loss": {
            name: outer.result_for(name).unit_balanced_log_loss for name in CANDIDATES
        },
        "all_selected_fits_passed_audit": all(
            fold.outer_evaluation.fit.audit().status.value == "pass" for fold in nested.folds
        ),
        "inner_fold_counts": [len(fold.inner_report.splits) for fold in nested.folds],
    }


def run(*, repetitions: int = 20, seed: int = 84_221) -> dict[str, Any]:
    """Aggregate matched repetitions under each generating structure."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        experiments = [
            experiment(
                regime=regime,
                seed=seed + regime_index * 10_000 + repetition * 10,
            )
            for repetition in range(repetitions)
        ]
        expected = EXPECTED[regime]
        total_outer_folds = sum(len(result["outer_fold_selections"]) for result in experiments)
        n_expected = sum(result["n_expected_selections"] for result in experiments)
        regimes[regime] = {
            "expected_model": expected,
            "expected_selection_rate": n_expected / total_outer_folds,
            "majority_recovery_rate": float(
                np.mean([result["majority_recovered"] for result in experiments])
            ),
            "selection_counts": {
                candidate: sum(
                    result["outer_fold_selections"].count(candidate) for result in experiments
                )
                for candidate in CANDIDATES
            },
            "mean_nested_outer_log_loss": float(
                np.mean([result["nested_outer_log_loss"] for result in experiments])
            ),
            "mean_fixed_candidate_outer_log_loss": {
                candidate: float(
                    np.mean(
                        [
                            result["fixed_candidate_outer_log_loss"][candidate]
                            for result in experiments
                        ]
                    )
                )
                for candidate in CANDIDATES
            },
            "all_selected_fits_passed_audit": all(
                result["all_selected_fits_passed_audit"] for result in experiments
            ),
            "runs": experiments,
        }
    return {
        "benchmark": "nested prospective stationary-versus-drift selection recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "subjects": 10,
            "sessions": 7,
            "trials_per_session": 40,
            "outer_training_origins": [5, 6],
            "inner_minimum_training_sessions": 3,
            "knots": list(KNOTS),
        },
        "selection_contract": {
            "primary_metric": "mean subject-level inner-fold log loss",
            "scored_columns": ["choice"],
            "tie_break": "declared candidate order",
            "outer_test_outcomes_available_during_selection": False,
            "candidate_order": list(CANDIDATES),
        },
        "regimes": regimes,
        "all_regimes_majority_recovered": all(
            result["majority_recovery_rate"] >= 0.5 for result in regimes.values()
        ),
    }


def _generator_model() -> SmoothModel:
    return smooth(
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02),
        over="session_order",
        knots=KNOTS,
        smoothness=3.0,
        shared_trajectory=True,
    )


def _candidates() -> dict[str, Any]:
    return {
        "static": BernoulliHistoryGLM(
            predictors=("stimulus",),
            choice_lags=1,
            l2=0.02,
        ),
        "shared_smooth": _generator_model(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=84_221)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(repetitions=arguments.repetitions, seed=arguments.seed)
    rendered = render(result)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
