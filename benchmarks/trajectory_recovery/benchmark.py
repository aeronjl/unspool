"""Compare five longitudinal models across four trajectory-generating regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.provenance import render
from unspool import (
    BernoulliHistoryGLM,
    HierarchicalBernoulliHistoryGLM,
    HierarchicalSmoothBernoulliHistoryGLM,
    SmoothBernoulliHistoryGLM,
    Study,
)

KNOTS = (0.0, 2.0, 4.0)
REGIMES = ("stationary_identical", "stable_individual", "shared_drift", "individual_drift")
METHODS = (
    "complete_pooling",
    "static_partial_pooling",
    "shared_smooth",
    "independent_smooth",
    "hierarchical_smooth",
)
EXPECTED_WINNER = {
    "stationary_identical": "complete_pooling",
    "stable_individual": "static_partial_pooling",
    "shared_drift": "shared_smooth",
    "individual_drift": "hierarchical_smooth",
}


def build_design(
    *, seed: int, n_subjects: int = 12, n_sessions: int = 5, trials_per_session: int = 50
) -> Study:
    """Construct a balanced design with a fixed session clock."""

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


def experiment(*, regime: str, seed: int) -> dict[str, dict[str, Any]]:
    """Run one matched prospective comparison for a generating regime."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    design = build_design(seed=seed)
    hierarchical_smooth = _hierarchical_smooth_model()
    population_paths, subject_deviations = _truth(regime, design.subjects, seed=seed + 1)
    simulation = hierarchical_smooth.simulate_with_effects(
        design,
        hierarchical_smooth.parameters_from_paths(population_paths),
        seed=seed + 2,
        subject_deviation_paths=subject_deviations,
    )
    train_indices = np.flatnonzero(simulation.study["session_order"] < 4)
    test_indices = np.flatnonzero(simulation.study["session_order"] == 4)
    train = simulation.study.take(train_indices)
    test = simulation.study.take(test_indices)
    truth = simulation.subject_knot_values

    complete = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=0.02)
    complete_fit = complete.fit(train)
    complete_paths = np.broadcast_to(complete_fit.estimates[None, :, None], truth.shape)

    static_partial = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",), choice_lags=0, l2=0.02, subject_scale=0.4
    )
    static_fit = static_partial.fit(train)
    static_paths = np.broadcast_to(static_fit.subject_coefficients[:, :, None], truth.shape)

    shared_smooth = SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=0,
        knots=KNOTS,
        smoothness=3.0,
        l2=0.02,
        shared_trajectory=True,
    )
    shared_fit = shared_smooth.fit(train)
    shared_paths = np.broadcast_to(shared_fit.estimates.reshape(2, 3)[None, :, :], truth.shape)

    hierarchical_fit = hierarchical_smooth.fit(train)

    independent_paths: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    independent_scores: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    independent_converged = True
    for subject in simulation.subjects:
        subject_train = _subject_study(train, subject)
        subject_test = _subject_study(test, subject)
        independent = SmoothBernoulliHistoryGLM(
            covariates=("stimulus",),
            choice_lags=0,
            knots=KNOTS,
            smoothness=3.0,
            l2=0.02,
        )
        independent_fit = independent.fit(subject_train)
        independent_paths.append(independent_fit.estimates.reshape(2, 3))
        independent_scores.append(independent.pointwise_log_prob(subject_test, independent_fit))
        independent_converged = independent_converged and independent_fit.diagnostics.converged

    paths = {
        "complete_pooling": complete_paths,
        "static_partial_pooling": static_paths,
        "shared_smooth": shared_paths,
        "independent_smooth": np.stack(independent_paths),
        "hierarchical_smooth": hierarchical_fit.subject_knot_values,
    }
    log_losses = {
        "complete_pooling": -float(np.mean(complete.pointwise_log_prob(test, complete_fit))),
        "static_partial_pooling": -float(
            np.mean(static_partial.pointwise_log_prob(test, static_fit))
        ),
        "shared_smooth": -float(np.mean(shared_smooth.pointwise_log_prob(test, shared_fit))),
        "independent_smooth": -float(np.mean(np.concatenate(independent_scores))),
        "hierarchical_smooth": -float(
            np.mean(hierarchical_smooth.pointwise_log_prob(test, hierarchical_fit))
        ),
    }
    convergence = {
        "complete_pooling": complete_fit.diagnostics.converged,
        "static_partial_pooling": static_fit.diagnostics.converged,
        "shared_smooth": shared_fit.diagnostics.converged,
        "independent_smooth": independent_converged,
        "hierarchical_smooth": hierarchical_fit.diagnostics.converged,
    }
    return {
        method: {
            "subject_trajectory_rmse": float(np.sqrt(np.mean((paths[method] - truth) ** 2))),
            "prospective_log_loss": log_losses[method],
            "all_fits_converged": convergence[method],
        }
        for method in METHODS
    }


def run(*, repetitions: int = 20, seed: int = 5419) -> dict[str, Any]:
    """Aggregate matched repetitions and retain the winning model per regime."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        runs = [
            experiment(regime=regime, seed=seed + regime_index * 10_000 + repeat * 10)
            for repeat in range(repetitions)
        ]
        methods = {
            method: {
                "mean_subject_trajectory_rmse": float(
                    np.mean([result[method]["subject_trajectory_rmse"] for result in runs])
                ),
                "mean_prospective_log_loss": float(
                    np.mean([result[method]["prospective_log_loss"] for result in runs])
                ),
                "all_fits_converged": all(result[method]["all_fits_converged"] for result in runs),
            }
            for method in METHODS
        }
        regimes[regime] = {
            "expected_winner": EXPECTED_WINNER[regime],
            "trajectory_rmse_winner": min(
                methods, key=lambda method: methods[method]["mean_subject_trajectory_rmse"]
            ),
            "prospective_log_loss_winner": min(
                methods, key=lambda method: methods[method]["mean_prospective_log_loss"]
            ),
            "methods": methods,
        }
    return {
        "benchmark": "factorial population and individual trajectory recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "subjects": 12,
            "training_sessions": 4,
            "test_sessions": 1,
            "trials_per_session": 50,
            "knots": list(KNOTS),
        },
        "scope": {
            "hyperparameters": "fixed equally across matched repetitions",
            "prediction": "final session for subjects represented in training",
            "trajectory_metric": "RMSE over realized subject coefficient values at knots",
        },
        "regimes": regimes,
        "all_expected_winners_recovered": all(
            result[metric] == result["expected_winner"]
            for result in regimes.values()
            for metric in ("trajectory_rmse_winner", "prospective_log_loss_winner")
        ),
    }


def _truth(
    regime: str, subjects: tuple[Any, ...], *, seed: int
) -> tuple[dict[str, list[float]], dict[Any, dict[str, list[float]]]]:
    generator = np.random.default_rng(seed)
    population = {
        "intercept": [-0.2, -0.2, -0.2],
        "stimulus": [1.0, 1.0, 1.0],
    }
    deviations = {
        subject: {"intercept": [0.0, 0.0, 0.0], "stimulus": [0.0, 0.0, 0.0]} for subject in subjects
    }
    if regime == "stable_individual":
        for subject in subjects:
            for coefficient in population:
                offset = float(generator.normal(0.0, 0.4))
                deviations[subject][coefficient] = [offset, offset, offset]
    elif regime == "shared_drift":
        population = {
            "intercept": [-0.5, -0.2, 0.1],
            "stimulus": [0.4, 1.0, 1.6],
        }
    elif regime == "individual_drift":
        population = {
            "intercept": [-0.3, -0.2, -0.1],
            "stimulus": [0.8, 1.0, 1.2],
        }
        centered_knots = np.asarray([-1.0, 0.0, 1.0])
        for subject in subjects:
            for coefficient in population:
                offset = generator.normal(0.0, 0.2)
                slope = generator.normal(0.0, 0.35)
                deviations[subject][coefficient] = (offset + slope * centered_knots).tolist()
    return population, deviations


def _hierarchical_smooth_model() -> HierarchicalSmoothBernoulliHistoryGLM:
    return HierarchicalSmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=0,
        knots=KNOTS,
        smoothness=3.0,
        l2=0.02,
        subject_scale=0.4,
        subject_smoothness=3.0,
    )


def _subject_study(study: Study, subject: Any) -> Study:
    return study.take(np.flatnonzero(study["subject"] == subject))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5419)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(repetitions=arguments.repetitions, seed=arguments.seed)
    rendered = render(result, allow_nan=True, sort_keys=False)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
