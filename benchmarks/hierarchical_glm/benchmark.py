"""Compare complete pooling, independent fits, and fixed-scale partial pooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from unspool import BernoulliHistoryGLM, HierarchicalBernoulliHistoryGLM, Study

POPULATION_PARAMETERS = {
    "intercept": -0.2,
    "stimulus": 1.0,
    "choice_lag_1": 0.35,
}
HETEROGENEITY_SCALES = (0.1, 0.5, 1.0)
METHODS = ("complete_pooling", "independent", "partial_pooling")


def build_design(
    *,
    seed: int,
    n_subjects: int = 12,
    n_sessions: int = 4,
    trials_per_session: int = 35,
) -> Study:
    """Construct a balanced longitudinal design without generating outcomes."""

    generator = np.random.default_rng(seed)
    subject: list[str] = []
    session: list[str] = []
    trial: list[int] = []
    session_order: list[int] = []
    stimulus: list[float] = []
    for subject_index in range(n_subjects):
        subject_name = f"mouse-{subject_index:02d}"
        for order in range(n_sessions):
            subject.extend([subject_name] * trials_per_session)
            session.extend([f"session-{order}"] * trials_per_session)
            trial.extend(range(trials_per_session))
            session_order.extend([order] * trials_per_session)
            stimulus.extend(generator.normal(size=trials_per_session))
    return Study(
        {
            "subject": subject,
            "session": session,
            "trial": trial,
            "session_order": session_order,
            "stimulus": stimulus,
        }
    )


def experiment(*, subject_scale: float, seed: int) -> dict[str, dict[str, Any]]:
    """Run one matched prospective comparison at a fixed generative scale."""

    hierarchical = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.05,
        subject_scale=subject_scale,
    )
    static = BernoulliHistoryGLM(
        covariates=hierarchical.covariates,
        choice_lags=hierarchical.choice_lags,
        l2=hierarchical.l2,
    )
    design = build_design(seed=seed)
    simulation = hierarchical.simulate_with_effects(
        design,
        POPULATION_PARAMETERS,
        seed=seed + 1,
    )
    train_indices = np.flatnonzero(simulation.study["session_order"] < 3)
    test_indices = np.flatnonzero(simulation.study["session_order"] == 3)
    train = simulation.study.take(train_indices)
    test = simulation.study.take(test_indices)
    truth = simulation.subject_coefficients

    complete_fit = static.fit(train)
    complete_coefficients = np.broadcast_to(complete_fit.estimates, truth.shape)
    partial_fit = hierarchical.fit(train)

    independent_coefficients: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    independent_log_probabilities: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    independent_converged = True
    for subject in simulation.subjects:
        subject_train = _subject_study(train, subject)
        subject_test = _subject_study(test, subject)
        subject_fit = static.fit(subject_train)
        independent_coefficients.append(subject_fit.estimates)
        independent_log_probabilities.append(static.pointwise_log_prob(subject_test, subject_fit))
        independent_converged = independent_converged and subject_fit.diagnostics.converged

    coefficients = {
        "complete_pooling": complete_coefficients,
        "independent": np.vstack(independent_coefficients),
        "partial_pooling": partial_fit.subject_coefficients,
    }
    mean_log_losses = {
        "complete_pooling": -float(np.mean(static.pointwise_log_prob(test, complete_fit))),
        "independent": -float(np.mean(np.concatenate(independent_log_probabilities))),
        "partial_pooling": -float(np.mean(hierarchical.pointwise_log_prob(test, partial_fit))),
    }
    convergence = {
        "complete_pooling": complete_fit.diagnostics.converged,
        "independent": independent_converged,
        "partial_pooling": partial_fit.diagnostics.converged,
    }
    return {
        method: {
            "subject_coefficient_rmse": float(
                np.sqrt(np.mean((coefficients[method] - truth) ** 2))
            ),
            "prospective_log_loss": mean_log_losses[method],
            "all_fits_converged": convergence[method],
        }
        for method in METHODS
    }


def run(*, repetitions: int = 20, seed: int = 8617) -> dict[str, Any]:
    """Aggregate matched repetitions across low, moderate, and high heterogeneity."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    regimes: dict[str, Any] = {}
    for scale_index, subject_scale in enumerate(HETEROGENEITY_SCALES):
        runs = [
            experiment(
                subject_scale=subject_scale,
                seed=seed + scale_index * 10_000 + repetition * 10,
            )
            for repetition in range(repetitions)
        ]
        regimes[str(subject_scale)] = {
            "subject_scale": subject_scale,
            "methods": {
                method: {
                    "mean_subject_coefficient_rmse": float(
                        np.mean(
                            [run_result[method]["subject_coefficient_rmse"] for run_result in runs]
                        )
                    ),
                    "mean_prospective_log_loss": float(
                        np.mean([run_result[method]["prospective_log_loss"] for run_result in runs])
                    ),
                    "all_fits_converged": all(
                        run_result[method]["all_fits_converged"] for run_result in runs
                    ),
                }
                for method in METHODS
            },
        }

    partial_wins = {
        metric: sum(
            regime["methods"]["partial_pooling"][metric]
            < min(
                regime["methods"]["complete_pooling"][metric],
                regime["methods"]["independent"][metric],
            )
            for regime in regimes.values()
        )
        for metric in ("mean_subject_coefficient_rmse", "mean_prospective_log_loss")
    }
    return {
        "benchmark": "fixed-scale hierarchical Bernoulli GLM recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "subjects": 12,
            "training_sessions": 3,
            "test_sessions": 1,
            "trials_per_session": 35,
        },
        "scope": {
            "subject_scale": "fixed to the known generative value before fitting",
            "prediction": "one future session for subjects represented in training",
            "claim": "MAP shrinkage mechanics, not learned variance or posterior prediction",
        },
        "regimes": regimes,
        "partial_pooling_wins": partial_wins,
    }


def _subject_study(study: Study, subject: Any) -> Study:
    indices = np.flatnonzero(study["subject"] == subject)
    return study.take(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8617)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("result.json"),
    )
    arguments = parser.parse_args()
    result = run(repetitions=arguments.repetitions, seed=arguments.seed)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
