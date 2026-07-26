"""Benchmark parameter-specific longitudinal DDM scale recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from unspool import HierarchicalSmoothWienerDriftDiffusion, Study

SUBJECT_COUNTS = (6, 12)
TRUE_SCALES = {"drift.stimulus": 0.22, "boundary": 0.07}
INITIAL_SCALES = {"drift.stimulus": 0.14, "boundary": 0.14}
SCALE_BOUNDS = (0.03, 0.5)
KNOTS = (0.0, 3.0)
TRAINING_SESSIONS = 3
TRIALS_PER_SESSION = 45


def build_design(*, n_subjects: int, seed: int) -> Study:
    """Construct a balanced prospective choice/response-time design."""

    generator = np.random.default_rng(seed)
    n_sessions = TRAINING_SESSIONS + 1
    n_trials = n_subjects * n_sessions * TRIALS_PER_SESSION
    subjects = tuple(f"mouse-{index}" for index in range(n_subjects))
    return Study(
        {
            "subject": [
                subject
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(TRIALS_PER_SESSION)
            ],
            "session": [
                f"session-{session}"
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(TRIALS_PER_SESSION)
            ],
            "trial": list(range(TRIALS_PER_SESSION)) * n_subjects * n_sessions,
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(TRIALS_PER_SESSION)
            ],
            "stimulus": generator.normal(size=n_trials),
        }
    )


def experiment(*, n_subjects: int, seed: int) -> dict[str, Any]:
    """Run one training-only scale recovery and future-session comparison."""

    design = build_design(n_subjects=n_subjects, seed=seed)
    generator = _model(subject_scales=TRUE_SCALES)
    study = generator.simulate(design, _population_truth(generator), seed=seed + 1)
    train, test = _prospective_partition(study)

    estimated_model = _model(subject_scales=INITIAL_SCALES, estimate=True)
    estimated_fit = estimated_model.fit(train)
    oracle_model = _model(subject_scales=TRUE_SCALES)
    oracle_fit = oracle_model.fit(train)
    intervals = estimated_fit.subject_scale_confidence_intervals_95
    standard_errors = estimated_fit.subject_scale_standard_error_map
    boundary = estimated_fit.subject_scale_at_boundary_map
    if intervals is None or standard_errors is None or boundary is None:
        raise AssertionError("estimated fit did not retain scale diagnostics")

    estimates = dict(estimated_fit.subject_scale_map)
    return {
        "seed": seed,
        "scale_estimates": estimates,
        "scale_standard_errors": dict(standard_errors),
        "intervals_95": {parameter: list(interval) for parameter, interval in intervals.items()},
        "interval_covers_truth": {
            parameter: interval[0] <= TRUE_SCALES[parameter] <= interval[1]
            for parameter, interval in intervals.items()
        },
        "scale_at_boundary": dict(boundary),
        "scale_estimation_converged": estimated_fit.scale_estimation_converged,
        "scale_estimation_iterations": estimated_fit.scale_estimation_iterations,
        "fit_converged": estimated_fit.diagnostics.converged,
        "fit_audit": estimated_fit.audit().to_dict(),
        "estimated_prospective_log_loss": -float(
            np.mean(estimated_model.pointwise_log_prob(test, estimated_fit))
        ),
        "oracle_prospective_log_loss": -float(
            np.mean(oracle_model.pointwise_log_prob(test, oracle_fit))
        ),
    }


def run(*, repetitions: int = 8, seed: int = 91_337) -> dict[str, Any]:
    """Aggregate scale recovery over two population sizes."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    regimes: dict[str, Any] = {}
    for count_index, n_subjects in enumerate(SUBJECT_COUNTS):
        runs = [
            experiment(
                n_subjects=n_subjects,
                seed=seed + count_index * 100_000 + repetition * 10,
            )
            for repetition in range(repetitions)
        ]
        parameter_summaries = {}
        squared_errors = []
        for parameter, truth in TRUE_SCALES.items():
            estimates = np.asarray(
                [result["scale_estimates"][parameter] for result in runs],
                dtype=np.float64,
            )
            errors = estimates - truth
            squared_errors.extend((errors**2).tolist())
            parameter_summaries[parameter] = {
                "truth": truth,
                "mean_estimate": float(np.mean(estimates)),
                "bias": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_standard_error": float(
                    np.mean([result["scale_standard_errors"][parameter] for result in runs])
                ),
                "coverage_95": float(
                    np.mean([result["interval_covers_truth"][parameter] for result in runs])
                ),
                "boundary_rate": float(
                    np.mean([result["scale_at_boundary"][parameter] for result in runs])
                ),
            }
        estimated_loss = np.asarray([result["estimated_prospective_log_loss"] for result in runs])
        oracle_loss = np.asarray([result["oracle_prospective_log_loss"] for result in runs])
        regimes[str(n_subjects)] = {
            "n_subjects": n_subjects,
            "joint_scale_rmse": float(np.sqrt(np.mean(squared_errors))),
            "scale_estimation_convergence_rate": float(
                np.mean([result["scale_estimation_converged"] for result in runs])
            ),
            "fit_convergence_rate": float(np.mean([result["fit_converged"] for result in runs])),
            "mean_scale_iterations": float(
                np.mean([result["scale_estimation_iterations"] for result in runs])
            ),
            "mean_estimated_prospective_log_loss": float(np.mean(estimated_loss)),
            "mean_oracle_prospective_log_loss": float(np.mean(oracle_loss)),
            "mean_excess_log_loss": float(np.mean(estimated_loss - oracle_loss)),
            "parameters": parameter_summaries,
            "runs": runs,
        }
    return {
        "benchmark": "parameter-specific longitudinal DDM scale recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "subject_counts": list(SUBJECT_COUNTS),
            "training_sessions": TRAINING_SESSIONS,
            "test_sessions": 1,
            "trials_per_session": TRIALS_PER_SESSION,
            "knots": list(KNOTS),
            "true_scales": TRUE_SCALES,
            "initial_scales": INITIAL_SCALES,
            "scale_bounds": list(SCALE_BOUNDS),
        },
        "scope": {
            "estimator": "bounded parameter-specific Laplace-EM",
            "interval": "local expected-prior curvature on log scale",
            "oracle": "fixed to each true generating scale",
            "prediction": "fourth session for subjects represented in training",
        },
        "regimes": regimes,
        "more_subjects_reduce_joint_scale_rmse": (
            regimes[str(SUBJECT_COUNTS[1])]["joint_scale_rmse"]
            < regimes[str(SUBJECT_COUNTS[0])]["joint_scale_rmse"]
        ),
    }


def _model(
    *,
    subject_scales: dict[str, float],
    estimate: bool = False,
) -> HierarchicalSmoothWienerDriftDiffusion:
    return HierarchicalSmoothWienerDriftDiffusion(
        covariates=("stimulus",),
        knots=KNOTS,
        varying_parameters=("drift.stimulus", "boundary"),
        subject_parameters=("drift.stimulus", "boundary"),
        smoothness=8.0,
        subject_parameter_scales=subject_scales,
        estimate_subject_scales=estimate,
        subject_scale_bounds=SCALE_BOUNDS,
        scale_max_iterations=20,
        scale_tolerance=0.03,
        subject_smoothness=8.0,
        n_restarts=1,
        max_iterations=400,
        tolerance=1e-6,
        simulation_time_step=0.001,
    )


def _population_truth(model: HierarchicalSmoothWienerDriftDiffusion) -> dict[str, float]:
    return model.parameters_from_paths(
        {
            "drift.intercept": 0.1,
            "drift.stimulus": [0.65, 1.2],
            "boundary": [1.4, 1.15],
            "starting_bias": 0.48,
            "nondecision_time": 0.25,
        }
    )


def _prospective_partition(study: Study) -> tuple[Study, Study]:
    train = study.take(np.flatnonzero(study["session_order"] < TRAINING_SESSIONS))
    test = study.take(np.flatnonzero(study["session_order"] == TRAINING_SESSIONS))
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91_337)
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
