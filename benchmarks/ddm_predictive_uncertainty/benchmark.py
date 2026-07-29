"""Benchmark scale uncertainty and unseen-subject random-effect prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.provenance import render
from unspool import HierarchicalSmoothWienerDriftDiffusion, ModelDataError, Study

TRAINING_SUBJECTS = 8
UNSEEN_SUBJECTS = 4
N_SESSIONS = 3
TRIALS_PER_SESSION = 35
KNOTS = (0.0, 2.0)
TRUE_SCALES = {"drift.stimulus": 0.2, "boundary": 0.08}
INITIAL_SCALES = {"drift.stimulus": 0.14, "boundary": 0.14}
SCALE_BOUNDS = (0.03, 0.5)
PREDICTIVE_DRAWS = 4096


def build_design(*, n_subjects: int, prefix: str, seed: int) -> Study:
    """Construct one balanced three-session longitudinal design."""

    generator = np.random.default_rng(seed)
    subjects = tuple(f"{prefix}-{index}" for index in range(n_subjects))
    n_trials = n_subjects * N_SESSIONS * TRIALS_PER_SESSION
    return Study(
        {
            "subject": [
                subject
                for subject in subjects
                for _session in range(N_SESSIONS)
                for _trial in range(TRIALS_PER_SESSION)
            ],
            "session": [
                f"session-{session}"
                for _subject in subjects
                for session in range(N_SESSIONS)
                for _trial in range(TRIALS_PER_SESSION)
            ],
            "trial": list(range(TRIALS_PER_SESSION)) * n_subjects * N_SESSIONS,
            "session_order": [
                session
                for _subject in subjects
                for session in range(N_SESSIONS)
                for _trial in range(TRIALS_PER_SESSION)
            ],
            "stimulus": generator.normal(size=n_trials),
        }
    )


def experiment(*, seed: int) -> dict[str, Any]:
    """Run one matched interval and unseen-subject prediction experiment."""

    generator_model = _model(subject_scales=TRUE_SCALES)
    truth = _population_truth(generator_model)
    training = generator_model.simulate(
        build_design(n_subjects=TRAINING_SUBJECTS, prefix="train", seed=seed),
        truth,
        seed=seed + 1,
    )
    unseen = generator_model.simulate(
        build_design(n_subjects=UNSEEN_SUBJECTS, prefix="new", seed=seed + 2),
        truth,
        seed=seed + 3,
    )
    estimated_model = _model(
        subject_scales=INITIAL_SCALES,
        estimate=True,
        scale_uncertainty="supplemented",
    )
    supplemented_error = None
    try:
        fit = estimated_model.fit(training)
    except ModelDataError as error:
        if str(error) != "supplemented scale uncertainty requires a stable EM rate matrix":
            raise
        supplemented_error = str(error)
        estimated_model = _model(
            subject_scales=INITIAL_SCALES,
            estimate=True,
            scale_uncertainty="local",
        )
        fit = estimated_model.fit(training)
    predictive = estimated_model.predict_new_subjects(
        unseen,
        fit,
        n_draws=PREDICTIVE_DRAWS,
        seed=seed + 4,
    )

    supplemented_intervals = (
        None if supplemented_error is not None else fit.subject_scale_confidence_intervals_95
    )
    local_intervals = fit.subject_scale_local_confidence_intervals_95
    if local_intervals is None:
        raise AssertionError("estimated fit did not retain local scale intervals")
    if supplemented_error is None and supplemented_intervals is None:
        raise AssertionError("stable supplemented fit did not retain scale intervals")
    plugin_pointwise = estimated_model.pointwise_log_prob(unseen, fit)
    plugin_subject = {
        subject: float(np.sum(plugin_pointwise[unseen["subject"] == subject]))
        for subject in unseen.subjects
    }
    predictive_subject = predictive.subject_joint_log_probability_map
    subject_improvements = {
        subject: predictive_subject[subject] - plugin_subject[subject]
        for subject in unseen.subjects
    }
    return {
        "seed": seed,
        "fit_audit": fit.audit().to_dict(),
        "scale_estimation_converged": fit.scale_estimation_converged,
        "scale_em_spectral_radius": fit.subject_scale_em_spectral_radius,
        "supplemented_interval_error": supplemented_error,
        "scale_estimates": dict(fit.subject_scale_map),
        "local_intervals": {
            parameter: list(interval) for parameter, interval in local_intervals.items()
        },
        "supplemented_intervals": None
        if supplemented_intervals is None
        else {parameter: list(interval) for parameter, interval in supplemented_intervals.items()},
        "local_covers_truth": {
            parameter: interval[0] <= TRUE_SCALES[parameter] <= interval[1]
            for parameter, interval in local_intervals.items()
        },
        "supplemented_covers_truth": None
        if supplemented_intervals is None
        else {
            parameter: interval[0] <= TRUE_SCALES[parameter] <= interval[1]
            for parameter, interval in supplemented_intervals.items()
        },
        "plugin_subject_log_probability": plugin_subject,
        "predictive_subject_log_probability": dict(predictive_subject),
        "subject_log_probability_improvement": subject_improvements,
        "predictive_effective_draws": dict(
            zip(
                predictive.subjects,
                predictive.subject_effective_draws.tolist(),
                strict=True,
            )
        ),
        "predictive_log_probability_mcse": dict(
            zip(
                predictive.subjects,
                predictive.subject_log_probability_mcse.tolist(),
                strict=True,
            )
        ),
    }


def run(*, repetitions: int = 20, seed: int = 63_901) -> dict[str, Any]:
    """Aggregate finite calibration and unseen-subject prediction evidence."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    runs = [experiment(seed=seed + repetition * 100) for repetition in range(repetitions)]
    interval_summaries = {}
    for parameter in TRUE_SCALES:
        supplemented_runs = [run for run in runs if run["supplemented_covers_truth"] is not None]
        interval_summaries[parameter] = {
            "truth": TRUE_SCALES[parameter],
            "local_coverage": float(
                np.mean([run["local_covers_truth"][parameter] for run in runs])
            ),
            "supplemented_coverage": float(
                np.mean([run["supplemented_covers_truth"][parameter] for run in supplemented_runs])
            ),
            "supplemented_interval_denominator": len(supplemented_runs),
            "mean_local_width": float(
                np.mean(
                    [
                        run["local_intervals"][parameter][1] - run["local_intervals"][parameter][0]
                        for run in runs
                    ]
                )
            ),
            "mean_supplemented_width": float(
                np.mean(
                    [
                        run["supplemented_intervals"][parameter][1]
                        - run["supplemented_intervals"][parameter][0]
                        for run in supplemented_runs
                    ]
                )
            ),
        }
    improvements = np.asarray(
        [
            improvement
            for run in runs
            for improvement in run["subject_log_probability_improvement"].values()
        ],
        dtype=np.float64,
    )
    effective_draws = np.asarray(
        [value for run in runs for value in run["predictive_effective_draws"].values()],
        dtype=np.float64,
    )
    monte_carlo_errors = np.asarray(
        [value for run in runs for value in run["predictive_log_probability_mcse"].values()],
        dtype=np.float64,
    )
    return {
        "benchmark": "longitudinal DDM scale and unseen-subject predictive uncertainty",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "training_subjects": TRAINING_SUBJECTS,
            "unseen_subjects_per_run": UNSEEN_SUBJECTS,
            "sessions": N_SESSIONS,
            "trials_per_session": TRIALS_PER_SESSION,
            "knots": list(KNOTS),
            "true_scales": TRUE_SCALES,
            "initial_scales": INITIAL_SCALES,
            "scale_bounds": list(SCALE_BOUNDS),
            "predictive_draws": PREDICTIVE_DRAWS,
        },
        "scope": {
            "local_interval": "expected-prior curvature",
            "supplemented_interval": "forced-EM observed-information correction",
            "predictive": "empirical-Bayes integration over one random path per new subject",
            "joint_score": "Monte Carlo log mean of each subject likelihood",
            "parameter_uncertainty_in_prediction": "not integrated",
        },
        "intervals": interval_summaries,
        "prediction": {
            "n_unseen_subjects": len(improvements),
            "mean_joint_log_probability_improvement": float(np.mean(improvements)),
            "median_joint_log_probability_improvement": float(np.median(improvements)),
            "random_effect_win_rate": float(np.mean(improvements > 0)),
            "median_effective_draws": float(np.median(effective_draws)),
            "minimum_effective_draws": float(np.min(effective_draws)),
            "maximum_log_probability_mcse": float(np.max(monte_carlo_errors)),
        },
        "diagnostics": {
            "all_outer_fits_converged": all(
                run["fit_audit"]["numerical"]["converged"] and run["scale_estimation_converged"]
                for run in runs
            ),
            "supplemented_interval_successes": int(
                np.sum([run["supplemented_interval_error"] is None for run in runs])
            ),
            "supplemented_interval_failures": int(
                np.sum([run["supplemented_interval_error"] is not None for run in runs])
            ),
            "mean_scale_em_spectral_radius": float(
                np.mean(
                    [
                        run["scale_em_spectral_radius"]
                        for run in runs
                        if run["scale_em_spectral_radius"] is not None
                    ]
                )
            ),
            "maximum_scale_em_spectral_radius": float(
                np.max(
                    [
                        run["scale_em_spectral_radius"]
                        for run in runs
                        if run["scale_em_spectral_radius"] is not None
                    ]
                )
            ),
        },
        "runs": runs,
    }


def _model(
    *,
    subject_scales: dict[str, float],
    estimate: bool = False,
    scale_uncertainty: str = "local",
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
        scale_tolerance=0.04,
        subject_scale_uncertainty=scale_uncertainty,
        subject_smoothness=8.0,
        n_restarts=1,
        max_iterations=350,
        tolerance=1e-6,
        simulation_time_step=0.001,
    )


def _population_truth(model: HierarchicalSmoothWienerDriftDiffusion) -> dict[str, float]:
    return model.parameters_from_paths(
        {
            "drift.intercept": 0.1,
            "drift.stimulus": [0.65, 1.3],
            "boundary": [1.4, 1.1],
            "starting_bias": 0.48,
            "nondecision_time": 0.25,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=63_901)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("result.json"),
    )
    arguments = parser.parse_args()
    result = run(repetitions=arguments.repetitions, seed=arguments.seed)
    rendered = render(result, allow_nan=True, sort_keys=False)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
