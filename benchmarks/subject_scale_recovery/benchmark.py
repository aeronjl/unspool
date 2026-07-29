"""Benchmark Laplace subject-scale recovery and future-session prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.hierarchical_glm.benchmark import POPULATION_PARAMETERS, build_design
from benchmarks.provenance import render
from unspool import HierarchicalBernoulliHistoryGLM, Study

SUBJECT_COUNTS = (8, 24)
TRUE_SCALES = (0.1, 0.5, 1.0)
SCALE_BOUNDS = (0.05, 1.5)
INITIAL_SCALE = 0.4


def experiment(*, n_subjects: int, true_scale: float, seed: int) -> dict[str, Any]:
    """Run one matched scale-recovery and future-session experiment."""

    design = build_design(
        seed=seed,
        n_subjects=n_subjects,
        n_sessions=4,
        trials_per_session=35,
    )
    generator = _model(subject_scale=true_scale)
    study = generator.simulate(design, POPULATION_PARAMETERS, seed=seed + 1)
    train, test = _prospective_partition(study)

    estimated_model = _model(
        subject_scale=INITIAL_SCALE,
        estimate_subject_scale=True,
    )
    estimated_fit = estimated_model.fit(train)
    oracle_model = _model(subject_scale=true_scale)
    oracle_fit = oracle_model.fit(train)

    interval = estimated_fit.subject_scale_confidence_interval_95
    if interval is None:
        raise AssertionError("estimated scale fit did not retain an uncertainty interval")
    estimated_prediction = estimated_model.predict(test, estimated_fit)
    oracle_prediction = oracle_model.predict(test, oracle_fit)
    outcomes = np.asarray(test["choice"], dtype=np.float64)
    return {
        "scale_estimate": estimated_fit.subject_scale,
        "scale_standard_error": estimated_fit.subject_scale_standard_error,
        "interval_95": list(interval),
        "interval_covers_truth": interval[0] <= true_scale <= interval[1],
        "scale_at_boundary": estimated_fit.subject_scale_at_boundary,
        "fit_converged": estimated_fit.diagnostics.converged,
        "estimated_prospective_log_loss": -float(
            np.mean(estimated_model.pointwise_log_prob(test, estimated_fit))
        ),
        "oracle_prospective_log_loss": -float(
            np.mean(oracle_model.pointwise_log_prob(test, oracle_fit))
        ),
        "estimated_brier_score": float(np.mean((estimated_prediction.probability - outcomes) ** 2)),
        "oracle_brier_score": float(np.mean((oracle_prediction.probability - outcomes) ** 2)),
    }


def run(*, repetitions: int = 20, seed: int = 2741) -> dict[str, Any]:
    """Aggregate recovery over sample size and true-scale regimes."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    regimes: dict[str, Any] = {}
    for count_index, n_subjects in enumerate(SUBJECT_COUNTS):
        for scale_index, true_scale in enumerate(TRUE_SCALES):
            runs = [
                experiment(
                    n_subjects=n_subjects,
                    true_scale=true_scale,
                    seed=(seed + count_index * 100_000 + scale_index * 10_000 + repetition * 10),
                )
                for repetition in range(repetitions)
            ]
            estimates = np.asarray([result["scale_estimate"] for result in runs])
            estimated_log_loss = np.asarray(
                [result["estimated_prospective_log_loss"] for result in runs]
            )
            oracle_log_loss = np.asarray([result["oracle_prospective_log_loss"] for result in runs])
            estimated_brier = np.asarray([result["estimated_brier_score"] for result in runs])
            oracle_brier = np.asarray([result["oracle_brier_score"] for result in runs])
            regime_name = f"subjects={n_subjects},scale={true_scale}"
            regimes[regime_name] = {
                "n_subjects": n_subjects,
                "true_scale": true_scale,
                "mean_estimate": float(np.mean(estimates)),
                "bias": float(np.mean(estimates - true_scale)),
                "rmse": float(np.sqrt(np.mean((estimates - true_scale) ** 2))),
                "mean_standard_error": float(
                    np.mean([result["scale_standard_error"] for result in runs])
                ),
                "coverage_95": float(np.mean([result["interval_covers_truth"] for result in runs])),
                "boundary_rate": float(np.mean([result["scale_at_boundary"] for result in runs])),
                "convergence_rate": float(np.mean([result["fit_converged"] for result in runs])),
                "mean_estimated_prospective_log_loss": float(np.mean(estimated_log_loss)),
                "mean_oracle_prospective_log_loss": float(np.mean(oracle_log_loss)),
                "mean_excess_log_loss": float(np.mean(estimated_log_loss - oracle_log_loss)),
                "mean_estimated_brier_score": float(np.mean(estimated_brier)),
                "mean_oracle_brier_score": float(np.mean(oracle_brier)),
                "mean_excess_brier_score": float(np.mean(estimated_brier - oracle_brier)),
            }

    more_subjects_reduce_rmse = {
        str(true_scale): (
            regimes[f"subjects=24,scale={true_scale}"]["rmse"]
            < regimes[f"subjects=8,scale={true_scale}"]["rmse"]
        )
        for true_scale in TRUE_SCALES
    }
    return {
        "benchmark": "Laplace subject-scale recovery and calibration",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "subject_counts": list(SUBJECT_COUNTS),
            "true_scales": list(TRUE_SCALES),
            "training_sessions": 3,
            "test_sessions": 1,
            "trials_per_session": 35,
            "initial_scale": INITIAL_SCALE,
            "scale_bounds": list(SCALE_BOUNDS),
        },
        "scope": {
            "estimator": "bounded Laplace marginal likelihood with one shared scale",
            "interval": "local-Hessian delta-method 95% interval on log scale",
            "oracle": "fixed to the true generating scale",
            "prediction": "one future session for subjects represented in training",
        },
        "regimes": regimes,
        "more_subjects_reduce_rmse": more_subjects_reduce_rmse,
    }


def _model(
    *,
    subject_scale: float,
    estimate_subject_scale: bool = False,
) -> HierarchicalBernoulliHistoryGLM:
    return HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.05,
        subject_scale=subject_scale,
        estimate_subject_scale=estimate_subject_scale,
        subject_scale_bounds=SCALE_BOUNDS,
    )


def _prospective_partition(study: Study) -> tuple[Study, Study]:
    train_indices = np.flatnonzero(study["session_order"] < 3)
    test_indices = np.flatnonzero(study["session_order"] == 3)
    return study.take(train_indices), study.take(test_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2741)
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
