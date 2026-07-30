"""Benchmark threshold-landmark resolution under decisive and marginal learning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from behavio import Study
from behavio.time import ClockKind, ClockSpec, ThresholdLandmarkClock
from benchmarks.provenance import render

DEFAULT_REPETITIONS = 30
DEFAULT_RESAMPLES = 200
ROOT_SEED = 12401
REGIMES = {
    "decisive": (0.15, 0.95),
    "marginal": (0.15, 0.72),
}


def build_study(*, regime: str, seed: int) -> Study:
    """Simulate one four-session binary performance trajectory."""

    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {tuple(REGIMES)}")
    n_sessions = 4
    trials_per_session = 30
    n_trials = n_sessions * trials_per_session
    transition = 50
    before, after = REGIMES[regime]
    probability = np.full(n_trials, before, dtype=np.float64)
    probability[transition:] = after
    correct = np.random.default_rng(seed).binomial(1, probability)
    return Study(
        {
            "subject": ["animal-0"] * n_trials,
            "session": [
                f"session-{session}"
                for session in range(n_sessions)
                for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions,
            "session_order": np.repeat(np.arange(n_sessions), trials_per_session),
            "cumulative_trial": np.arange(n_trials, dtype=np.float64),
            "correct": correct,
        }
    )


def experiment(*, regime: str, seed: int, n_resamples: int = DEFAULT_RESAMPLES) -> dict[str, Any]:
    """Fit one point landmark and its frozen training-only uncertainty distribution."""

    study = build_study(regime=regime, seed=seed)
    transform = ThresholdLandmarkClock(
        clock=ClockSpec(
            "cumulative_trial",
            ClockKind.CUMULATIVE_TRIAL,
            unit="observed_trial",
        ),
        metric="correct",
        output="trials_since_learning",
        threshold=0.8,
        window=15,
        consecutive=3,
        on_missing="nan",
    )
    fitted = transform.fit_with_uncertainty(
        study,
        n_resamples=n_resamples,
        seed=seed + 1,
        smoothing_window=7,
        interval_level=0.9,
    )
    if fitted.uncertainty is None:
        raise AssertionError("uncertainty-aware fit did not retain bootstrap evidence")
    estimate = fitted.uncertainty.estimates["animal-0"]
    clock_samples = fitted.transform_samples(study)
    interval = estimate.interval
    final_clock = clock_samples.values[:, -1]
    return {
        "point_landmark": estimate.point,
        "point_resolved": estimate.point is not None,
        "bootstrap_resolution_rate": estimate.resolution_rate,
        "bootstrap_median": estimate.median,
        "bootstrap_interval_90": None if interval is None else list(interval),
        "bootstrap_interval_width": None if interval is None else interval[1] - interval[0],
        "n_resolved": estimate.n_resolved,
        "n_resamples": len(estimate.samples),
        "final_clock_resolution_rate": float(np.mean(np.isfinite(final_clock))),
        "point_matches_transform": bool(
            estimate.point == fitted.landmarks["animal-0"]
            and np.isclose(
                fitted.transform(study).study["trials_since_learning"][-1],
                np.nan if estimate.point is None else 119.0 - estimate.point,
                equal_nan=True,
            )
        ),
    }


def run(
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = ROOT_SEED,
    check: bool = True,
) -> dict[str, Any]:
    """Aggregate landmark resolution over matched decisive and marginal regimes."""

    if repetitions < 1 or n_resamples < 1:
        raise ValueError("repetitions and n_resamples must be positive")
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        runs = [
            experiment(
                regime=regime,
                seed=seed + regime_index * 100_000 + repetition * 100,
                n_resamples=n_resamples,
            )
            for repetition in range(repetitions)
        ]
        interval_widths = [
            item["bootstrap_interval_width"]
            for item in runs
            if item["bootstrap_interval_width"] is not None
        ]
        point_landmarks = [
            item["point_landmark"] for item in runs if item["point_landmark"] is not None
        ]
        regimes[regime] = {
            "probability_before_after": list(REGIMES[regime]),
            "point_resolution_rate": float(np.mean([item["point_resolved"] for item in runs])),
            "mean_point_landmark_when_resolved": (
                None if not point_landmarks else float(np.mean(point_landmarks))
            ),
            "mean_bootstrap_resolution_rate": float(
                np.mean([item["bootstrap_resolution_rate"] for item in runs])
            ),
            "minimum_bootstrap_resolution_rate": float(
                np.min([item["bootstrap_resolution_rate"] for item in runs])
            ),
            "mean_interval_width_when_resolved": (
                None if not interval_widths else float(np.mean(interval_widths))
            ),
            "interval_available_rate": len(interval_widths) / repetitions,
            "point_transform_agreement_rate": float(
                np.mean([item["point_matches_transform"] for item in runs])
            ),
            "clock_sample_resolution_matches_landmark_rate": float(
                np.mean(
                    [
                        np.isclose(
                            item["final_clock_resolution_rate"],
                            item["bootstrap_resolution_rate"],
                        )
                        for item in runs
                    ]
                )
            ),
        }

    decisive = regimes["decisive"]
    marginal = regimes["marginal"]
    contract = {
        "point_estimates_are_preserved": all(
            result["point_transform_agreement_rate"] == 1.0 for result in regimes.values()
        ),
        "clock_samples_retain_unresolved_draws": all(
            result["clock_sample_resolution_matches_landmark_rate"] == 1.0
            for result in regimes.values()
        ),
        "decisive_learning_resolves_more_point_landmarks": (
            decisive["point_resolution_rate"] > marginal["point_resolution_rate"]
        ),
        "decisive_learning_has_higher_bootstrap_resolution": (
            decisive["mean_bootstrap_resolution_rate"] > marginal["mean_bootstrap_resolution_rate"]
        ),
        "decisive_learning_is_usually_bootstrap_resolved": (
            decisive["mean_bootstrap_resolution_rate"] > 0.9
        ),
        "marginal_learning_remains_incompletely_resolved": (
            marginal["mean_bootstrap_resolution_rate"] < 0.9
        ),
    }
    contract_passed = all(contract.values())
    if check and not contract_passed:
        raise AssertionError(f"landmark-uncertainty contract failed: {contract}")
    return {
        "benchmark": "threshold-landmark uncertainty and resolution",
        "seed": seed,
        "repetitions": repetitions,
        "n_resamples_per_fit": n_resamples,
        "design": {
            "n_subjects": 1,
            "n_sessions": 4,
            "trials_per_session": 30,
            "probability_transition_trial": 50,
            "threshold": 0.8,
            "detection_window": 15,
            "consecutive_windows": 3,
            "bootstrap_smoothing_window": 7,
            "interval_level": 0.9,
        },
        "scope": {
            "estimator": "causally smoothed plug-in Bernoulli bootstrap",
            "interval": "equal-tailed among resolved draws; resolution reported separately",
            "dependence": "conditional independence given the smoothed training trajectory",
            "claim": "design-specific landmark resolution, not a posterior credible interval",
        },
        "regimes": regimes,
        "contract": contract,
        "contract_passed": contract_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=ROOT_SEED)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(
        repetitions=arguments.repetitions,
        n_resamples=arguments.n_resamples,
        seed=arguments.seed,
        check=not arguments.no_check,
    )
    rendered = render(result, allow_nan=True)
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
