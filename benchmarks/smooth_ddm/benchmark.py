"""Compare static and smooth Wiener fits under stationary and changing truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from unspool import SmoothWienerDriftDiffusion, Study, WienerDriftDiffusion

N_SESSIONS = 6
TRAINING_SESSIONS = 5
TRIALS_PER_SESSION = 150
REGIMES = ("stationary", "changing")
METHODS = ("static", "smooth")
EXPECTED_WINNER = {"stationary": "static", "changing": "smooth"}


def build_design(*, seed: int) -> Study:
    """Construct six ordered sessions on a declared session clock."""

    generator = np.random.default_rng(seed)
    n_trials = N_SESSIONS * TRIALS_PER_SESSION
    return Study(
        {
            "subject": ["synthetic-subject"] * n_trials,
            "session": [
                f"session-{session}"
                for session in range(N_SESSIONS)
                for _ in range(TRIALS_PER_SESSION)
            ],
            "trial": list(range(TRIALS_PER_SESSION)) * N_SESSIONS,
            "session_order": [
                session for session in range(N_SESSIONS) for _ in range(TRIALS_PER_SESSION)
            ],
            "stimulus": generator.normal(size=n_trials),
        }
    )


def experiment(*, regime: str, seed: int) -> dict[str, Any]:
    """Run one matched recovery and final-session forecasting comparison."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    smooth = _smooth_model()
    static = _static_model()
    truth_paths = _truth_paths(regime)
    truth = smooth.parameters_from_paths(truth_paths)
    study = smooth.simulate(build_design(seed=seed), truth, seed=seed + 1)
    train = study.take(np.flatnonzero(study["session_order"] < TRAINING_SESSIONS))
    test = study.take(np.flatnonzero(study["session_order"] == TRAINING_SESSIONS))
    smooth_fit = smooth.fit(train)
    static_fit = static.fit(train)

    truth_matrix = np.column_stack(
        [
            np.asarray(truth_paths["drift.stimulus"][:TRAINING_SESSIONS]),
            np.asarray(truth_paths["boundary"][:TRAINING_SESSIONS]),
        ]
    )
    smooth_trajectory = smooth.parameter_trajectory(
        smooth_fit,
        times=range(TRAINING_SESSIONS),
    )
    smooth_matrix = np.column_stack(
        [
            smooth_trajectory.path("drift.stimulus"),
            smooth_trajectory.path("boundary"),
        ]
    )
    static_components = static.parameter_components(static_fit)
    static_matrix = np.broadcast_to(
        np.asarray(
            [
                static_components.drift_map["drift.stimulus"],
                static_components.boundary,
            ]
        ),
        truth_matrix.shape,
    )
    return {
        "seed": seed,
        "methods": {
            "static": _method_record(
                fit=static_fit,
                trajectory_rmse=_rmse(static_matrix, truth_matrix),
                future_log_loss=-float(np.mean(static.pointwise_log_prob(test, static_fit))),
            ),
            "smooth": _method_record(
                fit=smooth_fit,
                trajectory_rmse=_rmse(smooth_matrix, truth_matrix),
                future_log_loss=-float(np.mean(smooth.pointwise_log_prob(test, smooth_fit))),
            ),
        },
    }


def run(*, repetitions: int = 20, seed: int = 82_411) -> dict[str, Any]:
    """Aggregate paired regimes while retaining every audit and seed."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        runs = [
            experiment(
                regime=regime,
                seed=seed + regime_index * 100_000 + repeat * 10,
            )
            for repeat in range(repetitions)
        ]
        eligible = [
            row
            for row in runs
            if all(row["methods"][method]["audit"]["status"] != "fail" for method in METHODS)
        ]
        if not eligible:
            raise RuntimeError(f"no paired fits were eligible in {regime!r}")
        summaries = {
            method: {
                "mean_training_trajectory_rmse": float(
                    np.mean(
                        [row["methods"][method]["training_trajectory_rmse"] for row in eligible]
                    )
                ),
                "mean_future_log_loss": float(
                    np.mean([row["methods"][method]["future_log_loss"] for row in eligible])
                ),
                "all_fits_converged": all(row["methods"][method]["converged"] for row in runs),
            }
            for method in METHODS
        }
        regimes[regime] = {
            "expected_winner": EXPECTED_WINNER[regime],
            "n_paired_eligible": len(eligible),
            "trajectory_rmse_winner": min(
                summaries,
                key=lambda method: summaries[method]["mean_training_trajectory_rmse"],
            ),
            "future_log_loss_winner": min(
                summaries,
                key=lambda method: summaries[method]["mean_future_log_loss"],
            ),
            "smooth_future_wins": int(
                np.sum(
                    [
                        row["methods"]["smooth"]["future_log_loss"]
                        < row["methods"]["static"]["future_log_loss"]
                        for row in eligible
                    ]
                )
            ),
            "methods": summaries,
            "runs": runs,
        }
    return {
        "benchmark": "session-varying Wiener trajectory recovery and forecast",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "n_sessions": N_SESSIONS,
            "training_sessions": TRAINING_SESSIONS,
            "test_sessions": 1,
            "trials_per_session": TRIALS_PER_SESSION,
            "clock": "session_order",
            "knots": list(range(N_SESSIONS)),
        },
        "scope": {
            "varying_parameters": ["drift.stimulus", "boundary"],
            "stationary_parameters": [
                "drift.intercept",
                "starting_bias",
                "nondecision_time",
            ],
            "smoothness": _smooth_model().smoothness,
            "forecast": "sixth session from the first five; future knot fixed before fitting",
            "trajectory_metric": "RMSE for stimulus drift and boundary at training knots",
        },
        "truth": {regime: _truth_paths(regime) for regime in REGIMES},
        "regimes": regimes,
        "all_expected_winners_recovered": all(
            regimes[regime][metric] == EXPECTED_WINNER[regime]
            for regime in REGIMES
            for metric in ("trajectory_rmse_winner", "future_log_loss_winner")
        ),
    }


def _smooth_model() -> SmoothWienerDriftDiffusion:
    return SmoothWienerDriftDiffusion(
        covariates=("stimulus",),
        knots=tuple(float(session) for session in range(N_SESSIONS)),
        varying_parameters=("drift.stimulus", "boundary"),
        smoothness=10.0,
        n_restarts=3,
        max_iterations=400,
        simulation_time_step=0.0005,
    )


def _static_model() -> WienerDriftDiffusion:
    return WienerDriftDiffusion(
        covariates=("stimulus",),
        n_restarts=3,
        max_iterations=400,
        simulation_time_step=0.0005,
    )


def _truth_paths(regime: str) -> dict[str, float | list[float]]:
    if regime == "stationary":
        stimulus = [1.0] * N_SESSIONS
        boundary = [1.2] * N_SESSIONS
    elif regime == "changing":
        stimulus = [0.3, 0.55, 0.85, 1.2, 1.55, 1.9]
        boundary = [1.55, 1.45, 1.32, 1.18, 1.05, 0.95]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return {
        "drift.intercept": 0.1,
        "drift.stimulus": stimulus,
        "boundary": boundary,
        "starting_bias": 0.48,
        "nondecision_time": 0.25,
    }


def _method_record(
    *,
    fit: Any,
    trajectory_rmse: float,
    future_log_loss: float,
) -> dict[str, Any]:
    return {
        "training_trajectory_rmse": trajectory_rmse,
        "future_log_loss": future_log_loss,
        "converged": fit.diagnostics.converged,
        "objective": float(fit.diagnostics.objective),
        "audit": fit.audit().to_dict(),
    }


def _rmse(
    estimate: np.ndarray[Any, np.dtype[np.float64]], truth: np.ndarray[Any, np.dtype[np.float64]]
) -> float:
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=82_411)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(repetitions=arguments.repetitions, seed=arguments.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
