"""Benchmark contaminant-aware and naive Wiener fits on matched future sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmarks.provenance import render
from unspool import Study, UniformResponseTimeContaminant, WienerDriftDiffusion

N_SESSIONS = 5
TRIALS_PER_SESSION = 200
SHARED_PARAMETERS = (
    "drift.intercept",
    "drift.stimulus",
    "boundary",
    "starting_bias",
    "nondecision_time",
)


def build_design(*, seed: int) -> Study:
    """Build five ordered sessions with continuous evidence."""

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


def run(*, repeats: int = 20, seed: int = 73_901) -> dict[str, Any]:
    """Run paired recovery and final-session forecasts under uniform contamination."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    contaminant = UniformResponseTimeContaminant(
        time_bounds=(0.05, 3.0),
        probability_bounds=(0.0, 0.2),
    )
    robust = WienerDriftDiffusion(
        covariates=("stimulus",),
        contaminant=contaminant,
        nondecision_time_bounds=(0.1, 0.6),
        n_restarts=3,
        max_iterations=400,
        simulation_time_step=0.0001,
    )
    naive = WienerDriftDiffusion(
        covariates=("stimulus",),
        n_restarts=3,
        max_iterations=400,
        simulation_time_step=0.0001,
    )
    truth = robust.parameters_from_components(
        drift={"drift.intercept": 0.2, "drift.stimulus": 1.2},
        boundary=1.2,
        starting_bias=0.45,
        nondecision_time=0.25,
        contaminant_probability=0.05,
    )
    sequences = np.random.SeedSequence(seed).spawn(repeats)
    runs: list[dict[str, Any]] = []
    for sequence in sequences:
        design_seed, simulation_seed = sequence.generate_state(2, dtype=np.uint64)
        simulation = robust.simulate_with_contaminants(
            build_design(seed=int(design_seed)),
            truth,
            seed=int(simulation_seed),
        )
        train_indices = np.flatnonzero(simulation.study["session_order"] < N_SESSIONS - 1)
        test_indices = np.flatnonzero(simulation.study["session_order"] == N_SESSIONS - 1)
        train = simulation.study.take(train_indices)
        test = simulation.study.take(test_indices)
        robust_fit = robust.fit(train)
        naive_fit = naive.fit(train)
        robust_posterior = robust_fit.posterior_contaminant_probability
        train_truth = simulation.contaminants[train_indices]
        runs.append(
            {
                "design_seed": int(design_seed),
                "simulation_seed": int(simulation_seed),
                "n_train_contaminants": int(np.sum(train_truth)),
                "n_test_contaminants": int(np.sum(simulation.contaminants[test_indices])),
                "robust": _fit_record(robust_fit),
                "naive": _fit_record(naive_fit),
                "future_mean_log_loss": {
                    "robust": float(-np.mean(robust.pointwise_log_prob(test, robust_fit))),
                    "naive": float(-np.mean(naive.pointwise_log_prob(test, naive_fit))),
                },
                "responsibility": {
                    "true_contaminants": _optional_mean(robust_posterior[train_truth]),
                    "ordinary_trials": _optional_mean(robust_posterior[~train_truth]),
                    "expected_count": robust_fit.expected_contaminant_count,
                },
            }
        )
    eligible = [
        row
        for row in runs
        if row["robust"]["audit"]["status"] != "fail" and row["naive"]["audit"]["status"] != "fail"
    ]
    if not eligible:
        raise RuntimeError("no paired fits were eligible for benchmark summaries")
    parameter_rmse: dict[str, dict[str, float | bool]] = {}
    for parameter in SHARED_PARAMETERS:
        robust_rmse = _rmse(
            [row["robust"]["estimates"][parameter] for row in eligible],
            truth[parameter],
        )
        naive_rmse = _rmse(
            [row["naive"]["estimates"][parameter] for row in eligible],
            truth[parameter],
        )
        parameter_rmse[parameter] = {
            "robust": robust_rmse,
            "naive": naive_rmse,
            "robust_better": robust_rmse < naive_rmse,
        }
    robust_losses = np.asarray([row["future_mean_log_loss"]["robust"] for row in eligible])
    naive_losses = np.asarray([row["future_mean_log_loss"]["naive"] for row in eligible])
    contaminant_responsibility = np.asarray(
        [row["responsibility"]["true_contaminants"] for row in eligible],
        dtype=np.float64,
    )
    ordinary_responsibility = np.asarray(
        [row["responsibility"]["ordinary_trials"] for row in eligible],
        dtype=np.float64,
    )
    return {
        "benchmark": "uniform-contaminant Wiener recovery and future-session forecast",
        "seed": seed,
        "repeats": repeats,
        "design": {
            "n_sessions": N_SESSIONS,
            "trials_per_session": TRIALS_PER_SESSION,
            "n_train_trials": (N_SESSIONS - 1) * TRIALS_PER_SESSION,
            "n_test_trials": TRIALS_PER_SESSION,
        },
        "generator_signature": robust.signature,
        "naive_signature": naive.signature,
        "scored_columns": list(robust.scored_columns),
        "truth": dict(truth),
        "runs": runs,
        "summary": {
            "n_paired_eligible": len(eligible),
            "parameter_rmse": parameter_rmse,
            "contaminant_probability_rmse": _rmse(
                [row["robust"]["estimates"]["contaminant_probability"] for row in eligible],
                truth["contaminant_probability"],
            ),
            "future_mean_log_loss": {
                "robust": float(np.mean(robust_losses)),
                "naive": float(np.mean(naive_losses)),
                "naive_minus_robust": float(np.mean(naive_losses - robust_losses)),
                "robust_better_runs": int(np.sum(robust_losses < naive_losses)),
            },
            "mean_responsibility": {
                "true_contaminants": float(np.nanmean(contaminant_responsibility)),
                "ordinary_trials": float(np.nanmean(ordinary_responsibility)),
            },
        },
    }


def _fit_record(fit: Any) -> dict[str, Any]:
    return {
        "estimates": {name: float(value) for name, value in fit.parameters.items()},
        "standard_errors": {
            name: _finite_or_none(value) for name, value in fit.standard_error_map.items()
        },
        "objective": float(fit.diagnostics.objective),
        "selected_restart": fit.selected_restart,
        "restart_objectives": [float(value) for value in fit.restart_objectives],
        "audit": fit.audit().to_dict(),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _optional_mean(values: NDArray[np.float64]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _rmse(values: list[float], truth: float) -> float:
    errors = np.asarray(values, dtype=np.float64) - truth
    return float(np.sqrt(np.mean(errors**2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73_901)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(repeats=arguments.repeats, seed=arguments.seed)
    rendered = render(result)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
