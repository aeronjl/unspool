"""Compare Wiener population structures across three longitudinal regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from behavio import (
    HierarchicalSmoothWienerDriftDiffusion,
    SmoothWienerDriftDiffusion,
    Study,
    WienerDriftDiffusion,
)
from behavio.models._kernels.basis import linear_time_basis
from benchmarks.provenance import render

N_SUBJECTS = 5
N_SESSIONS = 5
TRAINING_SESSIONS = 4
TRIALS_PER_SESSION = 70
KNOTS = (0.0, 2.0, 4.0)
REGIMES = ("stationary_identical", "shared_change", "individual_change")
METHODS = ("complete_pooling", "shared_smooth", "independent_smooth", "hierarchical_smooth")
EXPECTED_WINNER = {
    "stationary_identical": "complete_pooling",
    "shared_change": "shared_smooth",
    "individual_change": "hierarchical_smooth",
}
PATH_PARAMETERS = ("drift.stimulus", "boundary")


def build_design(*, seed: int) -> Study:
    """Construct a balanced five-animal longitudinal choice/RT design."""

    generator = np.random.default_rng(seed)
    subjects = tuple(f"mouse-{index}" for index in range(N_SUBJECTS))
    n_trials = N_SUBJECTS * N_SESSIONS * TRIALS_PER_SESSION
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
            "trial": list(range(TRIALS_PER_SESSION)) * N_SUBJECTS * N_SESSIONS,
            "session_order": [
                session
                for _subject in subjects
                for session in range(N_SESSIONS)
                for _trial in range(TRIALS_PER_SESSION)
            ],
            "stimulus": generator.normal(size=n_trials),
        }
    )


def experiment(*, regime: str, seed: int) -> dict[str, Any]:
    """Run one matched structural recovery and final-session forecast."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    design = build_design(seed=seed)
    hierarchical = _hierarchical_model()
    population_paths, deviations = _truth(regime, design.subjects, seed=seed + 1)
    simulation = hierarchical.simulate_with_effects(
        design,
        hierarchical.parameters_from_paths(population_paths),
        seed=seed + 2,
        subject_deviation_paths=deviations,
    )
    train = simulation.study.take(
        np.flatnonzero(simulation.study["session_order"] < TRAINING_SESSIONS)
    )
    test = simulation.study.take(
        np.flatnonzero(simulation.study["session_order"] == TRAINING_SESSIONS)
    )
    evaluation_times = np.arange(TRAINING_SESSIONS, dtype=np.float64)
    truth_paths = _evaluated_truth(simulation.subject_knot_values, evaluation_times)

    complete = _complete_model()
    complete_fit = complete.fit(train)
    complete_components = complete.parameter_components(complete_fit)
    complete_path = np.asarray(
        [complete_components.drift_map["drift.stimulus"], complete_components.boundary]
    )
    complete_paths = np.broadcast_to(
        complete_path,
        (N_SUBJECTS, TRAINING_SESSIONS, len(PATH_PARAMETERS)),
    )

    shared = _shared_model()
    shared_fit = shared.fit(train)
    shared_paths = np.broadcast_to(
        _selected_trajectory(shared.parameter_trajectory(shared_fit, times=evaluation_times)),
        truth_paths.shape,
    )

    independent_paths: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    independent_scores: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    independent_audits: list[dict[str, Any]] = []
    independent_converged = True
    for subject in simulation.subjects:
        subject_train = _subject_study(train, subject)
        subject_test = _subject_study(test, subject)
        independent = _independent_model()
        independent_fit = independent.fit(subject_train)
        independent_paths.append(
            _selected_trajectory(
                independent.parameter_trajectory(independent_fit, times=evaluation_times)
            )
        )
        independent_scores.append(independent.pointwise_log_prob(subject_test, independent_fit))
        independent_audits.append(independent_fit.audit().to_dict())
        independent_converged &= independent_fit.diagnostics.converged

    hierarchical_fit = hierarchical.fit(train)
    hierarchical_paths = np.stack(
        [
            _selected_trajectory(
                hierarchical.subject_trajectory(
                    hierarchical_fit,
                    subject,
                    times=evaluation_times,
                )
            )
            for subject in simulation.subjects
        ]
    )
    records = {
        "complete_pooling": _method_record(
            trajectory_rmse=_rmse(complete_paths, truth_paths),
            future_log_loss=-float(np.mean(complete.pointwise_log_prob(test, complete_fit))),
            converged=complete_fit.diagnostics.converged,
            audits=[complete_fit.audit().to_dict()],
        ),
        "shared_smooth": _method_record(
            trajectory_rmse=_rmse(shared_paths, truth_paths),
            future_log_loss=-float(np.mean(shared.pointwise_log_prob(test, shared_fit))),
            converged=shared_fit.diagnostics.converged,
            audits=[shared_fit.audit().to_dict()],
        ),
        "independent_smooth": _method_record(
            trajectory_rmse=_rmse(np.stack(independent_paths), truth_paths),
            future_log_loss=-float(np.mean(np.concatenate(independent_scores))),
            converged=independent_converged,
            audits=independent_audits,
        ),
        "hierarchical_smooth": _method_record(
            trajectory_rmse=_rmse(hierarchical_paths, truth_paths),
            future_log_loss=-float(
                np.mean(hierarchical.pointwise_log_prob(test, hierarchical_fit))
            ),
            converged=hierarchical_fit.diagnostics.converged,
            audits=[hierarchical_fit.audit().to_dict()],
        ),
    }
    return {"seed": seed, "methods": records}


def run(*, repetitions: int = 20, seed: int = 64_219) -> dict[str, Any]:
    """Aggregate repeated matched experiments with complete audit evidence."""

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
            if all(
                all(audit["status"] != "fail" for audit in row["methods"][method]["audits"])
                for method in METHODS
            )
        ]
        if not eligible:
            raise RuntimeError(f"no complete matched runs were audit-eligible in {regime!r}")
        summaries = {
            method: {
                "mean_subject_trajectory_rmse": float(
                    np.mean([row["methods"][method]["subject_trajectory_rmse"] for row in eligible])
                ),
                "mean_future_log_loss": float(
                    np.mean([row["methods"][method]["future_log_loss"] for row in eligible])
                ),
                "all_fits_converged": all(
                    row["methods"][method]["all_fits_converged"] for row in runs
                ),
            }
            for method in METHODS
        }
        regimes[regime] = {
            "expected_winner": EXPECTED_WINNER[regime],
            "n_complete_eligible": len(eligible),
            "trajectory_rmse_winner": min(
                summaries,
                key=lambda method: summaries[method]["mean_subject_trajectory_rmse"],
            ),
            "future_log_loss_winner": min(
                summaries,
                key=lambda method: summaries[method]["mean_future_log_loss"],
            ),
            "methods": summaries,
            "runs": runs,
        }
    return {
        "benchmark": "hierarchical session-varying Wiener structural recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "subjects": N_SUBJECTS,
            "sessions": N_SESSIONS,
            "training_sessions": TRAINING_SESSIONS,
            "test_sessions": 1,
            "trials_per_session": TRIALS_PER_SESSION,
            "knots": list(KNOTS),
            "clock": "session_order",
        },
        "scope": {
            "varying_parameters": list(PATH_PARAMETERS),
            "subject_scale": _hierarchical_model().subject_scale,
            "subject_smoothness": _hierarchical_model().subject_smoothness,
            "prediction": "fifth session for animals represented in training",
            "trajectory_metric": "subject RMSE for stimulus drift and boundary in training",
        },
        "regimes": regimes,
        "all_expected_winners_recovered": all(
            regimes[regime][metric] == EXPECTED_WINNER[regime]
            for regime in REGIMES
            for metric in ("trajectory_rmse_winner", "future_log_loss_winner")
        ),
    }


def _hierarchical_model() -> HierarchicalSmoothWienerDriftDiffusion:
    return HierarchicalSmoothWienerDriftDiffusion(
        covariates=("stimulus",),
        knots=KNOTS,
        varying_parameters=PATH_PARAMETERS,
        subject_parameters=PATH_PARAMETERS,
        smoothness=8.0,
        subject_scale=0.2,
        subject_smoothness=8.0,
        n_restarts=2,
        max_iterations=350,
        simulation_time_step=0.001,
    )


def _shared_model() -> SmoothWienerDriftDiffusion:
    return SmoothWienerDriftDiffusion(
        covariates=("stimulus",),
        knots=KNOTS,
        varying_parameters=PATH_PARAMETERS,
        smoothness=8.0,
        shared_trajectory=True,
        n_restarts=2,
        max_iterations=350,
        simulation_time_step=0.001,
    )


def _independent_model() -> SmoothWienerDriftDiffusion:
    return SmoothWienerDriftDiffusion(
        covariates=("stimulus",),
        knots=KNOTS,
        varying_parameters=PATH_PARAMETERS,
        smoothness=8.0,
        n_restarts=2,
        max_iterations=350,
        simulation_time_step=0.001,
    )


def _complete_model() -> WienerDriftDiffusion:
    return WienerDriftDiffusion(
        covariates=("stimulus",),
        n_restarts=2,
        max_iterations=350,
        simulation_time_step=0.001,
    )


def _truth(
    regime: str,
    subjects: tuple[Any, ...],
    *,
    seed: int,
) -> tuple[dict[str, float | list[float]], dict[Any, dict[str, list[float]]]]:
    if regime == "stationary_identical":
        stimulus = [1.0, 1.0, 1.0]
        boundary = [1.2, 1.2, 1.2]
    elif regime in {"shared_change", "individual_change"}:
        stimulus = [0.45, 1.0, 1.55]
        boundary = [1.5, 1.25, 1.0]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    population: dict[str, float | list[float]] = {
        "drift.intercept": 0.1,
        "drift.stimulus": stimulus,
        "boundary": boundary,
        "starting_bias": 0.48,
        "nondecision_time": 0.25,
    }
    deviations = {
        subject: {parameter: [0.0] * len(KNOTS) for parameter in PATH_PARAMETERS}
        for subject in subjects
    }
    if regime == "individual_change":
        generator = np.random.default_rng(seed)
        centered = np.asarray([-1.0, 0.0, 1.0])
        for subject in subjects:
            stimulus_offset = generator.normal(0.0, 0.08)
            stimulus_slope = generator.normal(0.0, 0.22)
            boundary_offset = generator.normal(0.0, 0.06)
            boundary_slope = generator.normal(0.0, 0.12)
            deviations[subject]["drift.stimulus"] = (
                stimulus_offset + stimulus_slope * centered
            ).tolist()
            deviations[subject]["boundary"] = (boundary_offset + boundary_slope * centered).tolist()
    return population, deviations


def _evaluated_truth(
    subject_knot_values: np.ndarray[Any, np.dtype[np.float64]],
    times: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    basis = linear_time_basis(times, KNOTS)
    selected = subject_knot_values[:, [1, 2], :]
    return np.einsum("tk,spk->stp", basis, selected)


def _selected_trajectory(trajectory: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.column_stack([trajectory.path(parameter) for parameter in PATH_PARAMETERS])


def _method_record(
    *,
    trajectory_rmse: float,
    future_log_loss: float,
    converged: bool,
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "subject_trajectory_rmse": trajectory_rmse,
        "future_log_loss": future_log_loss,
        "all_fits_converged": converged,
        "audits": audits,
    }


def _subject_study(study: Study, subject: Any) -> Study:
    return study.take(np.flatnonzero(study["subject"] == subject))


def _rmse(
    estimate: np.ndarray[Any, np.dtype[np.float64]],
    truth: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=64_219)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(repetitions=arguments.repetitions, seed=arguments.seed)
    rendered = render(result)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
