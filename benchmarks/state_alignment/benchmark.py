"""Benchmark permutation-invariant state recovery under clear and overlapping emissions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from behavio import BernoulliGLMHMM, Study
from behavio.models import align_latent_states
from benchmarks.provenance import render

DEFAULT_REPETITIONS = 20
ROOT_SEED = 9107
REGIMES = {
    "clear": (-2.0, 2.0),
    "overlapping": (-0.25, 0.25),
}


def build_design(*, seed: int, n_sessions: int = 4, trials_per_session: int = 75) -> Study:
    """Return the fixed observed design shared by both recovery regimes."""

    generator = np.random.default_rng(seed)
    n_trials = n_sessions * trials_per_session
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
            "stimulus": generator.normal(size=n_trials),
        }
    )


def experiment(*, regime: str, seed: int) -> dict[str, Any]:
    """Simulate, fit, and align one realization from a named regime."""

    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {tuple(REGIMES)}")
    model = BernoulliGLMHMM(
        predictors=(),
        choice_lags=0,
        n_states=2,
        n_restarts=3,
        random_seed=seed,
        l2=0.01,
        max_iterations=400,
    )
    parameters = model.parameters_from_components(
        initial_probabilities=[0.5, 0.5],
        transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
        emissions={"intercept": REGIMES[regime]},
    )
    simulation = model.simulate_with_states(build_design(seed=seed), parameters, seed=seed + 1)
    fit = model.fit(simulation.study)
    probabilities = model.state_probabilities(simulation.study, fit).filtered
    alignment = align_latent_states(simulation.states, probabilities)
    reversed_alignment = align_latent_states(simulation.states, probabilities[:, ::-1])
    return {
        "fit_converged": fit.diagnostics.converged,
        "fit_label_ambiguous": fit.label_ambiguous,
        "alignment_ambiguous": alignment.ambiguous,
        "reference_to_inferred": list(alignment.reference_to_inferred),
        "best_score": alignment.best_score,
        "runner_up_score": alignment.runner_up_score,
        "score_gap": alignment.score_gap,
        "posterior_accuracy": alignment.posterior_accuracy,
        "decoded_accuracy": alignment.decoded_accuracy,
        "raw_decoded_accuracy": float(
            np.mean(np.argmax(probabilities, axis=1) == simulation.states)
        ),
        "reversed_raw_decoded_accuracy": float(
            np.mean(np.argmax(probabilities[:, ::-1], axis=1) == simulation.states)
        ),
        "permutation_invariant": bool(
            np.allclose(alignment.aligned_probabilities, reversed_alignment.aligned_probabilities)
            and np.isclose(alignment.best_score, reversed_alignment.best_score)
            and np.isclose(alignment.decoded_accuracy, reversed_alignment.decoded_accuracy)
        ),
    }


def run(
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = ROOT_SEED,
    check: bool = True,
) -> dict[str, Any]:
    """Aggregate state-alignment recovery across the two emission regimes."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        runs = [
            experiment(
                regime=regime,
                seed=seed + regime_index * 100_000 + repetition * 100,
            )
            for repetition in range(repetitions)
        ]
        regimes[regime] = {
            "emission_intercepts": list(REGIMES[regime]),
            "convergence_rate": float(np.mean([item["fit_converged"] for item in runs])),
            "fit_label_ambiguity_rate": float(
                np.mean([item["fit_label_ambiguous"] for item in runs])
            ),
            "alignment_ambiguity_rate": float(
                np.mean([item["alignment_ambiguous"] for item in runs])
            ),
            "mean_best_score": float(np.mean([item["best_score"] for item in runs])),
            "mean_runner_up_score": float(np.mean([item["runner_up_score"] for item in runs])),
            "mean_score_gap": float(np.mean([item["score_gap"] for item in runs])),
            "mean_posterior_accuracy": float(
                np.mean([item["posterior_accuracy"] for item in runs])
            ),
            "mean_decoded_accuracy": float(np.mean([item["decoded_accuracy"] for item in runs])),
            "mean_raw_decoded_accuracy": float(
                np.mean([item["raw_decoded_accuracy"] for item in runs])
            ),
            "mean_reversed_raw_decoded_accuracy": float(
                np.mean([item["reversed_raw_decoded_accuracy"] for item in runs])
            ),
            "permutation_invariance_rate": float(
                np.mean([item["permutation_invariant"] for item in runs])
            ),
        }

    clear = regimes["clear"]
    overlapping = regimes["overlapping"]
    contract = {
        "all_fits_converged": all(result["convergence_rate"] == 1.0 for result in regimes.values()),
        "all_aligned_metrics_permutation_invariant": all(
            result["permutation_invariance_rate"] == 1.0 for result in regimes.values()
        ),
        "clear_states_have_higher_decoded_accuracy": (
            clear["mean_decoded_accuracy"] > overlapping["mean_decoded_accuracy"]
        ),
        "clear_states_have_higher_posterior_accuracy": (
            clear["mean_posterior_accuracy"] > overlapping["mean_posterior_accuracy"]
        ),
        "clear_states_have_larger_assignment_gap": (
            clear["mean_score_gap"] > overlapping["mean_score_gap"]
        ),
        "overlap_does_not_reduce_alignment_ambiguity": (
            overlapping["alignment_ambiguity_rate"] >= clear["alignment_ambiguity_rate"]
        ),
    }
    contract_passed = all(contract.values())
    if check and not contract_passed:
        raise AssertionError(f"state-alignment recovery contract failed: {contract}")
    return {
        "benchmark": "permutation-invariant latent-state recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "n_subjects": 1,
            "n_sessions": 4,
            "trials_per_session": 75,
            "transition_matrix": [[0.95, 0.05], [0.05, 0.95]],
            "ambiguity_tolerance": 0.05,
        },
        "scope": {
            "posterior": (
                "outcome-filtered recursion conditional on full-study fitted parameters; "
                "each state update excludes future outcomes"
            ),
            "assignment": "balanced posterior mass over reference states",
            "truth_use": "alignment and recovery diagnostics only; never fitting",
        },
        "regimes": regimes,
        "contract": contract,
        "contract_passed": contract_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=ROOT_SEED)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(
        repetitions=arguments.repetitions,
        seed=arguments.seed,
        check=not arguments.no_check,
    )
    rendered = render(result, allow_nan=True)
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
