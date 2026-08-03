"""Select a session-dynamic GLM-HMM and test it against matched competitors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from behavio import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    BinaryQLearning,
    ScoreMetric,
    SessionDynamicBernoulliGLMHMM,
    Study,
    cohort_forward_session_splits,
    compare_models,
    nested_select_model,
)
from behavio.compose import smooth
from benchmarks.provenance import render

REGIMES = ("stationary", "session_dynamic")
COMPETITORS = (
    "stationary_glm_hmm",
    "observed_transition_glm_hmm",
    "smooth_drift",
    "q_learning",
)
STATE_COUNTS = (2, 3)
EMISSION_STEP_SCALES = (0.15, 0.35)
TRANSITION_CONCENTRATIONS = (12.0, 30.0)
TRUE_EMISSION_STEP_SCALE = 0.35
TRUE_TRANSITION_CONCENTRATION = 30.0


def build_design(*, seed: int, n_sessions: int = 7, trials_per_session: int = 75) -> Study:
    """Build a one-subject learning design shared by both generating regimes."""

    generator = np.random.default_rng(seed)
    n_rows = n_sessions * trials_per_session
    return Study.factorial(
        trials=trials_per_session,
        subjects="animal-a",
        sessions=n_sessions,
        columns={
            "stimulus": generator.normal(size=n_rows),
            # Rewards are deliberately choice-independent. The learning candidate is a
            # falsification control here, not a disguised generator.
            "reward": generator.binomial(1, 0.5, size=n_rows),
        },
    )


def experiment(*, regime: str, seed: int) -> dict[str, Any]:
    """Run one prospective selection/comparison and one full-path recovery check."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    design = build_design(seed=seed)
    truth_model = _truth_model(regime, seed=seed + 1)
    simulation = truth_model.simulate_with_states(
        design,
        _truth_parameters(truth_model),
        seed=seed + 2,
    )
    study = simulation.study
    outer_split = cohort_forward_session_splits(study, min_train_sessions=6)[0]

    def inner_splitter(training: Study):
        return cohort_forward_session_splits(training, min_train_sessions=4)

    dynamic_candidates = _dynamic_candidates(seed=seed + 3)
    nested = nested_select_model(
        dynamic_candidates,
        study,
        (outer_split,),
        inner_splitter,
        metrics=(ScoreMetric.LOG_LOSS,),
        bootstrap_resamples=50,
        inner_bootstrap_resamples=50,
        bootstrap_seed=seed + 4,
    )
    comparison = compare_models(
        _competitors(seed=seed + 5),
        study,
        (outer_split,),
        metrics=(ScoreMetric.LOG_LOSS,),
        bootstrap_resamples=50,
        bootstrap_seed=seed + 6,
    )
    selected_name = nested.folds[0].selected_model
    selected_model = dynamic_candidates[selected_name]
    scores = {
        "selected_session_dynamic": nested.unit_balanced_log_loss,
        **{name: comparison.result_for(name).unit_balanced_log_loss for name in COMPETITORS},
    }
    winner = min(scores, key=scores.__getitem__)
    selected_fit = selected_model.fit(study)
    recovery: dict[str, Any] | None = None
    if regime == "session_dynamic" and selected_model.n_states == simulation.n_states:
        trajectory = selected_model.trajectory_recovery(simulation, selected_fit)
        recovery = {
            "decoded_state_accuracy": trajectory.alignment.decoded_accuracy,
            "emission_rmse": trajectory.emission_rmse,
            "transition_rmse": trajectory.transition_rmse,
            "alignment_ambiguous": trajectory.alignment.ambiguous,
        }
    return {
        "selected_candidate": selected_name,
        "selected_n_states": selected_model.n_states,
        "selected_emission_step_scale": selected_model.emission_step_scale,
        "selected_transition_concentration": selected_model.transition_concentration,
        "inner_fold_count": len(nested.folds[0].inner_report.splits),
        "outer_train_session_orders": list(outer_split.train_session_orders["animal-a"]),
        "outer_test_session_orders": list(outer_split.test_session_orders["animal-a"]),
        "prospective_log_loss": scores,
        "prospective_winner": winner,
        "selected_fit_audit_status": nested.audit_status.value,
        "selected_partial_stage_converged": selected_fit.partial_converged,
        "selected_full_stage_converged": selected_fit.full_converged,
        "selected_path_recovery": recovery,
    }


def run(*, repetitions: int = 4, seed: int = 27_041) -> dict[str, Any]:
    """Aggregate paired stationary and dynamic generating repetitions."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        runs = [
            experiment(regime=regime, seed=seed + repetition * 100 + regime_index * 10)
            for repetition in range(repetitions)
        ]
        methods = ("selected_session_dynamic", *COMPETITORS)
        mean_scores = {
            method: float(np.mean([run["prospective_log_loss"][method] for run in runs]))
            for method in methods
        }
        recoveries = [
            run["selected_path_recovery"]
            for run in runs
            if run["selected_path_recovery"] is not None
        ]
        regimes[regime] = {
            "mean_prospective_log_loss": mean_scores,
            "mean_score_winner": min(mean_scores, key=mean_scores.__getitem__),
            "prospective_win_counts": {
                method: sum(run["prospective_winner"] == method for run in runs)
                for method in methods
            },
            "selection_counts": {
                name: sum(run["selected_candidate"] == name for run in runs)
                for name in _dynamic_candidates(seed=0)
            },
            "n_two_state_selections": sum(run["selected_n_states"] == 2 for run in runs),
            "all_selected_full_stages_converged": all(
                run["selected_full_stage_converged"] for run in runs
            ),
            "all_selected_partial_stages_converged": all(
                run["selected_partial_stage_converged"] for run in runs
            ),
            **(
                {
                    "n_aligned_path_recoveries": len(recoveries),
                    "mean_decoded_state_accuracy": float(
                        np.mean([recovery["decoded_state_accuracy"] for recovery in recoveries])
                    ),
                    "mean_emission_rmse": float(
                        np.mean([recovery["emission_rmse"] for recovery in recoveries])
                    ),
                    "mean_transition_rmse": float(
                        np.mean([recovery["transition_rmse"] for recovery in recoveries])
                    ),
                }
                if recoveries
                else {}
            ),
            "runs": runs,
        }
    contract_passed = (
        regimes["stationary"]["mean_score_winner"] == "stationary_glm_hmm"
        and regimes["session_dynamic"]["mean_score_winner"] == "selected_session_dynamic"
        and all(
            regimes[regime][stage]
            for regime in REGIMES
            for stage in (
                "all_selected_partial_stages_converged",
                "all_selected_full_stages_converged",
            )
        )
        and regimes["session_dynamic"].get("mean_decoded_state_accuracy", 0.0) >= 0.7
        and regimes["session_dynamic"].get("n_aligned_path_recoveries", 0) >= repetitions / 2
    )
    return {
        "benchmark": "training-only session-dynamic GLM-HMM selection and recovery",
        "seed": seed,
        "repetitions": repetitions,
        "design": {
            "subjects": 1,
            "sessions": 7,
            "trials_per_session": 75,
            "outer_training_sessions": 6,
            "outer_test_sessions": 1,
            "inner_minimum_training_sessions": 4,
        },
        "truth": {
            "states": 2,
            "dynamic_emission_step_scale": TRUE_EMISSION_STEP_SCALE,
            "dynamic_transition_concentration": TRUE_TRANSITION_CONCENTRATION,
        },
        "selection_contract": {
            "candidate_dimensions": [
                "n_states",
                "emission_step_scale",
                "transition_concentration",
            ],
            "state_counts": list(STATE_COUNTS),
            "emission_step_scales": list(EMISSION_STEP_SCALES),
            "transition_concentrations": list(TRANSITION_CONCENTRATIONS),
            "dynamic_em_tolerance": 1e-5,
            "dynamic_em_max_iterations_per_stage": 35,
            "primary_metric": "mean subject-level inner-fold log loss",
            "outer_test_outcomes_available_during_selection": False,
            "tie_break": "declared candidate order",
        },
        "competitors": list(COMPETITORS),
        "regimes": regimes,
        "contract_passed": contract_passed,
    }


def _base_model_arguments(seed: int) -> dict[str, Any]:
    return {
        "predictors": ("stimulus",),
        "choice_lags": 1,
        "l2": 0.02,
        "n_restarts": 1,
        "random_seed": seed,
        "max_iterations": 250,
        "tolerance": 1e-8,
    }


def _truth_model(regime: str, *, seed: int) -> BernoulliGLMHMM | SessionDynamicBernoulliGLMHMM:
    arguments = _base_model_arguments(seed)
    if regime == "stationary":
        return BernoulliGLMHMM(n_states=2, **arguments)
    return SessionDynamicBernoulliGLMHMM(
        n_states=2,
        emission_step_scale=TRUE_EMISSION_STEP_SCALE,
        transition_concentration=TRUE_TRANSITION_CONCENTRATION,
        dynamic_max_iterations=35,
        dynamic_tolerance=1e-5,
        **arguments,
    )


def _truth_parameters(
    model: BernoulliGLMHMM | SessionDynamicBernoulliGLMHMM,
) -> dict[str, float]:
    return dict(
        model.parameters_from_components(
            initial_probabilities=(0.5, 0.5),
            transition_matrix=((0.96, 0.04), (0.04, 0.96)),
            emissions={
                "intercept": (-2.4, 2.4),
                "stimulus": (0.4, 1.6),
                "choice_lag_1": (0.35, -0.15),
            },
        )
    )


def _dynamic_candidates(*, seed: int) -> dict[str, SessionDynamicBernoulliGLMHMM]:
    candidates: dict[str, SessionDynamicBernoulliGLMHMM] = {}
    for n_states in STATE_COUNTS:
        for sigma in EMISSION_STEP_SCALES:
            for alpha in TRANSITION_CONCENTRATIONS:
                name = f"K={n_states};sigma={sigma:g};alpha={alpha:g}"
                candidates[name] = SessionDynamicBernoulliGLMHMM(
                    n_states=n_states,
                    emission_step_scale=sigma,
                    transition_concentration=alpha,
                    dynamic_max_iterations=35,
                    dynamic_tolerance=1e-5,
                    **_base_model_arguments(seed),
                )
    return candidates


def _competitors(*, seed: int) -> dict[str, Any]:
    return {
        "stationary_glm_hmm": BernoulliGLMHMM(
            n_states=2,
            **_base_model_arguments(seed),
        ),
        "observed_transition_glm_hmm": BernoulliGLMHMM(
            n_states=2,
            transition_predictors=("session_order",),
            transition_l2=0.1,
            **_base_model_arguments(seed),
        ),
        "smooth_drift": smooth(
            BernoulliHistoryGLM(
                predictors=("stimulus",),
                choice_lags=1,
                l2=0.02,
                max_iterations=250,
                tolerance=1e-8,
            ),
            over="session_order",
            knots=(0.0, 3.0, 6.0),
            smoothness=3.0,
        ),
        "q_learning": BinaryQLearning(
            n_restarts=2,
            random_seed=seed,
            max_iterations=250,
            tolerance=1e-8,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=27_041)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(repetitions=arguments.repetitions, seed=arguments.seed)
    arguments.output.write_text(render(result), encoding="utf-8")


if __name__ == "__main__":
    main()
