"""Run a matched four-family prospective model-recovery grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from unspool import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    BinaryQLearning,
    ModelRecoveryScenario,
    SmoothBernoulliHistoryGLM,
    Study,
    run_model_recovery_grid,
)

N_SESSIONS = 5
MAX_TRIALS_PER_SESSION = 60
EXPECTED_SELECTED = {
    "sparse": ("static", "static", "smooth", "q-learning"),
    "dense": ("static", "smooth", "hmm", "q-learning"),
}


def build_design(*, trials_per_session: int) -> Study:
    """Build nested synthetic designs with stimulus, reward environment, and nuisance reward."""

    if not 1 <= trials_per_session <= MAX_TRIALS_PER_SESSION:
        raise ValueError(f"trials_per_session must lie between 1 and {MAX_TRIALS_PER_SESSION}")
    generator = np.random.default_rng(2040)
    stimulus = generator.normal(size=(N_SESSIONS, MAX_TRIALS_PER_SESSION))
    nuisance_reward = generator.binomial(
        1,
        0.5,
        size=(N_SESSIONS, MAX_TRIALS_PER_SESSION),
    )
    probability_one = np.asarray(
        [
            [0.8 if ((trial // 15) + session) % 2 else 0.2 for trial in range(trials_per_session)]
            for session in range(N_SESSIONS)
        ],
        dtype=np.float64,
    )
    n_trials = N_SESSIONS * trials_per_session
    return Study(
        {
            "subject": ["synthetic-mouse"] * n_trials,
            "session": [
                f"session-{session}"
                for session in range(N_SESSIONS)
                for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * N_SESSIONS,
            "session_order": [
                session for session in range(N_SESSIONS) for _ in range(trials_per_session)
            ],
            "stimulus": stimulus[:, :trials_per_session].reshape(-1),
            "reward_probability_0": (1.0 - probability_one).reshape(-1),
            "reward_probability_1": probability_one.reshape(-1),
            "reward": nuisance_reward[:, :trials_per_session].reshape(-1),
        }
    )


def experiment() -> tuple[
    tuple[ModelRecoveryScenario, ...],
    dict[str, BernoulliHistoryGLM | SmoothBernoulliHistoryGLM | BernoulliGLMHMM | BinaryQLearning],
]:
    """Return fixed candidate configurations and one parameter regime per family."""

    static = BernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.01,
    )
    smooth = SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        knots=tuple(range(N_SESSIONS)),
        smoothness=5.0,
        l2=0.01,
    )
    hidden_state = BernoulliGLMHMM(
        covariates=("stimulus",),
        choice_lags=1,
        n_states=2,
        n_restarts=1,
        random_seed=3,
        l2=0.01,
    )
    q_learning = BinaryQLearning(n_restarts=2, random_seed=4)
    candidates = {
        "static": static,
        "smooth": smooth,
        "hmm": hidden_state,
        "q-learning": q_learning,
    }
    scenarios = (
        ModelRecoveryScenario(
            name="stationary",
            truth_label="static",
            generator=static,
            parameters={"intercept": -0.1, "stimulus": 1.2, "choice_lag_1": 0.3},
        ),
        ModelRecoveryScenario(
            name="drifting",
            truth_label="smooth",
            generator=smooth,
            parameters=smooth.parameters_from_paths(
                {
                    "intercept": np.linspace(-0.6, 0.6, N_SESSIONS),
                    "stimulus": np.linspace(0.4, 2.0, N_SESSIONS),
                    "choice_lag_1": np.linspace(0.7, 0.1, N_SESSIONS),
                }
            ),
        ),
        ModelRecoveryScenario(
            name="switching",
            truth_label="hmm",
            generator=hidden_state,
            parameters=hidden_state.parameters_from_components(
                initial_probabilities=[0.5, 0.5],
                transition_matrix=[[0.93, 0.07], [0.07, 0.93]],
                emissions={
                    "intercept": [-1.5, 1.5],
                    "stimulus": [0.5, 1.5],
                    "choice_lag_1": [0.6, 0.1],
                },
            ),
        ),
        ModelRecoveryScenario(
            name="reward-learning",
            truth_label="q-learning",
            generator=q_learning,
            parameters=q_learning.parameters_from_components(
                learning_rate=0.3,
                inverse_temperature=5.0,
                choice_bias=0.0,
                perseveration=0.2,
            ),
        ),
    )
    return scenarios, candidates


def run(*, check: bool = True) -> dict[str, Any]:
    """Run the fixed grid and optionally enforce its qualitative selection contract."""

    scenarios, candidates = experiment()
    grid = run_model_recovery_grid(
        {
            "sparse": build_design(trials_per_session=30),
            "dense": build_design(trials_per_session=60),
        },
        scenarios,
        candidates,
        seed=77,
        min_train_sessions=3,
        tie_tolerance=0.002,
    )
    selections = {
        name: report.selected_labels
        for name, report in zip(grid.design_names, grid.reports, strict=True)
    }
    contract_passed = selections == EXPECTED_SELECTED
    if check and not contract_passed:
        raise AssertionError(
            f"recovery-grid selection contract failed: {selections} != {EXPECTED_SELECTED}"
        )

    cells: list[dict[str, Any]] = []
    for summary, report, design_seed in zip(grid.summary(), grid.reports, grid.seeds, strict=True):
        matrix = report.confusion_matrix()
        cells.append(
            {
                "design": summary.design_name,
                "design_seed": int(design_seed),
                "n_trials": summary.n_trials,
                "n_subjects": summary.n_subjects,
                "n_runs": summary.n_runs,
                "resolution_rate": summary.resolution_rate,
                "overall_accuracy": summary.overall_accuracy,
                "resolved_accuracy": summary.resolved_accuracy,
                "audit_warning_rate": summary.audit_warning_rate,
                "audit_failure_rate": summary.audit_failure_rate,
                "truth_labels": list(report.truth_labels),
                "selected_labels": list(report.selected_labels),
                "run_seeds": report.seeds.tolist(),
                "n_folds": report.n_folds.tolist(),
                "mean_log_probabilities": report.mean_log_probabilities.tolist(),
                "audit_statuses": [
                    [status.value for status in row] for row in report.audit_statuses
                ],
                "audit_issue_codes": [
                    [list(codes) for codes in row] for row in report.audit_issue_codes
                ],
                "confusion": {
                    "truth_labels": list(matrix.truth_labels),
                    "selected_labels": list(matrix.selected_labels),
                    "counts": matrix.counts.tolist(),
                    "rates": matrix.rates.tolist(),
                },
            }
        )
    return {
        "benchmark": "matched four-family prospective model-recovery grid",
        "root_seed": grid.root_seed,
        "candidate_labels": list(grid.reports[0].candidate_labels),
        "contract_passed": contract_passed,
        "designs": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-check", action="store_true", help="report without enforcing contract")
    parser.add_argument("--output", type=Path, help="also write the JSON result to this path")
    args = parser.parse_args()
    result = run(check=not args.no_check)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
