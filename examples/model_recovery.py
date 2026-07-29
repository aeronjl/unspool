"""Recover static and smooth model families with prospective session scoring."""

from __future__ import annotations

import numpy as np

from behavio import (
    BernoulliHistoryGLM,
    ModelRecoveryScenario,
    SmoothBernoulliHistoryGLM,
    Study,
    run_model_recovery,
)


def build_design(*, n_sessions: int = 10, trials_per_session: int = 120) -> Study:
    generator = np.random.default_rng(2027)
    n_trials = n_sessions * trials_per_session
    return Study(
        {
            "subject": ["synthetic-mouse"] * n_trials,
            "session": [
                f"session-{session}"
                for session in range(n_sessions)
                for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions,
            "session_order": [
                session for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=n_trials),
        }
    )


def main() -> None:
    design = build_design()
    static = BernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.01,
    )
    smooth = SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        knots=tuple(range(10)),
        smoothness=10.0,
        l2=0.01,
    )
    scenarios = [
        ModelRecoveryScenario(
            name="stationary",
            truth_label="static",
            generator=static,
            parameters={
                "intercept": -0.2,
                "stimulus": 1.2,
                "choice_lag_1": 0.4,
            },
        ),
        ModelRecoveryScenario(
            name="drifting",
            truth_label="smooth",
            generator=smooth,
            parameters=smooth.parameters_from_paths(
                {
                    "intercept": np.linspace(-0.5, 0.5, len(smooth.knots)),
                    "stimulus": np.linspace(0.2, 2.5, len(smooth.knots)),
                    "choice_lag_1": np.linspace(0.8, 0.1, len(smooth.knots)),
                }
            ),
        ),
    ]
    report = run_model_recovery(
        design,
        scenarios,
        {"static": static, "smooth": smooth},
        repeats=3,
        seed=12,
        min_train_sessions=3,
        tie_tolerance=0.001,
    )
    matrix = report.confusion_matrix()

    print("Prospective static-versus-smooth model recovery")
    print("truth       static    smooth  unresolved")
    for truth, rates in zip(matrix.truth_labels, matrix.rates, strict=True):
        print(
            f"{truth:<10} "
            + "  ".join(f"{rate:>8.1%}" if np.isfinite(rate) else f"{'n/a':>8}" for rate in rates)
        )
    print(f"resolution rate: {report.resolution_rate:.1%}")
    print(f"overall accuracy: {report.overall_accuracy:.1%}")
    print(f"accuracy among resolved runs: {report.resolved_accuracy:.1%}")


if __name__ == "__main__":
    main()
