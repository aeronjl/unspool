"""Evaluate filtered predictions at prospective origins inside a session."""

from __future__ import annotations

from behavio import (
    BernoulliHistoryGLM,
    Study,
    evaluate_splits,
    within_session_rolling_splits,
)


def build_design() -> Study:
    return Study.factorial(
        trials=40,
        subjects="synthetic-mouse",
        sessions=3,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=2026,
    )


def main() -> None:
    model = BernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.1,
    )
    study = model.simulate(
        build_design(),
        {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.5},
        seed=77,
    )
    splits = within_session_rolling_splits(
        study,
        min_train_sessions=2,
        min_train_trials=20,
        horizon=5,
        step=10,
    )
    evaluations = evaluate_splits(model, study, splits)

    print("Prospective within-session rolling origins")
    for evaluation in evaluations:
        split = evaluation.split
        print(
            f"origin={split.origin_session}:{split.origin_trial}, "
            f"fit={evaluation.fit.n_observations}, "
            f"context={len(split.prediction_context_indices)}, "
            f"test={split.test_trials}, "
            f"log-loss={evaluation.mean_log_loss:.3f}"
        )


if __name__ == "__main__":
    main()
