"""Prospectively compare static and smoothly drifting behavioural GLMs."""

from __future__ import annotations

import numpy as np

from behavio import (
    BernoulliHistoryGLM,
    FoldEvaluation,
    Study,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)
from behavio.compose import smooth as make_smooth


def build_design(*, n_sessions: int = 10, trials_per_session: int = 120) -> Study:
    return Study.factorial(
        trials=trials_per_session,
        subjects="synthetic-mouse",
        sessions=n_sessions,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=2026,
    )


def mean_log_loss(evaluations: tuple[FoldEvaluation, ...]) -> float:
    scores = np.concatenate([evaluation.pointwise_log_probability for evaluation in evaluations])
    return -float(np.mean(scores))


def main() -> None:
    design = build_design()
    knots = tuple(range(10))
    smooth = make_smooth(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.01),
        over="session_order",
        knots=tuple(float(knot) for knot in knots),
        smoothness=10.0,
    )
    truth = smooth.parameters_from_paths(
        {
            "intercept": np.linspace(-0.5, 0.5, len(knots)),
            "stimulus": np.linspace(0.2, 2.5, len(knots)),
            "choice_lag_1": np.linspace(0.8, 0.1, len(knots)),
        }
    )
    study = smooth.simulate(design, truth, seed=77)
    splits = forward_session_splits(study, min_train_sessions=3)

    static = BernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.01,
    )
    static_evaluations = evaluate_splits(static, study, splits)
    smooth_evaluations = evaluate_splits(smooth, study, splits)
    static_loss = mean_log_loss(static_evaluations)
    smooth_loss = mean_log_loss(smooth_evaluations)

    print("Prospective stationary-versus-smooth comparison")
    print(f"static mean log loss: {static_loss:.3f}")
    print(f"smooth mean log loss: {smooth_loss:.3f}")
    print(f"smooth improvement:   {static_loss - smooth_loss:.3f}")

    fit = smooth.fit(study)
    trajectory = smooth.coefficient_trajectory(fit)
    stimulus_column = trajectory.coefficient_names.index("stimulus")
    print("\nFitted stimulus trajectory")
    print("session_order  coefficient")
    for time, value in zip(trajectory.times, trajectory.values[:, stimulus_column], strict=True):
        print(f"{time:>13.0f}  {value:>11.3f}")

    alternative = smooth.parameters_from_paths(
        {
            "intercept": np.linspace(-0.3, 0.3, len(knots)),
            "stimulus": np.linspace(0.5, 2.0, len(knots)),
            "choice_lag_1": np.linspace(0.6, 0.2, len(knots)),
        }
    )
    recovery = run_parameter_recovery(
        smooth,
        design,
        [truth, alternative],
        seed=91,
    )
    mean_rmse = float(np.mean([row.rmse for row in recovery.summary()]))
    print("\nSmooth-path parameter recovery")
    print(
        f"{recovery.n_runs} runs, convergence={recovery.convergence_rate:.0%}, "
        f"mean knot-parameter RMSE={mean_rmse:.3f}"
    )


if __name__ == "__main__":
    main()
