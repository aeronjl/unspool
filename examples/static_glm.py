"""End-to-end synthetic analysis with the first Behavio reference model."""

from __future__ import annotations

from behavio import (
    BernoulliHistoryGLM,
    Study,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)


def build_design(*, n_sessions: int = 5, trials_per_session: int = 120) -> Study:
    """Create fixed experimental covariates without generating outcomes."""

    return Study.factorial(
        trials=trials_per_session,
        subjects="synthetic-mouse",
        sessions=n_sessions,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=2025,
    )


def main() -> None:
    design = build_design()
    model = BernoulliHistoryGLM(
        predictors=("stimulus",),
        choice_lags=1,
        l2=0.05,
    )
    truth = {
        "intercept": -0.2,
        "stimulus": 1.1,
        "choice_lag_1": 0.5,
    }
    study = model.simulate(design, truth, seed=17)

    splits = forward_session_splits(study, min_train_sessions=2)
    evaluations = evaluate_splits(model, study, splits)

    print("Prospective session folds")
    print("train_sessions  test_session  mean_log_loss  converged")
    for evaluation in evaluations:
        split = evaluation.split
        print(
            f"{len(split.train_sessions):>14}  {split.test_sessions[0]:>12}  "
            f"{evaluation.mean_log_loss:>13.3f}  "
            f"{evaluation.fit.diagnostics.converged!s:>9}"
        )

    parameter_sets = [
        {"intercept": -0.4, "stimulus": 0.7, "choice_lag_1": 0.2},
        truth,
        {"intercept": 0.2, "stimulus": 1.5, "choice_lag_1": 0.8},
    ]
    recovery = run_parameter_recovery(
        model,
        design,
        parameter_sets,
        repeats=2,
        seed=29,
    )

    print("\nDesign-specific parameter recovery")
    print(
        f"{recovery.n_runs} runs, {recovery.n_subjects} subject, "
        f"{recovery.n_trials} trials, convergence={recovery.convergence_rate:.0%}"
    )
    print("parameter          bias     rmse     corr  coverage")
    for summary in recovery.summary():
        print(
            f"{summary.parameter:<16} {summary.bias:>7.3f}  {summary.rmse:>7.3f}  "
            f"{summary.correlation:>7.3f}  {summary.coverage_95:>8.1%}"
        )


if __name__ == "__main__":
    main()
