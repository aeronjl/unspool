"""Compare and select static versus smooth trajectories without test leakage."""

from __future__ import annotations

from behavio import (
    BernoulliHistoryGLM,
    Study,
    cohort_forward_session_splits,
    compare_models,
    nested_select_model,
)
from behavio.compose import SmoothModel, smooth

KNOTS = (0.0, 2.0, 5.0)


def make_study() -> Study:
    design = Study.factorial(
        trials=30,
        subjects=tuple(f"mouse-{index}" for index in range(4)),
        sessions=6,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=51,
    )
    generator_model = smooth_model()
    return generator_model.simulate(
        design,
        generator_model.parameters_from_paths(
            {
                "intercept": [-0.5, 0.0, 0.5],
                "stimulus": [0.3, 1.0, 1.8],
                "choice_lag_1": [0.2, 0.2, 0.2],
            }
        ),
        seed=52,
    )


def smooth_model() -> SmoothModel:
    return smooth(
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02),
        over="session_order",
        knots=KNOTS,
        smoothness=3.0,
        shared_trajectory=True,
    )


def candidates():
    return {
        "static": BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02),
        "shared_smooth": smooth_model(),
    }


def main() -> None:
    study = make_study()
    outer_splits = cohort_forward_session_splits(study, min_train_sessions=5)
    report = compare_models(
        candidates(),
        study,
        outer_splits,
        bootstrap_resamples=200,
        bootstrap_seed=53,
    )
    print("Subject-balanced prospective comparison")
    for result in report.model_results:
        interval = result.unit_balanced_log_loss_interval
        print(
            f"{result.name}: loss={result.unit_balanced_log_loss:.3f}, "
            f"95% interval=({interval.lower:.3f}, {interval.upper:.3f}), "
            f"audit={result.audit_status.value}"
        )

    def inner_splitter(training: Study):
        return cohort_forward_session_splits(training, min_train_sessions=3)

    nested = nested_select_model(
        candidates(),
        study,
        outer_splits,
        inner_splitter,
        bootstrap_resamples=200,
        inner_bootstrap_resamples=100,
        bootstrap_seed=54,
    )
    print("\nTraining-only nested selection")
    print(f"selected: {nested.folds[0].selected_model}")
    print(f"outer loss: {nested.unit_balanced_log_loss:.3f}")


if __name__ == "__main__":
    main()
