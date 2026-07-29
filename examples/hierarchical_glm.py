"""Fit the bounded population GLM and inspect its prediction policy."""

from __future__ import annotations

from behavio import HierarchicalBernoulliHistoryGLM, Study


def build_design() -> Study:
    return Study.factorial(
        trials=60,
        subjects=("mouse-a", "mouse-b", "mouse-c", "mouse-d"),
        sessions=3,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=2106,
    )


def main() -> None:
    generator = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.05,
        subject_scale=0.45,
    )
    truth = {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.35}
    simulation = generator.simulate_with_effects(build_design(), truth, seed=71)
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.05,
        subject_scale=0.25,
        estimate_subject_scale=True,
        subject_scale_bounds=(0.05, 1.5),
    )
    fit = model.fit(simulation.study)

    print("Hierarchical Bernoulli GLM with estimated scale")
    print(f"converged: {fit.diagnostics.converged}")
    print(f"subject scale (estimated): {fit.subject_scale:.3f}")
    print(f"scale interval: {fit.subject_scale_confidence_interval_95}")
    print(f"scale at boundary: {fit.subject_scale_at_boundary}")
    print(f"unseen-subject policy: {fit.unseen_subject_policy}")
    print("\nsubject       true stimulus  fitted stimulus")
    stimulus_index = fit.parameter_names.index("stimulus")
    for index, subject in enumerate(fit.subjects):
        print(
            f"{subject:<13} "
            f"{simulation.subject_coefficients[index, stimulus_index]:>13.3f}  "
            f"{fit.subject_coefficients[index, stimulus_index]:>15.3f}"
        )

    print("\nPrediction policy")
    print(f"mouse-a fitted: {fit.subject_was_fitted('mouse-a')}")
    print(f"new-mouse fitted: {fit.subject_was_fitted('new-mouse')}")
    print(f"new-mouse coefficients: {dict(fit.coefficients_for('new-mouse'))}")


if __name__ == "__main__":
    main()
