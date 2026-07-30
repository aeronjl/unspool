"""Fit the bounded population GLM and inspect its prediction policy."""

from __future__ import annotations

from behavio import BernoulliHistoryGLM, Study
from behavio.compose import hierarchical


def build_design() -> Study:
    return Study.factorial(
        trials=60,
        subjects=("mouse-a", "mouse-b", "mouse-c", "mouse-d"),
        sessions=3,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=2106,
    )


def main() -> None:
    base = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.05)
    generator = hierarchical(base, over="subject", scale=0.45)
    truth = {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.35}
    simulation = generator.simulate_with_effects(build_design(), truth, seed=71)
    model = hierarchical(
        base,
        over="subject",
        scale=0.25,
        estimate_scale=True,
        scale_bounds=(0.05, 1.5),
    )
    fit = model.fit(simulation.study)

    print("Hierarchical Bernoulli GLM with estimated scale")
    print(f"converged: {fit.diagnostics.converged}")
    print(f"subject scale (estimated): {fit.scales[0]:.3f}")
    print(f"scale interval: {fit.scale_confidence_interval_95}")
    print(f"scale at boundary: {fit.scale_at_boundary}")
    print(f"unseen-group policy: {fit.unseen_group_policy}")
    print("\nsubject       true stimulus  fitted stimulus")
    stimulus_index = fit.varying_parameters.index("stimulus")
    for index, subject in enumerate(fit.groups):
        print(
            f"{subject:<13} "
            f"{simulation.group_parameters[index, stimulus_index]:>13.3f}  "
            f"{fit.group_parameters[index, stimulus_index]:>15.3f}"
        )

    print("\nPrediction policy")
    print(f"mouse-a fitted: {fit.group_was_fitted('mouse-a')}")
    print(f"new-mouse fitted: {fit.group_was_fitted('new-mouse')}")
    print(f"new-mouse parameters: {dict(fit.parameters_for('new-mouse'))}")


if __name__ == "__main__":
    main()
