"""Recover population and individual coefficient trajectories."""

from __future__ import annotations

import numpy as np

from behavio import BernoulliHistoryGLM, Study
from behavio.compose import hierarchical, smooth


def build_design() -> Study:
    return Study.factorial(
        trials=60,
        subjects=tuple(f"mouse-{index}" for index in range(6)),
        sessions=5,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=321,
    )


def main() -> None:
    paths = smooth(
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0, l2=0.02),
        over="session_order",
        knots=(0.0, 2.0, 4.0),
        smoothness=3.0,
    )
    model = hierarchical(paths, over="subject", scale=0.4)
    population_paths = {
        "intercept": [-0.3, -0.2, -0.1],
        "stimulus": [0.8, 1.0, 1.2],
    }
    centered_knots = np.asarray([-1.0, 0.0, 1.0])
    deviation_generator = np.random.default_rng(99)
    design = build_design()
    deviations = {}
    for subject in design.subjects:
        subject_paths = []
        for _ in paths.coefficient_names:
            offset = deviation_generator.normal(0.0, 0.15)
            slope = deviation_generator.normal(0.0, 0.3)
            subject_paths.extend((offset + slope * centered_knots).tolist())
        deviations[subject] = subject_paths

    simulation = model.simulate_with_effects(
        design,
        paths.parameters_from_paths(population_paths),
        seed=41,
        group_deviations=deviations,
    )
    fit = model.fit(simulation.study)
    population = paths.trajectory_from_knots(fit.estimates)
    subject = paths.trajectory_from_knots(np.asarray(list(fit.parameters_for("mouse-0").values())))

    print("Partially pooled coefficient trajectories")
    print(f"converged: {fit.diagnostics.converged}")
    print(f"unseen-group policy: {fit.unseen_group_policy}")
    print(f"population stimulus: {population.values[:, 1].round(3).tolist()}")
    print(f"mouse-0 stimulus:     {subject.values[:, 1].round(3).tolist()}")
    print(f"mouse-0 fitted: {fit.group_was_fitted('mouse-0')}")
    print(f"new-mouse fitted: {fit.group_was_fitted('new-mouse')}")


if __name__ == "__main__":
    main()
