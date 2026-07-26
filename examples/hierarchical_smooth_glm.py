"""Recover population and individual coefficient trajectories."""

from __future__ import annotations

import numpy as np

from unspool import HierarchicalSmoothBernoulliHistoryGLM, Study


def build_design() -> Study:
    generator = np.random.default_rng(321)
    subjects = tuple(f"mouse-{index}" for index in range(6))
    n_sessions = 5
    trials_per_session = 60
    n_rows = len(subjects) * n_sessions * trials_per_session
    return Study(
        {
            "subject": [
                subject
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "session": [
                f"session-{session}"
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * len(subjects) * n_sessions,
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=n_rows),
        }
    )


def main() -> None:
    model = HierarchicalSmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=0,
        knots=(0.0, 2.0, 4.0),
        smoothness=3.0,
        l2=0.02,
        subject_scale=0.4,
        subject_smoothness=3.0,
    )
    population_paths = {
        "intercept": [-0.3, -0.2, -0.1],
        "stimulus": [0.8, 1.0, 1.2],
    }
    centered_knots = np.asarray([-1.0, 0.0, 1.0])
    deviation_generator = np.random.default_rng(99)
    design = build_design()
    deviations = {}
    for subject in design.subjects:
        deviations[subject] = {}
        for coefficient in model.coefficient_names:
            offset = deviation_generator.normal(0.0, 0.15)
            slope = deviation_generator.normal(0.0, 0.3)
            deviations[subject][coefficient] = (offset + slope * centered_knots).tolist()

    simulation = model.simulate_with_effects(
        design,
        model.parameters_from_paths(population_paths),
        seed=41,
        subject_deviation_paths=deviations,
    )
    fit = model.fit(simulation.study)
    population = model.population_trajectory(fit)
    subject = model.subject_trajectory(fit, "mouse-0")

    print("Partially pooled coefficient trajectories")
    print(f"converged: {fit.diagnostics.converged}")
    print(f"unseen-subject policy: {fit.unseen_subject_policy}")
    print(f"population stimulus: {population.values[:, 1].round(3).tolist()}")
    print(f"mouse-0 stimulus:     {subject.values[:, 1].round(3).tolist()}")
    print(f"mouse-0 fitted: {fit.subject_was_fitted('mouse-0')}")
    print(f"new-mouse fitted: {fit.subject_was_fitted('new-mouse')}")


if __name__ == "__main__":
    main()
