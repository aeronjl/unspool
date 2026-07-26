"""Fit the bounded population GLM and inspect its prediction policy."""

from __future__ import annotations

import numpy as np

from unspool import HierarchicalBernoulliHistoryGLM, Study


def build_design() -> Study:
    generator = np.random.default_rng(2106)
    subjects = ("mouse-a", "mouse-b", "mouse-c", "mouse-d")
    n_sessions = 3
    trials_per_session = 60
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
            "trial": list(range(trials_per_session)) * n_sessions * len(subjects),
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=len(subjects) * n_sessions * trials_per_session),
        }
    )


def main() -> None:
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.05,
        subject_scale=0.45,
    )
    truth = {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.35}
    simulation = model.simulate_with_effects(build_design(), truth, seed=71)
    fit = model.fit(simulation.study)

    print("Fixed-scale hierarchical Bernoulli GLM")
    print(f"converged: {fit.diagnostics.converged}")
    print(f"subject scale (fixed): {fit.subject_scale}")
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
