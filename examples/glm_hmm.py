"""Fit and prospectively challenge a fixed-transition Bernoulli GLM-HMM."""

from __future__ import annotations

import numpy as np

from behavio import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    SmoothBernoulliHistoryGLM,
    Study,
    evaluate_splits,
    forward_session_splits,
)


def build_design(*, n_sessions: int = 6, trials_per_session: int = 100) -> Study:
    return Study.factorial(
        trials=trials_per_session,
        subjects="synthetic-mouse",
        sessions=n_sessions,
    )


def main() -> None:
    model = BernoulliGLMHMM(
        choice_lags=0,
        n_restarts=3,
        random_seed=7,
        l2=0.01,
    )
    truth = model.parameters_from_components(
        initial_probabilities=[0.5, 0.5],
        transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
        emissions={"intercept": [-2.0, 2.0]},
    )
    simulation = model.simulate_with_states(build_design(), truth, seed=2026)
    fit = model.fit(simulation.study)
    fitted = model.parameter_components(fit)
    recovery = model.state_recovery(simulation, fit)
    audit = fit.audit()

    print("Fixed-transition Bernoulli GLM-HMM")
    print("fitted transition matrix:")
    print(np.array2string(fitted.transition_matrix, precision=3))
    print(f"restart objectives: {np.round(fit.restart_objectives, 3).tolist()}")
    print(f"state occupancy:    {np.round(fit.state_occupancy, 3).tolist()}")
    print(f"label ambiguous:    {fit.label_ambiguous}")
    print(f"fit audit:          {audit.status.value} {list(audit.issue_codes)}")
    print(f"aligned filtered state accuracy: {recovery.decoded_accuracy:.1%}")
    print(f"alignment gap:      {recovery.score_gap:.3f}")
    print(f"alignment ambiguous: {recovery.ambiguous}")

    splits = forward_session_splits(simulation.study, min_train_sessions=5)
    static = BernoulliHistoryGLM(choice_lags=0, l2=0.01)
    smooth = SmoothBernoulliHistoryGLM(
        choice_lags=0,
        knots=tuple(range(6)),
        smoothness=10.0,
        l2=0.01,
    )
    print("\nProspective competing explanations")
    for label, candidate in (("static", static), ("smooth", smooth), ("GLM-HMM", model)):
        evaluation = evaluate_splits(candidate, simulation.study, splits)[0]
        print(f"{label:>8}: log-loss={evaluation.mean_log_loss:.3f}")


if __name__ == "__main__":
    main()
