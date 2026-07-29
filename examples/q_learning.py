"""Recover and prospectively challenge a compact binary Q-learning agent."""

from __future__ import annotations

from behavio import (
    BernoulliHistoryGLM,
    BinaryQLearning,
    SmoothBernoulliHistoryGLM,
    Study,
    evaluate_splits,
    forward_session_splits,
)


def build_design(*, n_sessions: int = 8, trials_per_session: int = 120) -> Study:
    probability_1: list[float] = []
    for session in range(n_sessions):
        for trial in range(trials_per_session):
            action_1_is_rich = ((trial // 30) + session) % 2
            probability_1.append(0.8 if action_1_is_rich else 0.2)
    n_trials = n_sessions * trials_per_session
    return Study(
        {
            "subject": ["synthetic-mouse"] * n_trials,
            "session": [
                f"session-{session}"
                for session in range(n_sessions)
                for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions,
            "session_order": [
                session for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "reward_probability_0": [1.0 - value for value in probability_1],
            "reward_probability_1": probability_1,
        }
    )


def main() -> None:
    model = BinaryQLearning(
        n_restarts=4,
        random_seed=4,
    )
    truth = model.parameters_from_components(
        learning_rate=0.25,
        inverse_temperature=5.0,
        choice_bias=0.15,
        perseveration=0.3,
    )
    study = model.simulate(build_design(), truth, seed=9)
    fit = model.fit(study)
    recovered = model.parameter_components(fit)
    values = model.value_trajectory(study, fit)
    audit = fit.audit()

    print("Session-reset binary Q-learning")
    print(f"learning rate:       {recovered.learning_rate:.3f}")
    print(f"inverse temperature: {recovered.inverse_temperature:.3f}")
    print(f"choice bias:         {recovered.choice_bias:.3f}")
    print(f"perseveration:       {recovered.perseveration:.3f}")
    print(f"restart objectives:  {fit.restart_objectives.round(3).tolist()}")
    print(f"fit audit:           {audit.status.value} {list(audit.issue_codes)}")
    print(f"first prediction error: {values.prediction_error[0]:.3f}")

    splits = forward_session_splits(study, min_train_sessions=7)
    static = BernoulliHistoryGLM(choice_lags=1, l2=0.01)
    smooth = SmoothBernoulliHistoryGLM(
        choice_lags=1,
        knots=tuple(range(8)),
        smoothness=10.0,
        l2=0.01,
    )
    print("\nProspective competing explanations")
    for label, candidate in (("static", static), ("smooth", smooth), ("Q-learning", model)):
        evaluation = evaluate_splits(candidate, study, splits)[0]
        print(f"{label:>10}: log-loss={evaluation.mean_log_loss:.3f}")


if __name__ == "__main__":
    main()
