from collections.abc import Mapping

import numpy as np
import pytest

from behavio import (
    BehaviourModel,
    BernoulliHistoryGLM,
    BinaryQLearning,
    ChoiceSpec,
    FitDiagnostics,
    FitResult,
    FittedModel,
    ModelDataError,
    PredictionMode,
    RewardSpec,
    SmoothBernoulliHistoryGLM,
    Study,
    TaskSpec,
    UnsupportedPredictionMode,
    evaluate_splits,
    export_fit,
    forward_session_splits,
    run_parameter_recovery,
    within_session_rolling_splits,
)
from behavio.models._kernels.bernoulli import ordered_session_indices


def volatile_design(*, n_sessions: int = 8, trials_per_session: int = 120) -> Study:
    probability_1: list[float] = []
    for session in range(n_sessions):
        for trial in range(trials_per_session):
            action_1_is_rich = ((trial // 30) + session) % 2
            probability_1.append(0.8 if action_1_is_rich else 0.2)
    n_trials = n_sessions * trials_per_session
    return Study(
        {
            "subject": ["a"] * n_trials,
            "session": [
                f"s{session}" for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions,
            "session_order": [
                session for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "reward_probability_0": [1.0 - value for value in probability_1],
            "reward_probability_1": probability_1,
        }
    )


def agent(*, n_restarts: int = 4) -> BinaryQLearning:
    return BinaryQLearning(
        n_restarts=n_restarts,
        random_seed=4,
        max_iterations=500,
    )


def agent_parameters(model: BinaryQLearning) -> Mapping[str, float]:
    return model.parameters_from_components(
        learning_rate=0.25,
        inverse_temperature=5.0,
        choice_bias=0.15,
        perseveration=0.3,
    )


def fixed_fit(model: BinaryQLearning, parameters: Mapping[str, float], n: int) -> FitResult:
    estimates = np.asarray([parameters[name] for name in model.parameter_names])
    return FitResult(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        estimates=estimates,
        standard_errors=np.zeros_like(estimates),
        covariance=np.zeros((len(estimates), len(estimates))),
        n_observations=n,
        diagnostics=FitDiagnostics(
            converged=True,
            optimizer="known truth",
            status=0,
            message="known truth",
            n_iterations=0,
            objective=0.0,
            gradient_norm=0.0,
            hessian_condition=1.0,
            boundary_estimate=False,
        ),
    )


def test_natural_parameters_round_trip_through_optimizer_coordinates() -> None:
    model = agent()
    parameters = agent_parameters(model)

    decoded = model.parameter_components(parameters)

    assert decoded.learning_rate == pytest.approx(0.25)
    assert decoded.inverse_temperature == pytest.approx(5.0)
    assert decoded.choice_bias == pytest.approx(0.15)
    assert decoded.perseveration == pytest.approx(0.3)
    assert tuple(parameters) == model.parameter_names
    assert model.required_task_columns == ("reward",)


def test_simulation_is_recursive_reproducible_and_action_contingent() -> None:
    model = agent()
    design = Study(
        {
            "subject": ["a"] * 20,
            "session": ["s1"] * 20,
            "trial": list(range(20)),
            "session_order": [0] * 20,
            "reward_probability_0": [0.0] * 20,
            "reward_probability_1": [1.0] * 20,
        }
    )

    first = model.simulate(design, agent_parameters(model), seed=12)
    second = model.simulate(design, agent_parameters(model), seed=12)

    np.testing.assert_array_equal(first["choice"], second["choice"])
    np.testing.assert_array_equal(first["reward"], second["reward"])
    np.testing.assert_array_equal(first["reward"], first["choice"])
    assert first["session"].tolist() == design["session"].tolist()


def test_value_trajectory_updates_after_choice_and_resets_each_session() -> None:
    model = agent()
    parameters = model.parameters_from_components(
        learning_rate=0.5,
        inverse_temperature=2.0,
    )
    study = Study(
        {
            "subject": ["a", "a", "a"],
            "session": ["s1", "s1", "s2"],
            "trial": [0, 1, 0],
            "session_order": [0, 0, 1],
            "choice": [1, 0, 1],
            "reward": [1.0, 0.0, 0.0],
        }
    )
    fit = fixed_fit(model, parameters, len(study))

    trajectory = model.value_trajectory(study, fit)

    np.testing.assert_allclose(
        trajectory.pre_choice,
        [[0.5, 0.5], [0.5, 0.75], [0.5, 0.5]],
    )
    np.testing.assert_allclose(
        trajectory.post_update,
        [[0.5, 0.75], [0.25, 0.75], [0.5, 0.25]],
    )
    np.testing.assert_allclose(trajectory.prediction_error, [0.5, -0.5, -0.5])
    np.testing.assert_allclose(trajectory.linear_predictor, [0.0, 0.5, 0.0])


def test_pointwise_scores_match_recursive_likelihood() -> None:
    model = agent()
    parameters = agent_parameters(model)
    study = model.simulate(
        volatile_design(n_sessions=2, trials_per_session=40),
        parameters,
        seed=21,
    )
    fit = fixed_fit(model, parameters, len(study))
    choices = model._choices(study)
    rewards = model._rewards(study)
    vector = np.asarray([parameters[name] for name in model.parameter_names])

    objective, _ = model._objective_gradient(
        vector,
        choices,
        rewards,
        ordered_session_indices(study),
    )
    scores = model.pointwise_log_prob(study, fit)

    assert -float(np.sum(scores)) == pytest.approx(objective, abs=1e-10)


def test_recursive_analytic_gradient_matches_finite_differences() -> None:
    model = agent(n_restarts=1)
    study = Study(
        {
            "subject": ["a"] * 8,
            "session": ["s1"] * 4 + ["s2"] * 4,
            "trial": [0, 1, 2, 3] * 2,
            "session_order": [0] * 4 + [1] * 4,
            "choice": [0, 1, 1, 0, 1, 0, 1, 1],
            "reward": [0.0, 1.0, 0.2, 0.8, 1.0, 0.0, 0.5, 1.0],
        }
    )
    parameters = agent_parameters(model)
    vector = np.asarray([parameters[name] for name in model.parameter_names])
    choices = model._choices(study)
    rewards = model._rewards(study)
    sessions = ordered_session_indices(study)

    _, analytic = model._objective_gradient(vector, choices, rewards, sessions)
    numeric = np.empty_like(analytic)
    for index in range(len(vector)):
        step = 1e-6 * (1.0 + abs(vector[index]))
        positive = vector.copy()
        negative = vector.copy()
        positive[index] += step
        negative[index] -= step
        positive_value, _ = model._objective_gradient(positive, choices, rewards, sessions)
        negative_value, _ = model._objective_gradient(negative, choices, rewards, sessions)
        numeric[index] = (positive_value - negative_value) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-6)


def test_fit_recovers_agent_and_retains_all_restart_outcomes() -> None:
    model = agent()
    study = model.simulate(volatile_design(), agent_parameters(model), seed=8)

    fit = model.fit(study)
    recovered = model.parameter_components(fit)

    assert fit.diagnostics.converged
    assert fit.restart_objectives.shape == (model.n_restarts,)
    assert fit.restart_converged.shape == (model.n_restarts,)
    assert len(fit.optimization_run.attempts) == model.n_restarts
    assert fit.optimization_run.selected_index == fit.selected_restart
    assert fit.optimization_run.problem["parameter_space_fingerprint"] == (
        model.parameter_space.fingerprint
    )
    assert fit.diagnostics.optimizer.startswith("scipy.optimize.minimize/L-BFGS-B")
    task = TaskSpec(
        choice=ChoiceSpec(options=(0, 1)),
        reward=RewardSpec(column="reward", minimum=0.0, maximum=1.0),
    )
    artifact = export_fit(FittedModel(model, task, fit, task.validate(study)), study)
    assert len(artifact.diagnostics["optimization_run"]["attempts"]) == model.n_restarts
    assert artifact.diagnostics["optimization_run"]["selected_index"] == fit.selected_restart
    assert fit.learning_rate == pytest.approx(recovered.learning_rate)
    assert fit.inverse_temperature == pytest.approx(recovered.inverse_temperature)
    assert recovered.learning_rate == pytest.approx(0.25, abs=0.08)
    assert recovered.inverse_temperature == pytest.approx(5.0, abs=1.0)
    assert recovered.choice_bias == pytest.approx(0.15, abs=0.3)
    assert recovered.perseveration == pytest.approx(0.3, abs=0.35)


def test_generic_parameter_recovery_uses_encoded_agent_coordinates() -> None:
    model = agent(n_restarts=3)
    parameters = agent_parameters(model)

    report = run_parameter_recovery(
        model,
        volatile_design(),
        [parameters],
        seed=31,
    )
    estimate_map = dict(zip(model.parameter_names, report.estimates[0], strict=True))
    recovered = model.parameter_components(estimate_map)

    assert report.convergence_rate == 1.0
    assert recovered.learning_rate == pytest.approx(0.25, abs=0.1)
    assert recovered.inverse_temperature == pytest.approx(5.0, abs=1.5)


def test_reward_learning_beats_static_and_smooth_glms_prospectively() -> None:
    model = agent(n_restarts=3)
    study = model.simulate(volatile_design(), agent_parameters(model), seed=9)
    splits = forward_session_splits(study, min_train_sessions=7)
    static = BernoulliHistoryGLM(choice_lags=1, l2=0.01)
    smooth = SmoothBernoulliHistoryGLM(
        choice_lags=1,
        knots=tuple(range(8)),
        smoothness=10.0,
        l2=0.01,
    )

    static_loss = evaluate_splits(static, study, splits)[0].mean_log_loss
    smooth_loss = evaluate_splits(smooth, study, splits)[0].mean_log_loss
    learning_loss = evaluate_splits(model, study, splits)[0].mean_log_loss

    assert learning_loss < static_loss - 0.1
    assert learning_loss < smooth_loss - 0.1


def test_within_session_evaluation_replays_value_context() -> None:
    model = agent(n_restarts=2)
    study = model.simulate(
        volatile_design(n_sessions=3, trials_per_session=60),
        agent_parameters(model),
        seed=17,
    )
    split = within_session_rolling_splits(
        study,
        min_train_sessions=1,
        min_train_trials=20,
        horizon=3,
        step=100,
    )[0]

    evaluation = evaluate_splits(model, study, [split])[0]
    session_rows = np.flatnonzero(study["session_order"] == split.origin_session_order)
    full_session = study.take(session_rows)
    full_prediction = model.predict(full_session, evaluation.fit)

    np.testing.assert_allclose(
        evaluation.prediction.probability,
        full_prediction.probability[20:23],
    )
    reset = model.predict(study.take(split.test_indices), evaluation.fit)
    assert not np.isclose(evaluation.prediction.probability[0], reset.probability[0])


def test_fitting_does_not_require_generative_environment_columns() -> None:
    model = agent(n_restarts=2)
    simulated = model.simulate(
        volatile_design(n_sessions=2, trials_per_session=40),
        agent_parameters(model),
        seed=19,
    )
    observed = Study(
        {
            name: simulated[name]
            for name in simulated.columns
            if name not in model.reward_probability_columns
        }
    )

    fit = model.fit(observed)

    assert fit.diagnostics.converged


def test_simulation_requires_a_valid_reward_environment() -> None:
    model = agent()
    no_environment = Study(
        {
            "subject": ["a"],
            "session": ["s1"],
            "trial": [0],
            "session_order": [0],
        }
    )

    with pytest.raises(ModelDataError, match="reward probability columns"):
        model.simulate(no_environment, agent_parameters(model), seed=1)


def test_invalid_observed_rewards_are_rejected() -> None:
    model = agent()
    study = Study(
        {
            "subject": ["a"],
            "session": ["s1"],
            "trial": [0],
            "session_order": [0],
            "choice": [1],
            "reward": [2.0],
        }
    )

    with pytest.raises(ModelDataError, match="between zero and one"):
        model.fit(study)


def test_smoothed_prediction_is_not_silently_substituted() -> None:
    model = agent()
    parameters = agent_parameters(model)
    study = model.simulate(
        volatile_design(n_sessions=1, trials_per_session=20),
        parameters,
        seed=7,
    )
    fit = fixed_fit(model, parameters, len(study))

    with pytest.raises(UnsupportedPredictionMode, match="filtered prediction"):
        model.predict(study, fit, mode=PredictionMode.SMOOTHED)
    assert isinstance(model, BehaviourModel)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("outcome", "trial", "outcome"),
        ("reward", "choice", "distinct"),
        ("reward_probability_columns", ("p", "p"), "distinct"),
        ("initial_value", -0.1, "initial_value"),
        ("n_restarts", 0, "n_restarts"),
        ("random_seed", -1, "random_seed"),
        ("max_iterations", True, "max_iterations"),
        ("tolerance", 0.0, "tolerance"),
        ("learning_rate_warning_threshold", 0.5, "learning_rate_warning_threshold"),
        ("inverse_temperature_warning_threshold", 0.5, "inverse_temperature_warning_threshold"),
        ("inverse_temperature_warning_threshold", 0.0, "inverse_temperature_warning_threshold"),
    ],
)
def test_agent_configuration_is_validated(
    argument: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BinaryQLearning(**{argument: value})
