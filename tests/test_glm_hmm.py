from collections.abc import Mapping

import numpy as np
import pytest

from unspool import (
    BehaviourModel,
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    FitDiagnostics,
    FitResult,
    PredictionMode,
    SmoothBernoulliHistoryGLM,
    Study,
    UnsupportedPredictionMode,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)
from unspool.models.glm import _ordered_session_indices


def design(*, n_sessions: int = 4, trials_per_session: int = 100) -> Study:
    generator = np.random.default_rng(91)
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
            "stimulus": generator.normal(size=n_trials),
        }
    )


def clear_model(*, n_restarts: int = 3) -> BernoulliGLMHMM:
    return BernoulliGLMHMM(
        choice_lags=0,
        l2=0.01,
        n_restarts=n_restarts,
        max_iterations=400,
        random_seed=7,
    )


def clear_parameters(model: BernoulliGLMHMM) -> Mapping[str, float]:
    return model.parameters_from_components(
        initial_probabilities=[0.5, 0.5],
        transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
        emissions={"intercept": [-2.0, 2.0]},
    )


def fixed_fit(model: BernoulliGLMHMM, parameters: Mapping[str, float], n: int) -> FitResult:
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


def test_parameter_components_are_canonicalized_and_round_trip() -> None:
    model = clear_model()

    parameters = model.parameters_from_components(
        initial_probabilities=[0.2, 0.8],
        transition_matrix=[[0.9, 0.1], [0.2, 0.8]],
        emissions={"intercept": [2.0, -2.0]},
    )
    components = model.parameter_components(parameters)

    np.testing.assert_allclose(components.initial_probabilities, [0.8, 0.2])
    np.testing.assert_allclose(components.transition_matrix, [[0.8, 0.2], [0.1, 0.9]])
    np.testing.assert_allclose(components.emission_coefficients[:, 0], [-2.0, 2.0])
    assert tuple(parameters) == model.parameter_names


def test_three_state_components_use_the_same_canonical_contract() -> None:
    model = BernoulliGLMHMM(choice_lags=0, n_states=3)

    parameters = model.parameters_from_components(
        initial_probabilities=[0.2, 0.3, 0.5],
        transition_matrix=[
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ],
        emissions={"intercept": [2.0, -2.0, 0.0]},
    )
    components = model.parameter_components(parameters)

    np.testing.assert_allclose(components.emission_coefficients[:, 0], [-2.0, 0.0, 2.0])
    np.testing.assert_allclose(components.initial_probabilities, [0.3, 0.5, 0.2])
    assert components.transition_matrix.shape == (3, 3)


def test_simulation_keeps_latent_truth_separate_and_reproducible() -> None:
    model = clear_model()
    parameters = clear_parameters(model)

    first = model.simulate_with_states(design(), parameters, seed=22)
    second = model.simulate_with_states(design(), parameters, seed=22)

    assert "latent_state" not in first.study.columns
    assert first.n_states == 2
    assert set(first.study["choice"].tolist()) <= {0, 1}
    np.testing.assert_array_equal(first.study["choice"], second.study["choice"])
    np.testing.assert_array_equal(first.states, second.states)
    with pytest.raises(ValueError, match="read-only"):
        first.states[0] = 1


def test_filtered_state_probabilities_reset_at_session_boundaries() -> None:
    model = clear_model()
    parameters = clear_parameters(model)
    simulated = model.simulate(design(), parameters, seed=33)
    fit = fixed_fit(model, parameters, len(simulated))

    probabilities = model.state_probabilities(simulated, fit)
    components = model.parameter_components(parameters)

    assert probabilities.predictive.shape == (len(simulated), 2)
    np.testing.assert_allclose(probabilities.predictive.sum(axis=1), 1.0)
    for session_rows in _ordered_session_indices(simulated):
        np.testing.assert_allclose(
            probabilities.predictive[session_rows[0]],
            components.initial_probabilities,
        )


def test_pointwise_filtered_score_matches_forward_likelihood() -> None:
    model = clear_model()
    parameters = clear_parameters(model)
    simulated = model.simulate(design(n_sessions=2, trials_per_session=30), parameters, seed=44)
    fit = fixed_fit(model, parameters, len(simulated))
    outcomes = model._outcomes(simulated)
    features = model._base_feature_matrix(simulated, outcomes)
    vector = np.asarray([parameters[name] for name in model.parameter_names])

    objective, _ = model._objective_gradient(
        vector,
        features,
        outcomes,
        _ordered_session_indices(simulated),
    )
    scores = model.pointwise_log_prob(simulated, fit)

    assert -float(np.sum(scores)) == pytest.approx(objective, abs=1e-9)


def test_analytic_likelihood_gradient_matches_finite_differences() -> None:
    study = Study(
        {
            "subject": ["a"] * 8,
            "session": ["s1"] * 4 + ["s2"] * 4,
            "trial": [0, 1, 2, 3] * 2,
            "session_order": [0] * 4 + [1] * 4,
            "stimulus": [-1.0, 0.2, 1.1, -0.3, 0.5, -0.7, 1.4, 0.1],
            "choice": [0, 1, 1, 0, 1, 0, 1, 1],
        }
    )
    model = BernoulliGLMHMM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=0.2,
        n_restarts=1,
        stickiness=1.7,
    )
    parameters = model.parameters_from_components(
        initial_probabilities=[0.6, 0.4],
        transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
        emissions={
            "intercept": [-0.8, 0.9],
            "stimulus": [0.4, 1.2],
            "choice_lag_1": [-0.2, 0.5],
        },
    )
    vector = np.asarray([parameters[name] for name in model.parameter_names])
    outcomes = model._outcomes(study)
    features = model._base_feature_matrix(study, outcomes)
    sessions = _ordered_session_indices(study)

    _, analytic = model._objective_gradient(vector, features, outcomes, sessions)
    numeric = np.empty_like(analytic)
    for index in range(len(vector)):
        step = 1e-6 * (1.0 + abs(vector[index]))
        positive = vector.copy()
        negative = vector.copy()
        positive[index] += step
        negative[index] -= step
        positive_value, _ = model._objective_gradient(positive, features, outcomes, sessions)
        negative_value, _ = model._objective_gradient(negative, features, outcomes, sessions)
        numeric[index] = (positive_value - negative_value) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-6)


def test_sticky_prior_adds_declared_self_transition_pseudocounts() -> None:
    study = design(n_sessions=2, trials_per_session=10)
    study = Study(
        {
            **{name: study[name] for name in study.columns},
            "choice": [0, 1] * 10,
        }
    )
    plain = BernoulliGLMHMM(choice_lags=0, n_restarts=1)
    sticky = BernoulliGLMHMM(choice_lags=0, n_restarts=1, stickiness=2.5)
    parameters = plain.parameters_from_components(
        initial_probabilities=[0.6, 0.4],
        transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
        emissions={"intercept": [-1.0, 1.0]},
    )
    plain_vector = np.asarray([parameters[name] for name in plain.parameter_names])
    sticky_vector = np.asarray([parameters[name] for name in sticky.parameter_names])
    outcomes = plain._outcomes(study)
    features = plain._base_feature_matrix(study, outcomes)
    sessions = _ordered_session_indices(study)

    plain_loss, _ = plain._objective_gradient(plain_vector, features, outcomes, sessions)
    sticky_loss, _ = sticky._objective_gradient(sticky_vector, features, outcomes, sessions)

    expected = -sticky.stickiness * np.log([0.8, 0.7]).sum()
    assert sticky_loss - plain_loss == pytest.approx(expected)
    assert "stickiness=2.5" in sticky.signature


def test_fit_recovers_clear_states_and_retains_restart_diagnostics() -> None:
    model = clear_model()
    simulation = model.simulate_with_states(design(), clear_parameters(model), seed=2)

    fit = model.fit(simulation.study)
    components = model.parameter_components(fit)
    recovery = model.state_recovery(simulation, fit)

    assert fit.diagnostics.converged
    assert fit.restart_objectives.shape == (model.n_restarts,)
    assert fit.restart_converged.shape == (model.n_restarts,)
    assert fit.canonical_permutation in ((0, 1), (1, 0))
    assert fit.state_occupancy.sum() == pytest.approx(1.0)
    assert fit.label_ambiguous is False
    assert fit.low_occupancy is False
    assert components.emission_coefficients[0, 0] < 0 < components.emission_coefficients[1, 0]
    assert np.all(np.diag(components.transition_matrix) > 0.85)
    assert recovery.decoded_accuracy > 0.85
    assert recovery.posterior_accuracy > 0.8
    assert recovery.ambiguous is False


def test_state_recovery_rejects_a_simulation_with_the_wrong_state_count() -> None:
    model = clear_model(n_restarts=1)
    parameters = clear_parameters(model)
    simulation = model.simulate_with_states(
        design(n_sessions=1, trials_per_session=10),
        parameters,
        seed=3,
    )
    fit = fixed_fit(model, parameters, len(simulation.study))
    three_state_model = BernoulliGLMHMM(choice_lags=0, n_states=3, n_restarts=1)

    with pytest.raises(ValueError, match="same number of states"):
        three_state_model.state_recovery(simulation, fit)


def test_generic_parameter_recovery_handles_canonical_hmm_labels() -> None:
    model = clear_model(n_restarts=2)
    parameters = clear_parameters(model)

    report = run_parameter_recovery(
        model,
        design(),
        [parameters],
        seed=81,
    )
    estimate_map = dict(zip(model.parameter_names, report.estimates[0], strict=True))
    recovered = model.parameter_components(estimate_map)

    assert report.convergence_rate == 1.0
    assert recovered.emission_coefficients[0, 0] < -1.0
    assert recovered.emission_coefficients[1, 0] > 1.0
    assert np.all(np.diag(recovered.transition_matrix) > 0.8)


def test_glm_hmm_supports_prospective_model_evaluation() -> None:
    model = clear_model(n_restarts=2)
    simulated = model.simulate(design(), clear_parameters(model), seed=55)
    splits = forward_session_splits(simulated, min_train_sessions=3)

    evaluations = evaluate_splits(model, simulated, splits)

    assert len(evaluations) == 1
    assert evaluations[0].prediction.probability.shape == (100,)
    assert np.all(np.isfinite(evaluations[0].pointwise_log_probability))
    assert isinstance(model, BehaviourModel)


def test_switching_truth_beats_static_and_smooth_glms_prospectively() -> None:
    switching = clear_model(n_restarts=2)
    study = switching.simulate(
        design(n_sessions=6),
        clear_parameters(switching),
        seed=2026,
    )
    splits = forward_session_splits(study, min_train_sessions=5)
    static = BernoulliHistoryGLM(choice_lags=0, l2=0.01)
    smooth = SmoothBernoulliHistoryGLM(
        choice_lags=0,
        knots=tuple(range(6)),
        smoothness=10.0,
        l2=0.01,
    )

    static_loss = evaluate_splits(static, study, splits)[0].mean_log_loss
    smooth_loss = evaluate_splits(smooth, study, splits)[0].mean_log_loss
    switching_loss = evaluate_splits(switching, study, splits)[0].mean_log_loss

    assert switching_loss < static_loss - 0.1
    assert switching_loss < smooth_loss - 0.1


def test_smoothed_state_prediction_is_not_silently_substituted() -> None:
    model = clear_model(n_restarts=1)
    parameters = clear_parameters(model)
    simulated = model.simulate(design(n_sessions=2, trials_per_session=20), parameters, seed=66)
    fit = fixed_fit(model, parameters, len(simulated))

    with pytest.raises(UnsupportedPredictionMode, match="filtered prediction"):
        model.predict(simulated, fit, mode=PredictionMode.SMOOTHED)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("n_states", 1, "n_states"),
        ("n_restarts", 0, "n_restarts"),
        ("random_seed", -1, "random_seed"),
        ("label_by", "missing", "label_by"),
        ("label_tolerance", -1.0, "label_tolerance"),
        ("state_occupancy_warning", 1.0, "state_occupancy_warning"),
        ("probability_warning_threshold", 0.5, "probability_warning_threshold"),
        ("stickiness", -1.0, "stickiness"),
    ],
)
def test_glm_hmm_configuration_is_validated(
    argument: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BernoulliGLMHMM(**{argument: value})
