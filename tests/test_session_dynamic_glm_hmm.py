import numpy as np
import pytest
from scipy.special import expit

from behavio import SessionDynamicBernoulliGLMHMM, Study
from behavio.models import ModelDataError, SessionDynamicGLMHMMSimulation
from behavio.models._kernels.bernoulli import ordered_session_indices


def design(*, sessions: int = 5, trials: int = 60, subject: str = "animal-a") -> Study:
    return Study.factorial(trials=trials, subjects=subject, sessions=sessions)


def model(**changes: object) -> SessionDynamicBernoulliGLMHMM:
    arguments: dict[str, object] = {
        "choice_lags": 0,
        "n_restarts": 1,
        "max_iterations": 300,
        "dynamic_max_iterations": 40,
        "dynamic_tolerance": 1e-7,
        "emission_step_scale": 0.2,
        "transition_concentration": 40.0,
        "random_seed": 9,
    }
    arguments.update(changes)
    return SessionDynamicBernoulliGLMHMM(**arguments)


def parameters(estimator: SessionDynamicBernoulliGLMHMM) -> dict[str, float]:
    return dict(
        estimator.parameters_from_components(
            initial_probabilities=[0.5, 0.5],
            transition_matrix=[[0.96, 0.04], [0.04, 0.96]],
            emissions={"intercept": [-2.5, 2.5]},
        )
    )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("emission_step_scale", 0.0, "emission_step_scale"),
        ("transition_concentration", 0.0, "transition_concentration"),
        ("dynamic_max_iterations", 0, "dynamic_max_iterations"),
        ("dynamic_tolerance", 0.0, "dynamic_tolerance"),
        ("stickiness", 1.0, "Dirichlet prior"),
        ("transition_predictors", ("arousal",), "different mechanisms"),
    ],
)
def test_configuration_refuses_undeclared_hybrids(
    argument: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        model(**{argument: value})


def test_simulation_retains_reproducible_session_trajectory_truth() -> None:
    estimator = model()
    first = estimator.simulate_with_trajectories(design(), parameters(estimator), seed=41)
    second = estimator.simulate_with_trajectories(design(), parameters(estimator), seed=41)

    assert isinstance(first, SessionDynamicGLMHMMSimulation)
    assert estimator.is_session_dynamic
    assert not estimator.is_dynamic
    assert "latent_state" not in first.study.columns
    assert first.emission_coefficients.shape == (5, 2, 1)
    assert first.transition_matrices.shape == (5, 2, 2)
    assert not np.array_equal(first.emission_coefficients[0], first.emission_coefficients[-1])
    np.testing.assert_allclose(first.transition_matrices.sum(axis=2), 1.0)
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_allclose(first.emission_coefficients, second.emission_coefficients)
    np.testing.assert_allclose(first.transition_matrices, second.transition_matrices)
    with pytest.raises(ValueError, match="read-only"):
        first.emission_coefficients[0, 0, 0] = 0.0


def test_transition_m_step_is_the_dirichlet_pseudocount_update() -> None:
    estimator = model(transition_concentration=5.0)
    sessions = ((0, 1, 2), (3, 4))
    expectations = np.zeros((5, 2, 2), dtype=np.float64)
    expectations[1] = [[0.7, 0.1], [0.05, 0.15]]
    expectations[2] = [[0.2, 0.1], [0.1, 0.7]]
    expectations[4] = [[0.4, 0.2], [0.3, 0.1]]
    global_transition = np.asarray([[0.8, 0.2], [0.25, 0.75]])

    observed = estimator._transition_m_step(
        expectations,
        sessions,
        global_transition,
    )
    counts = expectations[np.asarray(sessions[0])].sum(axis=0)
    expected = counts + 5.0 * global_transition
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(observed[0], expected)
    np.testing.assert_allclose(observed.sum(axis=2), 1.0)


def test_partial_stage_fits_one_shared_transition_matrix() -> None:
    estimator = model()
    sessions = ((0, 1, 2), (3, 4))
    expectations = np.zeros((5, 2, 2), dtype=np.float64)
    expectations[1] = [[0.7, 0.1], [0.05, 0.15]]
    expectations[2] = [[0.2, 0.1], [0.1, 0.7]]
    expectations[4] = [[0.4, 0.2], [0.3, 0.1]]
    fallback = np.asarray([[0.8, 0.2], [0.25, 0.75]])

    observed = estimator._stationary_transition_m_step(expectations, sessions, fallback)
    counts = expectations.sum(axis=0)

    np.testing.assert_allclose(observed, counts / counts.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(observed.sum(axis=1), 1.0)


def test_random_walk_emission_gradient_matches_finite_differences() -> None:
    estimator = model(l2=0.3)
    study = Study(
        {
            "subject": ["a"] * 8,
            "session": ["s0"] * 4 + ["s1"] * 4,
            "trial": [0, 1, 2, 3] * 2,
            "session_order": [0] * 4 + [1] * 4,
            "stimulus": [-1.0, 0.3, 0.8, -0.2, 0.4, -0.7, 1.2, 0.1],
            "choice": [0, 1, 1, 0, 1, 0, 1, 1],
        }
    )
    estimator = model(predictors=("stimulus",), l2=0.3)
    features = estimator.design_matrix(study)
    outcomes = estimator.outcomes(study)
    sessions = ordered_session_indices(study)
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.5, 0.5],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
        ]
    )
    vector = np.asarray([-0.8, 0.4, 0.9, 1.1, -0.6, 0.5, 1.0, 0.8])

    _, analytic = estimator._emission_m_step_objective(
        vector,
        features,
        outcomes,
        sessions,
        probabilities,
    )
    numeric = np.empty_like(analytic)
    for index in range(len(vector)):
        step = 1e-6 * (1.0 + abs(vector[index]))
        positive = vector.copy()
        negative = vector.copy()
        positive[index] += step
        negative[index] -= step
        positive_value, _ = estimator._emission_m_step_objective(
            positive,
            features,
            outcomes,
            sessions,
            probabilities,
        )
        negative_value, _ = estimator._emission_m_step_objective(
            negative,
            features,
            outcomes,
            sessions,
            probabilities,
        )
        numeric[index] = (positive_value - negative_value) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-6)


def test_fit_retains_paths_and_scores_the_same_filtered_likelihood() -> None:
    estimator = model()
    simulation = estimator.simulate_with_trajectories(
        design(trials=80),
        parameters(estimator),
        seed=7,
    )

    fit = estimator.fit(simulation.study)
    scores = estimator.pointwise_log_prob(simulation.study, fit)
    components = estimator.parameter_components(fit)
    posterior = estimator._dynamic_posterior(
        estimator.design_matrix(simulation.study),
        estimator.outcomes(simulation.study),
        ordered_session_indices(simulation.study),
        components.initial_probabilities,
        fit.session_emission_coefficients,
        fit.session_transition_matrices,
    )
    recovery = estimator.trajectory_recovery(simulation, fit)

    assert fit.diagnostics.converged
    assert fit.session_emission_coefficients.shape == (5, 2, 1)
    assert fit.session_transition_matrices.shape == (5, 2, 2)
    assert fit.partial_converged
    assert fit.partial_emission_optimizer_converged
    assert fit.full_converged
    assert len(fit.partial_objective_history) >= 2
    assert np.all(np.isfinite(fit.partial_objective_history))
    assert "partial stage converged; full stage converged" in fit.diagnostics.message
    assert fit.state_occupancy.sum() == pytest.approx(1.0)
    assert fit.uncertainty_policy == "not-estimated"
    assert np.all(np.isnan(fit.standard_errors))
    assert fit.audit().latent_states is not None
    assert fit.audit().restarts is not None
    assert -float(np.sum(scores)) == pytest.approx(-posterior.log_likelihood, abs=1e-9)
    assert recovery.alignment.decoded_accuracy > 0.75
    assert np.isfinite(recovery.emission_rmse)
    assert np.isfinite(recovery.transition_rmse)


def test_future_sessions_use_last_emissions_and_the_global_transition_center() -> None:
    estimator = model(dynamic_max_iterations=30)
    full = estimator.simulate(
        design(sessions=4, trials=50),
        parameters(estimator),
        seed=12,
    )
    training_rows = np.flatnonzero(full["session_order"] < 3)
    future_rows = np.flatnonzero(full["session_order"] == 3)
    training = full.take(training_rows)
    future = full.take(future_rows)
    fit = estimator.fit(training)

    prediction = estimator.predict(future, fit)
    fitted_base = estimator.parameter_components(fit)
    first_features = estimator.design_matrix(future)[0]
    expected_first = float(
        fitted_base.initial_probabilities
        @ expit(first_features @ fit.session_emission_coefficients[-1].T)
    )
    transitions = estimator.transition_probabilities(future, fit)

    assert fit.future_session_policy == "carry-last-emissions/use-global-transitions"
    assert prediction.probability[0] == pytest.approx(expected_first)
    np.testing.assert_allclose(transitions[0], fit.global_transition_matrix)


def test_prediction_refuses_unseen_past_sessions_and_unseen_subjects() -> None:
    estimator = model(dynamic_max_iterations=20)
    simulation = estimator.simulate(
        design(sessions=2, trials=30),
        parameters(estimator),
        seed=2,
    )
    fit = estimator.fit(simulation)
    past = Study.factorial(
        trials=5,
        subjects="animal-a",
        sessions=("unseen-past",),
        columns={"choice": [0, 1, 0, 1, 0]},
    )
    other_subject = Study.factorial(
        trials=5,
        subjects="animal-b",
        sessions=("future",),
        columns={"choice": [0, 1, 0, 1, 0]},
    )
    other_subject = Study(
        {
            **{name: other_subject[name] for name in other_subject.columns},
            "session_order": [3] * 5,
        }
    )

    with pytest.raises(ModelDataError, match="not fitted and is not prospectively later"):
        estimator.predict(past, fit)
    with pytest.raises(ModelDataError, match="unseen subject"):
        estimator.predict(other_subject, fit)


def test_multiple_subjects_require_an_explicit_population_model() -> None:
    estimator = model()
    multiple = Study.factorial(
        trials=10,
        subjects=("a", "b"),
        sessions=2,
        columns={"choice": [0, 1] * 20},
    )

    with pytest.raises(ModelDataError, match="one subject at a time"):
        estimator.fit(multiple)
