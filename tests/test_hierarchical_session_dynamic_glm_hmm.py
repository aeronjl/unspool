import numpy as np
import pytest
from scipy.special import expit

from behavio import HierarchicalSessionDynamicBernoulliGLMHMM, Study, hierarchical
from behavio.models import (
    HierarchicalSessionDynamicGLMHMMFitResult,
    HierarchicalSessionDynamicGLMHMMSimulation,
    ModelDataError,
    UnseenSubjectDynamicPrediction,
)
from behavio.models.hierarchical_session_dynamic_glm_hmm import _population_structure


def design(
    *,
    subjects: tuple[str, ...] = ("animal-a", "animal-b", "animal-c"),
    sessions: int = 4,
    trials: int = 50,
) -> Study:
    return Study.factorial(trials=trials, subjects=subjects, sessions=sessions)


def model(**changes: object) -> HierarchicalSessionDynamicBernoulliGLMHMM:
    arguments: dict[str, object] = {
        "choice_lags": 0,
        "n_restarts": 1,
        "max_iterations": 250,
        "dynamic_max_iterations": 35,
        "dynamic_tolerance": 1e-6,
        "population_emission_step_scale": 0.15,
        "subject_emission_scale": 0.25,
        "emission_step_scale": 0.12,
        "transition_concentration": 50.0,
        "random_seed": 13,
    }
    arguments.update(changes)
    return HierarchicalSessionDynamicBernoulliGLMHMM(**arguments)


def parameters(estimator: HierarchicalSessionDynamicBernoulliGLMHMM) -> dict[str, float]:
    return dict(
        estimator.parameters_from_components(
            initial_probabilities=(0.5, 0.5),
            transition_matrix=((0.96, 0.04), (0.04, 0.96)),
            emissions={"intercept": (-2.5, 2.5)},
        )
    )


@pytest.mark.parametrize(
    "name",
    ("population_emission_step_scale", "subject_emission_scale"),
)
def test_population_scales_must_be_positive(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        model(**{name: 0.0})


def test_simulation_retains_population_and_subject_path_truth() -> None:
    estimator = model()
    first = estimator.simulate_with_trajectories(
        design(),
        parameters(estimator),
        seed=31,
    )
    second = estimator.simulate_with_trajectories(
        design(),
        parameters(estimator),
        seed=31,
    )

    assert isinstance(first, HierarchicalSessionDynamicGLMHMMSimulation)
    assert estimator.is_population_dynamic
    assert first.subjects == ("animal-a", "animal-b", "animal-c")
    assert first.population_emission_coefficients.shape == (4, 2, 1)
    assert first.session_emission_coefficients.shape == (12, 2, 1)
    assert first.session_transition_matrices.shape == (12, 2, 2)
    assert "latent_state" not in first.study.columns
    assert not np.array_equal(
        first.population_emission_coefficients[0],
        first.population_emission_coefficients[-1],
    )
    np.testing.assert_allclose(first.session_transition_matrices.sum(axis=2), 1.0)
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_allclose(
        first.population_emission_coefficients,
        second.population_emission_coefficients,
    )
    np.testing.assert_allclose(
        first.session_emission_coefficients,
        second.session_emission_coefficients,
    )
    with pytest.raises(ValueError, match="read-only"):
        first.population_emission_coefficients[0, 0, 0] = 0.0


def test_population_and_subject_path_gradient_matches_finite_differences() -> None:
    study = Study.factorial(
        trials=4,
        subjects=("a", "b"),
        sessions=2,
        columns={"choice": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]},
    )
    estimator = model(l2=0.3)
    structure = _population_structure(study, require_multiple_subjects=True)
    features = estimator.design_matrix(study)
    outcomes = estimator.outcomes(study)
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.5, 0.5],
        ]
        * 4
    )
    generator = np.random.default_rng(9)
    population = generator.normal(0.0, 0.4, (2, 2, 1))
    emissions = generator.normal(0.0, 0.4, (4, 2, 1))
    vector = estimator._pack_emission_coordinate(population, emissions)

    _, analytic = estimator._population_emission_m_step_objective(
        vector,
        features,
        outcomes,
        probabilities,
        structure,
    )
    numeric = np.empty_like(analytic)
    for index in range(len(vector)):
        step = 1e-6 * (1.0 + abs(vector[index]))
        positive = vector.copy()
        negative = vector.copy()
        positive[index] += step
        negative[index] -= step
        positive_value, _ = estimator._population_emission_m_step_objective(
            positive,
            features,
            outcomes,
            probabilities,
            structure,
        )
        negative_value, _ = estimator._population_emission_m_step_objective(
            negative,
            features,
            outcomes,
            probabilities,
            structure,
        )
        numeric[index] = (positive_value - negative_value) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-6)


def test_fit_retains_population_paths_scores_and_recovery() -> None:
    estimator = model()
    simulation = estimator.simulate_with_trajectories(
        design(trials=60),
        parameters(estimator),
        seed=7,
    )

    fit = estimator.fit(simulation.study)
    scores = estimator.pointwise_log_prob(simulation.study, fit)
    components = estimator.parameter_components(fit)
    posterior = estimator._dynamic_posterior(
        estimator.design_matrix(simulation.study),
        estimator.outcomes(simulation.study),
        _population_structure(
            simulation.study,
            require_multiple_subjects=True,
        ).sessions,
        components.initial_probabilities,
        fit.session_emission_coefficients,
        fit.session_transition_matrices,
    )
    recovery = estimator.trajectory_recovery(simulation, fit)

    assert isinstance(fit, HierarchicalSessionDynamicGLMHMMFitResult)
    assert fit.partial_converged
    assert fit.full_converged
    assert fit.diagnostics.converged
    assert fit.population_emission_coefficients.shape == (4, 2, 1)
    assert fit.session_emission_coefficients.shape == (12, 2, 1)
    assert fit.subject_deviations.shape == (12, 2, 1)
    assert fit.population_label_crossings.shape == (3, 1)
    assert fit.subject_label_crossings.shape == (9, 1)
    assert fit.grouping == "subject"
    assert fit.groups == fit.subjects
    assert fit.uncertainty_policy == "observed-laplace-conditional-on-canonical-path"
    assert fit.uncertainty_label_policy == "conditional-on-one-whole-path-canonical-mode"
    assert fit.path_covariance_positive_definite
    assert fit.population_emission_standard_errors.shape == (4, 2, 1)
    assert fit.session_emission_standard_errors.shape == (12, 2, 1)
    assert fit.subject_deviation_standard_errors.shape == (12, 2, 1)
    assert fit.population_emission_covariance.shape == (8, 8)
    assert fit.session_emission_covariance.shape == (24, 24)
    assert fit.joint_emission_covariance.shape == (32, 32)
    assert np.all(np.isfinite(fit.population_emission_standard_errors))
    assert np.all(np.isfinite(fit.subject_deviation_standard_errors))
    assert np.all(np.isfinite(fit.session_transition_standard_errors))
    assert fit.population_emission_intervals().shape == (4, 2, 1, 2)
    assert fit.subject_deviation_intervals().shape == (12, 2, 1, 2)
    assert not fit.hyperparameters_estimated
    assert np.all(np.isnan(fit.hyperparameter_standard_errors))
    assert np.all(np.isnan(fit.standard_errors))
    assert fit.audit().latent_states is not None
    assert fit.audit().restarts is not None
    assert -float(np.sum(scores)) == pytest.approx(-posterior.log_likelihood, abs=1e-9)
    assert recovery.alignment.decoded_accuracy > 0.75
    assert recovery.subject_emission_rmse_by_subject.shape == (3,)
    assert np.isfinite(recovery.population_emission_rmse)
    assert np.isfinite(recovery.subject_emission_rmse)
    assert np.isfinite(recovery.transition_rmse)


def test_seen_future_and_unseen_subject_predictions_use_declared_population_plugins() -> None:
    estimator = model(dynamic_max_iterations=25, dynamic_tolerance=1e-5)
    simulation = estimator.simulate(
        design(subjects=("animal-a", "animal-b"), sessions=3, trials=45),
        parameters(estimator),
        seed=17,
    )
    fit = estimator.fit(simulation)
    unseen = _prediction_study(subject="animal-new", session="future", order=3)
    seen = _prediction_study(subject="animal-a", session="future", order=3)

    unseen_prediction = estimator.predict(unseen, fit)
    seen_prediction = estimator.predict(seen, fit)
    base = estimator.parameter_components(fit)
    unseen_expected = float(
        base.initial_probabilities @ expit(fit.population_emission_coefficients[-1, :, 0])
    )
    last_block = max(
        block for block, subject in enumerate(fit.path_subjects) if subject == "animal-a"
    )
    last_order = int(fit.session_orders[last_block])
    population_position = list(fit.population_session_orders).index(last_order)
    deviation = (
        fit.session_emission_coefficients[last_block]
        - fit.population_emission_coefficients[population_position]
    )
    seen_emissions = fit.population_emission_coefficients[-1] + deviation
    seen_expected = float(base.initial_probabilities @ expit(seen_emissions[:, 0]))

    assert fit.unseen_subject_policy == "population-path-plugin/use-global-transitions"
    assert fit.seen_future_session_policy.startswith("population-path-plus-carried")
    assert not fit.subject_was_fitted("animal-new")
    assert fit.subject_was_fitted("animal-a")
    assert unseen_prediction.probability[0] == pytest.approx(unseen_expected)
    assert seen_prediction.probability[0] == pytest.approx(seen_expected)
    np.testing.assert_allclose(
        estimator.transition_probabilities(unseen, fit)[0],
        fit.global_transition_matrix,
    )
    np.testing.assert_allclose(
        estimator.transition_probabilities(seen, fit)[0],
        fit.global_transition_matrix,
    )


def test_unseen_subject_prediction_integrates_coherent_paths_and_joint_scores() -> None:
    estimator = model(dynamic_max_iterations=25, dynamic_tolerance=1e-5)
    simulation = estimator.simulate(
        design(subjects=("animal-a", "animal-b", "animal-c"), sessions=3, trials=35),
        parameters(estimator),
        seed=17,
    )
    fit = estimator.fit(simulation)
    unseen = Study.factorial(
        trials=8,
        subjects=("animal-new-a", "animal-new-b"),
        sessions=("future-a", "future-b"),
        columns={"choice": [0, 1] * 16},
    )
    columns = {name: unseen[name] for name in unseen.columns}
    columns["session_order"] = np.tile(np.repeat([3, 4], 8), 2)
    unseen = Study(columns)

    first = estimator.predict_new_subjects(unseen, fit, n_draws=32, seed=29)
    second = estimator.predict_new_subjects(unseen, fit, n_draws=32, seed=29)

    assert isinstance(first, UnseenSubjectDynamicPrediction)
    assert first.includes_population_path_uncertainty
    assert first.label_path_ambiguous == fit.label_path_ambiguous
    assert first.draw_probabilities.shape == (32, len(unseen))
    assert first.draw_session_emission_coefficients.shape == (32, 4, 2, 1)
    assert first.draw_session_transition_matrices.shape == (32, 4, 2, 2)
    np.testing.assert_allclose(first.probability, np.mean(first.draw_probabilities, axis=0))
    np.testing.assert_allclose(
        first.draw_session_transition_matrices.sum(axis=-1),
        1.0,
    )
    np.testing.assert_allclose(first.draw_probabilities, second.draw_probabilities)
    assert np.all(first.subject_effective_draws >= 1.0)
    assert np.all(first.subject_log_probability_mcse >= 0.0)

    with pytest.raises(ValueError, match="entirely unseen"):
        estimator.predict_new_subjects(
            _prediction_study(subject="animal-a", session="future", order=3),
            fit,
            n_draws=8,
            seed=3,
        )


def test_hierarchy_hyperparameter_estimation_keeps_unstable_scale_information_visible() -> None:
    truth = model(
        dynamic_max_iterations=15,
        population_emission_step_scale=0.2,
        subject_emission_scale=0.3,
        emission_step_scale=0.15,
        transition_concentration=20.0,
    )
    simulation = truth.simulate(
        design(subjects=("a", "b", "c"), sessions=3, trials=25),
        parameters(truth),
        seed=8,
    )
    estimator = model(
        max_iterations=100,
        dynamic_max_iterations=12,
        estimate_hyperparameters=True,
        hyperparameter_max_iterations=1,
        hyperparameter_tolerance=10.0,
        population_emission_step_scale=0.2,
        subject_emission_scale=0.3,
        emission_step_scale=0.15,
        transition_concentration=20.0,
    )

    fit = estimator.fit(simulation)

    assert fit.hyperparameters_estimated
    assert fit.hyperparameter_estimation_converged
    assert fit.hyperparameter_estimates.shape == (4,)
    assert np.all(np.isfinite(fit.hyperparameter_estimates))
    assert fit.hyperparameter_covariance.shape == (4, 4)
    assert fit.gaussian_scale_em_rate_matrix.shape == (3, 3)
    assert np.isfinite(fit.hyperparameter_standard_errors[-1])
    scale_errors = fit.hyperparameter_standard_errors[:3]
    if np.all(np.isfinite(scale_errors)):
        assert np.all(scale_errors > 0)
    else:
        assert np.all(np.isnan(scale_errors))
        assert fit.gaussian_scale_em_spectral_radius >= 1.0
        assert "scale-unavailable" in fit.hyperparameter_uncertainty_policy


def test_seen_subject_unfitted_past_session_is_refused() -> None:
    estimator = model(dynamic_max_iterations=20, dynamic_tolerance=1e-5)
    simulation = estimator.simulate(
        design(subjects=("animal-a", "animal-b"), sessions=3, trials=30),
        parameters(estimator),
        seed=21,
    )
    fit = estimator.fit(simulation)
    past = _prediction_study(subject="animal-a", session="unseen-past", order=1)

    with pytest.raises(ModelDataError, match="not later than that subject's last training"):
        estimator.predict(past, fit)


def test_hierarchy_requires_multiple_subjects_and_refuses_another_generic_hierarchy() -> None:
    estimator = model()
    one_subject = Study.factorial(
        trials=10,
        subjects="a",
        sessions=2,
        columns={"choice": [0, 1] * 10},
    )

    with pytest.raises(ModelDataError, match="at least two subjects"):
        estimator.fit(one_subject)
    with pytest.raises(TypeError, match="already contains a population path"):
        hierarchical(estimator)


def _prediction_study(*, subject: str, session: str, order: int) -> Study:
    return Study(
        {
            "subject": [subject] * 5,
            "session": [session] * 5,
            "trial": list(range(5)),
            "session_order": [order] * 5,
            "choice": [0, 1, 0, 1, 0],
        }
    )
