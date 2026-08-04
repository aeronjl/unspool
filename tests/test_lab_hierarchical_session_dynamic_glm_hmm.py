import numpy as np
import pytest

from behavio import (
    LabHierarchicalSessionDynamicBernoulliGLMHMM,
    Study,
    hierarchical,
)
from behavio.evaluate.splits import leave_one_lab_out_splits
from behavio.models import (
    LabHierarchicalSessionDynamicGLMHMMFitResult,
    LabHierarchicalSessionDynamicGLMHMMSimulation,
    ModelDataError,
    UnseenLabDynamicPrediction,
    UnseenSubjectInLabDynamicPrediction,
)
from behavio.models.lab_hierarchical_session_dynamic_glm_hmm import _lab_structure


def design(
    *,
    labs: tuple[str, ...] = ("north", "south"),
    subjects_per_lab: int = 2,
    sessions: int = 3,
    trials: int = 35,
) -> Study:
    subjects = tuple(f"{lab}-{index}" for lab in labs for index in range(subjects_per_lab))
    study = Study.factorial(trials=trials, subjects=subjects, sessions=sessions)
    values = np.concatenate([np.repeat(lab, subjects_per_lab * sessions * trials) for lab in labs])
    columns = {name: study[name] for name in study.columns}
    columns["lab"] = values
    return Study(columns)


def model(**changes: object) -> LabHierarchicalSessionDynamicBernoulliGLMHMM:
    arguments: dict[str, object] = {
        "choice_lags": 0,
        "n_restarts": 1,
        "max_iterations": 180,
        "dynamic_max_iterations": 20,
        "dynamic_tolerance": 1e-5,
        "population_emission_step_scale": 0.15,
        "lab_emission_scale": 0.25,
        "lab_emission_step_scale": 0.1,
        "subject_emission_scale": 0.2,
        "emission_step_scale": 0.1,
        "transition_concentration": 40.0,
        "random_seed": 9,
    }
    arguments.update(changes)
    return LabHierarchicalSessionDynamicBernoulliGLMHMM(**arguments)


def parameters(
    estimator: LabHierarchicalSessionDynamicBernoulliGLMHMM,
) -> dict[str, float]:
    return dict(
        estimator.parameters_from_components(
            initial_probabilities=(0.5, 0.5),
            transition_matrix=((0.96, 0.04), (0.04, 0.96)),
            emissions={"intercept": (-2.5, 2.5)},
        )
    )


@pytest.mark.parametrize(
    "name",
    ("lab_emission_scale", "lab_emission_step_scale"),
)
def test_lab_scales_must_be_positive(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        model(**{name: 0.0})


def test_simulation_retains_every_nested_path_and_reuses_one_lab_path_per_order() -> None:
    estimator = model()
    first = estimator.simulate_with_trajectories(design(trials=12), parameters(estimator), seed=31)
    second = estimator.simulate_with_trajectories(design(trials=12), parameters(estimator), seed=31)

    assert isinstance(first, LabHierarchicalSessionDynamicGLMHMMSimulation)
    assert estimator.is_lab_hierarchical
    assert first.labs == ("north", "south")
    assert first.subject_labs == ("north", "north", "south", "south")
    assert first.population_emission_coefficients.shape == (3, 2, 1)
    assert first.lab_deviation_coefficients.shape == (6, 2, 1)
    assert first.lab_emission_coefficients.shape == (6, 2, 1)
    assert first.session_emission_coefficients.shape == (12, 2, 1)
    assert first.session_transition_matrices.shape == (12, 2, 2)
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_allclose(first.lab_deviation_coefficients, second.lab_deviation_coefficients)
    with pytest.raises(ValueError, match="read-only"):
        first.lab_deviation_coefficients[0, 0, 0] = 0.0


def test_nested_emission_gradient_matches_finite_differences() -> None:
    study = design(sessions=2, trials=3)
    columns = {name: study[name] for name in study.columns}
    columns["choice"] = np.tile([0, 1, 1], 8)
    study = Study(columns)
    estimator = model(l2=0.3)
    structure = _lab_structure(
        study,
        lab_column="lab",
        require_multiple_labs=True,
        require_replicated_subjects=True,
    )
    generator = np.random.default_rng(13)
    population = generator.normal(0.0, 0.3, (2, 2, 1))
    lab_deviations = generator.normal(0.0, 0.3, (4, 2, 1))
    emissions = generator.normal(0.0, 0.3, (8, 2, 1))
    vector = estimator._pack_lab_coordinate(population, lab_deviations, emissions)
    state_probabilities = generator.dirichlet((2.0, 2.0), size=len(study))
    arguments = (
        estimator.design_matrix(study),
        estimator.outcomes(study),
        state_probabilities,
        structure,
    )

    _, analytic = estimator._lab_emission_m_step_objective(vector, *arguments)
    numeric = np.empty_like(analytic)
    for index in range(len(vector)):
        step = 1e-6 * (1.0 + abs(vector[index]))
        positive = vector.copy()
        negative = vector.copy()
        positive[index] += step
        negative[index] -= step
        positive_value, _ = estimator._lab_emission_m_step_objective(positive, *arguments)
        negative_value, _ = estimator._lab_emission_m_step_objective(negative, *arguments)
        numeric[index] = (positive_value - negative_value) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, atol=3e-6, rtol=3e-6)


def test_fit_retains_nested_uncertainty_scores_and_truth_aligned_recovery() -> None:
    estimator = model()
    simulation = estimator.simulate_with_trajectories(
        design(trials=45), parameters(estimator), seed=7
    )

    fit = estimator.fit(simulation.study)
    scores = estimator.pointwise_log_prob(simulation.study, fit)
    recovery = estimator.trajectory_recovery(simulation, fit)

    assert isinstance(fit, LabHierarchicalSessionDynamicGLMHMMFitResult)
    assert fit.partial_converged
    assert fit.full_converged
    assert fit.diagnostics.converged
    assert fit.grouping == "lab"
    assert fit.groups == ("north", "south")
    assert fit.population_emission_coefficients.shape == (3, 2, 1)
    assert fit.lab_deviation_coefficients.shape == (6, 2, 1)
    assert fit.lab_emission_coefficients.shape == (6, 2, 1)
    assert fit.subject_deviations.shape == (12, 2, 1)
    assert fit.population_label_crossings.shape == (2, 1)
    assert fit.lab_label_crossings.shape == (4, 1)
    assert fit.subject_label_crossings.shape == (8, 1)
    assert fit.path_covariance_positive_definite
    assert fit.population_emission_standard_errors.shape == (3, 2, 1)
    assert fit.lab_deviation_standard_errors.shape == (6, 2, 1)
    assert fit.lab_emission_standard_errors.shape == (6, 2, 1)
    assert fit.subject_deviation_standard_errors.shape == (12, 2, 1)
    assert fit.joint_emission_covariance.shape == (42, 42)
    assert fit.lab_emission_intervals().shape == (6, 2, 1, 2)
    assert np.all(np.isfinite(scores))
    assert recovery.alignment.decoded_accuracy > 0.65
    assert recovery.lab_deviation_rmse_by_lab.shape == (2,)
    assert recovery.subject_emission_rmse_by_subject.shape == (4,)


def test_prediction_distinguishes_seen_subject_seen_lab_new_subject_and_new_lab() -> None:
    estimator = model(dynamic_max_iterations=15)
    simulation = estimator.simulate(design(trials=30), parameters(estimator), seed=17)
    fit = estimator.fit(simulation)
    seen_subject = prediction_study(subjects=("north-0",), labs=("north",), order=3)
    new_subject = prediction_study(subjects=("north-new",), labs=("north",), order=3)
    new_lab = prediction_study(subjects=("west-0", "west-1"), labs=("west", "west"), order=3)

    for target in (seen_subject, new_subject, new_lab):
        prediction = estimator.predict(target, fit)
        assert prediction.probability.shape == (len(target),)
        np.testing.assert_allclose(
            estimator.transition_probabilities(target, fit)[0],
            fit.global_transition_matrix,
        )
    assert fit.seen_subject_future_policy.startswith("population-plus-lab")
    assert fit.unseen_subject_seen_lab_policy.startswith("population-plus-lab")
    assert fit.unseen_lab_policy.startswith("population-path-zero-lab")

    subject_prediction = estimator.predict_new_subjects(new_subject, fit, n_draws=16, seed=3)
    lab_prediction = estimator.predict_new_labs(new_lab, fit, n_draws=16, seed=5)
    assert isinstance(subject_prediction, UnseenSubjectInLabDynamicPrediction)
    assert subject_prediction.subjects == ("north-new",)
    assert subject_prediction.draw_session_emission_coefficients.shape == (16, 1, 2, 1)
    assert np.all(subject_prediction.subject_effective_draws >= 1.0)
    assert isinstance(lab_prediction, UnseenLabDynamicPrediction)
    assert lab_prediction.labs == ("west",)
    assert lab_prediction.draw_lab_deviation_coefficients.shape == (16, 1, 2, 1)
    assert lab_prediction.lab_joint_log_probability.shape == (1,)
    assert np.all(lab_prediction.lab_effective_draws >= 1.0)

    with pytest.raises(ValueError, match="predict_new_labs requires entirely unseen"):
        estimator.predict_new_labs(new_subject, fit, n_draws=4, seed=1)
    with pytest.raises(ValueError, match="use predict_new_labs"):
        estimator.predict_new_subjects(new_lab, fit, n_draws=4, seed=1)


def test_nested_hyperparameter_estimation_reports_all_scales_and_instability() -> None:
    truth = model(dynamic_max_iterations=5, dynamic_tolerance=1e-4)
    simulation = truth.simulate(design(sessions=2, trials=12), parameters(truth), seed=23)
    estimator = model(
        max_iterations=100,
        dynamic_max_iterations=5,
        dynamic_tolerance=1e-4,
        estimate_hyperparameters=True,
        hyperparameter_max_iterations=1,
        hyperparameter_tolerance=10.0,
    )

    fit = estimator.fit(simulation)

    assert fit.hyperparameters_estimated
    assert fit.hyperparameter_estimation_converged
    assert fit.hyperparameter_names == (
        "population_emission_step_scale",
        "lab_emission_scale",
        "lab_emission_step_scale",
        "subject_emission_scale",
        "emission_step_scale",
        "transition_concentration",
    )
    assert fit.hyperparameter_estimates.shape == (6,)
    assert fit.hyperparameter_covariance.shape == (6, 6)
    assert fit.gaussian_scale_em_rate_matrix.shape == (5, 5)
    assert np.all(np.isfinite(fit.hyperparameter_estimates))
    assert np.isfinite(fit.hyperparameter_standard_errors[-1])
    assert fit.hyperparameters_at_boundary.shape == (6,)
    if np.all(np.isfinite(fit.hyperparameter_standard_errors[:5])):
        assert np.all(fit.hyperparameter_standard_errors[:5] > 0)
    else:
        assert np.all(np.isnan(fit.hyperparameter_standard_errors[:5]))
        assert "scale-unavailable" in fit.hyperparameter_uncertainty_policy


def test_lab_hierarchy_refuses_crossed_or_unreplicated_subjects_and_generic_wrapping() -> None:
    estimator = model()
    unreplicated = design(subjects_per_lab=1, trials=5)
    with pytest.raises(ModelDataError, match="two independent subjects"):
        estimator.fit(unreplicated)

    crossed = design(trials=5)
    columns = {name: crossed[name] for name in crossed.columns}
    mask = columns["subject"] == "north-0"
    labs = np.asarray(columns["lab"]).copy()
    labs[np.flatnonzero(mask)[-5:]] = "south"
    columns["lab"] = labs
    with pytest.raises(ModelDataError, match="exactly one laboratory"):
        estimator.fit(Study(columns))
    with pytest.raises(TypeError, match="already contains nested"):
        hierarchical(estimator)


def test_leave_lab_out_fit_predicts_the_entire_unseen_lab_without_subject_leakage() -> None:
    estimator = model(dynamic_max_iterations=8, dynamic_tolerance=1e-4)
    simulation = estimator.simulate(
        design(labs=("north", "south", "west"), sessions=2, trials=12),
        parameters(estimator),
        seed=29,
    )
    split = leave_one_lab_out_splits(simulation)[0]
    train = simulation.take(split.train_indices)
    test = simulation.take(split.test_indices)

    fit = estimator.fit(train)
    prediction = estimator.predict(test, fit)
    integrated = estimator.predict_new_labs(test, fit, n_draws=8, seed=4)

    assert split.test_groups == ("north",)
    assert set(fit.labs) == {"south", "west"}
    assert not set(fit.subjects) & set(test.subjects)
    assert prediction.probability.shape == (len(test),)
    assert integrated.labs == ("north",)
    assert integrated.lab_joint_log_probability.shape == (1,)


def prediction_study(*, subjects: tuple[str, ...], labs: tuple[str, ...], order: int) -> Study:
    study = Study.factorial(
        trials=5,
        subjects=subjects,
        sessions=("future",),
        columns={"choice": [0, 1, 0, 1, 0] * len(subjects)},
    )
    columns = {name: study[name] for name in study.columns}
    columns["lab"] = np.repeat(labs, 5)
    columns["session_order"] = np.full(len(study), order)
    return Study(columns)
