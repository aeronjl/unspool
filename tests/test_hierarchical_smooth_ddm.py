from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from behavio import (
    BehaviourModel,
    HierarchicalSmoothDriftDiffusionFitResult,
    HierarchicalSmoothWienerDriftDiffusion,
    ModelDataError,
    Study,
    UnseenSubjectPosteriorPrediction,
    cohort_forward_session_splits,
    evaluate_splits,
    leave_one_subject_out_splits,
    run_parameter_recovery,
)
from behavio.models.hierarchical_smooth_ddm import (
    _arrowhead_covariance,
    _louis_scale_information,
    _scale_standard_error,
    _scales_at_bounds,
)


def make_design(
    *,
    n_subjects: int = 4,
    n_sessions: int = 3,
    trials_per_session: int = 70,
    seed: int = 420,
) -> Study:
    generator = np.random.default_rng(seed)
    subjects = tuple(f"mouse-{index}" for index in range(n_subjects))
    n_rows = n_subjects * n_sessions * trials_per_session
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
            "trial": list(range(trials_per_session)) * n_subjects * n_sessions,
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(trials_per_session)
            ],
            "stimulus": generator.normal(size=n_rows),
        }
    )


def hierarchical_model(**changes: object) -> HierarchicalSmoothWienerDriftDiffusion:
    arguments = {
        "covariates": ("stimulus",),
        "knots": (0.0, 2.0),
        "varying_parameters": ("drift.stimulus", "boundary"),
        "subject_parameters": ("drift.stimulus", "boundary"),
        "smoothness": 8.0,
        "subject_scale": 0.18,
        "subject_smoothness": 8.0,
        "n_restarts": 2,
        "max_iterations": 300,
        "simulation_time_step": 0.001,
    }
    arguments.update(changes)
    return HierarchicalSmoothWienerDriftDiffusion(**arguments)  # type: ignore[arg-type]


def population_truth(model: HierarchicalSmoothWienerDriftDiffusion):
    return model.parameters_from_paths(
        {
            "drift.intercept": 0.1,
            "drift.stimulus": [0.6, 1.4],
            "boundary": [1.45, 1.1],
            "starting_bias": 0.48,
            "nondecision_time": 0.25,
        }
    )


def zero_deviations(
    model: HierarchicalSmoothWienerDriftDiffusion,
    design: Study,
) -> dict[str, dict[str, list[float]]]:
    return {
        subject: {
            parameter: [0.0] * len(model.knots) for parameter in model.subject_parameters or ()
        }
        for subject in design.subjects
    }


def test_hierarchical_smooth_ddm_has_a_stable_public_contract() -> None:
    model = hierarchical_model()

    assert isinstance(model, BehaviourModel)
    assert model.model_name == "hierarchical-smooth-wiener-drift-diffusion"
    assert model.subject_parameters == ("drift.stimulus", "boundary")
    assert model.parameter_names == (
        "drift.intercept",
        "drift.stimulus[session_order=0]",
        "drift.stimulus[session_order=2]",
        "boundary[session_order=0]",
        "boundary[session_order=2]",
        "starting_bias",
        "nondecision_time",
    )
    assert "subject_scale=0.18" in model.signature
    assert model.subject_parameter_scales is None
    assert "subject_parameter_scales=drift.stimulus:0.18,boundary:0.18" in model.signature
    assert "estimate_subject_scales=False" in model.signature
    assert "subject_smoothness=8.0" in model.signature


@pytest.mark.parametrize(
    "arguments",
    [
        {"subject_scale": 0.0},
        {"subject_parameter_scales": {"boundary": 0.2}},
        {
            "subject_parameter_scales": {
                "drift.stimulus": 0.2,
                "boundary": 0.0,
            }
        },
        {"estimate_subject_scales": 1},
        {"subject_scale_bounds": (0.2, 0.1)},
        {"estimate_subject_scales": True, "subject_scale": 0.01},
        {"scale_max_iterations": 0},
        {"scale_tolerance": 0.0},
        {"subject_scale_uncertainty": "posterior"},
        {"subject_scale_uncertainty": "supplemented"},
        {"subject_smoothness": 0.0},
        {"subject_parameters": ()},
        {"subject_parameters": ("nondecision_time",)},
        {"subject_parameters": ("boundary", "boundary")},
        {"shared_trajectory": True},
    ],
)
def test_hierarchical_smooth_ddm_configuration_is_validated(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        hierarchical_model(**arguments)


def test_simulation_retains_reproducible_subject_paths_outside_observed_data() -> None:
    model = hierarchical_model()
    design = make_design(n_subjects=3, trials_per_session=25)

    first = model.simulate_with_effects(design, population_truth(model), seed=8)
    second = model.simulate_with_effects(design, population_truth(model), seed=8)

    assert first.population_knot_values.shape == (5, 2)
    assert first.subject_deviation_knot_values.shape == (3, 2, 2)
    assert first.subject_knot_values.shape == (3, 5, 2)
    np.testing.assert_array_equal(first.study["choice"], second.study["choice"])
    np.testing.assert_array_equal(
        first.subject_deviation_knot_values,
        second.subject_deviation_knot_values,
    )
    assert "subject_deviation" not in first.study.columns
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.subject_knot_values.setflags(write=True)


def test_parameter_specific_scales_control_named_simulation_components() -> None:
    model = hierarchical_model(
        subject_parameter_scales={"drift.stimulus": 0.3, "boundary": 0.05},
        subject_smoothness=0.1,
    )
    design = make_design(n_subjects=40, n_sessions=1, trials_per_session=2)

    simulation = model.simulate_with_effects(design, population_truth(model), seed=81)
    empirical = np.std(simulation.subject_deviation_knot_values, axis=(0, 2))

    assert empirical[0] > 3.0 * empirical[1]


def test_subject_scales_are_estimated_by_parameter_with_retained_diagnostics() -> None:
    model = hierarchical_model(
        subject_parameter_scales={"drift.stimulus": 0.22, "boundary": 0.08},
        estimate_subject_scales=True,
        subject_scale_bounds=(0.04, 0.5),
        scale_max_iterations=12,
        scale_tolerance=0.03,
        subject_scale_uncertainty="supplemented",
        n_restarts=1,
    )
    design = make_design(n_subjects=8, trials_per_session=70)
    study = model.simulate(design, population_truth(model), seed=91)

    fit = model.fit(study)

    assert fit.subject_scales_estimated
    assert fit.scale_estimation_policy == "laplace-em"
    assert fit.subject_scale_uncertainty_policy == "supplemented-laplace-em"
    assert 1 <= fit.scale_estimation_iterations <= 12
    assert set(fit.subject_scale_map) == {"drift.stimulus", "boundary"}
    assert set(fit.subject_scale_standard_error_map or ()) == {
        "drift.stimulus",
        "boundary",
    }
    assert set(fit.subject_scale_at_boundary_map or ()) == {
        "drift.stimulus",
        "boundary",
    }
    assert set(fit.subject_scale_confidence_intervals_95 or ()) == {
        "drift.stimulus",
        "boundary",
    }
    assert fit.subject_scale_map["drift.stimulus"] > fit.subject_scale_map["boundary"]
    assert np.all(np.isfinite(fit.subject_parameter_scales))
    assert np.all(fit.subject_scale_standard_errors > 0)
    assert np.all(fit.subject_scale_local_standard_errors > 0)
    assert fit.subject_scale_covariance.shape == (2, 2)
    assert fit.subject_scale_em_rate_matrix.shape == (2, 2)
    assert fit.subject_scale_em_spectral_radius is not None
    assert fit.subject_scale_em_spectral_radius < 1.0
    assert set(fit.subject_scale_local_confidence_intervals_95 or ()) == {
        "drift.stimulus",
        "boundary",
    }
    assert all(
        0.04 <= lower < upper <= 0.5
        for lower, upper in (fit.subject_scale_confidence_intervals_95 or {}).values()
    )
    assert np.all(fit.subject_scale_standard_errors >= fit.subject_scale_local_standard_errors)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        fit.subject_parameter_scales.setflags(write=True)


def test_observed_scale_information_is_complete_information_minus_missing_information() -> None:
    generator = np.random.default_rng(4)
    knots = (0.0, 2.0, 5.0)
    n_knots = len(knots)
    n_subjects = 6
    smoothness = 3.0
    scales = np.asarray([0.30, 0.12])
    deviations = generator.normal(0.0, 0.15, (n_subjects, len(scales), n_knots))
    covariances = tuple(
        0.001 * (matrix @ matrix.T) + 0.002 * np.eye(len(scales) * n_knots)
        for matrix in generator.normal(0.0, 1.0, (n_subjects, len(scales) * n_knots, 6))
    )

    information = _louis_scale_information(
        scales,
        deviations,
        covariances,
        subject_smoothness=smoothness,
        knots=knots,
    )

    complete = np.diag(
        [
            (
                scale
                / _scale_standard_error(
                    scale,
                    deviations[:, index, :],
                    tuple(
                        covariance[
                            index * n_knots : (index + 1) * n_knots,
                            index * n_knots : (index + 1) * n_knots,
                        ]
                        for covariance in covariances
                    ),
                    subject_smoothness=smoothness,
                    knots=knots,
                )
            )
            ** 2
            for index, scale in enumerate(scales)
        ]
    )
    selectors = [
        np.diag(np.repeat(np.arange(len(scales)) == index, n_knots).astype(np.float64))
        for index in range(len(scales))
    ]
    means = deviations.reshape(n_subjects, -1)
    missing = np.empty((len(scales), len(scales)), dtype=np.float64)
    for row in range(len(scales)):
        for column in range(len(scales)):
            missing[row, column] = sum(
                2.0 * np.trace(selectors[row] @ covariance @ selectors[column] @ covariance)
                + 4.0 * mean @ selectors[row] @ covariance @ selectors[column] @ mean
                for covariance, mean in zip(covariances, means, strict=True)
            ) / (scales[row] ** 2 * scales[column] ** 2)

    np.testing.assert_allclose(information, complete - missing, rtol=1e-5)
    assert np.all(np.diag(information) < np.diag(complete))


def test_observed_scale_uncertainty_widens_the_uncorrected_curvature_interval() -> None:
    model = hierarchical_model(
        subject_parameter_scales={"drift.stimulus": 0.22, "boundary": 0.08},
        estimate_subject_scales=True,
        subject_scale_bounds=(0.04, 0.5),
        scale_max_iterations=12,
        scale_tolerance=0.03,
        n_restarts=1,
    )
    design = make_design(n_subjects=8, trials_per_session=70)
    study = model.simulate(design, population_truth(model), seed=91)

    fit = model.fit(study)

    assert model.subject_scale_uncertainty == "observed"
    assert fit.subject_scale_uncertainty_policy == "louis-observed-information"
    assert fit.subject_scale_em_rate_matrix is None
    assert fit.subject_scale_em_spectral_radius is None
    assert np.all(fit.subject_scale_standard_errors > fit.subject_scale_local_standard_errors)
    assert np.min(np.linalg.eigvalsh(fit.subject_scale_covariance)) > 0.0
    for parameter, (lower, upper) in (fit.subject_scale_confidence_intervals_95 or {}).items():
        local_lower, local_upper = (fit.subject_scale_local_confidence_intervals_95 or {})[
            parameter
        ]
        assert lower <= local_lower < local_upper <= upper


def test_subject_scale_boundary_diagnostics_use_declared_log_scale_tolerance() -> None:
    indicators = _scales_at_bounds(
        np.asarray([0.0505, 0.2, 0.99]),
        (0.05, 1.0),
        tolerance=0.02,
    )

    assert indicators.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        indicators.setflags(write=True)


def test_unseen_subject_prediction_integrates_one_random_path_per_subject() -> None:
    model = hierarchical_model(n_restarts=1)
    training_design = make_design(n_subjects=4, trials_per_session=45)
    training = model.simulate(training_design, population_truth(model), seed=191)
    fit = model.fit(training)
    unseen_design = make_design(n_subjects=2, trials_per_session=30, seed=192)
    unseen_columns = {name: unseen_design[name] for name in unseen_design.columns}
    unseen_columns["subject"] = np.asarray(
        [f"new-{subject}" for subject in unseen_design["subject"]]
    )
    unseen = model.simulate(Study(unseen_columns), population_truth(model), seed=193)

    first = model.predict_new_subjects(unseen, fit, n_draws=64, seed=194)
    second = model.predict_new_subjects(unseen, fit, n_draws=64, seed=194)
    plugin = model.predict(unseen, fit)

    assert isinstance(first, UnseenSubjectPosteriorPrediction)
    assert first.n_draws == 64
    assert first.subjects == ("new-mouse-0", "new-mouse-1")
    assert first.random_effect_draws.shape == (64, 2, 2, 2)
    assert first.draw_pointwise_log_probability.shape == (64, len(unseen))
    assert first.subject_joint_log_probability.shape == (2,)
    assert np.all((first.subject_effective_draws >= 1) & (first.subject_effective_draws <= 64))
    assert np.all(first.subject_log_probability_mcse >= 0)
    assert not np.array_equal(first.probability, plugin.probability)
    np.testing.assert_array_equal(first.probability, second.probability)
    np.testing.assert_array_equal(
        first.subject_joint_log_probability,
        second.subject_joint_log_probability,
    )
    np.testing.assert_array_equal(first.prediction.probability, first.probability)
    assert set(first.subject_joint_log_probability_map) == {
        "new-mouse-0",
        "new-mouse-1",
    }
    for subject_index in range(2):
        expected_joint = logsumexp(
            np.sum(
                first.draw_pointwise_log_probability[:, first.row_subject_indices == subject_index],
                axis=1,
            )
        ) - np.log(first.n_draws)
        assert first.subject_joint_log_probability[subject_index] == pytest.approx(expected_joint)
    with pytest.raises(ValueError, match="entirely unseen"):
        model.predict_new_subjects(training.take([0]), fit, n_draws=8, seed=1)


def test_joint_fit_recovers_population_and_subject_trajectory_outputs() -> None:
    model = hierarchical_model()
    design = make_design(trials_per_session=100)
    simulation = model.simulate_with_effects(
        design,
        population_truth(model),
        seed=14,
        subject_deviation_paths=zero_deviations(model, design),
    )

    fit = model.fit(simulation.study)
    population = model.population_trajectory(fit)
    subject = model.subject_trajectory(fit, "mouse-0", times=[0.0, 1.0, 2.0])

    assert isinstance(fit, HierarchicalSmoothDriftDiffusionFitResult)
    assert fit.diagnostics.converged
    assert fit.subjects == simulation.subjects
    assert fit.subject_deviations.shape == (4, 2, 2)
    assert fit.subject_standard_errors.shape == (4, 2, 2)
    assert fit.uncertainty_policy == "arrowhead-local-hessian"
    assert population.path("drift.stimulus").tolist() == pytest.approx([0.6, 1.4], abs=0.25)
    assert population.path("boundary").tolist() == pytest.approx([1.45, 1.1], abs=0.15)
    assert subject.values.shape == (3, 5)
    assert np.all(np.isfinite(fit.subject_knot_values))


def test_seen_and_unseen_subjects_use_the_declared_population_plugin() -> None:
    model = hierarchical_model()
    design = make_design(n_subjects=3, trials_per_session=60)
    study = model.simulate(design, population_truth(model), seed=31)
    fit = model.fit(study)
    source = study.take([0])
    unseen_columns = {name: source[name] for name in source.columns}
    unseen_columns["subject"] = np.asarray(["new-mouse"])
    unseen = Study(unseen_columns)

    seen_trajectory = model.subject_trajectory(fit, "mouse-0")
    unseen_trajectory = model.subject_trajectory(fit, "new-mouse")
    population = model.population_trajectory(fit)

    assert fit.subject_was_fitted("mouse-0")
    assert not fit.subject_was_fitted("new-mouse")
    assert not np.array_equal(seen_trajectory.values, population.values)
    np.testing.assert_allclose(unseen_trajectory.values, population.values)
    assert np.all(np.isfinite(model.predict(unseen, fit).probability))
    assert np.all(np.isfinite(model.pointwise_log_prob(unseen, fit)))


def test_explicit_subject_paths_and_hierarchical_fit_scope_are_validated() -> None:
    model = hierarchical_model()
    design = make_design(n_subjects=2, trials_per_session=20)
    parameters = population_truth(model)

    with pytest.raises(ValueError, match="every design subject"):
        model.simulate_with_effects(
            design,
            parameters,
            seed=1,
            subject_deviation_paths={"mouse-0": {}},
        )

    invalid = zero_deviations(model, design)
    invalid["mouse-0"]["boundary"] = [-2.0, -2.0]
    with pytest.raises(ValueError, match="outside the model bounds"):
        model.simulate_with_effects(
            design,
            parameters,
            seed=1,
            subject_deviation_paths=invalid,
        )

    one_subject = make_design(n_subjects=1, trials_per_session=20)
    with pytest.raises(ModelDataError, match="at least two subjects"):
        model.fit(model.simulate(one_subject, parameters, seed=1))


def test_population_and_session_holdouts_use_valid_declared_policies() -> None:
    model = hierarchical_model()
    design = make_design(n_subjects=4, trials_per_session=45)
    study = model.simulate(design, population_truth(model), seed=44)

    population_evaluations = evaluate_splits(
        model,
        study,
        leave_one_subject_out_splits(study),
    )
    session_evaluations = evaluate_splits(
        model,
        study,
        cohort_forward_session_splits(study, min_train_sessions=2),
    )

    assert len(population_evaluations) == 4
    for evaluation in population_evaluations:
        assert isinstance(evaluation.fit, HierarchicalSmoothDriftDiffusionFitResult)
        assert not evaluation.fit.subject_was_fitted(evaluation.split.held_out_group)
        assert np.all(np.isfinite(evaluation.pointwise_log_probability))
    assert len(session_evaluations) == 1
    assert np.all(np.isfinite(session_evaluations[0].pointwise_log_probability))


def test_generic_parameter_recovery_accepts_population_paths() -> None:
    model = hierarchical_model(n_restarts=1)
    report = run_parameter_recovery(
        model,
        make_design(n_subjects=3, trials_per_session=45),
        [population_truth(model)],
        seed=52,
    )

    assert report.n_runs == 1
    assert report.parameter_names == model.parameter_names
    assert report.audits[0].status.value in {"pass", "warning"}


def test_arrowhead_covariance_matches_dense_quadratic_inverse() -> None:
    population_hessian = np.asarray([[4.0, 0.2], [0.2, 3.0]])
    deviation_hessians = (np.asarray([[2.0]]), np.asarray([[2.5]]))
    crosses = (np.asarray([[0.3], [-0.1]]), np.asarray([[-0.2], [0.4]]))
    full_hessian = np.zeros((4, 4))
    full_hessian[:2, :2] = population_hessian
    for subject, (deviation_hessian, cross) in enumerate(
        zip(deviation_hessians, crosses, strict=True)
    ):
        index = 2 + subject
        full_hessian[index, index] = deviation_hessian[0, 0]
        full_hessian[:2, index : index + 1] = cross
        full_hessian[index : index + 1, :2] = cross.T

    def objective(values: np.ndarray) -> float:
        return 0.5 * float(values @ full_hessian @ values)

    population_covariance, subject_covariances, condition = _arrowhead_covariance(
        objective,
        np.zeros(4),
        population_bounds=[(-5.0, 5.0)] * 2,
        deviation_bounds=[(-5.0, 5.0)],
        n_subjects=2,
    )
    dense_covariance = np.linalg.inv(full_hessian)

    np.testing.assert_allclose(population_covariance, dense_covariance[:2, :2], rtol=1e-5)
    np.testing.assert_allclose(subject_covariances[0], dense_covariance[2:3, 2:3], rtol=1e-5)
    np.testing.assert_allclose(subject_covariances[1], dense_covariance[3:4, 3:4], rtol=1e-5)
    assert condition == pytest.approx(np.linalg.cond(full_hessian), rel=1e-5)
