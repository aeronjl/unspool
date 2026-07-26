import numpy as np
import pytest
from scipy.special import expit

from unspool import (
    BehaviourModel,
    FitDiagnostics,
    HierarchicalSmoothBernoulliHistoryGLM,
    HierarchicalSmoothGLMFitResult,
    ModelDataError,
    Study,
    evaluate_splits,
    leave_one_subject_out_splits,
)


def make_design(*, n_subjects: int = 6, n_sessions: int = 5, trials_per_session: int = 80) -> Study:
    generator = np.random.default_rng(9021)
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


def smooth_model(**changes: object) -> HierarchicalSmoothBernoulliHistoryGLM:
    arguments = {
        "covariates": ("stimulus",),
        "choice_lags": 0,
        "knots": (0.0, 2.0, 4.0),
        "smoothness": 4.0,
        "l2": 0.02,
        "subject_scale": 0.4,
        "subject_smoothness": 4.0,
    }
    arguments.update(changes)
    return HierarchicalSmoothBernoulliHistoryGLM(**arguments)


def zero_deviations(
    model: HierarchicalSmoothBernoulliHistoryGLM, study: Study
) -> dict[str, dict[str, list[float]]]:
    return {
        subject: {coefficient: [0.0] * len(model.knots) for coefficient in model.coefficient_names}
        for subject in study.subjects
    }


def known_fit(
    model: HierarchicalSmoothBernoulliHistoryGLM,
) -> HierarchicalSmoothGLMFitResult:
    return HierarchicalSmoothGLMFitResult(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        estimates=np.asarray([0.0, 0.0, 1.0, 1.0]),
        standard_errors=np.full(4, 0.1),
        covariance=np.eye(4) * 0.01,
        n_observations=30,
        diagnostics=FitDiagnostics(
            converged=True,
            optimizer="test",
            status=0,
            message="known fit",
            n_iterations=0,
            objective=0.0,
            gradient_norm=0.0,
            hessian_condition=1.0,
            boundary_estimate=False,
        ),
        subjects=("mouse-a", "mouse-b"),
        clock=model.time,
        knots=model.knots,
        coefficient_names=model.coefficient_names,
        subject_deviations=np.asarray(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[-1.0, -1.0], [0.0, 0.0]],
            ]
        ),
        subject_standard_errors=np.full((2, 2, 2), 0.2),
        subject_scale=model.subject_scale,
        subject_smoothness=model.subject_smoothness,
    )


def test_hierarchical_smooth_model_has_a_stable_public_contract() -> None:
    model = smooth_model(knots=(0.0, 2.0))

    assert isinstance(model, BehaviourModel)
    assert model.parameter_names == (
        "intercept[session_order=0]",
        "intercept[session_order=2]",
        "stimulus[session_order=0]",
        "stimulus[session_order=2]",
    )
    assert "subject_smoothness=4.0" in model.signature


@pytest.mark.parametrize(
    "arguments",
    [
        {"knots": (0.0,)},
        {"knots": (0.0, 0.0)},
        {"knots": (1.0, 0.0)},
        {"time": "choice"},
        {"smoothness": 0.0},
        {"subject_scale": 0.0},
        {"subject_smoothness": 0.0},
    ],
)
def test_hierarchical_smooth_configuration_is_validated(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        smooth_model(**arguments)


def test_simulation_retains_reproducible_subject_paths_outside_the_study() -> None:
    model = smooth_model()
    design = make_design(n_subjects=3, n_sessions=5, trials_per_session=20)
    population = model.parameters_from_paths(
        {"intercept": [-0.2, 0.0, 0.2], "stimulus": [0.5, 1.0, 1.5]}
    )

    first = model.simulate_with_effects(design, population, seed=8)
    second = model.simulate_with_effects(design, population, seed=8)

    assert first.population_knot_values.shape == (2, 3)
    assert first.subject_deviation_knot_values.shape == (3, 2, 3)
    assert first.subject_knot_values.shape == (3, 2, 3)
    assert np.array_equal(first.study["choice"], second.study["choice"])
    assert np.array_equal(first.subject_deviation_knot_values, second.subject_deviation_knot_values)
    assert "subject_deviation" not in first.study.columns
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.subject_knot_values.setflags(write=True)


def test_joint_fit_recovers_population_and_subject_trajectory_outputs() -> None:
    model = smooth_model()
    design = make_design()
    paths = {"intercept": [-0.2, 0.0, 0.2], "stimulus": [0.5, 1.0, 1.5]}
    simulation = model.simulate_with_effects(
        design,
        model.parameters_from_paths(paths),
        seed=14,
        subject_deviation_paths=zero_deviations(model, design),
    )

    fit = model.fit(simulation.study)
    population = model.population_trajectory(fit)
    subject = model.subject_trajectory(fit, "mouse-0", times=[0.0, 1.0, 4.0])

    assert fit.diagnostics.converged
    assert fit.subjects == simulation.subjects
    assert fit.subject_deviations.shape == (6, 2, 3)
    assert fit.subject_standard_errors.shape == (6, 2, 3)
    assert population.values[:, 1].tolist() == pytest.approx(paths["stimulus"], abs=0.2)
    assert subject.values.shape == (3, 2)
    assert np.all(np.isfinite(fit.subject_knot_values))


def test_seen_and_unseen_subjects_use_declared_trajectory_policy() -> None:
    model = smooth_model(knots=(0.0, 2.0))
    fit = known_fit(model)
    study = Study(
        {
            "subject": ["mouse-a", "mouse-b", "new-mouse"],
            "session": ["s", "s", "s"],
            "trial": [0, 0, 0],
            "session_order": [1, 1, 1],
            "stimulus": [0.0, 0.0, 0.0],
            "choice": [1, 0, 1],
        }
    )

    prediction = model.predict(study, fit)
    unseen = model.subject_trajectory(fit, "new-mouse", times=[1.0])

    assert prediction.probability.tolist() == pytest.approx([expit(1.0), expit(-1.0), 0.5])
    assert fit.subject_was_fitted("mouse-a")
    assert not fit.subject_was_fitted("new-mouse")
    assert np.allclose(unseen.values, [[0.0, 1.0]])
    assert model.pointwise_log_prob(study, fit).shape == (3,)


def test_explicit_subject_paths_and_model_scope_are_validated() -> None:
    model = smooth_model()
    design = make_design(n_subjects=2, n_sessions=5, trials_per_session=10)
    parameters = model.parameters_from_paths(
        {"intercept": [0.0, 0.0, 0.0], "stimulus": [1.0, 1.0, 1.0]}
    )

    with pytest.raises(ValueError, match="every design subject"):
        model.simulate_with_effects(
            design,
            parameters,
            seed=1,
            subject_deviation_paths={"mouse-0": {}},
        )

    one_subject = make_design(n_subjects=1, n_sessions=5, trials_per_session=10)
    with pytest.raises(ModelDataError, match="at least two subjects"):
        model.fit(model.simulate(one_subject, parameters, seed=1))

    outside_knots = make_design(n_subjects=2, n_sessions=6, trials_per_session=10)
    with pytest.raises(ModelDataError, match="fixed knot range"):
        model.simulate(outside_knots, parameters, seed=1)


def test_population_holdout_uses_the_unseen_trajectory_plugin() -> None:
    model = smooth_model(knots=(0.0, 2.0), choice_lags=0)
    design = make_design(n_subjects=3, n_sessions=3, trials_per_session=35)
    parameters = model.parameters_from_paths({"intercept": [-0.2, 0.2], "stimulus": [0.7, 1.2]})
    study = model.simulate(design, parameters, seed=44)

    evaluations = evaluate_splits(model, study, leave_one_subject_out_splits(study))

    assert len(evaluations) == 3
    for evaluation in evaluations:
        assert isinstance(evaluation.fit, HierarchicalSmoothGLMFitResult)
        assert not evaluation.fit.subject_was_fitted(evaluation.split.held_out_group)
        assert np.all(np.isfinite(evaluation.prediction.probability))
