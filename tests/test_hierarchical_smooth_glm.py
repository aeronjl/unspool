"""``hierarchical(smooth(...))``: the composed model that used to be its own class."""

import numpy as np
import pytest
from scipy.special import expit

from behavio import (
    BehaviourModel,
    BernoulliHistoryGLM,
    FitDiagnostics,
    ModelDataError,
    Study,
    evaluate_splits,
    leave_one_subject_out_splits,
)
from behavio.compose import (
    HierarchicalFitResult,
    HierarchicalModel,
    SmoothModel,
    hierarchical,
    smooth,
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


def paths_model(
    *,
    knots: tuple[float, ...] = (0.0, 2.0, 4.0),
    smoothness: float = 4.0,
    group_smoothness: float | None = None,
    choice_lags: int = 0,
    l2: float = 0.02,
    over: str = "session_order",
) -> SmoothModel:
    return smooth(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=choice_lags, l2=l2),
        over=over,
        knots=knots,
        smoothness=smoothness,
        group_smoothness=group_smoothness,
    )


def smooth_model(*, scale: float = 0.4, **changes: object) -> HierarchicalModel:
    return hierarchical(paths_model(**changes), over="subject", scale=scale)


def known_fit(model: HierarchicalModel) -> HierarchicalFitResult:
    return HierarchicalFitResult(
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
        grouping="subject",
        groups=("mouse-a", "mouse-b"),
        varying_parameters=model.varying_parameters,
        group_deviations=np.asarray([[1.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0]]),
        group_standard_errors=np.full((2, 4), 0.2),
        scales=np.asarray(model.effects.scales),
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
    assert model.model_name == "hierarchical-smooth-bernoulli-history-glm"
    assert "group_smoothness=4.0" in model.signature
    assert "over=subject" in model.signature


def test_hierarchy_is_the_outer_combinator() -> None:
    pooled = hierarchical(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0), over="subject"
    )

    with pytest.raises(TypeError, match="hierarchical\\(smooth\\(model\\)\\)"):
        smooth(pooled, over="session_order", knots=(0.0, 2.0))


@pytest.mark.parametrize(
    "arguments",
    [
        {"knots": (0.0,)},
        {"knots": (0.0, 0.0)},
        {"knots": (1.0, 0.0)},
        {"over": "choice"},
        {"smoothness": 0.0},
        {"group_smoothness": 0.0},
    ],
)
def test_hierarchical_smooth_configuration_is_validated(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        smooth_model(**arguments)

    with pytest.raises(ValueError):
        smooth_model(scale=0.0)


def test_simulation_retains_reproducible_subject_paths_outside_the_study() -> None:
    model = smooth_model()
    design = make_design(n_subjects=3, n_sessions=5, trials_per_session=20)
    population = model.model.parameters_from_paths(
        {"intercept": [-0.2, 0.0, 0.2], "stimulus": [0.5, 1.0, 1.5]}
    )

    first = model.simulate_with_effects(design, population, seed=8)
    second = model.simulate_with_effects(design, population, seed=8)

    assert first.population_parameters.shape == (6,)
    assert first.group_deviations.shape == (3, 6)
    assert first.group_parameters.shape == (3, 6)
    assert np.array_equal(first.study["choice"], second.study["choice"])
    assert np.array_equal(first.group_deviations, second.group_deviations)
    assert "subject_deviation" not in first.study.columns
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.group_parameters.setflags(write=True)


def test_joint_fit_recovers_population_and_subject_trajectory_outputs() -> None:
    model = smooth_model()
    paths = model.model
    design = make_design()
    truth = {"intercept": [-0.2, 0.0, 0.2], "stimulus": [0.5, 1.0, 1.5]}
    zero = dict.fromkeys(design.subjects, [0.0] * len(model.parameter_names))
    simulation = model.simulate_with_effects(
        design,
        paths.parameters_from_paths(truth),
        seed=14,
        group_deviations=zero,
    )

    fit = model.fit(simulation.study)
    population = paths.trajectory_from_knots(fit.estimates)
    subject = paths.trajectory_from_knots(
        np.asarray(list(fit.parameters_for("mouse-0").values())), times=[0.0, 1.0, 4.0]
    )

    assert fit.diagnostics.converged
    assert fit.groups == simulation.groups
    assert fit.group_deviations.shape == (6, 6)
    assert fit.group_standard_errors.shape == (6, 6)
    assert population.values[:, 1].tolist() == pytest.approx(truth["stimulus"], abs=0.2)
    assert subject.values.shape == (3, 2)
    assert np.all(np.isfinite(fit.group_parameters))


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
    unseen = model.model.trajectory_from_knots(
        np.asarray(list(fit.parameters_for("new-mouse").values())), times=[1.0]
    )

    assert prediction.probability.tolist() == pytest.approx([expit(1.0), expit(-1.0), 0.5])
    assert fit.group_was_fitted("mouse-a")
    assert not fit.group_was_fitted("new-mouse")
    assert np.allclose(unseen.values, [[0.0, 1.0]])
    assert model.pointwise_log_prob(study, fit).shape == (3,)


def test_explicit_subject_paths_and_model_scope_are_validated() -> None:
    model = smooth_model()
    design = make_design(n_subjects=2, n_sessions=5, trials_per_session=10)
    parameters = model.model.parameters_from_paths(
        {"intercept": [0.0, 0.0, 0.0], "stimulus": [1.0, 1.0, 1.0]}
    )

    with pytest.raises(ValueError, match="every design group"):
        model.simulate_with_effects(design, parameters, seed=1, group_deviations={"mouse-0": []})

    one_subject = make_design(n_subjects=1, n_sessions=5, trials_per_session=10)
    with pytest.raises(ModelDataError, match="at least two subject"):
        model.fit(model.simulate(one_subject, parameters, seed=1))

    outside_knots = make_design(n_subjects=2, n_sessions=6, trials_per_session=10)
    with pytest.raises(ModelDataError, match="fixed knot range"):
        model.simulate(outside_knots, parameters, seed=1)


def test_population_holdout_uses_the_unseen_trajectory_plugin() -> None:
    model = smooth_model(knots=(0.0, 2.0), choice_lags=0)
    design = make_design(n_subjects=3, n_sessions=3, trials_per_session=35)
    parameters = model.model.parameters_from_paths(
        {"intercept": [-0.2, 0.2], "stimulus": [0.7, 1.2]}
    )
    study = model.simulate(design, parameters, seed=44)

    evaluations = evaluate_splits(model, study, leave_one_subject_out_splits(study))

    assert len(evaluations) == 3
    for evaluation in evaluations:
        assert isinstance(evaluation.fit, HierarchicalFitResult)
        assert not evaluation.fit.group_was_fitted(evaluation.split.held_out_group)
        assert np.all(np.isfinite(evaluation.prediction.probability))


def test_a_group_deviation_path_inherits_the_population_roughness_prior() -> None:
    model = smooth_model(knots=(0.0, 2.0, 4.0), smoothness=4.0)
    columns = model.effects.columns(model.parameter_names)
    scales = model.effects.ordered_scales(model.parameter_names)
    penalty = model.model.group_penalty(columns, scales)

    off_diagonal = penalty - np.diag(np.diag(penalty))

    assert penalty.shape == (6, 6)
    assert np.any(off_diagonal != 0.0), "a deviation path must be penalised for jumping"

    free = hierarchical(paths_model(group_smoothness=1e-9), over="subject", scale=0.4)
    free_penalty = free.model.group_penalty(
        columns, free.effects.ordered_scales(free.parameter_names)
    )
    assert np.max(np.abs(free_penalty - np.diag(np.diag(free_penalty)))) < 1e-8
