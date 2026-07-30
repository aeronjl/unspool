from types import MappingProxyType

import numpy as np
import pytest
from scipy.special import expit

from behavio import BernoulliHistoryGLM, Study, evaluate_splits
from behavio.compose import HierarchicalFitResult, HierarchicalModel, hierarchical
from behavio.evaluate import leave_one_subject_out_splits
from behavio.models import BehaviourModel, FitDiagnostics, ModelDataError


def pooled_glm(
    *,
    predictors: tuple[str, ...] = ("stimulus",),
    choice_lags: int = 1,
    l2: float = 0.0,
    **effects: object,
) -> HierarchicalModel:
    base = BernoulliHistoryGLM(predictors=predictors, choice_lags=choice_lags, l2=l2)
    return hierarchical(base, over="subject", **effects)


def make_population_design(
    *, n_subjects: int = 4, n_sessions: int = 3, n_trials: int = 80
) -> Study:
    generator = np.random.default_rng(20260726)
    subject: list[str] = []
    session: list[str] = []
    trial: list[int] = []
    session_order: list[int] = []
    stimulus: list[float] = []
    for subject_index in range(n_subjects):
        subject_name = f"mouse-{subject_index}"
        for order in range(n_sessions):
            subject.extend([subject_name] * n_trials)
            session.extend([f"session-{order}"] * n_trials)
            trial.extend(range(n_trials))
            session_order.extend([order] * n_trials)
            stimulus.extend(generator.normal(size=n_trials))
    return Study(
        {
            "subject": subject,
            "session": session,
            "trial": trial,
            "session_order": session_order,
            "stimulus": stimulus,
        }
    )


def known_hierarchical_fit(model: HierarchicalModel) -> HierarchicalFitResult:
    return HierarchicalFitResult(
        model_name=model.model_name,
        model_signature=model.signature,
        parameter_names=model.parameter_names,
        estimates=np.asarray([0.0, 1.0]),
        standard_errors=np.asarray([0.1, 0.1]),
        covariance=np.eye(2) * 0.01,
        n_observations=20,
        diagnostics=FitDiagnostics(
            converged=True,
            optimizer="test",
            status=0,
            message="known test fit",
            n_iterations=0,
            objective=0.0,
            gradient_norm=0.0,
            hessian_condition=1.0,
            boundary_estimate=False,
        ),
        grouping="subject",
        groups=("mouse-a", "mouse-b"),
        varying_parameters=model.varying_parameters,
        group_deviations=np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
        group_standard_errors=np.full((2, 2), 0.2),
        scales=np.asarray(model.effects.scales),
    )


def test_hierarchical_model_has_a_bounded_public_contract() -> None:
    model = pooled_glm(choice_lags=0, scale=0.4)

    assert isinstance(model, BehaviourModel)
    assert model.parameter_names == ("intercept", "stimulus")
    assert model.varying_parameters == ("intercept", "stimulus")
    assert "over=subject" in model.signature
    assert "varying=all;scale=0.4" in model.signature

    with pytest.raises(ValueError, match="scale"):
        pooled_glm(scale=0.0)


def test_only_the_named_parameters_vary_by_group() -> None:
    model = pooled_glm(choice_lags=0, parameters=("intercept",), scale=0.3)
    study = model.simulate(
        make_population_design(n_subjects=3, n_sessions=2, n_trials=40),
        {"intercept": -0.2, "stimulus": 1.0},
        seed=5,
    )

    fit = model.fit(study)

    assert model.varying_parameters == ("intercept",)
    assert fit.varying_parameters == ("intercept",)
    assert fit.group_deviations.shape == (3, 1)
    assert fit.parameters_for("mouse-0")["stimulus"] == fit.parameters["stimulus"]
    assert fit.parameters_for("mouse-0")["intercept"] != fit.parameters["intercept"]

    with pytest.raises(ValueError, match="not parameters of this model"):
        pooled_glm(parameters=("bias",))


def test_per_parameter_scales_are_declared_and_reported() -> None:
    model = pooled_glm(choice_lags=0, scale=0.5, parameter_scales={"stimulus": 0.1})

    assert model.effects.scales.tolist() == [0.5, 0.1]
    assert "varying=all;scales=intercept:0.5,stimulus:0.1" in model.signature
    assert any("stimulus ~ Normal(0, 0.1)" in prior for prior in model.declared_priors)

    with pytest.raises(ValueError, match="non-varying parameters"):
        pooled_glm(parameters=("intercept",), parameter_scales={"stimulus": 0.1})


@pytest.mark.parametrize(
    "arguments",
    [
        {"estimate_scale": 1},
        {"scale_bounds": (0.0, 1.0)},
        {"scale_bounds": (1.0, 1.0)},
        {"scale_bounds": (1.0, 0.5)},
        {"scale": 2.0, "estimate_scale": True, "scale_bounds": (0.1, 1.0)},
    ],
)
def test_subject_scale_estimation_configuration_is_validated(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        pooled_glm(**arguments)


def test_simulation_retains_random_effect_truth_outside_observed_study() -> None:
    model = pooled_glm(scale=0.35)
    design = make_population_design(n_subjects=3, n_sessions=2, n_trials=20)
    parameters = {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.4}

    first = model.simulate_with_effects(design, parameters, seed=8)
    second = model.simulate_with_effects(design, parameters, seed=8)

    assert first.groups == design.subjects
    assert first.group_deviations.shape == (3, 3)
    assert first.group_parameters.shape == (3, 3)
    assert np.array_equal(first.study["choice"], second.study["choice"])
    assert np.array_equal(first.group_deviations, second.group_deviations)
    assert "subject_deviation" not in first.study.columns
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.group_deviations.setflags(write=True)


def test_joint_fit_exposes_population_and_shrunken_subject_estimates() -> None:
    model = pooled_glm(scale=0.45)
    truth = {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.35}
    simulated = model.simulate_with_effects(make_population_design(), truth, seed=14)

    fit = model.fit(simulated.study)

    assert fit.diagnostics.converged
    assert fit.groups == simulated.groups
    assert fit.group_deviations.shape == (4, 3)
    assert fit.group_standard_errors.shape == (4, 3)
    assert np.all(np.isfinite(fit.group_parameters))
    assert np.all(fit.group_standard_errors > 0)
    assert fit.estimates.tolist() == pytest.approx(list(truth.values()), abs=0.35)
    assert isinstance(fit.parameters_for("mouse-0"), MappingProxyType)
    assert fit.group_was_fitted("mouse-0")
    assert not fit.group_was_fitted("new-mouse")
    assert not fit.scale_estimated
    assert fit.scale_standard_error is None
    assert fit.scale_confidence_interval_95 is None


def test_laplace_fit_estimates_subject_scale_and_uncertainty() -> None:
    truth_scale = 0.5
    generator = pooled_glm(l2=0.05, scale=truth_scale)
    study = generator.simulate(
        make_population_design(n_subjects=12, n_sessions=4, n_trials=35),
        {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.35},
        seed=11,
    )
    model = pooled_glm(l2=0.05, scale=0.25, estimate_scale=True, scale_bounds=(0.05, 1.5))

    fit = model.fit(study)

    assert fit.diagnostics.converged
    assert fit.diagnostics.optimizer == "L-BFGS-B with Laplace marginal likelihood"
    assert fit.scale_estimated
    assert not fit.scale_at_boundary
    assert float(fit.scales[0]) == pytest.approx(truth_scale, abs=0.15)
    assert fit.scale_standard_error is not None
    assert fit.scale_standard_error > 0
    assert fit.scale_confidence_interval_95 is not None
    lower, upper = fit.scale_confidence_interval_95
    assert lower < truth_scale < upper


def test_estimated_scale_is_stable_to_distinct_initial_values() -> None:
    generator = pooled_glm(choice_lags=0, scale=0.6)
    study = generator.simulate(
        make_population_design(n_subjects=10, n_sessions=3, n_trials=35),
        {"intercept": -0.2, "stimulus": 1.0},
        seed=82,
    )

    estimates = [
        float(
            pooled_glm(
                choice_lags=0,
                scale=initial,
                estimate_scale=True,
                scale_bounds=(0.05, 1.5),
            )
            .fit(study)
            .scales[0]
        )
        for initial in (0.15, 1.2)
    ]

    assert estimates[0] == pytest.approx(estimates[1], abs=1e-3)


def test_small_heterogeneity_is_visible_as_a_scale_boundary() -> None:
    generator = pooled_glm(l2=0.05, scale=0.1)
    study = generator.simulate(
        make_population_design(n_subjects=12, n_sessions=4, n_trials=35),
        {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.35},
        seed=11,
    )
    model = pooled_glm(l2=0.05, scale=0.4, estimate_scale=True, scale_bounds=(0.05, 1.5))

    fit = model.fit(study)

    assert fit.diagnostics.converged
    assert fit.scale_at_boundary
    assert fit.diagnostics.boundary_estimate
    assert float(fit.scales[0]) == pytest.approx(0.05)


def test_prediction_declares_seen_and_unseen_subject_behavior() -> None:
    model = pooled_glm(choice_lags=0, scale=0.5)
    study = Study(
        {
            "subject": ["mouse-a", "mouse-b", "new-mouse"],
            "session": ["s", "s", "s"],
            "trial": [0, 0, 0],
            "session_order": [0, 0, 0],
            "stimulus": [0.0, 0.0, 0.0],
            "choice": [1, 0, 1],
        }
    )
    fit = known_hierarchical_fit(model)

    prediction = model.predict(study, fit)

    assert prediction.probability.tolist() == pytest.approx([expit(1.0), expit(-1.0), 0.5])
    assert fit.parameters_for("new-mouse") == {"intercept": 0.0, "stimulus": 1.0}
    assert model.pointwise_log_prob(study, fit).tolist() == pytest.approx(
        [np.log(expit(1.0)), np.log1p(-expit(-1.0)), np.log(0.5)]
    )


def test_hierarchical_fit_rejects_one_subject_and_static_fit_results() -> None:
    model = pooled_glm(choice_lags=0, scale=0.5)
    one_subject = make_population_design(n_subjects=1, n_sessions=1, n_trials=20)
    simulated = model.simulate(one_subject, {"intercept": 0.0, "stimulus": 1.0}, seed=3)

    with pytest.raises(ModelDataError, match="at least two subject"):
        model.fit(simulated)

    population_study = model.simulate(
        make_population_design(n_subjects=2, n_sessions=1, n_trials=20),
        {"intercept": 0.0, "stimulus": 1.0},
        seed=4,
    )
    static_model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0)
    static_fit = static_model.fit(population_study)
    with pytest.raises(ValueError, match="hierarchical group effects"):
        model.predict(population_study, static_fit)


def test_leave_subject_out_evaluation_uses_the_unseen_subject_policy() -> None:
    model = pooled_glm(choice_lags=0, scale=0.4)
    study = model.simulate(
        make_population_design(n_subjects=3, n_sessions=2, n_trials=40),
        {"intercept": -0.2, "stimulus": 1.0},
        seed=91,
    )

    evaluations = evaluate_splits(model, study, leave_one_subject_out_splits(study))

    assert len(evaluations) == 3
    for evaluation in evaluations:
        assert isinstance(evaluation.fit, HierarchicalFitResult)
        held_out_subject = evaluation.split.held_out_group
        assert not evaluation.fit.group_was_fitted(held_out_subject)
        assert np.all(np.isfinite(evaluation.prediction.probability))
