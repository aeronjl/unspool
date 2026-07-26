from types import MappingProxyType

import numpy as np
import pytest
from scipy.special import expit

from unspool import (
    BehaviourModel,
    BernoulliHistoryGLM,
    FitDiagnostics,
    HierarchicalBernoulliHistoryGLM,
    HierarchicalGLMFitResult,
    ModelDataError,
    Study,
    evaluate_splits,
    leave_one_subject_out_splits,
)


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


def known_hierarchical_fit(
    model: HierarchicalBernoulliHistoryGLM,
) -> HierarchicalGLMFitResult:
    return HierarchicalGLMFitResult(
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
        subjects=("mouse-a", "mouse-b"),
        subject_deviations=np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
        subject_standard_errors=np.full((2, 2), 0.2),
        subject_scale=model.subject_scale,
    )


def test_hierarchical_model_has_a_bounded_public_contract() -> None:
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",), choice_lags=0, subject_scale=0.4
    )

    assert isinstance(model, BehaviourModel)
    assert model.parameter_names == ("intercept", "stimulus")
    assert "subject_scale=0.4" in model.signature

    with pytest.raises(ValueError, match="subject_scale"):
        HierarchicalBernoulliHistoryGLM(subject_scale=0.0)


def test_simulation_retains_random_effect_truth_outside_observed_study() -> None:
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",), choice_lags=1, subject_scale=0.35
    )
    design = make_population_design(n_subjects=3, n_sessions=2, n_trials=20)
    parameters = {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.4}

    first = model.simulate_with_effects(design, parameters, seed=8)
    second = model.simulate_with_effects(design, parameters, seed=8)

    assert first.subjects == design.subjects
    assert first.subject_deviations.shape == (3, 3)
    assert first.subject_coefficients.shape == (3, 3)
    assert np.array_equal(first.study["choice"], second.study["choice"])
    assert np.array_equal(first.subject_deviations, second.subject_deviations)
    assert "subject_deviation" not in first.study.columns
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.subject_deviations.setflags(write=True)


def test_joint_fit_exposes_population_and_shrunken_subject_estimates() -> None:
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",), choice_lags=1, subject_scale=0.45
    )
    truth = {"intercept": -0.2, "stimulus": 1.0, "choice_lag_1": 0.35}
    simulated = model.simulate_with_effects(make_population_design(), truth, seed=14)

    fit = model.fit(simulated.study)

    assert fit.diagnostics.converged
    assert fit.subjects == simulated.subjects
    assert fit.subject_deviations.shape == (4, 3)
    assert fit.subject_standard_errors.shape == (4, 3)
    assert np.all(np.isfinite(fit.subject_coefficients))
    assert np.all(fit.subject_standard_errors > 0)
    assert fit.estimates.tolist() == pytest.approx(list(truth.values()), abs=0.35)
    assert isinstance(fit.coefficients_for("mouse-0"), MappingProxyType)
    assert fit.subject_was_fitted("mouse-0")
    assert not fit.subject_was_fitted("new-mouse")


def test_prediction_declares_seen_and_unseen_subject_behavior() -> None:
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",), choice_lags=0, subject_scale=0.5
    )
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
    assert fit.coefficients_for("new-mouse") == {"intercept": 0.0, "stimulus": 1.0}
    assert model.pointwise_log_prob(study, fit).tolist() == pytest.approx(
        [np.log(expit(1.0)), np.log1p(-expit(-1.0)), np.log(0.5)]
    )


def test_hierarchical_fit_rejects_one_subject_and_static_fit_results() -> None:
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",), choice_lags=0, subject_scale=0.5
    )
    one_subject = make_population_design(n_subjects=1, n_sessions=1, n_trials=20)
    simulated = model.simulate(one_subject, {"intercept": 0.0, "stimulus": 1.0}, seed=3)

    with pytest.raises(ModelDataError, match="at least two subjects"):
        model.fit(simulated)

    population_study = model.simulate(
        make_population_design(n_subjects=2, n_sessions=1, n_trials=20),
        {"intercept": 0.0, "stimulus": 1.0},
        seed=4,
    )
    static_model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=model.l2)
    static_fit = static_model.fit(population_study)
    with pytest.raises(ValueError, match="hierarchical subject effects"):
        model.predict(population_study, static_fit)


def test_leave_subject_out_evaluation_uses_the_unseen_subject_policy() -> None:
    model = HierarchicalBernoulliHistoryGLM(
        covariates=("stimulus",), choice_lags=0, subject_scale=0.4
    )
    study = model.simulate(
        make_population_design(n_subjects=3, n_sessions=2, n_trials=40),
        {"intercept": -0.2, "stimulus": 1.0},
        seed=91,
    )

    evaluations = evaluate_splits(model, study, leave_one_subject_out_splits(study))

    assert len(evaluations) == 3
    for evaluation in evaluations:
        assert isinstance(evaluation.fit, HierarchicalGLMFitResult)
        held_out_subject = evaluation.split.held_out_group
        assert not evaluation.fit.subject_was_fitted(held_out_subject)
        assert np.all(np.isfinite(evaluation.prediction.probability))
