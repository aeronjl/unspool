"""``mix()``: one mixture algebra where three hard-coded mechanisms used to be.

A lapse on a GLM, on a multinomial and on a GLM-HMM could not be written at all before this
combinator; a lapse on a psychometric curve needed its own class; a lapse on a
drift-diffusion model needed a constructor slot on that one family. What is tested here is
that all of them are the same expression, that the one weight it adds is estimated and
recovered rather than declared, that the orders it composes in are the ones that mean
something, and that a mixture the design cannot identify says so before it is fitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from behavio import (
    BehaviourModel,
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    BiasOnly,
    BinaryRLAgent,
    ChoiceSpec,
    DesignSpec,
    MultinomialLogit,
    NumericTerm,
    Study,
    UniformCategoryGuess,
    UniformChoiceGuess,
    UniformResponseGuess,
    WienerDriftDiffusion,
    compare_models,
    evaluate_splits,
    forward_session_splits,
    mix,
    run_parameter_recovery,
)
from behavio.compose import MixtureModel, hierarchical, smooth
from behavio.contracts.compose import PenalisedLinearEstimator
from behavio.contracts.mixture import (
    MIXTURE_LOGIT,
    MIXTURE_LOGIT_BOUND,
    MixtureComponent,
    mixture_weight,
)

KNOTS = (0.0, 3.0)


def design(
    *, n_subjects: int = 1, n_sessions: int = 4, n_trials: int = 250, seed: int = 7
) -> Study:
    generator = np.random.default_rng(seed)
    rows = n_subjects * n_sessions * n_trials
    return Study(
        {
            "subject": [
                f"m{subject}" for subject in range(n_subjects) for _ in range(n_sessions * n_trials)
            ],
            "session": [
                session
                for _ in range(n_subjects)
                for session in range(n_sessions)
                for _ in range(n_trials)
            ],
            "session_order": [
                session
                for _ in range(n_subjects)
                for session in range(n_sessions)
                for _ in range(n_trials)
            ],
            "trial": list(range(n_trials)) * (n_subjects * n_sessions),
            "stimulus": generator.normal(size=rows),
        }
    )


def glm(**changes: Any) -> BernoulliHistoryGLM:
    arguments: dict[str, Any] = {"covariates": ("stimulus",), "choice_lags": 0}
    arguments.update(changes)
    return BernoulliHistoryGLM(**arguments)


def lapsing_glm(**changes: Any) -> MixtureModel:
    return mix(glm(), UniformChoiceGuess(), **changes)


# --------------------------------------------------------------------------------------
# One idea, three families that could not express it
# --------------------------------------------------------------------------------------


def test_a_lapse_on_a_glm_is_one_extra_parameter_and_recovers() -> None:
    model = lapsing_glm(weight_bounds=(0.0, 0.3))
    truth = model.from_natural({"intercept": 0.2, "stimulus": 1.5, "lapse_rate": 0.12})
    study = model.simulate(design(n_trials=600), truth, seed=11)

    fit = model.fit(study)
    recovered = model.to_natural(fit.estimates)

    assert isinstance(model, BehaviourModel)
    assert isinstance(model, PenalisedLinearEstimator)
    assert model.parameter_names == ("intercept", "stimulus", MIXTURE_LOGIT)
    assert model.natural_names == ("intercept", "stimulus", "lapse_rate")
    assert recovered["lapse_rate"] == pytest.approx(0.12, abs=0.04)
    assert recovered["stimulus"] == pytest.approx(1.5, abs=0.3)
    assert fit.derived_quantities["lapse_rate"].standard_error > 0
    assert fit.diagnostics.converged
    assert not fit.diagnostics.boundary_estimate


def test_a_lapse_on_a_multinomial_guesses_uniformly_over_the_options_offered() -> None:
    choice = ChoiceSpec(options=("left", "right", "up"), available_options_column="available")
    base = MultinomialLogit(choice=choice, design=DesignSpec((NumericTerm("stimulus"),)))
    model = mix(base, UniformCategoryGuess(choice=choice), weight_bounds=(0.0, 0.3))
    source = design(n_trials=500)
    available = np.empty(len(source), dtype=object)
    available[:] = [
        ("left", "right") if index % 3 == 0 else ("left", "right", "up")
        for index in range(len(source))
    ]
    study = Study({**{name: source[name] for name in source.columns}, "available": available})
    truth = model.from_natural(
        {
            "category['right']::intercept": 0.5,
            "category['right']::stimulus": 3.0,
            "category['up']::intercept": -0.5,
            "category['up']::stimulus": -2.5,
            "lapse_rate": 0.15,
        }
    )

    simulation = model.simulate_with_component(study, truth, seed=3)
    fit = model.fit(simulation.study)
    recovered = model.to_natural(fit.estimates)
    probability = model.predict(simulation.study, fit).probability

    assert recovered["lapse_rate"] == pytest.approx(0.15, abs=0.05)
    assert recovered["category['right']::stimulus"] == pytest.approx(3.0, abs=0.6)
    # A guess never picks an option the trial did not offer, at any weight.
    unavailable = np.asarray([len(row) == 2 for row in simulation.study["available"]])
    assert np.all(probability[unavailable, 2] == 0.0)
    assert np.all(np.asarray(simulation.study["choice"])[unavailable] != "up")


def test_a_lapse_on_a_drift_diffusion_model_scores_the_joint_observation() -> None:
    base = WienerDriftDiffusion(
        covariates=("stimulus",),
        nondecision_time_bounds=(0.1, 0.6),
        n_restarts=2,
        simulation_time_step=0.001,
    )
    model = mix(base, UniformResponseGuess(time_bounds=(0.05, 3.0)), n_restarts=2)
    truth = model.from_natural(
        {
            "drift.intercept": 0.2,
            "drift.stimulus": 1.2,
            "boundary": 1.2,
            "starting_bias": 0.45,
            "nondecision_time": 0.25,
            "contaminant_rate": 0.08,
        }
    )

    simulation = model.simulate_with_component(design(n_trials=200), truth, seed=13)
    fit = model.fit(simulation.study)
    recovered = model.to_natural(fit.estimates)

    assert model.outcome_channels == ("choice", "response_time")
    assert recovered["contaminant_rate"] == pytest.approx(0.08, abs=0.035)
    assert recovered["nondecision_time"] == pytest.approx(0.25, abs=0.02)
    # The whole point of the mixture: a response faster than the non-decision time is not
    # an inadmissible fit, it is a trial the second process explains.
    fastest = float(np.min(simulation.study["response_time"]))
    assert fastest < recovered["nondecision_time"]
    assert np.all(np.isfinite(model.pointwise_log_prob(simulation.study, fit)))


# --------------------------------------------------------------------------------------
# The weight: estimated, bounded by declaration, and reported everywhere
# --------------------------------------------------------------------------------------


def test_the_weight_is_estimated_inside_a_declared_range_that_is_reported() -> None:
    model = lapsing_glm(weight_bounds=(0.02, 0.2))
    study = model.simulate(
        design(n_trials=100),
        model.from_natural({"intercept": 0.0, "stimulus": 1.5, "lapse_rate": 0.1}),
        seed=5,
    )
    description = model.describe(study)

    assert "weight_bounds=0.02,0.2" in model.signature
    assert any("lapse_rate estimated in [0.02, 0.2]" in prior for prior in description.priors)
    assert description.parameter_names[-1] == MIXTURE_LOGIT
    assert float(mixture_weight(-np.inf, model.weight_bounds)) == pytest.approx(0.02)
    assert float(mixture_weight(np.inf, model.weight_bounds)) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="0 <= lower < upper < 1"):
        lapsing_glm(weight_bounds=(0.3, 0.3))
    with pytest.raises(ValueError, match="0 <= lower < upper < 1"):
        lapsing_glm(weight_bounds=(0.0, 1.0))


def test_a_recovery_study_recovers_the_weight_alongside_the_model() -> None:
    model = lapsing_glm(weight_bounds=(0.0, 0.3))
    # A steep slope is what makes a lapse identifiable at all: the two trade off, so the
    # weight is estimated from how flat the asymptotes are relative to how steep the middle
    # is. At a shallow slope the same estimator is unbiased with several times the spread.
    truths = [
        dict(model.from_natural({"intercept": 0.1, "stimulus": 3.0, "lapse_rate": rate}))
        for rate in (0.03, 0.10, 0.20)
    ]

    report = run_parameter_recovery(model, design(n_trials=1_000), truths, seed=17)

    assert report.parameter_names == ("intercept", "stimulus", MIXTURE_LOGIT)
    assert report.audit_failure_rate == 0.0
    recovered = [model.to_natural(estimate)["lapse_rate"] for estimate in report.estimates]
    for expected, found in zip((0.03, 0.10, 0.20), recovered, strict=True):
        assert found == pytest.approx(expected, abs=0.06)


def test_a_saturated_weight_is_reported_as_a_boundary_estimate() -> None:
    model = lapsing_glm(weight_bounds=(0.0, 0.25))
    # A truly lapse-free study: the weight has nowhere to go but its floor, and a logit at
    # its floor is a number that has stopped meaning anything rather than a small rate.
    truth = dict(zip(model.parameter_names, (0.1, 2.0, -40.0), strict=True))
    study = model.simulate(design(n_trials=300), truth, seed=21)

    fit = model.fit(study)

    assert model.to_natural(fit.estimates)["lapse_rate"] < 1e-4
    assert abs(float(fit.estimates[-1])) >= MIXTURE_LOGIT_BOUND
    assert fit.diagnostics.boundary_estimate


# --------------------------------------------------------------------------------------
# Which orders mean something
# --------------------------------------------------------------------------------------


def test_hierarchy_over_a_mixture_gives_a_subject_varying_lapse_rate() -> None:
    model = hierarchical(
        lapsing_glm(weight_bounds=(0.0, 0.4)),
        over="subject",
        parameters=(MIXTURE_LOGIT,),
        scale=0.8,
    )
    truth = dict(model.model.from_natural({"intercept": 0.1, "stimulus": 2.0, "lapse_rate": 0.12}))
    simulation = model.simulate_with_effects(design(n_subjects=4, n_trials=250), truth, seed=23)

    fit = model.fit(simulation.study)

    assert fit.varying_parameters == (MIXTURE_LOGIT,)
    assert fit.group_deviations.shape == (4, 1)
    assert np.all(np.isfinite(fit.group_deviations))
    # Shrunken, but ordered with the truth: the animals drawn with the largest lapse
    # deviation are the ones estimated with the largest.
    assert (
        np.corrcoef(simulation.group_deviations.ravel(), fit.group_deviations.ravel())[0, 1] > 0.5
    )
    rates = [model.model.to_natural(vector)["lapse_rate"] for vector in fit.group_parameter_vectors]
    assert all(0.0 < rate < 0.4 for rate in rates)


def test_smoothness_over_a_mixture_gives_a_drifting_lapse_rate() -> None:
    model = smooth(
        lapsing_glm(weight_bounds=(0.0, 0.4)),
        over="session_order",
        knots=KNOTS,
        smoothness=1.0,
    )
    truth = model.parameters_from_paths(
        {"intercept": [0.0, 0.0], "stimulus": [2.0, 2.0], MIXTURE_LOGIT: [-3.0, 0.0]}
    )
    study = model.simulate(design(n_trials=400), truth, seed=27)

    fit = model.fit(study)

    assert model.parameter_names[-2:] == (
        f"{MIXTURE_LOGIT}[session_order=0]",
        f"{MIXTURE_LOGIT}[session_order=3]",
    )
    assert fit.estimates[-2] < fit.estimates[-1]


def test_naming_only_the_wrapped_parameters_leaves_the_weight_stationary() -> None:
    """This is the model ``mix(smooth(model))`` would have been, and why it is refused."""

    model = smooth(
        lapsing_glm(),
        over="session_order",
        knots=KNOTS,
        parameters=("intercept", "stimulus"),
    )

    assert model.parameter_names[-1] == MIXTURE_LOGIT
    assert model.varying_coefficients == ("intercept", "stimulus")


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            lambda: mix(smooth(glm(), over="session_order", knots=KNOTS), UniformChoiceGuess()),
            "write smooth\\(mix\\(model\\)\\)",
        ),
        (
            lambda: mix(hierarchical(glm(), over="subject"), UniformChoiceGuess()),
            "write hierarchical\\(mix\\(model\\)\\)",
        ),
        (
            lambda: mix(lapsing_glm(), UniformChoiceGuess()),
            "a model is mixed with one component",
        ),
    ],
)
def test_the_refused_orders_name_the_working_one(build: Any, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        build()


def test_the_full_stack_composes_in_exactly_one_order() -> None:
    stack = hierarchical(
        smooth(lapsing_glm(), over="session_order", knots=KNOTS, parameters=("stimulus",)),
        over="subject",
        parameters=(MIXTURE_LOGIT,),
        scale=0.5,
    )
    truth = dict(
        stack.model.parameters_from_paths(
            {"intercept": 0.1, "stimulus": [1.5, 2.5], MIXTURE_LOGIT: -2.0}
        )
    )
    study = stack.simulate(design(n_subjects=3, n_trials=150), truth, seed=31)

    fit = stack.fit(study)

    assert stack.parameter_names[-1] == MIXTURE_LOGIT
    assert fit.varying_parameters == (MIXTURE_LOGIT,)
    assert np.all(np.isfinite(fit.estimates))
    with pytest.raises(TypeError, match="hierarchy is the outer combinator"):
        smooth(stack, over="session_order", knots=KNOTS)


# --------------------------------------------------------------------------------------
# Saying no
# --------------------------------------------------------------------------------------


def test_a_model_whose_likelihood_is_a_recursion_is_refused_by_name() -> None:
    with pytest.raises(TypeError, match="latent-state mixture"):
        mix(BernoulliGLMHMM(covariates=("stimulus",)), UniformChoiceGuess())
    with pytest.raises(TypeError, match="recursion over trials"):
        mix(BinaryRLAgent(), UniformChoiceGuess())


def test_a_component_that_cannot_score_the_outcome_says_why() -> None:
    choice = ChoiceSpec(options=("left", "right", "up"))
    categorical = MultinomialLogit(choice=choice, design=DesignSpec((NumericTerm("stimulus"),)))

    with pytest.raises(TypeError, match="rather than a binary choice"):
        mix(categorical, UniformChoiceGuess(choice.column))
    with pytest.raises(TypeError, match="not the categories this guess"):
        mix(categorical, UniformCategoryGuess(choice=ChoiceSpec(options=("left", "right"))))
    with pytest.raises(TypeError, match="are not a mixture of one observation"):
        mix(glm(), UniformResponseGuess(time_bounds=(0.1, 2.0)))
    with pytest.raises(TypeError, match="MixtureComponent"):
        mix(glm(), object())


def test_a_component_is_a_contract_and_the_three_supplied_satisfy_it() -> None:
    components = (
        UniformChoiceGuess(),
        UniformCategoryGuess(choice=ChoiceSpec(options=("left", "right", "up"))),
        UniformResponseGuess(time_bounds=(0.1, 2.0)),
    )

    for component in components:
        assert isinstance(component, MixtureComponent)
        assert component.weight_name in {"lapse_rate", "contaminant_rate"}
        assert component.prediction_width >= 1


# --------------------------------------------------------------------------------------
# Identifiability, reported before the fit rather than after it
# --------------------------------------------------------------------------------------


def test_a_mixture_a_design_cannot_identify_reports_a_finding() -> None:
    """A lapse and a shallow slope trade off; with no slope at all they trade off exactly."""

    model = mix(BiasOnly(), UniformChoiceGuess())
    study = model.simulate(
        design(n_trials=200),
        model.from_natural({"intercept": 0.4, "lapse_rate": 0.1}),
        seed=33,
    )

    description = model.describe(study)
    codes = [finding.code for finding in description.findings]

    assert "unidentified_mixture" in codes
    assert all(finding.severity == "warning" for finding in description.findings)
    assert "traded against the model's own parameters" in str(description)
    # A design that varies is not reported, which is what makes the report worth reading.
    assert "unidentified_mixture" not in [
        finding.code for finding in lapsing_glm().describe(study).findings
    ]


def test_a_component_that_could_not_have_produced_anything_reports_a_finding() -> None:
    base = WienerDriftDiffusion(
        covariates=("stimulus",), nondecision_time_bounds=(0.1, 0.6), n_restarts=2
    )
    plausible = mix(base, UniformResponseGuess(time_bounds=(0.05, 3.0)))
    study = plausible.simulate(
        design(n_trials=150),
        plausible.from_natural(
            {
                "drift.intercept": 0.2,
                "drift.stimulus": 1.2,
                "boundary": 1.2,
                "starting_bias": 0.5,
                "nondecision_time": 0.25,
                "contaminant_rate": 0.05,
            }
        ),
        seed=37,
    )
    impossible = mix(base, UniformResponseGuess(time_bounds=(30.0, 40.0)))

    codes = [finding.code for finding in impossible.describe(study).findings]

    assert "unreachable_mixture_component" in codes
    assert "unreachable_mixture_component" not in [
        finding.code for finding in plausible.describe(study).findings
    ]


# --------------------------------------------------------------------------------------
# The arithmetic, checked against the closed form rather than against itself
# --------------------------------------------------------------------------------------


def test_the_mixed_density_is_the_two_component_average_written_out() -> None:
    model = lapsing_glm(weight_bounds=(0.0, 0.3))
    study = model.simulate(
        design(n_trials=120),
        model.from_natural({"intercept": 0.3, "stimulus": 1.4, "lapse_rate": 0.1}),
        seed=41,
    )
    coordinate = np.asarray([0.25, 1.3, -1.1])
    fitted = float(mixture_weight(coordinate[-1], model.weight_bounds))
    outcomes = model.outcomes(study)
    logistic = expit(coordinate[0] + coordinate[1] * np.asarray(study["stimulus"]))
    probability = (1.0 - fitted) * logistic + fitted * 0.5
    expected = np.log(np.where(outcomes == 1.0, probability, 1.0 - probability))

    cells = model.design_matrix(study) @ coordinate + model.predictor_offsets(study)
    scores = model.likelihood.pointwise_log_prob(cells, outcomes)

    assert scores == pytest.approx(expected)


def test_responsibilities_are_posterior_probabilities_and_not_labels() -> None:
    model = lapsing_glm(weight_bounds=(0.0, 0.3))
    truth = model.from_natural({"intercept": 0.0, "stimulus": 3.0, "lapse_rate": 0.15})
    simulation = model.simulate_with_component(design(n_trials=400), truth, seed=43)

    fit = model.fit(simulation.study)
    responsibility = model.component_responsibility(simulation.study, fit)

    assert np.all((responsibility >= 0) & (responsibility <= 1))
    assert responsibility.mean() == pytest.approx(model.weight(fit), abs=0.03)
    assert np.mean(responsibility[simulation.from_component]) > np.mean(
        responsibility[~simulation.from_component]
    )


def test_a_mixed_model_is_an_ordinary_estimator_to_everything_downstream() -> None:
    model = lapsing_glm(weight_bounds=(0.0, 0.3))
    study = model.simulate(
        design(n_trials=200),
        model.from_natural({"intercept": 0.1, "stimulus": 2.0, "lapse_rate": 0.1}),
        seed=47,
    )
    splits = forward_session_splits(study, min_train_sessions=3)

    evaluations = evaluate_splits(model, study, splits)
    comparison = compare_models({"lapsing": model, "plain": glm()}, study, splits)

    assert len(evaluations) == 1
    assert np.all(np.isfinite(evaluations[0].pointwise_log_probability))
    assert set(comparison.model_order) == {"lapsing", "plain"}
