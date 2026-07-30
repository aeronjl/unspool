"""``mix()``: one mixture algebra where three hard-coded mechanisms used to be.

A lapse on a GLM, on a multinomial and on a GLM-HMM could not be written at all before this
combinator; a lapse on a psychometric curve needed its own class; a lapse on a
drift-diffusion model needed a constructor slot on that one family. What is tested here is
that all of them are the same expression, that the one weight it adds is estimated and
recovered rather than declared, that the orders it composes in are the ones that mean
something, and that a mixture the design cannot identify says so before it is fitted.

``mix()`` reaches a model through either estimator contract, because the condition it
imposes is **row independence** and not the presence of a linear predictor. The
bounded-coordinate half of that claim is tested on the family that provoked it, in
``tests/test_economic.py``; what is tested here is the other half -- that opening the second
route left the first one alone. ``tests/fixtures/penalised_mixture_reference.json`` holds a
psychometric lapse and a drift-diffusion contaminant fitted at commit ``6c4521c``, the commit
immediately before the rewrite, and every number in it is asserted **exactly**. A tolerance
would have passed a mixture that had quietly become a different arithmetic; equality will not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from behavio import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    BiasOnly,
    BinaryRLAgent,
    ChoiceSpec,
    MultinomialLogit,
    Psychometric,
    PsychometricFunction,
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
from behavio.compose import MixtureModel, MixtureRowModel, hierarchical, smooth
from behavio.contracts.bounded import BoundedCoordinateEstimator
from behavio.contracts.compose import PenalisedLinearEstimator
from behavio.contracts.mixture import (
    MIXTURE_LOGIT,
    MIXTURE_LOGIT_BOUND,
    MixtureComponent,
    mixture_weight,
    require_mixable,
)
from behavio.design import DesignSpec, NumericTerm
from behavio.models import BehaviourModel
from behavio.models.economic import TemporalDiscounting

KNOTS = (0.0, 3.0)
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "penalised_mixture_reference.json").read_text(
        encoding="utf-8"
    )
)


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
    arguments: dict[str, Any] = {"predictors": ("stimulus",), "choice_lags": 0}
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
        predictors=("stimulus",),
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
    """Row independence is the condition, and a recursion is what fails it.

    Both of these compose through the same contract the economic families are mixed on, and
    both of them satisfy every member a mixture would call. What refuses them is the sentence
    each declares about its own likelihood, read before any member is touched.
    """

    with pytest.raises(TypeError, match="rows are not independent"):
        mix(BernoulliGLMHMM(predictors=("stimulus",)), UniformChoiceGuess())
    with pytest.raises(TypeError, match="recursion over trials"):
        mix(BinaryRLAgent(), UniformChoiceGuess())
    # The refusal is a declaration and not an inspection: both models pass the structural
    # test for the contract a mixture would have reached them through.
    assert isinstance(BinaryRLAgent(), BoundedCoordinateEstimator)
    assert isinstance(BernoulliGLMHMM(predictors=("stimulus",)), BoundedCoordinateEstimator)


def test_a_curve_that_already_estimates_a_lapse_is_refused_by_the_name_it_collides_with() -> None:
    """``PsychometricFunction`` is row-independent, so the gate lets it through -- and then
    the coordinate does not.

    Two rates estimated inside the link plus a symmetric mixture weight is an exactly
    non-identified model: raising ``guess_rate`` and ``lapse_rate`` together reproduces any
    weight. The refusal that catches it is the one that is actually true -- this model
    already reports a ``lapse_rate`` -- rather than a complaint about a missing member.
    """

    assert require_mixable(PsychometricFunction(), combinator="mix") == "bounded-coordinate"
    with pytest.raises(ValueError, match="already has a parameter named 'lapse_rate'"):
        mix(PsychometricFunction(), UniformChoiceGuess())


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


@dataclass(frozen=True, slots=True)
class ImpossibleGuess:
    """A guess whose support contains nothing, so it can never have produced a trial.

    Written here rather than shipped because it is not a process anybody models; it is the
    shape of a declaration made in the wrong unit, which is what the finding it provokes is
    for. A contaminant window in milliseconds on a study recorded in seconds is the real
    version, and it is already tested on the penalised route.
    """

    outcome: str = "choice"

    @property
    def component_name(self) -> str:
        return "impossible"

    @property
    def signature(self) -> str:
        return "impossible-guess[]"

    @property
    def weight_name(self) -> str:
        return "impossible_rate"

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        return ()

    @property
    def prediction_width(self) -> int:
        return 1

    def mixture_refusal(self, model: Any) -> str | None:
        del model
        return None

    def pointwise_log_density(self, study: Study, outcomes: np.ndarray) -> np.ndarray:
        del study
        return np.full(len(outcomes), -np.inf, dtype=np.float64)

    def prediction_probability(self, study: Study) -> np.ndarray:
        return np.full(len(study), 0.5, dtype=np.float64)

    def simulate_outcomes(
        self, study: Study, rows: np.ndarray, *, generator: np.random.Generator
    ) -> dict[str, np.ndarray]:
        del study, generator
        return {self.outcome: np.zeros(len(rows), dtype=np.int8)}


def discount_study(*, constant: bool, n_sessions: int = 2) -> Study:
    """A discounting design that either titrates or repeats one trial, plus observed choices."""

    ratios = (0.2, 0.5, 0.8) if not constant else (0.5,)
    delays = (7.0, 30.0, 365.0) if not constant else (30.0,)
    rows = [(ratio, delay) for _ in range(n_sessions * 20) for ratio in ratios for delay in delays]
    return Study(
        {
            "subject": ["a"] * len(rows),
            "session": ["a-s0"] * len(rows),
            "session_order": [0] * len(rows),
            "trial": list(range(len(rows))),
            "sooner_amount": [100.0 * ratio for ratio, _ in rows],
            "sooner_delay": [0.0] * len(rows),
            "later_amount": [100.0] * len(rows),
            "later_delay": [delay for _, delay in rows],
            "choice": np.random.default_rng(3).integers(0, 2, len(rows)).astype(np.int8),
        }
    )


def test_the_row_route_reports_the_identifiability_findings_that_mean_the_same_thing() -> None:
    """Which of the two findings carried over, and the sense in which the other one did.

    ``unreachable_mixture_component`` is a statement about the component and the observed
    outcomes, so it is the same statement on both routes and is computed by the same code.

    ``unidentified_mixture`` is not read off a design matrix here, because there is not one.
    It is read off the model's prediction at each of the deterministic restarts the model's
    own solver would use, so a design that does not distinguish two rows is still caught --
    a constant design gives a constant prediction at every coordinate -- while a design that
    does is not reported.
    """

    model = TemporalDiscounting(value_scale=100.0)
    lapsing = mix(model, UniformChoiceGuess(), weight_bounds=(0.0, 0.3))
    degenerate = lapsing.describe(discount_study(constant=True)).findings
    usable = lapsing.describe(discount_study(constant=False)).findings

    assert "unidentified_mixture" in [finding.code for finding in degenerate]
    assert "unidentified_mixture" not in [finding.code for finding in usable]
    assert all(finding.severity == "warning" for finding in (*degenerate, *usable))
    # A component whose support excludes every observation is unreachable on this route too,
    # and it is reported by the same code, because it is the same statement.
    unreachable = mix(model, ImpossibleGuess(), weight_bounds=(0.0, 0.3))
    codes = [
        finding.code for finding in unreachable.describe(discount_study(constant=False)).findings
    ]
    assert "unreachable_mixture_component" in codes


def test_the_row_route_forwards_the_wrapped_model_s_own_findings() -> None:
    """A mixture must not swallow what the model it wraps had to say about the study."""

    lapsing = mix(TemporalDiscounting(value_scale=100.0), UniformChoiceGuess())
    columns = {
        name: discount_study(constant=False)[name]
        for name in discount_study(constant=False).columns
    }
    columns["later_delay"] = np.zeros(len(columns["trial"]))

    codes = [finding.code for finding in lapsing.describe(Study(columns)).findings]

    assert "unidentified_discount_rate" in codes


def test_a_component_that_could_not_have_produced_anything_reports_a_finding() -> None:
    base = WienerDriftDiffusion(
        predictors=("stimulus",), nondecision_time_bounds=(0.1, 0.6), n_restarts=2
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


# --------------------------------------------------------------------------------------
# The route a model is mixed on, and the proof the old one did not move
# --------------------------------------------------------------------------------------


def test_which_mixture_a_model_gets_follows_from_its_contract_and_nothing_else() -> None:
    penalised = mix(glm(), UniformChoiceGuess())
    rows = mix(TemporalDiscounting(), UniformChoiceGuess(), weight_bounds=(0.0, 0.3))

    assert isinstance(penalised, MixtureModel)
    assert isinstance(penalised, PenalisedLinearEstimator)
    assert isinstance(rows, MixtureRowModel)
    assert isinstance(rows, BoundedCoordinateEstimator)
    # The discriminator every combinator uses: a mixed row model has no ``likelihood``, so
    # nothing downstream mistakes it for a penalised one.
    assert not hasattr(rows, "likelihood")
    assert hasattr(rows, "row_objective")
    # Both spell the estimated coordinate the same way, which is what makes them one idea.
    assert penalised.parameter_names[-1] == rows.parameter_names[-1] == MIXTURE_LOGIT
    assert penalised.mixture_component is not None and rows.mixture_component is not None


def reference_design(n_trials: int, *, seed: int) -> Study:
    """The design the pinned reference fits were produced on, rebuilt exactly."""

    generator = np.random.default_rng(seed)
    return Study(
        {
            "subject": ["m0"] * n_trials,
            "session": [index // 100 for index in range(n_trials)],
            "session_order": [index // 100 for index in range(n_trials)],
            "trial": list(range(n_trials)),
            "stimulus": generator.normal(size=n_trials),
        }
    )


def reference_case(name: str) -> tuple[Any, Study, dict[str, Any]]:
    """Rebuild one pinned case: the model, its study and the numbers it must reproduce."""

    record = REFERENCE[name]
    if name == "psychometric_lapse":
        model: Any = mix(Psychometric(), UniformChoiceGuess(), weight_bounds=(0.0, 0.3))
    else:
        model = mix(
            WienerDriftDiffusion(
                predictors=("stimulus",),
                nondecision_time_bounds=(0.1, 0.6),
                n_restarts=2,
                simulation_time_step=0.001,
            ),
            UniformResponseGuess(time_bounds=(0.05, 3.0)),
            n_restarts=2,
        )
    study = model.simulate(
        reference_design(record["design_rows"], seed=record["design_seed"]),
        dict(record["truth"]),
        seed=record["simulate_seed"],
    )
    return model, study, record


@pytest.mark.parametrize("name", ["psychometric_lapse", "drift_diffusion_contaminant"])
def test_the_penalised_route_reproduces_its_pinned_fits_to_the_last_bit(name: str) -> None:
    """Equality, not a tolerance: the two mixtures that existed before are the same objects.

    ``mix()`` grew a second route and a shared base class, and both of those are the kind of
    change that can move a fit by an ulp and be argued about afterwards. This asserts that
    neither moved anything at all -- the estimates, their standard errors, the whole
    covariance, the objective, the per-row scores, the responsibilities and the reported
    weight are ``==`` to the numbers the previous implementation produced.
    """

    model, study, record = reference_case(name)

    fit = model.fit(study)
    scores = model.pointwise_log_prob(study, fit)
    responsibility = model.component_responsibility(study, fit)
    probability = np.asarray(model.predict(study, fit).probability).reshape(-1)

    assert model.signature == record["signature"]
    assert list(fit.parameter_names) == record["parameter_names"]
    assert list(np.asarray(fit.estimates)) == record["estimates"]
    assert list(np.asarray(fit.standard_errors)) == record["standard_errors"]
    assert [list(row) for row in np.asarray(fit.covariance)] == record["covariance"]
    assert fit.diagnostics.objective == record["objective"]
    assert fit.diagnostics.gradient_norm == record["gradient_norm"]
    assert fit.diagnostics.boundary_estimate == record["boundary_estimate"]
    assert float(np.sum(scores)) == record["log_probability_sum"]
    assert list(scores[:20]) == record["pointwise_log_probability_head"]
    assert list(responsibility[:20]) == record["responsibility_head"]
    assert list(probability[:20]) == record["probability_head"]
    for quantity, (value, error) in record["derived"].items():
        assert fit.derived_quantities[quantity].value == value
        assert fit.derived_quantities[quantity].standard_error == error
