"""Normative belief updating, checked against closed forms rather than against itself.

Five of these tests would still pass a wrong implementation of the fits, and they are the
ones this file exists for.

The **ideal observer's posterior is closed form**, so its belief at ``retention=1`` is
:math:`(\\alpha_0 + n_1)/(\\nu + n)` exactly, and below one it is a discounted count that a
double loop can recompute without touching the recursion. Its **learning rate is a data-free
recursion** with a closed-form fixed point :math:`1/(\\nu + 1/(1-\\rho))`, so a leaky ideal
observer *is* a Rescorla-Wagner learner and the number is checkable.

The **HGF's level-two step size has a closed-form fixed point** too, and the way to see it is
to drive the filter with an observation of one half: the prediction error is then zero on
every trial, the belief never moves, and the step-size recursion runs on its own until it
reaches :func:`hgf_fixed_learning_rate` -- to twelve decimal places, from an equation derived
on paper and not from this module. That is the Rescorla-Wagner reduction, asserted where it
is exact, and the departure from it on a real binary sequence is bounded rather than glossed.

**The conventions that distinguish implementations are asserted individually**: that the
binary first level contributes a *variance* at level two, that :math:`\\kappa_2 = 0` makes the
third level inert to the first two exactly, and that the level-two update is a delta rule with
the reported learning rate. And the **analytic gradient** of both likelihoods is checked
against a central difference, because a wrong gradient converges to a wrong answer quietly.

The rest of the file is the composition claim -- that all three of ``smooth()``,
``hierarchical()`` and ``mix()`` work on a family written after they existed, without any of
them being touched -- and the identifiability evidence, which for the third level is negative
and is asserted as such.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavio import Study, run_parameter_recovery
from behavio.adapters.estimator_conformance import check_behaviour_estimator
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth
from behavio.contracts.bounded import (
    BOUNDED_COORDINATE,
    BoundedCoordinateEstimator,
    require_composable,
)
from behavio.contracts.natural import NaturalParameterisation
from behavio.models import ModelDataError
from behavio.models._kernels.bernoulli import ordered_session_indices
from behavio.models.base import PredictionMode, UnsupportedPredictionMode
from behavio.models.belief import (
    BELIEF_SENSITIVITY_FLOOR,
    VIOLATION_PENALTY,
    BeliefSoftmax,
    BetaBernoulliObserver,
    BetaBernoulliParameters,
    HierarchicalGaussianFilter,
    NegativePosteriorPrecision,
    UnitSquareSigmoid,
    beta_bernoulli_beliefs,
    hgf_beliefs,
    hgf_fixed_learning_rate,
)

# --------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------


def observation_sequence(
    *, length: int = 480, period: int | None = 20, rate: float = 0.85, seed: int = 0
) -> np.ndarray:
    """A binary sequence whose rate either reverses every ``period`` trials or does not."""

    generator = np.random.default_rng(seed)
    if period is None:
        return (generator.random(length) < rate).astype(np.float64)
    values = []
    current = 1.0 - rate
    for index in range(length):
        if index and index % period == 0:
            current = rate if current < 0.5 else 1.0 - rate
        values.append(float(generator.random() < current))
    return np.asarray(values, dtype=np.float64)


def design(
    subjects: tuple[str, ...] = ("a",),
    *,
    sessions: int = 1,
    trials: int = 480,
    period: int | None = 20,
    seed: int = 0,
) -> Study:
    """A reversal-learning design: one exogenous observation column, no responses yet."""

    columns: dict[str, list[object]] = {
        name: [] for name in ("subject", "session", "trial", "session_order", "observation")
    }
    for subject_index, subject in enumerate(subjects):
        trial = 0
        for session in range(sessions):
            values = observation_sequence(
                length=trials,
                period=period,
                seed=seed + 100 * subject_index + session,
            )
            for value in values:
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["observation"].append(int(value))
                trial += 1
    return Study(columns)


def blocks_of(study: Study) -> tuple[tuple[int, ...], ...]:
    return ordered_session_indices(study)


# --------------------------------------------------------------------------------------
# The ideal observer against its own closed form
# --------------------------------------------------------------------------------------


def test_a_lossless_observer_is_the_exact_beta_bernoulli_posterior_mean() -> None:
    """``retention=1`` is conjugate Bayes, and conjugate Bayes has a formula."""

    observations = observation_sequence(length=120, period=None, seed=3)
    trajectory = beta_bernoulli_beliefs(
        observations, retention=1.0, prior_mean=0.4, prior_strength=3.0
    )

    successes = np.concatenate([[0.0], np.cumsum(observations)[:-1]])
    expected = (3.0 * 0.4 + successes) / (3.0 + np.arange(len(observations)))

    assert np.array_equal(trajectory.belief, expected)


def test_a_leaky_observer_is_its_discounted_count_ratio() -> None:
    """The recursion is checked against the sum it is a recursion for, term by term."""

    observations = observation_sequence(length=90, period=15, seed=4)
    retention, mean, strength = 0.9, 0.4, 3.0
    trajectory = beta_bernoulli_beliefs(
        observations, retention=retention, prior_mean=mean, prior_strength=strength
    )

    expected = []
    for trial in range(len(observations)):
        weights = retention ** (trial - 1 - np.arange(trial))
        successes = float(weights @ observations[:trial])
        failures = float(weights @ (1.0 - observations[:trial]))
        expected.append((strength * mean + successes) / (strength + successes + failures))

    assert np.allclose(trajectory.belief, expected, atol=1e-12)


def test_the_observers_learning_rate_is_a_rescorla_wagner_rate_with_a_closed_form() -> None:
    """A leaky ideal observer *is* a delta rule, and the rate is a number, not an analogy.

    Two claims, both exact. The step-size recursion :math:`n_{k+1} = \\rho n_k + (1-\\rho)\\nu
    + 1` contains no observation, so two completely different sequences produce *identical*
    learning rates; and it converges to :math:`\\nu + 1/(1-\\rho)`, so the asymptotic rate is
    :math:`1/(\\nu + 1/(1-\\rho))`.
    """

    retention, strength = 0.92, 2.5
    first = beta_bernoulli_beliefs(
        observation_sequence(length=400, period=20, seed=5),
        retention=retention,
        prior_strength=strength,
    )
    second = beta_bernoulli_beliefs(
        observation_sequence(length=400, period=None, seed=6),
        retention=retention,
        prior_strength=strength,
    )

    # Equal to rounding: the two sequences accumulate the same total count in a different
    # order, so the sum of two floats differs in its last bits and the claim is not bitwise.
    assert np.allclose(first.learning_rate, second.learning_rate, rtol=0.0, atol=1e-15)

    parameters = BetaBernoulliParameters(
        retention=retention, prior_mean=0.5, prior_strength=strength
    )
    assert parameters.asymptotic_learning_rate == pytest.approx(
        1.0 / (strength + 1.0 / (1.0 - retention))
    )
    assert first.learning_rate[-1] == pytest.approx(parameters.asymptotic_learning_rate, abs=1e-9)


def test_the_observers_update_is_a_delta_rule_plus_a_decay_toward_the_prior() -> None:
    """The exact decomposition the module docstring claims, asserted term by term."""

    observations = observation_sequence(length=200, period=25, seed=7)
    retention, mean, strength = 0.88, 0.35, 4.0
    trajectory = beta_bernoulli_beliefs(
        observations, retention=retention, prior_mean=mean, prior_strength=strength
    )

    belief = trajectory.belief
    following = np.concatenate(
        [belief[1:], [trajectory.state("alpha")[-1] / np.sum(trajectory.states[-1])]]
    )
    rate = trajectory.learning_rate
    predicted = (
        belief
        + rate * (observations - belief)
        + rate * (1.0 - retention) * strength * (mean - belief)
    )

    assert np.allclose(following, predicted, atol=1e-12)


# --------------------------------------------------------------------------------------
# The HGF against the conventions implementations disagree about
# --------------------------------------------------------------------------------------


def test_the_level_two_step_size_reaches_the_closed_form_rescorla_wagner_rate() -> None:
    """The Rescorla-Wagner reduction, asserted where it is exact.

    An observation of one half makes the level-one prediction error zero on every trial, so
    the belief never moves and the Bernoulli variance stays at a quarter. The step-size
    recursion then runs alone, and what it converges to is the root of
    :math:`v\\sigma^2 + vc\\sigma - c = 0` -- an equation solved on paper. Once there the
    filter is exactly :math:`\\mu_2 \\mathrel{+}= \\sigma^{*}(u - \\hat{\\mu}_1)`.
    """

    for tonic in (-4.0, -1.0, 0.0, 2.0):
        trajectory = hgf_beliefs(np.full(600, 0.5), initial_belief=0.0, tonic_volatility=tonic)
        expected = hgf_fixed_learning_rate(tonic_volatility=tonic, belief=0.5)

        assert np.allclose(trajectory.belief, 0.5)
        assert trajectory.learning_rate[-1] == pytest.approx(expected, abs=1e-12)


def test_the_binary_filter_departs_from_a_fixed_rate_only_through_the_belief() -> None:
    """The honest qualification, bounded rather than glossed.

    On a real binary sequence the step size is not constant, because the Bernoulli variance
    of the current belief enters it. The departure is exactly the range of
    :func:`hgf_fixed_learning_rate` over the beliefs the filter visited -- so the observed
    step sizes must lie inside that interval, and the filter must still track a fixed-rate
    Rescorla-Wagner learner closely enough to be recognisable as one.
    """

    observations = observation_sequence(length=400, period=25, seed=8)
    tonic = -2.0
    trajectory = hgf_beliefs(observations, initial_belief=0.0, tonic_volatility=tonic)

    visited = np.clip(trajectory.belief, 1e-9, 1.0 - 1e-9)
    admissible = [
        hgf_fixed_learning_rate(tonic_volatility=tonic, belief=float(value))
        for value in (visited.min(), 0.5, visited.max())
    ]
    assert trajectory.learning_rate.min() >= min(admissible) - 1e-9
    assert trajectory.learning_rate.max() <= max(admissible) + 1e-9

    rate = hgf_fixed_learning_rate(tonic_volatility=tonic, belief=0.5)
    tendency = 0.0
    delta_rule = []
    for observation in observations:
        expected = 1.0 / (1.0 + np.exp(-tendency))
        delta_rule.append(expected)
        tendency += rate * (observation - expected)

    assert np.max(np.abs(trajectory.belief - np.asarray(delta_rule))) < 0.25
    assert np.mean(np.abs(trajectory.belief - np.asarray(delta_rule))) < 0.05


def test_the_level_two_update_is_a_delta_rule_with_the_reported_learning_rate() -> None:
    """``mu2 += sigma2 * (u - muhat1)`` exactly, at both level counts."""

    observations = observation_sequence(length=300, period=20, seed=9)
    for extra in ({}, {"volatility_coupling": 1.0, "meta_volatility": -4.0}):
        trajectory = hgf_beliefs(observations, initial_belief=0.3, tonic_volatility=-3.0, **extra)
        tendency = trajectory.state("tendency")
        previous = np.concatenate([[0.3], tendency[:-1]])
        step = trajectory.learning_rate * (observations - trajectory.belief)

        assert np.allclose(tendency - previous, step, atol=1e-12)


def test_the_binary_first_level_contributes_a_variance_and_not_a_precision() -> None:
    """The one line of the binary HGF a reader is most likely to get backwards.

    ``pi2 = pihat2 + muhat1 * (1 - muhat1)``. The trajectory reports ``pi2`` and ``sigma2``,
    and ``pihat2`` is recoverable from the previous trial's ``sigma2`` -- so the identity can
    be recomputed from the outside. Adding ``1 / (muhat1 * (1 - muhat1))`` instead, the
    reading the notation invites, disagrees by more than a factor of ten here.
    """

    observations = observation_sequence(length=200, period=20, seed=10)
    tonic = -2.5
    trajectory = hgf_beliefs(observations, initial_belief=0.0, tonic_volatility=tonic)

    posterior_variance = trajectory.learning_rate
    prior_variance = np.concatenate([[1.0], posterior_variance[:-1]]) + np.exp(tonic)
    variance = trajectory.belief * (1.0 - trajectory.belief)

    assert np.allclose(
        trajectory.state("tendency_precision"), 1.0 / prior_variance + variance, atol=1e-12
    )
    wrong = 1.0 / prior_variance + 1.0 / variance
    assert np.max(np.abs(wrong - trajectory.state("tendency_precision"))) > 10.0


def test_zero_coupling_makes_the_third_level_inert_to_the_first_two() -> None:
    """ "Volatility held constant" has an exact meaning, and this is it."""

    observations = observation_sequence(length=300, period=20, seed=11)
    two = hgf_beliefs(observations, initial_belief=0.2, tonic_volatility=-3.0)
    three = hgf_beliefs(
        observations,
        initial_belief=0.2,
        tonic_volatility=-3.0,
        volatility_coupling=1e-14,
        meta_volatility=-2.0,
    )

    assert np.array_equal(two.belief, three.belief)
    assert np.array_equal(two.learning_rate, three.learning_rate)


def test_a_very_volatile_filter_believes_the_last_observation() -> None:
    """The other limit with a known answer: no memory at all."""

    observations = observation_sequence(length=200, period=20, seed=12)
    trajectory = hgf_beliefs(observations, initial_belief=0.0, tonic_volatility=4.0)

    assert np.mean(np.abs(trajectory.belief[1:] - observations[:-1])) < 0.05


# --------------------------------------------------------------------------------------
# The third level's precision, which can leave the positive reals
# --------------------------------------------------------------------------------------


def unstable_arguments() -> dict[str, float]:
    """Parameters that drive the third level's posterior precision through zero."""

    return {
        "initial_belief": 0.0,
        "tonic_volatility": 3.0,
        "volatility_coupling": 5.0,
        "meta_volatility": 1.5,
    }


def test_a_negative_posterior_precision_is_refused_and_named() -> None:
    observations = observation_sequence(length=200, period=8, seed=13)

    with pytest.raises(NegativePosteriorPrecision, match="positive reals at trial"):
        hgf_beliefs(observations, **unstable_arguments())


def test_the_stability_of_a_parameter_vector_can_be_asked_without_raising() -> None:
    """A grid sweep should be able to find the edge of the admissible region, not hit it."""

    study = design(trials=200, period=8, seed=13)
    model = HierarchicalGaussianFilter(
        levels=3, volatility_coupling=None, response=UnitSquareSigmoid()
    )
    responses = model.simulate(
        study,
        model.parameters_from_components(
            tonic_volatility=-3.0,
            volatility_coupling=1.0,
            meta_volatility=-4.0,
            decision_noise=3.0,
        ),
        seed=1,
    )

    good = model.volatility_stability(
        responses,
        model.parameters_from_components(
            tonic_volatility=-3.0,
            volatility_coupling=1.0,
            meta_volatility=-4.0,
            decision_noise=3.0,
        ),
    )
    bad = model.volatility_stability(
        responses,
        model.parameters_from_components(
            tonic_volatility=3.0,
            volatility_coupling=5.0,
            meta_volatility=1.5,
            decision_noise=3.0,
        ),
    )

    assert good.admissible and good.first_violation is None
    assert good.minimum_precision > 0.0
    assert not bad.admissible and bad.first_violation is not None


def test_an_admissible_fit_reports_the_margin_it_succeeded_by() -> None:
    study = design(trials=240, period=20, seed=14)
    model = HierarchicalGaussianFilter(levels=3, response=UnitSquareSigmoid())
    responses = model.simulate(
        study,
        model.parameters_from_components(
            tonic_volatility=-3.0, meta_volatility=-4.0, decision_noise=3.0
        ),
        seed=2,
    )

    fit = model.fit(responses)

    assert fit.derived_value("minimum_volatility_precision") > 0.0
    assert fit.diagnostics.objective < VIOLATION_PENALTY * len(responses)


def test_a_fit_with_no_admissible_restart_is_refused_rather_than_reported() -> None:
    """Every restart declared inadmissible: no number is preferable to a number from outside.

    Forced by declaring the very parameters that violate, so that the model has nothing left
    to search but the region its own assumptions exclude.
    """

    study = design(trials=200, period=8, seed=13)
    model = HierarchicalGaussianFilter(
        levels=3, response=UnitSquareSigmoid(), **unstable_arguments()
    )
    responses = Study(
        {
            **{name: study[name] for name in study.columns},
            "choice": np.asarray(study["observation"], dtype=np.int8),
        }
    )

    with pytest.raises(NegativePosteriorPrecision, match="came to rest"):
        model.fit(responses)


# --------------------------------------------------------------------------------------
# The gradient
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        BetaBernoulliObserver(response=BeliefSoftmax(include_bias=True)),
        BetaBernoulliObserver(response=UnitSquareSigmoid(), prior_mean=None, prior_strength=None),
        HierarchicalGaussianFilter(levels=2, response=UnitSquareSigmoid()),
        HierarchicalGaussianFilter(
            levels=3,
            response=BeliefSoftmax(),
            initial_belief=None,
            volatility_coupling=None,
        ),
    ],
    ids=["observer-softmax", "observer-free-prior", "hgf-two", "hgf-three-free-kappa"],
)
def test_the_analytic_gradient_matches_a_central_difference(model) -> None:
    """A wrong gradient converges to a wrong answer quietly, so it is differenced."""

    study = design(trials=200, period=20, seed=15)
    point = model.initial_points(study)[0]
    responses = model.simulate(study, dict(zip(model.parameter_names, point, strict=True)), seed=3)
    outcomes = model.outcomes(responses)

    value, gradient = model._objective(point, responses, outcomes)
    numeric = np.empty_like(gradient)
    for index in range(len(point)):
        step = 1e-6 * (1.0 + abs(float(point[index])))
        upper = np.array(point, dtype=np.float64)
        upper[index] += step
        lower = np.array(point, dtype=np.float64)
        lower[index] -= step
        numeric[index] = (
            model._objective(upper, responses, outcomes)[0]
            - model._objective(lower, responses, outcomes)[0]
        ) / (2.0 * step)

    assert np.isfinite(value)
    assert np.allclose(gradient, numeric, rtol=2e-4, atol=2e-5)


# --------------------------------------------------------------------------------------
# The perceptual model and the response model are separate
# --------------------------------------------------------------------------------------


def test_the_unit_square_sigmoid_at_unit_noise_is_probability_matching() -> None:
    """``zeta = 1`` returns the belief itself, which is the check the algebra promises."""

    study = design(trials=120, period=20, seed=16)
    model = HierarchicalGaussianFilter(levels=2, response=UnitSquareSigmoid())
    parameters = model.parameters_from_components(tonic_volatility=-3.0, decision_noise=1.0)

    belief = model.belief_trajectory(study, parameters).belief
    probability = model.response_probability(study, parameters)

    assert np.allclose(probability, belief, atol=1e-9)


def test_one_perceptual_model_carries_two_response_models() -> None:
    """The field's own split, made a component: the belief is identical, the readout is not."""

    study = design(trials=200, period=20, seed=17)
    softmax = BetaBernoulliObserver(response=BeliefSoftmax())
    sigmoid = BetaBernoulliObserver(response=UnitSquareSigmoid())

    assert softmax.parameter_names == ("retention_logit", "inverse_temperature_log")
    assert sigmoid.parameter_names == ("retention_logit", "decision_noise_log")
    assert softmax.signature != sigmoid.signature

    first = softmax.belief_trajectory(
        study, softmax.parameters_from_components(retention=0.9, inverse_temperature=5.0)
    )
    second = sigmoid.belief_trajectory(
        study, sigmoid.parameters_from_components(retention=0.9, decision_noise=5.0)
    )

    assert np.array_equal(first.belief, second.belief)


def test_a_declared_parameter_leaves_the_model_rather_than_being_pinned_inside_it() -> None:
    study = design(trials=200, period=20, seed=18)
    declared = HierarchicalGaussianFilter(levels=3, response=UnitSquareSigmoid())
    freed = HierarchicalGaussianFilter(
        levels=3, response=UnitSquareSigmoid(), volatility_coupling=None
    )

    assert "volatility_coupling_log" not in declared.parameter_names
    assert "volatility_coupling_log" in freed.parameter_names
    assert declared.coordinate_box(study).shape[0] == len(declared.parameter_names)
    assert "volatility_coupling=1" in declared.signature
    assert declared.parameter_components(
        declared.parameters_from_components(
            tonic_volatility=-3.0, meta_volatility=-4.0, decision_noise=2.0
        )
    ).volatility_coupling == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# The estimator contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        BetaBernoulliObserver(response=BeliefSoftmax()),
        HierarchicalGaussianFilter(levels=2, response=UnitSquareSigmoid()),
        HierarchicalGaussianFilter(levels=3, response=UnitSquareSigmoid()),
    ],
    ids=["observer", "hgf-two", "hgf-three"],
)
def test_a_belief_model_satisfies_the_filtered_estimator_contract(model) -> None:
    study = design(trials=150, period=20, seed=19)
    point = model.initial_points(study)[0]
    responses = model.simulate(study, dict(zip(model.parameter_names, point, strict=True)), seed=4)

    report = check_behaviour_estimator(model, responses)

    assert report.passed, report.summary()
    assert isinstance(model, BoundedCoordinateEstimator)
    assert isinstance(model, NaturalParameterisation)
    assert require_composable(model, combinator="smooth") == BOUNDED_COORDINATE


def test_the_natural_coordinate_round_trips_and_its_jacobian_is_the_derivative() -> None:
    model = BetaBernoulliObserver(
        response=BeliefSoftmax(include_bias=True), prior_mean=None, prior_strength=None
    )
    natural = {
        "retention": 0.9,
        "prior_mean": 0.4,
        "prior_strength": 3.0,
        "inverse_temperature": 4.0,
        "choice_bias": -0.5,
    }
    estimated = model.from_natural(natural)
    vector = np.asarray([estimated[name] for name in model.parameter_names])

    assert dict(model.to_natural(vector)) == pytest.approx(natural)

    jacobian = model.natural_jacobian(vector)
    numeric = np.empty_like(jacobian)
    for index in range(len(vector)):
        step = 1e-6
        upper = np.array(vector)
        upper[index] += step
        lower = np.array(vector)
        lower[index] -= step
        high = np.asarray(list(model.to_natural(upper).values()))
        low = np.asarray(list(model.to_natural(lower).values()))
        numeric[:, index] = (high - low) / (2.0 * step)

    assert np.allclose(jacobian, numeric, atol=1e-6)


def test_a_fit_from_another_specification_is_refused() -> None:
    study = design(trials=120, period=20, seed=20)
    first = BetaBernoulliObserver(response=BeliefSoftmax())
    second = BetaBernoulliObserver(response=UnitSquareSigmoid())
    responses = first.simulate(
        study, first.parameters_from_components(retention=0.9, inverse_temperature=5.0), seed=5
    )
    fit = first.fit(responses)

    with pytest.raises(ValueError, match="different model specification"):
        second.predict(responses, fit)
    with pytest.raises(UnsupportedPredictionMode):
        first.predict(responses, fit, mode=PredictionMode.SMOOTHED)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"levels": 4}, "levels must be two or three"),
        ({"levels": 2, "meta_volatility": -3.0}, "two-level filter has no third level"),
        ({"volatility_coupling": -1.0}, "must be positive"),
        ({"outcome": "observation"}, "must be distinct"),
        ({"initial_variance": 0.0}, "finite and positive"),
    ],
)
def test_an_inadmissible_specification_is_refused_at_construction(arguments, message) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        HierarchicalGaussianFilter(**arguments)


def test_a_study_without_the_observation_column_is_refused() -> None:
    model = BetaBernoulliObserver(response=BeliefSoftmax())
    study = Study(
        {
            "subject": ["a"] * 4,
            "session": ["a-0"] * 4,
            "trial": [0, 1, 2, 3],
            "session_order": [0] * 4,
            "choice": [0, 1, 0, 1],
        }
    )

    with pytest.raises(ModelDataError, match="observation"):
        model.fit(study)


# --------------------------------------------------------------------------------------
# Recovery, which is design-specific evidence and is reported as such
# --------------------------------------------------------------------------------------


def test_the_ideal_observers_leak_and_temperature_recover() -> None:
    study = design(trials=480, period=20, seed=21)
    model = BetaBernoulliObserver(response=BeliefSoftmax())
    truth = model.parameters_from_components(retention=0.9, inverse_temperature=6.0)

    report = run_parameter_recovery(model, study, [dict(truth)], repeats=3, seed=101)
    summaries = {summary.parameter: summary for summary in report.summary()}

    assert summaries["retention_logit"].n_successful == 3
    assert abs(summaries["retention_logit"].bias) < 0.4
    assert summaries["retention_logit"].rmse < 0.6
    assert abs(summaries["inverse_temperature_log"].bias) < 0.2


def test_the_two_level_filters_volatility_and_decision_noise_recover() -> None:
    study = design(trials=480, period=20, seed=22)
    model = HierarchicalGaussianFilter(levels=2, response=UnitSquareSigmoid())
    truth = model.parameters_from_components(tonic_volatility=-3.0, decision_noise=4.0)

    report = run_parameter_recovery(model, study, [dict(truth)], repeats=3, seed=102)
    summaries = {summary.parameter: summary for summary in report.summary()}

    assert summaries["tonic_volatility"].n_successful == 3
    assert abs(summaries["tonic_volatility"].bias) < 0.5
    assert summaries["tonic_volatility"].rmse < 0.6
    assert abs(summaries["decision_noise_log"].bias) < 0.3


def test_the_third_levels_volatility_does_not_recover_and_describe_said_so_first() -> None:
    """The negative result, asserted, because a plausible number here would be worse.

    :math:`\\omega_3` is not identified by binary responses on any observation sequence a
    reversal-learning design produces: displacing it by a factor of :math:`e` moves the whole
    belief vector by under a tenth, which no set of responses can reveal. So the pre-fit
    finding fires, and the recovery it predicts fails -- while the two parameters the same
    study *can* see recover in the same run.
    """

    study = design(trials=480, period=20, seed=23)
    model = HierarchicalGaussianFilter(levels=3, response=UnitSquareSigmoid())
    truth = model.parameters_from_components(
        tonic_volatility=-3.0, meta_volatility=-4.0, decision_noise=4.0
    )
    responses = model.simulate(study, truth, seed=6)

    sensitivity = model.belief_sensitivity(model._observations(responses), blocks_of(responses))
    assert sensitivity["meta_volatility"] < BELIEF_SENSITIVITY_FLOOR
    assert sensitivity["tonic_volatility"] > BELIEF_SENSITIVITY_FLOOR
    codes = [finding.code for finding in model.describe(responses).findings]
    assert "belief_insensitive_parameter" in codes

    report = run_parameter_recovery(model, study, [dict(truth)], repeats=3, seed=103)
    summaries = {summary.parameter: summary for summary in report.summary()}

    assert abs(summaries["tonic_volatility"].bias) < 0.5
    assert abs(summaries["decision_noise_log"].bias) < 0.3
    assert summaries["meta_volatility"].rmse > 1.0


# --------------------------------------------------------------------------------------
# Identifiability hazards, surfaced before fitting
# --------------------------------------------------------------------------------------


def responses_for(model, study: Study, *, seed: int = 7, **natural: float) -> Study:
    return model.simulate(study, model.parameters_from_components(**natural), seed=seed)


def test_a_stationary_design_cannot_identify_a_leak_or_a_volatility() -> None:
    study = design(trials=480, period=None, seed=24)
    observer = BetaBernoulliObserver(response=BeliefSoftmax())
    responses = responses_for(observer, study, retention=0.9, inverse_temperature=5.0)

    codes = [finding.code for finding in observer.describe(responses).findings]
    assert "stationary_observations" in codes

    filter_model = HierarchicalGaussianFilter(levels=3, response=UnitSquareSigmoid())
    filter_responses = responses_for(
        filter_model, study, tonic_volatility=-3.0, meta_volatility=-4.0, decision_noise=3.0
    )
    filter_codes = [finding.code for finding in filter_model.describe(filter_responses).findings]
    assert "stationary_observations" in filter_codes


def test_a_reversing_design_is_not_called_stationary() -> None:
    """The statistic has to survive its own false-positive case as well as its true one."""

    study = design(trials=480, period=16, seed=25)
    observer = BetaBernoulliObserver(response=BeliefSoftmax())
    responses = responses_for(observer, study, retention=0.9, inverse_temperature=5.0)

    codes = [finding.code for finding in observer.describe(responses).findings]
    assert "stationary_observations" not in codes


def test_a_freed_coupling_reports_the_ridge_it_shares_with_the_meta_volatility() -> None:
    study = design(trials=240, period=20, seed=26)
    model = HierarchicalGaussianFilter(
        levels=3, response=UnitSquareSigmoid(), volatility_coupling=None
    )
    responses = responses_for(
        model,
        study,
        tonic_volatility=-3.0,
        volatility_coupling=1.0,
        meta_volatility=-4.0,
        decision_noise=3.0,
    )

    codes = [finding.code for finding in model.describe(responses).findings]
    assert "coupled_volatility_scale" in codes

    correlation = model.coupling_volatility_correlation(model.fit(responses))
    assert np.isnan(correlation) or -1.0 <= correlation <= 1.0


def test_a_long_block_cannot_identify_the_beta_prior() -> None:
    study = design(trials=480, period=20, seed=27)
    model = BetaBernoulliObserver(response=BeliefSoftmax(), prior_mean=None, prior_strength=None)
    responses = responses_for(
        model,
        study,
        retention=0.9,
        prior_mean=0.5,
        prior_strength=2.0,
        inverse_temperature=5.0,
    )

    codes = [finding.code for finding in model.describe(responses).findings]
    assert "washed_out_prior" in codes


def test_a_degenerate_observation_sequence_and_a_constant_response_are_reported() -> None:
    columns: dict[str, list[object]] = {
        "subject": ["a"] * 40,
        "session": ["a-0"] * 40,
        "trial": list(range(40)),
        "session_order": [0] * 40,
        "observation": [1] * 40,
        "choice": [1] * 40,
    }
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    codes = [finding.code for finding in model.describe(Study(columns)).findings]

    assert "degenerate_observations" in codes
    assert "constant_response" in codes


# --------------------------------------------------------------------------------------
# The three combinators, untouched
# --------------------------------------------------------------------------------------


def composed_study(seed: int = 30) -> Study:
    study = design(("a", "b"), sessions=2, trials=120, period=20, seed=seed)
    model = BetaBernoulliObserver(response=BeliefSoftmax())
    return model.simulate(
        study,
        model.parameters_from_components(retention=0.85, inverse_temperature=5.0),
        seed=seed,
    )


def test_hierarchical_pools_a_perceptual_parameter_across_subjects() -> None:
    responses = composed_study()
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    composed = hierarchical(model, over="subject", parameters=("retention_logit",))
    fit = composed.fit(responses)

    assert composed.parameter_names == model.parameter_names
    assert fit.diagnostics.converged
    assert len(fit.group_deviations) == 2


def test_smooth_gives_a_perceptual_parameter_a_path_over_sessions() -> None:
    responses = composed_study(seed=31)
    subject = Study(
        {
            name: np.asarray(responses[name])[np.asarray(responses["subject"]) == "a"]
            for name in responses.columns
        }
    )
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    composed = smooth(
        model, over="session_order", knots=(0.0, 1.0), parameters=("retention_logit",)
    )
    fit = composed.fit(subject)

    assert composed.parameter_names == (
        "retention_logit[session_order=0]",
        "retention_logit[session_order=1]",
        "inverse_temperature_log",
    )
    assert fit.diagnostics.converged


def test_mix_puts_a_lapse_on_an_observer_where_it_cannot_go_on_an_agent() -> None:
    """The structural difference from the reinforcement-learning families, asserted.

    An agent's value trace is written by the agent's own action, so no row has a density of
    its own and ``mix()`` refuses it. An observer's belief trajectory is written by the task's
    observations, so every row does, and the lapse leaves the perceptual model untouched --
    which is what makes this the right place for it rather than a convenience.
    """

    responses = composed_study(seed=32)
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    composed = mix(model, UniformChoiceGuess(), weight_bounds=(0.0, 0.25))
    fit = composed.fit(responses)

    assert composed.parameter_names == (
        "retention_logit",
        "inverse_temperature_log",
        "mixture_logit",
    )
    assert 0.0 <= fit.derived_value("lapse_rate") <= 0.25

    belief = model.belief_trajectory(
        responses, model.parameters_from_components(retention=0.85, inverse_temperature=5.0)
    )
    lapsed = model.belief_trajectory(
        Study(
            {
                **{name: responses[name] for name in responses.columns},
                "choice": 1 - np.asarray(responses["choice"], dtype=np.int8),
            }
        ),
        model.parameters_from_components(retention=0.85, inverse_temperature=5.0),
    )
    assert np.array_equal(belief.belief, lapsed.belief)


def test_the_three_combinators_stack_without_any_of_them_being_touched() -> None:
    responses = composed_study(seed=33)
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    composed = hierarchical(
        smooth(
            mix(model, UniformChoiceGuess()),
            over="session_order",
            knots=(0.0, 1.0),
            parameters=("retention_logit",),
        ),
        over="subject",
        parameters=("retention_logit",),
    )
    fit = composed.fit(responses)

    assert composed.parameter_names == (
        "retention_logit[session_order=0]",
        "retention_logit[session_order=1]",
        "inverse_temperature_log",
        "mixture_logit",
    )
    assert fit.diagnostics.converged


def test_the_filter_composes_the_same_way_the_observer_does() -> None:
    study = design(("a", "b"), sessions=2, trials=120, period=20, seed=34)
    model = HierarchicalGaussianFilter(levels=2, response=UnitSquareSigmoid())
    responses = model.simulate(
        study,
        model.parameters_from_components(tonic_volatility=-3.0, decision_noise=4.0),
        seed=8,
    )

    pooled = hierarchical(model, over="subject", parameters=("tonic_volatility",)).fit(responses)
    lapsed = mix(model, UniformChoiceGuess()).fit(responses)

    assert pooled.diagnostics.converged
    assert 0.0 <= lapsed.derived_value("lapse_rate") <= 0.25


# --------------------------------------------------------------------------------------
# The block rule, which cuts finer here than it does for a value-updating agent
# --------------------------------------------------------------------------------------


def test_a_perceptual_parameter_may_not_follow_a_within_session_clock() -> None:
    """The half of the block rule that survives: a belief trace stays unambiguous."""

    responses = composed_study(seed=35)
    subject = Study(
        {
            name: np.asarray(responses[name])[np.asarray(responses["subject"]) == "a"]
            for name in responses.columns
        }
    )
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    composed = smooth(model, over="trial", knots=(0.0, 239.0), parameters=("retention_logit",))

    with pytest.raises(ValueError, match="perceptual parameters must be constant"):
        composed.fit(subject)


def test_a_response_parameter_may_follow_a_within_session_clock() -> None:
    """The half that does not, and the reason ``row_blocks`` reports one row per block.

    A decision noise that drifts within a session leaves no ambiguity at all: the belief
    trajectory that trial's response is read against was written by the observations and is
    the same whatever the response model did. A value-updating agent has no such parameter.
    """

    responses = composed_study(seed=36)
    subject = Study(
        {
            name: np.asarray(responses[name])[np.asarray(responses["subject"]) == "a"]
            for name in responses.columns
        }
    )
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    composed = smooth(
        model, over="trial", knots=(0.0, 239.0), parameters=("inverse_temperature_log",)
    )
    fit = composed.fit(subject)

    assert composed.parameter_names == (
        "retention_logit",
        "inverse_temperature_log[trial=0]",
        "inverse_temperature_log[trial=239]",
    )
    assert fit.diagnostics.converged


def test_every_row_is_its_own_density_block() -> None:
    responses = composed_study(seed=37)
    model = BetaBernoulliObserver(response=BeliefSoftmax())

    objective = model.row_objective(responses)

    assert np.array_equal(objective.row_blocks, np.arange(len(responses)))
    assert objective.n_parameters == len(model.parameter_names)
