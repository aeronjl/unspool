"""The psychometric family: link identities, natural thresholds, and benchmark parity.

Each link is checked against a closed form written out independently of the module's own
link table, the analytic gradient is checked against central differences, and the erf
two-gamma curve is checked against the independent implementation committed beside the IBL
2021 benchmark. That benchmark deliberately keeps its own copy of the equation together
with the released Nelder-Mead restart schedule; this test is what stops the two drifting.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf, expit, ndtr

from behavio import (
    Study,
    UniformChoiceGuess,
    mix,
    model_capabilities,
    run_parameter_recovery,
)
from behavio.contracts import (
    BehaviourModel,
    DerivedQuantity,
    ModelDataError,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models import (
    Psychometric,
    PsychometricFitResult,
    PsychometricFunction,
    PsychometricLink,
    PsychometricParameters,
    erf_two_gamma_probability,
)
from benchmarks.ibl2021_psychometrics.psychometric import erf_psycho_2gammas

LINEAR_LINKS = (
    PsychometricLink.LOGISTIC,
    PsychometricLink.GAUSS,
    PsychometricLink.ERF,
    PsychometricLink.GUMBEL,
)


def design(n_trials: int = 2_400, *, low: float = -3.0, high: float = 3.0) -> Study:
    levels = np.linspace(low, high, 12)
    return Study(
        {
            "subject": ["a"] * n_trials,
            "session": ["s"] * n_trials,
            "trial": list(range(n_trials)),
            "session_order": [0] * n_trials,
            "stimulus": np.resize(levels, n_trials),
        }
    )


@pytest.mark.parametrize("link", list(PsychometricLink))
def test_the_threshold_is_the_stimulus_at_the_curve_midpoint(link: PsychometricLink) -> None:
    model = PsychometricFunction(link=link)
    threshold = 2.0 if link is PsychometricLink.WEIBULL else 0.4
    components = PsychometricParameters(
        threshold=threshold, width=1.3, guess_rate=0.04, lapse_rate=0.07
    )

    at_threshold = float(model.probability(np.asarray([threshold]), components)[0])

    assert at_threshold == pytest.approx(0.5 * (0.04 + (1.0 - 0.07)))


@pytest.mark.parametrize("link", list(PsychometricLink))
def test_every_link_is_monotone_and_saturates_at_its_declared_rates(
    link: PsychometricLink,
) -> None:
    model = PsychometricFunction(link=link)
    threshold = 2.0 if link is PsychometricLink.WEIBULL else 0.0
    grid = (
        np.geomspace(1e-8, 400.0, 400)
        if link is PsychometricLink.WEIBULL
        else np.linspace(-40.0, 40.0, 400)
    )
    components = PsychometricParameters(
        threshold=threshold, width=1.0, guess_rate=0.05, lapse_rate=0.1
    )

    curve = model.probability(grid, components)
    near_threshold = model.probability(
        np.linspace(0.5 * threshold, 2.0 * threshold, 50)
        if link is PsychometricLink.WEIBULL
        else np.linspace(-1.0, 1.0, 50),
        components,
    )

    assert np.all(np.diff(curve) >= 0)
    assert np.all(np.diff(near_threshold) > 0)
    assert curve[0] == pytest.approx(0.05, abs=1e-6)
    assert curve[-1] == pytest.approx(0.9, abs=1e-6)


def test_link_closed_forms_match_independent_expressions() -> None:
    grid = np.linspace(-4.0, 4.0, 21)
    positive = np.linspace(0.1, 8.0, 21)
    components = PsychometricParameters(threshold=0.3, width=1.4, guess_rate=0.02, lapse_rate=0.06)
    span = 1.0 - 0.02 - 0.06
    standardized = (grid - 0.3) / 1.4

    logistic = PsychometricFunction(link=PsychometricLink.LOGISTIC)
    gauss = PsychometricFunction(link=PsychometricLink.GAUSS)
    erf_link = PsychometricFunction(link=PsychometricLink.ERF)
    gumbel = PsychometricFunction(link=PsychometricLink.GUMBEL)
    weibull = PsychometricFunction(link=PsychometricLink.WEIBULL)

    assert np.allclose(logistic.probability(grid, components), 0.02 + span * expit(standardized))
    assert np.allclose(gauss.probability(grid, components), 0.02 + span * ndtr(standardized))
    assert np.allclose(
        erf_link.probability(grid, components), 0.02 + span * (erf(standardized) + 1.0) / 2.0
    )
    # Gumbel with the half-point shift that puts the threshold at 50 % of the link.
    shifted = standardized + np.log(np.log(2.0))
    assert np.allclose(
        gumbel.probability(grid, components), 0.02 + span * (1.0 - np.exp(-np.exp(shifted)))
    )
    # The 50 %-referenced Weibull: 1 - 2 ** -(x / threshold) ** (1 / width).
    weibull_components = PsychometricParameters(
        threshold=2.0, width=0.5, guess_rate=0.02, lapse_rate=0.06
    )
    expected = 0.02 + span * (1.0 - 2.0 ** -((positive / 2.0) ** 2.0))
    assert np.allclose(weibull.probability(positive, weibull_components), expected)


def test_the_erf_link_is_the_gauss_link_with_a_width_scaled_by_root_two() -> None:
    grid = np.linspace(-5.0, 5.0, 41)
    erf_link = PsychometricFunction(link=PsychometricLink.ERF)
    gauss = PsychometricFunction(link=PsychometricLink.GAUSS)

    erf_curve = erf_link.probability(
        grid, PsychometricParameters(threshold=0.2, width=1.0, guess_rate=0.03, lapse_rate=0.04)
    )
    gauss_curve = gauss.probability(
        grid,
        PsychometricParameters(
            threshold=0.2, width=1.0 / np.sqrt(2.0), guess_rate=0.03, lapse_rate=0.04
        ),
    )

    assert np.allclose(erf_curve, gauss_curve)


@pytest.mark.parametrize("link", list(PsychometricLink))
def test_the_analytic_gradient_matches_central_differences(link: PsychometricLink) -> None:
    low, high = (0.1, 10.0) if link is PsychometricLink.WEIBULL else (-3.0, 3.0)
    model = PsychometricFunction(link=link)
    threshold = 2.0 if link is PsychometricLink.WEIBULL else 0.3
    parameters = model.parameters_from_components(
        threshold=threshold, width=1.0, guess_rate=0.03, lapse_rate=0.05
    )
    study = model.simulate(design(1_200, low=low, high=high), parameters, seed=4)
    values = model._transform(model._stimulus(study))
    outcomes = model._outcomes(study)
    coordinate = np.asarray([parameters[name] for name in model.parameter_names])

    _, gradient = model._objective(coordinate, values, outcomes)
    numerical = np.zeros_like(gradient)
    for index in range(len(coordinate)):
        step = np.zeros_like(coordinate)
        step[index] = 1e-6
        numerical[index] = (
            model._objective(coordinate + step, values, outcomes)[0]
            - model._objective(coordinate - step, values, outcomes)[0]
        ) / 2e-6

    assert np.allclose(gradient, numerical, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("link", LINEAR_LINKS)
def test_a_fit_reports_a_natural_threshold_with_an_interval_covering_the_truth(
    link: PsychometricLink,
) -> None:
    model = PsychometricFunction(link=link)
    parameters = model.parameters_from_components(
        threshold=0.4, width=0.9, guess_rate=0.03, lapse_rate=0.05
    )
    study = model.simulate(design(6_000), parameters, seed=17)

    fit = model.fit(study)
    summary = model.summarize(fit)

    assert isinstance(model, BehaviourModel)
    assert isinstance(fit, PsychometricFitResult)
    assert model.parameter_names[0] == "threshold"
    assert fit.parameters["threshold"] == pytest.approx(summary.threshold)
    assert fit.standard_error_map["threshold"] == pytest.approx(summary.threshold_standard_error)
    assert summary.threshold == pytest.approx(0.4, abs=0.2)
    assert summary.threshold_interval[0] < 0.4 < summary.threshold_interval[1]
    assert 0 < summary.width_interval[0] < summary.width < summary.width_interval[1]
    assert 0 < summary.guess_rate_interval[0] < summary.guess_rate_interval[1] < 0.2
    assert 0 < summary.lapse_rate_interval[0] < summary.lapse_rate_interval[1] < 0.2
    assert summary.slope_at_threshold > 0
    assert summary.interval_level == 0.95
    # ``PsychometricFitResult.link`` is gone: the link is a declared configuration and is
    # already spelled out by the fit's own identity, so the subclass was duplicating it.
    assert fit.model_name == f"psychometric-{link.value}"
    assert model.summarize(fit).link is link


def test_the_weibull_threshold_is_estimated_on_a_log_coordinate() -> None:
    model = PsychometricFunction(link=PsychometricLink.WEIBULL)
    parameters = model.parameters_from_components(
        threshold=3.0, width=0.5, guess_rate=0.03, lapse_rate=0.04
    )
    study = model.simulate(design(6_000, low=0.2, high=12.0), parameters, seed=21)

    fit = model.fit(study)
    summary = model.summarize(fit)

    assert model.parameter_names[0] == "log_threshold"
    assert fit.parameters["log_threshold"] == pytest.approx(np.log(summary.threshold))
    assert summary.threshold == pytest.approx(3.0, rel=0.2)
    assert summary.threshold_interval[0] > 0
    assert summary.threshold_interval[0] < 3.0 < summary.threshold_interval[1]


def test_a_fixed_rate_leaves_the_parameter_vector_entirely() -> None:
    model = PsychometricFunction(
        link=PsychometricLink.GAUSS, fixed_guess_rate=0.5, maximum_lapse=0.2
    )
    parameters = model.parameters_from_components(threshold=0.2, width=1.0, lapse_rate=0.05)
    study = model.simulate(design(4_000), parameters, seed=9)

    fit = model.fit(study)
    summary = model.summarize(fit)

    assert model.parameter_names == ("threshold", "log_width", "lapse_logit")
    assert "guess_logit" not in parameters
    assert summary.guess_rate == 0.5
    assert summary.guess_rate_is_fixed is True
    assert summary.guess_rate_standard_error == 0.0
    assert summary.guess_rate_interval == (0.5, 0.5)
    assert summary.lapse_rate_is_fixed is False
    assert model.predict(study, fit).probability.min() >= 0.5 - 1e-9
    with pytest.raises(ValueError, match=r"fixed at 0\.5"):
        model.parameters_from_components(threshold=0.2, width=1.0, guess_rate=0.1, lapse_rate=0.05)


def test_asymmetric_guess_and_lapse_rates_are_separately_identified() -> None:
    model = PsychometricFunction(link=PsychometricLink.GAUSS, maximum_guess=0.3, maximum_lapse=0.3)
    parameters = model.parameters_from_components(
        threshold=0.0, width=0.7, guess_rate=0.02, lapse_rate=0.18
    )
    study = model.simulate(design(20_000, low=-6.0, high=6.0), parameters, seed=33)

    summary = model.summarize(model.fit(study))

    assert summary.guess_rate == pytest.approx(0.02, abs=0.02)
    assert summary.lapse_rate == pytest.approx(0.18, abs=0.03)
    assert summary.lapse_rate > summary.guess_rate


def test_the_psychometric_family_satisfies_the_prediction_and_scoring_contract() -> None:
    model = PsychometricFunction(link=PsychometricLink.LOGISTIC)
    parameters = model.parameters_from_components(
        threshold=0.1, width=1.0, guess_rate=0.04, lapse_rate=0.04
    )
    study = model.simulate(design(1_200), parameters, seed=2)
    fit = model.fit(study)

    prediction = model.predict(study, fit)
    scores = model.pointwise_log_prob(study, fit)
    outcomes = np.asarray(study["choice"], dtype=np.float64)

    assert model_capabilities(model).can_recover_parameters
    assert model_capabilities(model).scored_columns == ("choice",)
    assert prediction.mode is PredictionMode.FILTERED
    assert np.allclose(
        scores,
        outcomes * np.log(prediction.probability)
        + (1.0 - outcomes) * np.log1p(-prediction.probability),
    )
    assert fit.audit().model_name == model.model_name
    assert fit.diagnostics.gradient_norm == pytest.approx(0.0, abs=1e-2)
    assert len(fit.restart_objectives) == model.n_restarts
    assert 0 <= fit.selected_restart < model.n_restarts

    with pytest.raises(UnsupportedPredictionMode):
        model.predict(study, fit, mode=PredictionMode.SMOOTHED)


def test_the_psychometric_family_recovers_its_parameters() -> None:
    model = PsychometricFunction(link=PsychometricLink.GAUSS)
    truth = dict(
        model.parameters_from_components(threshold=0.3, width=1.0, guess_rate=0.03, lapse_rate=0.05)
    )

    report = run_parameter_recovery(model, design(4_800), [truth], repeats=3, seed=5)

    assert report.model_name == model.model_name
    assert np.all(report.converged)
    assert report.estimates[:, 0].mean() == pytest.approx(0.3, abs=0.15)
    assert np.exp(report.estimates[:, 1]).mean() == pytest.approx(1.0, abs=0.3)


def test_the_promoted_erf_two_gamma_curve_matches_the_benchmark_implementation() -> None:
    """Parity with the deliberately independent implementation beside the IBL benchmark."""

    generator = np.random.default_rng(0)
    contrasts = np.linspace(-100.0, 100.0, 41)
    for _ in range(20):
        parameters = np.asarray(
            [
                generator.uniform(-30.0, 30.0),
                generator.uniform(1.0, 60.0),
                generator.uniform(0.0, 0.25),
                generator.uniform(0.0, 0.25),
            ]
        )
        released = erf_psycho_2gammas(parameters, contrasts)
        promoted = erf_two_gamma_probability(
            contrasts,
            bias=parameters[0],
            threshold=parameters[1],
            lapse_low=parameters[2],
            lapse_high=parameters[3],
        )
        assert np.array_equal(released, promoted)


def test_the_erf_model_predicts_the_released_two_gamma_curve() -> None:
    contrasts = np.linspace(-100.0, 100.0, 25)
    model = PsychometricFunction(
        link=PsychometricLink.ERF,
        stimulus="signed_contrast",
        outcome="rightward",
        maximum_guess=0.4,
        maximum_lapse=0.4,
    )
    parameters = model.parameters_from_components(
        threshold=-4.0, width=18.0, guess_rate=0.06, lapse_rate=0.11
    )
    study = Study(
        {
            "subject": ["a"] * len(contrasts),
            "session": ["s"] * len(contrasts),
            "trial": list(range(len(contrasts))),
            "session_order": [0] * len(contrasts),
            "signed_contrast": contrasts,
            "rightward": np.zeros(len(contrasts), dtype=np.int8),
        }
    )
    fit = model.fit(study)
    known = type(fit)(
        model_name=fit.model_name,
        model_signature=fit.model_signature,
        parameter_names=fit.parameter_names,
        estimates=np.asarray([parameters[name] for name in model.parameter_names]),
        standard_errors=np.ones(len(model.parameter_names)),
        covariance=np.eye(len(model.parameter_names)),
        n_observations=len(study),
        diagnostics=fit.diagnostics,
        derived=(
            DerivedQuantity("threshold", -4.0),
            DerivedQuantity("width", 18.0),
            DerivedQuantity("guess_rate", 0.06),
            DerivedQuantity("lapse_rate", 0.11),
        ),
        restart_objectives=fit.restart_objectives,
        restart_converged=fit.restart_converged,
        restart_messages=fit.restart_messages,
        selected_restart=fit.selected_restart,
    )

    predicted = model.predict(study, known).probability
    released = erf_psycho_2gammas(np.asarray([-4.0, 18.0, 0.06, 0.11]), contrasts)

    # The estimated coordinate stores the two rates as bounded logits, so a round trip
    # through the model reproduces the released curve to floating-point tolerance rather
    # than bit for bit. ``erf_two_gamma_probability`` takes natural rates and is exact.
    assert np.allclose(predicted, released, rtol=0, atol=1e-12)


def test_the_pinned_logistic_baseline_is_untouched_by_the_new_family() -> None:
    """`Psychometric` stays exported, and gains a lapse only by being mixed."""

    baseline = Psychometric()
    study = baseline.simulate(design(600), {"intercept": -0.2, "stimulus": 1.1}, seed=10)
    lapse = mix(baseline, UniformChoiceGuess(), weight_bounds=(0.0, 0.2))
    lapse_study = lapse.simulate(
        design(600),
        lapse.from_natural({"intercept": 0.1, "stimulus": 1.0, "lapse_rate": 0.05}),
        seed=11,
    )

    assert baseline.fit(study).model_name == "psychometric"
    assert lapse.fit(lapse_study).model_name == "lapse-psychometric"
    assert PsychometricFunction().model_name == "psychometric-logistic"
    assert baseline.parameter_names == ("intercept", "stimulus")


def test_asymmetry_is_the_link_s_business_and_not_the_mixture_s() -> None:
    """A declared asymmetric guess is expressible; two *estimated* rates are the link's.

    The two-gamma form is algebraically a mixture -- weight ``guess + lapse`` on a Bernoulli
    guess of probability ``guess / (guess + lapse)`` -- so it is not that `mix` cannot reach
    it, but that reaching it needs a second estimated number inside the component.
    `PsychometricFunction` estimates both rates inside the link, which is where the shape of
    a curve belongs; `mix` estimates one weight over a component whose asymmetry is declared.
    """

    declared = mix(
        Psychometric(),
        UniformChoiceGuess(probability=0.7),
        weight_bounds=(0.0, 0.3),
    )
    parameters = declared.from_natural({"intercept": 0.0, "stimulus": 1.5, "lapse_rate": 0.2})
    study = declared.simulate(design(800), parameters, seed=3)
    fit = declared.fit(study)
    probability = declared.predict(study, fit).probability

    # A guess that favours one response makes the two asymptotes different, which a
    # symmetric mixture cannot do at all.
    assert probability.max() < 1.0 - 1e-6
    assert probability.min() > 1e-6
    assert 1.0 - probability.max() != pytest.approx(probability.min(), rel=0.2)
    assert declared.parameter_names == ("intercept", "stimulus", "mixture_logit")
    assert PsychometricFunction().parameter_names == (
        "threshold",
        "log_width",
        "guess_logit",
        "lapse_logit",
    )


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"stimulus": "choice", "outcome": "choice"}, "distinct"),
        ({"stimulus": "subject"}, "required Study column"),
        ({"maximum_lapse": 0.0}, "strictly between zero and one"),
        ({"maximum_guess": 0.6, "maximum_lapse": 0.6}, "positive range"),
        ({"fixed_guess_rate": 0.95, "maximum_lapse": 0.2}, "positive range"),
        ({"fixed_lapse_rate": 1.5}, r"must lie in \[0, 1\)"),
        ({"n_restarts": 0}, "positive integer"),
        ({"link": "cauchy"}, "not a valid PsychometricLink"),
    ],
)
def test_psychometric_configuration_is_validated(arguments: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PsychometricFunction(**arguments)


def test_the_weibull_link_rejects_non_positive_stimulus_levels() -> None:
    model = PsychometricFunction(link=PsychometricLink.WEIBULL)
    parameters = model.parameters_from_components(
        threshold=2.0, width=0.5, guess_rate=0.02, lapse_rate=0.02
    )

    with pytest.raises(ModelDataError, match="positive stimulus"):
        model.simulate(design(120, low=-1.0, high=4.0), parameters, seed=1)
    with pytest.raises(ValueError, match="positive threshold"):
        model.parameters_from_components(
            threshold=-1.0, width=0.5, guess_rate=0.02, lapse_rate=0.02
        )


def test_a_rate_on_its_bound_must_be_declared_fixed_rather_than_estimated() -> None:
    model = PsychometricFunction(maximum_lapse=0.2)

    with pytest.raises(ValueError, match="strictly inside"):
        model.parameters_from_components(threshold=0.0, width=1.0, guess_rate=0.05, lapse_rate=0.2)
    with pytest.raises(ValueError, match="must be supplied"):
        model.parameters_from_components(threshold=0.0, width=1.0, guess_rate=0.05)


def test_a_fit_cannot_be_read_under_a_different_link() -> None:
    logistic = PsychometricFunction(link=PsychometricLink.LOGISTIC)
    gauss = PsychometricFunction(link=PsychometricLink.GAUSS)
    parameters = logistic.parameters_from_components(
        threshold=0.0, width=1.0, guess_rate=0.03, lapse_rate=0.03
    )
    study = logistic.simulate(design(600), parameters, seed=3)
    fit = logistic.fit(study)

    with pytest.raises(ValueError, match="different model specification"):
        gauss.predict(study, fit)
