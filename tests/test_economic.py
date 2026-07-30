"""Economic and value-based choice, checked against closed forms rather than against itself.

Three of these tests would still pass a wrong implementation of the fits, and they are the
ones this file exists for.

A hyperbolic discounter's **indifference point** is closed form, so the choice probability at
that rate is exactly one half whatever the inverse temperature is; a model that got the
discount factor wrong cannot satisfy that for both discount functions at once. Prospect
theory's **fourfold pattern** is a qualitative prediction with a published sign in each of
four cells, and it is asserted twice -- once from Tversky and Kahneman's (1992) declared
medians, where no fitting is involved at all, and once from parameters recovered out of
simulated choices. And the **analytic gradient** of both likelihoods is checked against a
central difference, because a wrong gradient converges to a wrong answer quietly.

The rest of the file is the composition claim: that ``smooth()`` and ``hierarchical()`` work
on a family written after they existed, without either combinator being touched, and that
``mix()`` does not -- for a reason that is worth a test of its own, because it is not the
reason a reinforcement-learning agent declines one.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from behavio import Study, compare_models, evaluate_splits, run_parameter_recovery
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth
from behavio.contracts.bounded import (
    BOUNDED_COORDINATE,
    BoundedCoordinateEstimator,
    require_composable,
)
from behavio.evaluate import cohort_forward_session_splits
from behavio.models import BernoulliHistoryGLM, ModelDataError
from behavio.models.economic import (
    DiscountFunction,
    ProspectTheory,
    ProspectTheoryParameters,
    TemporalDiscounting,
    certainty_equivalent,
    indifference_discount_rate,
    prelec_weight,
)

#: Tversky and Kahneman's (1992) median estimates, with Prelec's curvature standing in for
#: their one-parameter weighting exponent. Every "known answer" below is read against these.
TK92 = ProspectTheoryParameters(
    gain_exponent=0.88,
    loss_exponent=0.88,
    loss_aversion=2.25,
    weighting_curvature=0.65,
    weighting_elevation=1.0,
    inverse_temperature=8.0,
)


# --------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------


def discount_design(
    subjects: Sequence[str],
    *,
    sessions: int = 1,
    delays: Sequence[float] = (7.0, 30.0, 90.0, 365.0),
) -> Study:
    """A titration over the smaller-sooner amount at several later delays."""

    ratios = (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97)
    columns: dict[str, list[object]] = {
        name: []
        for name in (
            "subject",
            "session",
            "trial",
            "session_order",
            "sooner_amount",
            "sooner_delay",
            "later_amount",
            "later_delay",
        )
    }
    for subject in subjects:
        trial = 0
        for session in range(sessions):
            for delay, ratio in itertools.product(delays, ratios):
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-s{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["sooner_amount"].append(100.0 * ratio)
                columns["sooner_delay"].append(0.0)
                columns["later_amount"].append(100.0)
                columns["later_delay"].append(float(delay))
                trial += 1
    return Study(columns)


def risk_design(subjects: Sequence[str], *, sessions: int = 1, mixed: bool = True) -> Study:
    """A gamble-versus-sure-thing titration over gains and losses.

    ``mixed`` adds trials that place a certain loss against a probabilistic gain, which is the
    only comparison in a design of single-non-zero-outcome prospects that can locate loss
    aversion. Turning it off is how the ``unidentified_loss_aversion`` finding is provoked.
    """

    probabilities = (0.05, 0.25, 0.5, 0.75, 0.95)
    magnitudes = (20.0, 50.0, 100.0, 200.0)
    ratios = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95)
    rows: list[tuple[object, ...]] = []
    for subject in subjects:
        trial = 0
        for session in range(sessions):
            for sign, probability, magnitude, ratio in itertools.product(
                (1.0, -1.0), probabilities, magnitudes, ratios
            ):
                rows.append(
                    (
                        subject,
                        f"{subject}-s{session}",
                        trial,
                        session,
                        sign * magnitude * ratio,
                        1.0,
                        sign * magnitude,
                        probability,
                    )
                )
                trial += 1
            if not mixed:
                continue
            for magnitude, probability, ratio in itertools.product(
                (50.0, 100.0), (0.4, 0.6), (0.1, 0.2, 0.35, 0.5)
            ):
                rows.append(
                    (
                        subject,
                        f"{subject}-s{session}",
                        trial,
                        session,
                        -magnitude * ratio,
                        1.0,
                        magnitude,
                        probability,
                    )
                )
                trial += 1
    names = (
        "subject",
        "session",
        "trial",
        "session_order",
        "option_0_outcome",
        "option_0_probability",
        "option_1_outcome",
        "option_1_probability",
    )
    values = zip(*rows, strict=True)
    return Study({name: list(column) for name, column in zip(names, values, strict=True)})


def with_choices(study: Study, *, seed: int = 0) -> Study:
    """Attach an arbitrary observed choice column, for tests that only read ``describe()``."""

    columns = {name: study[name] for name in study.columns}
    columns["choice"] = np.random.default_rng(seed).integers(0, 2, len(study)).astype(np.int8)
    return Study(columns)


def discounter(discount: str = "hyperbolic") -> TemporalDiscounting:
    return TemporalDiscounting(discount=DiscountFunction(discount), value_scale=100.0)


def prospect() -> ProspectTheory:
    return ProspectTheory(value_scale=100.0)


def tk92_coordinate(model: ProspectTheory) -> dict[str, float]:
    return dict(
        model.parameters_from_components(
            gain_exponent=TK92.gain_exponent,
            loss_exponent=TK92.loss_exponent,
            loss_aversion=TK92.loss_aversion,
            weighting_curvature=TK92.weighting_curvature,
            weighting_elevation=TK92.weighting_elevation,
            inverse_temperature=TK92.inverse_temperature,
        )
    )


def fourfold_attitudes(components: ProspectTheoryParameters) -> dict[tuple[str, str], str]:
    """Classify the four cells of the fourfold pattern by comparing a CE to its EV."""

    attitudes: dict[tuple[str, str], str] = {}
    for domain, outcome in (("gain", 100.0), ("loss", -100.0)):
        for likelihood, probability in (("unlikely", 0.05), ("likely", 0.9)):
            equivalent = float(certainty_equivalent(outcome, probability, components))
            expected = probability * outcome
            attitudes[(domain, likelihood)] = "seeking" if equivalent > expected else "averse"
    return attitudes


FOURFOLD = {
    ("gain", "likely"): "averse",
    ("gain", "unlikely"): "seeking",
    ("loss", "likely"): "seeking",
    ("loss", "unlikely"): "averse",
}


# --------------------------------------------------------------------------------------
# 1. Known answers: the discounting indifference point
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("discount", ["hyperbolic", "exponential"])
def test_the_closed_form_indifference_rate_makes_the_choice_a_coin(discount: str) -> None:
    """The check a wrong discount factor cannot pass.

    Mazur's hyperbola and the exponential give different indifference rates for the same
    trial, so a model that computed one factor and reported the other would satisfy this for
    at most one of the two parameterisations.
    """

    model = discounter(discount)
    study = discount_design(["a"])
    rates = model.indifference_rates(study)
    assert np.all(np.isfinite(rates))

    for rate in np.unique(np.round(rates, 10))[:4]:
        for temperature in (0.5, 8.0, 40.0):
            parameters = model.parameters_from_components(
                discount_rate=float(rate), inverse_temperature=temperature
            )
            probability = model.choice_probability(study, parameters)
            at_indifference = np.isclose(rates, rate)
            assert np.allclose(probability[at_indifference], 0.5, atol=1e-9)


def test_the_hyperbolic_indifference_rate_is_mazurs_algebra() -> None:
    rate = indifference_discount_rate(50.0, 0.0, 100.0, 30.0, discount="hyperbolic")

    assert float(rate) == pytest.approx((100.0 - 50.0) / (50.0 * 30.0), rel=1e-12)


def test_the_exponential_indifference_rate_is_the_log_amount_ratio() -> None:
    rate = indifference_discount_rate(50.0, 0.0, 100.0, 30.0, discount="exponential")

    assert float(rate) == pytest.approx(np.log(2.0) / 30.0, rel=1e-12)


def test_a_trial_no_positive_rate_can_equate_has_no_indifference_point() -> None:
    """The later option is also the smaller one, so no amount of discounting equates them."""

    assert np.isnan(indifference_discount_rate(100.0, 0.0, 50.0, 30.0))


# --------------------------------------------------------------------------------------
# 2. Known answers: the weighting function and the fourfold pattern
# --------------------------------------------------------------------------------------


def test_prelec_weighting_is_a_weighting_function() -> None:
    """Anchored at both ends, strictly increasing, and crossing the diagonal at ``1/e``.

    The third is the property that picked this form. Prelec's fixed point is
    :math:`\\exp(-\\delta^{1/(1-\\gamma)})`, which is :math:`1/e` for every curvature when the
    elevation is one -- so a single number says where the crossover is, whatever the curvature
    does around it.
    """

    grid = np.linspace(1e-6, 1.0, 4001)
    for curvature in (0.3, 0.65, 1.0, 1.8):
        weights = prelec_weight(grid, curvature=curvature)
        assert np.all(np.diff(weights) > 0.0)
        assert prelec_weight(1.0, curvature=curvature) == pytest.approx(1.0)
        assert prelec_weight(0.0, curvature=curvature) == pytest.approx(0.0)
        crossing = float(prelec_weight(1.0 / np.e, curvature=curvature))
        assert crossing == pytest.approx(1.0 / np.e, rel=1e-12)


def test_elevation_moves_the_curve_without_moving_its_curvature() -> None:
    """The separation that Tversky and Kahneman's one-parameter form cannot express."""

    grid = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    neutral = prelec_weight(grid, curvature=0.65, elevation=1.0)
    pessimistic = prelec_weight(grid, curvature=0.65, elevation=1.6)

    assert np.all(pessimistic < neutral)


def test_declared_tversky_kahneman_medians_give_the_fourfold_pattern() -> None:
    """No fitting at all: the published parameters, the published qualitative prediction."""

    assert fourfold_attitudes(TK92) == FOURFOLD


def test_loss_aversion_does_not_move_a_within_domain_certainty_equivalent() -> None:
    """Which is why the model warns when a design has no gain-against-loss trial."""

    doubled = ProspectTheoryParameters(
        gain_exponent=TK92.gain_exponent,
        loss_exponent=TK92.loss_exponent,
        loss_aversion=2.0 * TK92.loss_aversion,
        weighting_curvature=TK92.weighting_curvature,
        weighting_elevation=TK92.weighting_elevation,
        inverse_temperature=TK92.inverse_temperature,
    )

    assert certainty_equivalent(-100.0, 0.4, doubled) == pytest.approx(
        certainty_equivalent(-100.0, 0.4, TK92)
    )


def test_a_sure_prospect_is_never_probability_weighted() -> None:
    """Cumulative weighting's defining property, and the reason it is used here."""

    model = ProspectTheory(
        other_outcome_columns=("option_0_other", "option_1_other"), value_scale=100.0
    )
    study = Study(
        {
            "subject": ["a", "a"],
            "session": ["a-s0", "a-s0"],
            "trial": [0, 1],
            "session_order": [0, 0],
            "option_0_outcome": [40.0, -40.0],
            "option_0_other": [40.0, -40.0],
            "option_0_probability": [0.3, 0.3],
            "option_1_outcome": [40.0, -40.0],
            "option_1_other": [40.0, -40.0],
            "option_1_probability": [0.9, 0.9],
        }
    )
    values = model.subjective_values(study, tk92_coordinate(model))

    assert values[:, 0] == pytest.approx(values[:, 1])


# --------------------------------------------------------------------------------------
# 3. The analytic gradients
# --------------------------------------------------------------------------------------


def numeric_gradient(model, options, outcomes, vector: np.ndarray) -> np.ndarray:
    steps = np.eye(len(vector)) * 1e-6
    return np.asarray(
        [
            (
                model._objective(vector + step, options, outcomes)[0]
                - model._objective(vector - step, options, outcomes)[0]
            )
            / 2e-6
            for step in steps
        ]
    )


def test_the_discounting_gradient_matches_a_central_difference() -> None:
    model = discounter()
    study = model.simulate(
        discount_design(["a"]),
        model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0),
        seed=0,
    )
    options, outcomes = model.read_options(study), np.asarray(study["choice"], dtype=np.float64)
    vector = np.asarray([np.log(0.031), np.log(5.5)])

    _, gradient = model._objective(vector, options, outcomes)

    assert gradient == pytest.approx(numeric_gradient(model, options, outcomes, vector), abs=1e-6)


def test_the_prospect_theory_gradient_matches_a_central_difference() -> None:
    """Over a design that exercises all three cases of the cumulative weighting rule."""

    generator = np.random.default_rng(5)
    rows = 60
    study = Study(
        {
            "subject": ["a"] * rows,
            "session": ["a-s0"] * rows,
            "trial": list(range(rows)),
            "session_order": [0] * rows,
            "option_0_outcome": generator.choice([-80.0, -30.0, 0.0, 25.0, 60.0], rows),
            "option_0_other": generator.choice([-20.0, 0.0, 15.0], rows),
            "option_0_probability": generator.choice([0.05, 0.25, 0.5, 0.9, 1.0], rows),
            "option_1_outcome": generator.choice([-100.0, -40.0, 10.0, 50.0, 90.0], rows),
            "option_1_other": generator.choice([-50.0, 0.0, 30.0], rows),
            "option_1_probability": generator.choice([0.1, 0.4, 0.75, 0.95], rows),
            "choice": generator.integers(0, 2, rows).astype(np.int8),
        }
    )
    model = ProspectTheory(
        other_outcome_columns=("option_0_other", "option_1_other"), value_scale=100.0
    )
    options, outcomes = model.read_options(study), np.asarray(study["choice"], dtype=np.float64)
    vector = np.log(np.asarray([0.8, 0.9, 2.0, 0.7, 1.1, 3.0]))

    _, gradient = model._objective(vector, options, outcomes)

    assert gradient == pytest.approx(numeric_gradient(model, options, outcomes, vector), abs=1e-6)


# --------------------------------------------------------------------------------------
# 4. Recovery, and the fourfold pattern out of a fit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("discount", ["hyperbolic", "exponential"])
def test_a_discount_rate_and_a_temperature_recover_together(discount: str) -> None:
    model = discounter(discount)
    design = discount_design(["a"], sessions=6)
    truth = model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0)

    report = run_parameter_recovery(model, design, [truth], repeats=4, seed=11)

    summaries = {summary.parameter: summary for summary in report.summary()}
    assert abs(summaries["discount_rate_log"].bias) < 0.25
    assert abs(summaries["inverse_temperature_log"].bias) < 0.3
    assert summaries["discount_rate_log"].coverage_95 >= 0.75


def test_prospect_theory_recovers_all_six_parameters() -> None:
    model = prospect()
    design = risk_design(["a"], sessions=4)
    truth = tk92_coordinate(model)

    report = run_parameter_recovery(model, design, [truth], repeats=3, seed=7)

    for summary in report.summary():
        assert abs(summary.bias) < 0.3, summary.parameter
        assert summary.coverage_95 >= 0.6, summary.parameter


def test_a_fitted_prospect_theory_model_reproduces_the_fourfold_pattern() -> None:
    """The qualitative prediction has to survive the round trip, not only the algebra."""

    model = prospect()
    design = risk_design(["a"], sessions=4)
    study = model.simulate(design, tk92_coordinate(model), seed=4)

    fit = model.fit(study)

    assert fit.diagnostics.converged
    assert not fit.diagnostics.boundary_estimate
    assert fourfold_attitudes(model.parameter_components(fit)) == FOURFOLD


def test_a_fit_reports_the_natural_coordinate_beside_the_estimated_one() -> None:
    model = discounter()
    study = model.simulate(
        discount_design(["a"], sessions=2),
        model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0),
        seed=2,
    )

    fit = model.fit(study)

    assert set(fit.derived_values) == {"discount_rate", "inverse_temperature"}
    assert fit.derived_values["discount_rate"] == pytest.approx(
        float(np.exp(fit.estimates[0])), rel=1e-12
    )
    assert fit.derived_quantities["discount_rate"].standard_error > 0.0


# --------------------------------------------------------------------------------------
# 5. The composition contract, and which combinators apply
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", [discounter(), prospect()])
def test_both_families_compose_through_the_bounded_coordinate_contract(model: object) -> None:
    assert isinstance(model, BoundedCoordinateEstimator)
    assert require_composable(model, combinator="smooth") == BOUNDED_COORDINATE
    assert require_composable(model, combinator="hierarchical") == BOUNDED_COORDINATE


@pytest.mark.parametrize("model", [discounter(), prospect()])
def test_the_declared_box_is_finite_on_every_coordinate(model: object) -> None:
    study = discount_design(["a"]) if isinstance(model, TemporalDiscounting) else risk_design(["a"])
    box = model.coordinate_box(study)

    assert box.shape == (len(model.parameter_names), 2)
    assert np.all(np.isfinite(box)) and np.all(box[:, 1] > box[:, 0])


def test_hierarchy_recovers_per_subject_discount_rates() -> None:
    model = discounter()
    pooled = hierarchical(model, over="subject", parameters=("discount_rate_log",), scale=0.5)
    truth = model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0)
    simulation = pooled.simulate_with_effects(
        discount_design(["a", "b", "c", "d"], sessions=3), truth, seed=5
    )

    fit = pooled.fit(simulation.study)

    assert fit.groups == ("a", "b", "c", "d")
    assert fit.varying_parameters == ("discount_rate_log",)
    correlation = np.corrcoef(simulation.group_deviations.ravel(), fit.group_deviations.ravel())[
        0, 1
    ]
    assert correlation > 0.8


def test_hierarchy_works_on_prospect_theory_too() -> None:
    model = prospect()
    pooled = hierarchical(model, over="subject", parameters=("loss_aversion_log",), scale=0.4)
    simulation = pooled.simulate_with_effects(
        risk_design(["a", "b", "c"], sessions=2), tk92_coordinate(model), seed=6
    )

    fit = pooled.fit(simulation.study)

    assert fit.group_deviations.shape == (3, 1)
    assert np.all(np.isfinite(fit.group_deviations))


def test_a_group_deviation_keeps_a_discount_rate_positive() -> None:
    """A Gaussian on the rate itself is a negative rate a quarter of the time; on its log it
    is a rate, always. That is the whole reason the estimated coordinate is a logarithm."""

    model = discounter()
    deviations = model.draw_group_deviations(
        np.asarray([0], dtype=np.intp),
        np.asarray([1.5]),
        groups=400,
        generator=np.random.default_rng(0),
    ).ravel()

    assert np.all(0.02 * np.exp(deviations) > 0.0)
    assert np.any(0.02 + deviations <= 0.0)


def test_smoothness_recovers_a_discount_rate_that_drifts_across_sessions() -> None:
    """The experimental question the combinator makes askable without new modelling code."""

    model = discounter()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 5.0),
        parameters=("discount_rate_log",),
        smoothness=1.0,
    )
    paths = drifting.parameters_from_paths(
        {
            "discount_rate_log": (float(np.log(0.005)), float(np.log(0.08))),
            "inverse_temperature_log": float(np.log(8.0)),
        }
    )
    study = drifting.simulate(discount_design(["a"], sessions=6), paths, seed=9)

    fit = drifting.fit(study)

    assert drifting.parameter_names == (
        "discount_rate_log[session_order=0]",
        "discount_rate_log[session_order=5]",
        "inverse_temperature_log",
    )
    values = dict(zip(drifting.parameter_names, fit.estimates, strict=True))
    early = values["discount_rate_log[session_order=0]"]
    late = values["discount_rate_log[session_order=5]"]
    # Direction and order of magnitude, not equality: a roughness penalty shrinks a
    # two-knot path towards flatness, so the fitted ends bracket the truth from inside.
    assert early < late - 1.0
    assert 0.001 < float(np.exp(early)) < 0.02
    assert 0.03 < float(np.exp(late)) < 0.25


def test_a_within_session_clock_is_admissible_here_and_is_not_for_an_agent() -> None:
    """The clock-block rule the RL families needed does **not** apply to this family.

    ``smooth(BinaryQLearning(), over="trial")`` is a ``ModelDataError``, because a learning
    rate that changed mid-session leaves the value trace unable to say which of its values
    wrote which part. A discounting trial's likelihood reads that trial's amounts and delays
    and nothing else, so ``row_blocks`` is ``arange(n_rows)``, a coordinate may vary trial by
    trial, and the composition is defined. Whether it is *estimable* is a separate question
    the knots answer; that it is not refused is the point.
    """

    model = discounter()
    design = discount_design(["a"], sessions=1)
    study = model.simulate(
        design,
        model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0),
        seed=3,
    )
    assert np.array_equal(model.row_objective(study).row_blocks, np.arange(len(design)))

    drifting = smooth(
        model,
        over="trial",
        knots=(0.0, float(len(design) - 1)),
        parameters=("discount_rate_log",),
        smoothness=2.0,
    )

    fit = drifting.fit(study)

    assert fit.diagnostics.converged
    assert len(fit.estimates) == 3


def test_mix_refuses_the_family_and_says_why_in_its_own_words() -> None:
    """The one cell that does not open, and the sentence that explains it.

    ``mix()`` is gated on ``require_penalised_linear`` rather than on row independence, so a
    row-independent bounded-coordinate family is refused for a reason that is about the
    absence of a *linear predictor*, not about a recursion. Nothing was changed to make this
    message appear; the model declares ``penalised_linear_refusal``, exactly as the GLM-HMM
    and the reinforcement-learning agent do.
    """

    for model in (discounter(), prospect()):
        with pytest.raises(TypeError, match="no design matrix and no linear predictor"):
            mix(model, UniformChoiceGuess())


# --------------------------------------------------------------------------------------
# 6. The identifiability hazard, reported before the fit
# --------------------------------------------------------------------------------------


def constant_delay_design() -> Study:
    study = discount_design(["a"])
    columns = {name: study[name] for name in study.columns}
    columns["later_delay"] = np.zeros(len(study))
    return with_choices(Study(columns))


def test_a_design_with_no_delay_difference_cannot_identify_the_discount_rate() -> None:
    findings = discounter().describe(constant_delay_design()).findings

    assert [finding.code for finding in findings] == ["unidentified_discount_rate"]
    assert all(finding.severity == "warning" for finding in findings)
    assert "inverse temperature absorbs it exactly" in findings[0].message


def single_magnitude_design() -> Study:
    """Every gamble is over the same amount, so only the probability varies."""

    study = risk_design(["a"], mixed=False)
    columns = {name: study[name] for name in study.columns}
    outcomes = np.asarray(study["option_1_outcome"], dtype=np.float64)
    columns["option_1_outcome"] = np.sign(outcomes) * 100.0
    columns["option_0_outcome"] = np.sign(outcomes) * 100.0
    return with_choices(Study(columns))


def test_a_single_outcome_magnitude_leaves_the_curvature_unidentified() -> None:
    findings = prospect().describe(single_magnitude_design()).findings
    codes = [finding.code for finding in findings]

    assert codes.count("unidentified_utility_curvature") == 2
    assert "gain_exponent is not identified" in " ".join(finding.message for finding in findings)


def test_a_design_without_a_gain_against_a_loss_cannot_identify_loss_aversion() -> None:
    findings = prospect().describe(with_choices(risk_design(["a"], mixed=False))).findings

    assert "unidentified_loss_aversion" in {finding.code for finding in findings}


def test_the_same_design_with_cross_domain_trials_reports_nothing() -> None:
    assert prospect().describe(with_choices(risk_design(["a"]))).findings == ()


def test_a_gains_only_design_says_the_loss_branch_is_never_evaluated() -> None:
    study = risk_design(["a"], mixed=False)
    columns = {name: study[name] for name in study.columns}
    for name in ("option_0_outcome", "option_1_outcome"):
        columns[name] = np.abs(np.asarray(study[name], dtype=np.float64))
    findings = prospect().describe(with_choices(Study(columns))).findings

    assert "unobserved_loss_domain" in {finding.code for finding in findings}


def test_sure_things_only_leaves_the_weighting_function_with_nothing_to_do() -> None:
    study = risk_design(["a"], mixed=False)
    columns = {name: study[name] for name in study.columns}
    columns["option_1_probability"] = np.ones(len(study))
    findings = prospect().describe(with_choices(Study(columns))).findings

    assert "unidentified_probability_weighting" in {finding.code for finding in findings}


def test_an_undeclared_amount_unit_is_reported_before_it_reaches_the_box() -> None:
    """Amounts in pennies under a model that never said so."""

    study = with_choices(discount_design(["a"]))
    columns = {name: study[name] for name in study.columns}
    for name in ("sooner_amount", "later_amount"):
        columns[name] = 100.0 * np.asarray(study[name], dtype=np.float64)

    findings = TemporalDiscounting().describe(Study(columns)).findings

    assert "value_scale_mismatch" in {finding.code for finding in findings}
    assert TemporalDiscounting(value_scale=10_000.0).describe(Study(columns)).findings == ()


def test_a_composed_model_carries_the_findings_through() -> None:
    pooled = hierarchical(discounter(), over="subject", parameters=("discount_rate_log",))

    findings = pooled.describe(constant_delay_design()).findings

    assert "unidentified_discount_rate" in {finding.code for finding in findings}


def test_the_post_fit_half_of_the_hazard_is_a_correlation_on_the_fit() -> None:
    """A magnitude near one means the pair is only jointly identified."""

    model = prospect()
    study = model.simulate(risk_design(["a"], sessions=2), tk92_coordinate(model), seed=8)

    fit = model.fit(study)
    correlation = model.temperature_scale_correlation(fit, parameter="gain_exponent_log")

    assert -1.0 <= correlation <= 1.0
    with pytest.raises(ValueError, match="not an estimated utility coordinate"):
        model.temperature_scale_correlation(fit, parameter="inverse_temperature_log")


def test_a_wildly_mis_declared_scale_lands_the_temperature_on_its_box() -> None:
    model = TemporalDiscounting(value_scale=1e-6)
    study = discounter().simulate(
        discount_design(["a"], sessions=2),
        discounter().parameters_from_components(discount_rate=0.02, inverse_temperature=8.0),
        seed=2,
    )

    fit = model.fit(study)

    assert fit.diagnostics.boundary_estimate


# --------------------------------------------------------------------------------------
# 7. These are ordinary estimators
# --------------------------------------------------------------------------------------


def test_the_family_flows_through_evaluate_splits_and_compare_models() -> None:
    model = discounter()
    study = model.simulate(
        discount_design(["a", "b"], sessions=4),
        model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0),
        seed=12,
    )
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    evaluations = evaluate_splits(model, study, splits)
    comparison = compare_models(
        {"discounting": model, "guessing": BernoulliHistoryGLM(predictors=(), l2=1.0)},
        study,
        splits,
        bootstrap_resamples=50,
    )

    assert len(evaluations) == len(splits)
    assert all(
        np.all(np.isfinite(evaluation.pointwise_log_probability)) for evaluation in evaluations
    )
    assert comparison.winner == "discounting"


def test_a_hierarchical_prospect_model_is_still_an_ordinary_estimator() -> None:
    model = prospect()
    pooled = hierarchical(model, over="subject", parameters=("loss_aversion_log",), scale=0.4)
    study = model.simulate(risk_design(["a", "b"], sessions=3), tk92_coordinate(model), seed=15)
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    evaluations = evaluate_splits(pooled, study, splits)

    assert len(evaluations) == len(splits)


# --------------------------------------------------------------------------------------
# 8. Configuration and refusals
# --------------------------------------------------------------------------------------


def test_a_fixed_parameter_leaves_the_coordinate_rather_than_being_estimated() -> None:
    model = ProspectTheory(
        value_scale=100.0, fixed_loss_aversion=1.0, fixed_weighting_elevation=1.0
    )

    assert model.parameter_names == (
        "gain_exponent_log",
        "loss_exponent_log",
        "weighting_curvature_log",
        "inverse_temperature_log",
    )
    components = model.parameter_components(
        model.parameters_from_components(
            gain_exponent=0.9, loss_exponent=0.9, weighting_curvature=0.7, inverse_temperature=6.0
        )
    )
    assert components.loss_aversion == 1.0
    assert components.weighting_elevation == 1.0


def test_supplying_a_fixed_parameter_with_the_wrong_value_is_refused() -> None:
    model = ProspectTheory(fixed_loss_aversion=1.0)

    with pytest.raises(ValueError, match=r"loss_aversion is fixed at 1\.0"):
        model.parameters_from_components(
            gain_exponent=0.9,
            loss_exponent=0.9,
            loss_aversion=2.0,
            weighting_curvature=0.7,
            inverse_temperature=6.0,
        )


def test_a_non_positive_natural_parameter_cannot_be_encoded() -> None:
    with pytest.raises(ValueError, match="discount_rate must be finite and positive"):
        discounter().parameters_from_components(discount_rate=0.0, inverse_temperature=1.0)


def test_a_missing_column_is_a_model_data_error_and_a_finding() -> None:
    study = discount_design(["a"])
    columns = {name: study[name] for name in study.columns if name != "later_delay"}
    incomplete = Study(columns)

    assert "missing_column" in {
        finding.code for finding in discounter().describe(incomplete).findings
    }
    with pytest.raises(ModelDataError, match="later_delay"):
        discounter().fit(incomplete)


def test_a_negative_delay_is_refused() -> None:
    study = discount_design(["a"])
    columns = {name: study[name] for name in study.columns}
    columns["later_delay"] = -np.ones(len(study))

    with pytest.raises(ModelDataError, match="delays must be non-negative"):
        discounter().read_options(Study(columns))


def test_a_probability_outside_the_unit_interval_is_refused() -> None:
    study = risk_design(["a"])
    columns = {name: study[name] for name in study.columns}
    columns["option_1_probability"] = 1.5 * np.ones(len(study))

    with pytest.raises(ModelDataError, match=r"probabilities must lie in \[0, 1\]"):
        prospect().read_options(Study(columns))


def test_a_fit_from_another_specification_is_refused() -> None:
    model = discounter()
    study = model.simulate(
        discount_design(["a"]),
        model.parameters_from_components(discount_rate=0.02, inverse_temperature=8.0),
        seed=1,
    )
    fit = model.fit(study)

    with pytest.raises(ValueError, match="different model specification"):
        discounter("exponential").predict(study, fit)
