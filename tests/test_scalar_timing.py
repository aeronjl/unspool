"""Scalar timing, checked against the properties that make it a theory rather than a curve.

Four of these tests would still pass a plausible but wrong implementation, and they are the
ones this file exists for.

The **scalar property** is asserted directly: the coefficient of variation of simulated
reproductions is constant across a sixteen-fold range of targets, and the standard deviation
of the *predicted density* -- integrated off the tabulated grid rather than read off the
parameterisation -- is proportional to the target with the Weber fraction as its constant. A
clock with constant absolute variability cannot satisfy either.

The **bisection point** is asserted at the geometric mean of the anchors (Church & Deluty
1977) for the ratio rule and at the arithmetic mean for the difference rule, at three Weber
fractions, with no fitting involved. The two rules are then shown to be separated by exactly
one number on a single anchor pair, which is what makes the choice between them a modelling
declaration rather than something a fit can settle.

Both **analytic gradients** are checked against central differences, because a wrong gradient
converges to a wrong answer quietly.

The rest is the composition claim -- that ``smooth()``, ``hierarchical()`` and ``mix()`` work
on a family written after they existed, without any of them being touched -- and the
identifiability findings this family reports before anything is fitted.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from behavio import ScoreMetric, Study, compare_models, evaluate_splits, run_parameter_recovery
from behavio.adapters.estimator_conformance import assert_behaviour_estimator_conforms
from behavio.compare import UndeclaredMetric
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth
from behavio.contracts.bounded import (
    BOUNDED_COORDINATE,
    BoundedCoordinateEstimator,
    require_composable,
)
from behavio.contracts.estimator import DensityPrediction, Prediction
from behavio.contracts.mixture import MIXTURE_LOGIT
from behavio.evaluate import cohort_forward_session_splits
from behavio.models import BernoulliHistoryGLM, ModelDataError
from behavio.models.scalar_timing import (
    BisectionRule,
    DurationReproduction,
    TemporalBisection,
    bisection_threshold,
    memory_log_sd,
    weber_fraction_from_log_sd,
)

SHORT_ANCHOR = 2.0
LONG_ANCHOR = 8.0
GEOMETRIC_MEAN = 4.0
ARITHMETIC_MEAN = 5.0

TARGETS = (0.5, 1.0, 2.0, 4.0, 8.0)
PROBES = (2.0, 2.5, 3.2, 4.0, 5.0, 6.4, 8.0)


# --------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------


def reproduction_design(
    subjects: Sequence[str],
    *,
    sessions: int = 1,
    targets: Sequence[float] = TARGETS,
    repeats: int = 20,
) -> Study:
    """A reproduction schedule over targets spanning a sixteen-fold range."""

    columns: dict[str, list[object]] = {
        name: [] for name in ("subject", "session", "trial", "session_order", "target_duration")
    }
    for subject in subjects:
        trial = 0
        for session in range(sessions):
            for target, _ in itertools.product(targets, range(repeats)):
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-s{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["target_duration"].append(float(target))
                trial += 1
    return Study(columns)


def bisection_design(
    subjects: Sequence[str],
    *,
    sessions: int = 1,
    probes: Sequence[float] = PROBES,
    repeats: int = 20,
) -> Study:
    """A bisection schedule with probes spaced geometrically between the anchors."""

    columns: dict[str, list[object]] = {
        name: [] for name in ("subject", "session", "trial", "session_order", "probe_duration")
    }
    for subject in subjects:
        trial = 0
        for session in range(sessions):
            for probe, _ in itertools.product(probes, range(repeats)):
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-s{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["probe_duration"].append(float(probe))
                trial += 1
    return Study(columns)


def with_reports(study: Study, *, seed: int = 0) -> Study:
    """Attach an arbitrary observed report, for tests that only read ``describe()``."""

    columns = {name: study[name] for name in study.columns}
    columns["choice"] = np.random.default_rng(seed).integers(0, 2, len(study)).astype(np.int8)
    return Study(columns)


def with_reproductions(study: Study, *, seed: int = 0) -> Study:
    """Attach an arbitrary observed reproduction, for ``describe()``-only tests."""

    columns = {name: study[name] for name in study.columns}
    columns["reproduced_duration"] = np.full(len(study), 1.0) * (
        1.0 + 0.01 * np.random.default_rng(seed).random(len(study))
    )
    return Study(columns)


def bisector(rule: BisectionRule | str = BisectionRule.RATIO) -> TemporalBisection:
    return TemporalBisection(
        short_anchor=SHORT_ANCHOR, long_anchor=LONG_ANCHOR, rule=BisectionRule(rule)
    )


# --------------------------------------------------------------------------------------
# 1. Known answers: the scalar property
# --------------------------------------------------------------------------------------


def test_the_weber_parameterisation_is_an_exact_coefficient_of_variation() -> None:
    """The identity the whole module rests on, and its exact inverse."""

    for weber in (0.02, 0.1, 0.25, 0.6, 1.5):
        log_sd = float(memory_log_sd(weber))
        assert float(weber_fraction_from_log_sd(log_sd)) == pytest.approx(weber, rel=1e-12)
        # Analytic lognormal moments, written independently of the model.
        mean = float(np.exp(0.5 * log_sd**2))
        variance = float(np.exp(log_sd**2) * np.expm1(log_sd**2))
        assert np.sqrt(variance) / mean == pytest.approx(weber, rel=1e-12)


def test_simulated_reproductions_have_a_constant_coefficient_of_variation() -> None:
    """The scalar property, asserted on data rather than on the parameterisation.

    A clock whose error is a fixed number of milliseconds would give a coefficient of
    variation falling as ``1/T`` across this design, which spans a factor of sixteen.
    """

    model = DurationReproduction()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.18)
    study = model.simulate(reproduction_design(["a"], repeats=4000), truth, seed=0)
    targets = np.asarray(study["target_duration"], dtype=np.float64)
    reproductions = np.asarray(study["reproduced_duration"], dtype=np.float64)

    observed = np.asarray(
        [
            float(np.std(reproductions[targets == target], ddof=1))
            / float(np.mean(reproductions[targets == target]))
            for target in TARGETS
        ]
    )

    assert observed == pytest.approx(np.full(len(TARGETS), 0.18), abs=0.012)
    assert float(np.max(observed) - np.min(observed)) < 0.02


def test_the_reproduction_standard_deviation_is_proportional_to_the_target() -> None:
    """Weber's law for time as a regression: slope ``w``, and an intercept at zero."""

    model = DurationReproduction()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.18)
    study = model.simulate(reproduction_design(["a"], repeats=4000), truth, seed=1)
    targets = np.asarray(study["target_duration"], dtype=np.float64)
    reproductions = np.asarray(study["reproduced_duration"], dtype=np.float64)
    spreads = np.asarray(
        [float(np.std(reproductions[targets == target], ddof=1)) for target in TARGETS]
    )

    levels = np.asarray(TARGETS, dtype=np.float64)
    slope, intercept = np.polyfit(levels, spreads, 1)

    assert float(slope) == pytest.approx(0.18, abs=0.012)
    assert abs(float(intercept)) < 0.02 * float(np.max(levels))


def test_the_predicted_density_itself_has_the_scalar_property() -> None:
    """Integrated off the tabulated grid, so it checks the density and not the sampler.

    ``predict`` returns a :class:`DensityPrediction`; its first two moments are taken here by
    quadrature on the grid the model published, which is a different route to the same claim
    from the one the simulation test takes.
    """

    model = DurationReproduction(grid_points=2049)
    design = reproduction_design(["a"], repeats=1)
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.2)
    study = model.simulate(design, truth, seed=2)
    fit_free = model.fit(study)
    prediction = model.predict(design, _replace_estimates(model, fit_free, truth))

    grid = np.asarray(prediction.grid, dtype=np.float64)
    density = np.asarray(prediction.density, dtype=np.float64)
    mean = np.trapezoid(density * grid, grid, axis=-1)
    second = np.trapezoid(density * grid**2, grid, axis=-1)
    coefficient = np.sqrt(np.maximum(second - mean**2, 0.0)) / mean

    assert np.allclose(prediction.total_mass, 1.0, atol=1e-3)
    assert coefficient == pytest.approx(np.full(len(design), 0.2), rel=5e-3)
    assert mean == pytest.approx(
        np.asarray(design["target_duration"], dtype=np.float64) * np.exp(0.5 * 0.2**2),
        rel=5e-3,
    )


def _replace_estimates(model, fit, parameters):
    """Return ``fit`` with its estimates swapped for a declared coordinate."""

    from dataclasses import replace

    return replace(
        fit,
        estimates=np.asarray([parameters[name] for name in model.parameter_names]),
    )


# --------------------------------------------------------------------------------------
# 2. Known answers: where a bisection curve crosses one half
# --------------------------------------------------------------------------------------


def test_an_accurate_clock_bisects_at_the_geometric_mean() -> None:
    """Church and Deluty's (1977) result, with no fitting anywhere in it."""

    model = bisector()
    design = bisection_design(["a"], probes=(GEOMETRIC_MEAN,), repeats=1)

    for weber in (0.05, 0.2, 0.8):
        parameters = model.parameters_from_components(clock_rate=1.0, weber_fraction=weber)
        assert model.report_probability(design, parameters) == pytest.approx(0.5, abs=1e-12)
        assert model.bisection_point(parameters) == pytest.approx(GEOMETRIC_MEAN, rel=1e-12)


def test_the_difference_rule_bisects_at_the_arithmetic_mean_instead() -> None:
    """The rule the literature disagrees about, and the number it disagrees over."""

    model = bisector(BisectionRule.DIFFERENCE)
    design = bisection_design(["a"], probes=(ARITHMETIC_MEAN,), repeats=1)

    for weber in (0.05, 0.2, 0.8):
        parameters = model.parameters_from_components(clock_rate=1.0, weber_fraction=weber)
        assert model.report_probability(design, parameters) == pytest.approx(0.5, abs=1e-12)
        assert model.bisection_point(parameters) == pytest.approx(ARITHMETIC_MEAN, rel=1e-12)


def test_the_two_comparison_durations_are_the_published_means() -> None:
    assert bisection_threshold(2.0, 8.0) == pytest.approx(4.0)
    assert bisection_threshold(2.0, 8.0, rule="difference") == pytest.approx(5.0)
    # Church and Deluty's four anchor pairs: the two accounts disagree on every one.
    for short, long_ in ((1.0, 4.0), (2.0, 8.0), (3.0, 12.0), (4.0, 16.0)):
        assert bisection_threshold(short, long_) == pytest.approx(np.sqrt(short * long_))
        assert bisection_threshold(short, long_) < bisection_threshold(
            short, long_, rule="difference"
        )


def test_one_anchor_pair_cannot_separate_the_two_rules() -> None:
    """The identifiability statement that makes the rule a declaration, not a finding.

    Fitting the same reports under both rules gives the same Weber fraction and the same log
    likelihood; only the clock rate moves, and it moves by exactly the ratio of the two
    comparison durations. Church and Deluty separated the accounts by varying the anchor
    pair, which is a comparison *between* models rather than a parameter inside one.
    """

    ratio_model = bisector()
    difference_model = bisector(BisectionRule.DIFFERENCE)
    truth = ratio_model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25)
    study = ratio_model.simulate(bisection_design(["a"], sessions=4), truth, seed=3)

    ratio_fit = ratio_model.fit(study)
    difference_fit = difference_model.fit(study)

    ratio_natural = ratio_model.parameter_components(ratio_fit)
    difference_natural = difference_model.parameter_components(difference_fit)

    assert difference_fit.diagnostics.objective == pytest.approx(
        ratio_fit.diagnostics.objective, rel=1e-8
    )
    assert difference_natural.weber_fraction == pytest.approx(
        ratio_natural.weber_fraction, rel=1e-4
    )
    assert difference_natural.clock_rate / ratio_natural.clock_rate == pytest.approx(
        ARITHMETIC_MEAN / GEOMETRIC_MEAN, rel=1e-4
    )
    # And therefore the two rules place the bisection point at the same observed duration.
    assert difference_model.bisection_point(difference_fit) == pytest.approx(
        ratio_model.bisection_point(ratio_fit), rel=1e-4
    )


def test_the_bisection_curve_is_a_function_of_the_probe_over_the_comparison() -> None:
    """Superposition: scaling the whole task leaves every report probability where it was."""

    model = bisector()
    scaled = TemporalBisection(short_anchor=6.0, long_anchor=24.0)
    parameters = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25)
    design = bisection_design(["a"], repeats=1)
    scaled_columns = {name: design[name] for name in design.columns}
    scaled_columns["probe_duration"] = 3.0 * np.asarray(design["probe_duration"])

    assert model.report_probability(design, parameters) == pytest.approx(
        scaled.report_probability(Study(scaled_columns), parameters)
    )


# --------------------------------------------------------------------------------------
# 3. The analytic gradients
# --------------------------------------------------------------------------------------


def numeric_gradient(model, problem, vector: np.ndarray) -> np.ndarray:
    steps = np.eye(len(vector)) * 1e-6
    return np.asarray(
        [
            (
                model.shared_value_and_gradient(problem, vector + step)[0]
                - model.shared_value_and_gradient(problem, vector - step)[0]
            )
            / 2e-6
            for step in steps
        ]
    )


def test_the_reproduction_gradient_matches_a_central_difference() -> None:
    model = DurationReproduction(fixed_central_tendency=None)
    truth = model.parameters_from_components(
        clock_rate=1.0, weber_fraction=0.18, central_tendency=0.9
    )
    study = model.simulate(reproduction_design(["a"]), truth, seed=4)
    problem = model.read_problem(study)
    vector = np.log(np.asarray([1.15, 0.22, 0.85]))

    _, gradient = model.shared_value_and_gradient(problem, vector)

    assert gradient == pytest.approx(numeric_gradient(model, problem, vector), abs=1e-5)


def test_the_bisection_gradient_matches_a_central_difference() -> None:
    model = bisector()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25)
    study = model.simulate(bisection_design(["a"], sessions=2), truth, seed=5)
    problem = model.read_problem(study)
    vector = np.log(np.asarray([1.2, 0.31]))

    _, gradient = model.shared_value_and_gradient(problem, vector)

    assert gradient == pytest.approx(numeric_gradient(model, problem, vector), abs=1e-5)


# --------------------------------------------------------------------------------------
# 4. Recovery, and the claim the two paradigms share a clock
# --------------------------------------------------------------------------------------


def test_a_clock_rate_and_a_weber_fraction_recover_from_reproductions() -> None:
    model = DurationReproduction()
    design = reproduction_design(["a"], sessions=2)
    truth = model.parameters_from_components(clock_rate=1.05, weber_fraction=0.18)

    report = run_parameter_recovery(model, design, [dict(truth)], repeats=4, seed=11)

    summaries = {summary.parameter: summary for summary in report.summary()}
    assert abs(summaries["clock_rate_log"].bias) < 0.05
    assert abs(summaries["weber_fraction_log"].bias) < 0.15
    assert summaries["clock_rate_log"].coverage_95 >= 0.75
    assert summaries["weber_fraction_log"].coverage_95 >= 0.75


def test_a_central_tendency_exponent_recovers_when_it_is_estimated() -> None:
    model = DurationReproduction(fixed_central_tendency=None)
    design = reproduction_design(["a"], sessions=3)
    truth = model.parameters_from_components(
        clock_rate=1.0, weber_fraction=0.18, central_tendency=0.8
    )

    report = run_parameter_recovery(model, design, [dict(truth)], repeats=3, seed=13)

    for summary in report.summary():
        assert abs(summary.bias) < 0.2, summary.parameter


def test_a_clock_rate_and_a_weber_fraction_recover_from_bisection_reports() -> None:
    model = bisector()
    design = bisection_design(["a"], sessions=4)
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25)

    report = run_parameter_recovery(model, design, [dict(truth)], repeats=4, seed=17)

    summaries = {summary.parameter: summary for summary in report.summary()}
    assert abs(summaries["clock_rate_log"].bias) < 0.08
    assert abs(summaries["weber_fraction_log"].bias) < 0.2
    assert summaries["clock_rate_log"].coverage_95 >= 0.75


def test_one_weber_fraction_describes_both_paradigms() -> None:
    """Scalar timing's strongest testable claim, and neither paradigm can make it alone.

    The same clock is used to generate a reproduction study and a bisection study; the two
    independent fits are asked to agree about the Weber fraction. Nothing in the two
    likelihoods is shared -- one is a lognormal density on a duration and the other a probit
    on a binary report -- so agreement is evidence that the memory behind them is one memory.
    """

    weber, clock = 0.22, 1.0
    reproduction = DurationReproduction()
    bisection = bisector()
    reproduction_study = reproduction.simulate(
        reproduction_design(["a"], sessions=4),
        reproduction.parameters_from_components(clock_rate=clock, weber_fraction=weber),
        seed=19,
    )
    bisection_study = bisection.simulate(
        bisection_design(["a"], sessions=8),
        bisection.parameters_from_components(clock_rate=clock, weber_fraction=weber),
        seed=23,
    )

    from_durations = reproduction.parameter_components(reproduction.fit(reproduction_study))
    from_reports = bisection.parameter_components(bisection.fit(bisection_study))

    assert from_durations.weber_fraction == pytest.approx(weber, rel=0.15)
    assert from_reports.weber_fraction == pytest.approx(weber, rel=0.2)
    assert from_durations.clock_rate == pytest.approx(from_reports.clock_rate, rel=0.1)


def test_a_reproduction_fit_reports_the_natural_coordinate_beside_the_estimated_one() -> None:
    model = DurationReproduction()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.18)
    study = model.simulate(reproduction_design(["a"], sessions=2), truth, seed=27)

    fit = model.fit(study)

    assert set(fit.derived_values) == {"clock_rate", "weber_fraction"}
    assert fit.derived_values["weber_fraction"] == pytest.approx(
        float(np.exp(fit.estimates[1])), rel=1e-12
    )
    assert fit.derived_quantities["clock_rate"].standard_error > 0.0
    assert model.coefficient_of_variation(fit) == pytest.approx(
        fit.derived_values["weber_fraction"]
    )


def test_a_bisection_fit_reports_its_bisection_point() -> None:
    model = bisector()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25)
    study = model.simulate(bisection_design(["a"], sessions=4), truth, seed=29)

    fit = model.fit(study)

    assert fit.derived_values["bisection_point"] == pytest.approx(GEOMETRIC_MEAN, rel=0.1)
    assert fit.derived_quantities["bisection_point"].standard_error > 0.0
    assert fit.derived_values["bisection_point"] == pytest.approx(model.bisection_point(fit))


# --------------------------------------------------------------------------------------
# 5. A continuous observable, and how it reaches the prediction contract
# --------------------------------------------------------------------------------------


def test_reproduction_predicts_a_density_and_bisection_predicts_a_probability() -> None:
    """The two shapes of observation this module has, and the two contract types for them."""

    reproduction = DurationReproduction()
    bisection = bisector()
    reproduction_study = reproduction.simulate(
        reproduction_design(["a"]),
        reproduction.parameters_from_components(clock_rate=1.0, weber_fraction=0.2),
        seed=31,
    )
    bisection_study = bisection.simulate(
        bisection_design(["a"]),
        bisection.parameters_from_components(clock_rate=1.0, weber_fraction=0.25),
        seed=33,
    )

    density = reproduction.predict(reproduction_study, reproduction.fit(reproduction_study))
    probability = bisection.predict(bisection_study, bisection.fit(bisection_study))

    assert isinstance(density, DensityPrediction)
    assert density.outcome == "reproduced_duration"
    assert not density.is_defective
    assert density.n_observations == len(reproduction_study)
    assert isinstance(probability, Prediction)
    assert probability.n_observations == len(bisection_study)


def test_the_analytic_score_and_the_tabulated_density_agree() -> None:
    """``pointwise_log_prob`` is closed form; the grid is a tabulation of the same closed form.

    They are asserted equal to interpolation error rather than by construction, which is what
    stops a fold's score from depending on ``grid_points``.
    """

    model = DurationReproduction(grid_points=4097)
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.2)
    study = model.simulate(reproduction_design(["a"]), truth, seed=35)
    fit = model.fit(study)

    analytic = model.pointwise_log_prob(study, fit)
    tabulated = model.predict(study, fit).observed_log_density(
        np.asarray(study["reproduced_duration"], dtype=np.float64)
    )

    assert analytic == pytest.approx(tabulated, abs=2e-3)


def test_both_families_satisfy_the_estimator_conformance_harness() -> None:
    reproduction = DurationReproduction()
    bisection = bisector()
    reproduction_study = reproduction.simulate(
        reproduction_design(["a"], sessions=2),
        reproduction.parameters_from_components(clock_rate=1.0, weber_fraction=0.2),
        seed=37,
    )
    bisection_study = bisection.simulate(
        bisection_design(["a"], sessions=2),
        bisection.parameters_from_components(clock_rate=1.0, weber_fraction=0.25),
        seed=39,
    )

    assert_behaviour_estimator_conforms(reproduction, reproduction_study, seed=1)
    assert_behaviour_estimator_conforms(bisection, bisection_study, seed=1)


# --------------------------------------------------------------------------------------
# 6. The composition contract, and which combinators apply
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", [DurationReproduction(), TemporalBisection()])
def test_both_families_compose_through_the_bounded_coordinate_contract(model: object) -> None:
    assert isinstance(model, BoundedCoordinateEstimator)
    assert require_composable(model, combinator="smooth") == BOUNDED_COORDINATE
    assert require_composable(model, combinator="hierarchical") == BOUNDED_COORDINATE


@pytest.mark.parametrize("model", [DurationReproduction(), TemporalBisection()])
def test_the_declared_box_is_finite_on_every_coordinate(model: object) -> None:
    study = (
        reproduction_design(["a"])
        if isinstance(model, DurationReproduction)
        else bisection_design(["a"])
    )
    box = model.coordinate_box(study)

    assert box.shape == (len(model.parameter_names), 2)
    assert np.all(np.isfinite(box)) and np.all(box[:, 1] > box[:, 0])


def test_hierarchy_recovers_a_per_subject_weber_fraction() -> None:
    """The wanted model, obtained for zero lines of combinator code."""

    model = DurationReproduction()
    pooled = hierarchical(model, over="subject", parameters=("weber_fraction_log",), scale=0.5)
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.18)
    simulation = pooled.simulate_with_effects(
        reproduction_design(["a", "b", "c", "d"], sessions=2), truth, seed=41
    )

    fit = pooled.fit(simulation.study)

    assert fit.groups == ("a", "b", "c", "d")
    assert fit.varying_parameters == ("weber_fraction_log",)
    correlation = np.corrcoef(simulation.group_deviations.ravel(), fit.group_deviations.ravel())[
        0, 1
    ]
    assert correlation > 0.8


def test_hierarchy_works_on_bisection_too() -> None:
    model = bisector()
    pooled = hierarchical(model, over="subject", parameters=("weber_fraction_log",), scale=0.4)
    simulation = pooled.simulate_with_effects(
        bisection_design(["a", "b", "c"], sessions=3),
        model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25),
        seed=43,
    )

    fit = pooled.fit(simulation.study)

    assert fit.group_deviations.shape == (3, 1)
    assert np.all(np.isfinite(fit.group_deviations))


def test_a_group_deviation_keeps_a_weber_fraction_positive() -> None:
    """A Gaussian on the fraction itself is a negative fraction sometimes; on its log, never."""

    model = DurationReproduction()
    deviations = model.draw_group_deviations(
        np.asarray([1], dtype=np.intp),
        np.asarray([1.5]),
        groups=400,
        generator=np.random.default_rng(0),
    ).ravel()

    assert np.all(0.18 * np.exp(deviations) > 0.0)
    assert np.any(0.18 + deviations <= 0.0)


def test_smoothness_recovers_a_clock_rate_that_drifts_across_sessions() -> None:
    model = DurationReproduction()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 5.0),
        parameters=("clock_rate_log",),
        smoothness=1.0,
    )
    paths = drifting.parameters_from_paths(
        {
            "clock_rate_log": (float(np.log(0.7)), float(np.log(1.4))),
            "weber_fraction_log": float(np.log(0.18)),
        }
    )
    study = drifting.simulate(reproduction_design(["a"], sessions=6), paths, seed=45)

    fit = drifting.fit(study)

    assert drifting.parameter_names == (
        "clock_rate_log[session_order=0]",
        "clock_rate_log[session_order=5]",
        "weber_fraction_log",
    )
    values = dict(zip(drifting.parameter_names, fit.estimates, strict=True))
    early = float(np.exp(values["clock_rate_log[session_order=0]"]))
    late = float(np.exp(values["clock_rate_log[session_order=5]"]))
    assert early < late
    assert 0.6 < early < 0.9
    assert 1.2 < late < 1.6


def test_a_lapse_on_a_bisection_model_recovers_the_rate_and_the_weight() -> None:
    """``mix()`` reaches this family because a bisection report is a binary choice."""

    model = bisector()
    lapsing = mix(model, UniformChoiceGuess(), weight_bounds=(0.0, 0.3))
    truth = lapsing.from_natural({"clock_rate": 1.0, "weber_fraction": 0.25, "lapse_rate": 0.12})
    simulation = lapsing.simulate_with_component(
        bisection_design(["a"], sessions=12), truth, seed=47
    )

    fit = lapsing.fit(simulation.study)
    recovered = lapsing.to_natural(fit.estimates)

    assert lapsing.parameter_names == (
        "clock_rate_log",
        "weber_fraction_log",
        MIXTURE_LOGIT,
    )
    assert lapsing.natural_names == ("clock_rate", "weber_fraction", "lapse_rate")
    assert recovered["lapse_rate"] == pytest.approx(0.12, abs=0.06)
    assert recovered["clock_rate"] == pytest.approx(1.0, rel=0.15)
    assert fit.diagnostics.converged
    responsibility = lapsing.component_responsibility(simulation.study, fit)
    assert np.all((responsibility >= 0) & (responsibility <= 1))


def test_the_full_stack_composes_over_a_bisection_row_objective() -> None:
    """``hierarchical(smooth(mix(model)))``, on a family with no linear predictor at all."""

    lapsing = mix(bisector(), UniformChoiceGuess(), weight_bounds=(0.0, 0.4))
    stack = hierarchical(
        smooth(lapsing, over="session_order", knots=(0.0, 5.0), parameters=("clock_rate_log",)),
        over="subject",
        parameters=(MIXTURE_LOGIT,),
        scale=0.5,
        estimate_scale=False,
    )
    truth = dict(
        stack.model.parameters_from_paths(
            {
                "clock_rate_log": [float(np.log(0.85)), float(np.log(1.2))],
                "weber_fraction_log": float(np.log(0.25)),
                MIXTURE_LOGIT: -2.0,
            }
        )
    )
    study = stack.simulate(bisection_design(["a", "b", "c"], sessions=6), truth, seed=49)

    fit = stack.fit(study)

    assert stack.parameter_names[-1] == MIXTURE_LOGIT
    assert fit.varying_parameters == (MIXTURE_LOGIT,)
    assert fit.group_deviations.shape == (3, 1)
    assert np.all(np.isfinite(stack.pointwise_log_prob(study, fit)))


def test_a_continuous_outcome_has_no_shipped_mixture_component() -> None:
    """The one cell that does not open, and the reason is a component rather than a combinator.

    ``mix()`` itself is untouched and would work: a reproduction's rows are independent and
    the weight would ride in one extra column of the row coordinate exactly as it does for a
    discounting model. What is missing is a component that scores a **bare duration** --
    ``UniformChoiceGuess`` writes a binary choice and ``UniformResponseGuess`` writes a joint
    choice and latency, and ``require_mixable`` refuses both by comparing scored columns
    before any arithmetic happens.
    """

    with pytest.raises(TypeError, match="not a mixture of one observation"):
        mix(DurationReproduction(), UniformChoiceGuess())


def test_a_within_session_clock_is_admissible_for_a_timing_model() -> None:
    """These rows are independent, so ``smooth(model, over="trial")`` is defined."""

    model = DurationReproduction()
    design = reproduction_design(["a"], repeats=4)
    study = model.simulate(
        design, model.parameters_from_components(clock_rate=1.0, weber_fraction=0.2), seed=51
    )
    assert np.array_equal(model.row_objective(study).row_blocks, np.arange(len(design)))

    drifting = smooth(
        model,
        over="trial",
        knots=(0.0, float(len(design) - 1)),
        parameters=("clock_rate_log",),
        smoothness=2.0,
    )

    fit = drifting.fit(study)

    assert fit.diagnostics.converged
    assert len(fit.estimates) == 3


# --------------------------------------------------------------------------------------
# 7. The identifiability hazards, reported before the fit
# --------------------------------------------------------------------------------------


def test_a_narrow_target_range_cannot_see_the_scalar_property() -> None:
    study = with_reproductions(reproduction_design(["a"], targets=(2.0, 2.2), repeats=5))

    findings = DurationReproduction().describe(study).findings

    assert "narrow_target_range" in {finding.code for finding in findings}
    assert all(finding.severity == "warning" for finding in findings)


def test_one_target_duration_leaves_the_central_tendency_unidentified() -> None:
    study = with_reproductions(reproduction_design(["a"], targets=(4.0,), repeats=10))

    findings = DurationReproduction(fixed_central_tendency=None).describe(study).findings
    codes = {finding.code for finding in findings}

    assert "unidentified_central_tendency" in codes
    assert "narrow_target_range" in codes
    # Fixed at one, the exponent is not a parameter and the finding does not apply.
    assert "unidentified_central_tendency" not in {
        finding.code for finding in DurationReproduction().describe(study).findings
    }


def test_two_target_durations_leave_the_central_tendency_weakly_identified() -> None:
    study = with_reproductions(reproduction_design(["a"], targets=(1.0, 8.0), repeats=10))

    findings = DurationReproduction(fixed_central_tendency=None).describe(study).findings

    assert "weakly_identified_central_tendency" in {finding.code for finding in findings}


def test_a_wide_target_range_reports_nothing() -> None:
    study = with_reproductions(reproduction_design(["a"], repeats=5))

    assert DurationReproduction().describe(study).findings == ()


def test_too_few_probes_cannot_separate_a_bisection_location_from_a_slope() -> None:
    study = with_reports(bisection_design(["a"], probes=(3.0, 5.0), repeats=10))

    findings = bisector().describe(study).findings

    assert "too_few_probe_durations" in {finding.code for finding in findings}


def test_probes_on_one_side_make_the_bisection_point_an_extrapolation() -> None:
    study = with_reports(bisection_design(["a"], probes=(4.5, 5.5, 6.5), repeats=10))

    findings = bisector().describe(study).findings

    assert "probes_do_not_span_the_comparison" in {finding.code for finding in findings}


def test_probes_outside_the_anchors_are_reported() -> None:
    study = with_reports(bisection_design(["a"], probes=(1.0, 4.0, 12.0), repeats=10))

    findings = bisector().describe(study).findings

    assert "probes_outside_the_anchors" in {finding.code for finding in findings}


def test_close_anchors_cannot_tell_the_two_decision_rules_apart() -> None:
    model = TemporalBisection(short_anchor=4.0, long_anchor=5.0)
    study = with_reports(bisection_design(["a"], probes=(4.0, 4.5, 5.0), repeats=10))

    findings = model.describe(study).findings

    assert "narrow_anchor_ratio" in {finding.code for finding in findings}


def test_a_composed_model_carries_the_findings_through() -> None:
    pooled = hierarchical(
        DurationReproduction(), over="subject", parameters=("weber_fraction_log",)
    )
    study = with_reproductions(reproduction_design(["a"], targets=(2.0, 2.2), repeats=5))

    findings = pooled.describe(study).findings

    assert "narrow_target_range" in {finding.code for finding in findings}


def test_deterministic_reproductions_land_the_weber_fraction_on_its_box() -> None:
    model = DurationReproduction()
    design = reproduction_design(["a"], repeats=5)
    columns = {name: design[name] for name in design.columns}
    columns["reproduced_duration"] = np.asarray(design["target_duration"], dtype=np.float64)

    fit = model.fit(Study(columns))

    assert fit.diagnostics.boundary_estimate


# --------------------------------------------------------------------------------------
# 8. These are ordinary estimators
# --------------------------------------------------------------------------------------


def test_the_families_flow_through_evaluate_splits_and_compare_models() -> None:
    model = bisector()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25)
    study = model.simulate(bisection_design(["a", "b"], sessions=4), truth, seed=53)
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    evaluations = evaluate_splits(model, study, splits)
    comparison = compare_models(
        {"bisection": model, "guessing": BernoulliHistoryGLM(predictors=(), l2=1.0)},
        study,
        splits,
        bootstrap_resamples=50,
    )

    assert len(evaluations) == len(splits)
    assert all(
        np.all(np.isfinite(evaluation.pointwise_log_probability)) for evaluation in evaluations
    )
    assert comparison.winner == "bisection"


def test_a_reproduction_model_is_evaluated_on_its_density() -> None:
    model = DurationReproduction()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.18)
    study = model.simulate(reproduction_design(["a", "b"], sessions=4), truth, seed=55)
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    evaluations = evaluate_splits(model, study, splits)

    assert len(evaluations) == len(splits)
    assert all(
        np.all(np.isfinite(evaluation.pointwise_log_probability)) for evaluation in evaluations
    )


def _two_reproduction_candidates() -> tuple[Any, Any, Any, Any]:
    model = DurationReproduction()
    free = DurationReproduction(fixed_central_tendency=None)
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.18)
    study = model.simulate(reproduction_design(["a", "b"], sessions=3), truth, seed=56)
    return model, free, study, cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)


def test_a_declared_brier_column_is_refused_by_name_before_any_fold_is_fitted() -> None:
    """The default table still refuses, and now refuses at declaration.

    ``compare_models`` carries a Brier column by default, and a Brier score needs a discrete
    margin. An unlabelled density has none, so the comparison refuses instead of inventing
    one -- but it refuses from the candidate's own declared
    :attr:`~behavio.contracts.ModelCapabilities.score_metrics`, before a single fold is
    fitted, and the message names both the candidate and the rule that made it impossible.
    """

    from behavio.compare.models import UnscoreableByBrier

    model, free, study, splits = _two_reproduction_candidates()

    with pytest.raises(UnscoreableByBrier, match="no categorical margin") as refusal:
        compare_models(
            {"scalar": model, "vierordt": free},
            study,
            splits,
            bootstrap_resamples=10,
            outcome_column="reproduced_duration",
        )
    assert "'scalar'" in str(refusal.value) and "'brier'" in str(refusal.value)


def test_two_reproduction_candidates_are_ranked_on_a_declared_log_score() -> None:
    """The comparison table these families were locked out of, now reachable.

    Declaring the log score alone is what SDR-0063 called the missing half: it is the joint
    log density of the whole observation, it is defined for a density with no discrete
    margin, and it is what ranks the two candidates here. The report says which rule decided
    -- in ``ranked_by``, in the winner key and in the winner policy -- because a winner on
    the log score and a winner on the Brier score are different claims.
    """

    model, free, study, splits = _two_reproduction_candidates()

    report = compare_models(
        {"scalar": model, "vierordt": free},
        study,
        splits,
        bootstrap_resamples=64,
        outcome_column="reproduced_duration",
        metrics=(ScoreMetric.LOG_LOSS,),
    )

    assert report.metrics == (ScoreMetric.LOG_LOSS,)
    assert report.ranked_by is ScoreMetric.LOG_LOSS
    assert report.winner in {"scalar", "vierordt"}
    assert report.model_order == ("scalar", "vierordt")
    payload = report.to_dict()
    assert payload["declared_metrics"] == ["log-loss"]
    assert payload["winner_policy"] == "lowest unit-balanced log loss among non-failed audits"
    assert payload["winner_by_unit_balanced_log_loss"] == report.winner
    assert set(payload["models"]["scalar"]["unit_scores"][0]) == {"unit", "log_loss"}
    # A log density, not a log probability: a sharply peaked duration density can exceed
    # one, so the column this table carries is not bounded below by zero.
    assert np.all(np.isfinite(report.result_for("scalar").unit_log_losses))
    with pytest.raises(UndeclaredMetric, match="carries no 'brier' column"):
        _ = report.result_for("scalar").unit_brier_scores
    contrast = report.comparison_for("scalar", "vierordt")
    assert contrast.metric is ScoreMetric.LOG_LOSS
    assert contrast.left_minus_right.estimate == pytest.approx(
        report.result_for("scalar").unit_balanced_log_loss
        - report.result_for("vierordt").unit_balanced_log_loss
    )


# --------------------------------------------------------------------------------------
# 9. Configuration and refusals
# --------------------------------------------------------------------------------------


def test_a_non_positive_natural_parameter_cannot_be_encoded() -> None:
    with pytest.raises(ValueError, match="weber_fraction must be finite and positive"):
        DurationReproduction().parameters_from_components(clock_rate=1.0, weber_fraction=0.0)


def test_a_fixed_central_tendency_leaves_the_coordinate() -> None:
    model = DurationReproduction()

    assert model.parameter_names == ("clock_rate_log", "weber_fraction_log")
    assert DurationReproduction(fixed_central_tendency=None).parameter_names == (
        "clock_rate_log",
        "weber_fraction_log",
        "central_tendency_log",
    )
    components = model.parameter_components(
        model.parameters_from_components(clock_rate=1.0, weber_fraction=0.2)
    )
    assert components.central_tendency == 1.0


def test_supplying_a_fixed_central_tendency_with_the_wrong_value_is_refused() -> None:
    with pytest.raises(ValueError, match=r"central_tendency is fixed at 1\.0"):
        DurationReproduction().parameters_from_components(
            clock_rate=1.0, weber_fraction=0.2, central_tendency=0.8
        )


def test_a_non_positive_duration_is_refused() -> None:
    model = DurationReproduction()
    design = reproduction_design(["a"], repeats=2)
    columns = {name: design[name] for name in design.columns}
    columns["reproduced_duration"] = np.zeros(len(design))

    with pytest.raises(ModelDataError, match="must be strictly positive"):
        model.read_problem(Study(columns))


def test_anchors_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="short_anchor must be strictly below long_anchor"):
        TemporalBisection(short_anchor=8.0, long_anchor=2.0)


def test_a_bisection_outcome_must_be_binary() -> None:
    model = bisector()
    design = bisection_design(["a"], repeats=2)
    columns = {name: design[name] for name in design.columns}
    columns["choice"] = np.full(len(design), 2)

    with pytest.raises(ModelDataError, match="only zero and one"):
        model.outcomes(Study(columns))


def test_a_missing_column_is_a_model_data_error_and_a_finding() -> None:
    study = reproduction_design(["a"], repeats=2)
    incomplete = Study({name: study[name] for name in study.columns if name != "target_duration"})

    assert "missing_column" in {
        finding.code for finding in DurationReproduction().describe(incomplete).findings
    }
    with pytest.raises(ModelDataError, match="target_duration"):
        DurationReproduction().fit(incomplete)


def test_a_fit_from_another_specification_is_refused() -> None:
    model = bisector()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.25)
    study = model.simulate(bisection_design(["a"]), truth, seed=57)
    fit = model.fit(study)

    with pytest.raises(ValueError, match="different model specification"):
        bisector(BisectionRule.DIFFERENCE).predict(study, fit)


def test_the_smoothed_prediction_mode_is_refused_by_name() -> None:
    from behavio.models import UnsupportedPredictionMode

    model = DurationReproduction()
    truth = model.parameters_from_components(clock_rate=1.0, weber_fraction=0.2)
    study = model.simulate(reproduction_design(["a"], repeats=2), truth, seed=59)
    fit = model.fit(study)

    with pytest.raises(UnsupportedPredictionMode, match="only filtered prediction"):
        model.predict(study, fit, mode="smoothed")
