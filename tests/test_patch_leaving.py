"""Patch leaving, checked against Charnov's closed form rather than against itself.

Four groups of tests here would still pass a plausible but wrong implementation.

The **marginal value theorem** is asserted three ways that do not share code: against the
exact algebra for the hyperbolic gain, where :math:`t^{*} = \\sqrt{h\\tau}` and
:math:`R^{*} = A/(\\sqrt{h}+\\sqrt{\\tau})^2`; against the implicit equation
:math:`\\rho(\\tau + t^{*}) = e^{\\rho t^{*}} - 1` for the exponential gain; and against a brute
force grid maximisation of :math:`g(t)/(\\tau + t)` for both. Charnov's central comparative
static -- a longer travel time buys a longer stay -- is asserted as a sign, and the tangent
condition :math:`g'(t^{*}) = R^{*}` is asserted for every patch type of a heterogeneous
environment against one shared rate.

The **simulator reproduces that closed form**: with the giving-up rate set to :math:`R^{*}`
the noise-free leaving time is exactly :math:`t^{*}`, and as the decision noise goes to zero
the simulated residence times converge on it.

**Censoring is exercised rather than assumed.** A censored row's score is checked against an
independently written survival probability, ignoring the censoring is shown to bias the fitted
threshold in the direction it must, a duration longer than its declared limit is refused, and
the one place the prediction contract cannot follow the likelihood is asserted as a
disagreement rather than papered over.

The **analytic gradient** is checked against a central difference with censored rows present,
because the censored branch is a different formula and a wrong one converges quietly.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from behavio import Study, compare_models, evaluate_splits, run_parameter_recovery
from behavio.adapters.estimator_conformance import assert_behaviour_estimator_conforms
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth
from behavio.contracts.bounded import (
    BOUNDED_COORDINATE,
    BoundedCoordinateEstimator,
    require_composable,
)
from behavio.contracts.estimator import DensityPrediction
from behavio.evaluate import cohort_forward_session_splits
from behavio.models import ModelDataError
from behavio.models.patch_leaving import (
    GainFunction,
    PatchLeaving,
    instantaneous_intake_rate,
    marginal_value_rate,
    marginal_value_residence_time,
    patch_gain,
)

#: Three patch types, so the study can tell a rate threshold from a time threshold.
PATCHES = ((10.0, 2.0), (20.0, 4.0), (6.0, 1.0))
TRAVEL_TIME = 8.0


# --------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------


def patch_design(
    subjects: Sequence[str],
    *,
    sessions: int = 1,
    patches: Sequence[tuple[float, float]] = PATCHES,
    repeats: int = 25,
    travel_time: float = TRAVEL_TIME,
    session_cap: float | None = None,
) -> Study:
    """One row per patch visit, with the patch's own depletion schedule declared."""

    names = [
        "subject",
        "session",
        "trial",
        "session_order",
        "patch_yield",
        "patch_decay",
        "travel_time",
    ]
    if session_cap is not None:
        names.append("session_cap")
    columns: dict[str, list[object]] = {name: [] for name in names}
    for subject in subjects:
        trial = 0
        for session in range(sessions):
            for (amount, decay), _ in itertools.product(patches, range(repeats)):
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-s{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["patch_yield"].append(float(amount))
                columns["patch_decay"].append(float(decay))
                columns["travel_time"].append(float(travel_time))
                if session_cap is not None:
                    columns["session_cap"].append(float(session_cap))
                trial += 1
    return Study(columns)


def with_residence(study: Study, value: float = 3.0) -> Study:
    """Attach an arbitrary residence time, for tests that only read ``describe()``."""

    columns = {name: study[name] for name in study.columns}
    columns["residence_time"] = np.linspace(1.0, value, len(study))
    return Study(columns)


def forager(
    gain: GainFunction = GainFunction.HYPERBOLIC, *, censoring: bool = False
) -> PatchLeaving:
    return PatchLeaving(
        gain=gain,
        travel_time_column="travel_time",
        censoring_time_column="session_cap" if censoring else None,
    )


def environment_rate(study: Study, *, gain: GainFunction = GainFunction.HYPERBOLIC) -> float:
    return marginal_value_rate(
        np.asarray(study["patch_yield"], dtype=np.float64),
        np.asarray(study["patch_decay"], dtype=np.float64),
        TRAVEL_TIME,
        gain=gain,
    )


# --------------------------------------------------------------------------------------
# 1. Known answers: the marginal value theorem
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("handling", [0.5, 2.0, 7.0])
@pytest.mark.parametrize("travel", [1.0, 8.0, 40.0])
def test_the_hyperbolic_optimum_is_the_square_root_of_handling_times_travel(
    handling: float, travel: float
) -> None:
    """Charnov's rule has an exact solution for Holling's disc equation, and this is it."""

    amount = 12.0
    optimal = marginal_value_residence_time(amount, handling, travel, gain="hyperbolic")
    rate = marginal_value_rate(amount, handling, travel, gain="hyperbolic")

    assert float(optimal[0]) == pytest.approx(np.sqrt(handling * travel), rel=1e-9)
    assert rate == pytest.approx(amount / (np.sqrt(handling) + np.sqrt(travel)) ** 2, rel=1e-9)


@pytest.mark.parametrize("decay", [0.1, 0.3, 1.2])
@pytest.mark.parametrize("travel", [2.0, 10.0, 50.0])
def test_the_exponential_optimum_satisfies_charnovs_implicit_equation(
    decay: float, travel: float
) -> None:
    """Substituting the tangent condition into the exponential gain gives one equation.

    :math:`\\rho(\\tau + t^{*}) = e^{\\rho t^{*}} - 1`, derived by hand and asserted as a
    residual, so the root finder is checked against algebra rather than against itself.
    """

    optimal = float(marginal_value_residence_time(25.0, decay, travel, gain="exponential")[0])

    residual = decay * (travel + optimal) - np.expm1(decay * optimal)

    assert residual == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("gain", ["exponential", "hyperbolic"])
def test_the_optimum_maximises_the_long_run_rate_by_brute_force(gain: str) -> None:
    """The tangent construction, checked against a grid search that knows no calculus."""

    amount, decay, travel = 15.0, 2.0 if gain == "hyperbolic" else 0.4, 6.0
    optimal = float(marginal_value_residence_time(amount, decay, travel, gain=gain)[0])
    rate = marginal_value_rate(amount, decay, travel, gain=gain)

    grid = np.linspace(1e-6, 8.0 * max(optimal, 1.0), 2_000_001)
    rates = patch_gain(grid, patch_yield=amount, patch_decay=decay, gain=gain) / (travel + grid)
    best = float(grid[int(np.argmax(rates))])

    assert best == pytest.approx(optimal, rel=1e-3)
    assert float(np.max(rates)) == pytest.approx(rate, rel=1e-6)
    assert instantaneous_intake_rate(
        optimal, patch_yield=amount, patch_decay=decay, gain=gain
    ) == pytest.approx(rate, rel=1e-8)


@pytest.mark.parametrize("gain", ["exponential", "hyperbolic"])
def test_a_longer_journey_buys_a_longer_stay(gain: str) -> None:
    """Charnov's central comparative static, asserted as a sign rather than a number."""

    decay = 2.0 if gain == "hyperbolic" else 0.4
    stays = [
        float(marginal_value_residence_time(15.0, decay, travel, gain=gain)[0])
        for travel in (1.0, 4.0, 16.0, 64.0)
    ]

    assert all(later > earlier for earlier, later in itertools.pairwise(stays))


@pytest.mark.parametrize("gain", ["exponential", "hyperbolic"])
def test_a_faster_depleting_patch_is_left_sooner(gain: str) -> None:
    stays = [
        float(marginal_value_residence_time(15.0, decay, 8.0, gain=gain)[0])
        for decay in ((0.5, 1.0, 2.0, 4.0) if gain == "exponential" else (8.0, 4.0, 2.0, 1.0))
    ]

    assert all(later < earlier for earlier, later in itertools.pairwise(stays))


@pytest.mark.parametrize("gain", ["exponential", "hyperbolic"])
def test_a_single_patch_types_optimum_does_not_depend_on_its_richness(gain: str) -> None:
    """A discriminating prediction, and one both gain functions happen to make.

    :math:`g'/g` has no :math:`A` in it for either form, so a homogeneous environment's
    optimal residence time is set by the depletion schedule and the travel time alone.
    Richness matters only when patches *differ*, which the next test is about.
    """

    decay = 2.0 if gain == "hyperbolic" else 0.4
    stays = [
        float(marginal_value_residence_time(amount, decay, 8.0, gain=gain)[0])
        for amount in (2.0, 15.0, 400.0)
    ]

    assert stays == pytest.approx([stays[0]] * 3, rel=1e-8)


def test_a_heterogeneous_environment_has_one_rate_and_several_optimal_stays() -> None:
    """The theorem is about an environment: one threshold rate, many residence times."""

    amounts = np.asarray([amount for amount, _ in PATCHES])
    decays = np.asarray([decay for _, decay in PATCHES])

    rate = marginal_value_rate(amounts, decays, TRAVEL_TIME, gain="hyperbolic")
    stays = marginal_value_residence_time(amounts, decays, TRAVEL_TIME, gain="hyperbolic")

    assert len(np.unique(np.round(stays, 9))) == len(PATCHES)
    for amount, decay, stay in zip(amounts, decays, stays, strict=True):
        assert instantaneous_intake_rate(
            stay, patch_yield=amount, patch_decay=decay, gain="hyperbolic"
        ) == pytest.approx(rate, rel=1e-8)


def test_enriching_one_patch_type_shortens_the_stay_in_the_others() -> None:
    """Richness enters through the environment, which is the whole point of :math:`R^{*}`."""

    poor = marginal_value_residence_time([6.0, 6.0], [1.0, 4.0], TRAVEL_TIME, gain="hyperbolic")
    rich = marginal_value_residence_time([60.0, 6.0], [1.0, 4.0], TRAVEL_TIME, gain="hyperbolic")

    assert float(rich[1]) < float(poor[1])
    assert marginal_value_rate(
        [60.0, 6.0], [1.0, 4.0], TRAVEL_TIME, gain="hyperbolic"
    ) > marginal_value_rate([6.0, 6.0], [1.0, 4.0], TRAVEL_TIME, gain="hyperbolic")


# --------------------------------------------------------------------------------------
# 2. The simulator reproduces the closed form
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("gain", [GainFunction.EXPONENTIAL, GainFunction.HYPERBOLIC])
def test_the_optimal_threshold_gives_the_optimal_residence_time(gain: GainFunction) -> None:
    """Set the giving-up rate to Charnov's :math:`R^{*}` and the model leaves at :math:`t^{*}`."""

    patches = PATCHES if gain is GainFunction.HYPERBOLIC else ((10.0, 0.4), (20.0, 0.2))
    model = forager(gain)
    design = patch_design(["a"], patches=patches, repeats=1)
    optimal_rate = environment_rate(design, gain=gain)
    parameters = model.parameters_from_components(giving_up_rate=optimal_rate, decision_noise=0.05)

    noise_free = model.deterministic_residence_time(design, parameters)
    expected = marginal_value_residence_time(
        np.asarray(design["patch_yield"], dtype=np.float64),
        np.asarray(design["patch_decay"], dtype=np.float64),
        TRAVEL_TIME,
        gain=gain,
    )

    assert noise_free == pytest.approx(expected, rel=1e-9)


def test_a_nearly_noiseless_forager_simulates_the_marginal_value_residence_time() -> None:
    """The check the simulator itself has to pass, not only the arithmetic beside it."""

    model = forager()
    design = patch_design(["a"], repeats=40)
    optimal_rate = environment_rate(design)
    expected = marginal_value_residence_time(
        np.asarray(design["patch_yield"], dtype=np.float64),
        np.asarray(design["patch_decay"], dtype=np.float64),
        TRAVEL_TIME,
        gain="hyperbolic",
    )

    study = model.simulate(
        design,
        model.parameters_from_components(giving_up_rate=optimal_rate, decision_noise=1e-3),
        seed=0,
    )

    observed = np.asarray(study["residence_time"], dtype=np.float64)
    assert np.max(np.abs(observed - expected) / expected) < 0.01


def test_the_median_residence_time_is_the_threshold_crossing_at_realistic_noise() -> None:
    """With noise the leaving time is a distribution, and its median is where the rule bites."""

    model = forager()
    design = patch_design(["a"], repeats=4000)
    parameters = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.3)
    crossing = model.deterministic_residence_time(design, parameters)
    study = model.simulate(design, parameters, seed=1)

    times = np.asarray(study["residence_time"], dtype=np.float64)
    for decay in np.unique(np.asarray(design["patch_decay"], dtype=np.float64)):
        rows = np.asarray(design["patch_decay"], dtype=np.float64) == decay
        assert float(np.median(times[rows])) == pytest.approx(
            float(np.median(crossing[rows])), rel=0.03
        )


def test_the_predicted_density_integrates_to_one_and_agrees_with_the_simulator() -> None:
    """The survival function conditions on entry, so the leaving time is a proper density."""

    model = forager()
    design = patch_design(["a"], patches=(PATCHES[0],), repeats=1)
    parameters = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.35)
    study = model.simulate(
        patch_design(["a"], patches=(PATCHES[0],), repeats=6000), parameters, seed=2
    )
    fit = model.fit(study)
    prediction = model.predict(design, fit)

    grid = np.asarray(prediction.grid, dtype=np.float64)
    density = np.asarray(prediction.density, dtype=np.float64)[0]
    assert float(np.trapezoid(density, grid)) == pytest.approx(1.0, abs=2e-3)

    # The tabulated density's median matches the simulated one.
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(grid) * 0.5 * (density[1:] + density[:-1]))]
    )
    median = float(np.interp(0.5, cumulative, grid))
    assert median == pytest.approx(float(np.median(study["residence_time"])), rel=0.05)


def test_the_leaving_hazard_rises_as_the_patch_depletes() -> None:
    """The shape that makes this a giving-up rule rather than a fixed-duration one."""

    model = forager(GainFunction.EXPONENTIAL)
    design = patch_design(["a"], patches=((10.0, 0.4),), repeats=1)
    parameters = model.parameters_from_components(giving_up_rate=1.0, decision_noise=0.4)

    times = np.linspace(0.1, 20.0, 40)
    hazards = np.asarray(
        [
            float(model.leaving_hazard(design, parameters, times=np.asarray([time]))[0])
            for time in times
        ]
    )

    assert np.all(np.diff(hazards) > 0.0)
    assert float(hazards[-1]) < float(np.asarray(design["patch_decay"])[0]) / 0.4


# --------------------------------------------------------------------------------------
# 3. The analytic gradient
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


@pytest.mark.parametrize("gain", [GainFunction.EXPONENTIAL, GainFunction.HYPERBOLIC])
@pytest.mark.parametrize("censoring", [False, True])
def test_the_gradient_matches_a_central_difference(gain: GainFunction, censoring: bool) -> None:
    """Both branches of the censored likelihood, on both gain functions."""

    patches = PATCHES if gain is GainFunction.HYPERBOLIC else ((10.0, 0.4), (20.0, 0.2))
    model = forager(gain, censoring=censoring)
    design = patch_design(["a"], patches=patches, session_cap=6.0 if censoring else None)
    study = model.simulate(
        design,
        model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.35),
        seed=3,
    )
    problem = model.read_problem(study)
    if censoring:
        assert float(np.mean(problem.durations.censored)) > 0.05
    vector = np.log(np.asarray([0.52, 0.29]))

    _, gradient = model.shared_value_and_gradient(problem, vector)

    assert gradient == pytest.approx(numeric_gradient(model, problem, vector), abs=1e-5)


# --------------------------------------------------------------------------------------
# 4. Censoring, handled explicitly
# --------------------------------------------------------------------------------------


def test_a_censored_row_is_scored_by_an_independently_written_survival_probability() -> None:
    """The likelihood of a visit still in progress is ``log S(c)``, and here is ``S(c)``."""

    model = forager(censoring=True)
    design = patch_design(["a"], session_cap=5.0, repeats=10)
    parameters = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.35)
    study = model.simulate(design, parameters, seed=4)
    fit = model.fit(study)
    scores = model.pointwise_log_prob(study, fit)

    amounts = np.asarray(study["patch_yield"], dtype=np.float64)
    decays = np.asarray(study["patch_decay"], dtype=np.float64)
    times = np.asarray(study["residence_time"], dtype=np.float64)
    censored = times >= 5.0 - 1e-9
    threshold, noise = np.exp(fit.estimates[0]), np.exp(fit.estimates[1])

    def logistic(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-value))

    entry_rate = amounts * decays / decays**2
    observed_rate = amounts * decays / (times + decays) ** 2
    survival = logistic((np.log(observed_rate) - np.log(threshold)) / noise) / logistic(
        (np.log(entry_rate) - np.log(threshold)) / noise
    )

    assert np.any(censored)
    assert scores[censored] == pytest.approx(np.log(survival[censored]), rel=1e-9)


def test_ignoring_censoring_biases_the_giving_up_rate_upwards() -> None:
    """Why the declaration exists: a truncated visit read as a departure looks like a
    forager who left sooner, so the fitted threshold moves up towards a richer patch."""

    honest = forager(censoring=True)
    naive = forager()
    design = patch_design(["a"], session_cap=4.0, sessions=2)
    truth = honest.parameters_from_components(giving_up_rate=0.35, decision_noise=0.35)
    study = honest.simulate(design, truth, seed=5)
    censored_share = float(np.mean(np.asarray(study["residence_time"]) >= 4.0 - 1e-9))

    honest_rate = honest.parameter_components(honest.fit(study)).giving_up_rate
    naive_rate = naive.parameter_components(naive.fit(study)).giving_up_rate

    assert censored_share > 0.15
    assert honest_rate == pytest.approx(0.35, rel=0.2)
    assert naive_rate > honest_rate * 1.1


def test_parameters_recover_from_a_censored_study() -> None:
    model = forager(censoring=True)
    design = patch_design(["a"], session_cap=6.0, sessions=2)
    truth = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.35)

    report = run_parameter_recovery(model, design, [dict(truth)], repeats=4, seed=7)

    summaries = {summary.parameter: summary for summary in report.summary()}
    assert abs(summaries["giving_up_rate_log"].bias) < 0.15
    assert abs(summaries["decision_noise_log"].bias) < 0.2
    assert summaries["giving_up_rate_log"].coverage_95 >= 0.75


def test_a_residence_time_longer_than_its_declared_limit_is_refused() -> None:
    model = forager(censoring=True)
    design = patch_design(["a"], session_cap=4.0, repeats=3)
    columns = {name: design[name] for name in design.columns}
    columns["residence_time"] = np.full(len(design), 9.0)

    with pytest.raises(ModelDataError, match=r"exceeds its .* value"):
        model.read_problem(Study(columns))


def test_the_predicted_density_and_the_pointwise_score_disagree_on_a_censored_row() -> None:
    """The one place the prediction contract cannot follow the likelihood, asserted.

    ``predict`` returns a density of the *leaving time*, which is what the model claims about
    every row. A censored row's likelihood is a survival probability, and no member of
    ``ModelPrediction`` can carry one. So the two agree exactly where the observation is an
    event and deliberately differ where it is not -- and a consumer that scores the density
    instead of asking the model would misscore exactly the censored rows.
    """

    model = PatchLeaving(
        gain=GainFunction.HYPERBOLIC,
        travel_time_column="travel_time",
        censoring_time_column="session_cap",
        grid_points=8193,
    )
    design = patch_design(["a"], session_cap=5.0, repeats=10)
    parameters = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.35)
    study = model.simulate(design, parameters, seed=8)
    fit = model.fit(study)

    scores = model.pointwise_log_prob(study, fit)
    density = model.predict(study, fit)
    tabulated = density.observed_log_density(np.asarray(study["residence_time"], dtype=np.float64))
    censored = np.asarray(study["residence_time"], dtype=np.float64) >= 5.0 - 1e-9

    assert isinstance(density, DensityPrediction)
    assert scores[~censored] == pytest.approx(tabulated[~censored], abs=5e-3)
    assert np.all(scores[censored] > tabulated[censored])


def test_an_undeclared_censoring_pile_up_is_reported_before_the_fit() -> None:
    model = forager()
    design = patch_design(["a"], repeats=10)
    columns = {name: design[name] for name in design.columns}
    times = np.linspace(0.5, 4.0, len(design))
    times[times > 3.0] = 3.0
    columns["residence_time"] = times

    findings = model.describe(Study(columns)).findings

    assert "undeclared_censoring" in {finding.code for finding in findings}
    assert "Declare" in " ".join(finding.message for finding in findings)


def test_heavy_censoring_is_reported_with_the_prediction_gap_named() -> None:
    model = forager(censoring=True)
    design = patch_design(["a"], session_cap=1.5, repeats=10)
    study = model.simulate(
        design,
        model.parameters_from_components(giving_up_rate=0.3, decision_noise=0.35),
        seed=9,
    )

    findings = model.describe(study).findings
    codes = {finding.code for finding in findings}

    assert codes & {"heavy_censoring", "all_rows_censored"}


def test_a_study_in_which_nothing_ever_departed_says_so() -> None:
    model = forager(censoring=True)
    design = patch_design(["a"], session_cap=0.05, repeats=5)
    columns = {name: design[name] for name in design.columns}
    columns["residence_time"] = np.full(len(design), 0.05)

    findings = model.describe(Study(columns)).findings

    assert "all_rows_censored" in {finding.code for finding in findings}


# --------------------------------------------------------------------------------------
# 5. Recovery and the overstaying benchmark
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("gain", [GainFunction.EXPONENTIAL, GainFunction.HYPERBOLIC])
def test_a_giving_up_rate_and_a_decision_noise_recover_together(gain: GainFunction) -> None:
    patches = PATCHES if gain is GainFunction.HYPERBOLIC else ((10.0, 0.4), (20.0, 0.2))
    model = forager(gain)
    design = patch_design(["a"], patches=patches, sessions=2)
    truth = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.35)

    report = run_parameter_recovery(model, design, [dict(truth)], repeats=4, seed=11)

    for summary in report.summary():
        assert abs(summary.bias) < 0.12, summary.parameter
        assert summary.coverage_95 >= 0.75, summary.parameter


def test_a_fit_reports_charnovs_optimum_beside_the_threshold_it_estimated() -> None:
    """The overstaying result becomes a measured ratio rather than an assumption."""

    model = forager()
    design = patch_design(["a"], sessions=3)
    optimal_rate = environment_rate(design)
    truth = model.parameters_from_components(giving_up_rate=0.5 * optimal_rate, decision_noise=0.3)
    study = model.simulate(design, truth, seed=13)

    fit = model.fit(study)

    assert fit.derived_values["marginal_value_rate"] == pytest.approx(optimal_rate, rel=1e-9)
    assert fit.derived_values["overstaying_ratio"] == pytest.approx(2.0, rel=0.15)
    assert fit.derived_quantities["overstaying_ratio"].standard_error > 0.0
    assert fit.derived_values["optimal_residence_time"] == pytest.approx(
        float(
            np.mean(
                marginal_value_residence_time(
                    np.asarray(design["patch_yield"], dtype=np.float64),
                    np.asarray(design["patch_decay"], dtype=np.float64),
                    TRAVEL_TIME,
                    gain="hyperbolic",
                )
            )
        ),
        rel=1e-9,
    )


def test_an_undeclared_travel_time_leaves_the_likelihood_alone() -> None:
    """The theorem is a benchmark, never a constraint: without it the fit is the same fit."""

    with_travel = forager()
    without = PatchLeaving(gain=GainFunction.HYPERBOLIC)
    design = patch_design(["a"], sessions=2)
    truth = with_travel.parameters_from_components(giving_up_rate=0.4, decision_noise=0.3)
    study = with_travel.simulate(design, truth, seed=15)

    left = with_travel.fit(study)
    right = without.fit(study)

    assert right.estimates == pytest.approx(left.estimates, rel=1e-8)
    assert "overstaying_ratio" not in right.derived_values
    assert "overstaying_ratio" in left.derived_values


def test_the_estimator_conformance_harness_passes() -> None:
    model = forager(censoring=True)
    design = patch_design(["a"], session_cap=6.0, sessions=2)
    study = model.simulate(
        design,
        model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.35),
        seed=17,
    )

    assert_behaviour_estimator_conforms(model, study, seed=1)


# --------------------------------------------------------------------------------------
# 6. The composition contract
# --------------------------------------------------------------------------------------


def test_the_family_composes_through_the_bounded_coordinate_contract() -> None:
    model = forager()

    assert isinstance(model, BoundedCoordinateEstimator)
    assert require_composable(model, combinator="smooth") == BOUNDED_COORDINATE
    assert require_composable(model, combinator="hierarchical") == BOUNDED_COORDINATE


def test_the_derived_box_is_finite_on_every_coordinate() -> None:
    model = forager()
    study = model.simulate(
        patch_design(["a"], repeats=5),
        model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.3),
        seed=19,
    )

    box = model.coordinate_box(study)

    assert box.shape == (2, 2)
    assert np.all(np.isfinite(box)) and np.all(box[:, 1] > box[:, 0])


def test_hierarchy_recovers_a_per_subject_giving_up_rate() -> None:
    model = forager()
    pooled = hierarchical(model, over="subject", parameters=("giving_up_rate_log",), scale=0.5)
    truth = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.3)
    simulation = pooled.simulate_with_effects(
        patch_design(["a", "b", "c", "d"], sessions=2), truth, seed=21
    )

    fit = pooled.fit(simulation.study)

    assert fit.groups == ("a", "b", "c", "d")
    assert fit.varying_parameters == ("giving_up_rate_log",)
    correlation = np.corrcoef(simulation.group_deviations.ravel(), fit.group_deviations.ravel())[
        0, 1
    ]
    assert correlation > 0.8


def test_smoothness_recovers_a_threshold_that_drifts_across_sessions() -> None:
    """A patch-leaving threshold that changes over training, for no combinator code at all."""

    model = forager()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 5.0),
        parameters=("giving_up_rate_log",),
        smoothness=1.0,
    )
    paths = drifting.parameters_from_paths(
        {
            "giving_up_rate_log": (float(np.log(0.2)), float(np.log(1.0))),
            "decision_noise_log": float(np.log(0.3)),
        }
    )
    study = drifting.simulate(patch_design(["a"], sessions=6), paths, seed=23)

    fit = drifting.fit(study)

    assert drifting.parameter_names == (
        "giving_up_rate_log[session_order=0]",
        "giving_up_rate_log[session_order=5]",
        "decision_noise_log",
    )
    values = dict(zip(drifting.parameter_names, fit.estimates, strict=True))
    early = float(np.exp(values["giving_up_rate_log[session_order=0]"]))
    late = float(np.exp(values["giving_up_rate_log[session_order=5]"]))
    assert early < late
    assert 0.1 < early < 0.4
    assert 0.6 < late < 1.6


def test_hierarchy_and_smoothness_compose_over_a_censored_likelihood() -> None:
    model = forager(censoring=True)
    stack = hierarchical(
        smooth(
            model,
            over="session_order",
            knots=(0.0, 3.0),
            parameters=("giving_up_rate_log",),
        ),
        over="subject",
        parameters=("decision_noise_log",),
        scale=0.4,
        estimate_scale=False,
    )
    truth = dict(
        stack.model.parameters_from_paths(
            {
                "giving_up_rate_log": [float(np.log(0.3)), float(np.log(0.6))],
                "decision_noise_log": float(np.log(0.35)),
            }
        )
    )
    study = stack.simulate(
        patch_design(["a", "b", "c"], sessions=4, session_cap=6.0), truth, seed=25
    )

    fit = stack.fit(study)

    assert fit.varying_parameters == ("decision_noise_log",)
    assert fit.group_deviations.shape == (3, 1)
    assert np.all(np.isfinite(stack.pointwise_log_prob(study, fit)))


def test_a_residence_time_has_no_shipped_mixture_component() -> None:
    """``mix()`` is untouched and would work; what is missing is a component for a duration."""

    with pytest.raises(TypeError, match="not a mixture of one observation"):
        mix(forager(), UniformChoiceGuess())


# --------------------------------------------------------------------------------------
# 7. The identifiability hazards, reported before the fit
# --------------------------------------------------------------------------------------


def test_one_patch_type_cannot_tell_a_rate_threshold_from_a_time_threshold() -> None:
    """The most important finding this family has, and the reason MVT needs varied patches."""

    study = with_residence(patch_design(["a"], patches=((10.0, 2.0),), repeats=10))

    findings = forager().describe(study).findings

    assert "unidentified_leaving_rule" in {finding.code for finding in findings}
    assert "one rate threshold governs different patches" in " ".join(
        finding.message for finding in findings
    )


def test_several_patch_types_report_that_the_optimum_is_per_type() -> None:
    study = with_residence(patch_design(["a"], repeats=10))

    findings = forager().describe(study).findings

    assert "heterogeneous_environment" in {finding.code for finding in findings}
    assert "unidentified_leaving_rule" not in {finding.code for finding in findings}


def test_a_composed_model_carries_the_findings_through() -> None:
    pooled = hierarchical(forager(), over="subject", parameters=("giving_up_rate_log",))
    study = with_residence(patch_design(["a"], patches=((10.0, 2.0),), repeats=10))

    findings = pooled.describe(study).findings

    assert "unidentified_leaving_rule" in {finding.code for finding in findings}


# --------------------------------------------------------------------------------------
# 8. This is an ordinary estimator
# --------------------------------------------------------------------------------------


def test_the_family_flows_through_evaluate_splits() -> None:
    model = forager()
    truth = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.3)
    study = model.simulate(patch_design(["a", "b"], sessions=4), truth, seed=27)
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    evaluations = evaluate_splits(model, study, splits)
    wrong_gain = evaluate_splits(forager(GainFunction.EXPONENTIAL), study, splits)

    assert len(evaluations) == len(splits)
    assert all(
        np.all(np.isfinite(evaluation.pointwise_log_probability)) for evaluation in evaluations
    )
    # The gain function is a falsifiable claim about the schedule: the fold log score prefers
    # the one that generated the residence times.
    assert sum(
        float(np.sum(evaluation.pointwise_log_probability)) for evaluation in evaluations
    ) > sum(float(np.sum(evaluation.pointwise_log_probability)) for evaluation in wrong_gain)


def test_comparing_two_density_only_candidates_is_refused_by_name() -> None:
    """A gap in the comparison layer, asserted rather than worked around.

    :func:`~behavio.compare.compare_models` reports a Brier score beside the log score, and a
    Brier score is a scoring rule for a probability. An **unlabelled** density has no discrete
    margin to score -- a two-boundary diffusion has one, a residence time does not -- so the
    comparison refuses rather than inventing a number. The log score is defined for these
    candidates and :func:`~behavio.evaluate.evaluate_splits` reports it; what is missing is a
    way to ask ``compare_models`` for the log-score half alone.
    """

    from behavio.compare.models import UnscoreableByBrier

    model = forager()
    truth = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.3)
    study = model.simulate(patch_design(["a", "b"], sessions=3), truth, seed=28)
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    with pytest.raises(UnscoreableByBrier, match="no categorical margin"):
        compare_models(
            {"exponential": forager(GainFunction.EXPONENTIAL), "hyperbolic": model},
            study,
            splits,
            bootstrap_resamples=10,
            outcome_column="residence_time",
        )


# --------------------------------------------------------------------------------------
# 9. Configuration and refusals
# --------------------------------------------------------------------------------------


def test_a_non_positive_natural_parameter_cannot_be_encoded() -> None:
    with pytest.raises(ValueError, match="decision_noise must be finite and positive"):
        forager().parameters_from_components(giving_up_rate=0.4, decision_noise=0.0)


def test_a_non_positive_patch_is_refused() -> None:
    model = forager()
    design = patch_design(["a"], repeats=3)
    columns = {name: design[name] for name in design.columns}
    columns["patch_decay"] = np.zeros(len(design))

    with pytest.raises(ModelDataError, match="must be positive"):
        model.read_problem(with_residence(Study(columns)))


def test_a_negative_residence_time_is_refused() -> None:
    model = forager()
    design = patch_design(["a"], repeats=3)
    columns = {name: design[name] for name in design.columns}
    columns["residence_time"] = -np.ones(len(design))

    with pytest.raises(ModelDataError, match="must be non-negative"):
        model.read_problem(Study(columns))


def test_the_two_gain_functions_are_different_models() -> None:
    """Their decay parameters are not interchangeable and neither are their signatures."""

    left = forager(GainFunction.EXPONENTIAL)
    right = forager(GainFunction.HYPERBOLIC)

    assert left.model_name != right.model_name
    assert left.signature != right.signature
    assert patch_gain(3.0, patch_yield=10.0, patch_decay=2.0, gain="exponential") != pytest.approx(
        patch_gain(3.0, patch_yield=10.0, patch_decay=2.0, gain="hyperbolic")
    )


def test_a_fit_from_another_specification_is_refused() -> None:
    model = forager()
    truth = model.parameters_from_components(giving_up_rate=0.4, decision_noise=0.3)
    study = model.simulate(patch_design(["a"], repeats=5), truth, seed=29)
    fit = model.fit(study)

    with pytest.raises(ValueError, match="different model specification"):
        forager(GainFunction.EXPONENTIAL).predict(study, fit)


def test_an_environment_needs_a_positive_travel_time() -> None:
    with pytest.raises(ValueError, match="travel_time must be finite and positive"):
        marginal_value_rate(10.0, 2.0, 0.0, gain="hyperbolic")
