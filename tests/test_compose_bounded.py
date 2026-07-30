"""Hierarchy and smoothness over models whose coordinate is bounded, not linear.

Six cells of the model-by-combinator grid open here: ``smooth()`` and ``hierarchical()``
over :class:`BinaryQLearning`, :class:`BinaryRLAgent` and :class:`PsychometricFunction`.
None of the three is a penalised linear model -- two are recursions over trials and one is a
nonlinear link with two bounded rates -- so they compose through
:class:`behavio.contracts.bounded.BoundedCoordinateEstimator` instead.

The tests that matter most here are not the fits. They are the ones that would still pass a
wrong implementation of the fits: that a group's deviation is Gaussian on the *transformed*
coordinate rather than the natural one, that a clock which cuts through a value trace is
refused rather than averaged, and that a rate the data pin to its bound is reported as a
finding rather than shrunk with confidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from scipy.special import expit, logit

from behavio import (
    BernoulliHistoryGLM,
    BinaryQLearning,
    BinaryRLAgent,
    PsychometricFunction,
    Study,
    compare_models,
    evaluate_splits,
    run_parameter_recovery,
)
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth
from behavio.contracts.bounded import (
    BOUNDED_COORDINATE,
    PENALISED_LINEAR,
    BoundedCoordinateEstimator,
    require_composable,
)
from behavio.contracts.mixture import require_independent_rows
from behavio.evaluate import cohort_forward_session_splits, forward_session_splits
from behavio.models import ModelDataError
from behavio.models.rl import SoftmaxPolicy

# --------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------


def bandit_design(subjects: Sequence[str], *, n_sessions: int = 2, trials: int = 80) -> Study:
    """A reversing two-armed bandit, one reset block per subject and session."""

    columns: dict[str, list[object]] = {
        name: []
        for name in (
            "subject",
            "session",
            "trial",
            "session_order",
            "reward_probability_0",
            "reward_probability_1",
        )
    }
    for subject in subjects:
        for session in range(n_sessions):
            for trial in range(trials):
                rich = ((trial // 20) + session) % 2
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-s{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["reward_probability_1"].append(0.8 if rich else 0.2)
                columns["reward_probability_0"].append(0.2 if rich else 0.8)
    return Study(columns)


def psychometric_design(
    subjects: Sequence[str], *, n_sessions: int = 2, per_level: int = 14
) -> Study:
    """A fixed set of signed contrast levels, repeated within every session."""

    levels = (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0)
    columns: dict[str, list[object]] = {
        name: [] for name in ("subject", "session", "trial", "session_order", "stimulus")
    }
    for subject in subjects:
        trial = 0
        for session in range(n_sessions):
            for level in levels:
                for _ in range(per_level):
                    columns["subject"].append(subject)
                    columns["session"].append(f"{subject}-s{session}")
                    columns["trial"].append(trial)
                    columns["session_order"].append(session)
                    columns["stimulus"].append(level)
                    trial += 1
    return Study(columns)


def q_agent() -> BinaryQLearning:
    return BinaryQLearning(n_restarts=2, max_iterations=400)


def rl_agent() -> BinaryRLAgent:
    return BinaryRLAgent(policy=SoftmaxPolicy(include_bias=True), n_restarts=2, max_iterations=300)


def curve() -> PsychometricFunction:
    return PsychometricFunction(n_restarts=3)


# --------------------------------------------------------------------------------------
# 0. Which contract each model composes through
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", [q_agent(), rl_agent(), curve()])
def test_bounded_models_compose_through_the_bounded_contract(model: object) -> None:
    assert isinstance(model, BoundedCoordinateEstimator)
    assert require_composable(model, combinator="smooth") == BOUNDED_COORDINATE


def test_penalised_linear_models_still_compose_through_their_own_contract() -> None:
    assert require_composable(BernoulliHistoryGLM(), combinator="hierarchical") == PENALISED_LINEAR


def test_a_bounded_model_declares_a_finite_box_on_the_transformed_coordinate() -> None:
    """The box is what says a transform was applied; a Gaussian deviation needs it."""

    study = bandit_design(["a"], n_sessions=1, trials=20)
    box = q_agent().coordinate_box(study)

    assert box.shape == (4, 2)
    assert np.all(np.isfinite(box))


def test_mix_still_refuses_a_value_updating_agent() -> None:
    """Widening the mixture to row objectives did not widen it to recursions.

    ``mix()`` is gated on row independence rather than on a linear predictor, so this cell is
    closed by the agent's own ``independent_rows_refusal`` -- a sentence about the value
    trace, not about a missing member -- and it would stay closed if every member a mixture
    calls were added tomorrow.
    """

    with pytest.raises(TypeError, match="recursion over trials"):
        mix(BinaryRLAgent(), UniformChoiceGuess())
    with pytest.raises(TypeError, match="recursion over trials"):
        mix(q_agent(), UniformChoiceGuess())


def test_the_row_block_check_is_exact_once_a_study_exists() -> None:
    """The declaration is eager; ``row_blocks`` is the same question asked exactly.

    A model that declared nothing and turns out to recurse must not be mixed on the strength
    of having the members, so the condition is re-asked against the study. The agent's own
    row objective is the honest instance of a coarser blocking: one block per session.
    """

    study = q_agent().simulate(
        bandit_design(["a"], n_sessions=2, trials=20),
        q_agent().parameters_from_components(
            learning_rate=0.3, inverse_temperature=3.0, choice_bias=0.0, perseveration=0.0
        ),
        seed=4,
    )
    objective = q_agent().row_objective(study)

    assert len(np.unique(objective.row_blocks)) < objective.n_rows
    with pytest.raises(ValueError, match="one density per row"):
        require_independent_rows(objective, model_name="binary-q-learning", combinator="mix")


# --------------------------------------------------------------------------------------
# 1. Deviations are Gaussian on the unconstrained coordinate, not the natural one
# --------------------------------------------------------------------------------------


def test_group_deviations_are_drawn_on_the_transformed_coordinate() -> None:
    """The decisive test: the same draws are inadmissible on the natural coordinate.

    A learning rate of 0.1 with a Gaussian deviation of standard deviation 1.5 is a rate
    below zero about a quarter of the time. On the logit it is a rate in ``(0, 1)`` every
    time. Both halves are asserted, so an implementation that moved the deviation onto the
    natural scale fails here rather than producing plausible-looking numbers.
    """

    model = q_agent()
    columns = np.asarray([0], dtype=np.intp)
    generator = np.random.default_rng(0)

    deviations = model.draw_group_deviations(
        columns, np.asarray([1.5]), groups=400, generator=generator
    ).ravel()

    population_rate = 0.1
    transformed = expit(logit(population_rate) + deviations)
    assert np.all((transformed > 0.0) & (transformed < 1.0))
    assert np.any(population_rate + deviations <= 0.0)
    assert np.any(population_rate + deviations >= 1.0)


def test_a_simulated_group_rate_is_the_logit_deviation_not_the_natural_one() -> None:
    model = q_agent()
    pooled = hierarchical(model, over="subject", parameters=("learning_rate_logit",))
    population = model.parameters_from_components(
        learning_rate=0.1, inverse_temperature=4.0, choice_bias=0.0, perseveration=0.0
    )

    simulation = pooled.simulate_with_effects(
        bandit_design(["a", "b"], n_sessions=1, trials=20),
        population,
        seed=1,
        group_deviations={"a": [1.0], "b": [-1.0]},
    )

    rates = [
        model.parameter_components(
            dict(zip(model.parameter_names, vector, strict=True))
        ).learning_rate
        for vector in simulation.group_parameter_vectors
    ]
    assert rates == pytest.approx([expit(logit(0.1) + 1.0), expit(logit(0.1) - 1.0)], rel=1e-9)
    # A natural-scale deviation of +1.0 is not a rate at all, which is the whole point.
    assert rates[0] < 1.0


def test_a_lapse_rate_deviation_stays_inside_its_declared_bound() -> None:
    model = PsychometricFunction(maximum_lapse=0.2)
    pooled = hierarchical(model, over="subject", parameters=("lapse_logit",), scale=1.2)
    population = model.parameters_from_components(
        threshold=0.0, width=0.3, guess_rate=0.05, lapse_rate=0.02
    )

    simulation = pooled.simulate_with_effects(
        psychometric_design(["a", "b", "c", "d"], n_sessions=1, per_level=4),
        population,
        seed=2,
    )

    lapses = [
        model.parameter_components(dict(zip(model.parameter_names, vector, strict=True))).lapse_rate
        for vector in simulation.group_parameter_vectors
    ]
    assert all(0.0 < lapse < 0.2 for lapse in lapses)
    assert len(set(np.round(lapses, 6))) == 4


# --------------------------------------------------------------------------------------
# 2. The three hierarchical cells fit and recover
# --------------------------------------------------------------------------------------


def test_hierarchical_q_learning_recovers_population_and_group_effects() -> None:
    model = q_agent()
    pooled = hierarchical(model, over="subject", parameters=("choice_bias",), scale=0.6)
    truth = model.parameters_from_components(
        learning_rate=0.3, inverse_temperature=4.0, choice_bias=0.0, perseveration=0.2
    )
    simulation = pooled.simulate_with_effects(
        bandit_design(["a", "b", "c", "d"], n_sessions=2, trials=80), truth, seed=3
    )

    fit = pooled.fit(simulation.study)

    assert fit.varying_parameters == ("choice_bias",)
    assert fit.groups == ("a", "b", "c", "d")
    components = model.parameter_components(
        dict(zip(model.parameter_names, fit.estimates, strict=True))
    )
    assert components.learning_rate == pytest.approx(0.3, abs=0.2)
    assert components.inverse_temperature == pytest.approx(4.0, rel=0.6)
    truth_deviations = simulation.group_deviations.ravel()
    fitted_deviations = fit.group_deviations.ravel()
    assert np.corrcoef(truth_deviations, fitted_deviations)[0, 1] > 0.9
    # Shrinkage, not equality: a MAP deviation is pulled towards the population.
    assert np.all(np.abs(fitted_deviations) <= np.abs(truth_deviations) + 0.3)


def test_hierarchical_rl_agent_recovers_population_and_group_effects() -> None:
    model = rl_agent()
    pooled = hierarchical(model, over="subject", parameters=("choice_bias",), scale=0.6)
    truth = model.parameters_from_components(
        learning_rate=0.3, inverse_temperature=4.0, choice_bias=0.0
    )
    simulation = pooled.simulate_with_effects(
        bandit_design(["a", "b", "c"], n_sessions=1, trials=100), truth, seed=13
    )

    fit = pooled.fit(simulation.study)

    natural = model.parameter_components(
        dict(zip(model.parameter_names, fit.estimates, strict=True))
    )
    assert natural["learning_rate"] == pytest.approx(0.3, abs=0.25)
    assert (
        np.corrcoef(simulation.group_deviations.ravel(), fit.group_deviations.ravel())[0, 1] > 0.9
    )


def test_hierarchical_psychometric_recovers_population_and_group_effects() -> None:
    model = curve()
    pooled = hierarchical(model, over="subject", parameters=("threshold", "log_width"), scale=0.4)
    truth = model.parameters_from_components(
        threshold=0.0, width=0.35, guess_rate=0.04, lapse_rate=0.04
    )
    simulation = pooled.simulate_with_effects(
        psychometric_design(["a", "b", "c", "d", "e"], n_sessions=2, per_level=14),
        truth,
        seed=11,
    )

    fit = pooled.fit(simulation.study)

    components = model.parameter_components(
        dict(zip(model.parameter_names, fit.estimates, strict=True))
    )
    assert components.threshold == pytest.approx(0.0, abs=0.25)
    assert components.width == pytest.approx(0.35, rel=0.8)
    assert fit.varying_parameters == ("threshold", "log_width")
    assert fit.group_deviations.shape == (5, 2)
    assert np.corrcoef(simulation.group_deviations[:, 0], fit.group_deviations[:, 0])[0, 1] > 0.7


# --------------------------------------------------------------------------------------
# 3. The three smooth cells fit and recover
# --------------------------------------------------------------------------------------


def test_smooth_q_learning_recovers_a_drifting_policy_temperature() -> None:
    model = q_agent()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 5.0),
        parameters=("inverse_temperature_log",),
        smoothness=0.5,
    )
    paths = drifting.parameters_from_paths(
        {
            "learning_rate_logit": float(logit(0.3)),
            "inverse_temperature_log": (float(np.log(1.5)), float(np.log(6.0))),
            "choice_bias": 0.0,
            "perseveration": 0.0,
        }
    )
    study = drifting.simulate(bandit_design(["a"], n_sessions=6, trials=100), paths, seed=5)

    fit = drifting.fit(study)

    assert drifting.parameter_names == (
        "learning_rate_logit",
        "inverse_temperature_log[session_order=0]",
        "inverse_temperature_log[session_order=5]",
        "choice_bias",
        "perseveration",
    )
    trajectory = drifting.coefficient_trajectory(fit)
    first, last = trajectory.values[0, 1], trajectory.values[-1, 1]
    assert first < last - 0.5
    assert np.exp(first) == pytest.approx(1.5, rel=0.9)
    assert np.exp(last) == pytest.approx(6.0, rel=0.9)


def test_smooth_rl_agent_recovers_a_drifting_policy_temperature() -> None:
    model = rl_agent()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 5.0),
        parameters=("inverse_temperature_log",),
        smoothness=0.5,
    )
    paths = drifting.parameters_from_paths(
        {
            "learning_rate_logit": float(logit(0.3)),
            "inverse_temperature_log": (float(np.log(1.5)), float(np.log(6.0))),
            "choice_bias": 0.0,
        }
    )
    study = drifting.simulate(bandit_design(["a"], n_sessions=6, trials=100), paths, seed=19)

    fit = drifting.fit(study)

    values = dict(zip(drifting.parameter_names, fit.estimates, strict=True))
    assert (
        values["inverse_temperature_log[session_order=0]"]
        < values["inverse_temperature_log[session_order=5]"] - 0.5
    )


def test_smooth_psychometric_recovers_a_narrowing_width() -> None:
    model = curve()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 5.0),
        parameters=("log_width",),
        smoothness=1.0,
    )
    paths = drifting.parameters_from_paths(
        {
            "threshold": 0.0,
            "log_width": (float(np.log(0.6)), float(np.log(0.15))),
            "guess_logit": float(logit(0.04 / 0.2)),
            "lapse_logit": float(logit(0.04 / 0.2)),
        }
    )
    study = drifting.simulate(psychometric_design(["a"], n_sessions=6, per_level=18), paths, seed=7)

    fit = drifting.fit(study)

    values = dict(zip(drifting.parameter_names, fit.estimates, strict=True))
    assert np.exp(values["log_width[session_order=0]"]) == pytest.approx(0.6, rel=0.7)
    assert np.exp(values["log_width[session_order=5]"]) == pytest.approx(0.15, rel=0.7)
    assert values["log_width[session_order=5]"] < values["log_width[session_order=0]"] - 0.5


# --------------------------------------------------------------------------------------
# 4. A clock that cuts through a value trace is refused, not averaged
# --------------------------------------------------------------------------------------


def test_smoothing_an_agent_over_a_within_session_clock_is_refused() -> None:
    """The scientific decision the selector alone cannot express, made explicit.

    ``parameters=`` says *which* parameters drift. The clock says *how often*, and for a
    recursion the clock is the part that can be wrong: a learning rate that changes between
    two trials of one session leaves the value trace unable to say which of its values wrote
    which part of the trace.
    """

    study = q_agent().simulate(
        bandit_design(["a"], n_sessions=2, trials=40),
        q_agent().parameters_from_components(
            learning_rate=0.3, inverse_temperature=3.0, choice_bias=0.0, perseveration=0.0
        ),
        seed=4,
    )
    drifting = smooth(
        q_agent(), over="trial", knots=(0.0, 39.0), parameters=("learning_rate_logit",)
    )

    with pytest.raises(ModelDataError, match="constant within each block"):
        drifting.fit(study)


def test_grouping_that_splits_a_reset_block_is_refused() -> None:
    model = q_agent()
    study = model.simulate(
        bandit_design(["a", "b"], n_sessions=1, trials=40),
        model.parameters_from_components(
            learning_rate=0.3, inverse_temperature=3.0, choice_bias=0.0, perseveration=0.0
        ),
        seed=6,
    )
    pooled = hierarchical(model, over="reward_probability_1", parameters=("choice_bias",))

    with pytest.raises(ValueError, match="splits a block"):
        pooled.fit(study)


# --------------------------------------------------------------------------------------
# 5. The lapse-near-bound hazard is reported before the fit
# --------------------------------------------------------------------------------------


def lapse_free_study(*, flawless: str) -> Study:
    """A study in which one subject never errs at the easiest levels."""

    design = psychometric_design(["a", "b", "c"], n_sessions=1, per_level=10)
    stimulus = np.asarray(design["stimulus"], dtype=np.float64)
    subject = list(design["subject"])
    generator = np.random.default_rng(21)
    probability = 0.03 + 0.94 * expit(stimulus / 0.25)
    choices = generator.binomial(1, probability).astype(np.int8)
    easiest = stimulus >= float(np.quantile(stimulus, 0.75))
    for index, (name, extreme) in enumerate(zip(subject, easiest, strict=True)):
        if name == flawless and extreme:
            choices[index] = 1
    columns = {name: design[name] for name in design.columns}
    columns["choice"] = choices
    return Study(columns)


def test_a_group_whose_lapse_rate_is_pinned_to_its_bound_is_reported() -> None:
    model = PsychometricFunction(fixed_guess_rate=0.02)
    pooled = hierarchical(model, over="subject", parameters=("lapse_logit",), scale=0.5)
    study = lapse_free_study(flawless="b")

    findings = pooled.describe(study).findings

    codes = {finding.code for finding in findings}
    assert "unidentified_group_rate" in codes
    message = next(
        finding.message for finding in findings if finding.code == "unidentified_group_rate"
    )
    assert "lapse_rate" in message
    assert "b" in message
    assert all(finding.severity == "warning" for finding in findings)


def test_no_rate_finding_when_every_group_shows_the_asymptote() -> None:
    model = PsychometricFunction(fixed_guess_rate=0.02)
    pooled = hierarchical(model, over="subject", parameters=("lapse_logit",), scale=0.5)
    study = lapse_free_study(flawless="none-of-them")

    assert pooled.describe(study).findings == ()


def test_the_rate_finding_is_only_raised_for_the_rates_that_vary() -> None:
    model = PsychometricFunction(fixed_guess_rate=0.02)
    pooled = hierarchical(model, over="subject", parameters=("threshold",), scale=0.5)

    assert pooled.describe(lapse_free_study(flawless="b")).findings == ()


def test_a_saturated_composed_rate_still_sets_the_boundary_diagnostic() -> None:
    """The post-fit half of the same hazard, through the diagnostic that already exists."""

    model = PsychometricFunction(fixed_guess_rate=0.02)
    pooled = hierarchical(model, over="subject", parameters=("lapse_logit",), scale=3.0)

    fit = pooled.fit(lapse_free_study(flawless="b"))

    assert fit.diagnostics.boundary_estimate


# --------------------------------------------------------------------------------------
# 6. Composed bounded models are ordinary estimators
# --------------------------------------------------------------------------------------


def test_a_hierarchical_agent_flows_through_evaluate_splits_and_compare_models() -> None:
    model = q_agent()
    truth = model.parameters_from_components(
        learning_rate=0.3, inverse_temperature=4.0, choice_bias=0.1, perseveration=0.0
    )
    study = model.simulate(bandit_design(["a", "b"], n_sessions=3, trials=60), truth, seed=8)
    pooled = hierarchical(model, over="subject", parameters=("choice_bias",), scale=0.5)
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    evaluations = evaluate_splits(pooled, study, splits)
    comparison = compare_models(
        {"pooled": pooled, "guessing": BernoulliHistoryGLM(predictors=(), l2=1.0)},
        study,
        splits,
        bootstrap_resamples=50,
    )

    assert len(evaluations) == len(splits)
    assert all(np.isfinite(evaluation.mean_log_loss) for evaluation in evaluations)
    assert set(comparison.model_order) == {"pooled", "guessing"}


def test_a_smooth_curve_flows_through_evaluate_splits_and_recovery() -> None:
    model = curve()
    drifting = smooth(model, over="session_order", knots=(0.0, 3.0), parameters=("log_width",))
    paths = drifting.parameters_from_paths(
        {
            "threshold": 0.0,
            "log_width": (float(np.log(0.5)), float(np.log(0.2))),
            "guess_logit": float(logit(0.05 / 0.2)),
            "lapse_logit": float(logit(0.05 / 0.2)),
        }
    )
    design = psychometric_design(["a"], n_sessions=4, per_level=12)
    study = drifting.simulate(design, paths, seed=9)
    splits = forward_session_splits(study, min_train_sessions=3, horizon=1)

    evaluations = evaluate_splits(drifting, study, splits)
    report = run_parameter_recovery(drifting, design, [dict(paths)], repeats=2, seed=31)

    assert all(np.isfinite(evaluation.mean_log_loss) for evaluation in evaluations)
    assert len(report.summary()) == len(drifting.parameter_names)


def test_a_hierarchical_curve_survives_parameter_recovery() -> None:
    model = curve()
    pooled = hierarchical(model, over="subject", parameters=("threshold",), scale=0.4)
    design = psychometric_design(["a", "b", "c"], n_sessions=1, per_level=12)
    truth = dict(
        model.parameters_from_components(threshold=0.1, width=0.3, guess_rate=0.05, lapse_rate=0.05)
    )

    report = run_parameter_recovery(pooled, design, [truth], repeats=2, seed=41)

    assert len(report.summary()) == len(pooled.parameter_names)
    assert np.all(np.isfinite(report.estimates))


def test_describe_reports_a_composed_agents_clock_and_grouping() -> None:
    model = q_agent()
    drifting = smooth(model, over="session_order", knots=(0.0, 1.0))
    study = model.simulate(
        bandit_design(["a"], n_sessions=2, trials=30),
        model.parameters_from_components(
            learning_rate=0.3, inverse_temperature=3.0, choice_bias=0.0, perseveration=0.0
        ),
        seed=10,
    )

    description = drifting.describe(study)

    assert description.clock == "session_order"
    assert description.model_name == "smooth-binary-q-learning"
    assert not description.errors


# --------------------------------------------------------------------------------------
# 7. Hierarchy over a smooth bounded model, and estimating its scale
# --------------------------------------------------------------------------------------


def test_hierarchy_outside_smoothness_composes_for_a_bounded_model() -> None:
    model = q_agent()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 2.0),
        parameters=("inverse_temperature_log",),
        smoothness=1.0,
        shared_trajectory=True,
    )
    pooled = hierarchical(drifting, over="subject", parameters=("choice_bias",), scale=0.5)
    truth = drifting.parameters_from_paths(
        {
            "learning_rate_logit": float(logit(0.3)),
            "inverse_temperature_log": (float(np.log(2.0)), float(np.log(5.0))),
            "choice_bias": 0.0,
            "perseveration": 0.0,
        }
    )
    simulation = pooled.simulate_with_effects(
        bandit_design(["a", "b", "c"], n_sessions=3, trials=60), truth, seed=12
    )

    fit = pooled.fit(simulation.study)

    assert fit.parameter_names == drifting.parameter_names
    assert fit.group_deviations.shape == (3, 1)
    trajectory = pooled.group_trajectory(fit, "a")
    assert trajectory.values.shape == (2, 4)


def test_a_whole_path_varies_by_group_for_a_bounded_model() -> None:
    model = q_agent()
    drifting = smooth(
        model,
        over="session_order",
        knots=(0.0, 2.0),
        parameters=("inverse_temperature_log",),
        shared_trajectory=True,
    )
    pooled = hierarchical(
        drifting, over="subject", parameters=("inverse_temperature_log",), scale=0.4
    )

    assert pooled.varying_parameters == (
        "inverse_temperature_log[session_order=0]",
        "inverse_temperature_log[session_order=2]",
    )


def test_expectation_maximisation_estimates_a_bounded_models_group_scale() -> None:
    model = q_agent()
    pooled = hierarchical(
        model,
        over="subject",
        parameters=("choice_bias",),
        scale=0.4,
        estimate_scale=True,
        scale_estimator="laplace-em",
        scale_max_iterations=4,
        scale_uncertainty="local",
    )
    truth = model.parameters_from_components(
        learning_rate=0.3, inverse_temperature=4.0, choice_bias=0.0, perseveration=0.0
    )
    simulation = pooled.simulate_with_effects(
        bandit_design(["a", "b", "c", "d"], n_sessions=1, trials=80), truth, seed=14
    )

    fit = pooled.fit(simulation.study)

    assert fit.scale_estimated
    assert fit.scale_estimation_policy == "laplace-em"
    assert 0.05 <= fit.scale <= 2.0
    assert fit.scale_standard_errors is not None


def test_the_laplace_profile_declines_a_bounded_model_by_name() -> None:
    model = q_agent()
    pooled = hierarchical(model, over="subject", parameters=("choice_bias",), estimate_scale=True)
    truth = model.parameters_from_components(
        learning_rate=0.3, inverse_temperature=3.0, choice_bias=0.0, perseveration=0.0
    )
    study = pooled.simulate(bandit_design(["a", "b"], n_sessions=1, trials=40), truth, seed=15)

    with pytest.raises(ModelDataError, match="laplace-em"):
        pooled.fit(study)


# --------------------------------------------------------------------------------------
# 8. The row objective is the model's own likelihood, read a different way
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "design", "natural"),
    [
        (
            q_agent(),
            bandit_design(["a"], n_sessions=2, trials=40),
            {
                "learning_rate": 0.3,
                "inverse_temperature": 3.0,
                "choice_bias": 0.1,
                "perseveration": 0.0,
            },
        ),
        (
            rl_agent(),
            bandit_design(["a"], n_sessions=2, trials=40),
            {"learning_rate": 0.3, "inverse_temperature": 3.0, "choice_bias": 0.1},
        ),
        (
            curve(),
            psychometric_design(["a"], n_sessions=1, per_level=10),
            {"threshold": 0.1, "width": 0.3, "guess_rate": 0.05, "lapse_rate": 0.05},
        ),
    ],
    ids=["q-learning", "rl-agent", "psychometric"],
)
def test_a_constant_row_objective_reproduces_the_models_own_score(
    model: object, design: Study, natural: dict[str, float]
) -> None:
    """One coordinate repeated over every row is the single-level fit, not an approximation."""

    parameters = model.parameters_from_components(**natural)
    study = model.simulate(design, parameters, seed=16)
    vector = np.asarray([parameters[name] for name in model.parameter_names])
    rows = np.tile(vector, (len(study), 1))

    value, gradient = model.row_objective(study).value_and_gradient(rows)
    scored = model.pointwise_log_prob_rows(study, rows)

    assert value == pytest.approx(-float(np.sum(scored)), rel=1e-8)
    assert gradient.shape == rows.shape
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize(
    ("model", "design", "natural"),
    [
        (
            q_agent(),
            bandit_design(["a"], n_sessions=2, trials=30),
            {
                "learning_rate": 0.35,
                "inverse_temperature": 2.5,
                "choice_bias": -0.2,
                "perseveration": 0.1,
            },
        ),
        (
            curve(),
            psychometric_design(["a"], n_sessions=1, per_level=8),
            {"threshold": 0.05, "width": 0.4, "guess_rate": 0.06, "lapse_rate": 0.03},
        ),
    ],
    ids=["q-learning", "psychometric"],
)
def test_the_row_gradient_agrees_with_a_finite_difference(
    model: object, design: Study, natural: dict[str, float]
) -> None:
    parameters = model.parameters_from_components(**natural)
    study = model.simulate(design, parameters, seed=18)
    objective = model.row_objective(study)
    vector = np.asarray([parameters[name] for name in model.parameter_names])

    def scored(point: np.ndarray) -> float:
        return objective.value_and_gradient(np.tile(point, (len(study), 1)))[0]

    _, gradient = objective.value_and_gradient(np.tile(vector, (len(study), 1)))
    analytic = gradient.sum(axis=0)
    numeric = np.empty_like(analytic)
    for index in range(len(vector)):
        step = np.zeros_like(vector)
        step[index] = 1e-5
        numeric[index] = (scored(vector + step) - scored(vector - step)) / 2e-5

    assert analytic == pytest.approx(numeric, rel=1e-3, abs=1e-4)


def test_the_rate_finding_survives_a_smooth_model_in_between() -> None:
    """A path varies by group as a whole path, so naming any knot names the coefficient."""

    model = PsychometricFunction(fixed_guess_rate=0.02)
    drifting = smooth(model, over="session_order", knots=(0.0, 1.0), parameters=("lapse_logit",))
    pooled = hierarchical(
        drifting,
        over="subject",
        parameters=("lapse_logit",),
        scale=0.5,
    )

    findings = pooled.describe(lapse_free_study(flawless="c")).findings

    rate_findings = [finding for finding in findings if finding.code == "unidentified_group_rate"]
    assert len(rate_findings) == 1
    assert "c" in rate_findings[0].message
