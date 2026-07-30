"""The multinomial's three empty cells, reached through the combinators rather than classes.

``MultinomialLogit`` was already a penalised linear model and already the package's
reference for design handling; the only thing standing between it and
``behavio.compose`` was the shape of its linear predictor, which is one number per
*category* and not one number per row. These tests are what establishes that widening
:class:`behavio.contracts.compose.PenalisedLinearEstimator` by a cell axis was sufficient:
smooth, hierarchical and hierarchical-smooth multinomials fit and recover, availability and
omission survive composition, and no combinator learned anything about categories to make
it happen.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavio import (
    ChoiceSpec,
    MultinomialLogit,
    Study,
    cohort_forward_session_splits,
    compare_models,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
)
from behavio.compose import hierarchical, smooth
from behavio.contracts.compose import (
    PenalisedLinearEstimator,
    expand_group_design,
    group_blocks,
)
from behavio.design import DesignSpec, NumericTerm
from behavio.evaluate import leave_one_subject_out_splits
from behavio.models import BehaviourEstimator, BehaviourModel, CategoricalPrediction

OPTIONS = ("left", "right", "up")
KNOTS = (0.0, 3.0)

#: One static truth, and one truth in which two of the four coefficients drift.
STATIC = {
    "category['right']::intercept": 0.3,
    "category['right']::stimulus": 1.1,
    "category['up']::intercept": -0.2,
    "category['up']::stimulus": -0.7,
}
PATHS = {
    "category['right']::intercept": (0.6, -0.4),
    "category['right']::stimulus": (0.4, 1.6),
    "category['up']::intercept": (-0.2, -0.2),
    "category['up']::stimulus": (-0.9, -0.1),
}


def design(
    *,
    n_subjects: int = 1,
    n_sessions: int = 4,
    n_trials: int = 120,
    withhold_up_from: tuple[str, ...] = (),
    seed: int = 41,
) -> Study:
    """A three-action design in which ``up`` is withheld on one trial in seven.

    ``withhold_up_from`` names subjects that were never offered ``up`` at all, which is the
    case a hierarchical fit has to answer for: a deviation on a category a group never saw.
    """

    generator = np.random.default_rng(seed)
    subjects = [f"mouse-{index}" for index in range(n_subjects)]
    rows = n_subjects * n_sessions * n_trials
    available = np.empty(rows, dtype=object)
    available[:] = [
        ("left", "right")
        if subject in withhold_up_from or trial % 7 == 0
        else ("left", "right", "up")
        for subject in subjects
        for _session in range(n_sessions)
        for trial in range(n_trials)
    ]
    return Study(
        {
            "subject": [subject for subject in subjects for _ in range(n_sessions * n_trials)],
            "session": [
                f"session-{order}"
                for _subject in subjects
                for order in range(n_sessions)
                for _ in range(n_trials)
            ],
            "trial": list(range(n_trials)) * (n_sessions * n_subjects),
            "session_order": [
                order
                for _subject in subjects
                for order in range(n_sessions)
                for _ in range(n_trials)
            ],
            "stimulus": generator.normal(size=rows),
            "available": available,
        }
    )


def base_model(*, omissions: bool = False, l2: float = 0.05) -> MultinomialLogit:
    return MultinomialLogit(
        choice=ChoiceSpec(
            options=OPTIONS,
            omission_values=("omit",) if omissions else (),
            available_options_column="available",
        ),
        design=DesignSpec((NumericTerm("stimulus"),)),
        include_omission=omissions,
        l2=l2,
    )


# --------------------------------------------------------------------------------------
# The widened contract
# --------------------------------------------------------------------------------------


def test_a_multinomial_satisfies_the_penalised_linear_contract_with_one_cell_per_category() -> None:
    model = base_model()
    study = design(n_trials=20)

    assert isinstance(model, PenalisedLinearEstimator)
    assert isinstance(model, BehaviourModel)
    assert model.predictor_cells == (
        "category['left']",
        "category['right']",
        "category['up']",
    )

    matrix = model.design_matrix(study)
    assert matrix.shape == (len(study), 3, len(model.parameter_names))
    # The reference category is a cell with no parameters, not an absent cell.
    assert not matrix[:, 0, :].any()
    assert model.penalty_matrix().shape == (4, 4)

    offsets = model.predictor_offsets(study)
    assert offsets is not None
    withheld = np.asarray([len(options) == 2 for options in study["available"]])
    assert np.all(np.isneginf(offsets[withheld, 2]))
    assert np.all(offsets[~withheld] == 0.0)


def test_a_design_with_every_option_on_every_trial_declares_no_offsets_at_all() -> None:
    model = MultinomialLogit(
        choice=ChoiceSpec(options=OPTIONS),
        design=DesignSpec((NumericTerm("stimulus"),)),
    )

    assert model.predictor_offsets(design(n_trials=10)) is None


def test_expand_group_design_treats_a_cell_axis_exactly_as_it_treats_no_cell_axis() -> None:
    """The claim the previous wave made about ``expand_group_*``, checked rather than assumed.

    Grouping partitions rows. A cell axis sits between the row axis and the coordinate
    axis, so expanding a per-category design has to be the same block copy applied to every
    cell of a row -- which is what one ellipsis buys, and what this compares against doing
    it one cell at a time.
    """

    study = design(n_subjects=3, n_sessions=1, n_trials=4)
    model = base_model()
    blocks = group_blocks(study, "subject")
    matrix = model.design_matrix(study)
    columns = np.asarray([0, 2], dtype=np.intp)

    expanded = expand_group_design(matrix, blocks, columns)

    assert expanded.shape == (len(study), 3, 4 + 3 * 2)
    for cell in range(3):
        np.testing.assert_array_equal(
            expanded[:, cell, :], expand_group_design(matrix[:, cell, :], blocks, columns)
        )


def test_a_model_whose_design_contradicts_its_declared_cells_is_named_in_the_error() -> None:
    """The declaration has to be read by something or it is decoration."""

    from dataclasses import dataclass

    from behavio.contracts.compose import validate_predictor_shape

    @dataclass(frozen=True)
    class Mismatched:
        predictor_cells: tuple[str, ...]

    with pytest.raises(TypeError, match="must be 3-dimensional"):
        validate_predictor_shape(Mismatched(("a", "b")), np.zeros((4, 2)))
    with pytest.raises(TypeError, match="2 of them"):
        validate_predictor_shape(Mismatched(("a", "b", "c")), np.zeros((4, 2, 3)))
    with pytest.raises(TypeError, match="must be 2-dimensional"):
        validate_predictor_shape(Mismatched(()), np.zeros((4, 2, 3)))
    np.testing.assert_array_equal(
        validate_predictor_shape(Mismatched(("a", "b")), np.zeros((4, 2, 3))), np.zeros((4, 2, 3))
    )


def test_a_penalised_design_refuses_offsets_that_are_not_a_support_declaration() -> None:
    from behavio.contracts.compose import PenalisedDesign
    from behavio.models._kernels.multinomial import MultinomialLikelihood

    pieces = {
        "parameter_names": ("a", "b"),
        "design_matrix": np.zeros((5, 3, 2)),
        "outcomes": np.zeros(5),
        "penalty_matrix": np.eye(2),
        "likelihood": MultinomialLikelihood(OPTIONS),
    }

    assert PenalisedDesign(**pieces).n_cells == 3
    assert PenalisedDesign(**pieces).offsets is None
    assert PenalisedDesign(**pieces, offsets=np.full((5, 3), -np.inf)).predictor_shape == (5, 3)
    with pytest.raises(ValueError, match="shape of the linear predictor"):
        PenalisedDesign(**pieces, offsets=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="finite values or -inf"):
        PenalisedDesign(**pieces, offsets=np.full((5, 3), np.inf))
    with pytest.raises(ValueError, match="finite values or -inf"):
        PenalisedDesign(**pieces, offsets=np.full((5, 3), np.nan))


def test_a_glm_hmm_declines_the_contract_by_declaration_and_says_why() -> None:
    from behavio.models import BernoulliGLMHMM

    model = BernoulliGLMHMM(predictors=("stimulus",))

    assert "latent-state mixture" in model.penalised_linear_refusal
    assert "forward recursion" in model.penalised_linear_refusal
    with pytest.raises(TypeError, match="no canonical labelling"):
        smooth(model, knots=KNOTS)
    with pytest.raises(TypeError, match="reference-category logits"):
        hierarchical(model, over="subject")


# --------------------------------------------------------------------------------------
# The three cells
# --------------------------------------------------------------------------------------


def test_a_smooth_multinomial_recovers_drifting_per_category_coefficients() -> None:
    model = smooth(base_model(), over="session_order", knots=KNOTS, smoothness=1.0)
    truth = model.parameters_from_paths(PATHS)
    study = model.simulate(design(n_trials=400), truth, seed=3)

    fit = model.fit(study)

    assert fit.diagnostics.converged
    assert fit.parameter_names == model.parameter_names
    np.testing.assert_allclose(fit.estimates, model.parameter_vector(truth), atol=0.25)
    assert np.all(np.isfinite(model.pointwise_log_prob(study, fit)))
    trajectory = model.coefficient_trajectory(fit, times=(0.0, 1.5, 3.0))
    assert trajectory.values.shape == (3, 4)
    assert trajectory.coefficient_names == base_model().parameter_names


def test_a_hierarchical_multinomial_recovers_a_population_and_its_group_deviations() -> None:
    model = hierarchical(base_model(), over="subject", scale=0.4)
    panel = design(n_subjects=6, n_trials=150)
    simulation = model.simulate_with_effects(panel, STATIC, seed=7)

    fit = model.fit(simulation.study)

    assert fit.diagnostics.converged
    assert fit.groups == panel.subjects
    assert fit.varying_parameters == base_model().parameter_names
    np.testing.assert_allclose(
        fit.estimates, [STATIC[name] for name in model.parameter_names], atol=0.35
    )
    correlation = np.corrcoef(simulation.group_deviations.ravel(), fit.group_deviations.ravel())[
        0, 1
    ]
    assert correlation > 0.7
    prediction = model.predict(simulation.study, fit)
    assert isinstance(prediction, CategoricalPrediction)
    np.testing.assert_allclose(np.sum(prediction.probability, axis=1), 1.0)


def test_the_laplace_profile_over_a_scale_reads_per_category_cells_and_their_offsets() -> None:
    """The empirical-Bayes path writes its own objective, so it is the one at real risk.

    ``hierarchical(..., estimate_scale=True)`` profiles the group deviations out one group
    at a time with a hand-written conditional, rather than handing the whole problem to the
    model's solver. Blocking a study by group has to carry the offsets along with the design
    and the outcomes, and the conditional has to contract a per-cell gradient and a per-cell
    curvature block back onto the coordinate.
    """

    model = hierarchical(base_model(), over="subject", scale=0.4, estimate_scale=True)
    panel = design(n_subjects=4, n_trials=60)
    study = hierarchical(base_model(), over="subject", scale=0.4).simulate(panel, STATIC, seed=7)

    fit = model.fit(study)

    assert fit.scale_estimated
    assert fit.scale_bounds == (0.05, 2.0)
    assert 0.05 < float(fit.scales[0]) < 2.0
    assert fit.scale_standard_error is not None and fit.scale_standard_error > 0
    assert np.all(np.isfinite(fit.group_deviations))
    np.testing.assert_allclose(
        fit.estimates, [STATIC[name] for name in model.parameter_names], atol=0.4
    )


def test_a_hierarchical_smooth_multinomial_recovers_paths_that_vary_by_subject() -> None:
    paths = smooth(
        base_model(),
        over="session_order",
        knots=KNOTS,
        smoothness=1.0,
        shared_trajectory=True,
    )
    model = hierarchical(paths, over="subject", scale=0.3)
    panel = design(n_subjects=6, n_trials=150)
    simulation = model.simulate_with_effects(panel, paths.parameters_from_paths(PATHS), seed=13)

    fit = model.fit(simulation.study)

    assert fit.diagnostics.converged
    assert model.parameter_names == paths.parameter_names
    np.testing.assert_allclose(
        fit.estimates, paths.parameter_vector(paths.parameters_from_paths(PATHS)), atol=0.35
    )
    correlation = np.corrcoef(simulation.group_deviations.ravel(), fit.group_deviations.ravel())[
        0, 1
    ]
    assert correlation > 0.6
    subject = paths.trajectory_from_knots(
        np.asarray(list(fit.parameters_for("mouse-1").values())), times=(0.0, 3.0)
    )
    assert subject.values.shape == (2, 4)


# --------------------------------------------------------------------------------------
# Availability and omission, which composition must not quietly break
# --------------------------------------------------------------------------------------


def test_an_unavailable_category_keeps_probability_zero_through_both_combinators() -> None:
    panel = design(n_subjects=4, n_trials=60)
    withheld = np.asarray([len(options) == 2 for options in panel["available"]])
    paths = smooth(
        base_model(), over="session_order", knots=KNOTS, smoothness=1.0, shared_trajectory=True
    )
    models = {
        "smooth": paths,
        "hierarchical": hierarchical(base_model(), over="subject", scale=0.4),
        "hierarchical_smooth": hierarchical(paths, over="subject", scale=0.4),
    }
    study = models["hierarchical"].simulate(panel, STATIC, seed=11)

    for name, model in models.items():
        fit = model.fit(study)
        probability = model.predict(study, fit).probability
        assert np.all(probability[withheld, 2] == 0.0), name
        np.testing.assert_allclose(np.sum(probability, axis=1), 1.0, err_msg=name)
        assert np.all(np.isfinite(model.pointwise_log_prob(study, fit))), name
    assert "up" not in set(np.asarray(study["choice"])[withheld])


def test_a_group_never_offered_a_category_is_left_to_the_prior_rather_than_broadcast() -> None:
    """The modelling answer, which the arithmetic produces without being told to.

    A trial on which a category was not offered contributes zero probability to it, so it
    contributes zero gradient and zero curvature to that category's coefficients. A subject
    who was never offered ``up`` therefore leaves the ``up`` block of its deviation with no
    likelihood curvature at all: the joint MAP puts it at exactly the prior mean, and its
    standard error is exactly the prior standard deviation. Nothing is imputed, nothing is
    silently pooled, and the *population* ``up`` coefficients stay identified by the other
    subjects.
    """

    panel = design(n_subjects=3, n_trials=120, withhold_up_from=("mouse-2",), seed=3)
    model = hierarchical(base_model(), over="subject", scale=0.4)
    study = model.simulate(panel, STATIC, seed=5)

    fit = model.fit(study)

    absent = fit.groups.index("mouse-2")
    up_columns = [
        index
        for index, name in enumerate(fit.varying_parameters)
        if name.startswith("category['up']::")
    ]
    np.testing.assert_array_equal(fit.group_deviations[absent, up_columns], 0.0)
    np.testing.assert_allclose(fit.group_standard_errors[absent, up_columns], 0.4, atol=1e-9)
    assert np.all(
        np.abs(fit.group_deviations[absent])[
            [index for index in range(len(fit.varying_parameters)) if index not in up_columns]
        ]
        > 0.0
    )
    assert np.all(np.isfinite(fit.estimates))
    assert (
        model.predict(study, fit).probability[:, 2][np.asarray(study["subject"]) == "mouse-2"].max()
        == 0.0
    )


def test_omissions_stay_a_modeled_category_when_a_multinomial_is_composed() -> None:
    model = hierarchical(base_model(omissions=True), over="subject", scale=0.4)
    panel = design(n_subjects=4, n_trials=80)
    truth = {
        **STATIC,
        "category['omit']::intercept": -1.2,
        "category['omit']::stimulus": 0.15,
    }
    study = model.simulate(panel, truth, seed=17)

    fit = model.fit(study)
    codes = model.outcome_codes(study)
    prediction = model.predict(study, fit)

    assert model.categories == ("left", "right", "up", "omit")
    assert np.sum(codes == 3) > 0, "the simulation must actually produce omissions"
    assert prediction.categories == ("left", "right", "up", "omit")
    # The omission category is always available, including on trials that withheld `up`.
    withheld = np.asarray([len(options) == 2 for options in study["available"]])
    assert np.all(prediction.probability[withheld, 3] > 0.0)
    assert np.all(prediction.probability[withheld, 2] == 0.0)
    assert fit.varying_parameters == base_model(omissions=True).parameter_names


# --------------------------------------------------------------------------------------
# Parameter naming, and a composed multinomial as an ordinary estimator
# --------------------------------------------------------------------------------------


def test_parameter_naming_composes_mechanically_and_round_trips() -> None:
    """Per-category structure was always in the name, so nothing about naming had to move.

    A multinomial coefficient is ``category[<label>]::<feature>``; smoothing appends
    ``[<clock>=<knot>]`` as it does to any coefficient; and a hierarchical model reports the
    population coordinate unchanged, naming group deviations by group label rather than by
    a flattened string. ``category_parameter_names`` exists so that naming one category's
    coefficients in a ``hierarchical(...)`` call does not require writing ``repr`` quoting
    into a string literal.
    """

    base = base_model()
    paths = smooth(base, over="session_order", knots=KNOTS, smoothness=1.0)
    per_category = hierarchical(
        base, over="subject", parameters=base.category_parameter_names("up")
    )

    assert base.parameter_names == (
        "category['right']::intercept",
        "category['right']::stimulus",
        "category['up']::intercept",
        "category['up']::stimulus",
    )
    assert base.category_parameter_names("left") == (), "the reference category is not estimated"
    assert paths.parameter_names[:2] == (
        "category['right']::intercept[session_order=0]",
        "category['right']::intercept[session_order=3]",
    )
    assert per_category.parameter_names == base.parameter_names
    assert per_category.varying_parameters == (
        "category['up']::intercept",
        "category['up']::stimulus",
    )
    assert hierarchical(paths, over="subject").parameter_names == paths.parameter_names

    values = {name: 0.5 for name in paths.parameter_names}
    assert dict(paths.parameters_from_paths(PATHS)).keys() == values.keys()
    with pytest.raises(ValueError, match="not a modeled category"):
        base.category_parameter_names("sideways")


def test_a_composed_multinomial_passes_through_every_consumer() -> None:
    base = base_model()
    panel = design(n_subjects=4, n_trials=60)
    model = hierarchical(base, over="subject", scale=0.4)
    study = model.simulate(panel, STATIC, seed=7)

    assert isinstance(model, BehaviourEstimator)

    evaluations = evaluate_splits(model, study, leave_one_subject_out_splits(study))
    assert len(evaluations) == 4
    assert all(isinstance(item.prediction, CategoricalPrediction) for item in evaluations)
    assert all(item.outcome_codes is not None for item in evaluations)
    assert all(np.all(np.isfinite(item.pointwise_log_probability)) for item in evaluations)

    comparison = compare_models(
        {"static": base, "hierarchical": model},
        study,
        cohort_forward_session_splits(study, min_train_sessions=2),
        aggregation_column="session",
        outcome_column="choice",
    )
    assert {result.name for result in comparison.model_results} == {"static", "hierarchical"}
    assert all(0 <= result.pooled_brier_score <= 1 for result in comparison.model_results)

    recovery = run_parameter_recovery(model, panel, [STATIC], seed=5)
    assert recovery.n_runs == 1
    assert recovery.parameter_names == model.parameter_names

    description = model.describe(study)
    assert description.model_name == "hierarchical-multinomial-logit"
    assert description.parameter_names == model.parameter_names
    assert any("deviations by subject" in prior for prior in description.priors)
    assert not description.errors


def test_a_smooth_multinomial_passes_through_every_consumer() -> None:
    model = smooth(base_model(), over="session_order", knots=KNOTS, smoothness=1.0)
    study = model.simulate(design(n_trials=150), model.parameters_from_paths(PATHS), seed=19)

    evaluations = evaluate_splits(model, study, forward_session_splits(study, min_train_sessions=2))
    assert len(evaluations) == 2
    assert all(isinstance(item.prediction, CategoricalPrediction) for item in evaluations)
    assert all(item.outcome_codes is not None for item in evaluations)

    recovery = run_parameter_recovery(
        model, design(n_trials=150), [model.parameters_from_paths(PATHS)], seed=2
    )
    assert recovery.parameter_names == model.parameter_names

    description = model.describe(study)
    assert description.model_name == "smooth-multinomial-logit"
    assert any("random walk over session_order" in prior for prior in description.priors)
    assert not description.errors


def test_a_binary_glm_keeps_its_scalar_predictor_and_claims_no_category_coordinate() -> None:
    """The pass-throughs that make a composed multinomial categorical must not overreach.

    ``categories`` and ``outcome_codes`` are forwarded, so a combinator around a *binary*
    model has them as attributes and raises :class:`AttributeError` from either. That is
    the honest failure: :func:`behavio.evaluate_splits` only asks for a category coordinate
    once a model has returned a
    :class:`~behavio.contracts.estimator.CategoricalPrediction`, which a Bernoulli GLM
    never does.
    """

    from behavio import BernoulliHistoryGLM

    glm = BernoulliHistoryGLM(predictors=("stimulus",))
    drifting = smooth(glm, over="session_order", knots=KNOTS)

    assert glm.predictor_cells == ()
    assert glm.predictor_offsets(design(n_trials=5)) is None
    assert drifting.predictor_cells == ()
    with pytest.raises(AttributeError):
        _ = drifting.categories
    with pytest.raises(AttributeError):
        _ = hierarchical(glm, over="subject").categories
    assert hierarchical(base_model(), over="subject").categories == OPTIONS
