"""The combinators, and the proof that they are the models they replaced.

``tests/fixtures/pre_combinator_glm_reference.json`` holds fits produced by the deleted
``HierarchicalBernoulliHistoryGLM``, ``SmoothBernoulliHistoryGLM`` and
``HierarchicalSmoothBernoulliHistoryGLM`` classes at commit ``084a0c8``, the commit
immediately before they were replaced. Every model configuration, design and seed in that
fixture is reconstructed here from the combinators, so the comparison is old-versus-new and
not new-versus-itself. Without it there would be nothing establishing that
``hierarchical(smooth(glm))`` is the same model the sibling class was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from behavio import (
    BernoulliHistoryGLM,
    Study,
    cohort_forward_session_splits,
    compare_models,
    evaluate_splits,
    model_from_formula,
    run_model_recovery,
    run_parameter_recovery,
)
from behavio.compose import HierarchicalModel, SmoothModel, hierarchical, smooth
from behavio.contracts.compose import PenalisedLinearEstimator, VaryingEffects, group_blocks
from behavio.evaluate import leave_one_subject_out_splits
from behavio.models import BehaviourEstimator, BehaviourModel
from behavio.recovery import ModelRecoveryScenario

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "pre_combinator_glm_reference.json").read_text(
        encoding="utf-8"
    )
)
KNOTS = tuple(REFERENCE["knots"])
POPULATION = REFERENCE["population"]
PATHS = REFERENCE["paths"]


def design(*, n_subjects: int, n_sessions: int = 3, n_trials: int = 40) -> Study:
    """The exact design the reference fixture was generated on."""

    generator = np.random.default_rng(20260726)
    subject: list[str] = []
    session: list[str] = []
    trial: list[int] = []
    session_order: list[int] = []
    stimulus: list[float] = []
    for index in range(n_subjects):
        for order in range(n_sessions):
            subject.extend([f"mouse-{index}"] * n_trials)
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


def base_glm(*, l2: float = 0.1, choice_lags: int = 1) -> BernoulliHistoryGLM:
    return BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=choice_lags, l2=l2)


def paths_model(*, smoothness: float = 2.0, group_smoothness: float | None = None) -> SmoothModel:
    return smooth(
        base_glm(),
        over="session_order",
        knots=KNOTS,
        smoothness=smoothness,
        group_smoothness=group_smoothness,
    )


def identical(observed: Any, expected: Any) -> None:
    """Assert bit-for-bit equality against a reference the old code produced."""

    np.testing.assert_array_equal(
        np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    )


# --------------------------------------------------------------------------------------
# The acceptance test: each combinator reproduces the class it replaced
# --------------------------------------------------------------------------------------


def test_hierarchical_reproduces_the_hand_written_hierarchical_glm_bit_for_bit() -> None:
    reference = REFERENCE["reference"]["hierarchical"]
    model = hierarchical(base_glm(), over="subject", scale=0.4)

    simulation = model.simulate_with_effects(design(n_subjects=4), POPULATION, seed=7)
    study = simulation.study
    fit = model.fit(study)

    identical(study["choice"], reference["choice"])
    identical(simulation.group_deviations, reference["deviations"])
    identical(fit.estimates, reference["estimates"])
    identical(fit.standard_errors, reference["standard_errors"])
    identical(fit.group_deviations, reference["group_deviations"])
    identical(fit.group_standard_errors, reference["group_standard_errors"])
    identical(fit.diagnostics.objective, reference["objective"])
    identical(np.sum(model.pointwise_log_prob(study, fit)), reference["pointwise_log_prob_sum"])
    identical(model.predict(study, fit).probability[:8], reference["probability"])


def test_smooth_reproduces_the_hand_written_smooth_glm_bit_for_bit() -> None:
    reference = REFERENCE["reference"]["smooth"]
    model = paths_model()

    study = model.simulate(design(n_subjects=1), model.parameters_from_paths(PATHS), seed=11)
    fit = model.fit(study)

    identical(study["choice"], reference["choice"])
    identical(fit.estimates, reference["estimates"])
    identical(fit.standard_errors, reference["standard_errors"])
    identical(fit.diagnostics.objective, reference["objective"])
    identical(np.sum(model.pointwise_log_prob(study, fit)), reference["pointwise_log_prob_sum"])
    identical(
        model.coefficient_trajectory(fit, times=(0.0, 0.5, 1.5, 2.0)).values,
        reference["trajectory"],
    )


def test_hierarchical_smooth_reproduces_the_sibling_class_bit_for_bit() -> None:
    """The hand-written sibling turns out to be exactly the composition.

    ``HierarchicalSmoothBernoulliHistoryGLM`` subclassed the *base* GLM rather than the
    smooth one, so it was not obvious that it was ``hierarchical(smooth(glm))`` rather than
    a fourth thing. It is -- provided the subject deviation inherits the roughness prior of
    the path it deviates from, which is what
    :meth:`behavio.compose.SmoothModel.group_penalty` supplies and a naive ridge would not.
    """

    reference = REFERENCE["reference"]["hierarchical_smooth"]
    paths = paths_model()
    model = hierarchical(paths, over="subject", scale=0.4)

    simulation = model.simulate_with_effects(
        design(n_subjects=4), paths.parameters_from_paths(PATHS), seed=13
    )
    study = simulation.study
    fit = model.fit(study)

    identical(study["choice"], reference["choice"])
    identical(simulation.group_deviations, reference["deviations"])
    identical(fit.estimates, reference["estimates"])
    identical(fit.standard_errors, reference["standard_errors"])
    identical(fit.group_deviations, reference["group_deviations"])
    identical(fit.diagnostics.objective, reference["objective"])
    identical(np.sum(model.pointwise_log_prob(study, fit)), reference["pointwise_log_prob_sum"])
    identical(
        paths.trajectory_from_knots(fit.estimates, times=(0.0, 0.5, 1.5, 2.0)).values,
        reference["population_trajectory"],
    )
    identical(
        paths.trajectory_from_knots(
            np.asarray(list(fit.parameters_for("mouse-1").values())),
            times=(0.0, 0.5, 1.5, 2.0),
        ).values,
        reference["subject_trajectory"],
    )


def test_the_estimated_scale_path_agrees_to_the_last_bits_it_reassociated() -> None:
    """The empirical-Bayes path is the one place the arithmetic genuinely moved.

    The hand-written Laplace profile wrote its three quadratic forms in scalar,
    GLM-specific shorthand -- ``0.5 * l2 * (beta[1:] @ beta[1:])`` for the population
    penalty, ``0.5 * (1 / sigma**2) * (b @ b)`` for the deviation penalty, and
    ``p * log(sigma)`` for the Gaussian normaliser. The generic profile computes the same
    three numbers as ``0.5 * x' P x`` and ``-0.5 * logdet(P)``, because it is handed a
    penalty matrix and cannot know that this one is a scaled identity. Those are
    last-bit-level differences in the objective, amplified by a chaotic outer L-BFGS-B, and
    the tolerance below is the size of that amplification, not of a modelling change.
    """

    reference = REFERENCE["reference"]["hierarchical_estimated_scale"]
    model = hierarchical(base_glm(), over="subject", scale=0.4)
    study = model.simulate(design(n_subjects=4), POPULATION, seed=7)
    estimated = hierarchical(base_glm(), over="subject", scale=0.4, estimate_scale=True)

    fit = estimated.fit(study)

    assert fit.scale_estimated
    np.testing.assert_allclose(fit.estimates, reference["estimates"], atol=1e-6)
    assert float(fit.scales[0]) == pytest.approx(reference["scale"], abs=1e-5)
    assert fit.scale_standard_error == pytest.approx(reference["scale_standard_error"], abs=1e-5)
    assert fit.diagnostics.objective == pytest.approx(reference["objective"], abs=1e-8)


# --------------------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------------------


def test_the_base_glm_satisfies_the_composable_contract_and_the_wrappers_preserve_it() -> None:
    base = base_glm()

    assert isinstance(base, PenalisedLinearEstimator)
    assert isinstance(paths_model(), PenalisedLinearEstimator)
    assert isinstance(base, BehaviourModel)
    assert isinstance(paths_model(), BehaviourModel)
    assert isinstance(hierarchical(base, over="subject"), BehaviourModel)


def test_a_latent_state_model_is_refused_where_its_labels_cannot_be_pinned() -> None:
    """Both refusals are about labels, and they are refusals of different things.

    A path in clock time has no canonical labelling at all, because the ordering that names
    a GLM-HMM's states is an ordering of numbers and a smooth model's are paths. Letting
    *every* parameter vary by group would put a Gaussian ridge on reference-coded transition
    logits, which is a prior on a chart rather than on the simplex it charts. The cell that
    is open -- group deviations on named emission coefficients -- is exercised in
    ``tests/test_compose_glm_hmm.py``.
    """

    from behavio.models import BernoulliGLMHMM

    with pytest.raises(TypeError, match="no canonical labelling"):
        smooth(BernoulliGLMHMM(predictors=("stimulus",)), knots=KNOTS)
    with pytest.raises(TypeError, match="reference-category logits"):
        hierarchical(BernoulliGLMHMM(predictors=("stimulus",)), over="subject")
    assert isinstance(
        hierarchical(
            BernoulliGLMHMM(predictors=("stimulus",)), over="subject", parameters=("intercept",)
        ),
        BehaviourModel,
    )


def test_group_blocks_follow_first_appearance_order() -> None:
    panel = design(n_subjects=3, n_sessions=1, n_trials=2)

    blocks = group_blocks(panel, "subject")

    assert blocks.labels == panel.subjects
    assert blocks.row_block.tolist() == [0, 0, 1, 1, 2, 2]


def test_varying_effects_resolve_names_and_scales_in_model_order() -> None:
    effects = VaryingEffects.declare(
        ("intercept", "stimulus", "choice_lag_1"),
        over="subject",
        parameters=("choice_lag_1", "intercept"),
        scale=0.5,
        parameter_scales={"choice_lag_1": 0.2},
    )

    assert effects.columns(("intercept", "stimulus", "choice_lag_1")).tolist() == [0, 2]
    assert effects.ordered_parameters(("intercept", "stimulus", "choice_lag_1")) == (
        "intercept",
        "choice_lag_1",
    )
    assert effects.ordered_scales(("intercept", "stimulus", "choice_lag_1")).tolist() == [0.5, 0.2]


# --------------------------------------------------------------------------------------
# A composed model is an ordinary estimator everywhere
# --------------------------------------------------------------------------------------


def test_a_composed_model_passes_through_the_evaluation_and_recovery_engines() -> None:
    paths = paths_model()
    model = hierarchical(paths, over="subject", scale=0.4)
    panel = design(n_subjects=4)
    study = model.simulate(panel, paths.parameters_from_paths(PATHS), seed=13)

    assert isinstance(model, BehaviourEstimator)

    evaluations = evaluate_splits(model, study, leave_one_subject_out_splits(study))
    assert len(evaluations) == 4
    assert all(np.all(np.isfinite(item.pointwise_log_probability)) for item in evaluations)

    comparison = compare_models(
        {"static": base_glm(), "hierarchical_smooth": model},
        study,
        cohort_forward_session_splits(study, min_train_sessions=2),
    )
    assert {result.name for result in comparison.model_results} == {
        "static",
        "hierarchical_smooth",
    }

    recovery = run_parameter_recovery(
        model,
        panel,
        [paths.parameters_from_paths(PATHS)],
        seed=5,
    )
    assert recovery.n_runs == 1
    assert recovery.parameter_names == model.parameter_names

    description = model.describe(study)
    assert description.model_name == "hierarchical-smooth-bernoulli-history-glm"
    assert description.parameter_names == model.parameter_names
    assert any("deviations by subject" in prior for prior in description.priors)
    assert any("random walk over session_order" in prior for prior in description.priors)
    assert not description.errors


def test_model_recovery_can_tell_a_composed_model_from_its_base() -> None:
    panel = design(n_subjects=3, n_sessions=4, n_trials=40)
    static = base_glm()
    drifting = smooth(
        base_glm(), over="session_order", knots=(0.0, 3.0), smoothness=4.0, shared_trajectory=True
    )

    report = run_model_recovery(
        panel,
        (
            ModelRecoveryScenario(
                name="stationary",
                truth_label="static",
                generator=static,
                parameters=POPULATION,
            ),
        ),
        {"static": static, "smooth": drifting},
        repeats=1,
        seed=3,
        min_train_sessions=2,
    )

    assert report.repeats == 1
    assert set(report.candidate_labels) == {"static", "smooth"}


# --------------------------------------------------------------------------------------
# Parameter naming
# --------------------------------------------------------------------------------------


def test_parameter_naming_is_stable_across_both_combinators() -> None:
    base = base_glm()
    paths = smooth(base, over="session_order", knots=(0.0, 4.0), smoothness=1.0)
    pooled = hierarchical(base, over="subject", parameters=("intercept",))
    both = hierarchical(paths, over="subject")

    assert base.parameter_names == ("intercept", "stimulus", "choice_lag_1")
    assert paths.parameter_names == (
        "intercept[session_order=0]",
        "intercept[session_order=4]",
        "stimulus[session_order=0]",
        "stimulus[session_order=4]",
        "choice_lag_1[session_order=0]",
        "choice_lag_1[session_order=4]",
    )
    assert pooled.parameter_names == base.parameter_names
    assert pooled.varying_parameters == ("intercept",)
    assert both.parameter_names == paths.parameter_names
    assert both.model_name == "hierarchical-smooth-bernoulli-history-glm"
    assert paths.model_name == "smooth-bernoulli-history-glm"


def test_a_smooth_parameter_varies_by_group_as_a_whole_path() -> None:
    both = hierarchical(
        paths_model(),
        over="subject",
        parameters=("intercept[session_order=0]",),
    )

    study = hierarchical(paths_model(), over="subject", scale=0.4).simulate(
        design(n_subjects=2), paths_model().parameters_from_paths(PATHS), seed=2
    )

    with pytest.raises(ValueError, match="whole path"):
        both.fit(study)


# --------------------------------------------------------------------------------------
# The formula
# --------------------------------------------------------------------------------------


def test_a_formula_group_term_produces_a_model_that_fits() -> None:
    panel = design(n_subjects=4)
    truth = hierarchical(
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=0), over="subject", scale=0.4
    )
    study = truth.simulate(panel, {"intercept": -0.2, "stimulus": 1.0}, seed=21)

    model = model_from_formula(
        "choice ~ stimulus + (1 | subject)",
        BernoulliHistoryGLM(),
        scale=0.4,
    )
    fit = model.fit(study)

    assert isinstance(model, HierarchicalModel)
    assert model.grouping == "subject"
    assert model.varying_parameters == ("intercept",)
    assert model.parameter_names == ("intercept", "stimulus")
    assert fit.diagnostics.converged
    assert fit.group_deviations.shape == (4, 1)


def test_a_formula_can_let_a_slope_vary_too() -> None:
    model = model_from_formula(
        "choice ~ stimulus + (1 + stimulus | subject)",
        BernoulliHistoryGLM(),
        scale=0.3,
        parameter_scales={"stimulus": 0.1},
    )

    assert model.varying_parameters == ("intercept", "stimulus")
    assert model.effects.scales.tolist() == [0.3, 0.1]


def test_a_formula_without_a_group_term_configures_the_model_and_stops() -> None:
    model = model_from_formula("choice ~ stimulus + lag(choice, 1)", BernoulliHistoryGLM())

    assert isinstance(model, BernoulliHistoryGLM)
    assert model.parameter_names == ("intercept", "stimulus", "choice_lag_1")
    assert model.choice_lags == 0, "the formula spells the lag, so the shorthand must not repeat it"


def test_one_grouping_level_at_a_time() -> None:
    from behavio.design.formula import FormulaError

    with pytest.raises(FormulaError, match="one grouping level"):
        model_from_formula(
            "choice ~ stimulus + (1 | subject) + (1 | session)", BernoulliHistoryGLM()
        )


def test_a_composed_model_reports_both_of_its_axes_to_describe() -> None:
    """`describe()` must see through the outer combinator.

    Hierarchy declares no clock and smoothness declares no grouping, but a caller holds
    only the outermost wrapper. If it does not forward, a partially pooled model whose
    coefficients are paths reports itself as stationary and ungrouped -- which is exactly
    the pair of facts a reader needs to know what the fit means.
    """

    base = BernoulliHistoryGLM(predictors=("stimulus",))
    paths = smooth(base, over="session_order", knots=(0.0, 2.0))
    pooled = hierarchical(paths, over="subject")

    assert base.describe().clock is None
    assert base.describe().group is None
    assert paths.describe().clock == "session_order"
    assert pooled.describe().clock == "session_order"
    assert pooled.describe().group == "subject"
