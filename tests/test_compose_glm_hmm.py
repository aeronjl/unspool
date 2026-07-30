"""Composing a GLM-HMM: which cells opened, and the label evidence that opened them.

Three combinator cells were closed for this family. Two are still closed and the tests below
pin the sentences that close them. The third -- ``hierarchical()`` on the emission
coefficients -- is open, and the bar for opening it was never "the arithmetic fits": a
hierarchical latent-state model that quietly relabels one animal's states produces
well-typed nonsense with confidence intervals on it. So the evidence here is in three parts.

1. The **mechanism**: a GLM-HMM's likelihood is blind to per-subject relabelling, and the
   group prior is the only thing that is not. That is tested directly, on a study whose
   dynamics are symmetric so nothing else could be doing the work.
2. The **recovery**: a cohort in which one subject's states are genuinely permuted relative
   to the population -- so much so that fitting that subject alone puts its states in the
   opposite order -- is recovered with the population's labelling intact.
3. The **diagnostic**: :meth:`BernoulliGLMHMM.group_label_agreement` says which groups came
   back aligned and by what margin, because the argument in (1) is about the global optimum
   and a local optimizer is not obliged to find it.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
import pytest

from behavio import BernoulliGLMHMM, BernoulliHistoryGLM
from behavio.compare import compare_models
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth
from behavio.compose.hierarchy import HierarchicalFitResult
from behavio.contracts.bounded import uses_row_coefficients
from behavio.contracts.estimator import ModelDataError
from behavio.evaluate import cohort_forward_session_splits, evaluate_splits
from behavio.models.glm_hmm import GLMHMMParameters
from behavio.recovery import run_parameter_recovery
from behavio.trials import Study

# --------------------------------------------------------------------------------------
# Shared design and model
# --------------------------------------------------------------------------------------


def cohort(
    subjects: Sequence[str], *, n_sessions: int = 2, trials: int = 100, seed: int = 91
) -> Study:
    """One stimulus-driven design, one reset block per subject and session."""

    generator = np.random.default_rng(seed)
    columns: dict[str, list[object]] = {
        name: [] for name in ("subject", "session", "trial", "session_order", "stimulus")
    }
    for subject in subjects:
        for session in range(n_sessions):
            for trial in range(trials):
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-s{session}")
                columns["trial"].append(trial)
                columns["session_order"].append(session)
                columns["stimulus"].append(float(generator.normal()))
    return Study(columns)


def switching_model(*, n_restarts: int = 1, **overrides: object) -> BernoulliGLMHMM:
    return BernoulliGLMHMM(
        predictors=("stimulus",),
        choice_lags=0,
        l2=0.01,
        n_restarts=n_restarts,
        max_iterations=300,
        random_seed=7,
        **overrides,  # type: ignore[arg-type]
    )


def population_parameters(model: BernoulliGLMHMM) -> Mapping[str, float]:
    """A disengaged state and an engaged state, under deliberately symmetric dynamics.

    Symmetric transitions and a symmetric initial distribution are the point of this
    fixture, not an accident of it: they are what make relabelling one subject's emissions
    an *exact* symmetry of the likelihood, so the group prior is provably the only thing
    identifying the labels.
    """

    return model.parameters_from_components(
        initial_probabilities=[0.5, 0.5],
        transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
        emissions={"intercept": [-2.0, 2.0], "stimulus": [0.5, 3.5]},
    )


# One subject's intercepts are reversed relative to the population, and by enough that its
# own ``label_by`` ordering disagrees with the population's. The deviations sum to zero
# across subjects so the population remains what it was declared to be.
PERMUTED_COHORT_DEVIATIONS = {"a": [-1.3, 1.3], "b": [-1.3, 1.3], "c": [2.6, -2.6]}


@lru_cache(maxsize=1)
def permuted_cohort_study(model: BernoulliGLMHMM) -> tuple[Study, np.ndarray]:
    pooled = hierarchical(model, over="subject", parameters=("intercept",), scale=1.5)
    simulation = pooled.simulate_with_effects(
        cohort(["a", "b", "c"]),
        population_parameters(model),
        seed=3,
        group_deviations=PERMUTED_COHORT_DEVIATIONS,
    )
    return simulation.study, np.asarray(simulation.group_parameter_vectors)


@lru_cache(maxsize=1)
def permuted_cohort_fit(model: BernoulliGLMHMM) -> HierarchicalFitResult:
    """One joint fit, shared by the tests that read different things off it.

    Cached because the fit is the expensive part and every assertion below is about the same
    fit: a model is frozen and a study is immutable, so there is only one answer to compute.
    """

    study, _ = permuted_cohort_study(model)
    return hierarchical(model, over="subject", parameters=("intercept",), scale=1.5).fit(study)


def emissions_of(model: BernoulliGLMHMM, vector: np.ndarray) -> np.ndarray:
    return model.parameter_components(
        dict(zip(model.parameter_names, np.asarray(vector).tolist(), strict=True))
    ).emission_coefficients


# --------------------------------------------------------------------------------------
# 1. Which cells are closed, and what closes them
# --------------------------------------------------------------------------------------


def test_a_glm_hmm_still_declines_a_mixture() -> None:
    """Widening ``mix()`` to row objectives did not open this cell, and could not have.

    ``mix()`` is now gated on row independence rather than on a linear predictor, which is
    exactly the thing a forward recursion does not have. The sentence it reports is the
    model's own ``independent_rows_refusal`` and it is a modelling statement: a lapse on a
    GLM-HMM is a lapse on the emission, inside the recursion.
    """

    model = switching_model()

    with pytest.raises(TypeError) as error:
        mix(model, UniformChoiceGuess())

    message = str(error.value)
    assert "rows are not independent" in message
    assert "inside that recursion" in message
    assert "absorb the state switching" in message


def test_a_glm_hmm_declines_a_path_in_clock_time() -> None:
    model = switching_model()

    with pytest.raises(TypeError) as error:
        smooth(model, over="session_order", knots=(0.0, 1.0), parameters=("intercept",))

    message = str(error.value)
    assert "ordering of coefficient *paths*" in message
    assert "where the paths do not cross" in message


def test_a_glm_hmm_declines_group_deviations_on_the_simplex() -> None:
    model = switching_model()
    transition = "transition_logit[from=0,to=0|reference=1]"

    with pytest.raises(TypeError) as every:
        hierarchical(model, over="subject")
    with pytest.raises(TypeError) as simplex:
        hierarchical(model, over="subject", parameters=(transition,))
    with pytest.raises(TypeError) as per_state:
        hierarchical(model, over="subject", parameters=("state[0].intercept",))

    assert "reference-category logits" in str(every.value)
    assert "prior on that chart rather than on" in str(every.value)
    assert "state 1 is the reference" in str(simplex.value)
    assert "for every state at once or not at all" in str(per_state.value)


def test_a_sticky_glm_hmm_declines_composition_by_name() -> None:
    with pytest.raises(TypeError) as error:
        hierarchical(switching_model(stickiness=2.0), over="subject", parameters=("intercept",))

    assert "neither a per-row score nor a quadratic penalty" in str(error.value)


def test_a_grouping_that_cuts_a_session_is_refused_rather_than_averaged() -> None:
    model = switching_model()
    study = model.simulate(cohort(["a", "b"], trials=20), population_parameters(model), seed=1)

    with pytest.raises(ValueError, match="splits a block this model's likelihood recurses over"):
        hierarchical(model, over="trial", parameters=("intercept",)).fit(study)


def test_the_inherited_bernoulli_likelihood_is_withdrawn() -> None:
    model = switching_model()

    with pytest.raises(AttributeError, match="marginalises over a latent state path"):
        _ = model.likelihood
    assert uses_row_coefficients(model)


def test_a_common_scale_multiplier_is_declined_with_the_alternative_named() -> None:
    model = switching_model()
    study = model.simulate(cohort(["a", "b"], trials=20), population_parameters(model), seed=1)
    pooled = hierarchical(
        model,
        over="subject",
        parameters=("intercept",),
        scale=0.5,
        estimate_scale=True,
        scale_estimator="laplace-profile",
    )

    with pytest.raises(ModelDataError, match="scale_estimator='laplace-em'"):
        pooled.fit(study)


# --------------------------------------------------------------------------------------
# 2. What the coordinate looks like once it composes
# --------------------------------------------------------------------------------------


def test_only_emission_coefficients_are_ridged() -> None:
    model = switching_model()

    diagonal = np.diag(model.penalty_matrix())

    assert diagonal.shape == (len(model.parameter_names),)
    assert diagonal.tolist() == [0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0]


def test_a_coefficient_varies_by_group_for_every_state_at_once() -> None:
    model = switching_model(n_states=3)

    assert model.group_parameter_expansion("stimulus") == (
        "state[0].stimulus",
        "state[1].stimulus",
        "state[2].stimulus",
    )
    assert model.coordinate_box(cohort(["a"], trials=4)).shape == (
        len(model.parameter_names),
        2,
    )


def test_the_row_objective_blocks_by_session_and_refuses_a_coordinate_that_varies() -> None:
    model = switching_model()
    study = model.simulate(cohort(["a", "b"], trials=15), population_parameters(model), seed=2)
    objective = model.row_objective(study)
    vector = np.asarray(
        [population_parameters(model)[name] for name in model.parameter_names], dtype=np.float64
    )
    rows = np.tile(vector, (len(study), 1))

    value, gradient = objective.value_and_gradient(rows)
    rows[0, 0] += 0.1

    assert len(np.unique(objective.row_blocks)) == 4
    assert np.isfinite(value) and gradient.shape == rows.shape
    with pytest.raises(ValueError, match="constant within each block of the recursion"):
        objective.value_and_gradient(rows)


def test_the_row_gradient_matches_a_finite_difference() -> None:
    model = switching_model()
    study = model.simulate(cohort(["a", "b"], trials=25), population_parameters(model), seed=4)
    objective = model.row_objective(study)
    vector = np.asarray(
        [population_parameters(model)[name] for name in model.parameter_names], dtype=np.float64
    )
    rows = np.tile(vector, (len(study), 1))
    _, gradient = objective.value_and_gradient(rows)

    step = 1e-6
    for position in range(len(vector)):
        forward = rows.copy()
        forward[:, position] += step
        backward = rows.copy()
        backward[:, position] -= step
        difference = (
            objective.value_and_gradient(forward)[0] - objective.value_and_gradient(backward)[0]
        ) / (2.0 * step)
        assert difference == pytest.approx(float(gradient[:, position].sum()), abs=1e-4)


def test_simulate_rows_reproduces_the_models_own_simulator() -> None:
    model = switching_model()
    design = cohort(["a", "b"], trials=30)
    parameters = population_parameters(model)
    vector = np.asarray([parameters[name] for name in model.parameter_names], dtype=np.float64)

    direct = model.simulate(design, parameters, seed=17)
    by_row = model.simulate_rows(design, np.tile(vector, (len(design), 1)), seed=17)

    assert np.array_equal(np.asarray(direct["choice"]), np.asarray(by_row["choice"]))


def test_predict_rows_reproduces_the_models_own_filter() -> None:
    model = switching_model()
    design = cohort(["a", "b"], trials=30)
    parameters = population_parameters(model)
    study = model.simulate(design, parameters, seed=19)
    fit = model.fit(study)
    vector = np.asarray(fit.estimates, dtype=np.float64)

    direct = model.predict(study, fit)
    by_row = model.predict_rows(study, np.tile(vector, (len(study), 1)), mode=direct.mode)

    assert np.allclose(direct.probability, by_row.probability, atol=1e-12)
    assert np.allclose(
        model.pointwise_log_prob(study, fit),
        model.pointwise_log_prob_rows(study, np.tile(vector, (len(study), 1))),
        atol=1e-12,
    )


# --------------------------------------------------------------------------------------
# 3. Relabelling is linear, and it is the only symmetry hierarchy leaves standing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_relabelling_is_an_exact_linear_map_on_this_coordinate(n_states: int) -> None:
    model = BernoulliGLMHMM(predictors=("stimulus",), choice_lags=1, n_states=n_states)
    vector = np.random.default_rng(n_states).normal(size=len(model.parameter_names))
    components = model.parameter_components(
        dict(zip(model.parameter_names, vector.tolist(), strict=True))
    )

    for permutation in itertools.permutations(range(n_states)):
        order = np.asarray(permutation, dtype=np.intp)
        relabelled = GLMHMMParameters(
            initial_probabilities=components.initial_probabilities[order],
            transition_matrix=components.transition_matrix[np.ix_(order, order)],
            emission_coefficients=components.emission_coefficients[order],
            coefficient_names=model.coefficient_names,
        )
        expected = np.concatenate(
            (
                relabelled.emission_coefficients.ravel(),
                np.log(relabelled.initial_probabilities[:-1])
                - np.log(relabelled.initial_probabilities[-1]),
                (
                    np.log(relabelled.transition_matrix[:, :-1])
                    - np.log(relabelled.transition_matrix[:, -1:])
                ).ravel(),
            )
        )
        assert np.allclose(model.relabelling_map(permutation) @ vector, expected, atol=1e-10)


def test_the_likelihood_is_blind_to_labels_and_only_the_group_prior_is_not() -> None:
    """The whole argument for hierarchy on a latent-state model, in one test.

    Relabelling one subject's states is an exact symmetry of a GLM-HMM's likelihood when the
    dynamics are symmetric -- the forward recursion cannot tell the two apart, at any sample
    size. What can tell them apart is that a relabelled subject is *far from the population*,
    and the joint fit pays a Gaussian price for that. If the price were not there, the fitted
    deviations would be meaningless whatever the fit reported.
    """

    model = switching_model()
    study = model.simulate(cohort(["a", "b"], trials=40), population_parameters(model), seed=6)
    objective = model.row_objective(study)
    population = np.asarray(
        [population_parameters(model)[name] for name in model.parameter_names], dtype=np.float64
    )
    columns = np.asarray([0, 2], dtype=np.intp)  # both states' intercepts
    subject_b = np.asarray([value == "b" for value in study["subject"]])
    rows = np.tile(population, (len(study), 1))
    rows[subject_b[:, None] & (np.arange(len(population)) == 0)] += 0.4
    rows[subject_b[:, None] & (np.arange(len(population)) == 2)] -= 0.4

    swap = model.relabelling_map((1, 0))
    swapped = rows.copy()
    swapped[subject_b] = rows[subject_b] @ swap.T

    anchored, _ = objective.value_and_gradient(rows)
    relabelled, _ = objective.value_and_gradient(swapped)
    before = rows[subject_b][0][columns] - population[columns]
    after = swapped[subject_b][0][columns] - population[columns]

    assert relabelled == pytest.approx(anchored, abs=1e-9)
    assert float(after @ after) > 20.0 * float(before @ before)


def test_relabelling_every_subject_leaves_the_joint_problem_where_it_was() -> None:
    """The one label symmetry hierarchy does not break, which is why ``fit_rows`` resolves it."""

    model = switching_model()
    study = model.simulate(cohort(["a", "b"], trials=40), population_parameters(model), seed=6)
    objective = model.row_objective(study)
    population = np.asarray(
        [population_parameters(model)[name] for name in model.parameter_names], dtype=np.float64
    )
    rows = np.tile(population, (len(study), 1))
    swap = model.relabelling_map((1, 0))

    assert objective.value_and_gradient(rows @ swap.T)[0] == pytest.approx(
        objective.value_and_gradient(rows)[0], abs=1e-9
    )


def test_a_composed_fit_reports_the_population_in_canonical_label_order() -> None:
    model = switching_model()
    population = emissions_of(model, permuted_cohort_fit(model).estimates)

    label = model.coefficient_names.index(model.label_by)
    assert population[0, label] < population[1, label]


# --------------------------------------------------------------------------------------
# 4. Recovery with the labels genuinely at risk
# --------------------------------------------------------------------------------------


def test_fitting_the_permuted_subject_alone_really_does_switch_its_labels() -> None:
    """The risk is real before it is shown to be handled.

    Subject ``c``'s intercepts are reversed relative to the population's, so this model's own
    ``label_by`` canonicalisation -- which orders states by intercept -- puts ``c``'s
    stimulus-sensitive state at index 0 while every other subject's sits at index 1. Fitting
    subjects one at a time and stacking the results would compare state 0 of one animal with
    state 1 of another.
    """

    model = switching_model()
    study, _ = permuted_cohort_study(model)

    steep_state = {}
    for subject in ("a", "b", "c"):
        rows = [index for index, value in enumerate(study["subject"]) if value == subject]
        alone = Study({name: [study[name][index] for index in rows] for name in study.columns})
        emissions = emissions_of(model, model.fit(alone).estimates)
        steep_state[subject] = int(
            np.argmax(emissions[:, model.coefficient_names.index("stimulus")])
        )

    assert steep_state["a"] == steep_state["b"] == 1
    assert steep_state["c"] == 0


def test_a_hierarchical_glm_hmm_recovers_a_cohort_whose_labels_are_at_risk() -> None:
    """One joint fit keeps every subject on the population's labelling, and says so."""

    model = switching_model()
    _, truth_vectors = permuted_cohort_study(model)

    fit = permuted_cohort_fit(model)
    fitted = np.asarray(fit.group_parameter_vectors)
    agreement = model.group_label_agreement(fit)
    steep = model.coefficient_names.index("stimulus")

    # Every subject's steep state is the population's steep state, subject c included.
    for vector in fitted:
        assert int(np.argmax(emissions_of(model, vector)[:, steep])) == 1
    assert agreement.all_aligned
    assert agreement.relabelled_groups == ()

    # The reversal is reported as a deviation, not absorbed by a relabelling.
    label = model.coefficient_names.index(model.label_by)
    for vector, truth in zip(fitted, truth_vectors, strict=True):
        recovered = emissions_of(model, vector)[:, label]
        declared = emissions_of(model, truth)[:, label]
        assert np.sign(recovered[1] - recovered[0]) == np.sign(declared[1] - declared[0])

    # The population is recovered despite one subject sitting on the wrong side of it, and
    # the anchor's strength differs by subject exactly where it should.
    population = emissions_of(model, fit.estimates)
    assert population[:, label] == pytest.approx([-2.0, 2.0], abs=0.75)
    assert population[:, steep] == pytest.approx([0.5, 3.5], abs=0.75)
    assert agreement.margins[2] < 0.5 * min(agreement.margins[0], agreement.margins[1])


def test_group_label_agreement_names_a_group_whose_states_are_swapped() -> None:
    model = switching_model()
    fit = permuted_cohort_fit(model)
    swap = model.relabelling_map((1, 0))
    vectors = np.asarray(fit.group_parameter_vectors).copy()
    vectors[1] = swap @ vectors[1]

    agreement = model.group_label_agreement(
        _Relabelled(estimates=fit.estimates, groups=fit.groups, group_parameter_vectors=vectors)
    )

    assert agreement.aligned == (True, False, True)
    assert agreement.relabelled_groups == ("b",)


class _Relabelled:
    """A stand-in for a hierarchical fit whose second group came back permuted."""

    def __init__(
        self, *, estimates: np.ndarray, groups: tuple, group_parameter_vectors: np.ndarray
    ):
        self.estimates = estimates
        self.groups = groups
        self.group_parameter_vectors = group_parameter_vectors


# --------------------------------------------------------------------------------------
# 5. The composed model is an ordinary estimator
# --------------------------------------------------------------------------------------


def test_a_hierarchical_glm_hmm_flows_through_the_evaluation_stack() -> None:
    model = switching_model()
    pooled = hierarchical(model, over="subject", parameters=("intercept",), scale=1.0)
    truth = dict(population_parameters(model))
    design = cohort(["a", "b"], n_sessions=3, trials=60)
    study = pooled.simulate(design, truth, seed=5)
    splits = cohort_forward_session_splits(study, min_train_sessions=2, horizon=1)

    evaluations = evaluate_splits(pooled, study, splits)
    comparison = compare_models(
        {"pooled": pooled, "static": BernoulliHistoryGLM(predictors=("stimulus",), l2=0.5)},
        study,
        splits,
        bootstrap_resamples=50,
    )
    recovery = run_parameter_recovery(pooled, design, [truth], repeats=1, seed=3)
    description = pooled.describe(study)

    assert len(evaluations) == len(splits)
    assert all(np.isfinite(evaluation.mean_log_loss) for evaluation in evaluations)
    assert set(comparison.model_order) == {"pooled", "static"}
    assert len(recovery.summary()) == len(pooled.parameter_names)
    assert np.all(np.isfinite(recovery.estimates))
    assert description.errors == ()
    assert description.parameter_names == model.parameter_names
    assert any("deviations by subject" in prior for prior in description.priors)


def test_an_unseen_subject_is_predicted_by_integrating_the_fitted_random_effect() -> None:
    model = switching_model()
    pooled = hierarchical(model, over="subject", parameters=("intercept",), scale=1.0)
    truth = dict(population_parameters(model))
    study = pooled.simulate(cohort(["a", "b"], trials=40), truth, seed=5)
    fresh = pooled.simulate(cohort(["d"], trials=40, seed=13), truth, seed=7)
    fit = pooled.fit(study)

    predictive = pooled.predict_new_groups(fresh, fit, n_draws=8, seed=2)

    assert predictive.groups == ("d",)
    assert predictive.probability.shape == (len(fresh),)
    assert np.all(np.isfinite(predictive.group_joint_log_probability))
    assert not fit.group_was_fitted("d")


def test_three_states_expand_into_a_joint_coordinate_of_the_declared_width() -> None:
    model = switching_model(n_states=3)
    pooled = hierarchical(model, over="subject", parameters=("intercept",), scale=0.5)

    assert pooled.varying_parameters == (
        "state[0].intercept",
        "state[1].intercept",
        "state[2].intercept",
    )
    assert pooled.parameter_names == model.parameter_names
    assert pooled.scale_parameters == ("intercept",)
