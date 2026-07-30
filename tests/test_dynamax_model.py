"""The dynamax wrapper: can a *latent-state* foreign model satisfy the contract honestly?

The PyDDM wrapper asked whether a foreign optimizer could satisfy
:class:`~behavio.contracts.BehaviourEstimator`, and Bambi asked the same of the sampled
half. This one asks the question ``AGENTS.md`` cares about most and that no model in the
package had ever been able to pose: **can a model that genuinely smooths declare that it
smooths, and can the package tell the difference?** Everything else here -- the ragged EM
loop, the covariance differentiated out of the marginal likelihood, the canonical state
order -- exists so that the answer is measured rather than asserted.

Gated on the optional extra exactly as the PyDDM, Bambi, NWB, DANDI, ONE, PyBADS and
Parquet suites are: ``pytest.importorskip`` at module scope, so a checkout without
``behavio[dynamax]`` skips this file cleanly. The tests that assert what a machine
*without* the extra sees deliberately block the import instead.

Fitting is EM over a jitted jax graph, so the study and the fit are module-scoped fixtures
and the configuration is the smallest that still converges: two states separated by about
1.4 standard deviations, which is close enough that smoothing carries real information and
far enough that EM reaches a stationary point.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from behavio import ScoreMetric, Study, compare_models, forward_session_splits
from behavio.adapters import check_behaviour_estimator
from behavio.adapters.conformance import CheckStatus
from behavio.adapters.estimator_conformance import perturb_future_rows
from behavio.contracts import (
    BehaviourEstimator,
    DensityBehaviourEstimator,
    DensityPrediction,
    FitAuditStatus,
    GenerativeBehaviourModel,
    ModelDataError,
    PredictionMode,
    model_capabilities,
)
from behavio.evaluate import evaluate_splits
from behavio.foreign import DYNAMAX_EXTRA, DYNAMAX_SERIES, ForeignPackageUnavailableError
from behavio.recovery import run_parameter_recovery
from behavio.trials import SequenceGrouping, sequence_layout

pytest.importorskip("dynamax")

from behavio.foreign.dynamax import (
    PARAMETER_CORRESPONDENCE,
    DynamaxFitResult,
    DynamaxSwitchingAutoregression,
    SwitchingStateProbabilities,
    _Backend,
)

TRUTH: dict[str, Any] = {
    "initial": [0.5, 0.5],
    "transitions": [[0.92, 0.08], [0.12, 0.88]],
    "biases": [-1.0, 1.0],
    "weights": [[0.3], [0.3]],
    "variances": [0.7, 0.7],
}
"""Two overlapping, sticky regimes with the same autoregression and the same noise.

Deliberately overlapping. Two states six standard deviations apart are identified by each
observation on its own, their smoothed and filtered state posteriors coincide to machine
precision, and the conformance harness then reports -- correctly -- that a model claiming to
smooth did not measurably use the future. See
``test_the_smoothed_check_is_falsifiable_in_the_other_direction_too``.
"""


def wrapper(**overrides: Any) -> DynamaxSwitchingAutoregression:
    settings: dict[str, Any] = {
        "outcome": "speed",
        "n_states": 2,
        "num_lags": 1,
        "n_restarts": 2,
        "em_iterations": 200,
    }
    settings.update(overrides)
    return DynamaxSwitchingAutoregression(**settings)


def design(n_trials: int = 30) -> Study:
    """Two subjects, two sessions each: four sequences the state chain must reset between."""

    columns: dict[str, list[Any]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "session_order": [],
        "speed": [],
    }
    for subject in ("m1", "m2"):
        for order in (0, 1):
            for trial in range(n_trials):
                columns["subject"].append(subject)
                columns["session"].append(f"{subject}-d{order}")
                columns["trial"].append(trial)
                columns["session_order"].append(order)
                columns["speed"].append(0.0)
    return Study(columns)


@pytest.fixture(scope="module")
def truth() -> dict[str, float]:
    return dict(wrapper().parameters_from_components(**TRUTH))


@pytest.fixture(scope="module")
def study(truth: dict[str, float]) -> Study:
    return wrapper().simulate(design(), truth, seed=0)


@pytest.fixture(scope="module")
def fit(study: Study) -> DynamaxFitResult:
    return wrapper().fit(study)


# -- the contract ------------------------------------------------------------------------


def test_the_wrapper_is_a_behaviour_estimator_that_predicts_a_density() -> None:
    model = wrapper()

    assert isinstance(model, BehaviourEstimator)
    assert isinstance(model, GenerativeBehaviourModel)
    assert isinstance(model, DensityBehaviourEstimator)
    capabilities = model_capabilities(model)
    assert capabilities.scored_columns == ("speed",)
    assert capabilities.required_task_columns == ()
    assert capabilities.can_simulate and capabilities.can_recover_parameters
    assert not capabilities.is_sampled


def test_it_declares_both_prediction_modes_and_it_is_the_first_model_that_does() -> None:
    """Every first-party model declares filtered only; this one has a smoother."""

    assert wrapper().supported_prediction_modes == (
        PredictionMode.FILTERED,
        PredictionMode.SMOOTHED,
    )


def test_the_zero_lag_model_is_a_nested_gaussian_hmm() -> None:
    """``num_lags=0`` is the same model with every autoregressive weight fixed at zero.

    Its parameter names are a strict subset of the autoregression's, which is what makes
    the pair a nested comparison rather than two unrelated candidates.
    """

    plain, autoregressive = wrapper(num_lags=0), wrapper(num_lags=1)

    assert set(plain.parameter_names) < set(autoregressive.parameter_names)
    assert [name for name in autoregressive.parameter_names if "lag" in name] == [
        "state[0].lag1",
        "state[1].lag1",
    ]
    assert plain.signature != autoregressive.signature


def test_the_parameter_correspondence_names_the_two_that_could_silently_be_wrong() -> None:
    """dynamax renames the offset between families and stores a variance as a matrix."""

    assert "means" in PARAMETER_CORRESPONDENCE["state[k].bias"]
    assert "biases" in PARAMETER_CORRESPONDENCE["state[k].bias"]
    assert "covariance matrix" in PARAMETER_CORRESPONDENCE["state[k].variance"]


# -- the conformance harness -------------------------------------------------------------


def test_every_conformance_check_passes_including_the_smoothed_one(
    study: Study, fit: DynamaxFitResult, truth: dict[str, float]
) -> None:
    """The whole estimator half of the contract, executed rather than described."""

    report = check_behaviour_estimator(wrapper(), study, fit=fit, simulation_parameters=truth)

    assert report.passed, report.summary()
    executed = {check.name: check.status for check in report.checks}
    assert executed["filtered-prediction-ignores-future-rows"] is CheckStatus.PASSED
    assert executed["filtered-score-ignores-future-rows"] is CheckStatus.PASSED
    assert executed["smoothed-prediction-uses-future-rows"] is CheckStatus.PASSED
    # The one skip, and it is skipped because the model declares every mode there is.
    assert [check.name for check in report.skipped] == ["refuses-undeclared-prediction-modes"]


def test_the_filtered_prediction_is_unchanged_by_the_future_and_the_smoothed_one_is_not(
    study: Study, fit: DynamaxFitResult
) -> None:
    """The distinction ``AGENTS.md`` requires, measured on one fit rather than declared.

    Same fit, same rows, one perturbation: relabelling the second half of every sequence
    must leave the one-step-ahead predictive density of the first half bit-identical and
    must move the smoothed description of exactly those rows.
    """

    model = wrapper()
    perturbed, past = perturb_future_rows(study, columns=("speed",))
    assert past.size and not np.array_equal(np.asarray(study["speed"]), perturbed["speed"])

    filtered = model.predict(study, fit).density[past]
    filtered_again = model.predict(perturbed, fit).density[past]
    smoothed = model.predict(study, fit, mode=PredictionMode.SMOOTHED).density[past]
    smoothed_again = model.predict(perturbed, fit, mode=PredictionMode.SMOOTHED).density[past]

    assert np.array_equal(filtered, filtered_again)
    assert np.max(np.abs(smoothed - smoothed_again)) > 1e-6


def test_the_smoothed_check_is_falsifiable_in_the_other_direction_too(
    truth: dict[str, float],
) -> None:
    """Well-separated states make the harness report that smoothing did nothing, and it is right.

    This is the finding the check exists to produce and the reason the fixtures above use
    overlapping states. When each observation identifies its own state, the backward message
    carries no information, the smoothed and filtered state posteriors agree to machine
    precision, and a model that *does* smooth is indistinguishable from one that does not --
    on that data. The harness measures the model on the study it is given rather than
    reading its declaration, so it says so instead of passing.
    """

    model = wrapper(em_iterations=40, n_restarts=1)
    separated = dict(
        model.parameters_from_components(
            initial=[0.5, 0.5],
            transitions=[[0.9, 0.1], [0.15, 0.85]],
            biases=[-2.5, 2.5],
            weights=[[0.1], [0.1]],
            variances=[0.2, 0.2],
        )
    )
    obvious = model.simulate(design(), separated, seed=3)
    separated_fit = model.fit(obvious)
    perturbed, past = perturb_future_rows(obvious, columns=("speed",))

    smoothed = model.predict(obvious, separated_fit, mode=PredictionMode.SMOOTHED).density[past]
    replayed = model.predict(perturbed, separated_fit, mode=PredictionMode.SMOOTHED).density[past]

    assert np.max(np.abs(smoothed - replayed)) < 1e-9
    report = check_behaviour_estimator(model, obvious, fit=separated_fit)
    failures = [check.name for check in report.failures]
    assert failures == ["smoothed-prediction-uses-future-rows"]


# -- the scoring identity ------------------------------------------------------------------


def test_the_filtered_scores_sum_to_dynamaxs_own_marginal_log_likelihood(
    study: Study, fit: DynamaxFitResult
) -> None:
    """``sum_t log p(y_t | y_{1:t-1})`` is the chain rule, not an approximation of it.

    This is the claim that makes the pointwise score and the objective EM maximised the same
    quantity decomposed two ways. It holds only because the filtered mode mixes under
    ``predicted_probs``; mixing under ``filtered_probs`` would condition each row on itself
    and the sum would be nonsense.
    """

    model = wrapper()
    backend = _Backend(model)
    layout = sequence_layout(study, grouping=SequenceGrouping.SESSION)
    blocks = backend.blocks(study, layout)
    parameters = backend.parameters(model._components(fit))
    marginal = float(backend._log_joint(parameters, blocks) - backend._hmm.log_prior(parameters))

    scores = model.pointwise_log_prob(study, fit)

    assert float(np.sum(scores)) == pytest.approx(marginal, abs=1e-8)


def test_a_smoothed_score_is_larger_than_a_filtered_one_and_is_never_the_default(
    study: Study, fit: DynamaxFitResult
) -> None:
    """A smoothed score conditions on the row it is scoring, so it cannot be a held-out score."""

    model = wrapper()

    filtered = model.pointwise_log_prob(study, fit)
    smoothed = model.pointwise_log_prob(study, fit, mode=PredictionMode.SMOOTHED)

    assert np.sum(smoothed) > np.sum(filtered)
    assert np.array_equal(filtered, model.pointwise_log_prob(study, fit))
    assert model.predict(study, fit).mode is PredictionMode.FILTERED


# -- shape: the sequence layout ------------------------------------------------------------


def test_the_ragged_em_loop_reproduces_dynamaxs_own_fit_em_exactly(study: Study) -> None:
    """The adapter owns the batching and nothing else: same E-step, same M-step, same answer.

    On an equal-length batch dynamax can do the batching itself, so the two must agree. They
    agree to floating point, which is what licenses replacing ``fit_em``'s vmap over one
    padded array with a vmap over each length partition.
    """

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    model = wrapper(em_iterations=25, n_restarts=1)
    backend = _Backend(model)
    layout = sequence_layout(study, grouping=SequenceGrouping.SESSION)
    blocks = backend.blocks(study, layout)
    assert len(blocks) == 1 and len(blocks[0].positions) == 4
    start = backend._start(blocks, 0)

    ours, _ = backend.expectation_maximization(blocks, 0)
    theirs, _ = backend._hmm.fit_em(
        start, backend._props, blocks[0].emissions, blocks[0].inputs, num_iters=25, verbose=False
    )

    differences = jax.tree.leaves(
        jax.tree.map(lambda left, right: float(jnp.max(jnp.abs(left - right))), ours, theirs)
    )
    assert max(differences) < 1e-9


def test_zero_padding_a_ragged_batch_changes_the_answer_which_is_why_there_is_none(
    truth: dict[str, float],
) -> None:
    """dynamax's ``fit_em`` takes no mask, so padded rows are fitted as real observations.

    This is the measurement behind the wrapper's refusal to build a
    ``(n_sequences, max_length, 1)`` tensor. Behavioural sessions are ragged essentially
    always, and the effect is not a rounding difference.
    """

    jnp = pytest.importorskip("jax.numpy")
    ragged = _ragged_study(truth)
    model = wrapper(em_iterations=40, n_restarts=1)
    backend = _Backend(model)
    layout = sequence_layout(ragged, grouping=SequenceGrouping.SESSION)
    blocks = backend.blocks(ragged, layout)
    assert layout.lengths == (40, 33, 26, 19)
    honest, _ = backend.expectation_maximization(blocks, 0)

    longest = max(layout.lengths)
    padded = np.zeros((layout.n_sequences, longest, 1))
    padded_inputs = np.zeros((layout.n_sequences, longest, 1))
    for position, block in enumerate(layout.split(np.asarray(ragged["speed"], dtype=np.float64))):
        padded[position, : block.size, 0] = block
        padded_inputs[position, 1 : block.size, 0] = block[:-1]
    contaminated, _ = backend._hmm.fit_em(
        backend._start(blocks, 0),
        backend._props,
        jnp.asarray(padded),
        jnp.asarray(padded_inputs),
        num_iters=40,
        verbose=False,
    )

    gap = float(
        np.max(
            np.abs(np.asarray(honest.emissions.biases) - np.asarray(contaminated.emissions.biases))
        )
    )
    # The two states' true offsets are one apart, so this is most of the signal.
    assert gap > 0.5


def test_source_row_order_does_not_change_the_fit(truth: dict[str, float]) -> None:
    """``join(split(v)) == v``: interleaving two subjects' rows must change nothing.

    The failure this catches is silent. A per-sequence array assembled in the wrong order
    still has the right length, and a prediction written back in sorted order rather than
    source order still validates.
    """

    model = wrapper(em_iterations=40, n_restarts=1)
    ordered = model.simulate(design(), truth, seed=1)
    shuffled = ordered.take(np.argsort(np.asarray(ordered["trial"], dtype=np.int64), kind="stable"))
    assert not np.array_equal(np.asarray(ordered["speed"]), np.asarray(shuffled["speed"]))

    first, second = model.fit(ordered), model.fit(shuffled)

    assert np.allclose(first.estimates, second.estimates, atol=1e-9)
    reordered = model.pointwise_log_prob(shuffled, second)
    lookup = {
        (str(session), int(trial)): value
        for session, trial, value in zip(
            shuffled["session"], shuffled["trial"], reordered, strict=True
        )
    }
    direct = model.pointwise_log_prob(ordered, first)
    for index, (session, trial) in enumerate(
        zip(ordered["session"], ordered["trial"], strict=True)
    ):
        assert lookup[(str(session), int(trial))] == pytest.approx(direct[index], abs=1e-9)


def test_the_autoregressive_history_resets_at_a_session_boundary(
    study: Study, fit: DynamaxFitResult
) -> None:
    """A session's first trial is conditioned on dynamax's zeros, not on last night's last trial.

    The lag is built per sequence, so changing the *last* trial of one session cannot move
    the density of the *first* trial of the next -- which is what would happen if the
    wrapper lagged the flat study.
    """

    model = wrapper()
    layout = sequence_layout(study, grouping=SequenceGrouping.SESSION)
    tampered = np.array(study["speed"], copy=True)
    last_of_first = int(layout.sequences[0].indices[-1])
    first_of_second = int(layout.sequences[1].indices[0])
    tampered[last_of_first] += 25.0
    altered = Study({**{name: study[name] for name in study.columns}, "speed": tampered})

    before = model.pointwise_log_prob(study, fit)
    after = model.pointwise_log_prob(altered, fit)

    assert before[first_of_second] == pytest.approx(after[first_of_second], abs=1e-12)
    assert before[last_of_first] != pytest.approx(after[last_of_first], abs=1e-6)


def test_state_probabilities_reports_three_posteriors_that_are_not_the_same_thing(
    study: Study, fit: DynamaxFitResult
) -> None:
    """``predicted``, ``filtered`` and ``smoothed`` answer three different questions."""

    posteriors = wrapper().state_probabilities(study, fit)

    assert isinstance(posteriors, SwitchingStateProbabilities)
    assert posteriors.n_states == 2
    for name in ("predictive", "filtered", "smoothed"):
        values = getattr(posteriors, name)
        assert values.shape == (len(study), 2)
        assert np.allclose(values.sum(axis=1), 1.0)
    assert not np.allclose(posteriors.predictive, posteriors.filtered)
    assert not np.allclose(posteriors.filtered, posteriors.smoothed)
    decoded = wrapper().most_likely_states(study, fit)
    assert decoded.shape == (len(study),) and set(np.unique(decoded)) <= {0, 1}


# -- the covariance EM does not hand you ---------------------------------------------------


def test_the_covariance_is_the_curvature_of_the_objective_em_maximised(
    fit: DynamaxFitResult,
) -> None:
    """dynamax reports a pytree; jax reports the second derivative of the log joint."""

    assert fit.covariance_is_estimated
    assert fit.covariance.shape == (len(fit.parameter_names),) * 2
    assert np.allclose(fit.covariance, fit.covariance.T)
    assert np.all(np.isfinite(fit.standard_errors)) and np.all(fit.standard_errors >= 0)
    assert np.allclose(fit.standard_errors, np.sqrt(np.diag(fit.covariance)))
    assert fit.diagnostics.hessian_condition is not None
    assert np.isfinite(fit.diagnostics.hessian_condition)


def test_the_reported_covariance_is_singular_along_the_sum_to_one_directions(
    fit: DynamaxFitResult,
) -> None:
    """Reporting whole simplexes buys readability at the price of a rank-deficient covariance.

    That is the correct variance rather than a defect: a transition row cannot move in the
    direction that changes its total, so the variance in that direction is zero. It is also
    the reason the fit records the condition number of the *unconstrained* information.
    """

    names = list(fit.parameter_names)
    row = [names.index(f"transition[0->{target}]") for target in range(2)]
    block = fit.covariance[np.ix_(row, row)]

    assert float(np.sum(block)) == pytest.approx(0.0, abs=1e-8)
    assert np.linalg.matrix_rank(fit.covariance, tol=1e-8) < len(names)


def test_convergence_is_exact_stationarity_and_an_under_iterated_fit_fails_it(
    study: Study, fit: DynamaxFitResult
) -> None:
    """EM cannot fail, so "converged" had to be measured rather than read off a flag."""

    assert fit.diagnostics.converged is True
    assert fit.diagnostics.gradient_norm is not None
    assert fit.diagnostics.gradient_norm <= wrapper().gradient_tolerance
    assert fit.audit().status is FitAuditStatus.PASS

    stopped_early = wrapper(em_iterations=2, n_restarts=1).fit(study)

    assert stopped_early.diagnostics.converged is False
    assert stopped_early.diagnostics.gradient_norm > wrapper().gradient_tolerance
    assert stopped_early.audit().status is FitAuditStatus.FAIL


def test_the_fit_retains_the_evidence_an_em_run_of_a_latent_state_model_produces(
    study: Study, fit: DynamaxFitResult
) -> None:
    assert isinstance(fit, DynamaxFitResult)
    assert fit.n_observations == len(study)
    assert fit.n_sequences == 4
    assert fit.restart_objectives.size == 2
    assert 0 <= fit.selected_restart < 2
    assert fit.restart_objectives[fit.selected_restart] == pytest.approx(
        float(np.max(fit.restart_objectives))
    )
    assert fit.dynamax_version and fit.jax_version
    assert fit.derived_value("state_dwell_time") > 1.0
    assert 0.0 <= fit.grid_truncation < 1e-6
    assert abs(fit.grid_log_density_gap) < 0.1


# -- label switching -----------------------------------------------------------------------


def test_relabelling_the_states_changes_nothing_the_model_can_observe(
    study: Study, fit: DynamaxFitResult
) -> None:
    """The unidentifiability itself, measured: this is *why* the fit canonicalises.

    Permuting the states everywhere -- the initial distribution, both axes of the transition
    matrix, every emission parameter -- leaves the likelihood exactly where it was. An EM
    run therefore answers with whichever permutation its initialisation fell into, and
    nothing in the data can prefer one over the other.
    """

    model = wrapper()
    mirrored = model.parameters_from_components(
        initial=list(fit.estimates[[1, 0]]),
        transitions=[
            [fit.estimates[5], fit.estimates[4]],
            [fit.estimates[3], fit.estimates[2]],
        ],
        biases=list(fit.estimates[[7, 6]]),
        weights=[[fit.estimates[9]], [fit.estimates[8]]],
        variances=list(fit.estimates[[11, 10]]),
    )
    relabelled = replace(
        fit, estimates=np.asarray([mirrored[name] for name in fit.parameter_names])
    )

    assert not np.allclose(fit.estimates, relabelled.estimates)
    assert model.pointwise_log_prob(study, fit) == pytest.approx(
        model.pointwise_log_prob(study, relabelled), abs=1e-10
    )


def test_canonicalisation_sorts_the_states_and_permutes_both_axes_of_the_transitions() -> None:
    """A relabelling is not a row permutation: the transition matrix moves on both axes.

    Getting this half-right is the classic way to canonicalise an HMM into a different
    model, and it is silent -- the permuted matrix is still a valid transition matrix.
    """

    model = wrapper()
    backend = _Backend(model)
    descending = model.parameters_from_components(
        initial=[0.25, 0.75],
        transitions=[[0.8, 0.2], [0.4, 0.6]],
        biases=[2.0, -1.0],
        weights=[[0.1], [0.5]],
        variances=[0.3, 0.9],
    )
    parameters = backend.parameters(model._validated_components(descending))

    canonical, permutation = backend.canonical(parameters)

    assert permutation == (1, 0)
    assert canonical.biases == pytest.approx([-1.0, 2.0])
    assert canonical.initial == pytest.approx([0.75, 0.25])
    assert np.allclose(canonical.transitions, [[0.6, 0.4], [0.2, 0.8]])
    assert canonical.weights.ravel() == pytest.approx([0.5, 0.1])
    assert canonical.variances == pytest.approx([0.9, 0.3])


def test_the_fit_says_how_identified_its_canonical_order_is(fit: DynamaxFitResult) -> None:
    """Canonicalisation makes an order; it does not make the order mean anything."""

    assert sorted(fit.canonical_permutation) == [0, 1]
    assert fit.label_order_gap > wrapper().label_tolerance
    assert not fit.label_ambiguous
    assert not fit.low_occupancy
    assert float(np.sum(fit.state_occupancy)) == pytest.approx(1.0)


# -- prediction ----------------------------------------------------------------------------


def test_the_density_grid_is_a_function_of_the_fit_and_never_of_the_study(
    study: Study, fit: DynamaxFitResult
) -> None:
    """A grid derived from the rows being scored would leak the future into the past."""

    model = wrapper()
    elsewhere = Study(
        {
            **{name: study[name] for name in study.columns},
            "speed": np.asarray(study["speed"], dtype=np.float64) * 0.5,
        }
    )

    here = model.predict(study, fit)
    there = model.predict(elsewhere, fit)

    assert isinstance(here, DensityPrediction)
    assert np.array_equal(here.grid, fit.outcome_grid)
    assert np.array_equal(there.grid, fit.outcome_grid)
    assert here.outcome == "speed" and here.categories is None
    assert np.all(here.total_mass > 0.999)


def test_predict_and_predict_density_are_the_same_object(
    study: Study, fit: DynamaxFitResult
) -> None:
    model = wrapper()

    assert np.array_equal(
        model.predict(study, fit).density, model.predict_density(study, fit).density
    )


def test_a_fit_from_another_specification_is_refused(study: Study, fit: DynamaxFitResult) -> None:
    with pytest.raises(ValueError, match="different model specification"):
        wrapper(n_states=3).predict(study, fit)


def test_a_grid_the_rows_lie_outside_is_reported_rather_than_normalised_away(
    study: Study, fit: DynamaxFitResult
) -> None:
    """A held-out row far outside the training range has no density on the fitted grid."""

    model = wrapper()
    displaced = Study(
        {
            **{name: study[name] for name in study.columns},
            "speed": np.asarray(study["speed"], dtype=np.float64) + 500.0,
        }
    )

    with pytest.raises(ModelDataError, match="outside the range the training fold covered"):
        model.predict(displaced, fit)


# -- through the falsification stack --------------------------------------------------------


def test_the_wrapper_flows_through_prospective_folds_against_its_own_nested_null(
    study: Study,
) -> None:
    """The point of a wrapper: the foreign model reaches the machinery unchanged.

    The competitor is the nested one the ``num_lags`` field exists for -- a switching
    autoregression against the Gaussian-emission HMM it reduces to when every weight is
    zero -- which is a targeted competitor rather than an unrelated candidate, and is what
    ``AGENTS.md`` asks for before a latent state is interpreted.
    """

    splits = forward_session_splits(study)
    autoregressive, plain = _fold_models()

    richer = evaluate_splits(autoregressive, study, splits)
    simpler = evaluate_splits(plain, study, splits)

    assert richer.complete and simpler.complete
    assert len(richer.evaluations) == len(splits)
    for fold in richer.evaluations:
        assert np.isfinite(fold.total_log_probability)
        assert fold.prediction.n_observations == fold.split.test_indices.size
        assert isinstance(fold.prediction, DensityPrediction)
    assert sum(fold.total_log_probability for fold in richer.evaluations) > sum(
        fold.total_log_probability for fold in simpler.evaluations
    )


def test_a_declared_brier_column_is_refused_for_an_unlabelled_density(study: Study) -> None:
    """A finding about Behavio, not about dynamax, and still refused -- but earlier.

    The default comparison table carries a Brier column, and a Brier score is a scoring rule
    for a probability. PyDDM's density escapes this because it is *defective across the two
    boundaries*, so integrating the grid yields genuine choice probabilities. A switching
    autoregression predicts an unlabelled continuous density with no discrete margin at all,
    so there is no number to report. The wrapper declares as much through its capabilities,
    so the refusal arrives before any fold is fitted and names the candidate and the rule.
    """

    from behavio.compare.models import UnscoreableByBrier

    with pytest.raises(UnscoreableByBrier, match="no categorical margin") as refusal:
        compare_models(
            {"switching-ar": _fold_models()[0]},
            study,
            forward_session_splits(study),
            outcome_column="speed",
        )
    assert "'switching-ar'" in str(refusal.value) and "'brier'" in str(refusal.value)


def test_a_switching_autoregression_is_ranked_against_its_null_on_the_log_score(
    study: Study,
) -> None:
    """The third continuous-outcome family, in the comparison table SDR-0063 unlocked.

    A wrapped foreign model reaches the declared log-score table on exactly the same terms
    as a first-party family: it declares that its prediction carries no discrete margin, the
    caller declares the rule that does not need one, and the contest runs. The
    autoregression is ranked against its own lag-free null, which is a falsifiable claim
    about the same rows rather than a table with one row in it.

    Which of the two wins is deliberately not asserted. This table weights every session
    equally where the test above sums pooled log probability, and the winner is read among
    *audit-eligible* candidates only -- an EM fit that did not converge in a fold is
    excluded however low its score, which is the same rule that governs every other family.
    What is asserted is that the verdict is the table's own: the lowest equal-unit log loss
    among eligible candidates, with the contrast as the matched difference.
    """

    autoregressive, plain = _fold_models()

    report = compare_models(
        {"switching-ar": autoregressive, "switching-null": plain},
        study,
        forward_session_splits(study),
        outcome_column="speed",
        aggregation_column="session",
        bootstrap_resamples=64,
        metrics=(ScoreMetric.LOG_LOSS,),
    )

    assert report.ranked_by is ScoreMetric.LOG_LOSS
    assert report.to_dict()["declared_metrics"] == ["log-loss"]
    assert report.winner == min(
        report.eligible_model_order,
        key=lambda name: report.result_for(name).unit_balanced_log_loss,
    )
    contrast = report.comparison_for("switching-ar", "switching-null")
    assert contrast.metric is ScoreMetric.LOG_LOSS
    assert contrast.left_minus_right.estimate == pytest.approx(
        report.result_for("switching-ar").unit_balanced_log_loss
        - report.result_for("switching-null").unit_balanced_log_loss
    )


def test_parameter_recovery_runs_end_to_end(truth: dict[str, float]) -> None:
    """Simulator and fitter must mean the same thing by the same name, or recovery is theatre."""

    model = wrapper(em_iterations=200, n_restarts=1)

    report = run_parameter_recovery(model, design(), [truth], seed=5)

    assert report.model_name == "dynamax-switching-autoregression"
    assert report.model_signature == model.signature
    assert report.convergence_rate == 1.0
    summary = report.summary()
    biases = {row.parameter: row for row in summary if row.parameter.endswith(".bias")}
    assert set(biases) == {"state[0].bias", "state[1].bias"}
    for row in biases.values():
        assert np.isfinite(row.bias)


# -- validation and refusal ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_states": 1}, "at least two"),
        ({"num_lags": -1}, "non-negative"),
        ({"initialisation": "spectral"}, "initialisation must be one of"),
        ({"n_restarts": 0}, "positive integer"),
        ({"outcome": "subject"}, "identity columns"),
    ],
)
def test_an_inadmissible_configuration_is_refused_at_construction(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        wrapper(**overrides)


def test_describe_reports_the_study_specific_ways_an_em_fit_goes_wrong() -> None:
    flat = Study(
        {
            "subject": ["m1"] * 6,
            "session": ["d0"] * 6,
            "trial": list(range(6)),
            "session_order": [0] * 6,
            "speed": [1.0] * 6,
        }
    )

    description = wrapper().describe(flat)

    codes = {finding.code for finding in description.findings}
    assert "constant_outcome" in codes
    assert description.errors


def test_a_categorical_outcome_is_refused_before_a_sampler_starts() -> None:
    categorical = Study(
        {
            "subject": ["m1"] * 4,
            "session": ["d0"] * 4,
            "trial": [0, 1, 2, 3],
            "session_order": [0] * 4,
            "speed": ["left", "right", "left", "right"],
        }
    )

    description = wrapper().describe(categorical)

    assert {finding.code for finding in description.errors} == {"non_numeric_outcome"}


def test_a_short_sequence_is_warned_about_rather_than_silently_zero_padded() -> None:
    short = Study(
        {
            "subject": ["m1"] * 5,
            "session": ["d0", "d0", "d0", "d0", "d1"],
            "trial": [0, 1, 2, 3, 0],
            "session_order": [0, 0, 0, 0, 1],
            "speed": [0.1, 0.4, -0.2, 0.9, 0.3],
        }
    )

    codes = {finding.code for finding in wrapper(num_lags=1).describe(short).findings}

    assert "sequence_shorter_than_lag_order" in codes
    assert "few_trials_per_parameter" in codes


def test_two_fits_of_the_same_study_are_identical(study: Study) -> None:
    """EM is seeded, restarts are seeded, and nothing reads a global stream."""

    model = wrapper(em_iterations=40, n_restarts=1)

    assert np.array_equal(model.fit(study).estimates, model.fit(study).estimates)


def test_an_undeclared_parameter_set_is_refused_by_the_simulator(truth: dict[str, float]) -> None:
    broken = dict(truth)
    broken["transition[0->0]"] = 0.5

    with pytest.raises(ValueError, match="transition row"):
        wrapper().simulate(design(5), broken, seed=0)


# -- what a machine without the extra sees ------------------------------------------------


@pytest.fixture
def without_dynamax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import dynamax`` fail the way it does on a machine without the extra."""

    monkeypatch.setitem(sys.modules, "dynamax", None)


def test_the_module_imports_and_describes_itself_without_the_extra(
    without_dynamax: None,
) -> None:
    model = wrapper()

    assert isinstance(model, BehaviourEstimator)
    assert "backend=dynamax.em" in model.signature
    assert len(model.parameter_names) == 12
    assert not model.describe(_small_study()).errors


def test_every_numerical_entry_point_names_the_extra_it_needs(
    without_dynamax: None, truth: dict[str, float]
) -> None:
    model = wrapper()

    with pytest.raises(ForeignPackageUnavailableError) as fitting:
        model.fit(_small_study())
    with pytest.raises(ForeignPackageUnavailableError) as simulating:
        model.simulate(design(10), truth, seed=0)

    assert f"behavio[{DYNAMAX_EXTRA}]" in str(fitting.value)
    assert f"behavio[{DYNAMAX_EXTRA}]" in str(simulating.value)
    assert "jax<0.4.32" in str(fitting.value)


def test_an_unsupported_dynamax_series_is_refused_rather_than_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release can change EM's default initialisation, which is part of the fit."""

    module = ModuleType("dynamax")
    module.__version__ = "0.9.1"
    monkeypatch.setitem(sys.modules, "dynamax", module)

    with pytest.raises(ForeignPackageUnavailableError) as error:
        wrapper().fit(_small_study())

    assert "0.9.1" in str(error.value)
    assert DYNAMAX_SERIES in str(error.value)


def _fold_models() -> tuple[DynamaxSwitchingAutoregression, DynamaxSwitchingAutoregression]:
    """The autoregression and its nested null, at the smallest settings that still fit."""

    settings = {"em_iterations": 30, "n_restarts": 1}
    return wrapper(**settings), wrapper(num_lags=0, **settings)


def _small_study() -> Study:
    """Two short sessions with a varying outcome: enough to describe, too little to fit."""

    values = np.sin(np.arange(24, dtype=np.float64))
    return Study(
        {
            "subject": ["m1"] * 24,
            "session": ["d0"] * 12 + ["d1"] * 12,
            "trial": list(range(12)) * 2,
            "session_order": [0] * 12 + [1] * 12,
            "speed": values,
        }
    )


def _ragged_study(truth: dict[str, float]) -> Study:
    """Four sessions of unequal length, which is what a behavioural recording looks like."""

    model = wrapper()
    full = model.simulate(design(40), truth, seed=2)
    layout = sequence_layout(full, grouping=SequenceGrouping.SESSION)
    keep: list[int] = []
    for position, sequence in enumerate((*layout.sequences,)):
        keep.extend(int(index) for index in sequence.indices[: 40 - 7 * position])
    return full.take(np.asarray(sorted(keep), dtype=np.intp))
