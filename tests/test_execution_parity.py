"""One execution path: the two entry points must produce the same numbers.

``behavio.protocol.runner`` used to re-implement the fit-predict-score loop that
:func:`behavio.evaluate.folds.evaluate_splits` already performed, so the protocol path and the
interactive path were two programs computing the same quantities. They are now one, and
these tests hold that collapse in place from both directions:

* :func:`test_protocol_and_interactive_paths_agree_on_every_number` runs the *same* models
  over the *same* folds through ``run_protocol`` and through ``compare_models`` and
  requires exact agreement, so a future divergence in either stack fails here.
* :func:`test_collapse_reproduces_the_pre_collapse_numbers` pins the values the *previous,
  duplicated* implementation produced on this fixture. They were captured by running the
  identical fixture against commit ``f0e47b5`` -- the last commit before the collapse --
  and are asserted exactly rather than to a tolerance. Nothing in the collapse was allowed
  to move a number, and this is the record of that.

The fixture is deliberately not separable: the two candidates differ only in ridge
penalty, so their scores are close and the paired interval straddles zero. A degenerate
fixture with perfect prediction would have agreed trivially.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_compiler import capabilities
from test_protocol import example_protocol

from behavio.compare.models import (
    ComparisonMultiplicity,
    bootstrap_interval,
    compare_models,
)
from behavio.evaluate.folds import FoldFailurePolicy, evaluate_splits
from behavio.evaluate.splits import cohort_forward_session_splits
from behavio.models import BernoulliHistoryGLM, PredictionMode
from behavio.protocol.compiler import compile_execution_plan, materialize_protocol
from behavio.protocol.runner import run_protocol
from behavio.trials import Study

#: Every number below was produced by commit ``f0e47b5``, the last commit in which
#: ``runner._run_candidate_folds`` re-implemented the fold loop and ``runner`` carried its
#: own ``_bootstrap_interval``. They are pinned to eighteen significant figures because the
#: collapse was required to preserve them exactly, not approximately.
PRE_COLLAPSE_UNIT_LOG_LOSS = {
    "static": {"a": 0.6544592753873432, "b": 0.658366969973131},
    "smooth": {"a": 0.6521226777559658, "b": 0.6517905160125684},
}
PRE_COLLAPSE_PAIRED_INTERVAL = {
    "estimate": 0.004456525795970001,
    "lower": 0.0023365976313773906,
    "upper": 0.0065764539605626116,
    "confidence_level": 0.95,
}


def parity_study() -> Study:
    """A study whose two candidates genuinely disagree without either being hopeless."""

    generator = np.random.default_rng(20260729)
    rows: list[dict[str, object]] = []
    source_row = 0
    for subject in ("a", "b"):
        drift = 0.4 if subject == "a" else -0.3
        for session_order in range(4):
            for trial in range(12):
                stimulus = float(generator.normal())
                logit = 0.8 * stimulus + drift * session_order - 0.2
                choice = int(generator.random() < 1.0 / (1.0 + np.exp(-logit)))
                rows.append(
                    {
                        "subject": subject,
                        "session": f"{subject}-s{session_order}",
                        "trial": trial,
                        "session_order": session_order,
                        "choice": choice,
                        "stimulus": stimulus,
                        "species": "mouse",
                        "source_asset": "asset-01",
                        "source_row": source_row,
                    }
                )
                source_row += 1
    return Study.from_records(rows)


def parity_models() -> dict[str, BernoulliHistoryGLM]:
    """Exactly the two candidates the shared example protocol freezes."""

    return {
        candidate.name: BernoulliHistoryGLM(
            **{setting.name: setting.value for setting in candidate.hyperparameters}
        )
        for candidate in parity_protocol().candidates
    }


def parity_protocol(candidates=(), *, with_recovery=True, **comparison_overrides):
    """The frozen protocol both parity fixtures and the multiplicity fixture share.

    ``candidates`` and ``comparison_overrides`` exist so a caller needing a wider candidate
    grid or a different winner policy does not have to restate the cohort denominators,
    which are tied to :func:`parity_study`'s shape and would then have to be kept in sync
    in two files.
    """

    draft = example_protocol(candidates=candidates, with_recovery=with_recovery)
    return replace(
        draft,
        cohort=replace(
            draft.cohort,
            expected_subjects=2,
            expected_sessions=8,
            expected_observations=96,
        ),
        panel=replace(draft.panel, minimum_sessions=4),
        comparison=replace(
            draft.comparison,
            bootstrap_repetitions=2_000,
            seed=11,
            **comparison_overrides,
        ),
    ).freeze()


def parity_compiled(protocol=None, model_capabilities=None):
    protocol = parity_protocol() if protocol is None else protocol
    materialized = materialize_protocol(protocol, parity_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    return (
        compile_execution_plan(
            materialized,
            splits,
            capabilities=capabilities() if model_capabilities is None else model_capabilities,
        ),
        materialized.study,
        splits,
    )


def unit_log_losses(candidate) -> dict[str, float]:
    return {str(score.unit): score.log_loss for score in candidate.unit_scores}


def test_protocol_and_interactive_paths_agree_on_every_number() -> None:
    compiled, study, splits = parity_compiled()
    models = parity_models()

    run = run_protocol(compiled, models)
    report = compare_models(
        models,
        study,
        splits,
        aggregation_column="subject",
        outcome_column="choice",
        bootstrap_resamples=compiled.protocol.comparison.bootstrap_repetitions,
        bootstrap_seed=compiled.protocol.comparison.seed,
        confidence_level=compiled.protocol.comparison.interval_level,
    )

    for candidate in run.report.candidates:
        interactive = report.result_for(candidate.name)
        protocol_scores = unit_log_losses(candidate)
        interactive_scores = dict(
            zip(
                (str(unit) for unit in interactive.aggregation_units),
                (float(value) for value in interactive.unit_log_losses),
                strict=True,
            )
        )
        assert protocol_scores == interactive_scores
        assert candidate.score is not None
        assert candidate.score.unit_balanced_score == interactive.unit_balanced_log_loss
        assert candidate.score.pooled_score == interactive.pooled_log_loss
        assert candidate.calibration.brier_score == interactive.pooled_brier_score
        assert (
            candidate.score.unit_balanced_interval.to_dict()
            == interactive.unit_balanced_log_loss_interval.to_dict()
        )

    protocol_pair = run.report.paired_comparisons[0]
    interactive_pair = report.comparison_for(protocol_pair.left_model, protocol_pair.right_model)
    assert protocol_pair.left_minus_right.to_dict() == interactive_pair.left_minus_right.to_dict()
    assert (
        protocol_pair.bootstrap_probability_positive
        == interactive_pair.bootstrap_probability_positive
    )
    assert protocol_pair.two_sided_probability == interactive_pair.two_sided_probability
    assert protocol_pair.decisive == interactive_pair.decisive


def test_collapse_reproduces_the_pre_collapse_numbers() -> None:
    compiled, _, _ = parity_compiled()

    run = run_protocol(compiled, parity_models())

    for candidate in run.report.candidates:
        assert unit_log_losses(candidate) == PRE_COLLAPSE_UNIT_LOG_LOSS[candidate.name]
    assert run.report.paired_comparisons[0].left_minus_right.to_dict() == (
        PRE_COLLAPSE_PAIRED_INTERVAL
    )


def test_the_one_bootstrap_reproduces_the_runners_former_private_draws() -> None:
    """The collapse replaced ``rng.choice(values, ...)`` with shared index draws.

    ``Generator.choice`` with ``replace=True`` and no weights *is* ``integers`` under the
    hood, so the two produce the same resample from the same seed. That equivalence is the
    reason the collapse could be exact, and it is worth holding rather than assuming.
    """

    values = np.asarray([0.31, 0.72, 1.05, 0.44, 0.19, 0.88])
    generator = np.random.default_rng(4321)
    former = np.mean(generator.choice(values, size=(500, len(values)), replace=True), axis=1)
    tail = round((1 - 0.9) / 2, 15)

    interval = bootstrap_interval(values, resamples=500, seed=4321, confidence_level=0.9)

    assert interval.estimate == float(np.mean(values))
    assert interval.lower == float(np.quantile(former, tail))
    assert interval.upper == float(np.quantile(former, 1 - tail))


def test_retain_and_raise_are_the_same_loop_under_two_declarations() -> None:
    """The two former failure semantics are now one option, honoured in both directions."""

    _, study, splits = parity_compiled()
    model = parity_models()["static"]

    raising = evaluate_splits(model, study, splits, mode=PredictionMode.FILTERED)
    retaining = evaluate_splits(
        model,
        study,
        splits,
        mode=PredictionMode.FILTERED,
        on_failure=FoldFailurePolicy.RETAIN,
    )

    assert raising.policy is FoldFailurePolicy.RAISE
    assert raising.failures == ()
    assert raising.complete
    assert len(retaining) == len(raising)
    assert retaining.failures == ()
    for left, right in zip(raising, retaining, strict=True):
        assert np.array_equal(left.pointwise_log_probability, right.pointwise_log_probability)
        assert left.identifier == right.identifier

    with pytest.raises(RuntimeError, match="deliberate second-fold failure"):
        evaluate_splits(_BreaksOnTheSecondFold(model), study, splits, mode=PredictionMode.FILTERED)

    kept = evaluate_splits(
        _BreaksOnTheSecondFold(model),
        study,
        splits,
        mode=PredictionMode.FILTERED,
        on_failure=FoldFailurePolicy.RETAIN,
    )
    assert len(kept) == len(splits) - 1
    assert len(kept.failures) == 1
    assert kept.failures[0].exception_type == "RuntimeError"
    assert not kept.complete
    assert kept.policy is FoldFailurePolicy.RETAIN


def test_a_retained_evaluation_may_not_claim_the_raising_policy() -> None:
    _, study, splits = parity_compiled()
    kept = evaluate_splits(
        _BreaksOnTheSecondFold(parity_models()["static"]),
        study,
        splits,
        mode=PredictionMode.FILTERED,
        on_failure=FoldFailurePolicy.RETAIN,
    )

    with pytest.raises(ValueError, match="cannot retain fold failures"):
        replace(kept, policy=FoldFailurePolicy.RAISE)


def test_paired_comparisons_are_matched_by_construction() -> None:
    """A pair's difference must equal the difference of the two candidates' estimates."""

    compiled, _, _ = parity_compiled()

    run = run_protocol(compiled, parity_models())

    by_name = {candidate.name: candidate for candidate in run.report.candidates}
    for pair in run.report.paired_comparisons:
        left = by_name[pair.left_model].score
        right = by_name[pair.right_model].score
        assert left is not None and right is not None
        assert pair.left_minus_right.estimate == pytest.approx(
            left.unit_balanced_score - right.unit_balanced_score, abs=1e-15
        )
    assert run.report.ranking.family is not None
    assert run.report.ranking.family.n_comparisons == 1
    assert run.report.ranking.family.multiplicity is ComparisonMultiplicity.BENJAMINI_HOCHBERG


class _BreaksOnTheSecondFold:
    """Delegates to a real estimator but throws once, from the second fit onwards."""

    def __init__(self, inner: BernoulliHistoryGLM) -> None:
        self._inner = inner
        self._fits = 0

    model_name = property(lambda self: self._inner.model_name)
    signature = property(lambda self: self._inner.signature)
    scored_columns = property(lambda self: self._inner.scored_columns)
    required_task_columns = property(lambda self: self._inner.required_task_columns)
    supported_prediction_modes = property(lambda self: self._inner.supported_prediction_modes)

    def fit(self, study):
        self._fits += 1
        if self._fits == 2:
            raise RuntimeError("deliberate second-fold failure")
        return self._inner.fit(study)

    def predict(self, study, fit, *, mode=PredictionMode.FILTERED):
        return self._inner.predict(study, fit, mode=mode)

    def pointwise_log_prob(self, study, fit, *, mode=PredictionMode.FILTERED):
        return self._inner.pointwise_log_prob(study, fit, mode=mode)
