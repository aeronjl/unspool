"""Tests for common execution under an audited protocol plan."""

from dataclasses import replace

import pytest
from test_compiler import (
    capabilities,
    frozen_nested_protocol,
    frozen_small_protocol,
    source_study,
)

from behavio.compiler import compile_execution_plan, materialize_protocol
from behavio.models import BernoulliHistoryGLM
from behavio.protocol import ProtocolState, ScoreMetric, WinnerPolicy
from behavio.runner import (
    ProtocolRunError,
    RankingStatus,
    RunStage,
    run_nested_protocol,
    run_protocol,
)
from behavio.validation import cohort_forward_session_splits


def compiled_small_protocol():
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    return compile_execution_plan(materialized, splits, capabilities=capabilities())


def candidate_models():
    return {
        "static": BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=0.1),
        "smooth": BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=1.0),
    }


def test_runner_retains_fits_predictions_scores_calibration_and_comparisons() -> None:
    run = run_protocol(compiled_small_protocol(), candidate_models())

    assert run.protocol.state == ProtocolState.EVALUATED
    assert run.protocol.lifecycle[-1].artifact_fingerprint == run.report.fingerprint
    assert len(run.report.candidates) == 2
    assert len(run.report.paired_comparisons) == 1
    for candidate in run.report.candidates:
        assert candidate.eligible
        assert len(candidate.folds) == 1
        assert len(candidate.folds[0].predictions) == 4
        assert len(candidate.unit_scores) == 2
        assert candidate.pooled_log_loss is not None
        assert candidate.unit_balanced_log_loss_interval is not None
        assert candidate.calibration.available
        assert candidate.calibration.n_observations == 4
        assert len(candidate.folds[0].audit.issues) >= 0
    serialized = run.report.canonical_json()
    assert '"predictions"' in serialized
    assert '"covariance"' not in serialized
    assert "NaN" not in serialized


def test_declared_brier_score_controls_summaries_comparisons_and_ranking() -> None:
    protocol = frozen_small_protocol()
    protocol = replace(
        protocol,
        comparison=replace(
            protocol.comparison,
            metric=ScoreMetric.BRIER,
            winner_policy=WinnerPolicy.LOWEST_POINT_ESTIMATE,
        ),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())

    run = run_protocol(compiled, candidate_models())

    for candidate in run.report.candidates:
        assert candidate.score is not None
        assert candidate.score.metric == ScoreMetric.BRIER
        assert candidate.unit_balanced_brier_score is not None
        assert candidate.pooled_brier_score is not None
        assert candidate.unit_balanced_brier_interval is not None
        assert candidate.unit_balanced_log_loss is None
        assert candidate.pooled_log_loss is None
        assert candidate.unit_balanced_log_loss_interval is None
    left, right = run.report.candidates
    comparison = run.report.paired_comparisons[0]
    expected_difference = sum(
        left_score.brier_score - right_score.brier_score
        for left_score, right_score in zip(left.unit_scores, right.unit_scores, strict=True)
        if left_score.brier_score is not None and right_score.brier_score is not None
    ) / len(left.unit_scores)
    assert comparison.metric == ScoreMetric.BRIER
    assert comparison.left_minus_right.estimate == pytest.approx(expected_difference)
    assert (
        run.report.ranking.winner
        == min(
            run.report.candidates,
            key=lambda candidate: candidate.score.unit_balanced_score,  # type: ignore[union-attr]
        ).name
    )


def test_identical_candidates_retain_an_unresolved_ranking() -> None:
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=0.5)
    run = run_protocol(
        compiled_small_protocol(),
        {"static": model, "smooth": model},
    )

    comparison = run.report.paired_comparisons[0]
    assert comparison.left_minus_right.estimate == pytest.approx(0.0)
    assert comparison.left_minus_right.lower == pytest.approx(0.0)
    assert comparison.left_minus_right.upper == pytest.approx(0.0)
    assert run.report.ranking.status == RankingStatus.UNRESOLVED
    assert run.report.ranking.winner is None


class FailingFitModel:
    model_name = "deliberate-failure"
    signature = "deliberate-failure[v1]"

    def fit(self, study):
        raise RuntimeError(f"deliberate failure for {len(study)} rows")


def test_candidate_failure_is_retained_while_other_candidate_completes() -> None:
    run = run_protocol(
        compiled_small_protocol(),
        {
            "static": FailingFitModel(),
            "smooth": candidate_models()["smooth"],
        },
    )

    failed, completed = run.report.candidates
    assert not failed.eligible
    assert failed.folds == ()
    assert len(failed.failures) == 1
    assert failed.failures[0].stage == RunStage.FIT
    assert failed.failures[0].exception_type == "RuntimeError"
    assert "deliberate failure" in failed.failures[0].message
    assert completed.eligible
    assert run.report.ranking.status == RankingStatus.RESOLVED
    assert run.report.ranking.winner == "smooth"


def test_runner_rejects_registry_drift_from_frozen_candidates() -> None:
    with pytest.raises(ProtocolRunError, match="exactly match"):
        run_protocol(
            compiled_small_protocol(),
            {"static": candidate_models()["static"]},
        )


def test_runner_refuses_a_failed_pre_fit_audit() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    failed = compile_execution_plan(materialized, splits, capabilities={})

    with pytest.raises(ProtocolRunError, match="audited execution plan"):
        run_protocol(failed, candidate_models())


def compiled_nested(study=None):
    materialized = materialize_protocol(frozen_nested_protocol(), study or source_study())
    outer = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    return compile_execution_plan(
        materialized,
        outer,
        capabilities=capabilities(),
        inner_splitter=lambda training, _fold: cohort_forward_session_splits(
            training,
            min_train_sessions=1,
        ),
    )


def test_nested_runner_selects_inside_training_then_scores_untouched_outer_rows() -> None:
    run = run_nested_protocol(compiled_nested(), candidate_models())

    assert run.protocol.state == ProtocolState.EVALUATED
    assert run.protocol.lifecycle[-1].artifact_fingerprint == run.report.fingerprint
    assert run.report.eligible
    assert len(run.report.folds) == 1
    fold = run.report.folds[0]
    assert fold.selected_candidate in {"static", "smooth"}
    assert tuple(candidate.name for candidate in fold.inner_candidates) == ("static", "smooth")
    assert all(candidate.eligible for candidate in fold.inner_candidates)
    assert fold.outer_result is not None
    assert len(fold.outer_result.folds[0].predictions) == 4
    assert len(run.report.unit_scores) == 2
    assert sum(run.report.selection_counts.values()) == 1


def test_nested_selection_uses_its_declared_brier_score() -> None:
    protocol = frozen_nested_protocol()
    protocol = replace(
        protocol,
        selection=replace(protocol.selection, metric=ScoreMetric.BRIER),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    outer = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(
        materialized,
        outer,
        capabilities=capabilities(),
        inner_splitter=lambda training, _fold: cohort_forward_session_splits(
            training,
            min_train_sessions=1,
        ),
    )

    run = run_nested_protocol(compiled, candidate_models())

    fold = run.report.folds[0]
    assert all(
        candidate.score is not None and candidate.score.metric == ScoreMetric.BRIER
        for candidate in fold.inner_candidates
    )
    assert (
        fold.selected_candidate
        == min(
            fold.inner_candidates,
            key=lambda candidate: candidate.score.unit_balanced_score,  # type: ignore[union-attr]
        ).name
    )
    assert run.report.score is not None
    assert run.report.score.metric == ScoreMetric.LOG_LOSS


def test_nested_selection_and_outer_comparison_use_their_own_aggregation_units() -> None:
    protocol = frozen_nested_protocol()
    protocol = replace(
        protocol,
        selection=replace(protocol.selection, aggregation_unit="session"),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    outer = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(
        materialized,
        outer,
        capabilities=capabilities(),
        inner_splitter=lambda training, _fold: cohort_forward_session_splits(
            training,
            min_train_sessions=1,
        ),
    )

    run = run_nested_protocol(compiled, candidate_models())

    fold = run.report.folds[0]
    assert all(len(candidate.unit_scores) == 2 for candidate in fold.inner_candidates)
    assert {
        score.unit for candidate in fold.inner_candidates for score in candidate.unit_scores
    } == {
        "a-s1",
        "b-s1",
    }
    assert {score.unit for score in run.report.unit_scores} == {"a", "b"}


def test_outer_outcomes_cannot_change_inner_selection() -> None:
    baseline = run_nested_protocol(compiled_nested(), candidate_models())
    source = source_study()
    columns = {name: source[name].copy() for name in source.columns}
    outer_rows = columns["session_order"] == 2
    columns["choice"][outer_rows] = 1 - columns["choice"][outer_rows]
    changed = run_nested_protocol(
        compiled_nested(type(source).from_columns(columns)),
        candidate_models(),
    )

    assert changed.report.selected_candidates == baseline.report.selected_candidates
    assert changed.report.unit_balanced_log_loss != baseline.report.unit_balanced_log_loss


def test_flat_runner_refuses_nested_selection_protocol() -> None:
    with pytest.raises(ProtocolRunError, match="must use run_nested_protocol"):
        run_protocol(compiled_nested(), candidate_models())
