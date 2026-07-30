"""Tests for deterministic, claim-bounded protocol reports."""

from dataclasses import replace

import pytest
from test_compiler import capabilities, source_study
from test_exact_recovery import evaluated, models, recovery_case, recovery_protocol
from test_protocol import example_protocol
from test_runner import candidate_models, compiled_nested, declared_glm

from behavio.evaluate.splits import cohort_forward_session_splits
from behavio.protocol.compiler import compile_execution_plan, materialize_protocol
from behavio.protocol.exact_recovery import run_exact_recovery
from behavio.protocol.runner import run_nested_protocol, run_protocol
from behavio.protocol.schema import CandidateSpec, ProtocolState, ScoreMetric, Setting
from behavio.report.bounded import (
    ReportGenerationError,
    ReportItem,
    ReportItemKind,
    generate_bounded_report,
)


def report_items() -> tuple[ReportItem, ...]:
    return (
        ReportItem("denominators", ReportItemKind.TABLE, "Exact included cohort counts."),
        ReportItem("fold-scores", ReportItemKind.TABLE, "Scores for every prospective fold."),
        ReportItem("fit-audits", ReportItemKind.TABLE, "Normalized numerical findings."),
        ReportItem(
            "paired-score-contrast",
            ReportItemKind.FIGURE,
            "Animal-paired predictive contrast with its uncertainty interval.",
        ),
        ReportItem(
            "optimization",
            ReportItemKind.DIAGNOSTIC,
            "Convergence and boundary findings remain visible.",
        ),
        ReportItem(
            "calibration",
            ReportItemKind.DIAGNOSTIC,
            "Forecast probabilities are compared with observed frequencies.",
        ),
    )


def recovered(*, failing_gate: bool = False):
    if failing_gate:
        protocol = recovery_protocol(metric="custom-rate", threshold=0.8)
        return run_exact_recovery(
            evaluated(protocol),
            generators={"static": models()["static"]},
            candidates=models(),
            cases=(recovery_case(),),
            assessments={"candidate-recovery": lambda _study, _run, _case: 0.0},
        )
    return run_exact_recovery(
        evaluated(),
        generators={"static": models()["static"]},
        candidates=models(),
        cases=(recovery_case(),),
    )


def evaluated_without_recovery():
    protocol = example_protocol(with_recovery=False)
    protocol = replace(
        protocol,
        cohort=replace(
            protocol.cohort,
            expected_subjects=2,
            expected_sessions=6,
            expected_observations=12,
        ),
        panel=replace(protocol.panel, minimum_sessions=3),
        comparison=replace(protocol.comparison, bootstrap_repetitions=50),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())
    return run_protocol(compiled, models())


def test_report_is_deterministic_complete_and_advances_lifecycle() -> None:
    evidence = recovered()
    result = generate_bounded_report(evidence, items=report_items())

    assert result.protocol.state == ProtocolState.REPORTED
    assert result.protocol.lifecycle[-1].artifact_fingerprint == result.report.fingerprint
    assert result.report.recovery_fingerprint == evidence.report.fingerprint
    assert result.report.markdown.startswith("# Learning trajectory forecast\n")
    assert "## Frozen design" in result.report.markdown
    assert "## Candidate performance" in result.report.markdown
    assert "## Exact-design recovery" in result.report.markdown
    assert "## Limitations" in result.report.markdown
    assert "## Claim boundaries" in result.report.markdown
    assert "mechanistic identification" in result.report.markdown
    assert result.report.canonical_json() == result.report.canonical_json()


def test_failed_recovery_gate_is_rendered_as_a_blocked_claim() -> None:
    result = generate_bounded_report(recovered(failing_gate=True), items=report_items())

    assert result.report.blocked_claims == ("mechanistic identification",)
    assert "Failed or insufficient recovery gates additionally block:" in result.report.markdown
    assert "| candidate-recovery | model | custom-rate | 0.000000 | ≥ 0.8 | failed |" in (
        result.report.markdown
    )


def test_required_recovery_cannot_be_bypassed() -> None:
    with pytest.raises(ReportGenerationError, match="requires recovery"):
        generate_bounded_report(evaluated(), items=report_items())


def test_protocol_without_recovery_reports_after_evaluation() -> None:
    result = generate_bounded_report(evaluated_without_recovery(), items=report_items())

    assert result.protocol.state == ProtocolState.REPORTED
    assert result.recovery is None
    assert "did not require a recovery stage" in result.report.markdown


def test_missing_required_report_artifact_is_rejected() -> None:
    with pytest.raises(ReportGenerationError, match="required report evidence is missing"):
        generate_bounded_report(recovered(), items=report_items()[:-1])


def test_nested_selection_report_never_ranks_candidates_on_outer_outcomes() -> None:
    nested = run_nested_protocol(compiled_nested(), candidate_models())
    result = generate_bounded_report(nested, items=report_items())

    assert result.protocol.state == ProtocolState.REPORTED
    assert "## Nested selection performance" in result.report.markdown
    assert "### Training-only selections" in result.report.markdown
    assert "No candidate ranking was computed from outer-test outcomes." in result.report.markdown


def test_report_labels_the_declared_brier_score() -> None:
    protocol = example_protocol(with_recovery=False)
    protocol = replace(
        protocol,
        cohort=replace(
            protocol.cohort,
            expected_subjects=2,
            expected_sessions=6,
            expected_observations=12,
        ),
        panel=replace(protocol.panel, minimum_sessions=3),
        comparison=replace(
            protocol.comparison,
            metric=ScoreMetric.BRIER,
            bootstrap_repetitions=50,
        ),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())
    evaluated = run_protocol(compiled, models())

    result = generate_bounded_report(evaluated, items=report_items())

    assert "Unit-balanced brier" in result.report.markdown
    assert "Unit-balanced log loss" not in result.report.markdown


def evaluated_with_an_unverifiable_declaration():
    protocol = example_protocol(
        with_recovery=False,
        candidates=(
            declared_glm("static", predictors=("stimulus",), choice_lags=0, l2=0.1),
            CandidateSpec(
                name="smooth",
                implementation="behavio.models.BernoulliHistoryGLM",
                hyperparameters=(Setting("l2", 1.0), Setting("optimizer", "irls")),
                scored_columns=("choice",),
            ),
        ),
    )
    protocol = replace(
        protocol,
        cohort=replace(
            protocol.cohort,
            expected_subjects=2,
            expected_sessions=6,
            expected_observations=12,
        ),
        panel=replace(protocol.panel, minimum_sessions=3),
        comparison=replace(protocol.comparison, bootstrap_repetitions=50),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())
    return run_protocol(compiled, candidate_models())


def test_a_fully_verified_study_reports_no_declaration_section() -> None:
    result = generate_bounded_report(evaluated_without_recovery(), items=report_items())

    assert "Unverified declarations" not in result.report.markdown


def test_a_declaration_that_could_not_be_checked_is_disclosed_in_the_report() -> None:
    result = generate_bounded_report(
        evaluated_with_an_unverifiable_declaration(), items=report_items()
    )

    assert "### Unverified declarations" in result.report.markdown
    assert "| smooth | hyperparameter:optimizer |" in result.report.markdown
    assert "no field of that name" in result.report.markdown
    assert "recorded rather than treated as satisfied" in result.report.markdown
