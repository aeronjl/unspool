"""Tests for recovery through the identical compiled study design."""

from dataclasses import replace

import pytest
from test_compiler import capabilities, source_study
from test_protocol import example_protocol

from behavio.compiler import compile_execution_plan, materialize_protocol
from behavio.models import BernoulliHistoryGLM
from behavio.protocol import ProtocolState, RecoveryKind, Setting, WinnerPolicy
from behavio.protocol_recovery import (
    ExactRecoveryError,
    GateStatus,
    RecoveryCase,
    run_exact_recovery,
)
from behavio.runner import run_protocol
from behavio.validation import cohort_forward_session_splits


def recovery_protocol(*, kind=RecoveryKind.MODEL, metric="selection-rate", threshold=0.0):
    protocol = example_protocol()
    return replace(
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
            winner_policy=WinnerPolicy.LOWEST_POINT_ESTIMATE,
            bootstrap_repetitions=50,
        ),
        recovery=(
            replace(
                protocol.recovery[0],
                kind=kind,
                success_metric=metric,
                threshold=threshold,
                repetitions=2,
            ),
        ),
    ).freeze()


def models():
    return {
        "static": BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=0.1),
        "smooth": BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=1.0),
    }


def evaluated(protocol=None):
    materialized = materialize_protocol(protocol or recovery_protocol(), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())
    return run_protocol(compiled, models())


def recovery_case():
    return RecoveryCase(
        name="static-world",
        requirement="candidate-recovery",
        generating_candidate="static",
        expected_winner="static",
        parameters=(Setting("intercept", -0.2), Setting("stimulus", 1.1)),
    )


def test_exact_recovery_reuses_plan_and_advances_lifecycle() -> None:
    observed = evaluated()
    result = run_exact_recovery(
        observed,
        generators={"static": models()["static"]},
        candidates=models(),
        cases=(recovery_case(),),
    )

    assert result.protocol.state == ProtocolState.RECOVERED
    assert result.protocol.lifecycle[-1].artifact_fingerprint == result.report.fingerprint
    assert len(result.report.replicates) == 2
    assert result.report.failures == ()
    assert all(
        replicate.execution_plan_fingerprint == observed.compiled.plan.fingerprint
        for replicate in result.report.replicates
    )
    gate = result.report.gates[0]
    assert gate.status == GateStatus.PASSED
    assert gate.requested_repetitions == 2
    assert gate.effective_repetitions == 2
    assert result.report.blocked_claims == ()
    assert "choice" not in result.report.canonical_json()


def test_failed_gate_automatically_retains_constrained_claim() -> None:
    protocol = recovery_protocol(metric="custom-rate", threshold=0.8)
    result = run_exact_recovery(
        evaluated(protocol),
        generators={"static": models()["static"]},
        candidates=models(),
        cases=(recovery_case(),),
        assessments={"candidate-recovery": lambda _study, _run, _case: 0.0},
    )

    assert result.report.gates[0].status == GateStatus.FAILED
    assert result.report.gates[0].observed == 0.0
    assert result.report.blocked_claims == ("mechanistic identification",)


def test_parameter_recovery_has_a_built_in_rmse_assessment() -> None:
    protocol = recovery_protocol(
        kind=RecoveryKind.PARAMETER,
        metric="parameter-rmse",
        threshold=100.0,
    )
    result = run_exact_recovery(
        evaluated(protocol),
        generators={"static": models()["static"]},
        candidates=models(),
        cases=(recovery_case(),),
    )

    assert result.report.gates[0].direction == "at-most"
    assert result.report.gates[0].status == GateStatus.PASSED
    assert result.report.gates[0].observed is not None


def test_outcome_feature_recovery_requires_training_only_recompilation() -> None:
    protocol = recovery_protocol(
        kind=RecoveryKind.OUTCOME_DERIVED_FEATURE,
        metric="custom-rate",
        threshold=0.8,
    )
    with pytest.raises(ExactRecoveryError, match="transform recompiler"):
        run_exact_recovery(
            evaluated(protocol),
            generators={"static": models()["static"]},
            candidates=models(),
            cases=(recovery_case(),),
            assessments={"candidate-recovery": lambda _study, _run, _case: 1.0},
        )


def test_recovery_cases_must_cover_every_frozen_requirement() -> None:
    with pytest.raises(ExactRecoveryError, match="must cover every"):
        run_exact_recovery(
            evaluated(),
            generators={"static": models()["static"]},
            candidates=models(),
            cases=(replace(recovery_case(), requirement="unknown"),),
        )
