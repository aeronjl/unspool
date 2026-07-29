"""Tests for recovery through the identical compiled study design."""

from dataclasses import replace

import numpy as np
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
    _replicate_seed_plan,
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


def colliding_protocol():
    """Two requirements whose old arithmetic seed ranges overlapped on seeds 5-9."""

    protocol = example_protocol()
    first = replace(
        protocol.recovery[0],
        name="first-recovery",
        success_metric="custom-rate",
        threshold=0.0,
        repetitions=10,
        seed=0,
    )
    second = replace(first, name="second-recovery", seed=5)
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
        recovery=(first, second),
    ).freeze()


def test_requirements_with_overlapping_declared_seed_ranges_stay_independent() -> None:
    """Two gates that shared replicate seeds now draw genuinely different simulated data."""

    protocol = colliding_protocol()
    # The second requirement's case comes first, which is what made the old arithmetic
    # `requirement.seed + case_index * requirement.repetitions + repetition` overlap.
    cases = (
        replace(recovery_case(), name="second-case", requirement="second-recovery"),
        replace(recovery_case(), name="first-case", requirement="first-recovery"),
    )
    arithmetic = {
        "second-recovery": [5 + 0 * 10 + repetition for repetition in range(10)],
        "first-recovery": [0 + 1 * 10 + repetition for repetition in range(10)],
    }
    assert set(arithmetic["second-recovery"]) & set(arithmetic["first-recovery"]) == {
        10,
        11,
        12,
        13,
        14,
    }

    observed = evaluated(protocol)
    result = run_exact_recovery(
        observed,
        generators={"static": models()["static"]},
        candidates=models(),
        cases=cases,
        assessments={
            "first-recovery": lambda _study, _run, _case: 1.0,
            "second-recovery": lambda _study, _run, _case: 1.0,
        },
    )

    replicates = result.report.replicates
    seeds = {
        name: [item.seed for item in replicates if item.requirement == name]
        for name in ("first-recovery", "second-recovery")
    }

    assert len(replicates) == 20
    assert all(len(values) == 10 for values in seeds.values())
    assert not set(seeds["first-recovery"]) & set(seeds["second-recovery"])
    assert len(set(seeds["first-recovery"] + seeds["second-recovery"])) == 20
    assert len({gate.requirement for gate in result.report.gates}) == 2

    # The recovery report deliberately withholds simulated observations, so re-simulate
    # with the recorded seeds: paired replicates must now differ trial by trial instead of
    # repeating five perfectly correlated draws under two separate gates.
    generator = models()["static"]
    design = observed.compiled.materialized.study
    parameters = recovery_case().parameter_map
    outcomes = {
        name: [generator.simulate(design, parameters, seed=seed)["choice"] for seed in values]
        for name, values in seeds.items()
    }
    assert all(
        not np.array_equal(left, right)
        for left, right in zip(outcomes["first-recovery"], outcomes["second-recovery"], strict=True)
    )
    assert all(
        not np.array_equal(left, right)
        for left, right in zip(
            outcomes["first-recovery"], outcomes["first-recovery"][1:], strict=False
        )
    )


def test_replicate_seeds_are_disjoint_even_for_identically_declared_requirements() -> None:
    """Two requirements that declare the same seed still get separate streams."""

    protocol = colliding_protocol()
    first, second = protocol.recovery
    shared = replace(second, seed=first.seed)
    cases = (
        RecoveryCase(
            name="first-case",
            requirement=first.name,
            generating_candidate="static",
            parameters=(Setting("intercept", -0.2),),
        ),
        RecoveryCase(
            name="second-case",
            requirement=shared.name,
            generating_candidate="static",
            parameters=(Setting("intercept", -0.2),),
        ),
    )

    plan = _replicate_seed_plan(cases, {first.name: first, shared.name: shared})

    assert len(plan) == 2
    assert all(len(seeds) == first.repetitions for seeds in plan)
    assert not set(plan[0]) & set(plan[1])
    assert _replicate_seed_plan(cases, {first.name: first, shared.name: shared}) == plan
