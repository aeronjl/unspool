"""Exact-design recovery executed through the identical compiled protocol plan."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

from behavio.models import BehaviourEstimator, GenerativeBehaviourModel
from behavio.protocol.compiler import CompiledProtocol
from behavio.protocol.runner import ProtocolRun, run_protocol
from behavio.protocol.schema import (
    ProtocolState,
    ProtocolValidationError,
    RecoveryKind,
    RecoverySpec,
    Setting,
)
from behavio.trials import Study

RecoveryAssessment = Callable[[Study, ProtocolRun, "RecoveryCase"], float]
TransformRecompiler = Callable[[Study, CompiledProtocol], CompiledProtocol]


class ExactRecoveryError(RuntimeError):
    """Raised when declared recovery cannot use the exact audited design."""


class GateStatus(StrEnum):
    """Outcome of one predeclared recovery requirement."""

    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT = "insufficient-evidence"


@dataclass(frozen=True, slots=True)
class RecoveryCase:
    """One generative truth linked to a frozen recovery requirement."""

    name: str
    requirement: str
    generating_candidate: str
    parameters: tuple[Setting, ...]
    expected_winner: str | None = None
    generator_key: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.name, "recovery case name"),
            (self.requirement, "recovery requirement"),
            (self.generating_candidate, "generating candidate"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if self.expected_winner is not None and not self.expected_winner:
            raise ValueError("expected_winner must be null or non-empty")
        if self.generator_key is not None and not self.generator_key:
            raise ValueError("generator_key must be null or non-empty")
        names = [setting.name for setting in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("recovery parameters must be uniquely named")
        if any(
            isinstance(setting.value, (bool, tuple))
            or not isinstance(setting.value, (int, float))
            or not math.isfinite(float(setting.value))
            for setting in self.parameters
        ):
            raise ValueError("recovery parameters must be finite numeric scalars")

    @property
    def parameter_map(self) -> Mapping[str, float]:
        """Natural-scale parameters passed to the declared generator."""

        return {setting.name: float(setting.value) for setting in self.parameters}


@dataclass(frozen=True, slots=True)
class RecoveryFailure:
    """Retained simulation, recompilation, execution, or assessment failure."""

    requirement: str
    case: str
    repetition: int
    seed: int
    stage: str
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RecoveryReplicate:
    """One exact-plan simulated evaluation and scalar gate contribution."""

    requirement: str
    case: str
    repetition: int
    seed: int
    simulated_study_fingerprint: str
    execution_plan_fingerprint: str
    ranking_status: str
    winner: str | None
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("recovery replicate score must be finite")


@dataclass(frozen=True, slots=True)
class RecoveryGate:
    """Aggregated recovery evidence that automatically blocks failed claims."""

    requirement: str
    kind: RecoveryKind
    metric: str
    threshold: float
    direction: str
    observed: float | None
    requested_repetitions: int
    effective_repetitions: int
    status: GateStatus
    constrains_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RecoveryKind(self.kind))
        object.__setattr__(self, "status", GateStatus(self.status))
        if self.observed is not None and not math.isfinite(self.observed):
            raise ValueError("recovery gate observation must be finite or null")
        if self.status == GateStatus.INSUFFICIENT and self.observed is not None:
            raise ValueError("insufficient recovery gates cannot declare an observation")


@dataclass(frozen=True, slots=True)
class ExactRecoveryReport:
    """Portable recovery evidence tied to one protocol and one execution plan."""

    schema_version: str
    protocol_fingerprint: str
    execution_plan_fingerprint: str
    replicates: tuple[RecoveryReplicate, ...]
    failures: tuple[RecoveryFailure, ...]
    gates: tuple[RecoveryGate, ...]

    @property
    def fingerprint(self) -> str:
        """Content address of all exact-design recovery evidence."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def blocked_claims(self) -> tuple[str, ...]:
        """Claims that remain prohibited because their gate did not pass."""

        return tuple(
            dict.fromkeys(
                claim
                for gate in self.gates
                if gate.status != GateStatus.PASSED
                for claim in gate.constrains_claims
            )
        )

    @property
    def passed_requirements(self) -> tuple[str, ...]:
        """Requirements whose predeclared gate passed."""

        return tuple(gate.requirement for gate in self.gates if gate.status == GateStatus.PASSED)

    def to_dict(self) -> dict[str, Any]:
        """Return standards-compliant evidence without simulated raw observations."""

        return {
            "schema_version": self.schema_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "replicates": [_json_safe(asdict(item)) for item in self.replicates],
            "failures": [_json_safe(asdict(item)) for item in self.failures],
            "gates": [_json_safe(asdict(item)) for item in self.gates],
            "passed_requirements": list(self.passed_requirements),
            "blocked_claims": list(self.blocked_claims),
        }

    def canonical_json(self) -> str:
        """Serialize recovery evidence deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ExactRecoveryRun:
    """Recovered lifecycle state paired with its exact-design evidence."""

    protocol: Any
    evaluation: ProtocolRun
    report: ExactRecoveryReport

    def __post_init__(self) -> None:
        if self.protocol.state != ProtocolState.RECOVERED:
            raise ProtocolValidationError("exact recovery requires recovered lifecycle state")
        if self.protocol.fingerprint != self.report.protocol_fingerprint:
            raise ProtocolValidationError("recovery report and protocol fingerprints differ")


def run_exact_recovery(
    evaluation: ProtocolRun,
    *,
    generators: Mapping[str, GenerativeBehaviourModel],
    candidates: Mapping[str, BehaviourEstimator],
    cases: Sequence[RecoveryCase],
    assessments: Mapping[str, RecoveryAssessment] | None = None,
    transform_recompiler: TransformRecompiler | None = None,
) -> ExactRecoveryRun:
    """Simulate and refit every recovery case through the identical compiled row plan.

    Learned transforms can be recomputed by supplying ``transform_recompiler``.  It must
    preserve the protocol identity and exact fit/context/score/exclusion geometry.  An
    outcome-derived transform requirement is rejected when no such recompiler is supplied.
    """

    protocol = evaluation.protocol
    if protocol.state != ProtocolState.EVALUATED:
        raise ExactRecoveryError("recovery must follow the observed-data evaluation")
    if not protocol.recovery:
        raise ExactRecoveryError("the frozen protocol declares no recovery requirements")
    assessments = assessments or {}
    requirement_by_name = {item.name: item for item in protocol.recovery}
    candidate_names = {candidate.name for candidate in protocol.candidates}
    cases = tuple(cases)
    if not cases:
        raise ExactRecoveryError("at least one recovery case is required")
    unknown_requirements = sorted({case.requirement for case in cases} - set(requirement_by_name))
    missing_requirements = sorted(set(requirement_by_name) - {case.requirement for case in cases})
    if unknown_requirements or missing_requirements:
        raise ExactRecoveryError(
            "recovery cases must cover every frozen requirement; "
            f"missing={missing_requirements}, unknown={unknown_requirements}"
        )
    if set(candidates) != candidate_names:
        raise ExactRecoveryError("recovery candidates must exactly match the frozen model set")
    for case in cases:
        generator_key = case.generator_key or case.generating_candidate
        if generator_key not in generators:
            raise ExactRecoveryError(f"case {case.name!r} has no generator {generator_key!r}")
        if case.expected_winner is not None and case.expected_winner not in candidate_names:
            raise ExactRecoveryError(f"case {case.name!r} expects an undeclared winner")
        requirement = requirement_by_name[case.requirement]
        eligible_recovery_candidates = set(requirement.candidate_names) or candidate_names
        if case.generating_candidate not in eligible_recovery_candidates:
            raise ExactRecoveryError(
                f"case {case.name!r} generator is outside its recovery candidate set"
            )
        if (
            case.expected_winner is not None
            and case.expected_winner not in eligible_recovery_candidates
        ):
            raise ExactRecoveryError(
                f"case {case.name!r} expected winner is outside its recovery candidate set"
            )
        if (
            requirement.kind == RecoveryKind.OUTCOME_DERIVED_FEATURE
            and transform_recompiler is None
        ):
            raise ExactRecoveryError(
                "outcome-derived-feature recovery requires a training-only transform recompiler"
            )
        _assessment_for(requirement, assessments)

    seed_plan = _replicate_seed_plan(cases, requirement_by_name)
    replicates: list[RecoveryReplicate] = []
    failures: list[RecoveryFailure] = []
    for case_index, case in enumerate(cases):
        requirement = requirement_by_name[case.requirement]
        assessment = _assessment_for(requirement, assessments)
        for repetition in range(requirement.repetitions):
            seed = seed_plan[case_index][repetition]
            try:
                generator_key = case.generator_key or case.generating_candidate
                simulated = generators[generator_key].simulate(
                    evaluation.compiled.materialized.study,
                    case.parameter_map,
                    seed=seed,
                )
                _validate_simulated_design(evaluation.compiled.materialized.study, simulated)
            except Exception as error:
                failures.append(
                    _recovery_failure(requirement, case, repetition, seed, "simulate", error)
                )
                continue
            try:
                simulated_compiled = _compiled_with_study(evaluation.compiled, simulated)
                if transform_recompiler is not None:
                    simulated_compiled = transform_recompiler(simulated, simulated_compiled)
                _validate_exact_plan(evaluation.compiled, simulated_compiled)
            except Exception as error:
                failures.append(
                    _recovery_failure(requirement, case, repetition, seed, "recompile", error)
                )
                continue
            try:
                simulated_run = run_protocol(simulated_compiled, candidates)
            except Exception as error:
                failures.append(
                    _recovery_failure(requirement, case, repetition, seed, "evaluate", error)
                )
                continue
            try:
                score = float(assessment(simulated, simulated_run, case))
                if not math.isfinite(score):
                    raise ValueError("recovery assessment returned a non-finite score")
            except Exception as error:
                failures.append(
                    _recovery_failure(requirement, case, repetition, seed, "assess", error)
                )
                continue
            replicates.append(
                RecoveryReplicate(
                    requirement=requirement.name,
                    case=case.name,
                    repetition=repetition,
                    seed=seed,
                    simulated_study_fingerprint=_study_design_fingerprint(simulated),
                    execution_plan_fingerprint=simulated_compiled.plan.fingerprint,
                    ranking_status=simulated_run.report.ranking.status.value,
                    winner=simulated_run.report.ranking.winner,
                    score=score,
                )
            )

    gates = tuple(
        _gate(
            requirement,
            [item.score for item in replicates if item.requirement == requirement.name],
            requested=sum(
                requirement.repetitions for case in cases if case.requirement == requirement.name
            ),
        )
        for requirement in protocol.recovery
    )
    report = ExactRecoveryReport(
        schema_version="behavio.exact-recovery/1",
        protocol_fingerprint=protocol.fingerprint,
        execution_plan_fingerprint=evaluation.compiled.plan.fingerprint,
        replicates=tuple(replicates),
        failures=tuple(failures),
        gates=gates,
    )
    recovered = protocol.advance(
        ProtocolState.RECOVERED,
        artifact_fingerprint=report.fingerprint,
    )
    return ExactRecoveryRun(recovered, evaluation, report)


def _replicate_seed_plan(
    cases: tuple[RecoveryCase, ...],
    requirement_by_name: Mapping[str, RecoverySpec],
) -> tuple[tuple[int, ...], ...]:
    """Spawn one disjoint replicate seed stream per case, indexed by case position.

    Arithmetic seeds (``requirement.seed + case_index * repetitions + repetition``) let two
    requirements with different declared seeds overlap: ``seed=0, repetitions=10`` and
    ``seed=5`` share the replicates 5-9. Those replicates are then perfectly correlated
    while feeding two separate recovery gates as if they were independent evidence.

    Seeds are therefore spawned from a :class:`numpy.random.SeedSequence` tree, as in
    :func:`behavio.recovery.run_parameter_recovery` and
    :func:`behavio.recovery.models.run_model_recovery`. Each requirement gets its own root,
    whose entropy mixes the declared seed with a domain-separation tag derived from the
    requirement name, so two requirements cannot address the same stream even when they
    declare the same seed, and one requirement's replicates never depend on another's
    declared seed. Cases and repetitions within a requirement are disjoint by construction
    because they are distinct children of that root. The emitted seeds are then checked for
    global uniqueness, so a collision refuses the run instead of silently reporting
    correlated replicates as independent.
    """

    positions: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        positions.setdefault(case.requirement, []).append(index)
    plan: dict[int, tuple[int, ...]] = {}
    for name, indices in positions.items():
        requirement = requirement_by_name[name]
        root = np.random.SeedSequence(entropy=(requirement.seed, *_requirement_domain(name)))
        sequences = root.spawn(len(indices) * requirement.repetitions)
        for offset, index in enumerate(indices):
            start = offset * requirement.repetitions
            plan[index] = tuple(
                int(sequence.generate_state(1, dtype=np.uint64)[0])
                for sequence in sequences[start : start + requirement.repetitions]
            )
    emitted = [seed for seeds in plan.values() for seed in seeds]
    if len(set(emitted)) != len(emitted):
        raise ExactRecoveryError(
            "recovery replicate seeds are not disjoint; distinct requirements must not "
            "share a replicate stream"
        )
    return tuple(plan[index] for index in range(len(cases)))


def _requirement_domain(name: str) -> tuple[int, int]:
    """Return a stable 128-bit domain-separation tag for one requirement name."""

    digest = hashlib.sha256(f"behavio.exact-recovery/requirement/{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:16], "big")


def _assessment_for(
    requirement: RecoverySpec,
    assessments: Mapping[str, RecoveryAssessment],
) -> RecoveryAssessment:
    if requirement.name in assessments:
        return assessments[requirement.name]
    if requirement.success_metric == "selection-rate":
        return _selection_score_for(requirement)
    if requirement.success_metric == "parameter-rmse":
        return _parameter_rmse
    raise ExactRecoveryError(
        f"recovery requirement {requirement.name!r} needs an explicit assessment for "
        f"metric {requirement.success_metric!r}"
    )


def _selection_score_for(requirement: RecoverySpec) -> RecoveryAssessment:
    def assess(_study: Study, run: ProtocolRun, case: RecoveryCase) -> float:
        candidates = tuple(
            candidate
            for candidate in run.report.candidates
            if not requirement.candidate_names or candidate.name in requirement.candidate_names
        )
        eligible = tuple(candidate for candidate in candidates if candidate.eligible)
        selected = (
            min(eligible, key=lambda candidate: candidate.score.unit_balanced_score).name
            if eligible
            else None
        )
        expected = case.expected_winner or case.generating_candidate
        return float(selected == expected)

    return assess


def _parameter_rmse(_study: Study, run: ProtocolRun, case: RecoveryCase) -> float:
    result = next(
        candidate
        for candidate in run.report.candidates
        if candidate.name == case.generating_candidate
    )
    if not result.eligible:
        raise ValueError("generating candidate did not complete every recovery fold")
    truth = case.parameter_map
    squared_errors: list[float] = []
    for fold in result.folds:
        estimates = {
            name: float(value)
            for name, value in zip(
                fold.fit.parameter_names,
                fold.fit.estimates,
                strict=True,
            )
        }
        common = set(truth) & set(estimates)
        if not common:
            raise ValueError("no generating parameters are exposed by the fitted candidate")
        squared_errors.extend((estimates[name] - truth[name]) ** 2 for name in common)
    return float(np.sqrt(np.mean(squared_errors)))


def _compiled_with_study(compiled: CompiledProtocol, study: Study) -> CompiledProtocol:
    materialized = replace(compiled.materialized, study=study)
    return replace(compiled, materialized=materialized)


def _validate_exact_plan(original: CompiledProtocol, simulated: CompiledProtocol) -> None:
    if simulated.protocol.fingerprint != original.protocol.fingerprint:
        raise ExactRecoveryError("recovery recompilation changed the protocol identity")
    original_geometry = tuple(
        (
            fold.fit_rows,
            fold.prediction_context_rows,
            fold.scored_rows,
            fold.excluded_rows,
        )
        for fold in original.plan.folds
    )
    simulated_geometry = tuple(
        (
            fold.fit_rows,
            fold.prediction_context_rows,
            fold.scored_rows,
            fold.excluded_rows,
        )
        for fold in simulated.plan.folds
    )
    if simulated_geometry != original_geometry:
        raise ExactRecoveryError("recovery recompilation changed the exact row geometry")


def _validate_simulated_design(original: Study, simulated: Study) -> None:
    if len(simulated) != len(original):
        raise ExactRecoveryError("simulator changed the number of design rows")
    for column in ("subject", "session", "trial", "session_order"):
        if column not in simulated.columns or not np.array_equal(
            original[column], simulated[column]
        ):
            raise ExactRecoveryError(f"simulator changed canonical design column {column!r}")


def _gate(requirement: RecoverySpec, scores: Sequence[float], *, requested: int) -> RecoveryGate:
    direction = _metric_direction(requirement.success_metric)
    observed = float(np.mean(scores)) if scores else None
    if observed is None:
        status = GateStatus.INSUFFICIENT
    elif direction == "at-least":
        status = GateStatus.PASSED if observed >= requirement.threshold else GateStatus.FAILED
    else:
        status = GateStatus.PASSED if observed <= requirement.threshold else GateStatus.FAILED
    return RecoveryGate(
        requirement=requirement.name,
        kind=requirement.kind,
        metric=requirement.success_metric,
        threshold=requirement.threshold,
        direction=direction,
        observed=observed,
        requested_repetitions=requested,
        effective_repetitions=len(scores),
        status=status,
        constrains_claims=requirement.constrains_claims,
    )


def _metric_direction(metric: str) -> str:
    lower_tokens = ("rmse", "error", "false-positive", "false-selection", "loss")
    return "at-most" if any(token in metric for token in lower_tokens) else "at-least"


def _recovery_failure(
    requirement: RecoverySpec,
    case: RecoveryCase,
    repetition: int,
    seed: int,
    stage: str,
    error: Exception,
) -> RecoveryFailure:
    return RecoveryFailure(
        requirement=requirement.name,
        case=case.name,
        repetition=repetition,
        seed=seed,
        stage=stage,
        exception_type=type(error).__name__,
        message=str(error),
    )


def _study_design_fingerprint(study: Study) -> str:
    identity = [
        [
            _json_scalar(study["subject"][row]),
            _json_scalar(study["session"][row]),
            int(study["trial"][row]),
            int(study["session_order"][row]),
        ]
        for row in range(len(study))
    ]
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_scalar(value: Any) -> Any:
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ExactRecoveryError(f"design identity is not a finite JSON scalar: {value!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
