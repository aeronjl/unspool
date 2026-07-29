"""Materialize and compile frozen study protocols into reviewable execution plans."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from behavio.models import ModelCapabilities
from behavio.protocol import (
    PredicateOperator,
    ProtocolState,
    ProtocolValidationError,
    ScoreMetric,
    StudyProtocol,
    ValidationGeometry,
)
from behavio.study import Study


class ProtocolCompilationError(ValueError):
    """Raised when source data cannot be materialized under a frozen protocol."""


class AuditLevel(StrEnum):
    """Severity of one pre-fit protocol audit finding."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    """Non-disclosive link from a source identity to one canonical study row."""

    source_row: int
    derived_row: int
    source_identity_fingerprint: str
    derived_identity_fingerprint: str


@dataclass(frozen=True, slots=True)
class CohortManifest:
    """Deterministic cohort denominators and source-to-derived identity resolution."""

    protocol_fingerprint: str
    source_checksum: str
    study_fingerprint: str
    source_observations: int
    selected_observations: int
    excluded_observations: int
    selected_subjects: int
    selected_sessions: int
    selected_source_rows: tuple[int, ...]
    excluded_source_rows: tuple[int, ...]
    identities: tuple[IdentityRecord, ...]

    @property
    def fingerprint(self) -> str:
        """Content address of the complete non-raw cohort manifest."""

        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe representation."""

        return _to_json(asdict(self))

    def canonical_json(self) -> str:
        """Serialize the manifest using canonical JSON."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class MaterializedStudy:
    """Canonical cohort data paired with its manifest and lifecycle state."""

    protocol: StudyProtocol
    study: Study
    manifest: CohortManifest

    def __post_init__(self) -> None:
        if self.protocol.state != ProtocolState.MATERIALIZED:
            raise ProtocolValidationError(
                "a materialized study requires materialized protocol state"
            )
        if self.protocol.fingerprint != self.manifest.protocol_fingerprint:
            raise ProtocolValidationError("manifest and protocol fingerprints differ")
        if len(self.study) != self.manifest.selected_observations:
            raise ProtocolValidationError("manifest selected denominator differs from study length")


@dataclass(frozen=True, slots=True)
class CompiledFold:
    """Exact rows available to fitting, prediction context, scoring, or neither."""

    identifier: str
    fit_rows: tuple[int, ...]
    prediction_context_rows: tuple[int, ...]
    scored_rows: tuple[int, ...]
    excluded_rows: tuple[int, ...]
    fit_subjects: tuple[JSONIdentifier, ...]
    scored_subjects: tuple[JSONIdentifier, ...]
    fit_groups: tuple[JSONIdentifier, ...]
    scored_groups: tuple[JSONIdentifier, ...]
    inner_folds: tuple[CompiledInnerFold, ...] = ()


JSONIdentifier = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class CompiledInnerFold:
    """Training-only validation rows mapped back to canonical study positions."""

    identifier: str
    fit_rows: tuple[int, ...]
    prediction_context_rows: tuple[int, ...]
    scored_rows: tuple[int, ...]
    excluded_rows: tuple[int, ...]
    fit_subjects: tuple[JSONIdentifier, ...]
    scored_subjects: tuple[JSONIdentifier, ...]
    fit_groups: tuple[JSONIdentifier, ...]
    scored_groups: tuple[JSONIdentifier, ...]


@dataclass(frozen=True, slots=True)
class ProtocolAuditIssue:
    """One stable, machine-readable pre-fit finding."""

    level: AuditLevel
    code: str
    message: str
    fold: str | None = None
    candidate: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", AuditLevel(self.level))


@dataclass(frozen=True, slots=True)
class ProtocolAudit:
    """Complete pre-fit audit retained even when compilation cannot be approved."""

    issues: tuple[ProtocolAuditIssue, ...]

    @property
    def passed(self) -> bool:
        """Whether no error-level finding remains."""

        return not any(issue.level == AuditLevel.ERROR for issue in self.issues)

    @property
    def errors(self) -> tuple[ProtocolAuditIssue, ...]:
        """Error-level findings in deterministic check order."""

        return tuple(issue for issue in self.issues if issue.level == AuditLevel.ERROR)

    @property
    def warnings(self) -> tuple[ProtocolAuditIssue, ...]:
        """Warning-level findings in deterministic check order."""

        return tuple(issue for issue in self.issues if issue.level == AuditLevel.WARNING)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Serializable, pre-fit account of every fold and information boundary."""

    schema_version: str
    protocol_fingerprint: str
    cohort_manifest_fingerprint: str
    study_fingerprint: str
    folds: tuple[CompiledFold, ...]
    audit: ProtocolAudit

    @property
    def fingerprint(self) -> str:
        """Content address of folds and the audit that approved or rejected them."""

        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the portable plan without models, arrays, or executable objects."""

        return _to_json(asdict(self))

    def canonical_json(self) -> str:
        """Serialize the plan deterministically."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class CompiledProtocol:
    """Execution plan together with the audited lifecycle state when it passes."""

    protocol: StudyProtocol
    materialized: MaterializedStudy
    plan: ExecutionPlan

    def __post_init__(self) -> None:
        expected_state = (
            ProtocolState.AUDITED if self.plan.audit.passed else ProtocolState.MATERIALIZED
        )
        if self.protocol.state != expected_state:
            raise ProtocolValidationError(
                f"compiled protocol state must be {expected_state.value!r} for its audit"
            )


def materialize_protocol(protocol: StudyProtocol, source_study: Study) -> MaterializedStudy:
    """Apply cohort rules and freeze exact denominators before compiling any fold.

    Source identity columns are required and hashed into the manifest.  Raw observed values
    remain in memory in :class:`Study`; the portable manifest contains only row positions,
    counts, and content digests.
    """

    if protocol.state != ProtocolState.FROZEN:
        raise ProtocolCompilationError("only a frozen protocol can materialize source data")
    required_columns = {
        *protocol.source.identity_columns,
        *protocol.cohort.selection_columns,
        *(predicate.column for predicate in protocol.cohort.predicates),
        *(unit.column for unit in protocol.units),
        *(observation.column for observation in protocol.observations),
        *(clock.column for clock in protocol.clocks if not clock.fold_fitted),
    }
    missing = sorted(required_columns - set(source_study.columns))
    if missing:
        raise ProtocolCompilationError(f"source study is missing protocol columns: {missing}")

    mask = np.ones(len(source_study), dtype=np.bool_)
    for predicate in protocol.cohort.predicates:
        mask &= _predicate_mask(source_study[predicate.column], predicate.operator, predicate.value)
    selected = tuple(int(value) for value in np.flatnonzero(mask))
    excluded = tuple(int(value) for value in np.flatnonzero(~mask))
    if not selected:
        raise ProtocolCompilationError("cohort rules exclude every source observation")

    selected_study = source_study.take(selected)
    _validate_expected_denominators(protocol, selected_study)
    _validate_panel(protocol, selected_study)
    identities = _identity_records(protocol, source_study, selected)
    study_fingerprint = _study_fingerprint(selected_study)
    manifest = CohortManifest(
        protocol_fingerprint=protocol.fingerprint,
        source_checksum=protocol.source.checksum,
        study_fingerprint=study_fingerprint,
        source_observations=len(source_study),
        selected_observations=len(selected_study),
        excluded_observations=len(excluded),
        selected_subjects=len(selected_study.subjects),
        selected_sessions=len(
            {
                (_portable_scalar(subject), _portable_scalar(session))
                for subject, session in zip(
                    selected_study["subject"], selected_study["session"], strict=True
                )
            }
        ),
        selected_source_rows=selected,
        excluded_source_rows=excluded,
        identities=identities,
    )
    materialized_protocol = protocol.advance(
        ProtocolState.MATERIALIZED,
        artifact_fingerprint=manifest.fingerprint,
    )
    return MaterializedStudy(materialized_protocol, selected_study, manifest)


def compile_execution_plan(
    materialized: MaterializedStudy,
    splits: Sequence[Any],
    *,
    capabilities: Mapping[str, ModelCapabilities],
    inner_splitter: Callable[[Study, int], Sequence[Any]] | None = None,
) -> CompiledProtocol:
    """Compile fold rows and audit temporal, unit, group, and model boundaries.

    The function always returns an :class:`ExecutionPlan`.  A failed plan retains all
    findings and leaves the protocol at ``materialized``; a passing plan advances it to
    ``audited`` using the plan fingerprint as lifecycle evidence.
    """

    protocol = materialized.protocol
    study = materialized.study
    if protocol.state != ProtocolState.MATERIALIZED:
        raise ProtocolCompilationError("execution plans require materialized protocol state")

    issues: list[ProtocolAuditIssue] = []
    compiled_folds: list[CompiledFold] = []
    _audit_protocol_capabilities(protocol, study, capabilities, issues)
    _audit_transform_visibility(protocol, issues)
    if not splits:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "no-folds",
                "the declared validation design produced no folds",
            )
        )
    if protocol.selection is not None and inner_splitter is None:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "inner-splitter-missing",
                "nested selection requires a training-only inner splitter",
            )
        )
    if protocol.selection is None and inner_splitter is not None:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.WARNING,
                "unused-inner-splitter",
                "an inner splitter was supplied but the protocol declares no nested selection",
            )
        )

    group_column = _group_column(protocol)
    clock_column = _audit_clock_column(protocol, study, issues)
    for fold_index, split in enumerate(splits):
        identifier = f"fold-{fold_index:04d}"
        fit_rows = _indices(split, "train_indices", len(study), identifier, issues)
        scored_rows = _indices(split, "test_indices", len(study), identifier, issues)
        context_rows = _indices(
            split,
            "prediction_context_indices",
            len(study),
            identifier,
            issues,
            allow_missing=True,
        )
        fit_set = set(fit_rows)
        score_set = set(scored_rows)
        context_set = set(context_rows)
        if fit_set & score_set:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "fit-score-overlap",
                    "fitting and scored rows overlap",
                    fold=identifier,
                )
            )
        if not context_set <= fit_set:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "context-outside-fit",
                    "prediction-context rows must be available within fitting history",
                    fold=identifier,
                )
            )
        excluded_rows = tuple(sorted(set(range(len(study))) - fit_set - score_set))
        fit_subjects = _identifiers(study, "subject", fit_rows)
        scored_subjects = _identifiers(study, "subject", scored_rows)
        fit_groups = _identifiers(study, group_column, fit_rows) if group_column else ()
        scored_groups = _identifiers(study, group_column, scored_rows) if group_column else ()
        _audit_fold_boundaries(
            protocol.validation.geometry,
            study,
            identifier,
            fit_rows,
            scored_rows,
            fit_subjects,
            scored_subjects,
            fit_groups,
            scored_groups,
            clock_column,
            issues,
        )
        compiled_inner = (
            _compile_inner_folds(
                protocol,
                study,
                identifier,
                fold_index,
                fit_rows,
                inner_splitter,
                group_column,
                clock_column,
                issues,
            )
            if protocol.selection is not None and inner_splitter is not None and fit_rows
            else ()
        )
        compiled_folds.append(
            CompiledFold(
                identifier=identifier,
                fit_rows=fit_rows,
                prediction_context_rows=context_rows,
                scored_rows=scored_rows,
                excluded_rows=excluded_rows,
                fit_subjects=fit_subjects,
                scored_subjects=scored_subjects,
                fit_groups=fit_groups,
                scored_groups=scored_groups,
                inner_folds=compiled_inner,
            )
        )

    issues.append(
        ProtocolAuditIssue(
            AuditLevel.INFO,
            "row-roles-compiled",
            f"compiled {len(compiled_folds)} folds with explicit fit, context, score, "
            "and exclusion rows",
        )
    )
    audit = ProtocolAudit(tuple(issues))
    plan = ExecutionPlan(
        schema_version="behavio.execution-plan/1",
        protocol_fingerprint=protocol.fingerprint,
        cohort_manifest_fingerprint=materialized.manifest.fingerprint,
        study_fingerprint=materialized.manifest.study_fingerprint,
        folds=tuple(compiled_folds),
        audit=audit,
    )
    audited_protocol = (
        protocol.advance(ProtocolState.AUDITED, artifact_fingerprint=plan.fingerprint)
        if audit.passed
        else protocol
    )
    return CompiledProtocol(audited_protocol, materialized, plan)


def _compile_inner_folds(
    protocol: StudyProtocol,
    study: Study,
    outer_identifier: str,
    outer_index: int,
    outer_fit_rows: tuple[int, ...],
    inner_splitter: Callable[[Study, int], Sequence[Any]],
    group_column: str | None,
    clock_column: str,
    issues: list[ProtocolAuditIssue],
) -> tuple[CompiledInnerFold, ...]:
    outer_training = study.take(outer_fit_rows)
    try:
        inner_splits = tuple(inner_splitter(outer_training, outer_index))
    except Exception as error:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "inner-splitter-failed",
                f"training-only inner splitter failed: {type(error).__name__}: {error}",
                fold=outer_identifier,
            )
        )
        return ()
    if not inner_splits:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "no-inner-folds",
                "nested selection produced no inner folds",
                fold=outer_identifier,
            )
        )
        return ()
    compiled: list[CompiledInnerFold] = []
    outer_set = set(outer_fit_rows)
    for inner_index, split in enumerate(inner_splits):
        identifier = f"{outer_identifier}/inner-{inner_index:04d}"
        local_fit = _indices(split, "train_indices", len(outer_training), identifier, issues)
        local_scored = _indices(split, "test_indices", len(outer_training), identifier, issues)
        local_context = _indices(
            split,
            "prediction_context_indices",
            len(outer_training),
            identifier,
            issues,
            allow_missing=True,
        )
        fit_rows = tuple(outer_fit_rows[row] for row in local_fit)
        scored_rows = tuple(outer_fit_rows[row] for row in local_scored)
        context_rows = tuple(outer_fit_rows[row] for row in local_context)
        fit_set = set(fit_rows)
        scored_set = set(scored_rows)
        context_set = set(context_rows)
        if fit_set & scored_set:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "inner-fit-score-overlap",
                    "inner fitting and scored rows overlap",
                    fold=identifier,
                )
            )
        if not context_set <= fit_set:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "inner-context-outside-fit",
                    "inner prediction context must lie inside inner fitting history",
                    fold=identifier,
                )
            )
        if not (fit_set | scored_set) <= outer_set:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "outer-test-visible-to-selection",
                    "an inner fold addresses rows outside its outer training study",
                    fold=identifier,
                )
            )
        excluded_rows = tuple(sorted(outer_set - fit_set - scored_set))
        fit_subjects = _identifiers(study, "subject", fit_rows)
        scored_subjects = _identifiers(study, "subject", scored_rows)
        fit_groups = _identifiers(study, group_column, fit_rows) if group_column else ()
        scored_groups = _identifiers(study, group_column, scored_rows) if group_column else ()
        _audit_fold_boundaries(
            protocol.selection.inner_validation.geometry,
            study,
            identifier,
            fit_rows,
            scored_rows,
            fit_subjects,
            scored_subjects,
            fit_groups,
            scored_groups,
            clock_column,
            issues,
        )
        compiled.append(
            CompiledInnerFold(
                identifier,
                fit_rows,
                context_rows,
                scored_rows,
                excluded_rows,
                fit_subjects,
                scored_subjects,
                fit_groups,
                scored_groups,
            )
        )
    return tuple(compiled)


def _audit_protocol_capabilities(
    protocol: StudyProtocol,
    study: Study,
    capabilities: Mapping[str, ModelCapabilities],
    issues: list[ProtocolAuditIssue],
) -> None:
    unknown = sorted(set(capabilities) - {candidate.name for candidate in protocol.candidates})
    if unknown:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.WARNING,
                "unused-capabilities",
                f"capabilities were supplied for undeclared candidates: {unknown}",
            )
        )
    for candidate in protocol.candidates:
        declared = capabilities.get(candidate.name)
        if declared is None:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "missing-capabilities",
                    "candidate runtime capabilities were not supplied",
                    candidate=candidate.name,
                )
            )
            continue
        if declared.scored_columns != candidate.scored_columns:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "scored-columns-mismatch",
                    f"runtime scores {declared.scored_columns!r}, protocol declares "
                    f"{candidate.scored_columns!r}",
                    candidate=candidate.name,
                )
            )
        expected_mode = candidate.prediction_information.value
        if expected_mode not in {mode.value for mode in declared.prediction_modes}:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "prediction-mode-mismatch",
                    f"runtime does not support declared prediction information {expected_mode!r}",
                    candidate=candidate.name,
                )
            )
        if protocol.recovery and not declared.can_simulate:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "recovery-capability-missing",
                    "declared exact-design recovery requires a generative candidate",
                    candidate=candidate.name,
                )
            )
    unit_columns = {unit.name: unit.column for unit in protocol.units}
    aggregation_uses = [("comparison", protocol.comparison.aggregation_unit)]
    if protocol.selection is not None:
        aggregation_uses.append(("nested selection", protocol.selection.aggregation_unit))
    checked_columns: set[str] = set()
    for use, unit_name in aggregation_uses:
        aggregation_column = unit_columns[unit_name]
        if aggregation_column in checked_columns:
            continue
        checked_columns.add(aggregation_column)
        if aggregation_column not in study.columns:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "aggregation-column-missing",
                    f"study does not contain {use} aggregation column {aggregation_column!r}",
                )
            )
        elif any(_is_missing(value) for value in study[aggregation_column]):
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "aggregation-unit-missing",
                    f"{use} aggregation column {aggregation_column!r} contains missing values",
                )
            )
    metrics = {
        protocol.comparison.metric,
        *((protocol.selection.metric,) if protocol.selection is not None else ()),
    }
    if ScoreMetric.BRIER in metrics:
        outcome_column = protocol.candidates[0].scored_columns[0]
        invalid = sum(not _is_binary_value(value) for value in study[outcome_column])
        if invalid:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "brier-outcome-invalid",
                    f"Brier scoring requires binary 0/1 outcomes; found {invalid} invalid rows",
                )
            )


def _audit_transform_visibility(protocol: StudyProtocol, issues: list[ProtocolAuditIssue]) -> None:
    outputs = {
        column: transform
        for transform in protocol.transforms
        for column in transform.output_columns
    }
    for clock in protocol.clocks:
        if not clock.fold_fitted:
            continue
        transform = outputs.get(clock.column)
        if transform is None:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "fold-fitted-clock-unresolved",
                    f"fold-fitted clock {clock.name!r} has no declared producing transform",
                )
            )
        elif transform.visibility.value != "training-only":
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "fold-fitted-clock-leakage",
                    f"fold-fitted clock {clock.name!r} is not produced training-only",
                )
            )


def _audit_fold_boundaries(
    geometry: ValidationGeometry,
    study: Study,
    identifier: str,
    fit_rows: tuple[int, ...],
    scored_rows: tuple[int, ...],
    fit_subjects: tuple[JSONIdentifier, ...],
    scored_subjects: tuple[JSONIdentifier, ...],
    fit_groups: tuple[JSONIdentifier, ...],
    scored_groups: tuple[JSONIdentifier, ...],
    clock_column: str,
    issues: list[ProtocolAuditIssue],
) -> None:
    if not fit_rows or not scored_rows:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "empty-fold-role",
                "every fold requires at least one fitting and one scored row",
                fold=identifier,
            )
        )
        return
    held_out_subject = geometry in (
        ValidationGeometry.HELD_OUT_SUBJECT,
        ValidationGeometry.HELD_OUT_GROUP,
        ValidationGeometry.HELD_OUT_GROUP_FUTURE_SESSION,
    )
    if held_out_subject and set(fit_subjects) & set(scored_subjects):
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "subject-boundary-leakage",
                "held-out subjects appear in fitting rows",
                fold=identifier,
            )
        )
    if geometry in (
        ValidationGeometry.HELD_OUT_GROUP,
        ValidationGeometry.HELD_OUT_GROUP_FUTURE_SESSION,
    ) and set(fit_groups) & set(scored_groups):
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "group-boundary-leakage",
                "held-out groups appear in fitting rows",
                fold=identifier,
            )
        )
    if geometry in (
        ValidationGeometry.FUTURE_SESSION,
        ValidationGeometry.FUTURE_TRIAL,
        ValidationGeometry.HISTORICAL_COHORT_FUTURE_SESSION,
    ):
        _audit_temporal_overlap(study, identifier, fit_rows, scored_rows, clock_column, issues)


def _audit_temporal_overlap(
    study: Study,
    identifier: str,
    fit_rows: tuple[int, ...],
    scored_rows: tuple[int, ...],
    clock_column: str,
    issues: list[ProtocolAuditIssue],
) -> None:
    fit_by_subject = _rows_by_subject(study, fit_rows)
    scored_by_subject = _rows_by_subject(study, scored_rows)
    for subject in sorted(set(fit_by_subject) & set(scored_by_subject), key=str):
        train_clock = [study[clock_column][row] for row in fit_by_subject[subject]]
        score_clock = [study[clock_column][row] for row in scored_by_subject[subject]]
        try:
            valid = max(train_clock) < min(score_clock)
        except TypeError:
            valid = False
        if not valid:
            issues.append(
                ProtocolAuditIssue(
                    AuditLevel.ERROR,
                    "temporal-boundary-leakage",
                    f"subject {subject!r} has fitting coordinates at or after scored coordinates",
                    fold=identifier,
                )
            )


def _indices(
    split: Any,
    name: str,
    length: int,
    fold: str,
    issues: list[ProtocolAuditIssue],
    *,
    allow_missing: bool = False,
) -> tuple[int, ...]:
    if not hasattr(split, name):
        if allow_missing:
            return ()
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "split-field-missing",
                f"validation split does not provide {name}",
                fold=fold,
            )
        )
        return ()
    raw = np.asarray(getattr(split, name))
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "split-indices-invalid",
                f"{name} must be a one-dimensional integer array",
                fold=fold,
            )
        )
        return ()
    values = tuple(int(value) for value in raw)
    if len(set(values)) != len(values):
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "split-indices-duplicate",
                f"{name} contains duplicate rows",
                fold=fold,
            )
        )
    invalid = tuple(value for value in values if value < 0 or value >= length)
    if invalid:
        issues.append(
            ProtocolAuditIssue(
                AuditLevel.ERROR,
                "split-indices-out-of-range",
                f"{name} contains out-of-range rows: {invalid[:5]!r}",
                fold=fold,
            )
        )
    return tuple(sorted(value for value in set(values) if 0 <= value < length))


def _predicate_mask(
    values: np.ndarray[Any, Any], operator: PredicateOperator, expected: Any
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if operator == PredicateOperator.PRESENT:
        return np.fromiter((not _is_missing(value) for value in values), bool, len(values))
    if operator in (PredicateOperator.IN, PredicateOperator.NOT_IN):
        expected_keys = {_value_key(value) for value in expected}
        result = np.fromiter(
            (_value_key(value) in expected_keys for value in values), bool, len(values)
        )
        return ~result if operator == PredicateOperator.NOT_IN else result

    def compare(value: Any) -> bool:
        if _is_missing(value):
            return False
        scalar = _portable_scalar(value)
        if operator == PredicateOperator.EQUAL:
            return scalar == expected
        if operator == PredicateOperator.NOT_EQUAL:
            return scalar != expected
        if operator == PredicateOperator.GREATER_EQUAL:
            return bool(scalar >= expected)
        if operator == PredicateOperator.LESS_EQUAL:
            return bool(scalar <= expected)
        raise AssertionError(f"unhandled cohort operator: {operator}")

    try:
        return np.fromiter((compare(value) for value in values), bool, len(values))
    except TypeError as error:
        raise ProtocolCompilationError(
            f"cohort predicate values are not comparable under {operator.value!r}"
        ) from error


def _validate_expected_denominators(protocol: StudyProtocol, study: Study) -> None:
    sessions = {
        (_value_key(subject), _value_key(session))
        for subject, session in zip(study["subject"], study["session"], strict=True)
    }
    observed = {
        "subjects": len(study.subjects),
        "sessions": len(sessions),
        "observations": len(study),
    }
    expected = {
        "subjects": protocol.cohort.expected_subjects,
        "sessions": protocol.cohort.expected_sessions,
        "observations": protocol.cohort.expected_observations,
    }
    mismatches = {
        name: (value, observed[name])
        for name, value in expected.items()
        if value is not None and value != observed[name]
    }
    if mismatches:
        raise ProtocolCompilationError(
            f"materialized cohort denominators differ from protocol: {mismatches}"
        )


def _validate_panel(protocol: StudyProtocol, study: Study) -> None:
    sessions_by_subject: dict[str, set[str]] = {}
    clocks_by_subject: dict[str, set[str]] = {}
    clock = next(item for item in protocol.clocks if item.name == protocol.panel.common_clock)
    for row in range(len(study)):
        subject = _value_key(study["subject"][row])
        sessions_by_subject.setdefault(subject, set()).add(_value_key(study["session"][row]))
        if not clock.fold_fitted:
            clocks_by_subject.setdefault(subject, set()).add(_value_key(study[clock.column][row]))
    short = {
        subject: len(sessions)
        for subject, sessions in sessions_by_subject.items()
        if len(sessions) < protocol.panel.minimum_sessions
    }
    if short:
        raise ProtocolCompilationError(
            "subjects do not satisfy the declared minimum session count: "
            f"{dict(sorted(short.items()))}"
        )
    if protocol.panel.balance_required and not clock.fold_fitted:
        coordinate_sets = {tuple(sorted(values)) for values in clocks_by_subject.values()}
        if len(coordinate_sets) != 1:
            raise ProtocolCompilationError(
                "balanced panel requires identical common-clock coordinates for every subject"
            )


def _identity_records(
    protocol: StudyProtocol, source: Study, selected_rows: tuple[int, ...]
) -> tuple[IdentityRecord, ...]:
    seen: set[str] = set()
    records: list[IdentityRecord] = []
    for derived_row, source_row in enumerate(selected_rows):
        source_identity = tuple(
            _portable_value(source[column][source_row])
            for column in protocol.source.identity_columns
        )
        source_digest = _fingerprint(source_identity)
        if source_digest in seen:
            raise ProtocolCompilationError(
                "source identity columns do not uniquely identify selected observations"
            )
        seen.add(source_digest)
        derived_identity = tuple(
            _portable_value(source[column][source_row])
            for column in ("subject", "session", "trial")
        )
        records.append(
            IdentityRecord(
                source_row=source_row,
                derived_row=derived_row,
                source_identity_fingerprint=source_digest,
                derived_identity_fingerprint=_fingerprint(derived_identity),
            )
        )
    return tuple(records)


def _study_fingerprint(study: Study) -> str:
    return _fingerprint(
        {
            "columns": list(study.columns),
            "values": {
                column: [_portable_value(value) for value in study[column]]
                for column in study.columns
            },
        }
    )


def _audit_clock_column(
    protocol: StudyProtocol,
    study: Study,
    issues: list[ProtocolAuditIssue],
) -> str:
    if protocol.validation.geometry == ValidationGeometry.FUTURE_TRIAL:
        return "trial"
    clock = next(item for item in protocol.clocks if item.name == protocol.panel.common_clock)
    if clock.column in study.columns:
        return clock.column
    issues.append(
        ProtocolAuditIssue(
            AuditLevel.INFO,
            "temporal-boundary-delegated",
            f"fold-fitted clock {clock.name!r} will be audited after training-only transformation",
        )
    )
    return "session_order"


def _group_column(protocol: StudyProtocol) -> str | None:
    if protocol.validation.group_unit is None:
        return None
    return next(
        unit.column for unit in protocol.units if unit.name == protocol.validation.group_unit
    )


def _identifiers(
    study: Study, column: str | None, rows: tuple[int, ...]
) -> tuple[JSONIdentifier, ...]:
    if column is None:
        return ()
    values: list[JSONIdentifier] = []
    seen: set[str] = set()
    for row in rows:
        value = _portable_scalar(study[column][row])
        key = _canonical_json(value)
        if key not in seen:
            seen.add(key)
            values.append(value)
    return tuple(values)


def _rows_by_subject(study: Study, rows: tuple[int, ...]) -> dict[JSONIdentifier, list[int]]:
    result: dict[JSONIdentifier, list[int]] = {}
    for row in rows:
        result.setdefault(_portable_scalar(study["subject"][row]), []).append(row)
    return result


def _portable_scalar(value: Any) -> JSONIdentifier:
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ProtocolCompilationError(f"identifier is not a finite JSON scalar: {value!r}")


def _portable_value(value: Any) -> Any:
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"special_float": "nan"}
        if math.isinf(value):
            return {"special_float": "positive-infinity" if value > 0 else "negative-infinity"}
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (tuple, list)):
        return [_portable_value(item) for item in value]
    raise ProtocolCompilationError(f"study value is not deterministically serializable: {value!r}")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _is_binary_value(value: Any) -> bool:
    value = value.item() if isinstance(value, np.generic) else value
    return isinstance(value, (bool, int, float)) and value in (0, 1)


def _value_key(value: Any) -> str:
    return _canonical_json(_portable_value(value))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _to_json(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json(item) for item in value]
    return value
