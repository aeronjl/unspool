"""Typed, immutable declarations for reproducible longitudinal studies.

The protocol is intentionally a closed scientific contract rather than a workflow
configuration language.  It declares what is to be learned, what information may be used,
and how evidence will be judged; execution is handled by the protocol compiler and runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from typing import Any, Self

SCHEMA_VERSION = "behavio.study-protocol/1"

#: Schema names this format has been published under. The package was distributed as
#: ``unspool`` up to and including 0.1.0, and a frozen protocol is content-addressed, so a
#: protocol recorded before the rename cannot be restamped without invalidating its own
#: fingerprint. The payload is byte-identical apart from this name, so it is read as-is and
#: its recorded name is preserved on the reconstructed object.
SUPERSEDED_SCHEMA_VERSIONS = ("unspool.study-protocol/1",)
ACCEPTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, *SUPERSEDED_SCHEMA_VERSIONS)

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | tuple["JSONValue", ...]


class ProtocolValidationError(ValueError):
    """Raised when a study protocol is incomplete or internally inconsistent."""


class ProtocolLifecycleError(RuntimeError):
    """Raised when a protocol lifecycle transition is not permitted."""


class ProtocolState(StrEnum):
    """Controlled evidence lifecycle for one immutable scientific design."""

    DRAFT = "draft"
    FROZEN = "frozen"
    MATERIALIZED = "materialized"
    AUDITED = "audited"
    EVALUATED = "evaluated"
    RECOVERED = "recovered"
    REPORTED = "reported"


class PredicateOperator(StrEnum):
    """Closed set of outcome-independent cohort operations."""

    EQUAL = "equal"
    NOT_EQUAL = "not-equal"
    IN = "in"
    NOT_IN = "not-in"
    GREATER_EQUAL = "greater-equal"
    LESS_EQUAL = "less-equal"
    PRESENT = "present"


class ObservationRole(StrEnum):
    """How a declared column may be used by the study."""

    OUTCOME = "outcome"
    PREDICTOR = "predictor"
    AUXILIARY = "auxiliary"


class ObservationDataType(StrEnum):
    """Closed measurement vocabulary for one declared observation column.

    ``data_type`` was a free string until it acquired an enforced meaning. The member
    values *are* the wire format, so every protocol Behavio has ever serialized keeps
    round-tripping byte-identically and keeps its fingerprint: ``"binary"`` and
    ``"continuous"`` are the only strings any released declaration recorded, and both are
    members here. Widening this vocabulary later stays backward compatible; narrowing it
    would not, and would need the :data:`SUPERSEDED_SCHEMA_VERSIONS` treatment that the
    package rename used.
    """

    BINARY = "binary"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    COUNT = "count"
    ORDINAL = "ordinal"


class UnitRole(StrEnum):
    """Inferential role of a unit identifier."""

    EXPERIMENTAL = "experimental"
    REPEATED_MEASURES = "repeated-measures"
    AGGREGATION = "aggregation"


class TransformVisibility(StrEnum):
    """Information boundary for learned preprocessing."""

    TRAINING_ONLY = "training-only"
    FIXED_A_PRIORI = "fixed-a-priori"


class ValidationGeometry(StrEnum):
    """Supported prospective deployment geometries."""

    FUTURE_SESSION = "future-session"
    FUTURE_TRIAL = "future-trial"
    HELD_OUT_SUBJECT = "held-out-subject"
    HELD_OUT_GROUP = "held-out-group"
    HELD_OUT_GROUP_FUTURE_SESSION = "held-out-group-future-session"
    HISTORICAL_COHORT_FUTURE_SESSION = "historical-cohort-future-session"


class PredictionInformation(StrEnum):
    """Information available when predictions are constructed."""

    FILTERED = "filtered"
    SMOOTHED_DESCRIPTION = "smoothed-description"


class ScoreMetric(StrEnum):
    """Proper pointwise scoring rules supported by the common runner."""

    LOG_LOSS = "log-loss"
    BRIER = "brier"
    JOINT_LOG_LOSS = "joint-log-loss"


class AggregationWeighting(StrEnum):
    """How repeated observations contribute to the target estimand."""

    EQUAL_UNIT = "equal-unit"
    POOLED_OBSERVATION = "pooled-observation"


class WinnerPolicy(StrEnum):
    """Predeclared interpretation of comparison uncertainty."""

    INTERVAL_EXCLUDES_ZERO = "interval-excludes-zero"
    LOWEST_POINT_ESTIMATE = "lowest-point-estimate"
    NO_AUTOMATIC_WINNER = "no-automatic-winner"


class RecoveryKind(StrEnum):
    """Design-specific falsification checks run through the compiled plan."""

    MODEL = "model"
    PARAMETER = "parameter"
    OUTCOME_DERIVED_FEATURE = "outcome-derived-feature"


class SelectionTieBreak(StrEnum):
    """Closed deterministic tie policies for nested training-only selection."""

    DECLARED_ORDER = "declared-order"


@dataclass(frozen=True, slots=True)
class Setting:
    """One deterministic named scalar or tuple-valued declaration."""

    name: str
    value: JSONValue

    def __post_init__(self) -> None:
        _name(self.name, "setting name")
        object.__setattr__(self, "value", _json_value(self.value, f"setting {self.name!r}"))


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Exact source identity without embedding access credentials or raw data."""

    adapter: str
    release: str
    locator: str
    checksum_algorithm: str
    checksum: str
    identity_columns: tuple[str, ...]
    metadata: tuple[Setting, ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.adapter, "source adapter"),
            (self.release, "source release"),
            (self.locator, "source locator"),
            (self.checksum_algorithm, "checksum algorithm"),
            (self.checksum, "source checksum"),
        ):
            _name(value, label)
        _names(self.identity_columns, "source identity columns")
        _settings(self.metadata, "source metadata")


@dataclass(frozen=True, slots=True)
class CohortPredicate:
    """One auditable cohort inclusion or exclusion condition."""

    column: str
    operator: PredicateOperator
    value: JSONValue = None
    rationale: str = ""

    def __post_init__(self) -> None:
        _name(self.column, "cohort predicate column")
        object.__setattr__(self, "operator", PredicateOperator(self.operator))
        object.__setattr__(self, "value", _json_value(self.value, "cohort predicate value"))
        _name(self.rationale, "cohort predicate rationale")
        if self.operator == PredicateOperator.PRESENT and self.value is not None:
            raise ProtocolValidationError("a 'present' predicate cannot declare a value")
        if self.operator in (PredicateOperator.IN, PredicateOperator.NOT_IN) and not isinstance(
            self.value, tuple
        ):
            raise ProtocolValidationError("'in' predicates require a tuple-valued declaration")


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """Outcome-blind cohort construction and expected denominator contract."""

    predicates: tuple[CohortPredicate, ...]
    selection_columns: tuple[str, ...]
    outcome_blind: bool
    expected_subjects: int | None = None
    expected_sessions: int | None = None
    expected_observations: int | None = None

    def __post_init__(self) -> None:
        if not self.predicates:
            raise ProtocolValidationError("cohort predicates must not be empty")
        _names(self.selection_columns, "cohort selection columns")
        if not isinstance(self.outcome_blind, bool):
            raise ProtocolValidationError("cohort outcome_blind must be boolean")
        for value, label in (
            (self.expected_subjects, "expected subjects"),
            (self.expected_sessions, "expected sessions"),
            (self.expected_observations, "expected observations"),
        ):
            if value is not None and (isinstance(value, bool) or value < 1):
                raise ProtocolValidationError(f"{label} must be a positive integer or null")


@dataclass(frozen=True, slots=True)
class UnitSpec:
    """Experimental, repeated-measures, or score-aggregation unit."""

    name: str
    column: str
    role: UnitRole
    nested_within: str | None = None

    def __post_init__(self) -> None:
        _name(self.name, "unit name")
        _name(self.column, "unit column")
        object.__setattr__(self, "role", UnitRole(self.role))
        if self.nested_within is not None:
            _name(self.nested_within, "unit nesting parent")
            if self.nested_within == self.name:
                raise ProtocolValidationError("a unit cannot be nested within itself")


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """One observed variable, its scientific use, and its enforced data contract.

    ``data_type`` and ``allowed_values`` are checked against the materialized column by
    :func:`behavio.compiler.materialize_protocol`; they are declarations the cohort must
    satisfy, not annotations.

    Missing data is legitimate in behavioural work, so a declared ``allowed_values`` set
    licenses missing observations only when it contains ``None``. This mirrors
    :class:`behavio.task.ChoiceSpec`, which refuses undeclared choice values but retains
    omissions that the specification names. An empty ``allowed_values`` declares no value
    contract at all and therefore never rejects a missing observation.
    """

    column: str
    role: ObservationRole
    data_type: ObservationDataType
    unit: str | None = None
    allowed_values: tuple[JSONScalar, ...] = ()

    def __post_init__(self) -> None:
        _name(self.column, "observation column")
        object.__setattr__(self, "role", ObservationRole(self.role))
        object.__setattr__(self, "data_type", _observation_data_type(self.data_type))
        if self.unit is not None:
            _name(self.unit, "observation unit")
        values = tuple(
            _json_value(value, "allowed observation value") for value in self.allowed_values
        )
        if any(isinstance(value, tuple) for value in values):
            raise ProtocolValidationError("allowed observation values must be scalars")
        if len({_canonical_scalar(value) for value in values}) != len(values):
            raise ProtocolValidationError("allowed observation values must be unique")
        object.__setattr__(self, "allowed_values", values)


@dataclass(frozen=True, slots=True)
class ProtocolClockSpec:
    """Declared across-trial coordinate with explicit scope and alignment."""

    name: str
    column: str
    kind: str
    scope: str
    alignment: str
    fold_fitted: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.name, "clock name"),
            (self.column, "clock column"),
            (self.kind, "clock kind"),
            (self.scope, "clock scope"),
            (self.alignment, "clock alignment"),
        ):
            _name(value, label)
        if not isinstance(self.fold_fitted, bool):
            raise ProtocolValidationError("clock fold_fitted must be boolean")


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """Repeated longitudinal panel required by the scientific question."""

    subject_unit: str
    session_unit: str
    common_clock: str
    minimum_sessions: int
    balance_required: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.subject_unit, "panel subject unit"),
            (self.session_unit, "panel session unit"),
            (self.common_clock, "panel common clock"),
        ):
            _name(value, label)
        if isinstance(self.minimum_sessions, bool) or self.minimum_sessions < 1:
            raise ProtocolValidationError("panel minimum_sessions must be positive")
        if not isinstance(self.balance_required, bool):
            raise ProtocolValidationError("panel balance_required must be boolean")


@dataclass(frozen=True, slots=True)
class EstimandSpec:
    """Target population, outcome, contrast, and aggregation unit."""

    name: str
    population: str
    outcome_columns: tuple[str, ...]
    contrast: str
    aggregation_unit: str
    weighting: AggregationWeighting

    def __post_init__(self) -> None:
        for value, label in (
            (self.name, "estimand name"),
            (self.population, "estimand population"),
            (self.contrast, "estimand contrast"),
            (self.aggregation_unit, "estimand aggregation unit"),
        ):
            _name(value, label)
        _names(self.outcome_columns, "estimand outcome columns")
        object.__setattr__(self, "weighting", AggregationWeighting(self.weighting))


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Preprocessing or landmark transform and its permitted evidence boundary."""

    name: str
    implementation: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    visibility: TransformVisibility
    parameters: tuple[Setting, ...] = ()
    uses_outcomes: bool = False

    def __post_init__(self) -> None:
        _name(self.name, "transform name")
        _name(self.implementation, "transform implementation")
        _names(self.input_columns, "transform input columns")
        _names(self.output_columns, "transform output columns")
        object.__setattr__(self, "visibility", TransformVisibility(self.visibility))
        _settings(self.parameters, "transform parameters")
        if not isinstance(self.uses_outcomes, bool):
            raise ProtocolValidationError("transform uses_outcomes must be boolean")
        if self.uses_outcomes and self.visibility != TransformVisibility.TRAINING_ONLY:
            raise ProtocolValidationError("outcome-derived transforms must be fitted training-only")


@dataclass(frozen=True, slots=True)
class ValidationSpec:
    """Outer deployment geometry and its row-information rules."""

    geometry: ValidationGeometry
    splitter: str
    prediction_information: PredictionInformation
    group_unit: str | None = None
    origin: int | None = None
    horizon: tuple[int, ...] = ()
    settings: tuple[Setting, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", ValidationGeometry(self.geometry))
        _name(self.splitter, "validation splitter")
        object.__setattr__(
            self, "prediction_information", PredictionInformation(self.prediction_information)
        )
        if self.group_unit is not None:
            _name(self.group_unit, "validation group unit")
        if self.origin is not None and (isinstance(self.origin, bool) or self.origin < 0):
            raise ProtocolValidationError("validation origin must be non-negative or null")
        horizon = tuple(self.horizon)
        if any(isinstance(value, bool) or value < 0 for value in horizon):
            raise ProtocolValidationError("validation horizon must contain non-negative integers")
        if len(set(horizon)) != len(horizon):
            raise ProtocolValidationError("validation horizon values must be unique")
        _settings(self.settings, "validation settings")
        object.__setattr__(self, "horizon", horizon)
        held_out = self.geometry in (
            ValidationGeometry.HELD_OUT_GROUP,
            ValidationGeometry.HELD_OUT_GROUP_FUTURE_SESSION,
        )
        if held_out != (self.group_unit is not None):
            raise ProtocolValidationError(
                "held-out-group geometries require exactly one declared group_unit"
            )
        if self.prediction_information == PredictionInformation.SMOOTHED_DESCRIPTION:
            raise ProtocolValidationError("validation scoring cannot use smoothed descriptions")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One fixed model candidate; no runtime code or search space is serialized."""

    name: str
    implementation: str
    hyperparameters: tuple[Setting, ...]
    scored_columns: tuple[str, ...]
    prediction_information: PredictionInformation = PredictionInformation.FILTERED
    supports_unseen_subjects: bool = False
    supports_unseen_groups: bool = False

    def __post_init__(self) -> None:
        _name(self.name, "candidate name")
        _name(self.implementation, "candidate implementation")
        _settings(self.hyperparameters, "candidate hyperparameters")
        _names(self.scored_columns, "candidate scored columns")
        object.__setattr__(
            self, "prediction_information", PredictionInformation(self.prediction_information)
        )
        if not isinstance(self.supports_unseen_subjects, bool) or not isinstance(
            self.supports_unseen_groups, bool
        ):
            raise ProtocolValidationError("candidate support declarations must be boolean")


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """Common scoring, uncertainty, pairing, and winner declaration."""

    metric: ScoreMetric
    aggregation_unit: str
    weighting: AggregationWeighting
    interval_method: str
    interval_level: float
    bootstrap_repetitions: int
    seed: int
    paired: bool
    winner_policy: WinnerPolicy
    reference_candidate: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", ScoreMetric(self.metric))
        _name(self.aggregation_unit, "comparison aggregation unit")
        object.__setattr__(self, "weighting", AggregationWeighting(self.weighting))
        _name(self.interval_method, "comparison interval method")
        if self.weighting != AggregationWeighting.EQUAL_UNIT:
            raise ProtocolValidationError(
                "the protocol runner currently supports equal-unit comparison weighting only"
            )
        if self.interval_method != "paired-unit-bootstrap":
            raise ProtocolValidationError(
                "the protocol runner currently supports paired-unit-bootstrap intervals only"
            )
        if not math.isfinite(self.interval_level) or not 0 < self.interval_level < 1:
            raise ProtocolValidationError("comparison interval_level must lie strictly in (0, 1)")
        if isinstance(self.bootstrap_repetitions, bool) or self.bootstrap_repetitions < 1:
            raise ProtocolValidationError("bootstrap_repetitions must be positive")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ProtocolValidationError("comparison seed must be non-negative")
        if not isinstance(self.paired, bool):
            raise ProtocolValidationError("comparison paired must be boolean")
        if not self.paired:
            raise ProtocolValidationError("the protocol runner requires paired comparisons")
        object.__setattr__(self, "winner_policy", WinnerPolicy(self.winner_policy))
        if self.reference_candidate is not None:
            _name(self.reference_candidate, "comparison reference candidate")


@dataclass(frozen=True, slots=True)
class NestedSelectionSpec:
    """Candidate selection performed wholly inside each outer training study."""

    candidate_names: tuple[str, ...]
    inner_validation: ValidationSpec
    metric: ScoreMetric
    aggregation_unit: str
    tie_break: SelectionTieBreak
    bootstrap_repetitions: int
    seed: int
    refit_selected_on_outer_training: bool = True

    def __post_init__(self) -> None:
        _names(self.candidate_names, "nested selection candidate names")
        object.__setattr__(self, "metric", ScoreMetric(self.metric))
        _name(self.aggregation_unit, "nested selection aggregation unit")
        object.__setattr__(self, "tie_break", SelectionTieBreak(self.tie_break))
        if isinstance(self.bootstrap_repetitions, bool) or self.bootstrap_repetitions < 1:
            raise ProtocolValidationError("nested bootstrap_repetitions must be positive")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ProtocolValidationError("nested selection seed must be non-negative")
        if not isinstance(self.refit_selected_on_outer_training, bool):
            raise ProtocolValidationError("nested refit declaration must be boolean")
        if not self.refit_selected_on_outer_training:
            raise ProtocolValidationError(
                "nested selection must refit on each outer training study"
            )


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    """Exact-design recovery requirement and its reportability gate."""

    name: str
    kind: RecoveryKind
    required: bool
    repetitions: int
    seed: int
    success_metric: str
    threshold: float
    constrains_claims: tuple[str, ...]
    candidate_names: tuple[str, ...] = ()
    scenarios: tuple[Setting, ...] = ()

    def __post_init__(self) -> None:
        _name(self.name, "recovery name")
        object.__setattr__(self, "kind", RecoveryKind(self.kind))
        if not isinstance(self.required, bool):
            raise ProtocolValidationError("recovery required must be boolean")
        if isinstance(self.repetitions, bool) or self.repetitions < 1:
            raise ProtocolValidationError("recovery repetitions must be positive")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ProtocolValidationError("recovery seed must be non-negative")
        _name(self.success_metric, "recovery success metric")
        if not math.isfinite(self.threshold):
            raise ProtocolValidationError("recovery threshold must be finite")
        _names(self.constrains_claims, "recovery constrained claims", allow_empty=True)
        _names(self.candidate_names, "recovery candidate names", allow_empty=True)
        _settings(self.scenarios, "recovery scenarios")


@dataclass(frozen=True, slots=True)
class ReportingSpec:
    """Required retained artifacts and bounded scientific statements."""

    required_tables: tuple[str, ...]
    required_figures: tuple[str, ...]
    required_diagnostics: tuple[str, ...]
    limitations: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    retain_pointwise_predictions: bool = True

    def __post_init__(self) -> None:
        _names(self.required_tables, "required report tables", allow_empty=True)
        _names(self.required_figures, "required report figures", allow_empty=True)
        _names(self.required_diagnostics, "required report diagnostics", allow_empty=True)
        _names(self.limitations, "report limitations")
        _names(self.prohibited_claims, "prohibited claims")
        if not isinstance(self.retain_pointwise_predictions, bool):
            raise ProtocolValidationError("retain_pointwise_predictions must be boolean")


@dataclass(frozen=True, slots=True)
class ProtocolAmendment:
    """Traceable pre-evidence change linked to the preceding frozen protocol."""

    identifier: str
    parent_fingerprint: str
    reason: str
    changed_sections: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.identifier, "amendment identifier")
        _fingerprint(self.parent_fingerprint, "amendment parent fingerprint")
        _name(self.reason, "amendment reason")
        _names(self.changed_sections, "amendment changed sections")


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """One evidence-backed state transition without changing the protocol identity."""

    from_state: ProtocolState
    to_state: ProtocolState
    artifact_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_state", ProtocolState(self.from_state))
        object.__setattr__(self, "to_state", ProtocolState(self.to_state))
        _fingerprint(self.artifact_fingerprint, "lifecycle artifact fingerprint")


@dataclass(frozen=True, slots=True)
class StudyProtocol:
    """Complete, versioned declaration of a longitudinal modelling study.

    Instances are deeply immutable.  ``fingerprint`` identifies only the scientific
    declaration and amendment history, so it remains stable as evidence-backed lifecycle
    events are appended.  ``canonical_json()`` serializes the complete current record.
    """

    identifier: str
    title: str
    question: str
    source: SourceSpec
    cohort: CohortSpec
    units: tuple[UnitSpec, ...]
    observations: tuple[ObservationSpec, ...]
    clocks: tuple[ProtocolClockSpec, ...]
    panel: PanelSpec
    estimands: tuple[EstimandSpec, ...]
    transforms: tuple[TransformSpec, ...]
    validation: ValidationSpec
    candidates: tuple[CandidateSpec, ...]
    comparison: ComparisonSpec
    recovery: tuple[RecoverySpec, ...]
    reporting: ReportingSpec
    selection: NestedSelectionSpec | None = None
    schema_version: str = SCHEMA_VERSION
    amendments: tuple[ProtocolAmendment, ...] = ()
    state: ProtocolState = ProtocolState.DRAFT
    lifecycle: tuple[LifecycleEvent, ...] = ()

    def __post_init__(self) -> None:
        _name(self.identifier, "protocol identifier")
        _name(self.title, "protocol title")
        _name(self.question, "protocol question")
        if self.schema_version not in ACCEPTED_SCHEMA_VERSIONS:
            raise ProtocolValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected one of {ACCEPTED_SCHEMA_VERSIONS!r}"
            )
        object.__setattr__(self, "state", ProtocolState(self.state))
        for field_name in ("units", "observations", "clocks", "estimands", "candidates"):
            if not getattr(self, field_name):
                raise ProtocolValidationError(f"protocol {field_name} must not be empty")
        self._validate_references()
        self._validate_lifecycle()

    @property
    def fingerprint(self) -> str:
        """SHA-256 identity of the scientific declaration and amendment history."""

        return hashlib.sha256(self.scientific_json().encode("utf-8")).hexdigest()

    def to_dict(self, *, scientific_only: bool = False) -> dict[str, Any]:
        """Return a standards-compliant JSON value with stable field names."""

        value = _to_json(asdict(self))
        if scientific_only:
            value.pop("state")
            value.pop("lifecycle")
        return value

    def canonical_json(self) -> str:
        """Serialize the full protocol deterministically without executable objects."""

        return _canonical_json(self.to_dict())

    def scientific_json(self) -> str:
        """Serialize only fingerprinted scientific content deterministically."""

        return _canonical_json(self.to_dict(scientific_only=True))

    def freeze(self) -> Self:
        """Freeze a valid draft before any source materialization or model fitting."""

        if self.state != ProtocolState.DRAFT or self.lifecycle:
            raise ProtocolLifecycleError("only a pristine draft protocol can be frozen")
        event = LifecycleEvent(
            from_state=ProtocolState.DRAFT,
            to_state=ProtocolState.FROZEN,
            artifact_fingerprint=self.fingerprint,
        )
        return replace(self, state=ProtocolState.FROZEN, lifecycle=(event,))

    def amend(self, *, identifier: str, reason: str, **changes: Any) -> Self:
        """Create a new draft linked to this frozen, pre-evidence protocol.

        Amendments are forbidden once materialization begins.  The caller must explicitly
        name every changed top-level section; lifecycle and amendment history cannot be
        overwritten through ``changes``.
        """

        if self.state != ProtocolState.FROZEN:
            raise ProtocolLifecycleError("only a frozen pre-evidence protocol may be amended")
        forbidden = {"schema_version", "amendments", "state", "lifecycle"}
        unknown = set(changes) - {field.name for field in fields(self)}
        if unknown:
            raise TypeError(f"unknown protocol sections: {sorted(unknown)}")
        if forbidden & set(changes):
            raise ProtocolLifecycleError(
                f"amendment cannot replace controlled fields: {sorted(forbidden & set(changes))}"
            )
        if not changes:
            raise ProtocolValidationError("an amendment must change at least one section")
        unchanged = [name for name, value in changes.items() if value == getattr(self, name)]
        if unchanged:
            raise ProtocolValidationError(f"amendment sections are unchanged: {sorted(unchanged)}")
        amendment = ProtocolAmendment(
            identifier=identifier,
            parent_fingerprint=self.fingerprint,
            reason=reason,
            changed_sections=tuple(sorted(changes)),
        )
        return replace(
            self,
            **changes,
            amendments=(*self.amendments, amendment),
            state=ProtocolState.DRAFT,
            lifecycle=(),
        )

    def advance(self, to_state: ProtocolState, *, artifact_fingerprint: str) -> Self:
        """Append one permitted evidence-backed lifecycle transition."""

        target = ProtocolState(to_state)
        allowed = _allowed_transitions(self)
        if target not in allowed:
            raise ProtocolLifecycleError(
                f"cannot advance protocol from {self.state.value!r} to {target.value!r}; "
                f"allowed={tuple(state.value for state in allowed)!r}"
            )
        event = LifecycleEvent(self.state, target, artifact_fingerprint)
        return replace(self, state=target, lifecycle=(*self.lifecycle, event))

    def _validate_references(self) -> None:
        unit_names = _unique_named(self.units, "units")
        observation_columns = _unique_columned(self.observations, "observations")
        clock_names = _unique_named(self.clocks, "clocks")
        estimand_names = _unique_named(self.estimands, "estimands")
        candidate_names = _unique_named(self.candidates, "candidates")
        transform_names = _unique_named(self.transforms, "transforms")
        recovery_names = _unique_named(self.recovery, "recovery declarations")
        del estimand_names, transform_names, recovery_names

        for unit in self.units:
            if unit.nested_within is not None and unit.nested_within not in unit_names:
                raise ProtocolValidationError(
                    f"unit {unit.name!r} has unknown nesting parent {unit.nested_within!r}"
                )
        if not any(unit.role == UnitRole.EXPERIMENTAL for unit in self.units):
            raise ProtocolValidationError("protocol must declare an experimental unit")
        if not any(unit.role == UnitRole.REPEATED_MEASURES for unit in self.units):
            raise ProtocolValidationError("protocol must declare a repeated-measures unit")
        for name in (self.panel.subject_unit, self.panel.session_unit):
            if name not in unit_names:
                raise ProtocolValidationError(f"panel references unknown unit {name!r}")
        if self.panel.common_clock not in clock_names:
            raise ProtocolValidationError(
                f"panel references unknown clock {self.panel.common_clock!r}"
            )
        if self.comparison.aggregation_unit not in unit_names:
            raise ProtocolValidationError(
                f"comparison references unknown unit {self.comparison.aggregation_unit!r}"
            )
        if self.comparison.reference_candidate not in candidate_names | {None}:
            raise ProtocolValidationError("comparison reference_candidate is not a candidate")
        if self.validation.group_unit is not None and self.validation.group_unit not in unit_names:
            raise ProtocolValidationError("validation group_unit is not a declared unit")
        if self.selection is not None:
            if not set(self.selection.candidate_names) <= candidate_names:
                raise ProtocolValidationError("nested selection references unknown candidates")
            if self.selection.aggregation_unit not in unit_names:
                raise ProtocolValidationError(
                    "nested selection references an unknown aggregation unit"
                )
            inner_group = self.selection.inner_validation.group_unit
            if inner_group is not None and inner_group not in unit_names:
                raise ProtocolValidationError(
                    "nested inner validation references an unknown group unit"
                )

        outcome_columns = {
            item.column for item in self.observations if item.role == ObservationRole.OUTCOME
        }
        if not outcome_columns:
            raise ProtocolValidationError("protocol must declare at least one outcome observation")
        for estimand in self.estimands:
            if estimand.aggregation_unit not in unit_names:
                raise ProtocolValidationError(
                    f"estimand {estimand.name!r} references unknown aggregation unit"
                )
            if not set(estimand.outcome_columns) <= outcome_columns:
                raise ProtocolValidationError(
                    f"estimand {estimand.name!r} references non-outcome columns"
                )
        score_sets = {candidate.scored_columns for candidate in self.candidates}
        if len(score_sets) != 1:
            raise ProtocolValidationError(
                "all candidates must score the same complete observation columns"
            )
        declared_metrics = {
            self.comparison.metric,
            *((self.selection.metric,) if self.selection is not None else ()),
        }
        scored_columns = self.candidates[0].scored_columns
        if ScoreMetric.BRIER in declared_metrics and len(scored_columns) != 1:
            raise ProtocolValidationError(
                "Brier scoring requires exactly one declared binary outcome column"
            )
        if len(scored_columns) > 1 and any(
            metric != ScoreMetric.JOINT_LOG_LOSS for metric in declared_metrics
        ):
            raise ProtocolValidationError(
                "multiple scored outcome columns require joint-log-loss scoring"
            )
        if not any(
            estimand.outcome_columns == scored_columns
            and estimand.aggregation_unit == self.comparison.aggregation_unit
            and estimand.weighting == self.comparison.weighting
            for estimand in self.estimands
        ):
            raise ProtocolValidationError(
                "comparison columns, aggregation unit, and weighting must match a declared estimand"
            )
        for candidate in self.candidates:
            if not set(candidate.scored_columns) <= outcome_columns:
                raise ProtocolValidationError(
                    f"candidate {candidate.name!r} scores columns not declared as outcomes"
                )
            if candidate.prediction_information != self.validation.prediction_information:
                raise ProtocolValidationError(
                    f"candidate {candidate.name!r} prediction information conflicts with validation"
                )
            if (
                self.validation.geometry
                in (
                    ValidationGeometry.HELD_OUT_SUBJECT,
                    ValidationGeometry.HELD_OUT_GROUP,
                    ValidationGeometry.HELD_OUT_GROUP_FUTURE_SESSION,
                )
                and not candidate.supports_unseen_subjects
            ):
                raise ProtocolValidationError(
                    f"candidate {candidate.name!r} cannot predict unseen subjects"
                )
            if (
                self.validation.geometry
                in (
                    ValidationGeometry.HELD_OUT_GROUP,
                    ValidationGeometry.HELD_OUT_GROUP_FUTURE_SESSION,
                )
                and not candidate.supports_unseen_groups
            ):
                raise ProtocolValidationError(
                    f"candidate {candidate.name!r} cannot predict unseen groups"
                )
        for transform in self.transforms:
            missing = set(transform.input_columns) - observation_columns
            if missing:
                raise ProtocolValidationError(
                    f"transform {transform.name!r} has undeclared input columns: {sorted(missing)}"
                )
        selection_outcomes = set(self.cohort.selection_columns) & outcome_columns
        if self.cohort.outcome_blind and selection_outcomes:
            raise ProtocolValidationError(
                f"outcome-blind cohort selects on outcome columns: {sorted(selection_outcomes)}"
            )
        constrained_claims = {
            claim for recovery in self.recovery for claim in recovery.constrains_claims
        }
        unknown_claims = constrained_claims - set(self.reporting.prohibited_claims)
        if unknown_claims:
            raise ProtocolValidationError(
                "recovery gates must reference claims declared prohibited until supported; "
                f"unknown={sorted(unknown_claims)}"
            )
        for recovery in self.recovery:
            unknown_candidates = set(recovery.candidate_names) - candidate_names
            if unknown_candidates:
                raise ProtocolValidationError(
                    f"recovery {recovery.name!r} references unknown candidates: "
                    f"{sorted(unknown_candidates)}"
                )

    def _validate_lifecycle(self) -> None:
        if self.state == ProtocolState.DRAFT:
            if self.lifecycle:
                raise ProtocolValidationError("draft protocols cannot retain lifecycle evidence")
            return
        if not self.lifecycle:
            raise ProtocolValidationError("non-draft protocols require lifecycle evidence")
        current = ProtocolState.DRAFT
        for index, event in enumerate(self.lifecycle):
            if event.from_state != current:
                raise ProtocolValidationError(
                    f"lifecycle event {index} does not continue from {current.value!r}"
                )
            if event.to_state not in _allowed_transitions_for_state(current, bool(self.recovery)):
                raise ProtocolValidationError(
                    f"lifecycle event {index} is not a permitted transition"
                )
            if index == 0 and event.artifact_fingerprint != self.fingerprint:
                raise ProtocolValidationError("freeze event must contain the protocol fingerprint")
            current = event.to_state
        if current != self.state:
            raise ProtocolValidationError("lifecycle events do not terminate at protocol state")


def protocol_from_dict(value: dict[str, Any]) -> StudyProtocol:
    """Deserialize a protocol from plain JSON data and re-run every invariant."""

    if not isinstance(value, dict):
        raise TypeError("protocol value must be a JSON object")
    expected = {field.name for field in fields(StudyProtocol)}
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ProtocolValidationError(
            "protocol fields differ from schema; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    def many(name: str, cls: type[Any]) -> tuple[Any, ...]:
        raw = value[name]
        if not isinstance(raw, list):
            raise ProtocolValidationError(f"protocol field {name!r} must be an array")
        return tuple(_construct(cls, item) for item in raw)

    return StudyProtocol(
        identifier=value["identifier"],
        title=value["title"],
        question=value["question"],
        source=_construct(SourceSpec, value["source"]),
        cohort=_construct(CohortSpec, value["cohort"]),
        units=many("units", UnitSpec),
        observations=many("observations", ObservationSpec),
        clocks=many("clocks", ProtocolClockSpec),
        panel=_construct(PanelSpec, value["panel"]),
        estimands=many("estimands", EstimandSpec),
        transforms=many("transforms", TransformSpec),
        validation=_construct(ValidationSpec, value["validation"]),
        candidates=many("candidates", CandidateSpec),
        comparison=_construct(ComparisonSpec, value["comparison"]),
        recovery=many("recovery", RecoverySpec),
        reporting=_construct(ReportingSpec, value["reporting"]),
        selection=(
            _construct(NestedSelectionSpec, value["selection"])
            if value["selection"] is not None
            else None
        ),
        schema_version=value["schema_version"],
        amendments=many("amendments", ProtocolAmendment),
        state=value["state"],
        lifecycle=many("lifecycle", LifecycleEvent),
    )


def protocol_from_json(value: str) -> StudyProtocol:
    """Deserialize standards-compliant JSON without accepting NaN or infinity."""

    def reject_constant(constant: str) -> None:
        raise ProtocolValidationError(f"non-finite JSON constant is forbidden: {constant}")

    try:
        parsed = json.loads(value, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise ProtocolValidationError(f"invalid protocol JSON: {error}") from error
    return protocol_from_dict(parsed)


def _construct(cls: type[Any], value: Any) -> Any:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{cls.__name__} must be a JSON object")
    expected = {field.name for field in fields(cls)}
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ProtocolValidationError(
            f"{cls.__name__} fields differ from schema; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    converted = dict(value)
    if cls is NestedSelectionSpec:
        converted["inner_validation"] = _construct(ValidationSpec, converted["inner_validation"])
    tuple_fields: dict[type[Any], dict[str, type[Any] | None]] = {
        SourceSpec: {"identity_columns": None, "metadata": Setting},
        CohortSpec: {"predicates": CohortPredicate, "selection_columns": None},
        ObservationSpec: {"allowed_values": None},
        EstimandSpec: {"outcome_columns": None},
        TransformSpec: {
            "input_columns": None,
            "output_columns": None,
            "parameters": Setting,
        },
        ValidationSpec: {"horizon": None, "settings": Setting},
        CandidateSpec: {"hyperparameters": Setting, "scored_columns": None},
        RecoverySpec: {
            "constrains_claims": None,
            "candidate_names": None,
            "scenarios": Setting,
        },
        ReportingSpec: {
            "required_tables": None,
            "required_figures": None,
            "required_diagnostics": None,
            "limitations": None,
            "prohibited_claims": None,
        },
        ProtocolAmendment: {"changed_sections": None},
        NestedSelectionSpec: {"candidate_names": None},
    }
    for name, item_type in tuple_fields.get(cls, {}).items():
        raw = converted.get(name, [])
        if not isinstance(raw, list):
            raise ProtocolValidationError(f"{cls.__name__}.{name} must be an array")
        converted[name] = (
            tuple(_construct(item_type, item) for item in raw) if item_type else tuple(raw)
        )
    return cls(**converted)


def _allowed_transitions(protocol: StudyProtocol) -> tuple[ProtocolState, ...]:
    return _allowed_transitions_for_state(protocol.state, bool(protocol.recovery))


def _allowed_transitions_for_state(
    state: ProtocolState, has_recovery: bool
) -> tuple[ProtocolState, ...]:
    if state == ProtocolState.DRAFT:
        return (ProtocolState.FROZEN,)
    if state == ProtocolState.FROZEN:
        return (ProtocolState.MATERIALIZED,)
    if state == ProtocolState.MATERIALIZED:
        return (ProtocolState.AUDITED,)
    if state == ProtocolState.AUDITED:
        return (ProtocolState.EVALUATED,)
    if state == ProtocolState.EVALUATED:
        return (ProtocolState.RECOVERED,) if has_recovery else (ProtocolState.REPORTED,)
    if state == ProtocolState.RECOVERED:
        return (ProtocolState.REPORTED,)
    return ()


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
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    return value


def _json_value(value: Any, label: str) -> JSONValue:
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple):
        return tuple(_json_value(item, label) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ProtocolValidationError(f"{label} must be a finite JSON scalar or tuple")


def _canonical_scalar(value: JSONScalar) -> str:
    return _canonical_json(value)


def _observation_data_type(value: Any) -> ObservationDataType:
    try:
        return ObservationDataType(value)
    except ValueError:
        accepted = tuple(item.value for item in ObservationDataType)
        raise ProtocolValidationError(
            f"observation data type {value!r} is not one of {accepted!r}"
        ) from None


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{label} must be a non-empty string")


def _names(values: tuple[str, ...], label: str, *, allow_empty: bool = False) -> None:
    if isinstance(values, str):
        raise ProtocolValidationError(f"{label} must be a tuple")
    values = tuple(values)
    if not allow_empty and not values:
        raise ProtocolValidationError(f"{label} must not be empty")
    for value in values:
        _name(value, label)
    if len(set(values)) != len(values):
        raise ProtocolValidationError(f"{label} must be unique")


def _settings(values: tuple[Setting, ...], label: str) -> None:
    if isinstance(values, Setting):
        raise ProtocolValidationError(f"{label} must be a tuple")
    names = [value.name for value in values]
    if len(set(names)) != len(names):
        raise ProtocolValidationError(f"{label} names must be unique")


def _unique_named(values: tuple[Any, ...], label: str) -> set[str]:
    names = [value.name for value in values]
    if len(set(names)) != len(names):
        raise ProtocolValidationError(f"protocol {label} must have unique names")
    return set(names)


def _unique_columned(values: tuple[Any, ...], label: str) -> set[str]:
    columns = [value.column for value in values]
    if len(set(columns)) != len(columns):
        raise ProtocolValidationError(f"protocol {label} must have unique columns")
    return set(columns)


def _fingerprint(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError:
        parsed = b""
    if len(parsed) != 32 or value != value.lower():
        raise ProtocolValidationError(f"{label} must be a lowercase SHA-256 hex digest")
