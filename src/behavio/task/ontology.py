"""Declared task families, task protocols, and the canonical trial they produce.

A :class:`~behavio.task.spec.TaskSpec` says which column of *this* table holds the choice
and what its options are spelled as. It is deliberately silent about what the task *was*:
two studies of the same experiment, coding choice ``-1``/``+1`` and ``"left"``/``"right"``,
produce two unrelated specifications and nothing connects them. This module supplies the
layer above: a :class:`TaskFamily` names a reusable experimental idea and the variables it
turns on, a :class:`TaskProtocol` is one concrete realisation of that idea, and both are
written in the closed vocabulary of :mod:`behavio.task.vocabulary`.

The two layers meet in one direction only, which is what keeps them from becoming rivals:
**a declaration produces a specification, never the other way round.**
:meth:`ChoiceDeclaration.choice_spec` turns declared terms into a
:class:`~behavio.task.spec.ChoiceSpec`; :meth:`TaskProtocol.task_spec` turns a whole
protocol into a :class:`~behavio.task.spec.TaskSpec`; and
:func:`behavio.protocol.schema.observations_from_task_protocol` turns the same declaration
into the :class:`~behavio.protocol.schema.ObservationSpec` tuple a study protocol declares.
A term therefore has one definition, a column contract has one source, and the analysis
path reached by ``fit_model`` is the same one the ontology feeds.

Declarations are versioned and content-addressed exactly as
:class:`behavio.protocol.schema.StudyProtocol` is: :attr:`TaskFamily.fingerprint` is the
SHA-256 of the declaration's canonical JSON, computed by the one writer in
:mod:`behavio._internal.declaration`.

None of this is required. ``TaskSpec(choice=ChoiceSpec(options=(0, 1)))`` remains a
complete, supported task contract, and a GLM fitted to somebody's own CSV never names a
family.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from behavio._internal.declaration import (
    JSONScalar,
    canonical_json,
    canonical_scalar,
    content_fingerprint,
    json_value,
    require_name,
    require_names,
)
from behavio.task.response_times import ResponseTimeSpec, ResponseTimeUnit
from behavio.task.spec import ChoiceSpec, RewardSpec, TaskSpec
from behavio.task.vocabulary import (
    OMISSION_CHOICE_TERMS,
    ChoiceTerm,
    ChoiceType,
    CurationStatus,
    EvidenceType,
    FeedbackTerm,
    FeedbackType,
    Modality,
    ObservationDataType,
    ObservationRole,
    ResponseModality,
    Species,
    StimulusSide,
    choice_term_of,
    term,
    terms,
)

#: Version this module writes new declarations under. ``0.1.0`` is the version the curated
#: records that seeded this vocabulary were published at; it predates every member Behavio
#: added, so a record recorded under it may not carry one. This is the same discipline
#: :mod:`behavio.protocol.schema` applies to its own superseded versions: a declaration is
#: content-addressed, so silently writing a member its author never declared would change
#: the identity of something nobody amended.
ONTOLOGY_SCHEMA_VERSION = "0.2.0"

#: Versions this module reads. ``0.1.0`` records carry no Behavio-added member.
ACCEPTED_ONTOLOGY_SCHEMA_VERSIONS = ("0.2.0", "0.1.0")


class OntologyError(ValueError):
    """Raised when a task declaration is incomplete or internally inconsistent."""


class ProtocolScope(StrEnum):
    """Whether a task protocol is an abstract template or a concrete realisation."""

    TEMPLATE = "template"
    CONCRETE = "concrete"


class ClaimConfidence(StrEnum):
    """Curator confidence in one interpretive claim about a protocol."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --------------------------------------------------------------------------------------
# Shared record types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reference:
    """One bibliographic anchor for a declaration."""

    identifier: str
    citation: str
    url: str | None = None
    doi: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        require_name(self.identifier, "reference id", error=OntologyError)
        require_name(self.citation, "reference citation", error=OntologyError)
        for value, label in (
            (self.url, "reference url"),
            (self.doi, "reference doi"),
            (self.notes, "reference notes"),
        ):
            if value is not None:
                require_name(value, label, error=OntologyError)

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "id": self.identifier,
                "citation": self.citation,
                "url": self.url,
                "doi": self.doi,
                "notes": self.notes,
            },
            required=("id", "citation"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> Reference:
        payload = _object(value, "reference", ("id", "citation", "url", "doi", "notes"), ("id",))
        return cls(
            identifier=payload["id"],
            citation=payload.get("citation", ""),
            url=payload.get("url"),
            doi=payload.get("doi"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    """Who curated a declaration and when."""

    created: str
    updated: str
    curators: tuple[str, ...] = ()
    source_notes: str | None = None

    def __post_init__(self) -> None:
        require_name(self.created, "provenance created date", error=OntologyError)
        require_name(self.updated, "provenance updated date", error=OntologyError)
        require_names(self.curators, "provenance curators", allow_empty=True, error=OntologyError)
        if self.source_notes is not None:
            require_name(self.source_notes, "provenance source notes", error=OntologyError)
        object.__setattr__(self, "curators", tuple(self.curators))

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "created": self.created,
                "updated": self.updated,
                "curators": list(self.curators),
                "source_notes": self.source_notes,
            },
            required=("created", "updated"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> Provenance:
        payload = _object(
            value,
            "provenance",
            ("created", "updated", "curators", "source_notes"),
            ("created", "updated"),
        )
        return cls(
            created=payload["created"],
            updated=payload["updated"],
            curators=tuple(payload.get("curators", ())),
            source_notes=payload.get("source_notes"),
        )


@dataclass(frozen=True, slots=True)
class TrialPhase:
    """One named phase of a trial and how long it lasts."""

    name: str
    duration: str
    description: str | None = None
    contingent_on: str | None = None

    def __post_init__(self) -> None:
        require_name(self.name, "trial phase name", error=OntologyError)
        require_name(self.duration, "trial phase duration", error=OntologyError)
        for value, label in (
            (self.description, "trial phase description"),
            (self.contingent_on, "trial phase contingency"),
        ):
            if value is not None:
                require_name(value, label, error=OntologyError)

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "name": self.name,
                "duration": self.duration,
                "description": self.description,
                "contingent_on": self.contingent_on,
            },
            required=("name", "duration"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> TrialPhase:
        payload = _object(
            value,
            "trial phase",
            ("name", "duration", "description", "contingent_on"),
            ("name", "duration"),
        )
        return cls(
            name=payload["name"],
            duration=payload["duration"],
            description=payload.get("description"),
            contingent_on=payload.get("contingent_on"),
        )


@dataclass(frozen=True, slots=True)
class InterpretationClaim:
    """One sourced cognitive reading of an operational protocol, with its caveat."""

    label: str
    source: str
    confidence: ClaimConfidence
    caveat: str | None = None

    def __post_init__(self) -> None:
        require_name(self.label, "interpretation label", error=OntologyError)
        require_name(self.source, "interpretation source", error=OntologyError)
        object.__setattr__(
            self, "confidence", term(ClaimConfidence, self.confidence, "interpretation confidence")
        )
        if self.caveat is not None:
            require_name(self.caveat, "interpretation caveat", error=OntologyError)

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "label": self.label,
                "source": self.source,
                "confidence": self.confidence.value,
                "caveat": self.caveat,
            },
            required=("label", "source", "confidence"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> InterpretationClaim:
        payload = _object(
            value,
            "interpretive claim",
            ("label", "source", "confidence", "caveat"),
            ("label", "source", "confidence"),
        )
        return cls(
            label=payload["label"],
            source=payload["source"],
            confidence=payload["confidence"],
            caveat=payload.get("caveat"),
        )


# --------------------------------------------------------------------------------------
# Canonical variables: the named layer bound to a structural one
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalVariable:
    """One variable a task family turns on, optionally bound to a measured column.

    A curated family names its variables in prose -- ``"motion coherence"``,
    ``"stimulus side"`` -- and that alone is a real contribution: it says two experiments
    manipulate the same thing. It is not enough to fit anything, because nothing says which
    column carries it or what values are legal.

    Binding supplies that. A variable with a ``column`` also declares a
    :class:`~behavio.task.vocabulary.ObservationRole` and a
    :class:`~behavio.task.vocabulary.ObservationDataType`, which is exactly the information
    :class:`behavio.protocol.schema.ObservationSpec` requires, so one declaration serves the
    analysis path and the protocol's column contract without either restating the other. An
    unbound variable stays a name, and serializes as the plain string a curated record
    already holds.
    """

    name: str
    column: str | None = None
    role: ObservationRole | None = None
    data_type: ObservationDataType | None = None
    unit: str | None = None
    allowed_values: tuple[JSONScalar, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        require_name(self.name, "canonical variable name", error=OntologyError)
        bound_members = (self.role, self.data_type, self.unit, *self.allowed_values)
        if self.column is None and any(value is not None for value in bound_members):
            raise OntologyError(
                f"canonical variable {self.name!r} declares a role, type, unit or value set "
                "without a column; bind it to a column or leave it a name"
            )
        if self.column is not None:
            require_name(self.column, "canonical variable column", error=OntologyError)
            if self.role is None or self.data_type is None:
                raise OntologyError(
                    f"canonical variable {self.name!r} is bound to column {self.column!r} but "
                    "declares no role and data type; a bound variable is a typed observation"
                )
            object.__setattr__(
                self, "role", term(ObservationRole, self.role, "canonical variable role")
            )
            object.__setattr__(
                self,
                "data_type",
                term(ObservationDataType, self.data_type, "canonical variable data type"),
            )
        if self.unit is not None:
            require_name(self.unit, "canonical variable unit", error=OntologyError)
        if self.description is not None:
            require_name(self.description, "canonical variable description", error=OntologyError)
        values = tuple(
            json_value(value, "canonical variable allowed value", error=OntologyError)
            for value in self.allowed_values
        )
        if any(isinstance(value, tuple) for value in values):
            raise OntologyError("canonical variable allowed values must be scalars")
        if len({canonical_scalar(value) for value in values}) != len(values):
            raise OntologyError("canonical variable allowed values must be unique")
        object.__setattr__(self, "allowed_values", values)

    @property
    def bound(self) -> bool:
        """Whether the variable names a measured column."""

        return self.column is not None

    def to_wire(self) -> Any:
        """Serialize as a bare name when unbound and undescribed, as an object otherwise."""

        if not self.bound and self.description is None:
            return self.name
        return _compact(
            {
                "name": self.name,
                "column": self.column,
                "role": None if self.role is None else self.role.value,
                "data_type": None if self.data_type is None else self.data_type.value,
                "unit": self.unit,
                "allowed_values": list(self.allowed_values),
                "description": self.description,
            },
            required=("name",),
        )

    @classmethod
    def from_wire(cls, value: Any) -> CanonicalVariable:
        if isinstance(value, str):
            return cls(name=value)
        payload = _object(
            value,
            "canonical variable",
            ("name", "column", "role", "data_type", "unit", "allowed_values", "description"),
            ("name",),
        )
        return cls(
            name=payload["name"],
            column=payload.get("column"),
            role=payload.get("role"),
            data_type=payload.get("data_type"),
            unit=payload.get("unit"),
            allowed_values=tuple(payload.get("allowed_values", ())),
            description=payload.get("description"),
        )


# --------------------------------------------------------------------------------------
# Protocol-level declarations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StimulusDeclaration:
    """What evidence a protocol delivers, in which modality, on what schedule."""

    modalities: tuple[Modality, ...]
    variables: tuple[str, ...]
    evidence_type: EvidenceType
    evidence_schedule: str
    units: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "modalities", terms(Modality, self.modalities, "stimulus modalities")
        )
        require_names(self.variables, "stimulus variables", allow_empty=True, error=OntologyError)
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(
            self,
            "evidence_type",
            term(EvidenceType, self.evidence_type, "stimulus evidence type"),
        )
        require_name(self.evidence_schedule, "stimulus evidence schedule", error=OntologyError)
        require_names(self.units, "stimulus units", allow_empty=True, error=OntologyError)
        object.__setattr__(self, "units", tuple(self.units))
        if self.notes is not None:
            require_name(self.notes, "stimulus notes", error=OntologyError)

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "modalities": [value.value for value in self.modalities],
                "variables": list(self.variables),
                "evidence_type": self.evidence_type.value,
                "evidence_schedule": self.evidence_schedule,
                "units": list(self.units),
                "notes": self.notes,
            },
            required=("modalities", "variables", "evidence_type", "evidence_schedule"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> StimulusDeclaration:
        payload = _object(
            value,
            "stimulus",
            ("modalities", "variables", "evidence_type", "evidence_schedule", "units", "notes"),
            ("modalities", "variables", "evidence_type", "evidence_schedule"),
        )
        return cls(
            modalities=tuple(payload["modalities"]),
            variables=tuple(payload["variables"]),
            evidence_type=payload["evidence_type"],
            evidence_schedule=payload["evidence_schedule"],
            units=tuple(payload.get("units", ())),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ChoiceDeclaration:
    """What a protocol's alternatives are called, and what each of them means.

    ``alternatives`` holds the operational labels a protocol uses -- ``"left port"``,
    ``"direction A"``, ``"withhold lick"`` -- because that is what the apparatus and the
    paper say. ``terms`` is the parallel tuple of controlled
    :class:`~behavio.task.vocabulary.ChoiceTerm` values naming what each label *means*, and
    it is the piece without which no two protocols can be compared. A label that already
    spells a term needs no declaration; anything else does, and until it is supplied the
    alternative is unmapped and :meth:`choice_spec` refuses rather than guessing.

    Nothing here re-implements :class:`~behavio.task.spec.ChoiceSpec`. This declaration
    *produces* one: the terms become the option coordinate, and the omission terms become
    its omission values, so the structural contract a model validates against is derived
    from the named one rather than written twice.
    """

    choice_type: ChoiceType
    alternatives: tuple[str, ...]
    response_modalities: tuple[ResponseModality, ...]
    action_mapping: str
    notes: str | None = None
    terms: tuple[ChoiceTerm, ...] = ()
    column: str = "choice"

    def __post_init__(self) -> None:
        object.__setattr__(self, "choice_type", term(ChoiceType, self.choice_type, "choice type"))
        require_names(self.alternatives, "choice alternatives", error=OntologyError)
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        object.__setattr__(
            self,
            "response_modalities",
            terms(ResponseModality, self.response_modalities, "choice response modalities"),
        )
        require_name(self.action_mapping, "choice action mapping", error=OntologyError)
        if self.notes is not None:
            require_name(self.notes, "choice notes", error=OntologyError)
        declared = tuple(self.terms)
        if declared:
            resolved = tuple(term(ChoiceTerm, value, "choice term") for value in declared)
            if len(resolved) != len(self.alternatives):
                raise OntologyError(
                    "choice terms must name one term per alternative; "
                    f"{len(resolved)} terms for {len(self.alternatives)} alternatives"
                )
            if len(set(resolved)) != len(resolved):
                raise OntologyError("choice terms must not repeat a term")
            object.__setattr__(self, "terms", resolved)
        require_name(self.column, "choice column", error=OntologyError)

    @property
    def canonical_terms(self) -> tuple[ChoiceTerm | None, ...]:
        """The term each alternative means, declared or spelled outright, else ``None``."""

        if self.terms:
            return self.terms
        return tuple(choice_term_of(label) for label in self.alternatives)

    @property
    def unmapped_alternatives(self) -> tuple[str, ...]:
        """Alternatives whose canonical meaning nobody has declared."""

        return tuple(
            label
            for label, mapped in zip(self.alternatives, self.canonical_terms, strict=True)
            if mapped is None
        )

    def choice_spec(self) -> ChoiceSpec:
        """Return the structural choice contract these named alternatives define."""

        unmapped = self.unmapped_alternatives
        if unmapped:
            raise OntologyError(
                "cannot derive a ChoiceSpec: no controlled term is declared for "
                f"{list(unmapped)}. Add a `terms` entry naming what each alternative means."
            )
        mapped = [value for value in self.canonical_terms if value is not None]
        if ChoiceTerm.UNKNOWN in mapped:
            raise OntologyError(
                "'unknown' records that a curator could not determine the choice; it cannot "
                "be an option coordinate"
            )
        options = tuple(value.value for value in mapped if value not in OMISSION_CHOICE_TERMS)
        omissions = tuple(value.value for value in mapped if value in OMISSION_CHOICE_TERMS)
        if len(options) < 2:
            raise OntologyError(
                "a choice coordinate needs at least two non-omission terms; "
                f"{self.choice_type.value} declares {list(options)}"
            )
        return ChoiceSpec(
            column=self.column,
            options=options,
            omission_values=omissions,
        )

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "choice_type": self.choice_type.value,
                "alternatives": list(self.alternatives),
                "response_modalities": [value.value for value in self.response_modalities],
                "action_mapping": self.action_mapping,
                "notes": self.notes,
                "terms": [value.value for value in self.terms],
                "column": None if self.column == "choice" else self.column,
            },
            required=("choice_type", "alternatives", "response_modalities", "action_mapping"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> ChoiceDeclaration:
        payload = _object(
            value,
            "choice",
            (
                "choice_type",
                "alternatives",
                "response_modalities",
                "action_mapping",
                "notes",
                "terms",
                "column",
            ),
            ("choice_type", "alternatives", "response_modalities", "action_mapping"),
        )
        return cls(
            choice_type=payload["choice_type"],
            alternatives=tuple(payload["alternatives"]),
            response_modalities=tuple(payload["response_modalities"]),
            action_mapping=payload["action_mapping"],
            notes=payload.get("notes"),
            terms=tuple(payload.get("terms", ())),
            column=payload.get("column", "choice"),
        )


@dataclass(frozen=True, slots=True)
class FeedbackDeclaration:
    """What a protocol delivers after a response, described operationally."""

    feedback_type: FeedbackType
    reward: str | None = None
    penalty: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feedback_type", term(FeedbackType, self.feedback_type, "feedback type")
        )
        for value, label in (
            (self.reward, "feedback reward description"),
            (self.penalty, "feedback penalty description"),
            (self.notes, "feedback notes"),
        ):
            if value is not None:
                require_name(value, label, error=OntologyError)

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "feedback_type": self.feedback_type.value,
                "reward": self.reward,
                "penalty": self.penalty,
                "notes": self.notes,
            },
            required=("feedback_type",),
        )

    @classmethod
    def from_dict(cls, value: Any) -> FeedbackDeclaration:
        payload = _object(
            value,
            "feedback",
            ("feedback_type", "reward", "penalty", "notes"),
            ("feedback_type",),
        )
        return cls(
            feedback_type=payload["feedback_type"],
            reward=payload.get("reward"),
            penalty=payload.get("penalty"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class TrainingDeclaration:
    """The named stages a subject was carried through before the reported sessions."""

    stages: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        require_names(self.stages, "training stages", allow_empty=True, error=OntologyError)
        object.__setattr__(self, "stages", tuple(self.stages))
        if self.notes is not None:
            require_name(self.notes, "training notes", error=OntologyError)

    def to_dict(self) -> dict[str, Any]:
        return _compact({"stages": list(self.stages), "notes": self.notes}, required=())

    @classmethod
    def from_dict(cls, value: Any) -> TrainingDeclaration:
        payload = _object(value, "training", ("stages", "notes"), ())
        return cls(stages=tuple(payload.get("stages", ())), notes=payload.get("notes"))


# --------------------------------------------------------------------------------------
# Task family
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskFamily:
    """A reusable experimental idea, its canonical variables, and its choice vocabulary.

    A family is versioned and content-addressed: :attr:`fingerprint` is the SHA-256 of
    :meth:`canonical_json`, so a family that gained a variable is a different family and
    says so, exactly as a :class:`behavio.protocol.schema.StudyProtocol` does.
    """

    identifier: str
    name: str
    description: str
    modalities: tuple[Modality, ...]
    canonical_variables: tuple[CanonicalVariable, ...]
    choice_types: tuple[ChoiceType, ...]
    curation_status: CurationStatus
    references: tuple[Reference, ...]
    provenance: Provenance
    response_modalities: tuple[ResponseModality, ...] = ()
    aliases: tuple[str, ...] = ()
    choice_terms: tuple[ChoiceTerm, ...] = ()
    notes: str | None = None
    schema_version: str = ONTOLOGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_name(self.identifier, "task family id", error=OntologyError)
        require_name(self.name, "task family name", error=OntologyError)
        require_name(self.description, "task family description", error=OntologyError)
        _schema_version(self.schema_version, "task family")
        object.__setattr__(
            self, "modalities", terms(Modality, self.modalities, "task family modalities")
        )
        object.__setattr__(
            self, "canonical_variables", _variables(self.canonical_variables, "task family")
        )
        object.__setattr__(
            self, "choice_types", terms(ChoiceType, self.choice_types, "common choice types")
        )
        object.__setattr__(
            self,
            "response_modalities",
            terms(ResponseModality, self.response_modalities, "common response modalities"),
        )
        object.__setattr__(
            self, "curation_status", term(CurationStatus, self.curation_status, "curation status")
        )
        object.__setattr__(
            self, "choice_terms", terms(ChoiceTerm, self.choice_terms, "family choice terms")
        )
        require_names(self.aliases, "task family aliases", allow_empty=True, error=OntologyError)
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "provenance", self.provenance)
        if self.notes is not None:
            require_name(self.notes, "task family notes", error=OntologyError)
        if not isinstance(self.provenance, Provenance):
            raise OntologyError("task family provenance must be a Provenance record")
        if any(not isinstance(item, Reference) for item in self.references):
            raise OntologyError("task family references must be Reference records")
        _require_version_supports(
            self.schema_version,
            "task family",
            {"choice_terms": bool(self.choice_terms)},
        )

    @property
    def fingerprint(self) -> str:
        """SHA-256 identity of the declaration."""

        return content_fingerprint(self.canonical_json())

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Every canonical variable's name, in declaration order."""

        return tuple(variable.name for variable in self.canonical_variables)

    @property
    def bound_variables(self) -> tuple[CanonicalVariable, ...]:
        """Canonical variables bound to a measured column."""

        return tuple(variable for variable in self.canonical_variables if variable.bound)

    def choice_spec(self, *, column: str = "choice") -> ChoiceSpec:
        """Return the structural choice contract this family's declared terms define."""

        if not self.choice_terms:
            raise OntologyError(
                f"task family {self.identifier!r} declares no choice_terms, so no choice "
                "coordinate can be derived. Add the controlled terms its subjects can emit."
            )
        return ChoiceDeclaration(
            choice_type=self.choice_types[0] if self.choice_types else ChoiceType.MIXED,
            alternatives=tuple(value.value for value in self.choice_terms),
            response_modalities=self.response_modalities,
            action_mapping=f"declared by task family {self.identifier}",
            terms=self.choice_terms,
            column=column,
        ).choice_spec()

    def task_spec(self, *, column: str = "choice") -> TaskSpec:
        """Derive the task contract a model is fitted under from this declaration.

        This is the join between the ontology and the analysis path. The choice coordinate
        comes from :attr:`choice_terms`, the predictors from the canonical variables that
        were bound to columns, and the result is an ordinary
        :class:`~behavio.task.spec.TaskSpec` -- the same object ``fit_model`` already takes.
        """

        return TaskSpec(
            choice=self.choice_spec(column=column),
            predictors=_predictor_columns(self.bound_variables, column),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the declaration, omitting every member left at its default."""

        return _compact(
            {
                "object_type": "task_family",
                "schema_version": self.schema_version,
                "id": self.identifier,
                "name": self.name,
                "aliases": list(self.aliases),
                "description": self.description,
                "modalities": [value.value for value in self.modalities],
                "canonical_variables": [
                    variable.to_wire() for variable in self.canonical_variables
                ],
                "common_choice_types": [value.value for value in self.choice_types],
                "common_response_modalities": [value.value for value in self.response_modalities],
                "choice_terms": [value.value for value in self.choice_terms],
                "curation_status": self.curation_status.value,
                "references": [item.to_dict() for item in self.references],
                "provenance": self.provenance.to_dict(),
                "notes": self.notes,
            },
            required=TASK_FAMILY_REQUIRED,
        )

    def canonical_json(self) -> str:
        """Serialize the declaration deterministically."""

        return canonical_json(self.to_dict())


TASK_FAMILY_REQUIRED = (
    "object_type",
    "schema_version",
    "id",
    "name",
    "description",
    "modalities",
    "canonical_variables",
    "common_choice_types",
    "curation_status",
    "references",
    "provenance",
)

_FAMILY_FIELDS = (
    "object_type",
    "schema_version",
    "id",
    "name",
    "aliases",
    "description",
    "modalities",
    "canonical_variables",
    "common_choice_types",
    "common_response_modalities",
    "choice_terms",
    "curation_status",
    "references",
    "provenance",
    "notes",
)


def task_family_from_dict(value: Any) -> TaskFamily:
    """Reconstruct a task family from a plain record, re-running every invariant."""

    payload = _object(value, "task family", _FAMILY_FIELDS, TASK_FAMILY_REQUIRED)
    if payload["object_type"] != "task_family":
        raise OntologyError(f"object_type must be 'task_family'; got {payload['object_type']!r}")
    return TaskFamily(
        identifier=payload["id"],
        name=payload["name"],
        description=payload["description"],
        modalities=tuple(payload["modalities"]),
        canonical_variables=tuple(
            CanonicalVariable.from_wire(item) for item in payload["canonical_variables"]
        ),
        choice_types=tuple(payload["common_choice_types"]),
        curation_status=payload["curation_status"],
        references=tuple(Reference.from_dict(item) for item in payload["references"]),
        provenance=Provenance.from_dict(payload["provenance"]),
        response_modalities=tuple(payload.get("common_response_modalities", ())),
        aliases=tuple(payload.get("aliases", ())),
        choice_terms=tuple(payload.get("choice_terms", ())),
        notes=payload.get("notes"),
        schema_version=payload["schema_version"],
    )


# --------------------------------------------------------------------------------------
# Task protocol
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskProtocol:
    """One concrete realisation of a task family, in the same controlled vocabulary.

    A task protocol is what a canonical trial's ``protocol_id`` points at. It is not a
    :class:`behavio.protocol.schema.StudyProtocol`: that declares an *analysis* -- cohort,
    estimands, candidates, comparison -- whereas this declares the *experiment* the data
    came out of. The two meet at
    :func:`behavio.protocol.schema.observations_from_task_protocol`, which turns this
    declaration into the observation contract a study protocol enforces.
    """

    identifier: str
    family_identifier: str
    name: str
    description: str
    species: tuple[Species, ...]
    curation_status: CurationStatus
    stimulus: StimulusDeclaration
    choice: ChoiceDeclaration
    timing: tuple[TrialPhase, ...]
    feedback: FeedbackDeclaration
    references: tuple[Reference, ...]
    provenance: Provenance
    scope: ProtocolScope = ProtocolScope.CONCRETE
    template_identifier: str | None = None
    aliases: tuple[str, ...] = ()
    training: TrainingDeclaration | None = None
    apparatus: tuple[str, ...] = ()
    software: tuple[str, ...] = ()
    dataset_identifiers: tuple[str, ...] = ()
    implementation_identifiers: tuple[str, ...] = ()
    expected_analyses: tuple[str, ...] = ()
    interpretive_claims: tuple[InterpretationClaim, ...] = ()
    open_questions: tuple[str, ...] = ()
    variables: tuple[CanonicalVariable, ...] = ()
    reward: RewardSpec | None = None
    response_time: ResponseTimeSpec | None = None
    block_column: str | None = None
    schema_version: str = ONTOLOGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_name(self.identifier, "task protocol id", error=OntologyError)
        require_name(self.family_identifier, "task protocol family id", error=OntologyError)
        require_name(self.name, "task protocol name", error=OntologyError)
        require_name(self.description, "task protocol description", error=OntologyError)
        _schema_version(self.schema_version, "task protocol")
        object.__setattr__(self, "species", terms(Species, self.species, "protocol species"))
        object.__setattr__(
            self, "curation_status", term(CurationStatus, self.curation_status, "curation status")
        )
        object.__setattr__(self, "scope", term(ProtocolScope, self.scope, "protocol scope"))
        for value, label, kind in (
            (self.stimulus, "stimulus", StimulusDeclaration),
            (self.choice, "choice", ChoiceDeclaration),
            (self.feedback, "feedback", FeedbackDeclaration),
            (self.provenance, "provenance", Provenance),
        ):
            if not isinstance(value, kind):
                raise OntologyError(f"task protocol {label} must be a {kind.__name__}")
        if self.training is not None and not isinstance(self.training, TrainingDeclaration):
            raise OntologyError("task protocol training must be a TrainingDeclaration")
        object.__setattr__(self, "timing", tuple(self.timing))
        if any(not isinstance(item, TrialPhase) for item in self.timing):
            raise OntologyError("task protocol timing must contain TrialPhase records")
        object.__setattr__(self, "references", tuple(self.references))
        if any(not isinstance(item, Reference) for item in self.references):
            raise OntologyError("task protocol references must be Reference records")
        object.__setattr__(self, "interpretive_claims", tuple(self.interpretive_claims))
        if any(not isinstance(item, InterpretationClaim) for item in self.interpretive_claims):
            raise OntologyError("interpretive claims must be InterpretationClaim records")
        for name in (
            "aliases",
            "apparatus",
            "software",
            "dataset_identifiers",
            "implementation_identifiers",
            "expected_analyses",
            "open_questions",
        ):
            values = tuple(getattr(self, name))
            require_names(values, f"task protocol {name}", allow_empty=True, error=OntologyError)
            object.__setattr__(self, name, values)
        if self.template_identifier is not None:
            require_name(self.template_identifier, "template protocol id", error=OntologyError)
        object.__setattr__(self, "variables", _variables(self.variables, "task protocol"))
        if self.reward is not None and not isinstance(self.reward, RewardSpec):
            raise OntologyError("task protocol reward must be a RewardSpec")
        if self.response_time is not None:
            if not isinstance(self.response_time, ResponseTimeSpec):
                raise OntologyError("task protocol response_time must be a ResponseTimeSpec")
            if self.response_time.origin is None:
                raise OntologyError(
                    "a declared response time must name its origin: the event the clock "
                    "starts at. This is the fact a trials table does not carry and a "
                    "downstream reader cannot recover."
                )
        if self.block_column is not None:
            require_name(self.block_column, "task protocol block column", error=OntologyError)
        _require_version_supports(
            self.schema_version,
            "task protocol",
            {
                "variables": bool(self.variables),
                "reward": self.reward is not None,
                "response_time": self.response_time is not None,
                "block_column": self.block_column is not None,
                "choice.terms": bool(self.choice.terms),
                "choice.column": self.choice.column != "choice",
            },
        )

    @property
    def fingerprint(self) -> str:
        """SHA-256 identity of the declaration."""

        return content_fingerprint(self.canonical_json())

    @property
    def bound_variables(self) -> tuple[CanonicalVariable, ...]:
        """Canonical variables bound to a measured column."""

        return tuple(variable for variable in self.variables if variable.bound)

    def task_spec(self) -> TaskSpec:
        """Derive the task contract a model is fitted under from this declaration."""

        choice = self.choice.choice_spec()
        return TaskSpec(
            choice=choice,
            predictors=_predictor_columns(self.bound_variables, choice.column),
            reward=self.reward,
            response_time=self.response_time,
            block_column=self.block_column,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the declaration, omitting every member left at its default."""

        return _compact(
            {
                "object_type": "protocol",
                "schema_version": self.schema_version,
                "id": self.identifier,
                "family_id": self.family_identifier,
                # The scope is emitted whenever it is not the plain default *or* the record
                # names a template. A protocol that points at a template is placing itself
                # in a hierarchy, and a hierarchy is only readable if both ends say which
                # side of the template/concrete distinction they are on; leaving the
                # concrete end implicit there would make the pair harder to read than
                # either record alone.
                "protocol_scope": (
                    self.scope.value
                    if self.scope is not ProtocolScope.CONCRETE
                    or self.template_identifier is not None
                    else None
                ),
                "template_protocol_id": self.template_identifier,
                "name": self.name,
                "aliases": list(self.aliases),
                "description": self.description,
                "species": [value.value for value in self.species],
                "curation_status": self.curation_status.value,
                "stimulus": self.stimulus.to_dict(),
                "choice": self.choice.to_dict(),
                "timing": [item.to_dict() for item in self.timing],
                "feedback": self.feedback.to_dict(),
                "training": None if self.training is None else self.training.to_dict(),
                "apparatus": list(self.apparatus),
                "software": list(self.software),
                "dataset_ids": list(self.dataset_identifiers),
                "implementation_ids": list(self.implementation_identifiers),
                "expected_analyses": list(self.expected_analyses),
                "interpretive_claims": [item.to_dict() for item in self.interpretive_claims],
                "variables": [variable.to_wire() for variable in self.variables],
                "reward": None if self.reward is None else _reward_to_dict(self.reward),
                "response_time": (
                    None
                    if self.response_time is None
                    else _response_time_to_dict(self.response_time)
                ),
                "block_column": self.block_column,
                "references": [item.to_dict() for item in self.references],
                "provenance": self.provenance.to_dict(),
                "open_questions": list(self.open_questions),
            },
            required=TASK_PROTOCOL_REQUIRED,
        )

    def canonical_json(self) -> str:
        """Serialize the declaration deterministically."""

        return canonical_json(self.to_dict())


TASK_PROTOCOL_REQUIRED = (
    "object_type",
    "schema_version",
    "id",
    "family_id",
    "name",
    "description",
    "species",
    "curation_status",
    "stimulus",
    "choice",
    "timing",
    "feedback",
    "references",
    "provenance",
)

_PROTOCOL_FIELDS = (
    "object_type",
    "schema_version",
    "id",
    "family_id",
    "protocol_scope",
    "template_protocol_id",
    "name",
    "aliases",
    "description",
    "species",
    "curation_status",
    "stimulus",
    "choice",
    "timing",
    "feedback",
    "training",
    "apparatus",
    "software",
    "dataset_ids",
    "implementation_ids",
    "expected_analyses",
    "interpretive_claims",
    "variables",
    "reward",
    "response_time",
    "block_column",
    "references",
    "provenance",
    "open_questions",
)


def task_protocol_from_dict(value: Any) -> TaskProtocol:
    """Reconstruct a task protocol from a plain record, re-running every invariant."""

    payload = _object(value, "task protocol", _PROTOCOL_FIELDS, TASK_PROTOCOL_REQUIRED)
    if payload["object_type"] != "protocol":
        raise OntologyError(f"object_type must be 'protocol'; got {payload['object_type']!r}")
    training = payload.get("training")
    return TaskProtocol(
        identifier=payload["id"],
        family_identifier=payload["family_id"],
        name=payload["name"],
        description=payload["description"],
        species=tuple(payload["species"]),
        curation_status=payload["curation_status"],
        stimulus=StimulusDeclaration.from_dict(payload["stimulus"]),
        choice=ChoiceDeclaration.from_dict(payload["choice"]),
        timing=tuple(TrialPhase.from_dict(item) for item in payload["timing"]),
        feedback=FeedbackDeclaration.from_dict(payload["feedback"]),
        references=tuple(Reference.from_dict(item) for item in payload["references"]),
        provenance=Provenance.from_dict(payload["provenance"]),
        scope=payload.get("protocol_scope", ProtocolScope.CONCRETE),
        template_identifier=payload.get("template_protocol_id"),
        aliases=tuple(payload.get("aliases", ())),
        training=None if training is None else TrainingDeclaration.from_dict(training),
        apparatus=tuple(payload.get("apparatus", ())),
        software=tuple(payload.get("software", ())),
        dataset_identifiers=tuple(payload.get("dataset_ids", ())),
        implementation_identifiers=tuple(payload.get("implementation_ids", ())),
        expected_analyses=tuple(payload.get("expected_analyses", ())),
        interpretive_claims=tuple(
            InterpretationClaim.from_dict(item) for item in payload.get("interpretive_claims", ())
        ),
        open_questions=tuple(payload.get("open_questions", ())),
        variables=tuple(CanonicalVariable.from_wire(item) for item in payload.get("variables", ())),
        reward=_reward_from_dict(payload.get("reward")),
        response_time=_response_time_from_dict(payload.get("response_time")),
        block_column=payload.get("block_column"),
        schema_version=payload["schema_version"],
    )


# --------------------------------------------------------------------------------------
# The canonical trial record
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalTrial:
    """One harmonized trial, written entirely in the controlled vocabulary.

    This is the interchange row: whatever a source called its columns, a canonical trial
    says the subject chose ``left``, the feedback was ``error``, and the response time was
    measured from the event named in :attr:`response_time_origin`. That last field is the
    one a trials table almost never carries and no reader can recover, and it is why the
    record type is worth having at all rather than being replaced by a bare table.

    Every controlled field is validated on construction, so a record whose modality is not
    a :class:`~behavio.task.vocabulary.Modality` fails here rather than reaching a model.
    """

    protocol_id: str
    session_id: str
    trial_index: int
    stimulus_modality: Modality
    choice: ChoiceTerm
    dataset_id: str | None = None
    subject_id: str | None = None
    stimulus_value: float | None = None
    stimulus_units: str | None = None
    stimulus_side: StimulusSide = StimulusSide.UNKNOWN
    evidence_strength: float | None = None
    evidence_units: str | None = None
    correct: bool | None = None
    response_time: float | None = None
    response_time_origin: str | None = None
    feedback: FeedbackTerm = FeedbackTerm.UNKNOWN
    reward: float | None = None
    reward_units: str | None = None
    block_id: str | None = None
    prior_context: str | None = None
    training_stage: str | None = None
    task_variables: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_name(self.protocol_id, "canonical trial protocol_id", error=OntologyError)
        require_name(self.session_id, "canonical trial session_id", error=OntologyError)
        if isinstance(self.trial_index, bool) or not isinstance(self.trial_index, int):
            raise OntologyError("canonical trial trial_index must be an integer")
        object.__setattr__(
            self,
            "stimulus_modality",
            term(Modality, self.stimulus_modality, "canonical trial stimulus_modality"),
        )
        object.__setattr__(self, "choice", term(ChoiceTerm, self.choice, "canonical trial choice"))
        object.__setattr__(
            self,
            "stimulus_side",
            term(StimulusSide, self.stimulus_side, "canonical trial stimulus_side"),
        )
        object.__setattr__(
            self, "feedback", term(FeedbackTerm, self.feedback, "canonical trial feedback")
        )
        for name in (
            "dataset_id",
            "subject_id",
            "stimulus_units",
            "evidence_units",
            "response_time_origin",
            "reward_units",
            "block_id",
            "prior_context",
            "training_stage",
        ):
            value = getattr(self, name)
            if value is not None:
                require_name(value, f"canonical trial {name}", error=OntologyError)
        for name in ("stimulus_value", "evidence_strength", "response_time", "reward"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OntologyError(f"canonical trial {name} must be a number")
            object.__setattr__(self, name, float(value))
        if self.correct is not None and not isinstance(self.correct, bool):
            raise OntologyError("canonical trial correct must be boolean")
        for name in ("task_variables", "source"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise OntologyError(f"canonical trial {name} must be a mapping")
            if any(not isinstance(key, str) or not key for key in value):
                raise OntologyError(f"canonical trial {name} keys must be non-empty strings")
            object.__setattr__(self, name, MappingProxyType(dict(value)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trial, omitting every member left at its default."""

        return _compact(
            {
                "protocol_id": self.protocol_id,
                "dataset_id": self.dataset_id,
                "subject_id": self.subject_id,
                "session_id": self.session_id,
                "trial_index": self.trial_index,
                "stimulus_modality": self.stimulus_modality.value,
                "stimulus_value": self.stimulus_value,
                "stimulus_units": self.stimulus_units,
                "stimulus_side": (
                    None if self.stimulus_side is StimulusSide.UNKNOWN else self.stimulus_side.value
                ),
                "evidence_strength": self.evidence_strength,
                "evidence_units": self.evidence_units,
                "choice": self.choice.value,
                "correct": self.correct,
                "response_time": self.response_time,
                "response_time_origin": self.response_time_origin,
                "feedback": (
                    None if self.feedback is FeedbackTerm.UNKNOWN else self.feedback.value
                ),
                "reward": self.reward,
                "reward_units": self.reward_units,
                "block_id": self.block_id,
                "prior_context": self.prior_context,
                "training_stage": self.training_stage,
                "task_variables": dict(self.task_variables),
                "source": dict(self.source),
            },
            required=CANONICAL_TRIAL_REQUIRED,
        )


CANONICAL_TRIAL_REQUIRED = (
    "protocol_id",
    "session_id",
    "trial_index",
    "stimulus_modality",
    "choice",
)

_CANONICAL_TRIAL_FIELDS = tuple(item.name for item in fields(CanonicalTrial))


def canonical_trial_from_dict(value: Any) -> CanonicalTrial:
    """Reconstruct one canonical trial from a plain record, validating every term."""

    payload = _object(value, "canonical trial", _CANONICAL_TRIAL_FIELDS, CANONICAL_TRIAL_REQUIRED)
    known = {name: payload[name] for name in _CANONICAL_TRIAL_FIELDS if name in payload}
    return CanonicalTrial(**known)


def canonical_trials(records: Iterable[Mapping[str, Any]]) -> tuple[CanonicalTrial, ...]:
    """Validate a sequence of canonical trial records, naming the row that fails."""

    trials: list[CanonicalTrial] = []
    for index, record in enumerate(records):
        try:
            trials.append(canonical_trial_from_dict(record))
        except ValueError as error:
            raise OntologyError(f"canonical trial {index}: {error}") from error
    if not trials:
        raise OntologyError("at least one canonical trial is required")
    return tuple(trials)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _compact(values: Mapping[str, Any], *, required: Sequence[str]) -> dict[str, Any]:
    """Drop every optional member left at its default, preserving declaration order.

    A curated record omits what it does not say. Emitting ``"notes": null`` beside a record
    that never mentioned notes would make a round trip inexact and would change a
    content-addressed declaration's identity for no scientific reason.
    """

    keep = set(required)
    return {
        name: value
        for name, value in values.items()
        if name in keep or not (value is None or value == [] or value == {})
    }


def _object(
    value: Any,
    label: str,
    known: Sequence[str],
    required: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OntologyError(f"{label} must be a JSON object")
    unknown = sorted(set(value) - set(known))
    if unknown:
        raise OntologyError(f"{label} has unknown members: {unknown}")
    missing = sorted(set(required) - set(value))
    if missing:
        raise OntologyError(f"{label} is missing required members: {missing}")
    return value


def _schema_version(value: Any, label: str) -> None:
    require_name(value, f"{label} schema_version", error=OntologyError)
    if value not in ACCEPTED_ONTOLOGY_SCHEMA_VERSIONS:
        raise OntologyError(
            f"{label} schema_version {value!r} is not one of "
            f"{list(ACCEPTED_ONTOLOGY_SCHEMA_VERSIONS)}"
        )


def _require_version_supports(version: str, label: str, present: Mapping[str, bool]) -> None:
    if version != "0.1.0":
        return
    declared = sorted(name for name, used in present.items() if used)
    if declared:
        raise OntologyError(
            f"{label} schema_version '0.1.0' predates {declared}; record it under "
            f"{ONTOLOGY_SCHEMA_VERSION!r} to declare them"
        )


def _variables(values: Any, label: str) -> tuple[CanonicalVariable, ...]:
    resolved = tuple(
        value if isinstance(value, CanonicalVariable) else CanonicalVariable.from_wire(value)
        for value in values
    )
    names = [variable.name for variable in resolved]
    if len(set(names)) != len(names):
        raise OntologyError(f"{label} canonical variable names must be unique")
    columns = [variable.column for variable in resolved if variable.column is not None]
    if len(set(columns)) != len(columns):
        raise OntologyError(f"{label} canonical variables must bind distinct columns")
    return resolved


def _predictor_columns(
    variables: Sequence[CanonicalVariable], choice_column: str
) -> tuple[str, ...]:
    return tuple(
        variable.column
        for variable in variables
        if variable.role is ObservationRole.PREDICTOR
        and variable.column is not None
        and variable.column != choice_column
    )


def _reward_to_dict(spec: RewardSpec) -> dict[str, Any]:
    return _compact(
        {
            "column": spec.column,
            "minimum": spec.minimum,
            "maximum": spec.maximum,
            "allow_missing": spec.allow_missing or None,
            "units": spec.units,
        },
        required=("column",),
    )


def _reward_from_dict(value: Any) -> RewardSpec | None:
    if value is None:
        return None
    payload = _object(
        value, "reward", ("column", "minimum", "maximum", "allow_missing", "units"), ("column",)
    )
    return RewardSpec(
        column=payload["column"],
        minimum=payload.get("minimum"),
        maximum=payload.get("maximum"),
        allow_missing=bool(payload.get("allow_missing", False)),
        units=payload.get("units"),
    )


def _response_time_to_dict(spec: ResponseTimeSpec) -> dict[str, Any]:
    return {"column": spec.column, "unit": spec.unit.value, "origin": spec.origin}


def _response_time_from_dict(value: Any) -> ResponseTimeSpec | None:
    if value is None:
        return None
    payload = _object(value, "response_time", ("column", "unit", "origin"), ("origin",))
    return ResponseTimeSpec(
        column=payload.get("column", "response_time"),
        unit=ResponseTimeUnit(payload.get("unit", ResponseTimeUnit.SECONDS)),
        origin=payload["origin"],
    )
