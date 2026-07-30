"""JSON Schema for the task ontology, emitted from the declarations themselves.

A vocabulary that can only be *accepted* is not owned. The records in this ontology are
consumed by tools that are not written in Python -- a static site, a CI validator, a
reviewer looking at a diff -- and those tools validate against JSON Schema. So Behavio
emits the schema rather than expecting one to be handed to it, and the enumerations in the
emitted document are read off :mod:`behavio.task.vocabulary` rather than restated here. Add
a term to an enum and the published schema gains it; there is no second list to update.

The emitted documents follow the conventions of the generated schemas they replace --
``additionalProperties: false``, a title on every property, ``anyOf`` with an explicit null
for every optional scalar, ``$defs`` for shared records -- so a consumer can drop them in
place of what it generates today. Three deliberate differences from those earlier documents
are the point of the exercise rather than an accident, and each is checked by
``tests/test_task_ontology.py``:

1. Fields with a controlled vocabulary are emitted as ``enum`` rather than as bare
   ``string``. The vocabulary already existed; it was enforced in a separate validator that
   the schema knew nothing about, so a record could be schema-valid and still meaningless.
2. ``canonical_variables`` accepts an object as well as a string, which is how a variable
   is bound to a measured column and a data type.
3. A task protocol may declare ``reward``, ``response_time`` and ``variables``, which is
   what lets a declaration produce a :class:`~behavio.task.spec.TaskSpec`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from behavio.task.ontology import (
    CANONICAL_TRIAL_REQUIRED,
    TASK_FAMILY_REQUIRED,
    TASK_PROTOCOL_REQUIRED,
    ClaimConfidence,
    ProtocolScope,
)
from behavio.task.vocabulary import (
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
)

_SCALAR_TYPES = ("boolean", "integer", "null", "number", "string")


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def _string(name: str) -> dict[str, Any]:
    return {"title": _title(name), "type": "string"}


def _date(name: str) -> dict[str, Any]:
    return {"format": "date", "title": _title(name), "type": "string"}


def _integer(name: str) -> dict[str, Any]:
    return {"title": _title(name), "type": "integer"}


def _const(name: str, value: str) -> dict[str, Any]:
    return {"const": value, "title": _title(name), "type": "string"}


def _optional(name: str, kind: str) -> dict[str, Any]:
    return {"anyOf": [{"type": kind}, {"type": "null"}], "default": None, "title": _title(name)}


def _enum(name: str, values: Iterable[StrEnum], *, default: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "enum": [member.value for member in values],
        "title": _title(name),
        "type": "string",
    }
    if default is not None:
        schema["default"] = default
    return schema


def _array(name: str, item: Mapping[str, Any]) -> dict[str, Any]:
    return {"items": dict(item), "title": _title(name), "type": "array"}


def _enum_array(name: str, values: Iterable[StrEnum]) -> dict[str, Any]:
    return _array(name, {"enum": [member.value for member in values], "type": "string"})


def _mapping(name: str) -> dict[str, Any]:
    return {"additionalProperties": True, "title": _title(name), "type": "object"}


def _ref(definition: str) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{definition}"}


def _record(title: str, properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "additionalProperties": False,
        "properties": dict(properties),
        "title": title,
    }
    if required:
        schema["required"] = list(required)
    schema["type"] = "object"
    return schema


def _reference_definition() -> dict[str, Any]:
    return _record(
        "Reference",
        {
            "citation": _string("citation"),
            "doi": _optional("doi", "string"),
            "id": _string("id"),
            "notes": _optional("notes", "string"),
            "url": _optional("url", "string"),
        },
        ("id", "citation"),
    )


def _provenance_definition() -> dict[str, Any]:
    return _record(
        "Provenance",
        {
            "created": _date("created"),
            "curators": _array("curators", {"type": "string"}),
            "source_notes": _optional("source_notes", "string"),
            "updated": _date("updated"),
        },
        ("created", "updated"),
    )


def _canonical_variable_definition() -> dict[str, Any]:
    return _record(
        "CanonicalVariable",
        {
            "allowed_values": _array("allowed_values", {"type": list(_SCALAR_TYPES)}),
            "column": _optional("column", "string"),
            "data_type": {
                "anyOf": [
                    {"enum": [item.value for item in ObservationDataType], "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "title": _title("data_type"),
            },
            "description": _optional("description", "string"),
            "name": _string("name"),
            "role": {
                "anyOf": [
                    {"enum": [item.value for item in ObservationRole], "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "title": _title("role"),
            },
            "unit": _optional("unit", "string"),
        },
        ("name",),
    )


def _reward_definition() -> dict[str, Any]:
    return _record(
        "RewardSpec",
        {
            "allow_missing": {
                "default": False,
                "title": _title("allow_missing"),
                "type": "boolean",
            },
            "column": {"default": "reward", "title": _title("column"), "type": "string"},
            "maximum": _optional("maximum", "number"),
            "minimum": _optional("minimum", "number"),
            "units": _optional("units", "string"),
        },
        (),
    )


def _response_time_definition() -> dict[str, Any]:
    return _record(
        "ResponseTimeSpec",
        {
            "column": {"default": "response_time", "title": _title("column"), "type": "string"},
            "origin": _string("origin"),
            "unit": {
                "default": "seconds",
                "enum": ["seconds", "milliseconds"],
                "title": _title("unit"),
                "type": "string",
            },
        },
        ("origin",),
    )


def canonical_trial_json_schema() -> dict[str, Any]:
    """Return the JSON Schema one harmonized trial record must satisfy."""

    return {
        "additionalProperties": False,
        "properties": {
            "block_id": _optional("block_id", "string"),
            "choice": _enum("choice", ChoiceTerm),
            "correct": _optional("correct", "boolean"),
            "dataset_id": _optional("dataset_id", "string"),
            "evidence_strength": _optional("evidence_strength", "number"),
            "evidence_units": _optional("evidence_units", "string"),
            "feedback": _enum("feedback", FeedbackTerm, default=FeedbackTerm.UNKNOWN.value),
            "prior_context": _optional("prior_context", "string"),
            "protocol_id": _string("protocol_id"),
            "response_time": _optional("response_time", "number"),
            "response_time_origin": _optional("response_time_origin", "string"),
            "reward": _optional("reward", "number"),
            "reward_units": _optional("reward_units", "string"),
            "session_id": _string("session_id"),
            "source": _mapping("source"),
            "stimulus_modality": _enum("stimulus_modality", Modality),
            "stimulus_side": _enum(
                "stimulus_side", StimulusSide, default=StimulusSide.UNKNOWN.value
            ),
            "stimulus_units": _optional("stimulus_units", "string"),
            "stimulus_value": _optional("stimulus_value", "number"),
            "subject_id": _optional("subject_id", "string"),
            "task_variables": _mapping("task_variables"),
            "training_stage": _optional("training_stage", "string"),
            "trial_index": _integer("trial_index"),
        },
        "required": list(CANONICAL_TRIAL_REQUIRED),
        "title": "CanonicalTrial",
        "type": "object",
    }


def task_family_json_schema() -> dict[str, Any]:
    """Return the JSON Schema one task family record must satisfy."""

    return {
        "$defs": {
            "CanonicalVariable": _canonical_variable_definition(),
            "Provenance": _provenance_definition(),
            "Reference": _reference_definition(),
        },
        "additionalProperties": False,
        "properties": {
            "aliases": _array("aliases", {"type": "string"}),
            "canonical_variables": _array(
                "canonical_variables",
                {"anyOf": [{"type": "string"}, _ref("CanonicalVariable")]},
            ),
            "choice_terms": _enum_array("choice_terms", ChoiceTerm),
            "common_choice_types": _enum_array("common_choice_types", ChoiceType),
            "common_response_modalities": _enum_array(
                "common_response_modalities", ResponseModality
            ),
            "curation_status": _enum("curation_status", CurationStatus),
            "description": _string("description"),
            "id": _string("id"),
            "modalities": _enum_array("modalities", Modality),
            "name": _string("name"),
            "notes": _optional("notes", "string"),
            "object_type": _const("object_type", "task_family"),
            "provenance": _ref("Provenance"),
            "references": _array("references", _ref("Reference")),
            "schema_version": _string("schema_version"),
        },
        "required": list(TASK_FAMILY_REQUIRED),
        "title": "TaskFamily",
        "type": "object",
    }


def task_protocol_json_schema() -> dict[str, Any]:
    """Return the JSON Schema one task protocol record must satisfy."""

    choice = _record(
        "ChoiceSpec",
        {
            "action_mapping": _string("action_mapping"),
            "alternatives": _array("alternatives", {"type": "string"}),
            "choice_type": _enum("choice_type", ChoiceType),
            "column": {"default": "choice", "title": _title("column"), "type": "string"},
            "notes": _optional("notes", "string"),
            "response_modalities": _enum_array("response_modalities", ResponseModality),
            "terms": _enum_array("terms", ChoiceTerm),
        },
        ("choice_type", "alternatives", "response_modalities", "action_mapping"),
    )
    stimulus = _record(
        "StimulusSpec",
        {
            "evidence_schedule": _string("evidence_schedule"),
            "evidence_type": _enum("evidence_type", EvidenceType),
            "modalities": _enum_array("modalities", Modality),
            "notes": _optional("notes", "string"),
            "units": _array("units", {"type": "string"}),
            "variables": _array("variables", {"type": "string"}),
        },
        ("modalities", "variables", "evidence_type", "evidence_schedule"),
    )
    feedback = _record(
        "FeedbackSpec",
        {
            "feedback_type": _enum("feedback_type", FeedbackType),
            "notes": _optional("notes", "string"),
            "penalty": _optional("penalty", "string"),
            "reward": _optional("reward", "string"),
        },
        ("feedback_type",),
    )
    training = _record(
        "TrainingSpec",
        {"notes": _optional("notes", "string"), "stages": _array("stages", {"type": "string"})},
        (),
    )
    phase = _record(
        "Phase",
        {
            "contingent_on": _optional("contingent_on", "string"),
            "description": _optional("description", "string"),
            "duration": _string("duration"),
            "name": _string("name"),
        },
        ("name", "duration"),
    )
    claim = _record(
        "InterpretationClaim",
        {
            "caveat": _optional("caveat", "string"),
            "confidence": _enum("confidence", ClaimConfidence),
            "label": _string("label"),
            "source": _string("source"),
        },
        ("label", "source", "confidence"),
    )
    return {
        "$defs": {
            "CanonicalVariable": _canonical_variable_definition(),
            "ChoiceSpec": choice,
            "FeedbackSpec": feedback,
            "InterpretationClaim": claim,
            "Phase": phase,
            "Provenance": _provenance_definition(),
            "Reference": _reference_definition(),
            "ResponseTimeSpec": _response_time_definition(),
            "RewardSpec": _reward_definition(),
            "StimulusSpec": stimulus,
            "TrainingSpec": training,
        },
        "additionalProperties": False,
        "properties": {
            "aliases": _array("aliases", {"type": "string"}),
            "apparatus": _array("apparatus", {"type": "string"}),
            "block_column": _optional("block_column", "string"),
            "choice": _ref("ChoiceSpec"),
            "curation_status": _enum("curation_status", CurationStatus),
            "dataset_ids": _array("dataset_ids", {"type": "string"}),
            "description": _string("description"),
            "expected_analyses": _array("expected_analyses", {"type": "string"}),
            "family_id": _string("family_id"),
            "feedback": _ref("FeedbackSpec"),
            "id": _string("id"),
            "implementation_ids": _array("implementation_ids", {"type": "string"}),
            "interpretive_claims": _array("interpretive_claims", _ref("InterpretationClaim")),
            "name": _string("name"),
            "object_type": _const("object_type", "protocol"),
            "open_questions": _array("open_questions", {"type": "string"}),
            "protocol_scope": _enum(
                "protocol_scope", ProtocolScope, default=ProtocolScope.CONCRETE.value
            ),
            "provenance": _ref("Provenance"),
            "references": _array("references", _ref("Reference")),
            "response_time": _ref("ResponseTimeSpec"),
            "reward": _ref("RewardSpec"),
            "schema_version": _string("schema_version"),
            "software": _array("software", {"type": "string"}),
            "species": _enum_array("species", Species),
            "stimulus": _ref("StimulusSpec"),
            "template_protocol_id": _optional("template_protocol_id", "string"),
            "timing": _array("timing", _ref("Phase")),
            "training": _ref("TrainingSpec"),
            "variables": _array(
                "variables", {"anyOf": [{"type": "string"}, _ref("CanonicalVariable")]}
            ),
        },
        "required": list(TASK_PROTOCOL_REQUIRED),
        "title": "Protocol",
        "type": "object",
    }


ONTOLOGY_JSON_SCHEMAS = {
    "canonical_trial": canonical_trial_json_schema,
    "task_family": task_family_json_schema,
    "protocol": task_protocol_json_schema,
}
