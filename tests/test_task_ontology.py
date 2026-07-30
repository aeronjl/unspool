"""The task ontology: one vocabulary, expressed in the structural contract that existed.

The fixtures under ``tests/fixtures/atlas`` are the curated records this vocabulary was
derived from -- seven task families, eighteen task protocols, the controlled-term file they
were validated against, and the three JSON Schemas that were generated for them -- copied
verbatim as JSON. They are here because a vocabulary is only worth having if it fits the
records that motivated it, and the only way to know is to read them.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from behavio import ChoiceSpec, ResponseTimeSpec, RewardSpec, Study, TaskSpec, fit_model
from behavio.adapters import (
    CanonicalTrialError,
    CanonicalTrialSource,
    session_order_from_appearance,
    session_order_from_column,
    session_order_from_explicit,
    study_from_canonical_trials,
)
from behavio.contracts.adapter import SessionOrderPolicy, StudyAdapter, adapter_capabilities
from behavio.models import MultinomialLogit
from behavio.protocol.schema import (
    ObservationDataType,
    ObservationRole,
    ObservationSpec,
    observations_from_task_protocol,
)
from behavio.task import (
    CONTROLLED_VOCABULARIES,
    CanonicalTrial,
    CanonicalVariable,
    ChoiceDeclaration,
    ChoiceTerm,
    ChoiceType,
    CurationStatus,
    EvidenceType,
    FeedbackDeclaration,
    FeedbackType,
    Modality,
    OntologyError,
    Provenance,
    Reference,
    ResponseModality,
    Species,
    StimulusDeclaration,
    StimulusSide,
    TaskFamily,
    TaskProtocol,
    TrialPhase,
    VocabularyError,
    canonical_trial_json_schema,
    canonical_trials,
    task_family_from_dict,
    task_family_json_schema,
    task_protocol_from_dict,
    task_protocol_json_schema,
)
from behavio.task.vocabulary import term

FIXTURES = Path(__file__).parent / "fixtures" / "atlas"


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CURATED_FAMILIES: dict[str, Any] = _load("task_families.json")
CURATED_PROTOCOLS: dict[str, Any] = _load("protocols.json")
CURATED_VOCABULARY: dict[str, list[str]] = _load("core_vocabulary.json")

#: The protocol the task brief calls representative, and the one whose alternatives are
#: already spelled in controlled terms, so it exercises the whole path end to end.
REPRESENTATIVE = "ibl-visual-decision-v1"


# --------------------------------------------------------------------------------------
# A minimal JSON Schema reader.
#
# The point of emitting a schema is that a consumer outside Python validates records with
# it, so the tests have to actually validate rather than eyeball the document. Behavio adds
# no dependency for this: the subset the emitted schemas use is small and closed.
# --------------------------------------------------------------------------------------

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def schema_errors(instance: Any, schema: dict[str, Any], root: dict[str, Any]) -> list[str]:
    """Return every way ``instance`` fails ``schema``, as readable paths."""

    return _validate(instance, schema, root, "$")


def _validate(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> list[str]:
    if "$ref" in schema:
        name = schema["$ref"].removeprefix("#/$defs/")
        return _validate(value, root["$defs"][name], root, path)
    problems: list[str] = []
    if "anyOf" in schema:
        if all(_validate(value, option, root, path) for option in schema["anyOf"]):
            problems.append(f"{path}: {value!r} matches no branch of anyOf")
        return problems
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not in {schema['enum']}")
    expected = schema.get("type")
    if expected is not None:
        allowed = tuple(
            _TYPES[name] for name in ([expected] if isinstance(expected, str) else expected)
        )
        flat = tuple(
            item for entry in allowed for item in (entry if isinstance(entry, tuple) else (entry,))
        )
        if isinstance(value, bool) and bool not in flat:
            problems.append(f"{path}: {value!r} is not of type {expected}")
        elif not isinstance(value, flat):
            problems.append(f"{path}: {value!r} is not of type {expected}")
            return problems
    if isinstance(value, dict) and "properties" in schema:
        for name in schema.get("required", ()):
            if name not in value:
                problems.append(f"{path}: missing required member {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in schema["properties"]:
                    problems.append(f"{path}: unknown member {name!r}")
        for name, item in value.items():
            if name in schema["properties"]:
                problems += _validate(item, schema["properties"][name], root, f"{path}.{name}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            problems += _validate(item, schema["items"], root, f"{path}[{index}]")
    return problems


# --------------------------------------------------------------------------------------
# The vocabulary is one closed set per field, and it agrees with what was curated.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "modalities",
        "species",
        "choice_types",
        "response_modalities",
        "evidence_types",
        "feedback_types",
        "curation_statuses",
    ],
)
def test_behavio_publishes_the_curated_vocabulary_term_for_term(key: str) -> None:
    """Every controlled set the curated records were validated against is a Behavio enum.

    Set equality, not containment: a term Behavio invented would make records it emits
    unreadable to the tooling that already exists, and a term Behavio dropped would make
    existing records unreadable to Behavio.
    """

    assert list(CONTROLLED_VOCABULARIES[key]) == CURATED_VOCABULARY[key]


def test_the_trial_level_terms_are_published_alongside_the_record_level_ones() -> None:
    """The four sets the curated vocabulary file never carried, because it had no row type."""

    assert CONTROLLED_VOCABULARIES["choice_terms"] == (
        "left",
        "right",
        "go",
        "withhold",
        "no-response",
        "unknown",
    )
    assert CONTROLLED_VOCABULARIES["feedback_terms"] == ("reward", "error", "none", "unknown")
    assert CONTROLLED_VOCABULARIES["stimulus_sides"] == ("left", "right", "none", "unknown")
    assert CONTROLLED_VOCABULARIES["observation_roles"] == ("outcome", "predictor", "auxiliary")


def test_an_unknown_term_names_the_accepted_set() -> None:
    with pytest.raises(VocabularyError, match="visual, auditory"):
        term(Modality, "none", "stimulus modality")


def test_the_measurement_vocabulary_has_one_definition_shared_by_both_layers() -> None:
    """``ObservationDataType`` was defined in the protocol schema and is now defined once.

    Identity, not equality: a canonical variable and a protocol observation must be typed
    by the same object, or the two declarations could drift apart while both still passing.
    """

    from behavio.task.vocabulary import ObservationDataType as TaskDataType
    from behavio.task.vocabulary import ObservationRole as TaskRole

    assert ObservationDataType is TaskDataType
    assert ObservationRole is TaskRole
    # The member values are the wire format, so a serialized protocol is unaffected.
    assert ObservationDataType.BINARY.value == "binary"
    assert ObservationDataType.CONTINUOUS.value == "continuous"


# --------------------------------------------------------------------------------------
# Every curated record round-trips.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CURATED_FAMILIES))
def test_every_curated_task_family_round_trips_byte_for_byte(name: str) -> None:
    record = CURATED_FAMILIES[name]

    family = task_family_from_dict(record)

    assert family.to_dict() == record
    assert task_family_from_dict(family.to_dict()) == family


@pytest.mark.parametrize("name", sorted(CURATED_PROTOCOLS))
def test_every_curated_task_protocol_round_trips_byte_for_byte(name: str) -> None:
    record = CURATED_PROTOCOLS[name]

    protocol = task_protocol_from_dict(record)

    assert protocol.to_dict() == record
    assert task_protocol_from_dict(protocol.to_dict()) == protocol


def test_a_declaration_is_content_addressed_and_its_identity_tracks_its_content() -> None:
    family = task_family_from_dict(CURATED_FAMILIES["random-dot-motion"])

    assert len(family.fingerprint) == 64
    assert family.fingerprint == task_family_from_dict(family.to_dict()).fingerprint
    widened = replace(family, aliases=(*family.aliases, "dot motion"))
    assert widened.fingerprint != family.fingerprint


def test_the_curated_records_carry_no_member_their_recorded_version_predates() -> None:
    """``0.1.0`` is the version the curated records were published at, and it is honest."""

    record = dict(CURATED_FAMILIES["random-dot-motion"])
    assert record["schema_version"] == "0.1.0"
    record["choice_terms"] = ["left", "right"]

    with pytest.raises(OntologyError, match=r"predates \['choice_terms'\]"):
        task_family_from_dict(record)

    record["schema_version"] = "0.2.0"
    assert task_family_from_dict(record).choice_terms == (ChoiceTerm.LEFT, ChoiceTerm.RIGHT)


def test_an_unknown_member_is_refused_rather_than_dropped_on_the_way_through() -> None:
    record = {**CURATED_FAMILIES["random-dot-motion"], "difficulty": "hard"}

    with pytest.raises(OntologyError, match=r"unknown members: \['difficulty'\]"):
        task_family_from_dict(record)


# --------------------------------------------------------------------------------------
# Behavio emits the schema, and the curated records validate against what it emits.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CURATED_FAMILIES))
def test_every_curated_family_validates_against_the_emitted_schema(name: str) -> None:
    schema = task_family_json_schema()

    assert schema_errors(CURATED_FAMILIES[name], schema, schema) == []


@pytest.mark.parametrize("name", sorted(CURATED_PROTOCOLS))
def test_every_curated_protocol_validates_against_the_emitted_schema(name: str) -> None:
    schema = task_protocol_json_schema()

    assert schema_errors(CURATED_PROTOCOLS[name], schema, schema) == []


def test_the_emitted_trial_schema_is_the_generated_one_plus_the_modality_vocabulary() -> None:
    """One difference, and it is the whole argument for owning the vocabulary here.

    The generated schema typed ``stimulus_modality`` as a bare string while the curated
    records were separately checked against a modality list the schema knew nothing about.
    A row could therefore be schema-valid and still name a modality that does not exist --
    and one harmoniser does emit ``"none"``. Behavio emits the enum, so the check moves into
    the document the consumer already validates with.
    """

    generated = _load("canonical_trial.schema.json")
    emitted = canonical_trial_json_schema()

    assert emitted["required"] == generated["required"]
    modality = emitted["properties"].pop("stimulus_modality")
    assert modality == {
        "enum": list(CONTROLLED_VOCABULARIES["modalities"]),
        "title": "Stimulus Modality",
        "type": "string",
    }
    assert generated["properties"].pop("stimulus_modality") == {
        "title": "Stimulus Modality",
        "type": "string",
    }
    assert emitted == generated
    assert schema_errors({"stimulus_modality": "none"}, modality, emitted) != []


def test_the_emitted_family_schema_differs_only_where_behavio_added_something() -> None:
    generated = _load("task_family.schema.json")
    emitted = task_family_json_schema()

    assert emitted["required"] == generated["required"]
    # Behavio's two additive members.
    assert emitted["properties"].pop("choice_terms")["items"]["enum"] == list(
        CONTROLLED_VOCABULARIES["choice_terms"]
    )
    assert emitted["properties"].pop("canonical_variables")["items"] == {
        "anyOf": [{"type": "string"}, {"$ref": "#/$defs/CanonicalVariable"}]
    }
    assert emitted["$defs"].pop("CanonicalVariable")["required"] == ["name"]
    generated["properties"].pop("canonical_variables")
    # Three fields tightened from bare strings to the vocabulary they were always checked
    # against, and nothing else moved.
    tightened = {"modalities", "common_choice_types", "common_response_modalities"}
    for name in tightened:
        assert "enum" in emitted["properties"].pop(name)["items"]
        assert generated["properties"].pop(name)["items"] == {"type": "string"}
    assert "enum" in emitted["properties"].pop("curation_status")
    assert generated["properties"].pop("curation_status") == {
        "title": "Curation Status",
        "type": "string",
    }
    assert emitted == generated


def test_the_emitted_protocol_schema_adds_exactly_the_members_a_task_spec_needs() -> None:
    generated = _load("protocol.schema.json")
    emitted = task_protocol_json_schema()

    assert emitted["required"] == generated["required"]
    added = {"variables", "reward", "response_time", "block_column"}
    assert added <= set(emitted["properties"])
    assert added.isdisjoint(generated["properties"])
    assert {"terms", "column"} <= set(emitted["$defs"]["ChoiceSpec"]["properties"])
    assert {"terms", "column"}.isdisjoint(generated["$defs"]["ChoiceSpec"]["properties"])
    # The published ``$defs`` names are kept, so the document is a drop-in replacement.
    assert set(generated["$defs"]) < set(emitted["$defs"])


# --------------------------------------------------------------------------------------
# The named layer produces the structural one.
# --------------------------------------------------------------------------------------


def test_declared_terms_become_a_choice_spec_with_omissions_in_the_right_place() -> None:
    declaration = ChoiceDeclaration(
        choice_type=ChoiceType.TWO_ALTERNATIVE,
        alternatives=("left", "right", "no-response"),
        response_modalities=(ResponseModality.WHEEL,),
        action_mapping="wheel turns move the grating to the centre",
    )

    choice = declaration.choice_spec()

    assert isinstance(choice, ChoiceSpec)
    assert choice.options == ("left", "right")
    # ``no-response`` is a retained trial, not a third action.
    assert choice.omission_values == ("no-response",)
    # ``withhold`` is not an omission: withholding is the correct action in go/no-go.
    go_no_go = replace(declaration, alternatives=("go", "withhold"))
    assert go_no_go.choice_spec().options == ("go", "withhold")
    assert go_no_go.choice_spec().omission_values == ()


def test_an_operational_label_without_a_declared_meaning_stops_the_derivation() -> None:
    """The gap is made visible instead of being papered over with a guess."""

    declaration = ChoiceDeclaration(
        choice_type=ChoiceType.TWO_ALTERNATIVE,
        alternatives=("direction A", "direction B"),
        response_modalities=(ResponseModality.SACCADE,),
        action_mapping="saccade to the chosen target",
    )

    assert declaration.canonical_terms == (None, None)
    assert declaration.unmapped_alternatives == ("direction A", "direction B")
    with pytest.raises(OntologyError, match="direction A"):
        declaration.choice_spec()

    mapped = replace(declaration, terms=(ChoiceTerm.LEFT, ChoiceTerm.RIGHT))
    assert mapped.choice_spec().options == ("left", "right")


def test_the_uncurated_term_can_never_become_a_coordinate() -> None:
    declaration = ChoiceDeclaration(
        choice_type=ChoiceType.TWO_ALTERNATIVE,
        alternatives=("left", "right", "unknown"),
        response_modalities=(ResponseModality.WHEEL,),
        action_mapping="x",
    )

    with pytest.raises(OntologyError, match="could not determine the choice"):
        declaration.choice_spec()


@pytest.mark.parametrize("name", sorted(CURATED_PROTOCOLS))
def test_a_curated_protocol_either_derives_a_task_spec_or_says_what_is_missing(
    name: str,
) -> None:
    """Nine of the eighteen curated protocols reach the analysis path unchanged.

    The other nine name their alternatives operationally -- ``"left port"``,
    ``"direction A"`` -- which is exactly the information a structural specification cannot
    supply, and the failure names the labels that need a term rather than inventing one.
    """

    protocol = task_protocol_from_dict(CURATED_PROTOCOLS[name])

    if protocol.choice.unmapped_alternatives:
        with pytest.raises(OntologyError, match="no controlled term is declared"):
            protocol.task_spec()
        return
    task = protocol.task_spec()
    assert isinstance(task, TaskSpec)
    assert set(task.choice.options) <= set(CONTROLLED_VOCABULARIES["choice_terms"])


def test_the_representative_protocol_derives_the_contract_a_model_is_fitted_under() -> None:
    protocol = task_protocol_from_dict(CURATED_PROTOCOLS[REPRESENTATIVE])

    task = protocol.task_spec()

    assert task.choice.column == "choice"
    assert task.choice.options == ("left", "right")
    assert task.reward is None and task.response_time is None
    assert protocol.family_identifier == "family.visual-2afc-contrast"


def test_a_declared_family_reaches_the_fitting_path_rather_than_sitting_beside_it() -> None:
    """The join: a family produces a ``TaskSpec``, and ``fit_model`` takes it unmodified."""

    family = _bound_family()
    task = family.task_spec()

    assert task.choice.options == ("left", "right")
    assert task.predictors == ("contrast",)

    fitted = fit_model(MultinomialLogit(choice=task.choice), _canonical_study(), task=task)

    assert fitted.task is task
    assert fitted.validation.n_trials == 24
    assert set(fitted.validation.counts) == {"left", "right"}


def test_a_family_that_declares_no_terms_refuses_instead_of_inventing_a_coordinate() -> None:
    family = task_family_from_dict(CURATED_FAMILIES["random-dot-motion"])

    with pytest.raises(OntologyError, match="declares no choice_terms"):
        family.task_spec()


def test_a_bound_variable_is_a_typed_observation_or_it_is_not_bound() -> None:
    assert CanonicalVariable(name="motion coherence").bound is False
    with pytest.raises(OntologyError, match="declares no role and data type"):
        CanonicalVariable(name="contrast", column="contrast")
    with pytest.raises(OntologyError, match="without a column"):
        CanonicalVariable(name="contrast", unit="percent")


# --------------------------------------------------------------------------------------
# The declaration and the protocol's column contract have one source.
# --------------------------------------------------------------------------------------


def test_the_observation_contract_is_derived_from_the_same_declaration_as_the_task() -> None:
    """``allowed_values`` used to be restated beside ``ChoiceSpec.options``; now it is not."""

    protocol = _bound_protocol()

    observations = observations_from_task_protocol(protocol)
    task = protocol.task_spec()

    by_column = {item.column: item for item in observations}
    assert all(isinstance(item, ObservationSpec) for item in observations)
    assert by_column["choice"].role is ObservationRole.OUTCOME
    assert by_column["choice"].data_type is ObservationDataType.CATEGORICAL
    # Exactly the coordinate the model is validated against, omissions included.
    assert by_column["choice"].allowed_values == (
        *task.choice.options,
        *task.choice.omission_values,
    )
    assert by_column["response_time"].unit == "seconds"
    assert by_column["reward"].unit == "uL"
    assert by_column["reward"].role is ObservationRole.AUXILIARY
    assert by_column["contrast"].data_type is ObservationDataType.CONTINUOUS
    assert by_column["block_id"].role is ObservationRole.AUXILIARY


def test_the_derived_observation_contract_is_what_the_protocol_compiler_enforces() -> None:
    from behavio.protocol.compiler import _observation_violations

    protocol = _bound_protocol()
    study = _canonical_study()

    checked = [
        observation
        for observation in observations_from_task_protocol(protocol)
        if observation.column in study.columns
    ]
    assert {item.column for item in checked} == {
        "block_id",
        "choice",
        "contrast",
        "response_time",
        "reward",
    }
    for observation in checked:
        assert _observation_violations(observation, study) == ()

    # A row outside the declared coordinate is caught by the derived contract, without the
    # option set having been restated anywhere.
    broken = Study.from_columns(
        {name: _replaced(study, name, 0, "sideways") for name in study.columns}
    )
    choice = next(item for item in checked if item.column == "choice")
    violations = _observation_violations(choice, broken)
    assert violations and violations[0].rule.value == "allowed-values"


# --------------------------------------------------------------------------------------
# Canonical trials become a study, and the vocabulary is checked rather than trusted.
# --------------------------------------------------------------------------------------


def test_canonical_trials_become_a_study_with_a_recorded_chronology_rule() -> None:
    source = CanonicalTrialSource(
        trials=canonical_trials(_canonical_records()),
        session_order=session_order_from_explicit(["s1", "s2"]),
        protocol=_bound_protocol(),
    )

    study = source.read()

    assert isinstance(source, StudyAdapter)
    assert adapter_capabilities(source).session_order_policy is SessionOrderPolicy.DERIVED
    assert study.columns[:4] == ("subject", "session", "trial", "session_order")
    assert study.subjects == ("m1",)
    assert list(study["session_order"]) == [0, 0, 1, 1]
    assert list(study["choice"]) == ["left", "right", "right", "no-response"]
    # The derivation is recorded on every trial, so a derived chronology can never be
    # mistaken for a recorded one.
    assert set(study["source_session_order_rule"]) == {"explicit"}
    # A canonical field that no record carries does not become an all-missing column.
    assert "prior_context" not in study.columns
    # ``task_variables`` are flattened, because they are per-trial observations.
    assert list(study["signed_contrast"]) == [-1.0, 1.0, 0.5, 0.0]


@pytest.mark.parametrize(
    ("derivation", "expected"),
    [
        (session_order_from_appearance(), [0, 0, 1, 1]),
        (session_order_from_explicit(["s2", "s1"]), [1, 1, 0, 0]),
        (session_order_from_column("session_date"), [0, 0, 1, 1]),
    ],
)
def test_the_three_named_chronology_rules_are_the_table_reader_s_own(
    derivation: Any, expected: list[int]
) -> None:
    """No fourth vocabulary: a canonical record lacks chronology exactly as a CSV can."""

    records = [
        {**record, "source": {"session_date": "2026-01-0" + record["session_id"][-1]}}
        for record in _canonical_records()
    ]

    study = study_from_canonical_trials(records, session_order=derivation)

    assert list(study["session_order"]) == expected


def test_chronology_cannot_be_derived_from_a_key_that_is_not_there() -> None:
    with pytest.raises(CanonicalTrialError, match="no ordering key 'session_date'"):
        study_from_canonical_trials(
            _canonical_records(), session_order=session_order_from_column("session_date")
        )


def test_a_record_without_a_subject_is_refused_rather_than_given_one() -> None:
    records = [{**record} for record in _canonical_records()]
    records[2].pop("subject_id")

    with pytest.raises(CanonicalTrialError, match="trial 2 has no subject_id"):
        study_from_canonical_trials(records, session_order=session_order_from_appearance())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"choice": "sideways"}, "canonical trial choice must be one of"),
        ({"stimulus_modality": "none"}, "stimulus_modality must be one of"),
        ({"feedback": "punishment"}, "canonical trial feedback must be one of"),
        ({"stimulus_side": "middle"}, "stimulus_side must be one of"),
    ],
)
def test_a_term_outside_the_vocabulary_fails_at_the_row_that_carries_it(
    change: dict[str, Any], message: str
) -> None:
    records = [{**record} for record in _canonical_records()]
    records[1].update(change)

    with pytest.raises(OntologyError, match=f"canonical trial 1: .*{message.split(' must')[0]}"):
        canonical_trials(records)


def test_an_uncurated_choice_is_not_a_category_a_model_may_fit() -> None:
    records = [{**record} for record in _canonical_records()]
    records[1]["choice"] = "unknown"

    with pytest.raises(CanonicalTrialError, match="trial 1: choice is 'unknown'"):
        study_from_canonical_trials(records, session_order=session_order_from_appearance())


def test_trials_are_checked_against_the_protocol_they_claim_to_belong_to() -> None:
    protocol = _bound_protocol()
    records = [{**record} for record in _canonical_records()]

    records[0]["protocol_id"] = "protocol.something-else"
    with pytest.raises(CanonicalTrialError, match=r"protocol_id 'protocol\.something-else'"):
        study_from_canonical_trials(
            records, session_order=session_order_from_appearance(), protocol=protocol
        )

    records = [{**record, "stimulus_modality": "auditory"} for record in _canonical_records()]
    with pytest.raises(CanonicalTrialError, match="is not a modality protocol"):
        study_from_canonical_trials(
            records, session_order=session_order_from_appearance(), protocol=protocol
        )

    records = [{**record} for record in _canonical_records()]
    records[0]["choice"] = "go"
    with pytest.raises(CanonicalTrialError, match="is not an alternative protocol"):
        study_from_canonical_trials(
            records, session_order=session_order_from_appearance(), protocol=protocol
        )


def test_two_scales_in_one_column_is_refused_rather_than_pooled() -> None:
    """Two harmonisers write ``uL`` and ``mL`` for the same quantity. That is not one column."""

    records = [{**record, "reward": 2.0, "reward_units": "uL"} for record in _canonical_records()]
    records[3]["reward_units"] = "mL"

    with pytest.raises(CanonicalTrialError, match="reward_units is not constant"):
        study_from_canonical_trials(records, session_order=session_order_from_appearance())


def test_a_declared_response_time_origin_is_checked_against_the_trials() -> None:
    protocol = _bound_protocol()
    records = [
        {**record, "response_time": 0.5, "response_time_origin": "stimOn_times - goCue_times"}
        for record in _canonical_records()
    ]

    with pytest.raises(CanonicalTrialError, match="response_time_origin"):
        study_from_canonical_trials(
            records, session_order=session_order_from_appearance(), protocol=protocol
        )


def test_a_task_variable_that_is_not_on_every_row_cannot_become_a_column() -> None:
    records = [{**record} for record in _canonical_records()]
    records[2]["task_variables"] = {}

    with pytest.raises(CanonicalTrialError, match="different task_variables"):
        study_from_canonical_trials(records, session_order=session_order_from_appearance())

    records = [
        {**record, "task_variables": {**record["task_variables"], "choice": "left"}}
        for record in _canonical_records()
    ]
    with pytest.raises(CanonicalTrialError, match="collide with canonical study columns"):
        study_from_canonical_trials(records, session_order=session_order_from_appearance())


def test_a_converted_study_can_be_fitted_through_the_protocol_that_declared_it() -> None:
    """The whole path: curated declaration, harmonized rows, validated study, fitted model."""

    protocol = _bound_protocol()
    records = [{**record} for record in _canonical_records()] * 6
    records = [
        {**record, "session_id": f"s{1 + index // 12}", "trial_index": index % 12}
        for index, record in enumerate(records)
    ]

    study = study_from_canonical_trials(
        records, session_order=session_order_from_appearance(), protocol=protocol
    )
    task = protocol.task_spec()
    task_without_extras = replace(task, reward=None, response_time=None, block_column=None)

    fitted = fit_model(
        MultinomialLogit(choice=task_without_extras.choice, include_omission=True),
        study,
        task=replace(task_without_extras, predictors=()),
    )

    assert fitted.validation.n_trials == 24
    # The declared omission term is retained as a modelled category, not dropped.
    assert fitted.validation.n_omissions == 6
    assert fitted.validation.counts == {"left": 6, "right": 12}


# --------------------------------------------------------------------------------------
# The ontology is optional.
# --------------------------------------------------------------------------------------


def test_a_plain_task_contract_needs_no_family_no_protocol_and_no_term() -> None:
    """The shortest correct analysis is unchanged by everything above."""

    study = Study.from_columns(
        {
            "subject": ["m1"] * 8,
            "session": ["s1"] * 8,
            "session_order": [0] * 8,
            "trial": list(range(8)),
            "choice": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    task = TaskSpec(choice=ChoiceSpec(options=(0, 1)))

    assert task.validate(study).n_trials == 8
    # The two additive members default to "nobody said", not to a fabricated value.
    assert ResponseTimeSpec().origin is None
    assert RewardSpec().units is None


def test_a_declaration_must_say_where_its_response_time_clock_started() -> None:
    """The field NWB has no place for, required exactly where a claim is being made."""

    with pytest.raises(OntologyError, match="name its origin"):
        replace(_bound_protocol(), response_time=ResponseTimeSpec(column="response_time"))
    with pytest.raises(ValueError, match="non-empty string or None"):
        ResponseTimeSpec(origin="  ")


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _bound_family() -> TaskFamily:
    curated = task_family_from_dict(CURATED_FAMILIES["visual-2afc-contrast"])
    return replace(
        curated,
        schema_version="0.2.0",
        choice_terms=(ChoiceTerm.LEFT, ChoiceTerm.RIGHT),
        canonical_variables=(
            CanonicalVariable(
                name="contrast",
                column="contrast",
                role=ObservationRole.PREDICTOR,
                data_type=ObservationDataType.CONTINUOUS,
                unit="percent contrast",
            ),
            *curated.canonical_variables[1:],
        ),
    )


def _bound_protocol() -> TaskProtocol:
    return TaskProtocol(
        identifier="protocol.ibl-visual-decision-v1",
        family_identifier="family.visual-2afc-contrast",
        name="IBL visual decision task",
        description="Head-fixed mice report the side of a visual grating with a wheel.",
        species=(Species.MOUSE,),
        curation_status=CurationStatus.ADAPTER_READY,
        stimulus=StimulusDeclaration(
            modalities=(Modality.VISUAL,),
            variables=("contrast", "stimulus side"),
            evidence_type=EvidenceType.MIXED,
            evidence_schedule="a grating appears left or right at variable contrast",
            units=("percent contrast",),
        ),
        choice=ChoiceDeclaration(
            choice_type=ChoiceType.TWO_ALTERNATIVE,
            alternatives=("left", "right", "no-response"),
            response_modalities=(ResponseModality.WHEEL,),
            action_mapping="wheel turns move the grating toward the centre",
        ),
        timing=(TrialPhase(name="stimulus_and_response", duration="response-limited"),),
        feedback=FeedbackDeclaration(feedback_type=FeedbackType.MIXED, reward="sweetened water"),
        references=(Reference(identifier="ref.ibl-2021", citation="IBL et al., eLife, 2021."),),
        provenance=Provenance(created="2026-04-24", updated="2026-07-30"),
        variables=(
            CanonicalVariable(
                name="contrast",
                column="contrast",
                role=ObservationRole.PREDICTOR,
                data_type=ObservationDataType.CONTINUOUS,
                unit="percent contrast",
            ),
        ),
        reward=RewardSpec(column="reward", minimum=0.0, allow_missing=True, units="uL"),
        response_time=ResponseTimeSpec(origin="response_times - stimOn_times"),
        block_column="block_id",
    )


def _canonical_records() -> list[dict[str, Any]]:
    rows = [
        ("s1", 0, "left", -1.0, "left", "reward"),
        ("s1", 1, "right", 1.0, "right", "error"),
        ("s2", 0, "right", 0.5, "right", "reward"),
        ("s2", 1, "no-response", 0.0, "none", "none"),
    ]
    return [
        {
            "protocol_id": "protocol.ibl-visual-decision-v1",
            "dataset_id": "dataset.ibl-public-behavior",
            "subject_id": "m1",
            "session_id": session,
            "trial_index": index,
            "stimulus_modality": "visual",
            "stimulus_value": contrast,
            "stimulus_units": "percent contrast, signed left negative",
            "stimulus_side": side,
            "choice": choice,
            "feedback": feedback,
            "response_time": 0.42,
            "response_time_origin": "response_times - stimOn_times",
            "task_variables": {"signed_contrast": contrast},
        }
        for session, index, choice, contrast, side, feedback in rows
    ]


def _canonical_study() -> Study:
    generator = np.random.default_rng(7)
    rows = 24
    return Study.from_columns(
        {
            "subject": ["m1"] * rows,
            "session": ["s1"] * 12 + ["s2"] * 12,
            "session_order": [0] * 12 + [1] * 12,
            "trial": list(range(12)) * 2,
            "choice": ["left", "right"] * 12,
            "contrast": generator.normal(size=rows),
            "reward": generator.uniform(0.0, 3.0, size=rows),
            "response_time": generator.uniform(0.2, 1.5, size=rows),
            "block_id": ["b1"] * rows,
        }
    )


def _replaced(study: Study, name: str, row: int, value: Any) -> list[Any]:
    values = list(study[name])
    if name == "choice":
        values[row] = value
    return values


def test_the_canonical_trial_record_is_a_faithful_reader_of_its_own_schema() -> None:
    schema = canonical_trial_json_schema()
    record = _canonical_records()[0]

    assert schema_errors(record, schema, schema) == []
    trial = CanonicalTrial(**record)
    assert trial.choice is ChoiceTerm.LEFT
    assert trial.stimulus_side is StimulusSide.LEFT
    assert trial.to_dict() == record
    assert schema_errors(trial.to_dict(), schema, schema) == []
