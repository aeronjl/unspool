"""The controlled vocabulary of behavioural tasks: one closed set per named field.

This module is the *named* half of the task layer. :mod:`behavio.task.spec` is the
*structural* half: :class:`~behavio.task.spec.ChoiceSpec` says how a column is coded, which
option sits at coordinate 0, and which values are omissions, but it takes those options as
arbitrary labels because a source table is entitled to spell a leftward wheel turn ``-1``,
``"left port"``, or ``"turn_ccw"``. What it cannot say is that all three *mean the same
thing*, and without that no two datasets can be compared.

A term here supplies the missing half. :class:`ChoiceTerm` names the six things a choice can
be across every task Behavio has read; a declaration in :mod:`behavio.task.ontology` binds
one term to one source label; and :meth:`~behavio.task.ontology.ChoiceDeclaration.choice_spec`
turns the bound terms back into a :class:`~behavio.task.spec.ChoiceSpec`. The named layer is
therefore expressible in the structural one, and neither duplicates the other.

Two enums that used to live in :mod:`behavio.protocol.schema` are defined here now.
:class:`ObservationRole` and :class:`ObservationDataType` are the measurement vocabulary a
declared column is typed with, and a study protocol was not the only thing that needed
them: a canonical task variable is typed the same way, and having two spellings of
"this column holds a count" is exactly the defect this layer exists to remove. Their member
values are unchanged, so every protocol Behavio has ever serialized keeps its fingerprint,
and ``behavio.protocol.schema`` re-exports both under their original names.

:data:`CONTROLLED_VOCABULARIES` is the machine-readable index of every closed set, keyed by
the name each set is published under. It is derived from the enums rather than written
beside them, so a term is added in exactly one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeVar


class VocabularyError(ValueError):
    """Raised when a value is not a member of the closed set its field declares."""


class Modality(StrEnum):
    """Sensory modality a task's evidence is delivered in."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    SOMATOSENSORY = "somatosensory"
    VESTIBULAR = "vestibular"
    OLFACTORY = "olfactory"
    MULTISENSORY = "multisensory"


class Species(StrEnum):
    """Subject species a protocol was run in."""

    MOUSE = "mouse"
    RAT = "rat"
    NON_HUMAN_PRIMATE = "non-human-primate"
    HUMAN = "human"


class ChoiceType(StrEnum):
    """The response structure a task asks of its subject."""

    TWO_ALTERNATIVE = "2afc"
    GO_NO_GO = "go-no-go"
    DETECTION = "detection"
    DISCRIMINATION = "discrimination"
    MIXED = "mixed"
    CONFIDENCE_REPORT = "confidence-report"
    WAGERING = "wagering"
    RATING = "rating"
    FREE_RESPONSE = "free-response"


class ResponseModality(StrEnum):
    """The effector a response is made with."""

    WHEEL = "wheel"
    LEVER = "lever"
    LICK = "lick"
    SACCADE = "saccade"
    BUTTON_PRESS = "button-press"
    TOUCH = "touch"
    NOSE_POKE = "nose-poke"
    VERBAL = "verbal"
    JOYSTICK = "joystick"
    KEYBOARD = "keyboard"


class EvidenceType(StrEnum):
    """How a task's sensory evidence is scheduled over a trial."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    PULSE_TRAIN = "pulse-train"
    STOCHASTIC_MOTION = "stochastic-motion"
    STAIRCASE = "staircase"
    ADAPTIVE = "adaptive"
    BLOCK_PRIOR = "block-prior"
    CHANGE_DETECTION = "change-detection"
    MIXED = "mixed"


class FeedbackType(StrEnum):
    """What a protocol delivers after a response, at the protocol level."""

    REWARD = "reward"
    PUNISHMENT = "punishment"
    TIMEOUT = "timeout"
    AUDITORY_FEEDBACK = "auditory-feedback"
    VISUAL_FEEDBACK = "visual-feedback"
    NONE = "none"
    MIXED = "mixed"


class CurationStatus(StrEnum):
    """How far a declaration has been carried, from a stub to an expert review."""

    STUB = "stub"
    LITERATURE_CURATED = "literature-curated"
    SCHEMA_COMPLETE = "schema-complete"
    DATA_LINKED = "data-linked"
    ADAPTER_READY = "adapter-ready"
    HARMONIZED = "harmonized"
    ANALYSIS_VERIFIED = "analysis-verified"
    EXPERT_REVIEWED = "expert-reviewed"


class ChoiceTerm(StrEnum):
    """What a subject did on one trial, named rather than coded.

    :attr:`NO_RESPONSE` is a *retained* trial on which no action was taken; it is the one
    member :data:`OMISSION_CHOICE_TERMS` maps onto ``ChoiceSpec.omission_values``, so a
    declaration that admits it keeps those trials in the denominator instead of dropping
    them. :attr:`WITHHOLD` is different and is deliberately not an omission: withholding is
    the correct action in a go/no-go task, so it occupies a coordinate of its own.

    :attr:`UNKNOWN` records that a curator could not determine the choice. It is never a
    valid observation: a conversion that meets it fails rather than inventing a coordinate
    for "we do not know", because a model fitted to that category would be fitting the
    curation process.
    """

    LEFT = "left"
    RIGHT = "right"
    GO = "go"
    WITHHOLD = "withhold"
    NO_RESPONSE = "no-response"
    UNKNOWN = "unknown"


class FeedbackTerm(StrEnum):
    """What the subject received after one trial's response."""

    REWARD = "reward"
    ERROR = "error"
    NONE = "none"
    UNKNOWN = "unknown"


class StimulusSide(StrEnum):
    """Which side one trial's stimulus was presented on."""

    LEFT = "left"
    RIGHT = "right"
    NONE = "none"
    UNKNOWN = "unknown"


class ObservationRole(StrEnum):
    """How a declared column may be used by a study."""

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
    would not.
    """

    BINARY = "binary"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    COUNT = "count"
    ORDINAL = "ordinal"


#: Choice terms that name a retained trial with no action, rather than an action.
OMISSION_CHOICE_TERMS: tuple[ChoiceTerm, ...] = (ChoiceTerm.NO_RESPONSE,)

#: Terms that record the absence of curation rather than an observation. A conversion that
#: meets one of these fails: "unknown" is a statement about the record, not about the trial.
UNCURATED_TERMS: tuple[StrEnum, ...] = (
    ChoiceTerm.UNKNOWN,
    FeedbackTerm.UNKNOWN,
    StimulusSide.UNKNOWN,
)

#: Every closed set, keyed by the name it is published under. Consumers that validate
#: records against a vocabulary read this rather than re-listing members, so a term exists
#: in exactly one place: its enum.
CONTROLLED_VOCABULARIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        name: tuple(member.value for member in enum)
        for name, enum in (
            ("modalities", Modality),
            ("species", Species),
            ("choice_types", ChoiceType),
            ("response_modalities", ResponseModality),
            ("evidence_types", EvidenceType),
            ("feedback_types", FeedbackType),
            ("curation_statuses", CurationStatus),
            ("choice_terms", ChoiceTerm),
            ("feedback_terms", FeedbackTerm),
            ("stimulus_sides", StimulusSide),
            ("observation_roles", ObservationRole),
            ("observation_data_types", ObservationDataType),
        )
    }
)

_Term = TypeVar("_Term", bound=StrEnum)


def term(vocabulary: type[_Term], value: Any, label: str) -> _Term:
    """Return the vocabulary member ``value`` names, or fail listing the accepted set."""

    try:
        return vocabulary(value)
    except ValueError:
        accepted = ", ".join(member.value for member in vocabulary)
        raise VocabularyError(f"{label} must be one of {accepted}; got {value!r}") from None


def terms(vocabulary: type[_Term], values: Any, label: str) -> tuple[_Term, ...]:
    """Return a tuple of unique vocabulary members, preserving declaration order."""

    if isinstance(values, str):
        raise VocabularyError(f"{label} must be a sequence of terms, not a single string")
    resolved = tuple(term(vocabulary, value, label) for value in values)
    if len(set(resolved)) != len(resolved):
        raise VocabularyError(f"{label} must not repeat a term")
    return resolved


def choice_term_of(label: str) -> ChoiceTerm | None:
    """Return the choice term a source label spells outright, or ``None``.

    Concrete protocols name their alternatives operationally -- ``"left port"``,
    ``"direction A"``, ``"sure or wager option"`` -- and no rule recovers a canonical
    meaning from prose. Only a label that is already a term is read as one; everything else
    stays unmapped until a curator declares what it means, which is the honest answer and
    the one that makes the gap visible.
    """

    try:
        return ChoiceTerm(label)
    except ValueError:
        return None
