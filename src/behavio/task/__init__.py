"""The behavioural task contract: what a trial means before any model is fitted.

The area has two layers and they are not rivals.

:mod:`behavio.task.spec` is the **structural** layer. It declares the observed columns of a
study -- which one holds the choice, what its options are spelled as, which values are
omissions -- and :mod:`behavio.task.response_times` declares the unit and clock origin a
response time is measured in. It is what a model is validated against, and it takes labels
as it finds them, because a source table is entitled to code a leftward turn ``-1``.

:mod:`behavio.task.vocabulary` and :mod:`behavio.task.ontology` are the **named** layer.
They say that ``-1`` in one dataset and ``"left port"`` in another are both
:attr:`~behavio.task.vocabulary.ChoiceTerm.LEFT`, that both experiments belong to one
:class:`~behavio.task.ontology.TaskFamily`, and that the family turns on the same canonical
variables. The named layer *produces* the structural one --
:meth:`~behavio.task.ontology.TaskProtocol.task_spec` returns an ordinary
:class:`~behavio.task.spec.TaskSpec` -- so a term is defined once and reaches the analysis
path rather than sitting beside it.

Using the ontology is optional. ``TaskSpec(choice=ChoiceSpec(options=(0, 1)))`` is a
complete task contract and a GLM fitted to a personal CSV never names a family.

The area sits below :mod:`behavio.models`: it is typed against the estimator contract in
:mod:`behavio.contracts.estimator`, not against any concrete model, so a model may require
task columns without the task layer knowing the catalogue exists.
"""

from behavio.task.jsonschema import (
    ONTOLOGY_JSON_SCHEMAS,
    canonical_trial_json_schema,
    task_family_json_schema,
    task_protocol_json_schema,
)
from behavio.task.ontology import (
    ACCEPTED_ONTOLOGY_SCHEMA_VERSIONS,
    CANONICAL_TRIAL_REQUIRED,
    ONTOLOGY_SCHEMA_VERSION,
    TASK_FAMILY_REQUIRED,
    TASK_PROTOCOL_REQUIRED,
    CanonicalTrial,
    CanonicalVariable,
    ChoiceDeclaration,
    ClaimConfidence,
    FeedbackDeclaration,
    InterpretationClaim,
    OntologyError,
    ProtocolScope,
    Provenance,
    Reference,
    StimulusDeclaration,
    TaskFamily,
    TaskProtocol,
    TrainingDeclaration,
    TrialPhase,
    canonical_trial_from_dict,
    canonical_trials,
    task_family_from_dict,
    task_protocol_from_dict,
)
from behavio.task.response_times import ResponseTimes, ResponseTimeSpec, ResponseTimeUnit
from behavio.task.spec import (
    ChoiceData,
    ChoiceSpec,
    FittedModel,
    RewardSpec,
    TaskSpec,
    TaskValidation,
    TaskValidationError,
    fit_model,
)
from behavio.task.vocabulary import (
    CONTROLLED_VOCABULARIES,
    OMISSION_CHOICE_TERMS,
    UNCURATED_TERMS,
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
    VocabularyError,
    choice_term_of,
)

__all__ = [
    "ACCEPTED_ONTOLOGY_SCHEMA_VERSIONS",
    "CANONICAL_TRIAL_REQUIRED",
    "CONTROLLED_VOCABULARIES",
    "OMISSION_CHOICE_TERMS",
    "ONTOLOGY_JSON_SCHEMAS",
    "ONTOLOGY_SCHEMA_VERSION",
    "TASK_FAMILY_REQUIRED",
    "TASK_PROTOCOL_REQUIRED",
    "UNCURATED_TERMS",
    "CanonicalTrial",
    "CanonicalVariable",
    "ChoiceData",
    "ChoiceDeclaration",
    "ChoiceSpec",
    "ChoiceTerm",
    "ChoiceType",
    "ClaimConfidence",
    "CurationStatus",
    "EvidenceType",
    "FeedbackDeclaration",
    "FeedbackTerm",
    "FeedbackType",
    "FittedModel",
    "InterpretationClaim",
    "Modality",
    "ObservationDataType",
    "ObservationRole",
    "OntologyError",
    "ProtocolScope",
    "Provenance",
    "Reference",
    "ResponseModality",
    "ResponseTimeSpec",
    "ResponseTimeUnit",
    "ResponseTimes",
    "RewardSpec",
    "Species",
    "StimulusDeclaration",
    "StimulusSide",
    "TaskFamily",
    "TaskProtocol",
    "TaskSpec",
    "TaskValidation",
    "TaskValidationError",
    "TrainingDeclaration",
    "TrialPhase",
    "VocabularyError",
    "canonical_trial_from_dict",
    "canonical_trial_json_schema",
    "canonical_trials",
    "choice_term_of",
    "fit_model",
    "task_family_from_dict",
    "task_family_json_schema",
    "task_protocol_from_dict",
    "task_protocol_json_schema",
]
