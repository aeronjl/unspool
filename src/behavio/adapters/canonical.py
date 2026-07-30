"""Canonical trial records into a validated ``Study``, checked against the vocabulary.

A harmonized trial table is the interchange format a curation layer produces: one row per
trial, every term drawn from the closed sets in :mod:`behavio.task.vocabulary`. Turning it
into a :class:`~behavio.trials.Study` looks like a rename and is not, because three of the
four columns Behavio requires are only *nearly* present and the fourth is absent entirely:

- ``subject_id`` is optional in a canonical record and required by a study. A row without
  one is refused rather than filled in, because a study whose subject is invented cannot
  have its subject boundaries preserved, and preserving them is a stated requirement.
- ``trial_index`` becomes ``trial`` and must be non-negative.
- ``session_id`` becomes ``session``.
- ``session_order`` does not exist. No canonical record carries session chronology in any
  form, so the caller must name a derivation, exactly as
  :mod:`behavio.adapters.table` requires of a CSV that lacks the column. The same three
  rules are reused rather than reinvented --
  :func:`~behavio.adapters.table.session_order_from_column`,
  :func:`~behavio.adapters.table.session_order_from_explicit` and
  :func:`~behavio.adapters.table.session_order_from_appearance` -- and the rule that was
  applied is written to ``source_session_order_rule`` on every trial, so a study whose
  chronology was derived can never be mistaken for one whose source recorded it.

Validation is against the declaration, not a promise. When a
:class:`~behavio.task.ontology.TaskProtocol` is supplied, every trial's protocol identity,
choice term, stimulus modality and declared units are checked against it, and a term that
records the *absence* of curation -- ``choice: unknown`` -- is refused rather than becoming
a category a model could fit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from behavio.adapters.table import SessionOrderDerivation, SessionOrderRule
from behavio.contracts.adapter import SessionOrderPolicy, SourceType
from behavio.task.ontology import CanonicalTrial, TaskFamily, TaskProtocol, canonical_trials
from behavio.task.vocabulary import ChoiceTerm
from behavio.trials import REQUIRED_COLUMNS, Study

ADAPTER_NAME = "behavio.canonical-trials"
ADAPTER_VERSION = "1"

#: Written to every trial, naming the rule that produced the chronology.
PROVENANCE_COLUMNS = ("source_session_order_rule",)

#: Canonical record fields carried through unchanged, in emission order. ``task_variables``
#: and ``source`` are absent because neither is a scalar: the first is flattened, and the
#: second is per-row provenance of arbitrary shape that a trial table cannot hold.
_CARRIED = (
    "protocol_id",
    "dataset_id",
    "stimulus_modality",
    "stimulus_value",
    "stimulus_units",
    "stimulus_side",
    "evidence_strength",
    "evidence_units",
    "choice",
    "correct",
    "response_time",
    "response_time_origin",
    "feedback",
    "reward",
    "reward_units",
    "block_id",
    "prior_context",
    "training_stage",
)


class CanonicalTrialError(ValueError):
    """Raised when canonical trial records cannot become a valid ``Study``."""


@dataclass(frozen=True, slots=True)
class CanonicalTrialSource:
    """Harmonized trials, the declaration they were harmonized against, and a chronology.

    ``protocol`` and ``family`` are optional. Without them the record's own controlled
    terms are still enforced -- that happens when a
    :class:`~behavio.task.ontology.CanonicalTrial` is constructed -- but nothing checks that
    the trials belong to the task they claim to.
    Supplying a protocol is what turns the conversion from a rename into a check.
    """

    trials: tuple[CanonicalTrial, ...]
    session_order: SessionOrderDerivation
    protocol: TaskProtocol | None = None
    family: TaskFamily | None = None

    adapter_name: ClassVar[str] = ADAPTER_NAME
    adapter_version: ClassVar[str] = ADAPTER_VERSION
    source_type: ClassVar[SourceType] = SourceType.IN_MEMORY

    def __post_init__(self) -> None:
        trials = tuple(self.trials)
        if not trials:
            raise CanonicalTrialError("at least one canonical trial is required")
        if any(not isinstance(trial, CanonicalTrial) for trial in trials):
            raise TypeError("trials must contain CanonicalTrial records")
        if not isinstance(self.session_order, SessionOrderDerivation):
            raise TypeError(
                "session_order must be a SessionOrderDerivation: a canonical trial carries no "
                "session chronology, so the rule that supplies it has to be named"
            )
        if self.protocol is not None and not isinstance(self.protocol, TaskProtocol):
            raise TypeError("protocol must be a TaskProtocol")
        if self.family is not None and not isinstance(self.family, TaskFamily):
            raise TypeError("family must be a TaskFamily")
        if (
            self.protocol is not None
            and self.family is not None
            and self.protocol.family_identifier != self.family.identifier
        ):
            raise CanonicalTrialError(
                f"protocol {self.protocol.identifier!r} belongs to family "
                f"{self.protocol.family_identifier!r}, not {self.family.identifier!r}"
            )
        object.__setattr__(self, "trials", trials)

    @property
    def session_order_policy(self) -> SessionOrderPolicy:
        """Always ``DERIVED``: no canonical record carries session chronology."""

        return SessionOrderPolicy.DERIVED

    def read(self) -> Study:
        """Validate the trials against their declaration and return a canonical study."""

        return study_from_canonical_trials(
            self.trials,
            session_order=self.session_order,
            protocol=self.protocol,
            family=self.family,
        )


def study_from_canonical_trials(
    trials: Iterable[CanonicalTrial | Mapping[str, Any]],
    *,
    session_order: SessionOrderDerivation,
    protocol: TaskProtocol | None = None,
    family: TaskFamily | None = None,
) -> Study:
    """Convert canonical trial records into a study, checking every declared term."""

    records = _records(trials)
    source = CanonicalTrialSource(
        trials=records, session_order=session_order, protocol=protocol, family=family
    )
    _check_declaration(source.trials, source.protocol, source.family)
    subjects = _subjects(source.trials)
    sessions = [trial.session_id for trial in source.trials]
    columns: dict[str, list[Any]] = {
        "subject": subjects,
        "session": sessions,
        "trial": _trial_numbers(source.trials),
        "session_order": _session_orders(source.trials, subjects, sessions, session_order),
    }
    for name in _CARRIED:
        values = [_scalar(getattr(trial, name)) for trial in source.trials]
        if all(value is None for value in values):
            continue
        columns[name] = values
    for name, values in _task_variable_columns(source.trials, set(columns)).items():
        columns[name] = values
    columns["source_session_order_rule"] = [session_order.basis] * len(source.trials)
    try:
        return Study.from_columns({name: _array(values) for name, values in columns.items()})
    except ValueError as error:
        raise CanonicalTrialError(str(error)) from error


def _records(trials: Iterable[CanonicalTrial | Mapping[str, Any]]) -> tuple[CanonicalTrial, ...]:
    items = list(trials)
    if items and all(isinstance(item, CanonicalTrial) for item in items):
        return tuple(items)  # type: ignore[arg-type]
    return canonical_trials(item for item in items if isinstance(item, Mapping))


def _check_declaration(
    trials: Sequence[CanonicalTrial],
    protocol: TaskProtocol | None,
    family: TaskFamily | None,
) -> None:
    for index, trial in enumerate(trials):
        if trial.choice is ChoiceTerm.UNKNOWN:
            raise CanonicalTrialError(
                f"trial {index}: choice is 'unknown', which records that a curator could not "
                "determine it. Drop the row or curate it; it cannot become an observation."
            )
    _check_constant(trials, "stimulus_units")
    _check_constant(trials, "evidence_units")
    _check_constant(trials, "reward_units")
    if protocol is not None:
        _check_protocol(trials, protocol)
    if family is not None:
        _check_family(trials, family)


def _check_protocol(trials: Sequence[CanonicalTrial], protocol: TaskProtocol) -> None:
    declared_terms = {value for value in protocol.choice.canonical_terms if value is not None}
    modalities = set(protocol.stimulus.modalities)
    for index, trial in enumerate(trials):
        if trial.protocol_id != protocol.identifier:
            raise CanonicalTrialError(
                f"trial {index}: protocol_id {trial.protocol_id!r} is not {protocol.identifier!r}"
            )
        if declared_terms and trial.choice not in declared_terms:
            raise CanonicalTrialError(
                f"trial {index}: choice {trial.choice.value!r} is not an alternative protocol "
                f"{protocol.identifier!r} declares "
                f"({sorted(value.value for value in declared_terms)})"
            )
        if trial.stimulus_modality not in modalities:
            raise CanonicalTrialError(
                f"trial {index}: stimulus_modality {trial.stimulus_modality.value!r} is not a "
                f"modality protocol {protocol.identifier!r} declares "
                f"({sorted(value.value for value in modalities)})"
            )
    _check_units_against(
        trials,
        "reward_units",
        None if protocol.reward is None else protocol.reward.units,
        protocol.identifier,
    )
    if protocol.response_time is not None:
        _check_units_against(
            trials, "response_time_origin", protocol.response_time.origin, protocol.identifier
        )


def _check_family(trials: Sequence[CanonicalTrial], family: TaskFamily) -> None:
    modalities = set(family.modalities)
    terms = set(family.choice_terms)
    for index, trial in enumerate(trials):
        if trial.stimulus_modality not in modalities:
            raise CanonicalTrialError(
                f"trial {index}: stimulus_modality {trial.stimulus_modality.value!r} is not a "
                f"modality family {family.identifier!r} declares "
                f"({sorted(value.value for value in modalities)})"
            )
        if terms and trial.choice not in terms:
            raise CanonicalTrialError(
                f"trial {index}: choice {trial.choice.value!r} is not a term family "
                f"{family.identifier!r} declares "
                f"({sorted(value.value for value in terms)})"
            )


def _check_constant(trials: Sequence[CanonicalTrial], name: str) -> None:
    """A unit that changes between rows silently mixes two scales in one column."""

    observed = {getattr(trial, name) for trial in trials} - {None}
    if len(observed) > 1:
        raise CanonicalTrialError(
            f"{name} is not constant across these trials ({sorted(observed)}); a single study "
            "column cannot hold two scales. Convert the source, or read the sources separately."
        )


def _check_units_against(
    trials: Sequence[CanonicalTrial], name: str, declared: str | None, protocol_id: str
) -> None:
    if declared is None:
        return
    observed = {getattr(trial, name) for trial in trials} - {None}
    unexpected = sorted(observed - {declared})
    if unexpected:
        raise CanonicalTrialError(
            f"{name} {unexpected} does not match {declared!r}, which protocol {protocol_id!r} "
            "declares"
        )


def _subjects(trials: Sequence[CanonicalTrial]) -> list[str]:
    missing = [index for index, trial in enumerate(trials) if trial.subject_id is None]
    if missing:
        raise CanonicalTrialError(
            f"trial {missing[0]} has no subject_id. A canonical record may omit it; a study may "
            "not, because subject boundaries are what a longitudinal analysis is defined over. "
            "Supply the subject on every record."
        )
    return [str(trial.subject_id) for trial in trials]


def _trial_numbers(trials: Sequence[CanonicalTrial]) -> list[int]:
    negative = [index for index, trial in enumerate(trials) if trial.trial_index < 0]
    if negative:
        raise CanonicalTrialError(
            f"trial {negative[0]} has trial_index {trials[negative[0]].trial_index}; a study's "
            "trial numbers are non-negative positions within a session"
        )
    return [int(trial.trial_index) for trial in trials]


def _session_orders(
    trials: Sequence[CanonicalTrial],
    subjects: Sequence[str],
    sessions: Sequence[str],
    derivation: SessionOrderDerivation,
) -> list[int]:
    if derivation.rule is SessionOrderRule.EXPLICIT:
        ranks = {session: rank for rank, session in enumerate(derivation.ordering or ())}
        unknown = sorted({session for session in sessions if session not in ranks})
        if unknown:
            raise CanonicalTrialError(
                f"the explicit session ordering does not mention {unknown}. Every session "
                "identifier in the records must appear in the ordering."
            )
        return [ranks[session] for session in sessions]

    if derivation.rule is SessionOrderRule.APPEARANCE:
        seen: dict[str, dict[str, int]] = {}
        orders: list[int] = []
        for subject, session in zip(subjects, sessions, strict=True):
            known = seen.setdefault(subject, {})
            if session not in known:
                known[session] = len(known)
            orders.append(known[session])
        return orders

    key_name = str(derivation.column)
    keys: dict[tuple[str, str], Any] = {}
    values: list[Any] = []
    for index, trial in enumerate(trials):
        value = _ordering_key(trial, key_name, index)
        values.append(value)
        pair = (subjects[index], sessions[index])
        known_value = keys.setdefault(pair, value)
        if known_value != value:
            raise CanonicalTrialError(
                f"ordering key {key_name!r} is not constant within subject {pair[0]!r} session "
                f"{pair[1]!r} ({known_value!r} and {value!r}); session chronology must be a "
                "property of the session, not of the trial"
            )
    by_subject: dict[str, list[tuple[Any, str]]] = {}
    for (subject, session), value in keys.items():
        by_subject.setdefault(subject, []).append((value, session))
    ranks_by_pair: dict[tuple[str, str], int] = {}
    for subject, entries in by_subject.items():
        try:
            ordered = sorted(entries, key=lambda entry: entry[0])
        except TypeError:
            raise CanonicalTrialError(
                f"ordering key {key_name!r} mixes value types for subject {subject!r}; a "
                "chronology cannot be derived from values that do not compare"
            ) from None
        for rank, (_value, session) in enumerate(ordered):
            ranks_by_pair[(subject, session)] = rank
    return [
        ranks_by_pair[(subject, session)]
        for subject, session in zip(subjects, sessions, strict=True)
    ]


def _ordering_key(trial: CanonicalTrial, name: str, index: int) -> Any:
    for holder in (trial.task_variables, trial.source):
        if name in holder:
            value = holder[name]
            if value is None:
                break
            return value
    raise CanonicalTrialError(
        f"trial {index} has no ordering key {name!r} in task_variables or source; session "
        "chronology cannot be derived from a value that is not there"
    )


def _task_variable_columns(
    trials: Sequence[CanonicalTrial], taken: set[str]
) -> dict[str, list[Any]]:
    names = tuple(trials[0].task_variables)
    for index, trial in enumerate(trials[1:], start=1):
        if tuple(trial.task_variables) != names:
            missing = sorted(set(names) - set(trial.task_variables))
            extra = sorted(set(trial.task_variables) - set(names))
            raise CanonicalTrialError(
                f"trial {index} declares different task_variables from trial 0; "
                f"missing={missing}, extra={extra}. A study column has to exist on every row."
            )
    collisions = sorted(set(names) & (taken | set(REQUIRED_COLUMNS) | set(PROVENANCE_COLUMNS)))
    if collisions:
        raise CanonicalTrialError(
            f"task_variables {collisions} collide with canonical study columns; rename them in "
            "the source before converting"
        )
    return {name: [_scalar(trial.task_variables[name]) for trial in trials] for name in names}


def _scalar(value: Any) -> Any:
    return value.value if hasattr(value, "value") and isinstance(value, str) else value


def _array(values: Sequence[Any]) -> NDArray[Any]:
    kinds = {type(value) for value in values if value is not None}
    missing = any(value is None for value in values)
    if kinds == {bool} and not missing:
        return np.asarray(values, dtype=bool)
    if kinds == {int} and not missing:
        return np.asarray(values, dtype=np.int64)
    if kinds <= {int, float} and kinds and bool not in kinds:
        return np.asarray(
            [np.nan if value is None else float(value) for value in values], dtype=np.float64
        )
    if kinds == {str} and not missing:
        return np.asarray(values, dtype=np.str_)
    return np.asarray(values, dtype=object)
