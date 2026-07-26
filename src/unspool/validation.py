"""Longitudinal validation schemes with explicit leakage semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from unspool.study import Study

SplitScheme = Literal[
    "forward-session",
    "leave-one-session-out",
    "within-session-rolling-origin",
]
PopulationSplitScheme = Literal["leave-one-subject-out", "leave-one-lab-out"]
CohortSplitScheme = Literal["cohort-forward-session"]


@dataclass(frozen=True, slots=True)
class ValidationSplit:
    """Source-row positions and temporal metadata for one subject-level fold.

    ``prediction_context_indices`` are observed training rows replayed to initialize a
    filtered prediction. They are never part of ``test_indices`` and are not scored.
    """

    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]
    subject: Any
    train_sessions: tuple[Any, ...]
    test_sessions: tuple[Any, ...]
    train_session_orders: tuple[int, ...]
    test_session_orders: tuple[int, ...]
    scheme: SplitScheme
    prediction_context_indices: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )
    origin_session: Any | None = None
    origin_session_order: int | None = None
    origin_trial: int | None = None
    test_trials: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.scheme not in (
            "forward-session",
            "leave-one-session-out",
            "within-session-rolling-origin",
        ):
            raise ValueError(f"unknown validation scheme: {self.scheme!r}")

        train_sessions = tuple(self.train_sessions)
        test_sessions = tuple(self.test_sessions)
        train_orders = _validated_session_orders(self.train_session_orders, "train_session_orders")
        test_orders = _validated_session_orders(self.test_session_orders, "test_session_orders")
        train = _validated_indices(self.train_indices, "train_indices")
        test = _validated_indices(self.test_indices, "test_indices")
        context = _validated_indices(
            self.prediction_context_indices,
            "prediction_context_indices",
            allow_empty=True,
        )
        if np.intersect1d(train, test).size:
            raise ValueError("training and test indices must not overlap")
        if np.intersect1d(context, test).size:
            raise ValueError("prediction context and test indices must not overlap")
        if np.setdiff1d(context, train).size:
            raise ValueError("prediction context must be drawn from training indices")
        if len(train_sessions) != len(train_orders):
            raise ValueError("every training session must have a session order")
        if len(test_sessions) != len(test_orders):
            raise ValueError("every test session must have a session order")
        if not train_sessions or not test_sessions:
            raise ValueError("training and test sessions must not be empty")
        if self.scheme == "within-session-rolling-origin":
            origin_order, origin_trial, test_trials = _validate_trial_origin(
                origin_session=self.origin_session,
                origin_session_order=self.origin_session_order,
                origin_trial=self.origin_trial,
                test_trials=self.test_trials,
            )
            if not context.size:
                raise ValueError("within-session splits require prediction context")
            if test_orders != (origin_order,) or set(train_orders) & set(test_orders) != {
                origin_order
            }:
                raise ValueError(
                    "within-session training and testing may overlap only at the origin session"
                )
            if max(train_orders) != origin_order:
                raise ValueError(
                    "all complete training sessions must precede the within-session origin"
                )
            if len(test_sessions) != 1 or not _equal(test_sessions[0], self.origin_session):
                raise ValueError("within-session test metadata must identify the origin session")
            origin_positions = [
                position for position, order in enumerate(train_orders) if order == origin_order
            ]
            if len(origin_positions) != 1 or not _equal(
                train_sessions[origin_positions[0]], self.origin_session
            ):
                raise ValueError("the origin session must appear exactly once in training metadata")
        else:
            origin_order = None
            origin_trial = None
            test_trials = ()
            if set(train_orders) & set(test_orders):
                raise ValueError("training and test session orders must not overlap")
            if context.size:
                raise ValueError("prediction context is only valid for within-session splits")
            if (
                any(
                    value is not None
                    for value in (self.origin_session, self.origin_session_order, self.origin_trial)
                )
                or self.test_trials
            ):
                raise ValueError("trial-origin metadata is only valid for within-session splits")
            if self.scheme == "forward-session" and max(train_orders) >= min(test_orders):
                raise ValueError("forward-session training must occur strictly before testing")
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)
        object.__setattr__(self, "prediction_context_indices", context)
        object.__setattr__(self, "train_sessions", train_sessions)
        object.__setattr__(self, "test_sessions", test_sessions)
        object.__setattr__(self, "train_session_orders", train_orders)
        object.__setattr__(self, "test_session_orders", test_orders)
        object.__setattr__(self, "origin_session_order", origin_order)
        object.__setattr__(self, "origin_trial", origin_trial)
        object.__setattr__(self, "test_trials", test_trials)

    @property
    def prospective(self) -> bool:
        """Whether the split forbids training on observations from the test set's future."""

        return self.scheme in ("forward-session", "within-session-rolling-origin")


@dataclass(frozen=True, slots=True)
class PopulationValidationSplit:
    """Whole-population fold with explicit held-out subject and group metadata.

    Population folds contain complete subjects on both sides. ``group_column`` names the
    unit being held out: ``subject`` for subject generalization or a source column such as
    ``lab`` for group generalization.
    """

    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]
    train_subjects: tuple[Any, ...]
    test_subjects: tuple[Any, ...]
    train_groups: tuple[Any, ...]
    test_groups: tuple[Any, ...]
    held_out_group: Any
    group_column: str
    scheme: PopulationSplitScheme
    prediction_context_indices: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )

    def __post_init__(self) -> None:
        if self.scheme not in ("leave-one-subject-out", "leave-one-lab-out"):
            raise ValueError(f"unknown population validation scheme: {self.scheme!r}")
        if not isinstance(self.group_column, str) or not self.group_column:
            raise ValueError("group_column must be a non-empty string")

        train = _validated_indices(self.train_indices, "train_indices")
        test = _validated_indices(self.test_indices, "test_indices")
        context = _validated_indices(
            self.prediction_context_indices,
            "prediction_context_indices",
            allow_empty=True,
        )
        train_subjects = _validated_identifiers(self.train_subjects, "train_subjects")
        test_subjects = _validated_identifiers(self.test_subjects, "test_subjects")
        train_groups = _validated_identifiers(self.train_groups, "train_groups")
        test_groups = _validated_identifiers(self.test_groups, "test_groups")
        held_out_group = _validated_identifier(self.held_out_group, "held_out_group")

        if np.intersect1d(train, test).size:
            raise ValueError("training and test indices must not overlap")
        if context.size:
            raise ValueError("population splits do not use prediction context")
        if _identifier_keys(train_subjects) & _identifier_keys(test_subjects):
            raise ValueError("population training and test subjects must be disjoint")
        if _identifier_keys(train_groups) & _identifier_keys(test_groups):
            raise ValueError("population training and test groups must be disjoint")
        if len(test_groups) != 1 or not _equal(test_groups[0], held_out_group):
            raise ValueError("test_groups must contain exactly the held-out group")
        if self.scheme == "leave-one-subject-out":
            if self.group_column != "subject":
                raise ValueError("leave-one-subject-out folds must group by 'subject'")
            if train_groups != train_subjects or test_groups != test_subjects:
                raise ValueError("subject-holdout group metadata must match subject metadata")
        elif self.group_column == "subject":
            raise ValueError("leave-one-lab-out folds require a non-subject group column")

        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)
        object.__setattr__(self, "prediction_context_indices", context)
        object.__setattr__(self, "train_subjects", train_subjects)
        object.__setattr__(self, "test_subjects", test_subjects)
        object.__setattr__(self, "train_groups", train_groups)
        object.__setattr__(self, "test_groups", test_groups)
        object.__setattr__(self, "held_out_group", held_out_group)

    @property
    def prospective(self) -> bool:
        """Whether all observations from the held-out population unit are excluded."""

        return True


@dataclass(frozen=True, slots=True)
class CohortValidationSplit:
    """A prospective session-origin fold fitted jointly across a subject cohort.

    Session metadata is keyed by subject because source session identifiers and source
    chronology need not be shared across animals. All subjects nevertheless contribute
    the same number of training sessions and the same forecast horizon.
    """

    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]
    subjects: tuple[Any, ...]
    train_sessions: Mapping[Any, tuple[Any, ...]]
    test_sessions: Mapping[Any, tuple[Any, ...]]
    train_session_orders: Mapping[Any, tuple[int, ...]]
    test_session_orders: Mapping[Any, tuple[int, ...]]
    train_session_count: int
    scheme: CohortSplitScheme = "cohort-forward-session"
    prediction_context_indices: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )

    def __post_init__(self) -> None:
        if self.scheme != "cohort-forward-session":
            raise ValueError(f"unknown cohort validation scheme: {self.scheme!r}")
        _require_positive_integer(self.train_session_count, "train_session_count")

        train = _validated_indices(self.train_indices, "train_indices")
        test = _validated_indices(self.test_indices, "test_indices")
        context = _validated_indices(
            self.prediction_context_indices,
            "prediction_context_indices",
            allow_empty=True,
        )
        subjects = _validated_identifiers(self.subjects, "subjects")
        if np.intersect1d(train, test).size:
            raise ValueError("training and test indices must not overlap")
        if context.size:
            raise ValueError("cohort splits do not use prediction context")

        train_sessions = _validated_subject_mapping(self.train_sessions, subjects, "train_sessions")
        test_sessions = _validated_subject_mapping(self.test_sessions, subjects, "test_sessions")
        train_orders = _validated_subject_order_mapping(
            self.train_session_orders, subjects, "train_session_orders"
        )
        test_orders = _validated_subject_order_mapping(
            self.test_session_orders, subjects, "test_session_orders"
        )
        for subject in subjects:
            if len(train_sessions[subject]) != self.train_session_count:
                raise ValueError(
                    "every subject must contribute train_session_count training sessions"
                )
            if len(train_sessions[subject]) != len(train_orders[subject]):
                raise ValueError("every training session must have a session order")
            if len(test_sessions[subject]) != len(test_orders[subject]):
                raise ValueError("every test session must have a session order")
            if not test_sessions[subject]:
                raise ValueError("test sessions must not be empty")
            if set(train_orders[subject]) & set(test_orders[subject]):
                raise ValueError("training and test session orders must not overlap")
            if max(train_orders[subject]) >= min(test_orders[subject]):
                raise ValueError("cohort training must occur strictly before testing")

        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)
        object.__setattr__(self, "prediction_context_indices", context)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "train_sessions", MappingProxyType(train_sessions))
        object.__setattr__(self, "test_sessions", MappingProxyType(test_sessions))
        object.__setattr__(self, "train_session_orders", MappingProxyType(train_orders))
        object.__setattr__(self, "test_session_orders", MappingProxyType(test_orders))

    @property
    def prospective(self) -> bool:
        """Whether every subject is trained strictly before its test sessions."""

        return True


ValidationFold = ValidationSplit | PopulationValidationSplit | CohortValidationSplit


def forward_session_splits(
    study: Study,
    *,
    min_train_sessions: int = 1,
    horizon: int = 1,
    step: int = 1,
) -> tuple[ValidationSplit, ...]:
    """Create expanding-history, forward-session folds within each subject.

    Every test session occurs strictly after every training session according to the
    explicit ``session_order`` column. Each fold contains complete sessions, and returned
    indices always refer to the study's original row positions.
    """

    _require_positive_integer(min_train_sessions, "min_train_sessions")
    _require_positive_integer(horizon, "horizon")
    _require_positive_integer(step, "step")

    splits: list[ValidationSplit] = []
    for subject in study.subjects:
        ordered_sessions = _sessions_for_subject(study, subject)
        train_count = min_train_sessions
        while train_count + horizon <= len(ordered_sessions):
            train = ordered_sessions[:train_count]
            test = ordered_sessions[train_count : train_count + horizon]
            splits.append(
                _make_split(
                    study,
                    subject=subject,
                    train=train,
                    test=test,
                    scheme="forward-session",
                )
            )
            train_count += step
    return tuple(splits)


def cohort_forward_session_splits(
    study: Study,
    *,
    min_train_sessions: int = 1,
    horizon: int = 1,
    step: int = 1,
    require_all_subjects: bool = True,
) -> tuple[CohortValidationSplit, ...]:
    """Create expanding-history prospective folds fitted across a subject cohort.

    Unlike :func:`forward_session_splits`, each returned fold joins every eligible
    subject into one training and test set. By default, fold generation stops before the
    first origin unavailable for any subject, keeping the cohort fixed. Setting
    ``require_all_subjects=False`` permits later folds to contain only subjects with
    sufficient follow-up; membership remains explicit in each fold.
    """

    _require_positive_integer(min_train_sessions, "min_train_sessions")
    _require_positive_integer(horizon, "horizon")
    _require_positive_integer(step, "step")
    if not isinstance(require_all_subjects, bool):
        raise TypeError("require_all_subjects must be a boolean")

    sessions_by_subject = {
        _identifier_key(subject): _sessions_for_subject(study, subject)
        for subject in study.subjects
    }
    splits: list[CohortValidationSplit] = []
    train_count = min_train_sessions
    while True:
        eligible_subjects = tuple(
            subject
            for subject in study.subjects
            if train_count + horizon <= len(sessions_by_subject[_identifier_key(subject)])
        )
        if require_all_subjects and len(eligible_subjects) != len(study.subjects):
            break
        if not eligible_subjects:
            break

        train_sessions: dict[Any, tuple[Any, ...]] = {}
        test_sessions: dict[Any, tuple[Any, ...]] = {}
        train_orders: dict[Any, tuple[int, ...]] = {}
        test_orders: dict[Any, tuple[int, ...]] = {}
        train_order_sets: dict[Any, set[int]] = {}
        test_order_sets: dict[Any, set[int]] = {}
        for subject in eligible_subjects:
            key = _identifier_key(subject)
            ordered_sessions = sessions_by_subject[key]
            training = ordered_sessions[:train_count]
            testing = ordered_sessions[train_count : train_count + horizon]
            train_sessions[key] = tuple(session for _, session in training)
            test_sessions[key] = tuple(session for _, session in testing)
            train_orders[key] = tuple(order for order, _ in training)
            test_orders[key] = tuple(order for order, _ in testing)
            train_order_sets[key] = set(train_orders[key])
            test_order_sets[key] = set(test_orders[key])

        train_mask = np.fromiter(
            (
                _identifier_key(subject) in train_order_sets
                and int(order) in train_order_sets[_identifier_key(subject)]
                for subject, order in zip(study["subject"], study["session_order"], strict=True)
            ),
            dtype=np.bool_,
            count=len(study),
        )
        test_mask = np.fromiter(
            (
                _identifier_key(subject) in test_order_sets
                and int(order) in test_order_sets[_identifier_key(subject)]
                for subject, order in zip(study["subject"], study["session_order"], strict=True)
            ),
            dtype=np.bool_,
            count=len(study),
        )
        splits.append(
            CohortValidationSplit(
                train_indices=np.flatnonzero(train_mask),
                test_indices=np.flatnonzero(test_mask),
                subjects=eligible_subjects,
                train_sessions=train_sessions,
                test_sessions=test_sessions,
                train_session_orders=train_orders,
                test_session_orders=test_orders,
                train_session_count=train_count,
            )
        )
        train_count += step
    return tuple(splits)


def leave_one_session_out_splits(study: Study) -> tuple[ValidationSplit, ...]:
    """Hold out each complete session within subject in turn.

    Training folds include sessions on both sides of the held-out session. This scheme is
    useful for interpolation and session-robustness checks, but it is deliberately marked
    as non-prospective because training data can come from the held-out session's future.
    Subjects with fewer than two sessions do not produce a fold.
    """

    splits: list[ValidationSplit] = []
    for subject in study.subjects:
        ordered_sessions = _sessions_for_subject(study, subject)
        for held_out_index, held_out in enumerate(ordered_sessions):
            if len(ordered_sessions) < 2:
                continue
            training = ordered_sessions[:held_out_index] + ordered_sessions[held_out_index + 1 :]
            splits.append(
                _make_split(
                    study,
                    subject=subject,
                    train=training,
                    test=(held_out,),
                    scheme="leave-one-session-out",
                )
            )
    return tuple(splits)


def leave_one_subject_out_splits(study: Study) -> tuple[PopulationValidationSplit, ...]:
    """Hold out every complete subject while training on all other subjects."""

    return _leave_one_population_group_out(
        study,
        groups=study.subjects,
        group_column="subject",
        scheme="leave-one-subject-out",
    )


def leave_one_lab_out_splits(
    study: Study,
    *,
    lab_column: str = "lab",
) -> tuple[PopulationValidationSplit, ...]:
    """Hold out every complete lab while preventing subjects from crossing folds.

    The lab column is optional in the core ``Study`` schema, so its name is explicit and
    validated here. Every subject must map to exactly one non-missing lab.
    """

    if not isinstance(lab_column, str) or not lab_column:
        raise ValueError("lab_column must be a non-empty string")
    if lab_column == "subject":
        raise ValueError("lab_column must differ from the subject column")
    if lab_column not in study.columns:
        raise ValueError(f"study does not contain lab column {lab_column!r}")
    groups = _groups_with_constant_subject_assignment(study, lab_column)
    return _leave_one_population_group_out(
        study,
        groups=groups,
        group_column=lab_column,
        scheme="leave-one-lab-out",
    )


def within_session_rolling_splits(
    study: Study,
    *,
    min_train_sessions: int = 0,
    min_train_trials: int = 1,
    horizon: int = 1,
    step: int = 1,
) -> tuple[ValidationSplit, ...]:
    """Create expanding-history origins within each eligible subject/session.

    Training contains every earlier session plus the observed prefix of the current
    session. Testing contains the next ``horizon`` observed trials from that session.
    ``prediction_context_indices`` retains the current-session prefix so filtered models
    can construct history at the origin while scoring only future test trials.
    """

    _require_nonnegative_integer(min_train_sessions, "min_train_sessions")
    _require_positive_integer(min_train_trials, "min_train_trials")
    _require_positive_integer(horizon, "horizon")
    _require_positive_integer(step, "step")

    splits: list[ValidationSplit] = []
    for subject in study.subjects:
        ordered_sessions = _sessions_for_subject(study, subject)
        for session_index in range(min_train_sessions, len(ordered_sessions)):
            origin_order, origin_session = ordered_sessions[session_index]
            prior_orders = {order for order, _ in ordered_sessions[:session_index]}
            prior_rows = tuple(
                row
                for row in range(len(study))
                if _equal(study["subject"][row], subject)
                and int(study["session_order"][row]) in prior_orders
            )
            training_sessions = ordered_sessions[: session_index + 1]
            session_rows = _trial_rows_for_session(
                study,
                subject=subject,
                session_order=origin_order,
            )
            train_count = min_train_trials
            while train_count + horizon <= len(session_rows):
                context_rows = session_rows[:train_count]
                test_rows = session_rows[train_count : train_count + horizon]
                training_rows = np.asarray(
                    sorted((*prior_rows, *context_rows)),
                    dtype=np.intp,
                )
                context_indices = np.asarray(sorted(context_rows), dtype=np.intp)
                test_indices = np.asarray(sorted(test_rows), dtype=np.intp)
                splits.append(
                    ValidationSplit(
                        train_indices=training_rows,
                        test_indices=test_indices,
                        subject=subject,
                        train_sessions=tuple(session for _, session in training_sessions),
                        test_sessions=(origin_session,),
                        train_session_orders=tuple(order for order, _ in training_sessions),
                        test_session_orders=(origin_order,),
                        scheme="within-session-rolling-origin",
                        prediction_context_indices=context_indices,
                        origin_session=origin_session,
                        origin_session_order=origin_order,
                        origin_trial=int(study["trial"][context_rows[-1]]),
                        test_trials=tuple(int(study["trial"][row]) for row in test_rows),
                    )
                )
                train_count += step
    return tuple(splits)


def _leave_one_population_group_out(
    study: Study,
    *,
    groups: tuple[Any, ...],
    group_column: str,
    scheme: PopulationSplitScheme,
) -> tuple[PopulationValidationSplit, ...]:
    if len(groups) < 2:
        return ()

    group_values = study[group_column]
    splits: list[PopulationValidationSplit] = []
    for held_out_group in groups:
        test_mask = np.fromiter(
            (_equal(value, held_out_group) for value in group_values),
            dtype=np.bool_,
            count=len(study),
        )
        train_indices = np.flatnonzero(~test_mask)
        test_indices = np.flatnonzero(test_mask)
        train_subjects = _unique_identifiers(study["subject"][train_indices])
        test_subjects = _unique_identifiers(study["subject"][test_indices])
        train_groups = _unique_identifiers(group_values[train_indices])
        test_groups = _unique_identifiers(group_values[test_indices])
        splits.append(
            PopulationValidationSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
                train_groups=train_groups,
                test_groups=test_groups,
                held_out_group=_scalar(held_out_group),
                group_column=group_column,
                scheme=scheme,
            )
        )
    return tuple(splits)


def _groups_with_constant_subject_assignment(study: Study, group_column: str) -> tuple[Any, ...]:
    subject_groups: dict[Any, Any] = {}
    groups: list[Any] = []
    seen_groups: set[Any] = set()
    for row in range(len(study)):
        subject = _validated_identifier(study["subject"][row], f"subject at row {row}")
        group = _validated_identifier(study[group_column][row], f"{group_column} at row {row}")
        subject_key = _identifier_key(subject)
        known_group = subject_groups.setdefault(subject_key, group)
        if not _equal(known_group, group):
            raise ValueError(
                f"every subject must map to exactly one {group_column}; "
                f"subject {subject!r} maps to {known_group!r} and {group!r}"
            )
        group_key = _identifier_key(group)
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            groups.append(group)
    return tuple(groups)


def _unique_identifiers(values: NDArray[Any]) -> tuple[Any, ...]:
    identifiers: list[Any] = []
    seen: set[Any] = set()
    for position, value in enumerate(values):
        identifier = _validated_identifier(value, f"identifier at position {position}")
        key = _identifier_key(identifier)
        if key not in seen:
            seen.add(key)
            identifiers.append(identifier)
    return tuple(identifiers)


def _sessions_for_subject(study: Study, subject: Any) -> tuple[tuple[int, Any], ...]:
    sessions: dict[int, Any] = {}
    for row in range(len(study)):
        if _equal(study["subject"][row], subject):
            order = int(study["session_order"][row])
            sessions[order] = _scalar(study["session"][row])
    return tuple(sorted(sessions.items()))


def _trial_rows_for_session(
    study: Study,
    *,
    subject: Any,
    session_order: int,
) -> tuple[int, ...]:
    rows = (
        row
        for row in range(len(study))
        if _equal(study["subject"][row], subject)
        and int(study["session_order"][row]) == session_order
    )
    return tuple(sorted(rows, key=lambda row: (int(study["trial"][row]), row)))


def _make_split(
    study: Study,
    *,
    subject: Any,
    train: tuple[tuple[int, Any], ...],
    test: tuple[tuple[int, Any], ...],
    scheme: SplitScheme,
) -> ValidationSplit:
    train_orders = tuple(order for order, _ in train)
    test_orders = tuple(order for order, _ in test)
    train_order_set = set(train_orders)
    test_order_set = set(test_orders)

    subject_mask = np.fromiter(
        (_equal(value, subject) for value in study["subject"]), dtype=np.bool_, count=len(study)
    )
    orders = study["session_order"]
    train_mask = subject_mask & np.fromiter(
        (int(value) in train_order_set for value in orders), dtype=np.bool_, count=len(study)
    )
    test_mask = subject_mask & np.fromiter(
        (int(value) in test_order_set for value in orders), dtype=np.bool_, count=len(study)
    )

    return ValidationSplit(
        train_indices=np.flatnonzero(train_mask),
        test_indices=np.flatnonzero(test_mask),
        subject=subject,
        train_sessions=tuple(session for _, session in train),
        test_sessions=tuple(session for _, session in test),
        train_session_orders=train_orders,
        test_session_orders=test_orders,
        scheme=scheme,
    )


def _validated_indices(
    values: NDArray[np.intp],
    name: str,
    *,
    allow_empty: bool = False,
) -> NDArray[np.intp]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if array.size == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if np.any(array < 0):
        raise ValueError(f"{name} must not contain negative positions")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} must not contain duplicate positions")
    result = np.array(array, dtype=np.intp, copy=True)
    result.setflags(write=False)
    view = result.view()
    view.setflags(write=False)
    return view


def _validated_session_orders(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    orders = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in orders):
        raise TypeError(f"{name} must contain integers")
    if len(set(orders)) != len(orders):
        raise ValueError(f"{name} must not contain duplicates")
    return orders


def _validated_identifiers(values: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    identifiers = tuple(
        _validated_identifier(value, f"{name}[{position}]") for position, value in enumerate(values)
    )
    if not identifiers:
        raise ValueError(f"{name} must not be empty")
    if len(_identifier_keys(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must not contain duplicates")
    return identifiers


def _validated_subject_mapping(
    values: Mapping[Any, tuple[Any, ...]],
    subjects: tuple[Any, ...],
    name: str,
) -> dict[Any, tuple[Any, ...]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by subject")
    expected = _identifier_keys(subjects)
    observed = {_identifier_key(key) for key in values}
    if observed != expected:
        raise ValueError(f"{name} keys must exactly match subjects")
    result: dict[Any, tuple[Any, ...]] = {}
    for subject in subjects:
        key = _identifier_key(subject)
        entries = tuple(
            _validated_identifier(value, f"{name}[{subject!r}]") for value in values[key]
        )
        if not entries:
            raise ValueError(f"{name}[{subject!r}] must not be empty")
        result[key] = entries
    return result


def _validated_subject_order_mapping(
    values: Mapping[Any, tuple[int, ...]],
    subjects: tuple[Any, ...],
    name: str,
) -> dict[Any, tuple[int, ...]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by subject")
    expected = _identifier_keys(subjects)
    observed = {_identifier_key(key) for key in values}
    if observed != expected:
        raise ValueError(f"{name} keys must exactly match subjects")
    return {
        _identifier_key(subject): _validated_session_orders(values[_identifier_key(subject)], name)
        for subject in subjects
    }


def _validated_identifier(value: Any, name: str) -> Any:
    identifier = _scalar(value)
    if identifier is None:
        raise ValueError(f"{name} must not be missing")
    if isinstance(identifier, (float, complex, np.floating, np.complexfloating)) and np.isnan(
        identifier
    ):
        raise ValueError(f"{name} must not be missing")
    if isinstance(identifier, (np.datetime64, np.timedelta64)) and np.isnat(identifier):
        raise ValueError(f"{name} must not be missing")
    try:
        hash(identifier)
    except TypeError:
        raise TypeError(f"{name} must be hashable") from None
    return identifier


def _identifier_key(value: Any) -> Any:
    return _scalar(value)


def _identifier_keys(values: tuple[Any, ...]) -> set[Any]:
    return {_identifier_key(value) for value in values}


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_trial_origin(
    *,
    origin_session: Any | None,
    origin_session_order: int | None,
    origin_trial: int | None,
    test_trials: tuple[int, ...],
) -> tuple[int, int, tuple[int, ...]]:
    if origin_session is None:
        raise ValueError("within-session splits require origin_session")
    if (
        isinstance(origin_session_order, bool)
        or not isinstance(origin_session_order, int)
        or origin_session_order < 0
    ):
        raise ValueError("origin_session_order must be a non-negative integer")
    if isinstance(origin_trial, bool) or not isinstance(origin_trial, int) or origin_trial < 0:
        raise ValueError("origin_trial must be a non-negative integer")
    trials = tuple(test_trials)
    if not trials:
        raise ValueError("within-session splits require test_trials")
    if any(isinstance(trial, bool) or not isinstance(trial, int) or trial < 0 for trial in trials):
        raise ValueError("test_trials must contain non-negative integers")
    if len(set(trials)) != len(trials) or tuple(sorted(trials)) != trials:
        raise ValueError("test_trials must be unique and increasing")
    if trials[0] <= origin_trial:
        raise ValueError("test trials must occur after origin_trial")
    return origin_session_order, origin_trial, trials


def _equal(left: Any, right: Any) -> bool:
    return bool(_scalar(left) == _scalar(right))


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
