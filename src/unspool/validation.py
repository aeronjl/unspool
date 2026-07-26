"""Longitudinal validation schemes with explicit leakage semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from unspool.study import Study

SplitScheme = Literal["forward-session", "leave-one-session-out"]


@dataclass(frozen=True, slots=True)
class ValidationSplit:
    """Source-row positions and metadata for one subject-level validation fold."""

    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]
    subject: Any
    train_sessions: tuple[Any, ...]
    test_sessions: tuple[Any, ...]
    train_session_orders: tuple[int, ...]
    test_session_orders: tuple[int, ...]
    scheme: SplitScheme

    def __post_init__(self) -> None:
        if self.scheme not in ("forward-session", "leave-one-session-out"):
            raise ValueError(f"unknown validation scheme: {self.scheme!r}")

        train_sessions = tuple(self.train_sessions)
        test_sessions = tuple(self.test_sessions)
        train_orders = _validated_session_orders(self.train_session_orders, "train_session_orders")
        test_orders = _validated_session_orders(self.test_session_orders, "test_session_orders")
        train = _validated_indices(self.train_indices, "train_indices")
        test = _validated_indices(self.test_indices, "test_indices")
        if np.intersect1d(train, test).size:
            raise ValueError("training and test indices must not overlap")
        if len(train_sessions) != len(train_orders):
            raise ValueError("every training session must have a session order")
        if len(test_sessions) != len(test_orders):
            raise ValueError("every test session must have a session order")
        if not train_sessions or not test_sessions:
            raise ValueError("training and test sessions must not be empty")
        if set(train_orders) & set(test_orders):
            raise ValueError("training and test session orders must not overlap")
        if self.scheme == "forward-session" and max(train_orders) >= min(test_orders):
            raise ValueError("forward-session training must occur strictly before testing")
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)
        object.__setattr__(self, "train_sessions", train_sessions)
        object.__setattr__(self, "test_sessions", test_sessions)
        object.__setattr__(self, "train_session_orders", train_orders)
        object.__setattr__(self, "test_session_orders", test_orders)

    @property
    def prospective(self) -> bool:
        """Whether the split forbids training on observations from the test set's future."""

        return self.scheme == "forward-session"


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


def _sessions_for_subject(study: Study, subject: Any) -> tuple[tuple[int, Any], ...]:
    sessions: dict[int, Any] = {}
    for row in range(len(study)):
        if _equal(study["subject"][row], subject):
            order = int(study["session_order"][row])
            sessions[order] = _scalar(study["session"][row])
    return tuple(sorted(sessions.items()))


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


def _validated_indices(values: NDArray[np.intp], name: str) -> NDArray[np.intp]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if array.size == 0:
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


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _equal(left: Any, right: Any) -> bool:
    return bool(_scalar(left) == _scalar(right))


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
