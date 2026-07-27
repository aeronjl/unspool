"""Canonical trial-level data contract for longitudinal studies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

REQUIRED_COLUMNS: Final = ("subject", "session", "trial", "session_order")


class StudyValidationError(ValueError):
    """Raised when columns do not satisfy the longitudinal study contract."""


class Study:
    """An immutable, columnar view of trial-level longitudinal data.

    A study has one row per observed trial and four required columns:

    - ``subject``: stable subject identifier;
    - ``session``: stable session identifier within a subject;
    - ``trial``: non-negative integer trial position within a session;
    - ``session_order``: non-negative integer chronology within a subject.

    Additional source columns are retained unchanged. Row order is also retained; Unspool
    never infers chronology from row position or silently sorts source data.
    """

    __slots__ = ("_columns", "_length", "_subjects")

    def __init__(self, columns: Mapping[str, Sequence[Any] | NDArray[Any]]) -> None:
        arrays = _copy_columns(columns)
        length, subjects = _validate_columns(arrays)
        self._columns = MappingProxyType(arrays)
        self._length = length
        self._subjects = subjects

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def from_columns(cls, columns: Mapping[str, Sequence[Any] | NDArray[Any]]) -> Study:
        """Construct a study from equally sized one-dimensional columns."""

        return cls(columns)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> Study:
        """Construct a study from records with identical fields."""

        rows = list(records)
        if not rows:
            raise StudyValidationError("a study must contain at least one trial")

        names = tuple(rows[0])
        expected = set(names)
        for row_number, row in enumerate(rows[1:], start=1):
            if set(row) != expected:
                missing = sorted(expected - set(row))
                extra = sorted(set(row) - expected)
                raise StudyValidationError(
                    f"record {row_number} has different fields; missing={missing}, extra={extra}"
                )

        return cls({name: [row[name] for row in rows] for name in names})

    @classmethod
    def from_dataframe(cls, frame: Any) -> Study:
        """Construct a study from a pandas-like dataframe without retaining its index.

        Column and row order are preserved. The dataframe index is deliberately ignored:
        longitudinal identity and chronology must be carried by explicit columns.
        """

        if not hasattr(frame, "columns") or not hasattr(frame, "__getitem__"):
            raise TypeError("frame must provide dataframe-like columns and column access")
        names = tuple(frame.columns)
        if not names:
            raise StudyValidationError("a dataframe must contain columns")
        if any(not isinstance(name, str) or not name for name in names):
            raise StudyValidationError("dataframe column names must be non-empty strings")
        if len(set(names)) != len(names):
            raise StudyValidationError("dataframe column names must be unique")
        return cls({name: np.asarray(frame[name]) for name in names})

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, name: str) -> NDArray[Any]:
        """Return a read-only column by name."""

        try:
            return _read_only_view(self._columns[name])
        except KeyError:
            raise KeyError(f"unknown study column: {name!r}") from None

    def __repr__(self) -> str:
        return (
            f"Study(n_trials={len(self)}, n_subjects={len(self.subjects)}, "
            f"columns={self.columns!r})"
        )

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names in source order."""

        return tuple(self._columns)

    @property
    def subjects(self) -> tuple[Any, ...]:
        """Subject identifiers in first-appearance order."""

        return self._subjects

    def chronological_indices(self) -> NDArray[np.intp]:
        """Return row positions ordered by subject, session chronology, and trial.

        Subject order follows first appearance. The returned positions can be used for an
        explicit sorted view while the study itself continues to preserve source row order.
        """

        subject_rank = {_key(subject): rank for rank, subject in enumerate(self.subjects)}
        ordered = sorted(
            range(len(self)),
            key=lambda index: (
                subject_rank[_key(self["subject"][index])],
                int(self["session_order"][index]),
                int(self["trial"][index]),
                index,
            ),
        )
        return _read_only_array(ordered, dtype=np.intp)

    def take(self, indices: Sequence[int] | NDArray[np.integer[Any]]) -> Study:
        """Return a validated study containing selected source-row positions."""

        positions = np.asarray(indices)
        if positions.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if positions.dtype.kind not in "iu":
            raise TypeError("indices must contain integers")
        return type(self)({name: values[positions] for name, values in self._columns.items()})


def _copy_columns(
    columns: Mapping[str, Sequence[Any] | NDArray[Any]],
) -> dict[str, NDArray[Any]]:
    if not isinstance(columns, Mapping):
        raise TypeError("columns must be a mapping from names to one-dimensional values")
    if not columns:
        raise StudyValidationError("a study must contain columns")

    arrays: dict[str, NDArray[Any]] = {}
    for name, values in columns.items():
        if not isinstance(name, str) or not name:
            raise StudyValidationError("column names must be non-empty strings")
        if name in ("trial", "session_order"):
            raw_values = np.asarray(values, dtype=object)
            if raw_values.ndim == 1 and any(
                isinstance(value, (bool, np.bool_)) for value in raw_values
            ):
                raise StudyValidationError(f"{name!r} must contain integers, not booleans")
        array = np.array(values, copy=True)
        if array.ndim != 1:
            raise StudyValidationError(f"column {name!r} must be one-dimensional")
        array.setflags(write=False)
        arrays[name] = array
    return arrays


def _validate_columns(arrays: Mapping[str, NDArray[Any]]) -> tuple[int, tuple[Any, ...]]:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in arrays]
    if missing_columns:
        raise StudyValidationError(f"missing required columns: {missing_columns}")

    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise StudyValidationError(f"all columns must have equal length; observed {lengths}")
    length = next(iter(lengths.values()))
    if length == 0:
        raise StudyValidationError("a study must contain at least one trial")

    for name in ("subject", "session"):
        for row, value in enumerate(arrays[name]):
            if _is_missing(value):
                raise StudyValidationError(f"{name!r} is missing at row {row}")
            try:
                hash(_key(value))
            except TypeError:
                raise StudyValidationError(
                    f"{name!r} identifiers must be hashable; row {row} is {value!r}"
                ) from None

    for name in ("trial", "session_order"):
        for row, value in enumerate(arrays[name]):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise StudyValidationError(
                    f"{name!r} must contain integers; row {row} is {value!r}"
                )
            if value < 0:
                raise StudyValidationError(f"{name!r} must be non-negative; row {row} is {value!r}")

    subjects: list[Any] = []
    seen_subjects: set[Any] = set()
    session_orders: dict[tuple[Any, Any], int] = {}
    sessions_at_order: dict[tuple[Any, int], Any] = {}
    trial_keys: set[tuple[Any, Any, int]] = set()

    for row in range(length):
        subject = _key(arrays["subject"][row])
        session = _key(arrays["session"][row])
        trial = int(arrays["trial"][row])
        session_order = int(arrays["session_order"][row])

        if subject not in seen_subjects:
            seen_subjects.add(subject)
            subjects.append(_key(arrays["subject"][row]))

        session_key = (subject, session)
        known_order = session_orders.setdefault(session_key, session_order)
        if known_order != session_order:
            raise StudyValidationError(
                "session_order must be constant within each subject/session; "
                f"{session_key!r} has both {known_order} and {session_order}"
            )

        order_key = (subject, session_order)
        known_session = sessions_at_order.setdefault(order_key, session)
        if known_session != session:
            raise StudyValidationError(
                "session_order must identify exactly one session within each subject; "
                f"subject {subject!r}, order {session_order} maps to "
                f"{known_session!r} and {session!r}"
            )

        trial_key = (subject, session, trial)
        if trial_key in trial_keys:
            raise StudyValidationError(f"duplicate subject/session/trial key: {trial_key!r}")
        trial_keys.add(trial_key)

    return length, tuple(subjects)


def _key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _read_only_array(values: Sequence[Any], *, dtype: np.dtype[Any] | type[Any]) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return _read_only_view(array)


def _read_only_view(array: NDArray[Any]) -> NDArray[Any]:
    view = array.view()
    view.setflags(write=False)
    return view
