"""Canonical trial-level data contract for longitudinal studies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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

    Additional source columns are retained unchanged. Row order is also retained; Behavio
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
    def from_dataframe(
        cls,
        frame: Any,
        *,
        subject: str = "subject",
        session: str = "session",
        trial: str = "trial",
        session_order: str = "session_order",
    ) -> Study:
        """Construct a study from a pandas-like dataframe without retaining its index.

        The four keyword arguments map source columns onto Behavio's canonical identity
        and chronology names. Column and row order are preserved, with mapped columns
        renamed in place. The dataframe index is deliberately ignored: longitudinal
        identity and chronology must be carried by explicit columns.
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
        mapping = {
            "subject": subject,
            "session": session,
            "trial": trial,
            "session_order": session_order,
        }
        if any(not isinstance(name, str) or not name for name in mapping.values()):
            raise StudyValidationError("dataframe column mappings must be non-empty strings")
        if len(set(mapping.values())) != len(mapping):
            raise StudyValidationError("dataframe column mappings must be unique")
        missing = [source for source in mapping.values() if source not in names]
        if missing:
            raise StudyValidationError(f"dataframe is missing mapped columns: {missing}")

        canonical_by_source = {source: canonical for canonical, source in mapping.items()}
        renamed = tuple(canonical_by_source.get(name, name) for name in names)
        if len(set(renamed)) != len(renamed):
            raise StudyValidationError(
                "dataframe column mapping collides with an existing canonical column"
            )
        return cls({canonical_by_source.get(name, name): np.asarray(frame[name]) for name in names})

    @classmethod
    def factorial(
        cls,
        *,
        trials: int,
        subjects: int | str | Sequence[Any] = 1,
        sessions: int | str | Sequence[Any] = 1,
        session_label: Callable[[Any, int], Any] | None = None,
        columns: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> Study:
        """Construct the fully crossed subject x session x trial grid of a planned design.

        This is the design a simulation, a recovery study, or a worked example starts from:
        every subject runs every session, and every session runs the same number of trials.
        Rows are emitted subject-major, then in session order, then in trial order, which
        is the chronological order :meth:`chronological_indices` would return.

        ``subjects`` and ``sessions`` are either a count, which labels them ``subject-0``
        and ``session-0`` upwards, a single string, or an explicit sequence of labels.
        ``session_label`` overrides session naming with ``label(subject, order)`` for
        designs where each subject's sessions carry the subject's name; its results must
        stay unique within a subject, because ``session_order`` must identify exactly one
        session per subject.

        ``session_order`` is the zero-based position of a session within its subject, so it
        is constant inside a ``(subject, session)`` pair and injective within a subject --
        the two invariants :class:`Study` enforces.

        ``columns`` adds per-trial columns. A value is either a constant, broadcast to
        every row; a sequence of exactly one value per row; or a *draw*, a callable
        ``draw(generator, n_rows)`` that receives a seeded :class:`numpy.random.Generator`.
        Draws require ``seed`` and consume that one generator in ``columns`` order, so a
        grid is reproducible from its arguments alone and no unseeded global stream can
        reach it.
        """

        _positive_integer(trials, "trials")
        subject_labels = _grid_labels(subjects, "subjects", "subject")
        session_defaults = _grid_labels(sessions, "sessions", "session")
        n_rows = len(subject_labels) * len(session_defaults) * trials

        subject_values: list[Any] = []
        session_values: list[Any] = []
        trial_values: list[int] = []
        order_values: list[int] = []
        for subject in subject_labels:
            seen: set[Any] = set()
            for order, default in enumerate(session_defaults):
                label = default if session_label is None else session_label(subject, order)
                if _is_missing(label):
                    raise StudyValidationError("session labels must not be missing")
                try:
                    key = _key(label)
                    duplicate = key in seen
                except TypeError:
                    raise StudyValidationError("session labels must be hashable") from None
                if duplicate:
                    raise StudyValidationError(
                        "session labels must be unique within a subject; "
                        f"subject {subject!r} repeats {label!r}"
                    )
                seen.add(key)
                for trial in range(trials):
                    subject_values.append(subject)
                    session_values.append(label)
                    trial_values.append(trial)
                    order_values.append(order)

        grid: dict[str, Any] = {
            "subject": subject_values,
            "session": session_values,
            "trial": trial_values,
            "session_order": order_values,
        }
        generator: np.random.Generator | None = None
        for name, value in (columns or {}).items():
            if not isinstance(name, str) or not name:
                raise StudyValidationError("column names must be non-empty strings")
            if name in REQUIRED_COLUMNS:
                raise StudyValidationError(f"factorial builds {name!r}; it cannot be supplied")
            if callable(value):
                if seed is None:
                    raise StudyValidationError(
                        f"column {name!r} is a random draw, so factorial needs a seed"
                    )
                if generator is None:
                    generator = np.random.default_rng(seed)
                grid[name] = _drawn_column(value, generator, n_rows, name)
            elif isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
                grid[name] = [value] * n_rows
            else:
                if len(value) != n_rows:
                    raise StudyValidationError(
                        f"column {name!r} has {len(value)} values; the grid has {n_rows} rows"
                    )
                grid[name] = value
        if seed is not None and generator is None:
            raise StudyValidationError("seed was supplied but no column is a random draw")
        return cls(grid)

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


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise StudyValidationError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _grid_labels(value: int | str | Sequence[Any], name: str, prefix: str) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        count = _positive_integer(value, name)
        return tuple(f"{prefix}-{index}" for index in range(count))
    if not isinstance(value, (Sequence, np.ndarray)):
        raise StudyValidationError(f"{name} must be a count, a label, or a sequence of labels")
    labels = tuple(_key(item) for item in value)
    if not labels:
        raise StudyValidationError(f"{name} must name at least one {prefix}")
    if any(_is_missing(label) for label in labels):
        raise StudyValidationError(f"{name} must not contain missing labels")
    try:
        unique = len(set(labels)) == len(labels)
    except TypeError:
        raise StudyValidationError(f"{name} labels must be hashable") from None
    if not unique:
        raise StudyValidationError(f"{name} labels must be unique")
    return labels


def _drawn_column(
    draw: Callable[[np.random.Generator, int], Any],
    generator: np.random.Generator,
    n_rows: int,
    name: str,
) -> NDArray[Any]:
    values = np.asarray(draw(generator, n_rows))
    if values.ndim != 1 or len(values) != n_rows:
        raise StudyValidationError(
            f"the draw for column {name!r} must return {n_rows} one-dimensional values"
        )
    return values


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
