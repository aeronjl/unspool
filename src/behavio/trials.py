"""Canonical trial-level data contract for longitudinal studies.

:class:`Study` is the table. :class:`SequenceLayout` is the one derivation of its session
and subject boundaries, and it lives here rather than in ``behavio.adapters`` because it is
a fact about a study's own chronology: it names nothing above this module, and a model
wrapper, a smoother and a plotting routine all need the same answer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
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

    def __getstate__(self) -> dict[str, Any]:
        """Return the columns alone, so a study can cross a process boundary.

        A study is stored as a :class:`~types.MappingProxyType` over read-only arrays, and
        a mapping proxy cannot be pickled. Default pickling of a ``__slots__`` object would
        try to send that proxy verbatim and fail with ``cannot pickle 'mappingproxy'``,
        which is a statement about a storage detail rather than about the data. Sending the
        plain column mapping instead is enough: it is the only thing
        :meth:`__init__` needs, and ``_length`` and ``_subjects`` are derived from it.
        """

        return {"columns": dict(self._columns)}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Rebuild a study from its columns, re-running the full contract.

        Reconstruction goes through the same validation and the same read-only copy that
        :meth:`__init__` performs, rather than restoring slots directly. Unpickling yields
        writeable arrays, so trusting the sent state would produce a mutable "immutable"
        study -- and a study that arrived from another process would be the one place in
        the package where the data contract had not actually been checked.
        """

        arrays = _copy_columns(state["columns"])
        length, subjects = _validate_columns(arrays)
        object.__setattr__(self, "_columns", MappingProxyType(arrays))
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_subjects", subjects)

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


class SequenceGrouping(StrEnum):
    """What counts as one contiguous sequence of trials.

    :attr:`SESSION` is the default because a session is the unit a recording, a filtered
    state estimate, and a latent-state reset are all defined over. :attr:`SUBJECT`
    concatenates a subject's sessions in ``session_order`` and is what a model that carries
    latent state across sessions -- a learner whose values persist overnight -- needs.
    """

    SESSION = "session"
    SUBJECT = "subject"


class SequenceLayoutError(ValueError):
    """Raised when per-sequence blocks do not match the layout they are joined against."""


@dataclass(frozen=True, slots=True)
class TrialSequence:
    """One contiguous run of trials and the source rows it was drawn from.

    ``indices`` are positions into the *source* study, in chronological order within the
    sequence. They are not necessarily contiguous or increasing: a study may interleave two
    subjects' rows, and Behavio never reorders a source table to make that tidy.
    """

    subject: Any
    session: Any | None
    session_order: int | None
    indices: NDArray[np.intp]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.intp)
        if indices.ndim != 1 or not indices.size:
            raise SequenceLayoutError("a trial sequence must own at least one source row")
        if len(set(indices.tolist())) != len(indices):
            raise SequenceLayoutError("a trial sequence must not repeat a source row")
        if (self.session is None) != (self.session_order is None):
            raise SequenceLayoutError("a session-grouped sequence needs a session and an order")
        indices.setflags(write=False)
        object.__setattr__(self, "indices", indices)

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def name(self) -> str:
        """A readable identifier, stable for a given study and grouping."""

        if self.session is None:
            return f"{self.subject}"
        return f"{self.subject}/{self.session}"


@dataclass(frozen=True, slots=True)
class SequenceLayout:
    """The sequence structure of one study, and the maps between rows and sequences.

    A :class:`Study` is a flat columnar table in source row order. Almost every consumer
    that leaves this package wants something else: ``ssm`` and ``dynamax`` want a list of
    per-sequence arrays, HSSM wants a dataframe with a subject code per row, and a
    state-space smoother wants to know where one recording stops and the next begins. Each
    wrapper therefore has to work out where the boundaries are, run the foreign code over
    the pieces, and put the answers back on the study's own rows -- and every wrapper that
    re-derives that independently gets it wrong in its own way. The failure is silent: a
    per-sequence array assembled in the wrong order still has the right length, and a
    prediction written back in sorted order rather than source order still validates.

    This is that derivation, written once. It is built from
    :meth:`Study.chronological_indices`, so it cannot disagree with the package's own notion
    of chronology, and its two operations are exact inverses::

        layout = sequence_layout(study)
        blocks = layout.split(study["choice"])          # per-sequence, chronological
        restored = layout.join(blocks)                  # back in source row order

    ``join(split(values)) == values`` for every column of every study, which is the
    invariant a wrapper needs and the one it is most likely to break.
    """

    grouping: SequenceGrouping
    n_rows: int
    sequences: tuple[TrialSequence, ...]

    def __post_init__(self) -> None:
        grouping = SequenceGrouping(self.grouping)
        sequences = tuple(self.sequences)
        if not sequences:
            raise SequenceLayoutError("a layout must contain at least one sequence")
        if any(not isinstance(sequence, TrialSequence) for sequence in sequences):
            raise SequenceLayoutError("sequences must be TrialSequence records")
        covered = np.concatenate([sequence.indices for sequence in sequences])
        if covered.size != self.n_rows or len(set(covered.tolist())) != self.n_rows:
            raise SequenceLayoutError("the sequences must partition every source row exactly once")
        object.__setattr__(self, "grouping", grouping)
        object.__setattr__(self, "sequences", sequences)

    @classmethod
    def of(
        cls,
        study: Study,
        *,
        grouping: SequenceGrouping | str = SequenceGrouping.SESSION,
    ) -> SequenceLayout:
        """Derive the layout of ``study`` from its declared chronology."""

        return sequence_layout(study, grouping=grouping)

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterator[TrialSequence]:
        return iter(self.sequences)

    @property
    def n_sequences(self) -> int:
        """How many contiguous sequences the study contains under this grouping."""

        return len(self.sequences)

    @property
    def lengths(self) -> tuple[int, ...]:
        """Trial count of each sequence, in sequence order."""

        return tuple(len(sequence) for sequence in self.sequences)

    @property
    def names(self) -> tuple[str, ...]:
        """Readable identifiers of each sequence, in sequence order."""

        return tuple(sequence.name for sequence in self.sequences)

    @property
    def order(self) -> NDArray[np.intp]:
        """Source-row positions of every trial, concatenated in sequence order.

        This is the permutation that turns the source table into the per-sequence view. It
        equals :meth:`Study.chronological_indices` for :attr:`SequenceGrouping.SESSION`
        and for :attr:`SequenceGrouping.SUBJECT` alike, because both group runs of that one
        ordering.
        """

        return _read_only_view(np.concatenate([sequence.indices for sequence in self.sequences]))

    @property
    def sequence_of_row(self) -> NDArray[np.intp]:
        """For each source row, the position of the sequence it belongs to."""

        codes = np.empty(self.n_rows, dtype=np.intp)
        for position, sequence in enumerate(self.sequences):
            codes[sequence.indices] = position
        return _read_only_view(codes)

    @property
    def position_in_sequence(self) -> NDArray[np.intp]:
        """For each source row, its zero-based chronological position in its sequence."""

        positions = np.empty(self.n_rows, dtype=np.intp)
        for sequence in self.sequences:
            positions[sequence.indices] = np.arange(len(sequence), dtype=np.intp)
        return _read_only_view(positions)

    def subject_codes(self, study: Study) -> NDArray[np.intp]:
        """For each source row, the index of its subject in ``study.subjects``.

        This is the ``subj_idx`` column a hierarchical foreign package asks for, derived
        rather than invented: the code is the subject's first-appearance rank, which is the
        same order :attr:`Study.subjects` reports.
        """

        self._require_study(study)
        rank = {_key(subject): index for index, subject in enumerate(study.subjects)}
        codes = np.asarray([rank[_key(value)] for value in study["subject"]], dtype=np.intp)
        return _read_only_view(codes)

    def split(self, values: Sequence[Any] | NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Cut one source-ordered column into per-sequence, chronologically ordered blocks."""

        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.n_rows:
            raise SequenceLayoutError(
                f"values must have one leading entry per source row; expected {self.n_rows}, "
                f"got {0 if array.ndim < 1 else array.shape[0]}"
            )
        return tuple(_read_only_view(array[sequence.indices]) for sequence in self.sequences)

    def column(self, study: Study, name: str) -> tuple[NDArray[Any], ...]:
        """Split one named study column, checking the study is the one this layout describes."""

        self._require_study(study)
        return self.split(study[name])

    def join(self, blocks: Sequence[Sequence[Any] | NDArray[Any]]) -> NDArray[Any]:
        """Write per-sequence blocks back onto source rows, restoring source order exactly.

        This is the inverse of :meth:`split`. It is the operation a wrapper gets wrong when
        it concatenates a foreign package's per-sequence output and returns it as if the
        study had been sorted, which is a silent misalignment whenever the source table was
        not already in chronological order.
        """

        supplied = list(blocks)
        if len(supplied) != self.n_sequences:
            raise SequenceLayoutError(
                f"expected one block per sequence ({self.n_sequences}), got {len(supplied)}"
            )
        arrays = [np.asarray(block) for block in supplied]
        for position, (array, sequence) in enumerate(zip(arrays, self.sequences, strict=True)):
            if array.ndim < 1 or array.shape[0] != len(sequence):
                observed = 0 if array.ndim < 1 else array.shape[0]
                raise SequenceLayoutError(
                    f"block {position} ({self.sequences[position].name}) has {observed} rows; "
                    f"the sequence has {len(sequence)}"
                )
        trailing = {array.shape[1:] for array in arrays}
        if len(trailing) != 1:
            raise SequenceLayoutError("every block must have the same trailing shape")
        stacked = np.concatenate(arrays, axis=0)
        restored = np.empty_like(stacked)
        restored[self.order] = stacked
        return _read_only_view(restored)

    def _require_study(self, study: Study) -> None:
        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        if len(study) != self.n_rows:
            raise SequenceLayoutError(
                f"this layout describes {self.n_rows} rows; the study has {len(study)}"
            )


def sequence_layout(
    study: Study,
    *,
    grouping: SequenceGrouping | str = SequenceGrouping.SESSION,
) -> SequenceLayout:
    """Return the contiguous trial sequences of ``study`` under one grouping.

    Sequences appear in chronological order: subjects in first-appearance order, then
    ``session_order``, then ``trial``. That is exactly the order
    :meth:`Study.chronological_indices` defines, and this function groups runs of it
    rather than sorting again, so the two can never disagree.
    """

    if not isinstance(study, Study):
        raise TypeError("study must be a Study")
    mode = SequenceGrouping(grouping)
    order = np.asarray(study.chronological_indices(), dtype=np.intp)
    subjects = study["subject"]
    sessions = study["session"]
    orders = study["session_order"]

    sequences: list[TrialSequence] = []
    start = 0
    for position in range(1, len(order) + 1):
        if position < len(order) and _same_sequence(
            mode, subjects, sessions, order[position - 1], order[position]
        ):
            continue
        block = order[start:position]
        head = int(block[0])
        sequences.append(
            TrialSequence(
                subject=_key(subjects[head]),
                session=None if mode is SequenceGrouping.SUBJECT else _key(sessions[head]),
                session_order=None if mode is SequenceGrouping.SUBJECT else int(orders[head]),
                indices=block,
            )
        )
        start = position
    return SequenceLayout(grouping=mode, n_rows=len(study), sequences=tuple(sequences))


def _same_sequence(
    mode: SequenceGrouping,
    subjects: NDArray[Any],
    sessions: NDArray[Any],
    left: np.intp,
    right: np.intp,
) -> bool:
    if _key(subjects[left]) != _key(subjects[right]):
        return False
    if mode is SequenceGrouping.SUBJECT:
        return True
    return bool(_key(sessions[left]) == _key(sessions[right]))


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
        _canonicalize_signed_zeros(array)
        array.setflags(write=False)
        arrays[name] = array
    return arrays


def _canonicalize_signed_zeros(array: NDArray[Any]) -> None:
    """Replace every ``-0.0`` in ``array`` with ``+0.0``, in place.

    IEEE 754 keeps two zeros and they compare equal, so no computation downstream of a
    study can tell them apart. Provenance can. Behavio hashes a study's *values* into
    ``FitArtifact`` content addresses through ``json.dumps``, which writes ``-0.0`` and
    ``0.0`` as different text, so a column built two arithmetically equivalent ways used
    to land on two digests: ``np.einsum`` normalises ``-0.0`` to ``+0.0`` and elementwise
    ``*`` does not, ``0.0 * -1`` is ``-0.0`` and ``0.0 * 1`` is not, and a reader that
    parses ``"-0"`` differs from one that parses ``"0"``. Three separate defects of that
    shape were fixed one at a time before the invariant was moved here, where it belongs:
    a column's digest must depend on the column's values and not on the code path that
    produced them.

    This is the *ingest* boundary and only the ingest boundary. Intermediate arrays --
    a :class:`behavio.design.FeatureBlock`, a design matrix -- are not canonicalised,
    because they are not hashed; a design term still multiplies exactly as IEEE says it
    should, and its ``-0.0`` becomes ``0.0`` only if and when it is stored in a study.
    """

    if array.dtype.kind in "fc":
        array[array == 0] = 0
        return
    if array.dtype.kind != "O":
        return
    for index, value in enumerate(array):
        if isinstance(value, (float, np.floating)) and value == 0 and np.signbit(value):
            array[index] = type(value)(0.0)


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
