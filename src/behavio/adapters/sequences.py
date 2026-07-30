"""One derivation of session boundaries and source-row restoration, shared by every adapter.

:class:`behavio.Study` is a flat columnar table in source row order. Almost every foreign
model implementation wants something else: ``ssm`` and ``dynamax`` want a list of per-sequence
arrays, HSSM wants a dataframe with a subject code per row, and a state-space smoother wants
to know where one recording stops and the next begins. Each wrapper therefore has to work out
where the boundaries are, run the foreign code over the pieces, and put the answers back on
the study's own rows -- and every wrapper that re-derives that independently gets it wrong in
its own way. The failure is silent: a per-sequence array assembled in the wrong order still
has the right length, and a prediction written back in sorted order rather than source order
still validates.

:class:`SequenceLayout` is that derivation, written once. It is built from
:meth:`behavio.Study.chronological_indices`, so it cannot disagree with the package's own
notion of chronology, and its two operations are exact inverses:

    layout = sequence_layout(study)
    blocks = layout.split(study["choice"])          # per-sequence, chronological
    restored = layout.join(blocks)                  # back in source row order

``join(split(values)) == values`` for every column of every study, which is the invariant a
wrapper needs and the one it is most likely to break.

The helper deliberately lives in ``behavio.adapters``: it names nothing above
``behavio.trials``, and it is tooling for the author of an adapter or a model wrapper rather
than part of the estimator contract's type surface. It would sit equally well beside
:class:`~behavio.trials.Study` itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from behavio.trials import Study


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
    """The sequence structure of one study, and the maps between rows and sequences."""

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
        equals :meth:`behavio.Study.chronological_indices` for :attr:`SequenceGrouping.SESSION`
        and for :attr:`SequenceGrouping.SUBJECT` alike, because both group runs of that one
        ordering.
        """

        return _protected(np.concatenate([sequence.indices for sequence in self.sequences]))

    @property
    def sequence_of_row(self) -> NDArray[np.intp]:
        """For each source row, the position of the sequence it belongs to."""

        codes = np.empty(self.n_rows, dtype=np.intp)
        for position, sequence in enumerate(self.sequences):
            codes[sequence.indices] = position
        return _protected(codes)

    @property
    def position_in_sequence(self) -> NDArray[np.intp]:
        """For each source row, its zero-based chronological position in its sequence."""

        positions = np.empty(self.n_rows, dtype=np.intp)
        for sequence in self.sequences:
            positions[sequence.indices] = np.arange(len(sequence), dtype=np.intp)
        return _protected(positions)

    def subject_codes(self, study: Study) -> NDArray[np.intp]:
        """For each source row, the index of its subject in ``study.subjects``.

        This is the ``subj_idx`` column a hierarchical foreign package asks for, derived
        rather than invented: the code is the subject's first-appearance rank, which is the
        same order :attr:`behavio.Study.subjects` reports.
        """

        self._require_study(study)
        rank = {_key(subject): index for index, subject in enumerate(study.subjects)}
        codes = np.asarray([rank[_key(value)] for value in study["subject"]], dtype=np.intp)
        return _protected(codes)

    def split(self, values: Sequence[Any] | NDArray[Any]) -> tuple[NDArray[Any], ...]:
        """Cut one source-ordered column into per-sequence, chronologically ordered blocks."""

        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.n_rows:
            raise SequenceLayoutError(
                f"values must have one leading entry per source row; expected {self.n_rows}, "
                f"got {0 if array.ndim < 1 else array.shape[0]}"
            )
        return tuple(_protected(array[sequence.indices]) for sequence in self.sequences)

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
        return _protected(restored)

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
    :meth:`behavio.Study.chronological_indices` defines, and this function groups runs of it
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


def _key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _protected(array: NDArray[Any]) -> NDArray[Any]:
    view = array.view()
    view.setflags(write=False)
    return view


__all__ = [
    "SequenceGrouping",
    "SequenceLayout",
    "SequenceLayoutError",
    "TrialSequence",
    "sequence_layout",
]
