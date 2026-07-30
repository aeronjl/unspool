"""The one derivation of session boundaries and source-row restoration."""

import numpy as np
import pytest

from behavio import Study
from behavio.adapters import (
    SequenceGrouping,
    SequenceLayout,
    SequenceLayoutError,
    TrialSequence,
    sequence_layout,
)


def shuffled_study(seed: int = 0) -> Study:
    """Two subjects, two sessions each, with source rows deliberately out of order."""

    rows = []
    for subject in ("m1", "m2"):
        for order, session in enumerate(("day-2", "day-1")):
            for trial in range(4):
                rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "trial": trial,
                        "session_order": 1 - order,
                        "choice": trial % 2,
                        "value": float(trial) + (0.5 if subject == "m2" else 0.0),
                    }
                )
    generator = np.random.default_rng(seed)
    permuted = [rows[index] for index in generator.permutation(len(rows))]
    return Study.from_records(permuted)


def test_sequences_are_chronological_and_partition_the_study() -> None:
    study = shuffled_study()
    layout = sequence_layout(study)

    assert layout.n_sequences == 4
    assert layout.lengths == (4, 4, 4, 4)
    assert layout.names == ("m1/day-1", "m1/day-2", "m2/day-1", "m2/day-2")
    assert np.array_equal(layout.order, study.chronological_indices())


def test_split_then_join_restores_source_row_order_exactly() -> None:
    study = shuffled_study()
    layout = sequence_layout(study)

    for column in ("choice", "value", "trial"):
        restored = layout.join(layout.split(study[column]))
        assert np.array_equal(restored, np.asarray(study[column]))


def test_split_orders_each_sequence_by_trial_not_by_source_position() -> None:
    study = shuffled_study()
    layout = sequence_layout(study)

    for block in layout.column(study, "trial"):
        assert np.array_equal(block, np.arange(4))


def test_join_accepts_blocks_with_trailing_dimensions() -> None:
    study = shuffled_study()
    layout = sequence_layout(study)
    blocks = [np.arange(len(sequence) * 3).reshape(len(sequence), 3) for sequence in layout]

    restored = layout.join(blocks)

    assert restored.shape == (len(study), 3)
    assert np.array_equal(restored[layout.order], np.concatenate(blocks))


def test_subject_grouping_concatenates_sessions_in_chronological_order() -> None:
    study = shuffled_study()
    layout = sequence_layout(study, grouping=SequenceGrouping.SUBJECT)

    assert layout.n_sequences == 2
    assert layout.lengths == (8, 8)
    assert layout.names == ("m1", "m2")
    sessions = layout.column(study, "session")
    assert list(sessions[0]) == ["day-1"] * 4 + ["day-2"] * 4


def test_row_maps_answer_which_sequence_and_where_within_it() -> None:
    study = shuffled_study()
    layout = sequence_layout(study)

    codes = layout.sequence_of_row
    positions = layout.position_in_sequence
    for row in range(len(study)):
        sequence = layout.sequences[codes[row]]
        assert int(sequence.indices[positions[row]]) == row


def test_subject_codes_follow_first_appearance_in_the_source_table() -> None:
    study = shuffled_study()
    layout = sequence_layout(study)

    codes = layout.subject_codes(study)
    assert set(codes.tolist()) == {0, 1}
    for row in range(len(study)):
        assert study.subjects[codes[row]] == study["subject"][row]


def test_a_layout_refuses_blocks_that_do_not_match_it() -> None:
    study = shuffled_study()
    layout = sequence_layout(study)
    blocks = list(layout.split(study["choice"]))

    with pytest.raises(SequenceLayoutError, match="one block per sequence"):
        layout.join(blocks[:-1])
    with pytest.raises(SequenceLayoutError, match="rows; the sequence has"):
        layout.join([*blocks[:-1], blocks[-1][:-1]])
    with pytest.raises(SequenceLayoutError, match="one leading entry per source row"):
        layout.split(np.arange(3))


def test_a_layout_must_partition_its_rows() -> None:
    indices = np.asarray([0, 1], dtype=np.intp)
    with pytest.raises(SequenceLayoutError, match="partition every source row"):
        SequenceLayout(
            grouping=SequenceGrouping.SESSION,
            n_rows=3,
            sequences=(
                TrialSequence(subject="m1", session="d1", session_order=0, indices=indices),
            ),
        )


def test_a_trial_sequence_needs_a_session_and_an_order_together() -> None:
    with pytest.raises(SequenceLayoutError, match="session and an order"):
        TrialSequence(subject="m1", session="d1", session_order=None, indices=np.asarray([0]))


def test_the_layout_is_read_only_and_rejects_a_foreign_study() -> None:
    study = shuffled_study()
    layout = SequenceLayout.of(study)

    assert not layout.order.flags.writeable
    assert not layout.sequences[0].indices.flags.writeable
    with pytest.raises(SequenceLayoutError, match="describes 16 rows"):
        layout.column(study.take(np.arange(8)), "choice")
    with pytest.raises(TypeError, match="must be a Study"):
        sequence_layout({"subject": ["m1"]})
