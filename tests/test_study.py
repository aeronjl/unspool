import numpy as np
import pytest

from behavio import REQUIRED_COLUMNS, Study, StudyValidationError


def valid_columns() -> dict[str, list[object]]:
    return {
        "subject": ["mouse-a", "mouse-a", "mouse-a", "mouse-b"],
        "session": ["late", "early", "early", "only"],
        "trial": [0, 1, 0, 0],
        "session_order": [1, 0, 0, 0],
        "choice": [1, 0, 1, 1],
    }


def test_study_preserves_source_columns_and_row_order() -> None:
    columns = valid_columns()
    study = Study.from_columns(columns)

    assert REQUIRED_COLUMNS == ("subject", "session", "trial", "session_order")
    assert study.columns == tuple(columns)
    assert study.subjects == ("mouse-a", "mouse-b")
    assert study["session"].tolist() == ["late", "early", "early", "only"]
    assert study["choice"].tolist() == [1, 0, 1, 1]
    assert repr(study).startswith("Study(n_trials=4, n_subjects=2")


def test_study_copies_input_and_exposes_read_only_columns() -> None:
    choice = np.array([1, 0, 1, 1])
    columns = valid_columns()
    columns["choice"] = choice
    study = Study(columns)

    choice[0] = 99
    assert study["choice"][0] == 1
    with pytest.raises(ValueError, match="read-only"):
        study["choice"][0] = 2
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        study["choice"].setflags(write=True)
    with pytest.raises(AttributeError, match="immutable"):
        study._length = 99


def test_chronological_indices_do_not_mutate_source_order() -> None:
    study = Study(valid_columns())

    indices = study.chronological_indices()

    assert indices.tolist() == [2, 1, 0, 3]
    assert study["session"].tolist() == ["late", "early", "early", "only"]
    with pytest.raises(ValueError, match="read-only"):
        indices[0] = 3


def test_take_returns_a_revalidated_study() -> None:
    study = Study(valid_columns())

    subset = study.take([2, 0, 3])

    assert subset["session"].tolist() == ["early", "late", "only"]
    assert subset["choice"].tolist() == [1, 1, 1]


def test_from_records_requires_consistent_fields() -> None:
    study = Study.from_records(
        [
            {"subject": "a", "session": "s1", "trial": 0, "session_order": 0},
            {"subject": "a", "session": "s1", "trial": 1, "session_order": 0},
        ]
    )
    assert len(study) == 2

    with pytest.raises(StudyValidationError, match="different fields"):
        Study.from_records(
            [
                {"subject": "a", "session": "s1", "trial": 0, "session_order": 0},
                {"subject": "a", "session": "s1", "trial": 1},
            ]
        )


def test_from_dataframe_uses_explicit_columns_and_ignores_index() -> None:
    class Frame:
        columns = tuple(valid_columns())
        index = (100, 50, 20, 10)

        def __getitem__(self, name: str) -> np.ndarray:
            return np.asarray(valid_columns()[name])

    study = Study.from_dataframe(Frame())

    assert study.columns == Frame.columns
    assert study["trial"].tolist() == [0, 1, 0, 0]
    assert "index" not in study.columns


def test_from_dataframe_maps_source_identity_columns_to_the_canonical_contract() -> None:
    source = {
        "mouse": ["a", "a"],
        "session_id": ["first", "first"],
        "trial_index": [0, 1],
        "training_day": [0, 0],
        "stimulus": [-1.0, 1.0],
    }

    class Frame:
        columns = tuple(source)

        def __getitem__(self, name: str) -> np.ndarray:
            return np.asarray(source[name])

    study = Study.from_dataframe(
        Frame(),
        subject="mouse",
        session="session_id",
        trial="trial_index",
        session_order="training_day",
    )

    assert study.columns == (*REQUIRED_COLUMNS, "stimulus")
    assert study["subject"].tolist() == ["a", "a"]
    assert study["stimulus"].tolist() == [-1.0, 1.0]


def test_from_dataframe_rejects_missing_or_colliding_mappings() -> None:
    class Frame:
        columns = tuple(valid_columns())

        def __getitem__(self, name: str) -> np.ndarray:
            return np.asarray(valid_columns()[name])

    with pytest.raises(StudyValidationError, match="missing mapped"):
        Study.from_dataframe(Frame(), subject="mouse")
    with pytest.raises(StudyValidationError, match="mappings must be unique"):
        Study.from_dataframe(Frame(), subject="subject", session="subject")


def test_from_dataframe_rejects_non_tabular_and_duplicate_columns() -> None:
    with pytest.raises(TypeError, match="dataframe-like"):
        Study.from_dataframe(object())

    class DuplicateFrame:
        columns = ("subject", "subject")

        def __getitem__(self, name: str) -> list[str]:
            return [name]

    with pytest.raises(StudyValidationError, match="unique"):
        Study.from_dataframe(DuplicateFrame())


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_required_columns_are_enforced(missing: str) -> None:
    columns = valid_columns()
    del columns[missing]

    with pytest.raises(StudyValidationError, match="missing required columns"):
        Study(columns)


def test_columns_must_have_equal_nonzero_length() -> None:
    columns = valid_columns()
    columns["choice"] = [1]
    with pytest.raises(StudyValidationError, match="equal length"):
        Study(columns)

    empty = {name: [] for name in REQUIRED_COLUMNS}
    with pytest.raises(StudyValidationError, match="at least one trial"):
        Study(empty)


@pytest.mark.parametrize("name", ["subject", "session"])
def test_canonical_identifiers_cannot_be_missing(name: str) -> None:
    columns = valid_columns()
    columns[name][0] = None

    with pytest.raises(StudyValidationError, match="is missing"):
        Study(columns)


@pytest.mark.parametrize("name", ["trial", "session_order"])
@pytest.mark.parametrize("bad_value", [-1, 1.0, True])
def test_ordinal_columns_require_nonnegative_integers(name: str, bad_value: object) -> None:
    columns = valid_columns()
    columns[name][0] = bad_value

    with pytest.raises(StudyValidationError, match=name):
        Study(columns)


def test_trial_keys_must_be_unique() -> None:
    columns = valid_columns()
    columns["trial"][1] = 0

    with pytest.raises(StudyValidationError, match="duplicate subject/session/trial"):
        Study(columns)


def test_session_order_is_constant_within_session() -> None:
    columns = valid_columns()
    columns["session_order"][1] = 2

    with pytest.raises(StudyValidationError, match="constant within each subject/session"):
        Study(columns)


def test_session_order_identifies_one_session_per_subject() -> None:
    columns = valid_columns()
    columns["session_order"][0] = 0

    with pytest.raises(StudyValidationError, match="exactly one session"):
        Study(columns)


# --------------------------------------------------------------------------------------
# Study.factorial: the crossed subject x session x trial grid of a planned design
# --------------------------------------------------------------------------------------


def test_factorial_crosses_subjects_sessions_and_trials_in_chronological_order() -> None:
    study = Study.factorial(trials=2, subjects=("mouse-a", "mouse-b"), sessions=3)

    assert len(study) == 12
    assert study.columns == REQUIRED_COLUMNS
    assert study.subjects == ("mouse-a", "mouse-b")
    assert study["subject"].tolist() == ["mouse-a"] * 6 + ["mouse-b"] * 6
    assert (
        study["session"].tolist()
        == ["session-0"] * 2
        + ["session-1"] * 2
        + ["session-2"] * 2
        + ["session-0"] * 2
        + ["session-1"] * 2
        + ["session-2"] * 2
    )
    assert study["trial"].tolist() == [0, 1] * 6
    assert study["session_order"].tolist() == [0, 0, 1, 1, 2, 2] * 2
    assert study.chronological_indices().tolist() == list(range(12))


def test_factorial_session_order_satisfies_both_study_invariants() -> None:
    study = Study.factorial(trials=2, subjects=2, sessions=3)

    by_session: dict[tuple[object, object], set[int]] = {}
    by_order: dict[tuple[object, int], set[object]] = {}
    for row in range(len(study)):
        subject = study["subject"][row]
        session = study["session"][row]
        order = int(study["session_order"][row])
        by_session.setdefault((subject, session), set()).add(order)
        by_order.setdefault((subject, order), set()).add(session)

    # Constant within a (subject, session), and injective within a subject.
    assert all(len(orders) == 1 for orders in by_session.values())
    assert all(len(sessions) == 1 for sessions in by_order.values())


def test_factorial_labels_counts_and_accepts_a_single_string_for_one_level() -> None:
    counted = Study.factorial(trials=1, subjects=2, sessions=2)
    named = Study.factorial(trials=1, subjects="synthetic-mouse", sessions="only")

    assert counted["subject"].tolist() == ["subject-0", "subject-0", "subject-1", "subject-1"]
    assert counted["session"].tolist() == ["session-0", "session-1"] * 2
    assert named["subject"].tolist() == ["synthetic-mouse"]
    assert named["session"].tolist() == ["only"]


def test_factorial_session_labels_may_be_derived_from_the_subject() -> None:
    study = Study.factorial(
        trials=1,
        subjects=("mouse-a", "mouse-b"),
        sessions=2,
        session_label=lambda subject, order: f"{subject}-day-{order + 1}",
    )

    assert study["session"].tolist() == [
        "mouse-a-day-1",
        "mouse-a-day-2",
        "mouse-b-day-1",
        "mouse-b-day-2",
    ]
    assert study["session_order"].tolist() == [0, 1, 0, 1]


def test_factorial_rejects_session_labels_that_collide_within_a_subject() -> None:
    with pytest.raises(StudyValidationError, match="unique within a subject"):
        Study.factorial(trials=1, sessions=2, session_label=lambda subject, order: "one-day")


def test_factorial_broadcasts_constants_and_takes_one_value_per_row() -> None:
    study = Study.factorial(
        trials=2,
        subjects=("mouse-a", "mouse-b"),
        columns={"lab": "lab-1", "cue": [0.1, 0.2, 0.3, 0.4]},
    )

    assert study["lab"].tolist() == ["lab-1"] * 4
    assert study["cue"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert study.columns == (*REQUIRED_COLUMNS, "lab", "cue")


def test_factorial_rejects_a_sequence_that_is_not_one_value_per_row() -> None:
    with pytest.raises(StudyValidationError, match="has 3 values; the grid has 4 rows"):
        Study.factorial(trials=2, subjects=2, columns={"cue": [0.1, 0.2, 0.3]})


def test_factorial_random_draws_are_seeded_and_reproducible() -> None:
    def draw(rng: np.random.Generator, n_rows: int) -> np.ndarray:
        return rng.normal(size=n_rows)

    first = Study.factorial(trials=5, subjects=2, sessions=2, columns={"x": draw}, seed=7)
    again = Study.factorial(trials=5, subjects=2, sessions=2, columns={"x": draw}, seed=7)
    other = Study.factorial(trials=5, subjects=2, sessions=2, columns={"x": draw}, seed=8)

    np.testing.assert_array_equal(np.asarray(first["x"]), np.asarray(again["x"]))
    assert not np.array_equal(np.asarray(first["x"]), np.asarray(other["x"]))
    # The seed alone fixes the stream, so the grid is reproducible from its arguments.
    np.testing.assert_array_equal(np.asarray(first["x"]), np.random.default_rng(7).normal(size=20))


def test_factorial_draws_consume_one_generator_in_column_order() -> None:
    study = Study.factorial(
        trials=2,
        columns={
            "first": lambda rng, n_rows: rng.normal(size=n_rows),
            "second": lambda rng, n_rows: rng.normal(size=n_rows),
        },
        seed=11,
    )

    expected = np.random.default_rng(11)
    np.testing.assert_array_equal(np.asarray(study["first"]), expected.normal(size=2))
    np.testing.assert_array_equal(np.asarray(study["second"]), expected.normal(size=2))


def test_factorial_refuses_an_unseeded_draw_and_an_unused_seed() -> None:
    with pytest.raises(StudyValidationError, match="factorial needs a seed"):
        Study.factorial(trials=2, columns={"x": lambda rng, n_rows: rng.normal(size=n_rows)})
    with pytest.raises(StudyValidationError, match="no column is a random draw"):
        Study.factorial(trials=2, columns={"x": 1.0}, seed=3)


def test_factorial_rejects_a_draw_that_returns_the_wrong_shape() -> None:
    with pytest.raises(StudyValidationError, match="must return 4 one-dimensional values"):
        Study.factorial(
            trials=4, columns={"x": lambda rng, n_rows: rng.normal(size=(2, 2))}, seed=1
        )


def test_factorial_will_not_let_a_column_override_the_required_identity() -> None:
    with pytest.raises(StudyValidationError, match="factorial builds 'session_order'"):
        Study.factorial(trials=2, columns={"session_order": [0, 0]})


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"trials": 0}, "trials must be a positive integer"),
        ({"trials": 1.5}, "trials must be a positive integer"),
        ({"trials": 2, "subjects": 0}, "subjects must be a positive integer"),
        ({"trials": 2, "sessions": ()}, "sessions must name at least one session"),
        ({"trials": 2, "subjects": ("a", "a")}, "subjects labels must be unique"),
        ({"trials": 2, "subjects": 2.5}, "must be a count, a label, or a sequence"),
    ],
)
def test_factorial_rejects_a_grid_that_cannot_be_a_study(
    keywords: dict[str, object], match: str
) -> None:
    with pytest.raises(StudyValidationError, match=match):
        Study.factorial(**keywords)  # type: ignore[arg-type]


def test_factorial_reproduces_a_hand_rolled_grid_exactly() -> None:
    n_sessions, trials_per_session = 5, 6
    n_trials = n_sessions * trials_per_session
    hand = Study(
        {
            "subject": ["synthetic-mouse"] * n_trials,
            "session": [
                f"session-{session}"
                for session in range(n_sessions)
                for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * n_sessions,
            "session_order": [
                session for session in range(n_sessions) for _ in range(trials_per_session)
            ],
            "stimulus": np.random.default_rng(2025).normal(size=n_trials),
        }
    )

    built = Study.factorial(
        trials=trials_per_session,
        subjects="synthetic-mouse",
        sessions=n_sessions,
        columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
        seed=2025,
    )

    assert built.columns == hand.columns
    for column in hand.columns:
        np.testing.assert_array_equal(np.asarray(built[column]), np.asarray(hand[column]))
