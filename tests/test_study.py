import numpy as np
import pytest

from unspool import REQUIRED_COLUMNS, Study, StudyValidationError


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
