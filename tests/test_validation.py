import numpy as np
import pytest

from unspool import Study, forward_session_splits, leave_one_session_out_splits
from unspool.validation import ValidationSplit


def longitudinal_study() -> Study:
    return Study(
        {
            "subject": ["a", "b", "a", "a", "b", "a", "a", "b"],
            "session": ["a-2", "b-1", "a-1", "a-3", "b-2", "a-1", "a-2", "b-1"],
            "trial": [0, 1, 0, 0, 0, 1, 1, 0],
            "session_order": [1, 0, 0, 2, 1, 0, 1, 0],
            "choice": [1, 0, 1, 1, 0, 0, 1, 1],
        }
    )


def one_session_study() -> Study:
    return Study(
        {
            "subject": ["a", "a"],
            "session": ["only", "only"],
            "trial": [0, 2],
            "session_order": [4, 4],
        }
    )


def test_forward_splits_are_expanding_complete_and_prospective() -> None:
    study = longitudinal_study()

    splits = forward_session_splits(study)

    assert len(splits) == 3
    first, second, subject_b = splits

    assert first.subject == "a"
    assert first.train_sessions == ("a-1",)
    assert first.test_sessions == ("a-2",)
    assert first.train_indices.tolist() == [2, 5]
    assert first.test_indices.tolist() == [0, 6]
    assert first.prospective is True
    assert first.scheme == "forward-session"

    assert second.train_sessions == ("a-1", "a-2")
    assert second.test_sessions == ("a-3",)
    assert second.train_session_orders == (0, 1)
    assert second.test_session_orders == (2,)

    assert subject_b.subject == "b"
    assert subject_b.train_indices.tolist() == [1, 7]
    assert subject_b.test_indices.tolist() == [4]


def test_forward_split_horizon_and_step_are_explicit() -> None:
    study = longitudinal_study()

    splits = forward_session_splits(study, min_train_sessions=1, horizon=2, step=2)

    assert len(splits) == 1
    assert splits[0].train_sessions == ("a-1",)
    assert splits[0].test_sessions == ("a-2", "a-3")


@pytest.mark.parametrize(
    ("argument", "value"),
    [("min_train_sessions", 0), ("horizon", -1), ("step", True)],
)
def test_forward_split_arguments_must_be_positive_integers(argument: str, value: object) -> None:
    with pytest.raises(ValueError, match=argument):
        forward_session_splits(longitudinal_study(), **{argument: value})


def test_leave_one_session_out_is_whole_session_but_not_prospective() -> None:
    study = longitudinal_study()

    splits = leave_one_session_out_splits(study)

    assert len(splits) == 5
    middle = splits[1]
    assert middle.subject == "a"
    assert middle.train_sessions == ("a-1", "a-3")
    assert middle.test_sessions == ("a-2",)
    assert middle.train_indices.tolist() == [2, 3, 5]
    assert middle.test_indices.tolist() == [0, 6]
    assert middle.prospective is False
    assert middle.scheme == "leave-one-session-out"


def test_subjects_without_enough_sessions_produce_no_folds() -> None:
    study = one_session_study()

    assert forward_session_splits(study) == ()
    assert leave_one_session_out_splits(study) == ()


def test_validation_indices_are_read_only_and_disjoint() -> None:
    split = forward_session_splits(longitudinal_study())[0]

    assert not np.intersect1d(split.train_indices, split.test_indices).size
    with pytest.raises(ValueError, match="read-only"):
        split.train_indices[0] = 99
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        split.train_indices.setflags(write=True)


def test_validation_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        ValidationSplit(
            train_indices=np.array([0, 1]),
            test_indices=np.array([1, 2]),
            subject="a",
            train_sessions=("s1",),
            test_sessions=("s2",),
            train_session_orders=(0,),
            test_session_orders=(1,),
            scheme="forward-session",
        )


def test_forward_split_rejects_reversed_chronology() -> None:
    with pytest.raises(ValueError, match="strictly before"):
        ValidationSplit(
            train_indices=np.array([0]),
            test_indices=np.array([1]),
            subject="a",
            train_sessions=("s2",),
            test_sessions=("s1",),
            train_session_orders=(1,),
            test_session_orders=(0,),
            scheme="forward-session",
        )


def test_validation_split_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="unknown validation scheme"):
        ValidationSplit(
            train_indices=np.array([0]),
            test_indices=np.array([1]),
            subject="a",
            train_sessions=("s1",),
            test_sessions=("s2",),
            train_session_orders=(0,),
            test_session_orders=(1,),
            scheme="random",  # type: ignore[arg-type]
        )
