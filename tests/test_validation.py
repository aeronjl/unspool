import numpy as np
import pytest

from behavio import (
    Study,
    cohort_forward_session_splits,
    forward_session_splits,
    historical_cohort_forecast_splits,
    leave_one_lab_out_splits,
    leave_one_session_out_splits,
    leave_one_subject_out_splits,
    within_session_rolling_splits,
)
from behavio.validation import ValidationSplit


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


def shuffled_within_session_study() -> Study:
    return Study(
        {
            "subject": ["a", "b", "a", "a", "b", "a", "a", "a", "b", "b", "b"],
            "session": [
                "a-2",
                "b-1",
                "a-1",
                "a-2",
                "b-1",
                "a-1",
                "a-2",
                "a-2",
                "b-2",
                "b-2",
                "b-2",
            ],
            "trial": [2, 1, 0, 0, 0, 1, 3, 1, 0, 2, 1],
            "session_order": [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def population_study() -> Study:
    return Study(
        {
            "subject": ["a", "c", "b", "a", "d", "c", "b", "d"],
            "session": ["a-1", "c-1", "b-1", "a-1", "d-1", "c-1", "b-1", "d-1"],
            "trial": [0, 0, 0, 1, 0, 1, 1, 1],
            "session_order": [0] * 8,
            "lab": ["north", "south", "north", "north", "east", "south", "north", "east"],
        }
    )


def aligned_forecast_study() -> Study:
    subjects = ("d", "b", "a", "c")
    return Study(
        {
            "subject": [subject for subject in subjects for _session in range(5) for _ in range(2)],
            "session": [
                f"{subject}-{session}"
                for subject in subjects
                for session in range(5)
                for _ in range(2)
            ],
            "trial": [0, 1] * (len(subjects) * 5),
            "session_order": [
                session for _subject in subjects for session in range(5) for _ in range(2)
            ],
            "choice": [0, 1] * (len(subjects) * 5),
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


def test_cohort_forward_splits_fit_all_subjects_at_a_shared_origin() -> None:
    splits = cohort_forward_session_splits(longitudinal_study())

    assert len(splits) == 1
    split = splits[0]
    assert split.subjects == ("a", "b")
    assert split.train_sessions == {"a": ("a-1",), "b": ("b-1",)}
    assert split.test_sessions == {"a": ("a-2",), "b": ("b-2",)}
    assert split.train_session_orders == {"a": (0,), "b": (0,)}
    assert split.test_session_orders == {"a": (1,), "b": (1,)}
    assert split.train_indices.tolist() == [1, 2, 5, 7]
    assert split.test_indices.tolist() == [0, 4, 6]
    assert split.train_session_count == 1
    assert split.scheme == "cohort-forward-session"
    assert split.prospective


def test_cohort_forward_splits_can_explicitly_shrink_the_cohort() -> None:
    splits = cohort_forward_session_splits(longitudinal_study(), require_all_subjects=False)

    assert len(splits) == 2
    assert splits[0].subjects == ("a", "b")
    assert splits[1].subjects == ("a",)
    assert splits[1].train_sessions == {"a": ("a-1", "a-2")}
    assert splits[1].test_sessions == {"a": ("a-3",)}


def test_cohort_forward_split_horizon_step_and_arguments_are_explicit() -> None:
    split = cohort_forward_session_splits(
        longitudinal_study(), horizon=2, step=2, require_all_subjects=False
    )[0]

    assert split.subjects == ("a",)
    assert split.test_session_orders == {"a": (1, 2)}
    with pytest.raises(TypeError, match="require_all_subjects"):
        cohort_forward_session_splits(
            longitudinal_study(),
            require_all_subjects=1,  # type: ignore[arg-type]
        )


def test_historical_cohort_forecast_protects_new_animals_future() -> None:
    study = aligned_forecast_study()

    splits = historical_cohort_forecast_splits(
        study,
        context_session_count=2,
        horizon=2,
        n_folds=2,
    )

    assert len(splits) == 2
    first = splits[0]
    assert first.reference_subjects == ("b", "d")
    assert first.forecast_subjects == ("a", "c")
    assert first.reference_session_orders == (0, 1, 2, 3, 4)
    assert first.context_session_orders == (0, 1)
    assert first.test_session_orders == (3, 4)
    assert first.context_sessions == {"a": ("a-0", "a-1"), "c": ("c-0", "c-1")}
    assert first.test_sessions == {"a": ("a-3", "a-4"), "c": ("c-3", "c-4")}
    assert len(first.train_indices) == 28
    assert len(first.prediction_context_indices) == 8
    assert len(first.test_indices) == 8
    assert not np.intersect1d(first.train_indices, first.test_indices).size
    assert np.setdiff1d(first.prediction_context_indices, first.train_indices).size == 0
    training_subjects = study["subject"][first.train_indices]
    training_orders = study["session_order"][first.train_indices]
    assert not np.any(np.isin(training_subjects, first.forecast_subjects) & (training_orders == 2))
    assert first.scheme == "historical-cohort-session-forecast"
    assert first.prospective

    second = splits[1]
    assert second.reference_subjects == ("a", "c")
    assert second.forecast_subjects == ("b", "d")


def test_historical_cohort_forecast_requires_a_common_aligned_panel() -> None:
    study = aligned_forecast_study()
    keep = ~((study["subject"] == "d") & (study["session_order"] == 0))

    with pytest.raises(ValueError, match="common aligned session orders"):
        historical_cohort_forecast_splits(
            study.take(np.flatnonzero(keep)),
            context_session_count=2,
            horizon=2,
            n_folds=2,
        )
    with pytest.raises(ValueError, match="n_folds"):
        historical_cohort_forecast_splits(
            study,
            context_session_count=2,
            horizon=2,
            n_folds=5,
        )


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


def test_leave_one_subject_out_holds_out_complete_subjects() -> None:
    splits = leave_one_subject_out_splits(population_study())

    assert len(splits) == 4
    first = splits[0]
    assert first.scheme == "leave-one-subject-out"
    assert first.group_column == "subject"
    assert first.held_out_group == "a"
    assert first.train_subjects == ("c", "b", "d")
    assert first.test_subjects == ("a",)
    assert first.train_groups == first.train_subjects
    assert first.test_groups == first.test_subjects
    assert first.train_indices.tolist() == [1, 2, 4, 5, 6, 7]
    assert first.test_indices.tolist() == [0, 3]
    assert first.prospective


def test_leave_one_lab_out_holds_out_subjects_and_labs_together() -> None:
    splits = leave_one_lab_out_splits(population_study())

    assert len(splits) == 3
    north = splits[0]
    assert north.scheme == "leave-one-lab-out"
    assert north.group_column == "lab"
    assert north.held_out_group == "north"
    assert north.train_groups == ("south", "east")
    assert north.test_groups == ("north",)
    assert north.train_subjects == ("c", "d")
    assert north.test_subjects == ("a", "b")
    assert north.train_indices.tolist() == [1, 4, 5, 7]
    assert north.test_indices.tolist() == [0, 2, 3, 6]
    assert not np.intersect1d(north.train_subjects, north.test_subjects).size


def test_leave_one_lab_out_rejects_cross_lab_subject_leakage() -> None:
    study = population_study()
    columns = {name: study[name] for name in study.columns}
    columns["lab"] = ["north", "south", "north", "west", "east", "south", "north", "east"]

    with pytest.raises(ValueError, match="subject 'a' maps to 'north' and 'west'"):
        leave_one_lab_out_splits(Study(columns))


def test_population_splitters_require_multiple_valid_groups() -> None:
    assert leave_one_subject_out_splits(one_session_study()) == ()
    with pytest.raises(ValueError, match="does not contain lab column"):
        leave_one_lab_out_splits(population_study(), lab_column="institution")
    with pytest.raises(ValueError, match="lab_column must differ"):
        leave_one_lab_out_splits(population_study(), lab_column="subject")


def test_leave_one_lab_out_accepts_an_explicit_source_column() -> None:
    study = population_study()
    columns = {name: study[name] for name in study.columns if name != "lab"}
    columns["institution"] = study["lab"]

    splits = leave_one_lab_out_splits(Study(columns), lab_column="institution")

    assert len(splits) == 3
    assert splits[0].group_column == "institution"
    assert splits[0].held_out_group == "north"


def test_within_session_origins_expand_history_without_sorting_source_rows() -> None:
    study = shuffled_within_session_study()

    splits = within_session_rolling_splits(
        study,
        min_train_sessions=1,
        min_train_trials=2,
    )

    assert len(splits) == 3
    first, second, subject_b = splits
    assert first.subject == "a"
    assert first.scheme == "within-session-rolling-origin"
    assert first.prospective is True
    assert first.train_sessions == ("a-1", "a-2")
    assert first.test_sessions == ("a-2",)
    assert first.train_indices.tolist() == [2, 3, 5, 7]
    assert first.prediction_context_indices.tolist() == [3, 7]
    assert first.test_indices.tolist() == [0]
    assert first.origin_session == "a-2"
    assert first.origin_session_order == 1
    assert first.origin_trial == 1
    assert first.test_trials == (2,)

    assert second.train_indices.tolist() == [0, 2, 3, 5, 7]
    assert second.prediction_context_indices.tolist() == [0, 3, 7]
    assert second.test_indices.tolist() == [6]
    assert second.origin_trial == 2
    assert second.test_trials == (3,)

    assert subject_b.subject == "b"
    assert subject_b.train_indices.tolist() == [1, 4, 8, 10]
    assert subject_b.prediction_context_indices.tolist() == [8, 10]
    assert subject_b.test_indices.tolist() == [9]


def test_within_session_horizon_and_step_count_observed_trials() -> None:
    splits = within_session_rolling_splits(
        shuffled_within_session_study(),
        min_train_sessions=1,
        min_train_trials=1,
        horizon=2,
        step=2,
    )

    assert len(splits) == 2
    assert splits[0].subject == "a"
    assert splits[0].origin_trial == 0
    assert splits[0].test_trials == (1, 2)
    assert splits[1].subject == "b"
    assert splits[1].test_trials == (1, 2)


def test_within_session_can_begin_without_a_prior_complete_session() -> None:
    splits = within_session_rolling_splits(
        one_session_study(),
        min_train_sessions=0,
        min_train_trials=1,
    )

    assert len(splits) == 1
    assert splits[0].train_sessions == ("only",)
    assert splits[0].origin_trial == 0
    assert splits[0].test_trials == (2,)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("min_train_sessions", -1),
        ("min_train_sessions", True),
        ("min_train_trials", 0),
        ("horizon", -1),
        ("step", True),
    ],
)
def test_within_session_arguments_validate_counts(argument: str, value: object) -> None:
    with pytest.raises(ValueError, match=argument):
        within_session_rolling_splits(shuffled_within_session_study(), **{argument: value})


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

    within = within_session_rolling_splits(
        shuffled_within_session_study(),
        min_train_sessions=1,
        min_train_trials=2,
    )[0]
    with pytest.raises(ValueError, match="read-only"):
        within.prediction_context_indices[0] = 99


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


def test_only_within_session_splits_may_carry_prediction_context() -> None:
    with pytest.raises(ValueError, match="only valid for within-session"):
        ValidationSplit(
            train_indices=np.array([0]),
            test_indices=np.array([1]),
            subject="a",
            train_sessions=("s1",),
            test_sessions=("s2",),
            train_session_orders=(0,),
            test_session_orders=(1,),
            scheme="forward-session",
            prediction_context_indices=np.array([0]),
        )


def test_within_session_split_requires_complete_origin_metadata() -> None:
    with pytest.raises(ValueError, match="origin_session"):
        ValidationSplit(
            train_indices=np.array([0]),
            test_indices=np.array([1]),
            subject="a",
            train_sessions=("s1",),
            test_sessions=("s1",),
            train_session_orders=(0,),
            test_session_orders=(0,),
            scheme="within-session-rolling-origin",
            prediction_context_indices=np.array([0]),
        )


def test_within_session_split_rejects_complete_sessions_after_the_origin() -> None:
    with pytest.raises(ValueError, match="must precede"):
        ValidationSplit(
            train_indices=np.array([0, 1]),
            test_indices=np.array([2]),
            subject="a",
            train_sessions=("s1", "s3", "s2"),
            test_sessions=("s2",),
            train_session_orders=(0, 2, 1),
            test_session_orders=(1,),
            scheme="within-session-rolling-origin",
            prediction_context_indices=np.array([1]),
            origin_session="s2",
            origin_session_order=1,
            origin_trial=4,
            test_trials=(5,),
        )


def test_every_first_party_splitter_names_its_folds_by_scientific_coordinate() -> None:
    study = longitudinal_study()
    population = population_study()
    aligned = aligned_forecast_study()

    assert [split.identifier for split in forward_session_splits(study)] == [
        "forward-session/subject=a/forecast-sessions=1",
        "forward-session/subject=a/forecast-sessions=2",
        "forward-session/subject=b/forecast-sessions=1",
    ]
    assert [split.identifier for split in leave_one_session_out_splits(study)] == [
        "leave-one-session-out/subject=a/held-out-session=0",
        "leave-one-session-out/subject=a/held-out-session=1",
        "leave-one-session-out/subject=a/held-out-session=2",
        "leave-one-session-out/subject=b/held-out-session=0",
        "leave-one-session-out/subject=b/held-out-session=1",
    ]
    assert [split.identifier for split in within_session_rolling_splits(study)] == [
        "within-session-rolling-origin/subject=a/session=0/origin-trial=0",
        "within-session-rolling-origin/subject=a/session=1/origin-trial=0",
        "within-session-rolling-origin/subject=b/session=0/origin-trial=0",
    ]
    assert [split.identifier for split in cohort_forward_session_splits(study)] == [
        "cohort-forward-session/train-sessions=1"
    ]
    # A leave-one-out fold names the unit it holds out, and a lab fold names the column
    # the unit came from: `lab=north` and `institution=north` are different holdouts.
    assert [split.identifier for split in leave_one_subject_out_splits(population)] == [
        "leave-one-subject-out/subject=a",
        "leave-one-subject-out/subject=c",
        "leave-one-subject-out/subject=b",
        "leave-one-subject-out/subject=d",
    ]
    assert [split.identifier for split in leave_one_lab_out_splits(population)] == [
        "leave-one-lab-out/lab=north",
        "leave-one-lab-out/lab=south",
        "leave-one-lab-out/lab=east",
    ]
    forecasts = historical_cohort_forecast_splits(
        aligned, context_session_count=2, horizon=1, n_folds=2
    )
    assert [split.identifier for split in forecasts] == [
        "historical-cohort-session-forecast/fold=0-of-2",
        "historical-cohort-session-forecast/fold=1-of-2",
    ]


def test_fold_names_do_not_depend_on_position_in_the_returned_tuple() -> None:
    """A name derived from a coordinate survives filtering; a positional one would not."""

    study = longitudinal_study()
    splits = forward_session_splits(study)

    subset = tuple(split for split in splits if split.subject == "b")

    assert [split.identifier for split in subset] == [
        "forward-session/subject=b/forecast-sessions=1"
    ]
    assert subset[0].identifier == splits[-1].identifier


def test_fold_names_render_numeric_subjects_and_keep_integers_apart_from_floats() -> None:
    study = Study(
        {
            "subject": [11, 11, 12, 12],
            "session": ["a-0", "a-1", "b-0", "b-1"],
            "trial": [0, 0, 0, 0],
            "session_order": [0, 1, 0, 1],
        }
    )

    assert [split.identifier for split in forward_session_splits(study)] == [
        "forward-session/subject=11/forecast-sessions=1",
        "forward-session/subject=12/forecast-sessions=1",
    ]
    # A float subject keeps its decimal point. Rendering 11.0 as "11" would give it the
    # same fold name as integer subject 11, and the whole point of a name is to key a
    # record by it.
    float_subject = ValidationSplit(
        train_indices=np.array([0]),
        test_indices=np.array([1]),
        subject=11.0,
        train_sessions=("s0",),
        test_sessions=("s1",),
        train_session_orders=(0,),
        test_session_orders=(1,),
        scheme="forward-session",
    )
    assert float_subject.identifier == "forward-session/subject=11.0/forecast-sessions=1"
