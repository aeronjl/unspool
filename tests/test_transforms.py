from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from unspool import (
    ClockKind,
    ClockSpec,
    FittedStudyTransform,
    LandmarkNotFoundError,
    Study,
    StudyTransform,
    ThresholdLandmarkClock,
    fit_transform_split,
    fit_transform_splits,
    forward_session_splits,
    leave_one_session_out_splits,
    with_cumulative_trial_clock,
)


def landmark_study() -> Study:
    return Study(
        {
            "subject": ["a"] * 8,
            "session": ["s1"] * 5 + ["s2"] * 3,
            "trial": [0, 1, 2, 3, 4, 0, 1, 2],
            "session_order": [0] * 5 + [1] * 3,
            "clock": np.arange(8, dtype=float),
            "correct": [0, 0, 1, 1, 1, 1, 0, 0],
        }
    )


def threshold_transform(*, on_missing: str = "error") -> ThresholdLandmarkClock:
    return ThresholdLandmarkClock(
        clock=ClockSpec("clock", ClockKind.CUMULATIVE_TRIAL, unit="observed_trial"),
        metric="correct",
        output="since_learning",
        threshold=1.0,
        window=2,
        consecutive=2,
        on_missing=on_missing,  # type: ignore[arg-type]
    )


def test_threshold_landmark_is_the_first_confirmed_detection_time() -> None:
    transform = threshold_transform()

    fitted = transform.fit(landmark_study())
    result = fitted.transform(landmark_study())

    assert fitted.landmarks == {"a": 4.0}
    assert result.study["since_learning"].tolist() == [
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    assert result.clock.kind is ClockKind.LANDMARK_RELATIVE
    assert result.clock.unit == "observed_trial"


def test_fitted_state_is_immutable_and_retains_training_provenance() -> None:
    fitted = threshold_transform().fit(landmark_study())
    provenance = fitted.provenance

    assert isinstance(threshold_transform(), StudyTransform)
    assert isinstance(fitted, FittedStudyTransform)
    assert provenance.n_fit_trials == 8
    assert provenance.fit_subjects == ("a",)
    assert provenance.learned_values == {"a": 4.0}
    assert provenance.transform_signature == threshold_transform().signature
    with pytest.raises(TypeError):
        provenance.learned_values["a"] = 99.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        fitted.n_fit_trials = 99  # type: ignore[misc]


def test_transform_never_relearns_from_held_out_metric_values() -> None:
    fitted = threshold_transform().fit(landmark_study())
    held_out = Study(
        {
            "subject": ["a", "a"],
            "session": ["s3", "s3"],
            "trial": [0, 1],
            "session_order": [2, 2],
            "clock": [8.0, 9.0],
            "correct": [0, 0],
        }
    )
    changed_metric = Study(
        {
            "subject": ["a", "a"],
            "session": ["s3", "s3"],
            "trial": [0, 1],
            "session_order": [2, 2],
            "clock": [8.0, 9.0],
            "correct": [1, 1],
        }
    )

    first = fitted.transform(held_out).study["since_learning"]
    second = fitted.transform(changed_metric).study["since_learning"]

    np.testing.assert_array_equal(first, [4.0, 5.0])
    np.testing.assert_array_equal(second, first)


def test_missing_landmarks_are_errors_or_explicit_nan() -> None:
    no_learning = Study(
        {
            "subject": ["a"] * 4,
            "session": ["s1"] * 4,
            "trial": [0, 1, 2, 3],
            "session_order": [0] * 4,
            "clock": [0.0, 1.0, 2.0, 3.0],
            "correct": [0, 0, 0, 0],
        }
    )

    with pytest.raises(LandmarkNotFoundError, match="'a'"):
        threshold_transform().fit(no_learning)

    fitted = threshold_transform(on_missing="nan").fit(no_learning)
    result = fitted.transform(no_learning)
    assert fitted.landmarks == {"a": None}
    assert np.isnan(result.study["since_learning"]).all()
    assert result.clock.allow_missing is True


def test_fitted_landmark_rejects_subjects_absent_during_fit() -> None:
    fitted = threshold_transform().fit(landmark_study())
    unknown = Study(
        {
            "subject": ["b"],
            "session": ["s1"],
            "trial": [0],
            "session_order": [0],
            "clock": [0.0],
            "correct": [1],
        }
    )

    with pytest.raises(ValueError, match="not present when the landmark was fitted"):
        fitted.transform(unknown)


def prospective_study() -> Study:
    base = Study(
        {
            "subject": ["a"] * 9,
            "session": ["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3,
            "trial": [0, 1, 2] * 3,
            "session_order": [0] * 3 + [1] * 3 + [2] * 3,
            "correct": [0, 1, 1, 1, 1, 1, 0, 0, 0],
        }
    )
    return with_cumulative_trial_clock(base).study


def test_fold_helper_fits_only_on_prospective_training_rows() -> None:
    study = prospective_study()
    split = forward_session_splits(study, min_train_sessions=2)[0]
    transform = ThresholdLandmarkClock(
        clock=ClockSpec("cumulative_trial", ClockKind.CUMULATIVE_TRIAL, unit="observed_trial"),
        metric="correct",
        threshold=1.0,
        window=2,
        consecutive=1,
    )

    result = fit_transform_split(transform, study, split)

    assert result.fitted_transform.provenance.n_fit_trials == 6
    assert result.fitted_transform.provenance.learned_values == {"a": 2.0}
    assert result.testing.study["landmark_time"].tolist() == [4.0, 5.0, 6.0]


def test_fold_helper_rejects_nonprospective_splits_by_default() -> None:
    study = prospective_study()
    split = leave_one_session_out_splits(study)[0]

    with pytest.raises(ValueError, match="not prospective"):
        fit_transform_split(threshold_transform(on_missing="nan"), study, split)


def test_multiple_splits_refit_the_landmark_at_each_origin() -> None:
    study = prospective_study()
    transform = ThresholdLandmarkClock(
        clock=ClockSpec("cumulative_trial", ClockKind.CUMULATIVE_TRIAL),
        metric="correct",
        threshold=1.0,
        window=2,
    )

    results = fit_transform_splits(transform, study, forward_session_splits(study))

    assert len(results) == 2
    assert results[0].fitted_transform.provenance.learned_values == {"a": 2.0}
    assert results[1].fitted_transform.provenance.learned_values == {"a": 2.0}


def test_transform_does_not_overwrite_existing_output_column() -> None:
    with pytest.raises(ValueError, match="already exists"):
        ThresholdLandmarkClock(
            clock=ClockSpec("clock", ClockKind.CUMULATIVE_TRIAL),
            metric="correct",
            output="correct",
        ).fit(landmark_study())
