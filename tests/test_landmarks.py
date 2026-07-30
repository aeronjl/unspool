from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from behavio import Study, forward_session_splits
from behavio.evaluate import leave_one_session_out_splits
from behavio.time import (
    BootstrapThresholdLandmarkClock,
    ClockKind,
    ClockSpec,
    FittedStudyTransform,
    LandmarkClockSamples,
    LandmarkNotFoundError,
    StudyTransform,
    ThresholdLandmarkClock,
    fit_transform_split,
    fit_transform_splits,
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


def test_landmark_uncertainty_is_reproducible_and_retains_the_point_estimate() -> None:
    transform = threshold_transform()

    first = transform.fit_with_uncertainty(
        landmark_study(),
        n_resamples=100,
        seed=17,
        smoothing_window=2,
        interval_level=0.8,
    )
    second = transform.fit_with_uncertainty(
        landmark_study(),
        n_resamples=100,
        seed=17,
        smoothing_window=2,
        interval_level=0.8,
    )

    assert first.landmarks == {"a": 4.0}
    assert first.uncertainty is not None
    assert second.uncertainty is not None
    estimate = first.uncertainty.estimates["a"]
    np.testing.assert_allclose(
        estimate.samples,
        second.uncertainty.estimates["a"].samples,
        equal_nan=True,
    )
    assert estimate.point == 4.0
    assert estimate.samples.shape == (100,)
    assert 0.0 <= estimate.resolution_rate <= 1.0
    assert first.uncertainty.n_fit_trials == len(landmark_study())
    assert "resamples=100" in first.uncertainty.signature
    with pytest.raises(ValueError, match="read-only"):
        estimate.samples[0] = 2.0


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


def test_uncertainty_transform_never_reads_held_out_metric_values() -> None:
    fitted = threshold_transform().fit_with_uncertainty(
        landmark_study(),
        n_resamples=50,
        seed=31,
    )
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

    first = fitted.transform_samples(held_out)
    second = fitted.transform_samples(changed_metric)

    assert isinstance(first, LandmarkClockSamples)
    assert first.values.shape == (50, 2)
    assert first.clock.allow_missing is True
    np.testing.assert_allclose(first.values, second.values, equal_nan=True)
    resolved = np.isfinite(first.values[:, 0])
    np.testing.assert_allclose(first.values[resolved, 1] - first.values[resolved, 0], 1.0)
    with pytest.raises(ValueError, match="read-only"):
        first.values[0, 0] = 1.0


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


def test_unresolved_bootstrap_draws_remain_explicit() -> None:
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
    transform = ThresholdLandmarkClock(
        clock=ClockSpec("clock", ClockKind.CUMULATIVE_TRIAL),
        metric="correct",
        threshold=1.0,
        window=4,
        consecutive=4,
        on_missing="nan",
    )

    fitted = transform.fit_with_uncertainty(no_learning, n_resamples=20, seed=9)
    assert fitted.uncertainty is not None
    estimate = fitted.uncertainty.estimates["a"]

    assert estimate.point is None
    assert estimate.n_resolved == 0
    assert estimate.resolution_rate == 0.0
    assert estimate.median is None
    assert estimate.interval is None
    assert np.isnan(fitted.transform_samples(no_learning).values).all()


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


def test_bootstrap_landmark_uses_the_generic_fold_helper_without_leakage() -> None:
    study = prospective_study()
    split = forward_session_splits(study, min_train_sessions=2)[0]
    landmark = ThresholdLandmarkClock(
        clock=ClockSpec("cumulative_trial", ClockKind.CUMULATIVE_TRIAL),
        metric="correct",
        threshold=1.0,
        window=2,
    )
    transform = BootstrapThresholdLandmarkClock(
        landmark,
        n_resamples=40,
        seed=4,
        smoothing_window=2,
        interval_level=0.8,
    )

    result = fit_transform_split(transform, study, split)
    fitted = result.fitted_transform

    assert isinstance(transform, StudyTransform)
    assert fitted.provenance.n_fit_trials == 6
    assert fitted.uncertainty is not None
    assert fitted.uncertainty.n_fit_trials == 6
    assert fitted.uncertainty.estimates["a"].samples.shape == (40,)
    samples = fitted.transform_samples(study.take(split.test_indices))
    assert samples.values.shape == (40, 3)


def test_landmark_uncertainty_rejects_nonbinary_metrics() -> None:
    study = landmark_study()
    columns = {name: study[name] for name in study.columns}
    columns["correct"] = np.linspace(0.0, 1.0, len(study))

    with pytest.raises(ValueError, match="binary metric"):
        threshold_transform(on_missing="nan").fit_with_uncertainty(
            Study(columns),
            n_resamples=10,
        )


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
