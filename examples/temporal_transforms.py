"""Build explicit clocks and fit a behavioural landmark without test leakage."""

from __future__ import annotations

from unspool import (
    ClockKind,
    ClockSpec,
    Study,
    ThresholdLandmarkClock,
    fit_transform_splits,
    forward_session_splits,
    with_cumulative_trial_clock,
    with_elapsed_time_clock,
)


def build_study() -> Study:
    return Study(
        {
            "subject": ["mouse-1"] * 12,
            "session": ["day-1"] * 4 + ["day-2"] * 4 + ["day-3"] * 4,
            "trial": [0, 1, 2, 3] * 3,
            "session_order": [0] * 4 + [1] * 4 + [2] * 4,
            "timestamp_hours": [0, 1, 2, 3, 24, 25, 26, 27, 48, 49, 50, 51],
            "correct": [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        }
    )


def main() -> None:
    study = with_cumulative_trial_clock(build_study()).study
    study = with_elapsed_time_clock(
        study,
        source="timestamp_hours",
        source_kind="numeric",
        unit="hours",
    ).study
    landmark = ThresholdLandmarkClock(
        clock=ClockSpec(
            "cumulative_trial",
            ClockKind.CUMULATIVE_TRIAL,
            unit="observed_trial",
        ),
        metric="correct",
        output="trials_since_learning",
        threshold=1.0,
        window=2,
        consecutive=2,
    )
    splits = forward_session_splits(study, min_train_sessions=2)
    results = fit_transform_splits(landmark, study, splits)

    print("Explicit design clocks")
    print(f"cumulative trials: {study['cumulative_trial'][[0, -1]].tolist()}")
    print(f"elapsed hours:     {study['elapsed_time'][[0, -1]].tolist()}")
    print("\nTraining-fold landmark")
    for result in results:
        provenance = result.fitted_transform.provenance
        print(f"learned values: {dict(provenance.learned_values)}")
        print(f"fit trials:     {provenance.n_fit_trials}")
        print(f"test-relative:  {result.testing.study['trials_since_learning'].tolist()}")


if __name__ == "__main__":
    main()
