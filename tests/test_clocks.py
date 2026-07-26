import numpy as np
import pytest

from unspool import (
    ClockKind,
    ClockScope,
    ClockSpec,
    ClockValidationError,
    Study,
    session_order_clock,
    with_cumulative_trial_clock,
    with_elapsed_time_clock,
)


def shuffled_study() -> Study:
    return Study(
        {
            "subject": ["a", "b", "a", "a", "b", "a"],
            "session": ["a-2", "b-1", "a-1", "a-2", "b-1", "a-1"],
            "trial": [0, 1, 1, 1, 0, 0],
            "session_order": [1, 0, 0, 1, 0, 0],
            "timestamp": [
                "2025-01-03",
                "2025-02-01T12:00",
                "2025-01-01T12:00",
                "2025-01-03T12:00",
                "2025-02-01",
                "2025-01-01",
            ],
            "exposure_hours": [48.0, 12.0, 12.0, 60.0, 0.0, 0.0],
        }
    )


def test_session_order_clock_validates_the_canonical_clock() -> None:
    clock = session_order_clock()

    assert clock.validate(shuffled_study()) is clock
    assert clock.kind is ClockKind.SESSION_ORDER
    assert clock.scope is ClockScope.SUBJECT
    assert clock.unit == "session"


def test_cumulative_trial_clock_uses_chronology_and_preserves_rows() -> None:
    original = shuffled_study()

    result = with_cumulative_trial_clock(original, start=1)

    assert result.study["session"].tolist() == original["session"].tolist()
    assert result.study["cumulative_trial"].tolist() == [3, 2, 2, 4, 1, 1]
    assert result.clock.kind is ClockKind.CUMULATIVE_TRIAL
    assert result.clock.unit == "observed_trial"


def test_elapsed_datetime_clock_has_a_subject_specific_origin() -> None:
    result = with_elapsed_time_clock(
        shuffled_study(), source="timestamp", output="elapsed_days", unit="days"
    )

    assert result.study["elapsed_days"].tolist() == [2.0, 0.5, 0.5, 2.5, 0.0, 0.0]
    assert result.clock.column == "elapsed_days"
    assert result.clock.kind is ClockKind.ELAPSED_TIME
    assert result.clock.unit == "days"


def test_elapsed_numeric_clock_assumes_the_requested_unit() -> None:
    result = with_elapsed_time_clock(
        shuffled_study(),
        source="exposure_hours",
        output="elapsed_hours",
        source_kind="numeric",
        unit="hours",
    )

    assert result.study["elapsed_hours"].tolist() == [48.0, 12.0, 12.0, 60.0, 0.0, 0.0]


def test_clock_validation_rejects_decreasing_values() -> None:
    study = Study(
        {
            "subject": ["a", "a"],
            "session": ["s1", "s2"],
            "trial": [0, 0],
            "session_order": [0, 1],
            "bad_clock": [2.0, 1.0],
        }
    )

    with pytest.raises(ClockValidationError, match="decreases within subject"):
        ClockSpec("bad_clock", ClockKind.CUSTOM).validate(study)


def test_categorical_task_phase_must_be_declared_as_such() -> None:
    study = Study(
        {
            "subject": ["a", "a"],
            "session": ["s1", "s2"],
            "trial": [0, 0],
            "session_order": [0, 1],
            "phase": ["training", "reversal"],
        }
    )
    phase = ClockSpec(
        "phase",
        ClockKind.TASK_PHASE,
        scope=ClockScope.GLOBAL,
        numeric=False,
        monotonic_within_subject=False,
    )

    assert phase.validate(study) is phase


@pytest.mark.parametrize("start", [-1, True, 1.5])
def test_cumulative_trial_start_must_be_a_nonnegative_integer(start: object) -> None:
    with pytest.raises(ValueError, match="start"):
        with_cumulative_trial_clock(shuffled_study(), start=start)  # type: ignore[arg-type]


def test_clock_builders_do_not_overwrite_existing_columns() -> None:
    study = shuffled_study()

    with pytest.raises(ValueError, match="already exists"):
        with_cumulative_trial_clock(study, output="timestamp")
    with pytest.raises(ValueError, match="required Study column"):
        with_elapsed_time_clock(study, source="timestamp", output="trial")


def test_elapsed_time_rejects_missing_or_reversed_sources() -> None:
    reversed_time = Study(
        {
            "subject": ["a", "a"],
            "session": ["s1", "s2"],
            "trial": [0, 0],
            "session_order": [0, 1],
            "time": [2.0, 1.0],
        }
    )

    with pytest.raises(ClockValidationError, match="missing elapsed-time source"):
        with_elapsed_time_clock(reversed_time, source="absent")
    with pytest.raises(ClockValidationError, match="decreases within subject"):
        with_elapsed_time_clock(reversed_time, source="time", source_kind="numeric")


def test_clock_missingness_is_explicit() -> None:
    study = Study(
        {
            "subject": ["a", "a"],
            "session": ["s1", "s1"],
            "trial": [0, 1],
            "session_order": [0, 0],
            "clock": [0.0, np.nan],
        }
    )

    with pytest.raises(ClockValidationError, match="finite values"):
        ClockSpec("clock", ClockKind.CUSTOM).validate(study)
    ClockSpec("clock", ClockKind.CUSTOM, allow_missing=True).validate(study)
