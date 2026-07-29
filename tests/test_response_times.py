import numpy as np
import pytest

from behavio import ResponseTimes, ResponseTimeSpec, ResponseTimeUnit, Study


def study_with_response_times(values: list[object]) -> Study:
    return Study(
        {
            "subject": ["a"] * len(values),
            "session": ["s"] * len(values),
            "trial": list(range(len(values))),
            "session_order": [0] * len(values),
            "rt_ms": values,
        }
    )


def test_response_time_spec_validates_units_and_converts_to_seconds() -> None:
    spec = ResponseTimeSpec(column="rt_ms", unit=ResponseTimeUnit.MILLISECONDS)

    response_times = spec.read(study_with_response_times([250.0, 500.0, 750.0]))

    assert isinstance(response_times, ResponseTimes)
    assert response_times.unit is ResponseTimeUnit.MILLISECONDS
    assert response_times.seconds.tolist() == pytest.approx([0.25, 0.5, 0.75])
    assert response_times.from_seconds(np.asarray([1.0])).tolist() == [1000.0]
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        response_times.seconds.setflags(write=True)


@pytest.mark.parametrize("values", [[0.2, 0.0], [0.2, np.nan], [0.2, "slow"]])
def test_response_time_spec_rejects_nonpositive_or_nonnumeric_values(
    values: list[object],
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        ResponseTimeSpec(column="rt_ms", unit="milliseconds").read(
            study_with_response_times(values)
        )


def test_response_time_spec_rejects_missing_and_reserved_columns() -> None:
    with pytest.raises(ValueError, match="required Study column"):
        ResponseTimeSpec(column="trial")
    with pytest.raises(ValueError, match="missing response-time"):
        ResponseTimeSpec().read(study_with_response_times([100.0]))
