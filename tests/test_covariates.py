"""Timestamped behavioural covariates and their validity mask."""

from __future__ import annotations

import numpy as np

from behavio.observed.covariates import BehaviorCovariate


def test_covariate_validity_excludes_nonfinite_values() -> None:
    covariate = BehaviorCovariate(
        subject="mouse-1",
        session="day-1",
        name="speed",
        time_s=np.array([0.0, 1.0]),
        values=np.array([1.0, np.nan]),
        valid=np.array([True, True]),
        unit="cm/s",
        source="pose",
        clock_id="video",
    )
    assert covariate.valid.tolist() == [True, False]
