import json
from pathlib import Path

import pytest

from benchmarks.ddm_subject_scale_recovery.benchmark import run

RESULT_PATH = Path("benchmarks/ddm_subject_scale_recovery/result.json")


def test_ddm_scale_benchmark_validates_cheap_configuration_errors() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        run(repetitions=0)
    with pytest.raises(ValueError, match="seed"):
        run(repetitions=1, seed=-1)


def test_pinned_ddm_scale_result_retains_recovery_and_calibration_limits() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["repetitions"] == 8
    assert result["more_subjects_reduce_joint_scale_rmse"]
    assert set(result["regimes"]) == {"6", "12"}
    assert result["regimes"]["12"]["joint_scale_rmse"] < result["regimes"]["6"]["joint_scale_rmse"]
    for regime in result["regimes"].values():
        assert regime["scale_estimation_convergence_rate"] == 1.0
        assert regime["fit_convergence_rate"] == 1.0
        assert len(regime["runs"]) == 8
        for parameter in ("drift.stimulus", "boundary"):
            summary = regime["parameters"][parameter]
            assert 0.0 <= summary["coverage_95"] <= 1.0
            assert 0.0 <= summary["boundary_rate"] <= 1.0
