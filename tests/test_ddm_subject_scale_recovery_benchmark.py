import json
from pathlib import Path

import pytest

from benchmarks.ddm_subject_scale_recovery.benchmark import run

RESULT_PATH = Path("benchmarks/ddm_subject_scale_recovery/result.json")
NOMINAL_COVERAGE = 0.95
MEASURED_LOUIS_COVERAGE = {
    ("6", "drift.stimulus"): 1.0,
    ("6", "boundary"): 1.0,
    ("12", "drift.stimulus"): 1.0,
    ("12", "boundary"): 0.875,
}
MINIMUM_CELL_COVERAGE = 0.875
MEASURED_DRIFT_SCALE_SHRINKAGE = {"6": 0.15826942667779975, "12": 0.17411506944628036}


def test_ddm_scale_benchmark_validates_cheap_configuration_errors() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        run(repetitions=0)
    with pytest.raises(ValueError, match="seed"):
        run(repetitions=1, seed=-1)


def test_pinned_ddm_scale_result_retains_recovery_and_interval_calibration() -> None:
    """The default Louis intervals cover 0.875-1.000 of truths, at or above nominal."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["repetitions"] == 8
    assert result["scope"]["interval"] == "Louis observed information on log scale"
    assert result["more_subjects_reduce_joint_scale_rmse"]
    assert set(result["regimes"]) == {"6", "12"}
    assert result["regimes"]["12"]["joint_scale_rmse"] < result["regimes"]["6"]["joint_scale_rmse"]
    covered = 0
    cells = 0
    for name, regime in result["regimes"].items():
        assert regime["scale_estimation_convergence_rate"] == 1.0
        assert regime["fit_convergence_rate"] == 1.0
        assert len(regime["runs"]) == 8
        for parameter in ("drift.stimulus", "boundary"):
            summary = regime["parameters"][parameter]
            measured = MEASURED_LOUIS_COVERAGE[name, parameter]
            assert summary["coverage_95"] == pytest.approx(measured, abs=1e-9)
            assert summary["coverage_95"] >= MINIMUM_CELL_COVERAGE
            assert 0.0 <= summary["boundary_rate"] <= 1.0
            covered += summary["coverage_95"] * len(regime["runs"])
            cells += len(regime["runs"])

    assert covered / cells >= NOMINAL_COVERAGE


def test_pinned_ddm_scale_result_keeps_the_unresolved_drift_scale_shrinkage_visible() -> None:
    """Corrected intervals cover, but the drift-scale point estimate is still 21-28% low."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    truth = result["design"]["true_scales"]["drift.stimulus"]

    for name, regime in result["regimes"].items():
        summary = regime["parameters"]["drift.stimulus"]
        assert summary["mean_estimate"] == pytest.approx(
            MEASURED_DRIFT_SCALE_SHRINKAGE[name], abs=1e-9
        )
        assert summary["mean_estimate"] < truth
        assert summary["mean_standard_error"] > abs(summary["bias"])
