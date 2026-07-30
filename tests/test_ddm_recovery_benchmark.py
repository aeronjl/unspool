import json
from pathlib import Path

import pytest

from benchmarks.ddm_recovery.benchmark import run

pytestmark = pytest.mark.benchmark


def test_pinned_ddm_recovery_result_retains_runs_and_resolution_limit() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "ddm_recovery" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["scored_columns"] == ["choice", "response_time"]
    assert result["repeats_per_design"] == 20
    assert result["rmse_decreased_for_every_parameter"] is True
    assert result["simulation"]["time_step_seconds"] == 0.0001
    for design in result["designs"].values():
        assert design["n_runs"] == 20
        assert design["audit_rates"] == {"fail": 0.0, "pass": 1.0, "warning": 0.0}
        assert len(design["runs"]) == 20
        assert all(row["correlation"] is None for row in design["summary"])


def test_ddm_recovery_benchmark_is_reproducible_at_small_scale() -> None:
    first = run(repeats=2, seed=41)
    second = run(repeats=2, seed=41)

    assert first == second
    assert all(
        design["n_runs"] == 2 and design["audit_rates"]["fail"] == 0.0
        for design in first["designs"].values()
    )
