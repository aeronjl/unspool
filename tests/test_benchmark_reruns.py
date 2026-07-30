"""Nightly re-execution of the offline benchmarks that publish their own numerical gate.

These tests recompute the science instead of reading it back, so they are marked slow and
deselected from the default run. The nightly workflow selects them with ``-m slow``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.provenance import PROVENANCE_KEY

pytestmark = pytest.mark.benchmark

ROOT = Path(__file__).parents[1]
GATED_OFFLINE_BENCHMARKS = (
    "landmark_uncertainty",
    "recovery_grid",
    "state_alignment",
    "trajectory_shapes",
    "weak_signal_recovery",
)


def _subprocess_environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
    }


@pytest.mark.slow
@pytest.mark.parametrize("name", GATED_OFFLINE_BENCHMARKS)
def test_offline_benchmark_reruns_and_passes_its_own_contract(name: str, tmp_path: Path) -> None:
    output = tmp_path / f"{name}.json"
    completed = subprocess.run(
        [sys.executable, "-m", f"benchmarks.{name}.benchmark", "--output", str(output)],
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=_subprocess_environment(),
        text=True,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]

    committed = json.loads((ROOT / "benchmarks" / name / "result.json").read_text(encoding="utf-8"))
    rerun = json.loads(output.read_text(encoding="utf-8"))

    assert committed["contract_passed"] is True
    assert rerun["contract_passed"] is True
    assert rerun["benchmark"] == committed["benchmark"]
    assert PROVENANCE_KEY in rerun


def test_the_slow_tier_names_only_offline_benchmarks_that_expose_a_gate() -> None:
    for name in GATED_OFFLINE_BENCHMARKS:
        directory = ROOT / "benchmarks" / name
        committed = json.loads((directory / "result.json").read_text(encoding="utf-8"))

        assert not (directory / "fetch_data.py").exists(), name
        assert not (directory / "manifest.json").exists(), name
        assert "contract_passed" in committed, name
