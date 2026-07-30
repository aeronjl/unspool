"""The provenance block committed beside a benchmark result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.provenance import (
    HISTORICAL_LIBRARY_NAMES,
    PROTOCOL_SCHEMA_VERSION,
    PROVENANCE_KEY,
    canonical_libraries,
    environment,
    render,
)

ROOT = Path(__file__).parents[1]
RESULT_PATHS = sorted(ROOT.glob("benchmarks/*/result.json"))
PROVENANCE_FIELDS = {"schema_version", "git_describe", "python", "libraries"}
BASE_LIBRARIES = {"numpy", "scipy", "behavio"}
STAMPED_BENCHMARKS = (
    "cell2025_protocol",
    "ddm_contaminants",
    "ddm_predictive_uncertainty",
    "ddm_recovery",
    "ddm_subject_scale_recovery",
    "hierarchical_glm",
    "hierarchical_smooth_ddm",
    "ibl2021_decision_models",
    "landmark_uncertainty",
    "nested_selection",
    "recovery_grid",
    "smooth_ddm",
    "state_alignment",
    "subject_scale_recovery",
    "trajectory_recovery",
    "trajectory_shapes",
    "weak_signal_recovery",
)


def _stamp(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")).get(PROVENANCE_KEY)


@pytest.mark.parametrize("path", RESULT_PATHS, ids=[path.parent.name for path in RESULT_PATHS])
def test_any_committed_provenance_block_is_complete(path: Path) -> None:
    stamp = _stamp(path)
    if stamp is None:
        pytest.skip(f"{path.parent.name} predates the provenance contract")

    assert set(stamp) == PROVENANCE_FIELDS
    assert stamp["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert stamp["git_describe"] and stamp["git_describe"] != "unknown"
    assert stamp["python"].count(".") == 2
    assert set(canonical_libraries(stamp["libraries"])) >= BASE_LIBRARIES
    assert all(version != "not installed" for version in stamp["libraries"].values())


@pytest.mark.parametrize("name", STAMPED_BENCHMARKS)
def test_the_regenerated_benchmarks_carry_their_stamp(name: str) -> None:
    assert _stamp(ROOT / "benchmarks" / name / "result.json") is not None


def test_pre_rename_stamps_are_still_checked_under_their_historical_name() -> None:
    """Renaming the distribution must not quietly retire the library check on old stamps.

    Every committed stamp predates the ``unspool`` to ``behavio`` rename and is left as
    written. This asserts the alias is doing real work rather than sitting unused: at least
    one stamp still names the old distribution, and every stamp resolves to this package.
    """

    stamps = [stamp for stamp in map(_stamp, RESULT_PATHS) if stamp is not None]

    assert stamps
    assert any(set(stamp["libraries"]) & set(HISTORICAL_LIBRARY_NAMES) for stamp in stamps)
    assert all("behavio" in canonical_libraries(stamp["libraries"]) for stamp in stamps)


def test_a_freshly_written_stamp_records_the_current_distribution_name() -> None:
    """New results must stamp ``behavio``; only the committed history keeps the old name."""

    libraries = environment()["libraries"]

    assert "behavio" in libraries
    assert not set(libraries) & set(HISTORICAL_LIBRARY_NAMES)


def test_the_stamp_carries_no_wall_clock_nondeterminism() -> None:
    """A rerun on an unchanged tree must be byte-identical, so no timestamp may enter."""

    first = environment()
    second = environment()

    assert first == second
    assert render({"value": 1}) == render({"value": 1})
    assert not {key for key in first if "time" in key or "date" in key}


def test_stamping_an_already_stamped_result_is_refused() -> None:
    with pytest.raises(ValueError, match=PROVENANCE_KEY):
        render({PROVENANCE_KEY: {}})
