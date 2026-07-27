from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from benchmarks.chen2021_bandit.benchmark import contract_matches, load_study
from benchmarks.chen2021_bandit.fetch_data import _safe_extract

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "benchmarks" / "chen2021_bandit" / "result.json"
HEADER = ",left,right,choice,reward,state,RT,retrieval,initiation\n"
ROW = "0,0.3,0.7,2,1,1,0.4,0.2,0.1\n"


def _source_tree(root: Path) -> Path:
    source = root / "cleaned up restless final data"
    for session in range(1, 9):
        directory = source / f"session{session}"
        directory.mkdir(parents=True)
        for mouse in range(1, 33):
            (directory / f"{mouse}.csv").write_text(HEADER + ROW, encoding="utf-8")
    return source


def test_source_tree_maps_to_canonical_longitudinal_study(tmp_path: Path) -> None:
    study = load_study(_source_tree(tmp_path), trials_per_session=1)

    assert len(study) == 8 * 32
    assert len(study.subjects) == 32
    mouse_sessions = tuple(dict.fromkeys(study["session"][study["subject"] == "mouse-01"].tolist()))
    assert mouse_sessions == tuple(f"session-{value}" for value in range(1, 9))
    assert study["session_order"].tolist()[:2] == [0, 0]
    assert set(study["choice"].tolist()) == {1}
    assert set(study["sex"].tolist()) == {"male", "female"}
    assert set(study["source_state"].tolist()) == {1}


def test_source_tree_rejects_schema_drift(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    (source / "session1" / "1.csv").write_text("left,right\n0.3,0.7\n")

    with pytest.raises(ValueError, match="unexpected columns"):
        load_study(source, trials_per_session=1)


def test_safe_extractor_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.csv", "unsafe")

    with pytest.raises(RuntimeError, match="unsafe ZIP member"):
        _safe_extract(archive, tmp_path / "output")


def test_committed_result_preserves_the_scientific_contract() -> None:
    payload = json.loads(RESULT.read_text())

    assert payload["contract_passed"]
    assert contract_matches(payload)
    assert payload["data"]["archive_sha256"] == (
        "90f0f9fa843a16788d0dcd7b857f81db068e8d18b8dd4eabf20ccaee3b67db04"
    )
    assert payload["design"]["n_trials"] == 25_279
    assert payload["design"]["n_subjects"] == 32
    assert payload["design"]["source_state_role"].endswith("never treated as ground truth")

    comparison = payload["comparison"]
    assert comparison["bootstrap"] == {
        "confidence_level": 0.95,
        "interval": "percentile",
        "resamples": 5_000,
        "seed": 2_402,
        "unit": "subject",
    }
    difference = comparison["pairwise_log_loss_differences"][
        "win-stay-lose-shift_minus_q-learning"
    ]["left_minus_right"]
    assert difference["estimate"] == pytest.approx(0.00852669693425429)
    assert difference["lower"] < 0 < difference["upper"]


def test_committed_result_retains_recovery_warnings_and_aligned_figure_data() -> None:
    payload = json.loads(RESULT.read_text())
    recovery = payload["recovery"]

    assert recovery["confusion"]["counts"] == [[5, 0, 0], [0, 5, 0]]
    assert sum(status == "warning" for row in recovery["audit_statuses"] for status in row) == 4
    assert all(
        codes == ["boundary_estimate"]
        for row, statuses in zip(
            recovery["audit_issue_codes"], recovery["audit_statuses"], strict=True
        )
        for codes, status in zip(row, statuses, strict=True)
        if status == "warning"
    )

    example = payload["example_heldout_session"]
    lengths = {
        len(example[name])
        for name in (
            "trial",
            "choice",
            "reward",
            "reward_probability_0",
            "reward_probability_1",
            "q_value_0",
            "q_value_1",
        )
    }
    assert lengths == {100}
