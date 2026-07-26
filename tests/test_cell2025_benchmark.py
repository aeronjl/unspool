import csv
from pathlib import Path

from benchmarks.cell2025.benchmark import calculate_session_metrics, load_study


def _write_synthetic_source(path: Path) -> None:
    fieldnames = [
        "expRef",
        "trialNumber",
        "repeatNumber",
        "contrastLeft",
        "contrastRight",
        "choice",
        "choiceCompleteTime",
        "feedback",
        "stimulusOnsetTime",
        "isExpertMouse",
        "isShapedMouse",
        "sessionNum",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for session in range(1, 10):
            for trial in range(1, 81):
                side = -1 if trial <= 30 else 0 if trial <= 50 else 1
                writer.writerow(
                    {
                        "expRef": f"2026-01-{session:02d}_1_DAP001",
                        "trialNumber": trial,
                        "repeatNumber": 1,
                        "contrastLeft": 0.5 if side == -1 else 0.0,
                        "contrastRight": 0.5 if side == 1 else 0.0,
                        "choice": "Right" if (trial + session) % 3 else "Left",
                        "choiceCompleteTime": 2.0 + (trial % 7) / 100,
                        "feedback": "Rewarded" if trial % 4 else "Unrewarded",
                        "stimulusOnsetTime": 1.0,
                        "isExpertMouse": 1,
                        "isShapedMouse": 0,
                        "sessionNum": session,
                    }
                )


def test_cell2025_adapter_builds_a_session_aware_study(tmp_path: Path) -> None:
    source = tmp_path / "behaviour.csv"
    _write_synthetic_source(source)

    study = load_study(source)
    metrics = calculate_session_metrics(study)

    assert len(study) == 720
    assert study.subjects == ("DAP001",)
    assert study["session_order"].min() == 0
    assert study["session_order"].max() == 8
    assert study["paper_session_order"].min() == 1
    assert study["paper_session_order"].max() == 9
    assert len(metrics) == 9
    assert all(metric.subject == "DAP001" for metric in metrics)


def test_distinct_source_sessions_survive_a_paper_session_collision(tmp_path: Path) -> None:
    source = tmp_path / "collision.csv"
    fieldnames = [
        "expRef",
        "trialNumber",
        "repeatNumber",
        "contrastLeft",
        "contrastRight",
        "choice",
        "choiceCompleteTime",
        "feedback",
        "stimulusOnsetTime",
        "isExpertMouse",
        "isShapedMouse",
        "sessionNum",
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for date in ("2026-01-01", "2026-01-02"):
            for trial in range(1, 41):
                side = -1 if trial <= 15 else 0 if trial <= 25 else 1
                writer.writerow(
                    {
                        "expRef": f"{date}_1_DAP001",
                        "trialNumber": trial,
                        "repeatNumber": 1,
                        "contrastLeft": 0.5 if side == -1 else 0.0,
                        "contrastRight": 0.5 if side == 1 else 0.0,
                        "choice": "Right" if trial % 2 else "Left",
                        "choiceCompleteTime": 2.0 + (trial % 7) / 100,
                        "feedback": "Rewarded" if trial % 4 else "Unrewarded",
                        "stimulusOnsetTime": 1.0,
                        "isExpertMouse": 1,
                        "isShapedMouse": 0,
                        "sessionNum": 1,
                    }
                )

    study = load_study(source)
    metrics = calculate_session_metrics(study)

    assert len(set(study["session"])) == 2
    assert set(study["session_order"]) == {0, 1}
    assert set(study["paper_session_order"]) == {1}
    assert len(metrics) == 1
