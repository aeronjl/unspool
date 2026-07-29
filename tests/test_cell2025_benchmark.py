import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.cell2025.benchmark import (
    EXPECTED,
    calculate_figure1_subject_metrics,
    calculate_session_metrics,
    contract_matches,
    load_study,
)

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "cell2025"


def _write_synthetic_source(
    path: Path,
    *,
    session_numbers: tuple[int, ...] = tuple(range(1, 10)),
) -> None:
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
        for date_index, session in enumerate(session_numbers, start=1):
            for trial in range(1, 81):
                side = -1 if trial <= 30 else 0 if trial <= 50 else 1
                writer.writerow(
                    {
                        "expRef": f"2026-01-{date_index:02d}_1_DAP001",
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
    assert set(study["signed_contrast"]) == {-0.5, 0.0, 0.5}
    assert np.all(study["left_contrast"] <= 0)
    assert np.all(study["right_contrast"] >= 0)
    assert np.all(study["response_time"] > 0)
    assert np.array_equal(study["source_trial"], study["trial"])
    assert len(metrics) == 9
    assert all(metric.subject == "DAP001" for metric in metrics)


def test_figure1_late_window_uses_paper_days_not_five_observed_rows(tmp_path: Path) -> None:
    source = tmp_path / "sparse-days.csv"
    _write_synthetic_source(source, session_numbers=(1, 2, 4, 5, 6, 7, 8, 15, 16))

    study = load_study(source)
    session_metrics = calculate_session_metrics(study)
    summary = calculate_figure1_subject_metrics(
        study,
        session_metrics=session_metrics,
    )[0]
    late_rows = [row for row in session_metrics if 11 < row.session_order <= 16]

    assert len(late_rows) == 2
    assert summary.late_bias == np.mean([row.zero_bias for row in late_rows])
    assert summary.late_slope_difference == np.mean(
        [row.right_slope - row.left_slope for row in late_rows]
    )


def test_figure1gi_audit_freezes_the_panel_contract() -> None:
    audit = json.loads((BENCHMARK / "figure1gi_audit.json").read_text(encoding="utf-8"))

    assert audit["schema_version"] == 1
    assert audit["paper"]["pdf_page"] == 3
    assert audit["released_analysis"]["commit"] == ("2faa4680d5e9c0d6a9df516e3dede8c641e39a72")
    assert audit["released_analysis"]["colour_variable"] == "prop_below"
    assert set(audit["panels"]) == {"1G", "1I"}
    assert np.isclose(
        audit["panels"]["1G"]["reproduced_r"],
        EXPECTED["early_late_bias_r"],
        rtol=1e-15,
    )
    assert np.isclose(
        audit["panels"]["1I"]["reproduced_r"],
        EXPECTED["early_bias_late_slope_r"],
        rtol=1e-15,
    )
    assert "paper-day" in audit["panels"]["1G"]["y"]
    assert audit["behavio_display"]["bootstrap_seed"] == 202501
    assert audit["behavio_display"]["classification"] == "published-parity"


def test_committed_result_carries_the_audited_figure1gi_statistics() -> None:
    result = json.loads((BENCHMARK / "result.json").read_text(encoding="utf-8"))
    audit = json.loads((BENCHMARK / "figure1gi_audit.json").read_text(encoding="utf-8"))

    assert result["contract_passed"]
    assert result["n_subjects"] == audit["source_data"]["retained_animals"] == 30
    assert result["n_trials"] == audit["source_data"]["retained_trials"] == 192_238
    assert result["source_member_sha256"] == audit["source_data"]["member_sha256"]
    assert result["early_late_bias_r"] == audit["panels"]["1G"]["reproduced_r"]
    assert result["early_late_bias_p"] == audit["panels"]["1G"]["reproduced_p"]
    assert result["early_bias_late_slope_r"] == audit["panels"]["1I"]["reproduced_r"]
    assert result["early_bias_late_slope_p"] == audit["panels"]["1I"]["reproduced_p"]
    assert contract_matches({key: result[key] for key in EXPECTED})


def test_reproduced_correlations_agree_with_the_values_printed_in_the_paper() -> None:
    """Compare the reproduction to the paper, not only to the benchmark's own pin."""

    audit = json.loads((BENCHMARK / "figure1gi_audit.json").read_text(encoding="utf-8"))
    contract = json.loads((BENCHMARK / "published_claims.json").read_text(encoding="utf-8"))
    claims = {claim["id"]: claim for claim in contract["claims"]}

    assert contract["paper"]["doi"] == audit["paper"]["doi"]
    for panel, correlation, probability in (
        ("1G", "early_late_bias_r", "early_late_bias_p"),
        ("1I", "early_bias_late_slope_r", "early_bias_late_slope_p"),
    ):
        claim = claims[correlation]
        reported = audit["panels"][panel]["reported_r"]
        reproduced = audit["panels"][panel]["reproduced_r"]

        assert claim["published_value"] == reported
        assert claim["observed_value"] == reproduced
        assert abs(reproduced - reported) < claim["tolerance"]["value"]
        assert claim["status"] == "pass"

        bound = claims[probability]
        assert audit["panels"][panel]["reported_p"] == f"p < {bound['published_value']:g}"
        assert audit["panels"][panel]["reproduced_p"] < bound["published_value"]
        assert bound["status"] == "pass"

    assert claims["n_subjects"]["published_value"] == audit["panels"]["1G"]["n"] == 30


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
