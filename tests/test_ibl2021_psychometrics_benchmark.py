"""The IBL 2021 published-parity replication: its contract, its fit, and its re-run.

The default tier runs offline in milliseconds. It checks the pinned cohort manifest, the
reimplemented psychometric fit, and the agreement between the committed ``result.json`` and
the ``published_claims.json`` contract. The nightly slow tier re-executes the benchmark
against the cached public trial tables and requires the committed gate to be reproduced.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.ibl2021_psychometrics.fetch_data import (
    DEFAULT_CACHE,
    INSTITUTIONS,
    LAB_TO_INSTITUTION,
    load_manifest,
    manifest_digest,
)
from benchmarks.ibl2021_psychometrics.psychometric import (
    CRITERION_BOX,
    DEFAULT_SEED,
    REPORTED_BOX,
    contrast_summary,
    erf_psycho_2gammas,
    fit_psychometric,
)
from benchmarks.provenance import PROVENANCE_KEY

pytestmark = pytest.mark.benchmark

ROOT = Path(__file__).parents[1]
DIRECTORY = ROOT / "benchmarks" / "ibl2021_psychometrics"
EXPECTED_SUBJECTS = 138
EXPECTED_SESSIONS = 3058


def _result() -> dict[str, object]:
    return json.loads((DIRECTORY / "result.json").read_text(encoding="utf-8"))


def _claims() -> dict[str, object]:
    return json.loads((DIRECTORY / "published_claims.json").read_text(encoding="utf-8"))


def test_the_frozen_protocol_is_committed_beside_the_benchmark() -> None:
    protocol = (DIRECTORY / "PROTOCOL.md").read_text(encoding="utf-8")

    assert "Frozen: 2026-07-28" in protocol
    assert "Prior exposure" in protocol
    assert "Amendment 1" in protocol


def test_manifest_pins_the_cohort_and_verifies_its_own_digest() -> None:
    manifest = load_manifest()
    rows = manifest["sessions"]

    assert manifest_digest(rows) == manifest["sessions_sha256"]
    assert manifest["n_subjects"] == EXPECTED_SUBJECTS
    assert manifest["n_sessions"] == EXPECTED_SESSIONS == len(rows)
    assert manifest["release_tag"] == "2021_Q1_IBL_et_al_Behaviour"
    assert manifest["licence"] == "CC-BY-4.0"
    assert {row["institution"] for row in rows} == set(INSTITUTIONS)
    assert all(row["task_protocol"].startswith("_iblrig_tasks_trainingChoiceWorld") for row in rows)


def test_every_pinned_dataset_carries_a_uuid_a_size_and_two_checksums() -> None:
    """The release index and the served objects can disagree, so both are pinned."""

    rows = load_manifest()["sessions"]

    for row in rows:
        assert len(row["md5"]) == 32, row["dataset_id"]
        assert len(row["content_md5"]) == 32, row["dataset_id"]
        assert row["file_size"] > 0 and row["content_size"] > 0, row["dataset_id"]
        assert row["dataset_path"] == "alf/_ibl_trials.table.pqt"
        assert row["session_path"].startswith(f"{row['lab']}/Subjects/{row['subject']}/")
    assert len({row["dataset_id"] for row in rows}) == len(rows)


def test_the_release_index_disagrees_with_the_served_objects_for_one_dataset() -> None:
    """A retained finding: ONE's own ``check_hash=True`` does not catch this mismatch."""

    manifest = load_manifest()
    mismatched = [
        row["dataset_id"]
        for row in manifest["sessions"]
        if row["md5"] != row["content_md5"] or row["file_size"] != row["content_size"]
    ]

    assert mismatched == manifest["index_content_mismatches"]
    assert manifest["n_index_content_mismatches"] == len(mismatched) == 1


def test_lab_to_institution_recovers_the_papers_seven_institutions() -> None:
    assert set(LAB_TO_INSTITUTION.values()) == set(INSTITUTIONS)
    assert len(INSTITUTIONS) == 7
    assert LAB_TO_INSTITUTION["hoferlab"] == LAB_TO_INSTITUTION["mrsicflogellab"] == "SWC"
    assert LAB_TO_INSTITUTION["churchlandlab"] == LAB_TO_INSTITUTION["zadorlab"] == "CSHL"


def test_the_psychometric_matches_the_published_formula() -> None:
    """``gamma + (1 - gamma - lambda) * (erf((c - mu) / sigma) + 1) / 2``."""

    parameters = np.array([0.0, 20.0, 0.05, 0.10])

    assert erf_psycho_2gammas(parameters, np.array([0.0]))[0] == pytest.approx(0.05 + 0.85 * 0.5)
    assert erf_psycho_2gammas(parameters, np.array([-1e6]))[0] == pytest.approx(0.05, abs=1e-9)
    assert erf_psycho_2gammas(parameters, np.array([1e6]))[0] == pytest.approx(0.90, abs=1e-9)


def test_the_fit_recovers_parameters_it_generated() -> None:
    levels = np.array([-100.0, -50.0, -25.0, -12.5, 0.0, 12.5, 25.0, 50.0, 100.0])
    truth = np.array([3.0, 15.0, 0.06, 0.08])
    counts = np.full(levels.size, 4000.0)
    proportions = erf_psycho_2gammas(truth, levels)

    estimate = fit_psychometric(levels, counts, proportions, box=REPORTED_BOX)

    assert estimate == pytest.approx(truth, abs=0.5)


def test_the_two_published_boxes_differ_where_the_released_code_differs() -> None:
    contrasts = np.array([-100.0, 0.0, 100.0])

    assert CRITERION_BOX.minimum(contrasts)[1] == 0.0
    assert CRITERION_BOX.maximum(contrasts)[1] == 100.0
    assert REPORTED_BOX.minimum(contrasts)[1] == 5.0
    assert REPORTED_BOX.maximum(contrasts)[1] == 40.0
    assert CRITERION_BOX.start(contrasts)[0] == 0.0
    assert REPORTED_BOX.start(contrasts)[0] == 0.0


def test_the_fit_is_deterministic_under_its_pinned_seed() -> None:
    levels = np.array([-100.0, -25.0, 0.0, 25.0, 100.0])
    counts = np.full(levels.size, 300.0)
    proportions = np.array([0.03, 0.2, 0.5, 0.78, 0.95])

    first = fit_psychometric(levels, counts, proportions, box=REPORTED_BOX, seed=DEFAULT_SEED)
    second = fit_psychometric(levels, counts, proportions, box=REPORTED_BOX, seed=DEFAULT_SEED)

    assert np.array_equal(first, second)


def test_the_fit_declines_when_too_few_contrasts_were_presented() -> None:
    """``fit_psychfunc`` returns NaN below four unique stimulus levels; so does this."""

    levels = np.array([-100.0, 0.0, 100.0])
    estimate = fit_psychometric(
        levels, np.full(3, 100.0), np.array([0.05, 0.5, 0.95]), box=REPORTED_BOX
    )

    assert np.all(np.isnan(estimate))


def test_contrast_summary_ignores_no_go_trials() -> None:
    signed = np.array([-25.0, -25.0, 0.0, 25.0, 25.0])
    rightward = np.array([0.0, np.nan, 1.0, 1.0, 1.0])

    levels, counts, proportions = contrast_summary(signed, rightward)

    assert np.array_equal(levels, np.array([-25.0, 0.0, 25.0]))
    assert np.array_equal(counts, np.array([1.0, 1.0, 2.0]))
    assert np.array_equal(proportions, np.array([0.0, 1.0, 1.0]))


def test_the_committed_result_registers_the_gate_and_its_classification() -> None:
    result = _result()

    assert result["contract_passed"] is False
    assert result["classification"] == "failed-parity"
    assert result["n_institutions"] == 7
    assert result["rightward_choice_code"] == -1
    assert result["licence"] == "CC-BY-4.0"
    assert PROVENANCE_KEY in result


def test_the_classification_follows_the_gate() -> None:
    result = _result()
    expected = "published-parity" if result["contract_passed"] else "failed-parity"

    assert result["classification"] == expected


def test_the_contract_and_the_result_agree_claim_by_claim() -> None:
    result = _result()
    statuses = result["claim_status"]

    for claim in _claims()["claims"]:
        if claim["status"] == "waived":
            assert claim["id"] not in statuses, claim["id"]
            continue
        assert statuses[claim["id"]] == claim["status"], claim["id"]
    assert set(statuses) == {
        claim["id"] for claim in _claims()["claims"] if claim["status"] != "waived"
    }


def test_the_single_retained_failure_is_the_cohort_size() -> None:
    """The honest failure is kept: the public release exposes far fewer mice than n = 140."""

    failed = [claim["id"] for claim in _claims()["claims"] if claim["status"] == "fail"]

    assert failed == ["n_proficient_subjects"]


def test_the_seed_sensitivity_diagnostic_is_negligible() -> None:
    """Pinning the restart seed must not be doing the reproduction's work."""

    assert _result()["threshold_seed_sensitivity_pct"] < 0.01


def _cache_is_populated() -> bool:
    manifest_path = DIRECTORY / "manifest.json"
    if not manifest_path.is_file() or not DEFAULT_CACHE.is_dir():
        return False
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))["sessions"]
    return all(
        (DEFAULT_CACHE / row["session_path"] / row["dataset_path"]).is_file() for row in rows
    )


@pytest.mark.slow
def test_the_benchmark_reruns_and_reproduces_its_committed_gate(tmp_path: Path) -> None:
    pytest.importorskip("one", reason="the IBL replication requires behavio[ibl]")
    if not _cache_is_populated():
        pytest.skip("run `python -m benchmarks.ibl2021_psychometrics.fetch_data` first")

    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ibl2021_psychometrics.benchmark",
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src")))},
        text=True,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    committed = _result()
    rerun = json.loads(output.read_text(encoding="utf-8"))

    assert rerun["contract_passed"] == committed["contract_passed"]
    assert rerun["classification"] == committed["classification"]
    assert rerun["claim_status"] == committed["claim_status"]
    assert rerun["n_proficient_subjects"] == committed["n_proficient_subjects"]
    assert rerun["manifest_sha256"] == committed["manifest_sha256"]
    for name in (
        "threshold_during_training_pct",
        "easy_trial_error_pct_at_proficiency",
        "threshold_pct_at_proficiency",
        "training_days_to_proficiency",
        "training_kilotrials_to_proficiency",
    ):
        assert rerun[name] == pytest.approx(committed[name], rel=1e-9), name
    assert PROVENANCE_KEY in rerun


def test_the_retained_failure_carries_its_own_diagnosis() -> None:
    """The shortfall is data availability, not a criterion this benchmark got wrong."""

    result = _result()
    short = result["n_non_proficient_with_too_few_sessions"]
    behavioural = result["n_non_proficient_failing_behavioural_gates"]
    borderline = result["n_non_proficient_failing_only_psychometric_bounds"]

    assert result["n_proficient_subjects"] + short + behavioural + borderline == 138
    assert behavioural == 41
    assert borderline == 10
    assert result["proficiency_rate"] == pytest.approx(84 / 138)


def test_recruiting_every_borderline_mouse_would_still_miss_the_published_cohort() -> None:
    """Bounds how much of the failure could possibly be implementation rather than data."""

    result = _result()
    optimistic = (
        result["n_proficient_subjects"]
        + result["n_non_proficient_failing_only_psychometric_bounds"]
        + result["n_non_proficient_with_too_few_sessions"]
    )

    assert optimistic < 100
    assert optimistic < 0.95 * 140
