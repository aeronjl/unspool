"""Offline checks for the Ashwood 2022 GLM-HMM replication, plus its nightly re-run.

The fast tests never touch the 218 MiB public archive: they exercise the coding rules,
state ordering and scoring arithmetic on synthetic input, so a silent change to Ashwood's
preprocessing conventions fails in milliseconds. The re-run test is marked slow and skips
when the archive has not been fetched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from behavio import Study
from benchmarks.ashwood2022_glmhmm.benchmark import (
    CLAIM_CHECKS,
    COVARIATES,
    EXAMPLE_ANIMAL,
    L2,
    PRIOR_SIGMA,
    PUBLISHED,
    Cohort,
    SessionRecord,
    _remap_choice,
    animal_study,
    bits_per_trial,
    claim_classification,
    contract_failures,
    paper_state_order,
    session_folds,
)
from benchmarks.ashwood2022_glmhmm.fetch_data import ARCHIVE_SHA256, DEFAULT_DESTINATION
from benchmarks.provenance import PROVENANCE_KEY

pytestmark = pytest.mark.benchmark

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "ashwood2022_glmhmm"


def _cohort(*, n_sessions: int = 3, violations: tuple[int, ...] = ()) -> Cohort:
    generator = np.random.default_rng(0)
    records = []
    for index in range(n_sessions):
        contrast = generator.choice([-1.0, -0.25, 0.0, 0.25, 1.0], size=10)
        choice = (contrast > 0).astype(np.int64)
        if index == 0:
            for position in violations:
                choice[position] = -1
        records.append(
            SessionRecord(
                animal="M1",
                lab="testlab",
                session=f"M1-2020-01-{index + 1:02d}-001",
                date=f"2020-01-{index + 1:02d}",
                number="001",
                signed_contrast=contrast,
                choice=choice,
                reward=np.where(choice == (contrast > 0).astype(np.int64), 1.0, -1.0),
            )
        )
    return Cohort(sessions=tuple(records), animals=("M1",), stimulus_mean=0.0, stimulus_scale=1.0)


def test_choice_coding_follows_the_reference_implementation() -> None:
    """Clockwise is left, counter-clockwise is right, no-response is a violation."""

    remapped = _remap_choice(np.asarray([1, -1, 0, 1, -1]))

    assert remapped.tolist() == [0, 1, -1, 0, 1]


def test_the_prior_scale_is_the_paper_s_sigma() -> None:
    assert PRIOR_SIGMA == 2.0
    assert pytest.approx(1.0 / PRIOR_SIGMA**2) == L2


def test_violation_rows_are_dropped_and_history_carries_across_them() -> None:
    """Removing a violation must leave the win-stay covariate on the last real choice."""

    study = animal_study(_cohort(n_sessions=1, violations=(3,)), "M1")
    complete = animal_study(_cohort(n_sessions=1), "M1")

    assert len(study) == len(complete) - 1
    assert set(np.unique(study["wsls"]).tolist()) <= {-1.0, 0.0, 1.0}
    assert float(study["wsls"][0]) == 0.0, "the first trial of a session has no history"


def test_study_columns_carry_session_chronology() -> None:
    study = animal_study(_cohort(n_sessions=4), "M1")

    assert isinstance(study, Study)
    assert study.subjects == ("M1",)
    assert sorted(set(study["session_order"].tolist())) == [0, 1, 2, 3]
    assert set(COVARIATES) <= set(study.columns)


def test_paper_state_order_is_engaged_then_biased_left_then_biased_right() -> None:
    """Engaged has the largest stimulus weight; biased-left has the smaller bias."""

    # columns are (intercept, stimulus, wsls, choice_lag_1)
    coefficients = np.asarray(
        [
            [+1.5, 0.4, 0.1, 0.1],  # weak stimulus, high bias -> biased right
            [-2.0, 0.3, 0.1, 0.1],  # weak stimulus, low bias -> biased left
            [+0.1, 8.0, 0.1, 0.1],  # strong stimulus -> engaged
        ]
    )

    assert paper_state_order(coefficients) == (2, 1, 0)


def test_bits_per_trial_is_zero_for_the_null_and_positive_above_it() -> None:
    """The null scores the observed held-out choices at the training-set rate."""

    outcomes = np.repeat([1.0, 0.0], 50)
    null = np.full(100, np.log(0.5))

    assert bits_per_trial(null, outcomes, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert bits_per_trial(null + 0.1, outcomes, 0.5) == pytest.approx(0.1 / np.log(2.0))


def test_bits_per_trial_uses_observed_outcomes_not_their_expectation() -> None:
    """A skewed held-out fold must be scored on the choices it actually contains."""

    outcomes = np.repeat([1.0, 0.0], [90, 10])
    rate = 0.5
    scores = np.full(100, np.log(0.5))
    expected = (100 * np.log(0.5) - (90 * np.log(rate) + 10 * np.log(1 - rate))) / (
        100 * np.log(2.0)
    )

    assert bits_per_trial(scores, outcomes, rate) == pytest.approx(expected)


def test_session_folds_partition_every_trial_exactly_once() -> None:
    study = animal_study(_cohort(n_sessions=10), "M1")
    folds = session_folds(study, n_folds=5)

    combined = np.concatenate(folds)
    assert len(folds) == 5
    assert sorted(combined.tolist()) == list(range(len(study)))
    for fold in folds:
        sessions = {str(value) for value in study["session"][fold]}
        others = {
            str(value) for other in folds if other is not fold for value in study["session"][other]
        }
        assert not sessions & others, "folds must hold out whole sessions"


def test_contract_failures_names_every_value_that_missed_its_band() -> None:
    exact = dict(PUBLISHED)

    assert contract_failures(exact) == ()
    assert "n_animals" in contract_failures({**exact, "n_animals": 36})
    assert "example_mouse_n_trials" in contract_failures({**exact, "example_n_trials": 5039})
    assert "median_engaged_dwell_time" in contract_failures(
        {**exact, "median_dwell_times": (5.0, 13.0, 12.0)}
    )
    assert contract_failures({**exact, "median_dwell_times": (29.0, 13.0, 12.0)}) == ()


def test_the_gate_never_asserts_an_arg_max_over_state_counts() -> None:
    """The paper selects three states on parsimony grounds and reports no arg-max."""

    assert "selected_state_count" not in PUBLISHED
    assert contract_failures({**PUBLISHED, "selected_state_count": 5}) == ()


def test_every_claim_is_classified_on_the_evidence_ladder() -> None:
    """A claim that lands is published parity; one that misses is a retained failure."""

    classification = claim_classification(dict(PUBLISHED))

    assert set(classification) == {identifier for identifier, *_ in CLAIM_CHECKS}
    assert set(classification.values()) == {"published-parity"}
    missed = claim_classification({**PUBLISHED, "n_animals": 36})
    assert missed["n_animals"] == "failed-parity"


def test_committed_result_matches_the_committed_claims() -> None:
    """The published-parity contract and the result must agree on the same cohort."""

    result = json.loads((BENCHMARK / "result.json").read_text(encoding="utf-8"))
    claims = json.loads((BENCHMARK / "published_claims.json").read_text(encoding="utf-8"))

    assert result["source_sha256"] == ARCHIVE_SHA256
    assert result["example_animal"] == EXAMPLE_ANIMAL
    assert claims["data"]["member_sha256"] == ARCHIVE_SHA256
    assert result["n_animals"] == PUBLISHED["n_animals"]
    assert (result["contract_passed"] is True) == (not result["contract_failures"])
    assert (BENCHMARK / "PROTOCOL.md").is_file()


def test_protocol_is_dated_and_declares_its_substitutions() -> None:
    protocol = (BENCHMARK / "PROTOCOL.md").read_text(encoding="utf-8")

    assert "Frozen 2026-07-28" in protocol
    assert "Declared substitutions" in protocol
    assert "ssm" in protocol


@pytest.mark.slow
def test_benchmark_reruns_and_reproduces_its_committed_result(tmp_path: Path) -> None:
    """Recompute the science rather than reading it back, when the archive is present."""

    if not DEFAULT_DESTINATION.exists():
        pytest.skip("run benchmarks.ashwood2022_glmhmm.fetch_data first")

    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ashwood2022_glmhmm.benchmark",
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
    committed = json.loads((BENCHMARK / "result.json").read_text(encoding="utf-8"))
    rerun = json.loads(output.read_text(encoding="utf-8"))

    assert PROVENANCE_KEY in rerun
    assert rerun["benchmark"] == committed["benchmark"]
    assert rerun["n_animals"] == committed["n_animals"]
    assert rerun["n_source_trials"] == committed["n_source_trials"]
    assert rerun["selected_state_count"] == committed["selected_state_count"]
    assert rerun["contract_passed"] == committed["contract_passed"]
    assert rerun["contract_failures"] == committed["contract_failures"]
    assert rerun["median_dwell_times"] == pytest.approx(committed["median_dwell_times"], rel=1e-6)
