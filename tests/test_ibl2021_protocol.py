"""Regression gates for the public IBL migration to the 0.20 protocol workflow."""

import json
from pathlib import Path

import pytest

from benchmarks.ibl2021_nested_selection.benchmark import CANDIDATES
from benchmarks.ibl2021_protocol.benchmark import (
    LEGACY_RESULT,
    PROTOCOL_RESULT,
    build_protocol,
    recorded_parity,
)
from unspool import ProtocolState, protocol_from_dict, protocol_from_json


@pytest.mark.parametrize("target", ("same-animal", "held-out-lab"))
def test_ibl_protocol_declares_source_panel_candidates_and_nested_selection(target: str) -> None:
    protocol = build_protocol(target)

    assert protocol.state == ProtocolState.DRAFT
    assert protocol.source.release == "2021_Q1_IBL_et_al_Behaviour"
    assert protocol.source.checksum_algorithm == "sha256-manifest"
    assert protocol.cohort.outcome_blind
    assert protocol.cohort.expected_subjects == 78
    assert protocol.cohort.expected_sessions == 468
    assert protocol.cohort.expected_observations == 46_152
    assert protocol.panel.minimum_sessions == 6
    assert protocol.panel.common_clock == "endpoint-window-position"
    assert tuple(candidate.name for candidate in protocol.candidates) == CANDIDATES
    assert protocol.selection is not None
    assert protocol.selection.candidate_names == CANDIDATES
    assert protocol.selection.refit_selected_on_outer_training
    assert protocol.comparison.winner_policy.value == "no-automatic-winner"
    if target == "held-out-lab":
        assert protocol.validation.geometry.value == "held-out-group-future-session"
        assert protocol.validation.group_unit == "lab"
        assert all(candidate.supports_unseen_subjects for candidate in protocol.candidates)
        assert all(candidate.supports_unseen_groups for candidate in protocol.candidates)
    else:
        assert protocol.validation.geometry.value == "future-session"
        assert protocol.validation.group_unit is None


@pytest.mark.parametrize("target", ("same-animal", "held-out-lab"))
def test_ibl_protocol_canonical_round_trip_is_exact(target: str) -> None:
    frozen = build_protocol(target).freeze()

    restored = protocol_from_json(frozen.canonical_json())

    assert restored == frozen
    assert restored.fingerprint == frozen.fingerprint


@pytest.mark.parametrize("target", ("same-animal", "held-out-lab"))
def test_ibl_declaration_matches_the_pinned_legacy_design(target: str) -> None:
    parity = recorded_parity(target)

    assert parity.passed
    assert parity.source_parity
    assert parity.denominator_parity
    assert parity.candidate_parity
    assert parity.outer_geometry_parity
    assert parity.nested_selection_parity


@pytest.mark.parametrize(
    ("target", "legacy_key", "outer_folds", "inner_folds"),
    (
        ("same-animal", "within_subject_future_session", 1, 2),
        ("held-out-lab", "held_out_lab_future_session", 9, 8),
    ),
)
def test_recorded_ibl_protocol_has_exact_numerical_and_geometry_parity(
    target: str,
    legacy_key: str,
    outer_folds: int,
    inner_folds: int,
) -> None:
    migrated = json.loads(PROTOCOL_RESULT.read_text(encoding="utf-8"))[target]
    legacy = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))[legacy_key]

    assert migrated["parity"]["passed"]
    assert all(
        migrated["parity"][name]
        for name in (
            "source_parity",
            "denominator_parity",
            "candidate_parity",
            "outer_geometry_parity",
            "nested_selection_parity",
            "score_parity",
            "interval_parity",
            "audit_parity",
        )
    )
    assert migrated["cohort"]["selected_subjects"] == 78
    assert migrated["cohort"]["selected_sessions"] == 468
    assert migrated["cohort"]["selected_observations"] == 46_152
    assert migrated["plan"]["audit_passed"]
    assert len(migrated["plan"]["folds"]) == outer_folds
    assert {fold["inner_folds"] for fold in migrated["plan"]["folds"]} == {inner_folds}
    assert migrated["evaluation"]["selection_counts"] == legacy["selection_counts"]
    assert migrated["evaluation"]["unit_balanced_log_loss"] == legacy["subject_balanced_log_loss"]
    assert migrated["evaluation"]["pooled_log_loss"] == legacy["pooled_trial_log_loss"]
    interval = migrated["evaluation"]["unit_balanced_log_loss_interval"]
    assert [interval["lower"], interval["upper"]] == legacy[
        "subject_bootstrap_log_loss_95_interval"
    ]


@pytest.mark.parametrize("target", ("same-animal", "held-out-lab"))
def test_recorded_ibl_protocol_is_schema_valid_and_evaluated(target: str) -> None:
    recorded = json.loads(PROTOCOL_RESULT.read_text(encoding="utf-8"))[target]
    protocol = protocol_from_dict(recorded["protocol"])

    assert protocol.state == ProtocolState.EVALUATED
    assert protocol.fingerprint == recorded["parity"]["protocol_fingerprint"]
    assert [event.to_state.value for event in protocol.lifecycle] == [
        "frozen",
        "materialized",
        "audited",
        "evaluated",
    ]
    assert Path(PROTOCOL_RESULT).stat().st_size < 300_000
