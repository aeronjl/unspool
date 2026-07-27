"""Regression gates for the Cell 2025 migration to the 0.20 protocol workflow."""

import json
from pathlib import Path

import pytest

from benchmarks.cell2025_flagship.benchmark import MODEL_ORDER
from benchmarks.cell2025_protocol.benchmark import (
    DEFAULT_DESTINATION,
    LEGACY_RESULT,
    PROTOCOL_RESULT,
    build_protocol,
    compile_protocol,
)
from unspool import ProtocolState, RecoveryKind, protocol_from_dict


def test_cell_protocol_declares_the_complete_scientific_design() -> None:
    protocol = build_protocol()

    assert protocol.state == ProtocolState.DRAFT
    assert protocol.cohort.outcome_blind
    assert protocol.cohort.expected_subjects == 30
    assert protocol.cohort.expected_sessions == 390
    assert protocol.cohort.expected_observations == 73_042
    assert tuple(candidate.name for candidate in protocol.candidates) == MODEL_ORDER
    assert protocol.validation.geometry.value == "historical-cohort-future-session"
    assert protocol.comparison.aggregation_unit == "animal"
    assert protocol.comparison.bootstrap_repetitions == 5_000
    assert protocol.transforms[0].uses_outcomes
    assert protocol.transforms[0].visibility.value == "training-only"
    assert tuple(item.kind for item in protocol.recovery) == (
        RecoveryKind.MODEL,
        RecoveryKind.PARAMETER,
        RecoveryKind.OUTCOME_DERIVED_FEATURE,
    )


@pytest.mark.skipif(
    not DEFAULT_DESTINATION.exists(),
    reason="checksum-pinned Cell source table is not present in the local public-data cache",
)
def test_cell_public_panel_compiles_with_exact_denominators_and_fold_geometry() -> None:
    compiled = compile_protocol()
    recorded = json.loads(PROTOCOL_RESULT.read_text(encoding="utf-8"))

    assert compiled.protocol.state == ProtocolState.AUDITED
    assert compiled.plan.audit.passed
    assert compiled.materialized.manifest.selected_subjects == 30
    assert compiled.materialized.manifest.selected_sessions == 390
    assert compiled.materialized.manifest.selected_observations == 73_042
    observed = [
        (
            len(fold.fit_rows),
            len(fold.prediction_context_rows),
            len(fold.scored_rows),
            len(fold.excluded_rows),
        )
        for fold in compiled.plan.folds
    ]
    expected = [
        (
            fold["fit_rows"],
            fold["prediction_context_rows"],
            fold["scored_rows"],
            fold["excluded_rows"],
        )
        for fold in recorded["plan"]["folds"]
    ]
    assert observed == expected


def test_recorded_cell_protocol_run_has_full_numerical_parity() -> None:
    migrated = json.loads(PROTOCOL_RESULT.read_text(encoding="utf-8"))
    legacy = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))["historical_cohort_forecast"]

    assert migrated["parity"]["passed"]
    assert all(
        migrated["parity"][name]
        for name in (
            "denominator_parity",
            "fold_parity",
            "candidate_parity",
            "score_parity",
            "interval_parity",
            "audit_parity",
        )
    )
    assert migrated["evaluation"]["ranking"]["status"] == "unresolved"
    assert migrated["evaluation"]["ranking"]["winner"] is None
    for name in MODEL_ORDER:
        observed = migrated["evaluation"]["candidates"][name]
        expected = legacy["models"][name]
        assert observed["unit_balanced_log_loss"] == expected["unit_balanced_log_loss"]
        assert observed["pooled_log_loss"] == expected["pooled_log_loss"]
        assert (
            observed["unit_balanced_log_loss_interval"]
            == (expected["unit_balanced_log_loss_interval"])
        )
        assert observed["audit_status"] == expected["audit_status"]


def test_recorded_cell_protocol_is_schema_valid_and_lifecycle_complete_for_evaluation() -> None:
    recorded = json.loads(PROTOCOL_RESULT.read_text(encoding="utf-8"))
    protocol = protocol_from_dict(recorded["protocol"])

    assert protocol.state == ProtocolState.EVALUATED
    assert protocol.fingerprint == recorded["parity"]["protocol_fingerprint"]
    assert [event.to_state.value for event in protocol.lifecycle] == [
        "frozen",
        "materialized",
        "audited",
        "evaluated",
    ]
    assert Path(PROTOCOL_RESULT).stat().st_size < 250_000
