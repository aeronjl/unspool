"""Tests for the immutable study-protocol declaration and lifecycle."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from behavio.protocol.schema import (
    PROTOCOL_SCHEMA_VERSION,
    AggregationWeighting,
    CandidateSpec,
    CohortPredicate,
    CohortSpec,
    ComparisonMultiplicity,
    ComparisonSpec,
    EstimandSpec,
    LifecycleEvent,
    ObservationDataType,
    ObservationRole,
    ObservationSpec,
    PanelSpec,
    PredicateOperator,
    PredictionInformation,
    ProtocolClockSpec,
    ProtocolLifecycleError,
    ProtocolState,
    ProtocolValidationError,
    RecoveryKind,
    RecoverySpec,
    ReportingSpec,
    ScoreMetric,
    Setting,
    SourceSpec,
    StudyProtocol,
    TransformSpec,
    TransformVisibility,
    UnitRole,
    UnitSpec,
    ValidationGeometry,
    ValidationSpec,
    WinnerPolicy,
    protocol_from_dict,
    protocol_from_json,
)


def default_candidates() -> tuple[CandidateSpec, ...]:
    """Declare exactly the estimators the runner fixtures actually supply.

    The declaration is the object under test everywhere downstream: ``run_protocol``
    verifies each supplied estimator against its frozen ``implementation`` and
    ``hyperparameters``, so a fixture that declares one model and runs another would be
    the very defect these tests exist to catch.
    """

    common = (Setting("predictors", ("stimulus",)), Setting("choice_lags", 0))
    return (
        CandidateSpec(
            name="static",
            implementation="behavio.models.BernoulliHistoryGLM",
            hyperparameters=(*common, Setting("l2", 0.1)),
            scored_columns=("choice",),
        ),
        CandidateSpec(
            name="smooth",
            implementation="behavio.models.BernoulliHistoryGLM",
            hyperparameters=(*common, Setting("l2", 1.0)),
            scored_columns=("choice",),
        ),
    )


def example_protocol(
    *,
    with_recovery: bool = True,
    candidates: tuple[CandidateSpec, ...] = (),
) -> StudyProtocol:
    recovery = (
        RecoverySpec(
            name="candidate-recovery",
            kind=RecoveryKind.MODEL,
            required=True,
            repetitions=20,
            seed=919,
            success_metric="selection-rate",
            threshold=0.8,
            constrains_claims=("mechanistic identification",),
            scenarios=(Setting("signal", "matched-design"),),
        ),
    )
    return StudyProtocol(
        identifier="learning-forecast-v1",
        title="Learning trajectory forecast",
        question="Do learning-history models improve future-session prediction?",
        source=SourceSpec(
            adapter="nwb",
            release="2026-01",
            locator="dandi:000000/0.1.0",
            checksum_algorithm="sha256",
            checksum="a" * 64,
            identity_columns=("source_asset", "source_row"),
            metadata=(Setting("license", "CC-BY-4.0"),),
        ),
        cohort=CohortSpec(
            predicates=(
                CohortPredicate(
                    column="species",
                    operator=PredicateOperator.EQUAL,
                    value="mouse",
                    rationale="target population",
                ),
            ),
            selection_columns=("species", "session_order"),
            outcome_blind=True,
            expected_subjects=12,
            expected_sessions=72,
        ),
        units=(
            UnitSpec("animal", "subject", UnitRole.EXPERIMENTAL),
            UnitSpec("session", "session", UnitRole.REPEATED_MEASURES, "animal"),
        ),
        observations=(
            ObservationSpec("choice", ObservationRole.OUTCOME, "binary", allowed_values=(0, 1)),
            ObservationSpec("stimulus", ObservationRole.PREDICTOR, "continuous"),
        ),
        clocks=(
            ProtocolClockSpec(
                name="training-session",
                column="session_order",
                kind="ordinal-session",
                scope="subject",
                alignment="first-eligible-session",
            ),
        ),
        panel=PanelSpec(
            subject_unit="animal",
            session_unit="session",
            common_clock="training-session",
            minimum_sessions=6,
            balance_required=True,
        ),
        estimands=(
            EstimandSpec(
                name="animal-balanced-future-log-loss",
                population="eligible animals under the declared release",
                outcome_columns=("choice",),
                contrast="candidate minus reference future-session score",
                aggregation_unit="animal",
                weighting=AggregationWeighting.EQUAL_UNIT,
            ),
        ),
        transforms=(
            TransformSpec(
                name="stimulus-scale",
                implementation="behavio.time.transforms.Standardize",
                input_columns=("stimulus",),
                output_columns=("stimulus_z",),
                visibility=TransformVisibility.TRAINING_ONLY,
            ),
        ),
        validation=ValidationSpec(
            geometry=ValidationGeometry.FUTURE_SESSION,
            splitter="behavio.evaluate.splits.cohort_forward_session_splits",
            prediction_information=PredictionInformation.FILTERED,
            origin=4,
            horizon=(5,),
        ),
        candidates=candidates or default_candidates(),
        comparison=ComparisonSpec(
            metric=ScoreMetric.LOG_LOSS,
            aggregation_unit="animal",
            weighting=AggregationWeighting.EQUAL_UNIT,
            interval_method="paired-unit-bootstrap",
            interval_level=0.95,
            bootstrap_repetitions=2_000,
            seed=2025,
            paired=True,
            winner_policy=WinnerPolicy.INTERVAL_EXCLUDES_ZERO,
            reference_candidate="static",
        ),
        recovery=recovery if with_recovery else (),
        reporting=ReportingSpec(
            required_tables=("denominators", "fold-scores", "fit-audits"),
            required_figures=("paired-score-contrast",),
            required_diagnostics=("optimization", "calibration"),
            limitations=("one empirical cohort",),
            prohibited_claims=("mechanistic identification",),
        ),
    )


def test_protocol_is_deeply_immutable_and_has_versioned_fingerprint() -> None:
    protocol = example_protocol()

    assert protocol.schema_version == PROTOCOL_SCHEMA_VERSION
    assert len(protocol.fingerprint) == 64
    assert protocol.state == ProtocolState.DRAFT
    with pytest.raises(FrozenInstanceError):
        protocol.title = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        protocol.source.release = "changed"  # type: ignore[misc]


def test_canonical_serialization_round_trips_without_changing_identity() -> None:
    frozen = example_protocol().freeze()
    restored = protocol_from_json(frozen.canonical_json())

    assert restored == frozen
    assert restored.canonical_json() == frozen.canonical_json()
    assert restored.fingerprint == frozen.fingerprint
    assert "NaN" not in frozen.canonical_json()


def test_fingerprint_stays_fixed_as_lifecycle_evidence_accumulates() -> None:
    frozen = example_protocol().freeze()
    materialized = frozen.advance(ProtocolState.MATERIALIZED, artifact_fingerprint="b" * 64)
    audited = materialized.advance(ProtocolState.AUDITED, artifact_fingerprint="c" * 64)

    assert frozen.fingerprint == materialized.fingerprint == audited.fingerprint
    assert frozen.canonical_json() != materialized.canonical_json()
    assert [event.to_state for event in audited.lifecycle] == [
        ProtocolState.FROZEN,
        ProtocolState.MATERIALIZED,
        ProtocolState.AUDITED,
    ]


def test_lifecycle_requires_ordered_evidence_and_recovery_before_reporting() -> None:
    protocol = example_protocol().freeze()
    with pytest.raises(ProtocolLifecycleError, match="cannot advance"):
        protocol.advance(ProtocolState.AUDITED, artifact_fingerprint="b" * 64)

    protocol = protocol.advance(ProtocolState.MATERIALIZED, artifact_fingerprint="b" * 64)
    protocol = protocol.advance(ProtocolState.AUDITED, artifact_fingerprint="c" * 64)
    protocol = protocol.advance(ProtocolState.EVALUATED, artifact_fingerprint="d" * 64)
    with pytest.raises(ProtocolLifecycleError, match=r"allowed=.*recovered"):
        protocol.advance(ProtocolState.REPORTED, artifact_fingerprint="e" * 64)
    protocol = protocol.advance(ProtocolState.RECOVERED, artifact_fingerprint="e" * 64)
    protocol = protocol.advance(ProtocolState.REPORTED, artifact_fingerprint="f" * 64)

    assert protocol.state == ProtocolState.REPORTED


def test_protocol_without_recovery_may_report_after_evaluation() -> None:
    protocol = example_protocol(with_recovery=False).freeze()
    for state, digest in (
        (ProtocolState.MATERIALIZED, "b" * 64),
        (ProtocolState.AUDITED, "c" * 64),
        (ProtocolState.EVALUATED, "d" * 64),
        (ProtocolState.REPORTED, "e" * 64),
    ):
        protocol = protocol.advance(state, artifact_fingerprint=digest)
    assert protocol.state == ProtocolState.REPORTED


def test_amendment_links_parent_and_must_be_refrozen_before_evidence() -> None:
    original = example_protocol().freeze()
    amended = original.amend(
        identifier="amendment-01",
        reason="Public release corrected the expected session denominator.",
        cohort=replace(original.cohort, expected_sessions=71),
    )

    assert amended.state == ProtocolState.DRAFT
    assert amended.lifecycle == ()
    assert amended.amendments[-1].parent_fingerprint == original.fingerprint
    assert amended.amendments[-1].changed_sections == ("cohort",)
    assert amended.fingerprint != original.fingerprint
    assert amended.freeze().lifecycle[0].artifact_fingerprint == amended.fingerprint

    materialized = original.advance(ProtocolState.MATERIALIZED, artifact_fingerprint="b" * 64)
    with pytest.raises(ProtocolLifecycleError, match="pre-evidence"):
        materialized.amend(identifier="late", reason="too late", title="changed")


def test_outcome_blind_cohort_cannot_select_on_declared_outcome() -> None:
    protocol = example_protocol()
    cohort = replace(protocol.cohort, selection_columns=("species", "choice"))
    with pytest.raises(ProtocolValidationError, match="selects on outcome"):
        replace(protocol, cohort=cohort)


def test_outcome_derived_transform_must_be_training_only() -> None:
    with pytest.raises(ProtocolValidationError, match="training-only"):
        TransformSpec(
            name="learning-landmark",
            implementation="behavio.time.landmarks.ThresholdLandmarkClock",
            input_columns=("choice",),
            output_columns=("relative_trial",),
            visibility=TransformVisibility.FIXED_A_PRIORI,
            uses_outcomes=True,
        )


def test_candidates_must_score_the_same_complete_observation() -> None:
    protocol = example_protocol()
    incompatible = replace(protocol.candidates[1], scored_columns=("choice", "response_time"))
    with pytest.raises(ProtocolValidationError, match="same complete observation"):
        replace(protocol, candidates=(protocol.candidates[0], incompatible))


def test_brier_scoring_requires_one_binary_outcome_column() -> None:
    protocol = example_protocol()
    observations = (
        *protocol.observations,
        ObservationSpec("response_time", ObservationRole.OUTCOME, "continuous"),
    )
    candidates = tuple(
        replace(candidate, scored_columns=("choice", "response_time"))
        for candidate in protocol.candidates
    )
    with pytest.raises(ProtocolValidationError, match="Brier scoring requires exactly one"):
        replace(
            protocol,
            observations=observations,
            candidates=candidates,
            comparison=replace(protocol.comparison, metric=ScoreMetric.BRIER),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"weighting": AggregationWeighting.POOLED_OBSERVATION}, "equal-unit"),
        ({"interval_method": "percentile-bootstrap"}, "paired-unit-bootstrap"),
        ({"paired": False}, "requires paired"),
    ),
)
def test_comparison_rejects_declarations_the_runner_cannot_honor(change, message) -> None:
    comparison = example_protocol().comparison
    with pytest.raises(ProtocolValidationError, match=message):
        replace(comparison, **change)


def test_comparison_must_match_a_declared_estimand() -> None:
    protocol = example_protocol()
    with pytest.raises(ProtocolValidationError, match="must match a declared estimand"):
        replace(
            protocol,
            estimands=(replace(protocol.estimands[0], aggregation_unit="session"),),
        )


def test_recovery_gates_only_reference_explicitly_prohibited_claims() -> None:
    protocol = example_protocol()
    recovery = replace(protocol.recovery[0], constrains_claims=("causal effect",))
    with pytest.raises(ProtocolValidationError, match=r"unknown=.*causal effect"):
        replace(protocol, recovery=(recovery,))


def test_tampered_lifecycle_is_rejected_on_construction() -> None:
    protocol = example_protocol()
    with pytest.raises(ProtocolValidationError, match="freeze event"):
        replace(
            protocol,
            state=ProtocolState.FROZEN,
            lifecycle=(LifecycleEvent(ProtocolState.DRAFT, ProtocolState.FROZEN, "b" * 64),),
        )


def test_observation_data_type_is_a_closed_vocabulary_that_round_trips() -> None:
    protocol = example_protocol().freeze()
    payload = protocol.to_dict()

    assert [item["data_type"] for item in payload["observations"]] == ["binary", "continuous"]
    restored = protocol_from_json(protocol.canonical_json())
    assert restored.observations[0].data_type is ObservationDataType.BINARY
    assert restored.fingerprint == protocol.fingerprint
    assert restored.canonical_json() == protocol.canonical_json()

    with pytest.raises(ProtocolValidationError, match="is not one of"):
        ObservationSpec("choice", ObservationRole.OUTCOME, "spike-count")


def test_allowed_values_may_declare_missing_observations_explicitly() -> None:
    spec = ObservationSpec(
        "response_time",
        ObservationRole.AUXILIARY,
        ObservationDataType.CONTINUOUS,
        allowed_values=(0.25, None),
    )

    assert spec.allowed_values == (0.25, None)
    with pytest.raises(ProtocolValidationError, match="must be unique"):
        ObservationSpec(
            "response_time",
            ObservationRole.AUXILIARY,
            ObservationDataType.CONTINUOUS,
            allowed_values=(None, None),
        )


def test_the_declared_multiplicity_defaults_to_the_adjustment_the_runner_applied() -> None:
    protocol = example_protocol()

    assert protocol.schema_version == "behavio.study-protocol/2"
    assert protocol.comparison.multiplicity is ComparisonMultiplicity.BENJAMINI_HOCHBERG


def test_a_declared_multiplicity_round_trips_through_protocol_json() -> None:
    for adjustment in ComparisonMultiplicity:
        protocol = replace(
            example_protocol(),
            comparison=replace(example_protocol().comparison, multiplicity=adjustment),
        )

        restored = protocol_from_json(protocol.canonical_json())

        assert restored.comparison.multiplicity is adjustment
        assert restored == protocol
        assert restored.fingerprint == protocol.fingerprint
    # The declaration is part of the content address, which is the whole point: two
    # protocols identical but for the adjustment that picks their winner are two protocols.
    corrected = example_protocol()
    uncorrected = replace(
        corrected,
        comparison=replace(corrected.comparison, multiplicity=ComparisonMultiplicity.NONE),
    )
    assert corrected.fingerprint != uncorrected.fingerprint


def test_a_version_one_payload_still_loads_and_keeps_its_own_fingerprint() -> None:
    """A frozen protocol is content-addressed and its freeze event quotes that address."""

    protocol = example_protocol().freeze()
    recorded = json.loads(protocol.canonical_json())
    # Exactly what a protocol frozen before the member existed looks like on disk.
    del recorded["comparison"]["multiplicity"]
    recorded["schema_version"] = "behavio.study-protocol/1"
    recorded["lifecycle"][0]["artifact_fingerprint"] = _legacy_fingerprint(recorded)

    restored = protocol_from_dict(recorded)

    assert restored.schema_version == "behavio.study-protocol/1"
    assert restored.comparison.multiplicity is ComparisonMultiplicity.BENJAMINI_HOCHBERG
    assert restored.fingerprint == recorded["lifecycle"][0]["artifact_fingerprint"]
    assert json.loads(restored.canonical_json()) == recorded
    assert restored.state is ProtocolState.FROZEN


def _legacy_fingerprint(recorded: dict) -> str:
    scientific = {
        key: value for key, value in recorded.items() if key not in ("state", "lifecycle")
    }
    payload = json.dumps(
        scientific, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_a_superseded_schema_cannot_smuggle_in_an_adjustment_it_predates() -> None:
    with pytest.raises(ProtocolValidationError, match="predates the comparison"):
        replace(
            example_protocol(),
            schema_version="behavio.study-protocol/1",
            comparison=replace(
                example_protocol().comparison,
                multiplicity=ComparisonMultiplicity.BONFERRONI,
            ),
        )
    recorded = json.loads(example_protocol().canonical_json())
    recorded["schema_version"] = "behavio.study-protocol/1"
    with pytest.raises(ProtocolValidationError, match="must not carry one"):
        protocol_from_dict(recorded)


def test_an_amendment_is_recorded_under_the_schema_it_is_written_in() -> None:
    legacy = replace(
        example_protocol(),
        schema_version="behavio.study-protocol/1",
    ).freeze()

    amended = legacy.amend(
        identifier="declare-the-adjustment",
        reason="state the multiplicity correction the runner was already applying",
        comparison=replace(legacy.comparison, multiplicity=ComparisonMultiplicity.BONFERRONI),
    )

    assert amended.schema_version == PROTOCOL_SCHEMA_VERSION
    assert amended.comparison.multiplicity is ComparisonMultiplicity.BONFERRONI
    assert amended.amendments[-1].parent_fingerprint == legacy.fingerprint
