"""Tests for protocol materialization and pre-fit execution-plan auditing."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_protocol import example_protocol

from behavio.evaluate.splits import cohort_forward_session_splits
from behavio.models import ModelCapabilities, PredictionMode
from behavio.protocol.compiler import (
    AuditLevel,
    ObservationContractRule,
    ProtocolCompilationError,
    compile_execution_plan,
    materialize_protocol,
    validate_observation_contract,
)
from behavio.protocol.schema import (
    NestedSelectionSpec,
    ObservationDataType,
    ObservationRole,
    ObservationSpec,
    ProtocolState,
    ScoreMetric,
    SelectionTieBreak,
)
from behavio.trials import Study


def source_study() -> Study:
    rows: list[dict[str, object]] = []
    source_row = 0
    for subject, species in (("a", "mouse"), ("b", "mouse")):
        for session_order in range(3):
            for trial in range(2):
                rows.append(
                    {
                        "subject": subject,
                        "session": f"{subject}-s{session_order}",
                        "trial": trial,
                        "session_order": session_order,
                        "choice": (source_row + trial) % 2,
                        "stimulus": float(trial * 2 - 1),
                        "species": species,
                        "source_asset": "asset-01",
                        "source_row": source_row,
                    }
                )
                source_row += 1
    rows.append(
        {
            "subject": "excluded-rat",
            "session": "rat-s0",
            "trial": 0,
            "session_order": 0,
            "choice": 1,
            "stimulus": 0.25,
            "species": "rat",
            "source_asset": "asset-01",
            "source_row": source_row,
        }
    )
    return Study.from_records(rows)


def frozen_small_protocol(candidates=()):
    protocol = example_protocol(candidates=candidates)
    cohort = replace(
        protocol.cohort,
        expected_subjects=2,
        expected_sessions=6,
        expected_observations=12,
    )
    panel = replace(protocol.panel, minimum_sessions=3)
    return replace(protocol, cohort=cohort, panel=panel).freeze()


def capabilities() -> dict[str, ModelCapabilities]:
    value = ModelCapabilities(
        scored_columns=("choice",),
        prediction_modes=(PredictionMode.FILTERED,),
        can_simulate=True,
        can_recover_parameters=True,
    )
    return {"static": value, "smooth": value}


def frozen_nested_protocol():
    draft = example_protocol(with_recovery=False)
    selection = NestedSelectionSpec(
        candidate_names=("static", "smooth"),
        inner_validation=replace(
            draft.validation,
            origin=1,
            horizon=(1,),
            settings=(),
        ),
        metric=ScoreMetric.LOG_LOSS,
        aggregation_unit="animal",
        tie_break=SelectionTieBreak.DECLARED_ORDER,
        bootstrap_repetitions=50,
        seed=17,
    )
    return replace(
        draft,
        cohort=replace(
            draft.cohort,
            expected_subjects=2,
            expected_sessions=6,
            expected_observations=12,
        ),
        panel=replace(draft.panel, minimum_sessions=3),
        selection=selection,
    ).freeze()


def test_materialization_applies_cohort_and_resolves_source_identities() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())

    assert materialized.protocol.state == ProtocolState.MATERIALIZED
    assert len(materialized.study) == 12
    assert materialized.study.subjects == ("a", "b")
    assert materialized.manifest.source_observations == 13
    assert materialized.manifest.selected_observations == 12
    assert materialized.manifest.excluded_source_rows == (12,)
    assert len(materialized.manifest.identities) == 12
    assert materialized.manifest.identities[0].source_row == 0
    assert materialized.manifest.identities[-1].derived_row == 11
    assert len(materialized.manifest.fingerprint) == 64
    assert materialized.protocol.lifecycle[-1].artifact_fingerprint == (
        materialized.manifest.fingerprint
    )
    assert "choice" not in materialized.manifest.canonical_json()


def test_compiler_emits_complete_row_roles_and_approves_audited_plan() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    splits = cohort_forward_session_splits(
        materialized.study,
        min_train_sessions=2,
        horizon=1,
    )
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())

    assert compiled.plan.audit.passed
    assert compiled.protocol.state == ProtocolState.AUDITED
    assert len(compiled.plan.folds) == 1
    fold = compiled.plan.folds[0]
    assert fold.fit_rows == (0, 1, 2, 3, 6, 7, 8, 9)
    assert fold.prediction_context_rows == ()
    assert fold.scored_rows == (4, 5, 10, 11)
    assert fold.excluded_rows == ()
    assert fold.fit_subjects == ("a", "b")
    assert fold.scored_subjects == ("a", "b")
    assert compiled.protocol.lifecycle[-1].artifact_fingerprint == compiled.plan.fingerprint
    assert compiled.plan.canonical_json() == compiled.plan.canonical_json()


def test_missing_runtime_capability_rejects_plan_without_losing_audit() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(
        materialized,
        splits,
        capabilities={"static": capabilities()["static"]},
    )

    assert not compiled.plan.audit.passed
    assert compiled.protocol.state == ProtocolState.MATERIALIZED
    assert [(issue.code, issue.candidate) for issue in compiled.plan.audit.errors] == [
        ("missing-capabilities", "smooth")
    ]


def test_temporal_leakage_is_an_explicit_error_not_an_exception() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    leaky = SimpleNamespace(
        train_indices=np.array([0, 2, 4, 6, 8, 10]),
        test_indices=np.array([1, 3, 5, 7, 9, 11]),
        prediction_context_indices=np.array([], dtype=np.intp),
    )
    compiled = compile_execution_plan(materialized, (leaky,), capabilities=capabilities())

    assert not compiled.plan.audit.passed
    temporal = [
        issue for issue in compiled.plan.audit.issues if issue.code == "temporal-boundary-leakage"
    ]
    assert len(temporal) == 2
    assert all(issue.level == AuditLevel.ERROR for issue in temporal)


def test_context_must_be_available_in_fit_and_never_scored() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    invalid = SimpleNamespace(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        prediction_context_indices=np.array([8]),
    )
    compiled = compile_execution_plan(materialized, (invalid,), capabilities=capabilities())

    codes = {issue.code for issue in compiled.plan.audit.errors}
    assert codes == {"context-outside-fit"}


def test_materialization_rejects_denominator_drift() -> None:
    protocol = frozen_small_protocol()
    wrong = replace(
        protocol,
        cohort=replace(protocol.cohort, expected_observations=11),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()

    with pytest.raises(ProtocolCompilationError, match="denominators differ"):
        materialize_protocol(wrong, source_study())


def test_source_identity_collisions_are_rejected() -> None:
    study = source_study()
    columns = {column: study[column].copy() for column in study.columns}
    columns["source_row"][1] = columns["source_row"][0]

    with pytest.raises(ProtocolCompilationError, match="do not uniquely identify"):
        materialize_protocol(frozen_small_protocol(), Study.from_columns(columns))


def test_empty_validation_design_is_a_retained_audit_failure() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    compiled = compile_execution_plan(materialized, (), capabilities=capabilities())

    assert not compiled.plan.audit.passed
    assert [issue.code for issue in compiled.plan.audit.errors] == ["no-folds"]
    assert compiled.protocol.state == ProtocolState.MATERIALIZED


def test_nested_selection_compiles_inner_rows_only_from_outer_training() -> None:
    materialized = materialize_protocol(frozen_nested_protocol(), source_study())
    outer = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(
        materialized,
        outer,
        capabilities=capabilities(),
        inner_splitter=lambda training, _fold: cohort_forward_session_splits(
            training,
            min_train_sessions=1,
        ),
    )

    assert compiled.plan.audit.passed
    outer_fold = compiled.plan.folds[0]
    assert len(outer_fold.inner_folds) == 1
    inner = outer_fold.inner_folds[0]
    assert set(inner.fit_rows) <= set(outer_fold.fit_rows)
    assert set(inner.scored_rows) <= set(outer_fold.fit_rows)
    assert not set(inner.fit_rows) & set(outer_fold.scored_rows)
    assert not set(inner.scored_rows) & set(outer_fold.scored_rows)
    assert inner.fit_rows == (0, 1, 6, 7)
    assert inner.scored_rows == (2, 3, 8, 9)


def test_nested_protocol_without_inner_splitter_fails_pre_fit_audit() -> None:
    materialized = materialize_protocol(frozen_nested_protocol(), source_study())
    outer = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, outer, capabilities=capabilities())

    assert not compiled.plan.audit.passed
    assert [issue.code for issue in compiled.plan.audit.errors] == ["inner-splitter-missing"]


def test_brier_scoring_rejects_non_binary_outcomes_before_fit() -> None:
    protocol = frozen_small_protocol()
    # The outcome is declared categorical here so that materialization's observation
    # contract admits the value and the compiler's Brier audit is what refuses the plan.
    protocol = replace(
        protocol,
        comparison=replace(
            protocol.comparison, metric=ScoreMetric.BRIER, metrics=(ScoreMetric.BRIER,)
        ),
        observations=(
            ObservationSpec("choice", ObservationRole.OUTCOME, ObservationDataType.CATEGORICAL),
            *protocol.observations[1:],
        ),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    study = source_study()
    columns = {column: study[column].copy() for column in study.columns}
    columns["choice"][0] = 2
    materialized = materialize_protocol(protocol, Study.from_columns(columns))
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)

    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())

    assert not compiled.plan.audit.passed
    assert "brier-outcome-invalid" in {issue.code for issue in compiled.plan.audit.errors}


def protocol_with_observations(*observations: ObservationSpec):
    """Freeze the small protocol with a replaced observation contract."""

    protocol = frozen_small_protocol()
    return replace(
        protocol,
        observations=observations,
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()


def study_with(**overrides) -> Study:
    study = source_study()
    columns: dict[str, object] = {
        column: study[column].astype(object).copy() for column in study.columns
    }
    for column, values in overrides.items():
        if column in columns:
            for row, value in values.items():
                columns[column][row] = value
        else:
            columns[column] = list(values)
    return Study.from_columns(columns)


def test_declared_allowed_values_are_checked_against_the_materialized_column() -> None:
    protocol = protocol_with_observations(
        ObservationSpec(
            "choice",
            ObservationRole.OUTCOME,
            ObservationDataType.BINARY,
            allowed_values=(0, 1),
        ),
        ObservationSpec("stimulus", ObservationRole.PREDICTOR, ObservationDataType.CONTINUOUS),
    )

    with pytest.raises(ProtocolCompilationError) as error:
        materialize_protocol(protocol, study_with(choice={0: 7, 5: 7}))

    message = str(error.value)
    assert "'choice'" in message
    assert "2 of 12 rows" in message
    assert "row 0=7" in message and "row 5=7" in message


def test_declared_data_type_is_checked_independently_of_allowed_values() -> None:
    protocol = protocol_with_observations(
        ObservationSpec("choice", ObservationRole.OUTCOME, ObservationDataType.BINARY),
        ObservationSpec("stimulus", ObservationRole.PREDICTOR, ObservationDataType.COUNT),
    )

    violations = validate_observation_contract(protocol, study_with().take(tuple(range(12))))

    assert [violation.column for violation in violations] == ["stimulus"]
    violation = violations[0]
    assert violation.rule == ObservationContractRule.DATA_TYPE
    assert violation.data_type == "count"
    assert violation.n_violating_rows == 6
    assert "non-negative whole numbers" in violation.message


def test_missing_observations_are_legitimate_unless_values_are_declared() -> None:
    undeclared_values = protocol_with_observations(
        ObservationSpec("choice", ObservationRole.OUTCOME, ObservationDataType.BINARY),
        ObservationSpec("stimulus", ObservationRole.PREDICTOR, ObservationDataType.CONTINUOUS),
        ObservationSpec("response_time", ObservationRole.AUXILIARY, ObservationDataType.CONTINUOUS),
    )
    declared_missing = protocol_with_observations(
        ObservationSpec("choice", ObservationRole.OUTCOME, ObservationDataType.BINARY),
        ObservationSpec("stimulus", ObservationRole.PREDICTOR, ObservationDataType.CONTINUOUS),
        ObservationSpec(
            "response_time",
            ObservationRole.AUXILIARY,
            ObservationDataType.CONTINUOUS,
            allowed_values=(0.25, None),
        ),
    )
    aborted = study_with(response_time=[0.25] * 12 + [float("nan")])

    assert validate_observation_contract(undeclared_values, aborted) == ()
    assert validate_observation_contract(declared_missing, aborted) == ()
    assert materialize_protocol(undeclared_values, aborted).manifest.selected_observations == 12


def test_an_undeclared_missing_observation_is_reported_as_a_missing_value_violation() -> None:
    protocol = protocol_with_observations(
        ObservationSpec(
            "choice",
            ObservationRole.OUTCOME,
            ObservationDataType.BINARY,
            allowed_values=(0, 1),
        ),
        ObservationSpec("stimulus", ObservationRole.PREDICTOR, ObservationDataType.CONTINUOUS),
    )

    violations = validate_observation_contract(protocol, study_with(choice={3: float("nan")}))

    assert len(violations) == 1
    assert violations[0].rule == ObservationContractRule.MISSING_VALUE
    assert violations[0].example_rows == (3,)
    assert "do not include null" in violations[0].message
    assert violations[0].to_dict()["rule"] == "missing-value"
