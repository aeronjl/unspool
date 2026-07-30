"""Same-animal and held-out-lab IBL protocols with nested training-only selection.

The committed ``result.json`` is **stale in its ``model_signature`` strings**. The
hierarchical and hierarchical-smooth candidates are now
``hierarchical(...)`` and ``hierarchical(smooth(...))`` compositions, whose signatures nest
the wrapped model's signature instead of naming a single class. The fits themselves are
bit-identical -- the composition reproduces the deleted classes exactly -- so only the
recorded identity strings move, and they move on the next re-run against the real data.

It is **also stale in every fingerprint**, for a second and unrelated reason.
:class:`~behavio.protocol.ComparisonSpec` now declares its ``multiplicity``, which moved
the schema to ``behavio.study-protocol/2``. The recorded payloads are version 1 protocols
and keep their own fingerprints when read back, but :func:`build_protocol` now emits
version 2, so each target's ``parity.protocol_fingerprint``, plan fingerprint, evaluation
fingerprint and four lifecycle ``artifact_fingerprint`` values will move on the next
re-run. This protocol declares ``NO_AUTOMATIC_WINNER``, so the adjustment decides nothing
here at all; **no score, interval, selection or audit changes with the fingerprints**.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from behavio import (
    AggregationWeighting,
    CandidateSpec,
    CohortPredicate,
    CohortSpec,
    ComparisonMultiplicity,
    ComparisonSpec,
    CompiledProtocol,
    EstimandSpec,
    NestedProtocolRun,
    NestedSelectionSpec,
    ObservationRole,
    ObservationSpec,
    PanelSpec,
    PredicateOperator,
    PredictionInformation,
    ProtocolClockSpec,
    ReportingSpec,
    ScoreMetric,
    SelectionTieBreak,
    Setting,
    SourceSpec,
    StudyProtocol,
    UnitRole,
    UnitSpec,
    ValidationGeometry,
    ValidationSpec,
    WinnerPolicy,
    cohort_forward_session_splits,
    compile_execution_plan,
    leave_one_lab_out_session_forecast_splits,
    materialize_protocol,
    model_capabilities,
    run_nested_protocol,
)
from benchmarks.ibl2021_nested_selection.benchmark import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANDIDATES,
    INNER_BOOTSTRAP_RESAMPLES,
    KNOTS,
    _candidates,
)
from benchmarks.ibl2021_prospective.benchmark import (
    TRAIN_SESSION_COUNT,
    build_panel,
)
from benchmarks.ibl2021_replicated.benchmark import DEFAULT_CACHE, load_study
from benchmarks.ibl2021_replicated.manifest import (
    EXPECTED_MANIFEST_SHA256,
    load_manifest,
)
from benchmarks.provenance import render

Target = Literal["same-animal", "held-out-lab"]
LEGACY_RESULT = Path(__file__).parents[1] / "ibl2021_nested_selection" / "result.json"
PROTOCOL_RESULT = Path(__file__).with_name("result.json")


@dataclass(frozen=True, slots=True)
class IBLProtocolParity:
    """Pinned design and numerical parity summary for one deployment target."""

    target: Target
    protocol_fingerprint: str
    source_parity: bool
    denominator_parity: bool
    candidate_parity: bool
    outer_geometry_parity: bool
    nested_selection_parity: bool
    score_parity: bool | None = None
    interval_parity: bool | None = None
    audit_parity: bool | None = None

    @property
    def passed(self) -> bool:
        """Whether every evaluated parity gate passed."""

        return all(value is not False for value in asdict(self).values())


def build_protocol(target: Target) -> StudyProtocol:
    """Return a complete draft protocol for one IBL deployment geometry."""

    if target not in ("same-animal", "held-out-lab"):
        raise ValueError(f"unknown IBL protocol target: {target!r}")
    held_out = target == "held-out-lab"
    outer_seed = BOOTSTRAP_SEED + (10_000 if held_out else 0)
    candidates = _candidate_specs(held_out=held_out)
    outer_validation = ValidationSpec(
        (
            ValidationGeometry.HELD_OUT_GROUP_FUTURE_SESSION
            if held_out
            else ValidationGeometry.FUTURE_SESSION
        ),
        (
            "behavio.validation.leave_one_lab_out_session_forecast_splits"
            if held_out
            else "behavio.validation.cohort_forward_session_splits"
        ),
        PredictionInformation.FILTERED,
        group_unit="lab" if held_out else None,
        origin=TRAIN_SESSION_COUNT - 1,
        horizon=(5,),
        settings=(
            (
                Setting("train_session_count", TRAIN_SESSION_COUNT)
                if held_out
                else Setting("min_train_sessions", TRAIN_SESSION_COUNT)
            ),
            Setting("horizon", 1),
            *((Setting("lab_column", "lab"),) if held_out else ()),
        ),
    )
    inner_validation = ValidationSpec(
        (
            ValidationGeometry.HELD_OUT_GROUP_FUTURE_SESSION
            if held_out
            else ValidationGeometry.FUTURE_SESSION
        ),
        (
            "behavio.validation.leave_one_lab_out_session_forecast_splits"
            if held_out
            else "behavio.validation.cohort_forward_session_splits"
        ),
        PredictionInformation.FILTERED,
        group_unit="lab" if held_out else None,
        origin=3 if held_out else 2,
        horizon=(4,) if held_out else (3, 4),
        settings=(
            (Setting("train_session_count", 4) if held_out else Setting("min_train_sessions", 3)),
            Setting("horizon", 1),
            *((Setting("lab_column", "lab"),) if held_out else ()),
        ),
    )
    return StudyProtocol(
        identifier=f"ibl2021-nested-{target}-v1",
        title=(
            "IBL 2021 held-out-lab future-session selection"
            if held_out
            else "IBL 2021 same-animal future-session selection"
        ),
        question=(
            "Which fixed hierarchical trajectory candidate transfers to a final ordinal "
            "session in an unseen laboratory?"
            if held_out
            else "Which fixed hierarchical trajectory candidate forecasts the final ordinal "
            "session of represented animals?"
        ),
        source=SourceSpec(
            adapter="ibl-one-exact-dataset",
            release="2021_Q1_IBL_et_al_Behaviour",
            locator="doi:10.7554/eLife.63711",
            checksum_algorithm="sha256-manifest",
            checksum=EXPECTED_MANIFEST_SHA256,
            identity_columns=("source_ibl_dataset_id", "trial"),
            metadata=(
                Setting("outcome-blind-manifest", True),
                Setting("source-sessions", 468),
                Setting("source-trials-before-panel", 260_833),
                Setting("source-row-cap-before-choice-eligibility", 100),
            ),
        ),
        cohort=CohortSpec(
            predicates=(
                CohortPredicate(
                    "phase",
                    PredicateOperator.IN,
                    ("early", "late_training"),
                    "retain the three early and three final pre-transition sessions",
                ),
            ),
            selection_columns=("phase", "session_order", "lab"),
            outcome_blind=True,
            expected_subjects=78,
            expected_sessions=468,
            expected_observations=46_152,
        ),
        units=(
            UnitSpec("animal", "subject", UnitRole.EXPERIMENTAL),
            UnitSpec("session", "session", UnitRole.REPEATED_MEASURES, "animal"),
            UnitSpec("lab", "lab", UnitRole.AGGREGATION),
        ),
        observations=(
            ObservationSpec("choice", ObservationRole.OUTCOME, "binary", allowed_values=(0, 1)),
            ObservationSpec("stimulus", ObservationRole.PREDICTOR, "continuous"),
        ),
        clocks=(
            ProtocolClockSpec(
                "endpoint-window-position",
                "session_order",
                "ordinal-session",
                "animal",
                "three-early-plus-three-final-pre-transition-sessions",
            ),
        ),
        panel=PanelSpec("animal", "session", "endpoint-window-position", 6, True),
        estimands=(
            EstimandSpec(
                (
                    "animal-balanced-unseen-lab-final-session-log-loss"
                    if held_out
                    else "animal-balanced-represented-final-session-log-loss"
                ),
                (
                    "eligible animals in each entirely held-out empirical lab"
                    if held_out
                    else "all eligible represented animals"
                ),
                ("choice",),
                "selected procedure's untouched final-session predictive log loss",
                "animal",
                AggregationWeighting.EQUAL_UNIT,
            ),
        ),
        transforms=(),
        validation=outer_validation,
        candidates=candidates,
        comparison=ComparisonSpec(
            ScoreMetric.LOG_LOSS,
            "animal",
            AggregationWeighting.EQUAL_UNIT,
            "paired-unit-bootstrap",
            0.95,
            BOOTSTRAP_RESAMPLES,
            outer_seed,
            True,
            WinnerPolicy.NO_AUTOMATIC_WINNER,
            multiplicity=ComparisonMultiplicity.BENJAMINI_HOCHBERG,
        ),
        recovery=(),
        reporting=ReportingSpec(
            ("denominators", "inner-selections", "outer-scores", "fit-audits"),
            ("ibl-nested-selection",),
            ("optimization", "calibration", "outer-boundary-audit"),
            (
                "endpoint positions are ordinal and do not represent uniform elapsed time",
                "cohort entry is conditioned on the training-policy transition",
                "held-out-lab inference concerns nine empirical laboratories",
                "the upper smoothness grid boundary was selected in the pinned analysis",
            ),
            (
                "causal effect of laboratory",
                "generalization to a population of laboratories",
                "unique cognitive mechanism",
                "selection stability beyond the declared candidate grid",
            ),
        ),
        selection=NestedSelectionSpec(
            tuple(candidate.name for candidate in candidates),
            inner_validation,
            ScoreMetric.LOG_LOSS,
            "animal",
            SelectionTieBreak.DECLARED_ORDER,
            INNER_BOOTSTRAP_RESAMPLES,
            outer_seed,
        ),
    )


def compile_protocol(
    target: Target,
    cache_directory: Path = DEFAULT_CACHE,
) -> CompiledProtocol:
    """Retrieve the exact public release cache, materialize, and compile both geometries."""

    panel = build_panel(load_study(cache_directory))
    protocol = build_protocol(target).freeze()
    materialized = materialize_protocol(protocol, panel)
    models = _candidates()
    if target == "same-animal":
        outer = cohort_forward_session_splits(
            materialized.study,
            min_train_sessions=TRAIN_SESSION_COUNT,
            horizon=1,
        )

        def inner(training, _fold):
            return cohort_forward_session_splits(training, min_train_sessions=3, horizon=1)

    else:
        outer = leave_one_lab_out_session_forecast_splits(
            materialized.study,
            train_session_count=TRAIN_SESSION_COUNT,
            horizon=1,
        )

        def inner(training, _fold):
            return leave_one_lab_out_session_forecast_splits(
                training,
                train_session_count=4,
                horizon=1,
            )

    return compile_execution_plan(
        materialized,
        outer,
        capabilities={name: model_capabilities(model) for name, model in models.items()},
        inner_splitter=inner,
    )


def run_observed(
    target: Target,
    cache_directory: Path = DEFAULT_CACHE,
) -> NestedProtocolRun:
    """Execute nested selection on the exact public panel."""

    return run_nested_protocol(compile_protocol(target, cache_directory), _candidates())


def recorded_parity(target: Target) -> IBLProtocolParity:
    """Verify the typed declaration against the pinned 0.16 result contract."""

    manifest = load_manifest()
    legacy = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
    key = (
        "within_subject_future_session"
        if target == "same-animal"
        else "held_out_lab_future_session"
    )
    result = legacy[key]
    protocol = build_protocol(target)
    source_parity = (
        protocol.source.checksum == manifest["sessions_sha256"]
        and protocol.source.release == manifest["release_tag"]
        and protocol.cohort.outcome_blind
    )
    denominator_parity = (
        protocol.cohort.expected_subjects == legacy["panel"]["subjects"]
        and protocol.cohort.expected_sessions == legacy["panel"]["sessions"]
        and protocol.cohort.expected_observations == legacy["panel"]["trials"]
    )
    candidate_parity = (
        tuple(candidate.name for candidate in protocol.candidates)
        == tuple(legacy["selection_contract"]["candidate_order"])
        == CANDIDATES
    )
    expected_folds = 1 if target == "same-animal" else legacy["panel"]["labs"]
    outer_geometry_parity = (
        result["outer_folds"] == expected_folds
        and result["outer_scored_trials"] == legacy["panel"]["trials_by_position"]["5"]
    )
    nested_selection_parity = result["selection_counts"] == {
        "static": 0,
        "drift_smoothness_1": 0,
        "drift_smoothness_3": 0,
        "drift_smoothness_9": expected_folds,
    }
    return IBLProtocolParity(
        target,
        protocol.fingerprint,
        source_parity,
        denominator_parity,
        candidate_parity,
        outer_geometry_parity,
        nested_selection_parity,
    )


def numerical_parity(target: Target, run: NestedProtocolRun) -> IBLProtocolParity:
    """Add exact outer score, interval, audit, and selection parity to the design gates."""

    base = recorded_parity(target)
    legacy = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
    key = (
        "within_subject_future_session"
        if target == "same-animal"
        else "held_out_lab_future_session"
    )
    expected = legacy[key]
    interval = run.report.unit_balanced_log_loss_interval
    expected_interval = expected["subject_bootstrap_log_loss_95_interval"]
    return IBLProtocolParity(
        target,
        run.protocol.fingerprint,
        base.source_parity,
        base.denominator_parity,
        base.candidate_parity,
        base.outer_geometry_parity,
        run.report.selection_counts == expected["selection_counts"],
        score_parity=(
            abs(run.report.unit_balanced_log_loss - expected["subject_balanced_log_loss"]) < 1e-12
            and abs(run.report.pooled_log_loss - expected["pooled_trial_log_loss"]) < 1e-12
        ),
        interval_parity=(
            interval is not None
            and abs(interval.lower - expected_interval[0]) < 1e-12
            and abs(interval.upper - expected_interval[1]) < 1e-12
        ),
        audit_parity=run.report.eligible and expected["outer_audit_status"] != "fail",
    )


def _base_settings(settings: tuple[Setting, ...], *, depth: int = 1) -> tuple[Setting, ...]:
    """Re-key settings onto the wrapped model of a composed candidate.

    A protocol candidate is one implementation name plus flat scalar settings, so a
    composition is spelled by reference: ``base`` names the wrapped implementation and a
    ``base.`` prefix carries its own settings, nested once per layer.
    """

    prefix = "base." * depth
    return tuple(Setting(f"{prefix}{setting.name}", setting.value) for setting in settings)


def _candidate_specs(*, held_out: bool) -> tuple[CandidateSpec, ...]:
    common = (
        Setting("covariates", ("stimulus",)),
        Setting("choice_lags", 1),
        Setting("l2", 0.02),
    )
    candidates = [
        CandidateSpec(
            "static",
            "behavio.compose.hierarchical",
            (
                Setting("base", "behavio.models.BernoulliHistoryGLM"),
                *_base_settings(common),
                Setting("over", "subject"),
                Setting("scale", 0.4),
            ),
            ("choice",),
            supports_unseen_subjects=held_out,
            supports_unseen_groups=held_out,
        )
    ]
    for smoothness in (1.0, 3.0, 9.0):
        candidates.append(
            CandidateSpec(
                f"drift_smoothness_{int(smoothness)}",
                "behavio.compose.hierarchical",
                (
                    Setting("base", "behavio.compose.smooth"),
                    Setting("base.base", "behavio.models.BernoulliHistoryGLM"),
                    *_base_settings(common, depth=2),
                    Setting("base.over", "session_order"),
                    Setting("base.knots", KNOTS),
                    Setting("base.smoothness", smoothness),
                    Setting("over", "subject"),
                    Setting("scale", 0.4),
                ),
                ("choice",),
                supports_unseen_subjects=held_out,
                supports_unseen_groups=held_out,
            )
        )
    return tuple(candidates)


def compact_result(run: NestedProtocolRun) -> dict[str, object]:
    """Retain reviewable numerical evidence without duplicating pointwise predictions."""

    manifest = run.compiled.materialized.manifest
    interval = run.report.unit_balanced_log_loss_interval
    return {
        "protocol": run.protocol.to_dict(),
        "cohort": {
            "fingerprint": manifest.fingerprint,
            "source_observations": manifest.source_observations,
            "selected_observations": manifest.selected_observations,
            "selected_subjects": manifest.selected_subjects,
            "selected_sessions": manifest.selected_sessions,
        },
        "plan": {
            "fingerprint": run.compiled.plan.fingerprint,
            "audit_passed": run.compiled.plan.audit.passed,
            "issues": [asdict(issue) for issue in run.compiled.plan.audit.issues],
            "folds": [
                {
                    "identifier": fold.identifier,
                    "fit_rows": len(fold.fit_rows),
                    "prediction_context_rows": len(fold.prediction_context_rows),
                    "scored_rows": len(fold.scored_rows),
                    "excluded_rows": len(fold.excluded_rows),
                    "inner_folds": len(fold.inner_folds),
                }
                for fold in run.compiled.plan.folds
            ],
        },
        "evaluation": {
            "schema_version": run.report.schema_version,
            "fingerprint": run.report.fingerprint,
            "eligible": run.report.eligible,
            "selected_candidates": run.report.selected_candidates,
            "selection_counts": run.report.selection_counts,
            "pooled_log_loss": run.report.pooled_log_loss,
            "unit_balanced_log_loss": run.report.unit_balanced_log_loss,
            "unit_balanced_log_loss_interval": interval.to_dict() if interval else None,
            "calibration": asdict(run.report.calibration),
            "folds": [
                {
                    "outer_fold": fold.outer_fold,
                    "selected_candidate": fold.selected_candidate,
                    "selection_failure": fold.selection_failure,
                    "inner_candidates": {
                        candidate.name: {
                            "eligible": candidate.eligible,
                            "audit_status": candidate.audit_status.value,
                            "unit_balanced_log_loss": candidate.unit_balanced_log_loss,
                        }
                        for candidate in fold.inner_candidates
                    },
                    "outer_result": (
                        {
                            "eligible": fold.outer_result.eligible,
                            "audit_status": fold.outer_result.audit_status.value,
                            "pooled_log_loss": fold.outer_result.pooled_log_loss,
                            "unit_balanced_log_loss": (fold.outer_result.unit_balanced_log_loss),
                            "scored_observations": sum(
                                score.n_observations for score in fold.outer_result.unit_scores
                            ),
                        }
                        if fold.outer_result
                        else None
                    ),
                }
                for fold in run.report.folds
            ],
        },
    }


def _main() -> None:
    result = {}
    for target in ("same-animal", "held-out-lab"):
        run = run_observed(target)
        parity = numerical_parity(target, run)
        result[target] = {
            **compact_result(run),
            "parity": {**asdict(parity), "passed": parity.passed},
        }
    PROTOCOL_RESULT.write_text(render(result), encoding="utf-8")


if __name__ == "__main__":
    _main()
