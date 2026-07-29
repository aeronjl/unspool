"""Compile and execute the Cell 2025 flagship through the 0.20 protocol workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from behavio import (
    AggregationWeighting,
    CandidateSpec,
    CohortPredicate,
    CohortSpec,
    ComparisonSpec,
    CompiledProtocol,
    EstimandSpec,
    ObservationRole,
    ObservationSpec,
    PanelSpec,
    PredicateOperator,
    PredictionInformation,
    ProtocolClockSpec,
    ProtocolRun,
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
    compile_execution_plan,
    historical_cohort_forecast_splits,
    materialize_protocol,
    model_capabilities,
    run_protocol,
)
from benchmarks.cell2025.benchmark import load_study
from benchmarks.cell2025.fetch_data import (
    DEFAULT_DESTINATION,
    FIGSHARE_ARTICLE_DOI,
    MEMBER_SHA256,
    sha256,
)
from benchmarks.cell2025_flagship.benchmark import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONTEXT_PAPER_DAYS,
    EARLY_BIAS_RECOVERY_REPEATS,
    EARLY_BIAS_RECOVERY_SEED,
    FORECAST_HORIZON,
    KNOTS,
    MODEL_ORDER,
    MODEL_RECOVERY_REPEATS,
    MODEL_RECOVERY_SEED,
    N_FOLDS,
    PARAMETER_RECOVERY_REPEATS,
    PARAMETER_RECOVERY_SEED,
    _models,
    build_forecast_panel,
)
from benchmarks.provenance import render

LEGACY_RESULT = Path(__file__).parents[1] / "cell2025_flagship" / "result.json"
PROTOCOL_RESULT = Path(__file__).with_name("result.json")


@dataclass(frozen=True, slots=True)
class CellProtocolParity:
    """Machine-readable parity gates against the pinned 0.19 flagship evidence."""

    protocol_fingerprint: str
    denominator_parity: bool
    fold_parity: bool
    candidate_parity: bool
    score_parity: bool | None
    interval_parity: bool | None
    audit_parity: bool | None

    @property
    def passed(self) -> bool:
        """Whether every gate that was evaluated passed."""

        values = (
            self.denominator_parity,
            self.fold_parity,
            self.candidate_parity,
            self.score_parity,
            self.interval_parity,
            self.audit_parity,
        )
        return all(value is not False for value in values)


def build_protocol() -> StudyProtocol:
    """Return the complete draft protocol for the Cell behavioural forecast."""

    psychometric = ("left_contrast", "right_contrast")
    late_phase = (
        *psychometric,
        "forecast_phase",
        "forecast_phase_left_contrast",
        "forecast_phase_right_contrast",
    )
    common = (Setting("choice_lags", 0), Setting("l2", 0.02))
    candidates = (
        CandidateSpec(
            "pooled_psychometric",
            "behavio.models.BernoulliHistoryGLM",
            (Setting("covariates", psychometric), *common),
            ("choice",),
        ),
        CandidateSpec(
            "late_phase_psychometric",
            "behavio.models.BernoulliHistoryGLM",
            (Setting("covariates", late_phase), *common),
            ("choice",),
        ),
        CandidateSpec(
            "early_bias_forecast",
            "behavio.models.BernoulliHistoryGLM",
            (
                Setting(
                    "covariates",
                    (
                        *late_phase,
                        "early_bias",
                        "early_bias_forecast_phase",
                        "early_bias_forecast_left_contrast",
                        "early_bias_forecast_right_contrast",
                    ),
                ),
                *common,
            ),
            ("choice",),
        ),
        CandidateSpec(
            "static_partial_pooling",
            "behavio.models.HierarchicalBernoulliHistoryGLM",
            (
                Setting("covariates", psychometric),
                Setting("subject_scale", 0.4),
                *common,
            ),
            ("choice",),
        ),
        CandidateSpec(
            "shared_smooth_trajectory",
            "behavio.models.SmoothBernoulliHistoryGLM",
            (
                Setting("covariates", psychometric),
                Setting("knots", KNOTS),
                Setting("smoothness", 3.0),
                Setting("shared_trajectory", True),
                *common,
            ),
            ("choice",),
        ),
        CandidateSpec(
            "hierarchical_smooth_trajectory",
            "behavio.models.HierarchicalSmoothBernoulliHistoryGLM",
            (
                Setting("covariates", psychometric),
                Setting("knots", KNOTS),
                Setting("smoothness", 3.0),
                Setting("subject_scale", 0.4),
                Setting("subject_smoothness", 3.0),
                *common,
            ),
            ("choice",),
        ),
    )
    structural = (
        "pooled_psychometric",
        "static_partial_pooling",
        "shared_smooth_trajectory",
        "hierarchical_smooth_trajectory",
    )
    return StudyProtocol(
        identifier="cell2025-behavioural-forecast-v1",
        title="Cell 2025 behavioural learning trajectories",
        question=(
            "Do early behavioural measurements forecast an animal's final-session choice "
            "policy beyond pooled, phase-control, and longitudinal alternatives?"
        ),
        source=SourceSpec(
            adapter="cell2025-figshare-csv",
            release="Cell-2025-publication-release",
            locator=f"doi:{FIGSHARE_ARTICLE_DOI}",
            checksum_algorithm="sha256",
            checksum=MEMBER_SHA256,
            identity_columns=("subject", "source_session", "source_trial"),
            metadata=(Setting("derived-session-contract", "paper-day"),),
        ),
        cohort=CohortSpec(
            predicates=(
                CohortPredicate(
                    "phase",
                    PredicateOperator.IN,
                    ("context", "forecast"),
                    "retain the frozen first-eight and final-five paper-day windows",
                ),
            ),
            selection_columns=("phase", "session_order", "paper_session_order"),
            outcome_blind=True,
            expected_subjects=30,
            expected_sessions=390,
            expected_observations=73_042,
        ),
        units=(
            UnitSpec("animal", "subject", UnitRole.EXPERIMENTAL),
            UnitSpec("paper-day", "session", UnitRole.REPEATED_MEASURES, "animal"),
        ),
        observations=(
            ObservationSpec("choice", ObservationRole.OUTCOME, "binary", allowed_values=(0, 1)),
            ObservationSpec("left_contrast", ObservationRole.PREDICTOR, "continuous"),
            ObservationSpec("right_contrast", ObservationRole.PREDICTOR, "continuous"),
            ObservationSpec("forecast_phase", ObservationRole.PREDICTOR, "binary"),
            ObservationSpec(
                "forecast_phase_left_contrast", ObservationRole.PREDICTOR, "continuous"
            ),
            ObservationSpec(
                "forecast_phase_right_contrast", ObservationRole.PREDICTOR, "continuous"
            ),
            ObservationSpec("early_bias", ObservationRole.PREDICTOR, "continuous"),
            ObservationSpec("early_bias_forecast_phase", ObservationRole.PREDICTOR, "continuous"),
            ObservationSpec(
                "early_bias_forecast_left_contrast",
                ObservationRole.PREDICTOR,
                "continuous",
            ),
            ObservationSpec(
                "early_bias_forecast_right_contrast",
                ObservationRole.PREDICTOR,
                "continuous",
            ),
        ),
        clocks=(
            ProtocolClockSpec(
                "aligned-paper-day",
                "session_order",
                "ordinal-session",
                "animal",
                "first-eight-plus-final-five",
            ),
        ),
        panel=PanelSpec("animal", "paper-day", "aligned-paper-day", 13, True),
        estimands=(
            EstimandSpec(
                "animal-balanced-final-five-log-loss",
                "the 30 outcome-blindly eligible animals in the released cohort",
                ("choice",),
                "candidate minus reference final-five-session log loss",
                "animal",
                AggregationWeighting.EQUAL_UNIT,
            ),
        ),
        transforms=(
            TransformSpec(
                "early-bias-feature",
                "benchmarks.cell2025_flagship._recompute_early_bias_features",
                ("choice", "left_contrast", "right_contrast"),
                (
                    "early_bias",
                    "early_bias_forecast_phase",
                    "early_bias_forecast_left_contrast",
                    "early_bias_forecast_right_contrast",
                ),
                TransformVisibility.TRAINING_ONLY,
                (Setting("paper-days", tuple(range(4, 9))),),
                uses_outcomes=True,
            ),
        ),
        validation=ValidationSpec(
            ValidationGeometry.HISTORICAL_COHORT_FUTURE_SESSION,
            "behavio.validation.historical_cohort_forecast_splits",
            PredictionInformation.FILTERED,
            settings=(
                Setting("context_session_count", len(CONTEXT_PAPER_DAYS)),
                Setting("horizon", FORECAST_HORIZON),
                Setting("n_folds", N_FOLDS),
            ),
        ),
        candidates=candidates,
        comparison=ComparisonSpec(
            ScoreMetric.LOG_LOSS,
            "animal",
            AggregationWeighting.EQUAL_UNIT,
            "paired-unit-bootstrap",
            0.95,
            BOOTSTRAP_RESAMPLES,
            BOOTSTRAP_SEED,
            True,
            WinnerPolicy.INTERVAL_EXCLUDES_ZERO,
            "pooled_psychometric",
        ),
        recovery=(
            RecoverySpec(
                "structural-model-recovery",
                RecoveryKind.MODEL,
                True,
                MODEL_RECOVERY_REPEATS,
                MODEL_RECOVERY_SEED,
                "selection-rate",
                0.75,
                ("unique latent mechanism",),
                candidate_names=structural,
                scenarios=(Setting("worlds", structural),),
            ),
            RecoverySpec(
                "hierarchical-path-recovery",
                RecoveryKind.PARAMETER,
                True,
                PARAMETER_RECOVERY_REPEATS,
                PARAMETER_RECOVERY_SEED,
                "parameter-rmse",
                0.4,
                ("precise individual trajectory recovery",),
                candidate_names=("hierarchical_smooth_trajectory",),
            ),
            RecoverySpec(
                "early-bias-feature-recovery",
                RecoveryKind.OUTCOME_DERIVED_FEATURE,
                True,
                EARLY_BIAS_RECOVERY_REPEATS,
                EARLY_BIAS_RECOVERY_SEED,
                "selection-rate",
                0.8,
                ("stable natural strategy types",),
                candidate_names=("late_phase_psychometric", "early_bias_forecast"),
                scenarios=(
                    Setting(
                        "worlds",
                        (
                            "null_no_subject_signal",
                            "context_predicts_late_asymmetry",
                            "reward_history_without_stable_strategy",
                        ),
                    ),
                ),
            ),
        ),
        reporting=ReportingSpec(
            ("denominators", "fold-scores", "fit-audits", "recovery-gates"),
            ("cell2025-forecast", "cell2025-recovery"),
            ("optimization", "calibration", "boundary-estimates"),
            (
                "one internally validated cohort",
                "historical-cohort deployment assumes completed reference trajectories",
                "early bias is outcome-derived and must be recomputed inside training data",
                "the behavioural workflow does not reproduce neural or multimodal analyses",
            ),
            (
                "unique latent mechanism",
                "precise individual trajectory recovery",
                "stable natural strategy types",
                "causal effect of learning history",
                "transport to another laboratory",
            ),
        ),
    )


def compile_protocol(path: Path = DEFAULT_DESTINATION) -> CompiledProtocol:
    """Load the checksum-pinned public table and compile the frozen protocol."""

    observed = sha256(path)
    if observed != MEMBER_SHA256:
        raise ValueError(f"Cell source checksum mismatch: {observed}")
    panel = build_forecast_panel(load_study(path))
    protocol = build_protocol().freeze()
    materialized = materialize_protocol(protocol, panel)
    settings = {setting.name: setting.value for setting in protocol.validation.settings}
    splits = historical_cohort_forecast_splits(materialized.study, **settings)
    models = _models()
    return compile_execution_plan(
        materialized,
        splits,
        capabilities={name: model_capabilities(model) for name, model in models.items()},
    )


def run_observed(path: Path = DEFAULT_DESTINATION) -> ProtocolRun:
    """Execute the six candidates under the audited protocol plan."""

    return run_protocol(compile_protocol(path), _models())


def parity(run: ProtocolRun | None = None) -> CellProtocolParity:
    """Check denominator, fold, candidate, and optional numerical 0.19 parity."""

    compiled = run.compiled if run else compile_protocol()
    legacy = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))["historical_cohort_forecast"]
    panel = legacy["panel"]
    denominator_parity = (
        compiled.materialized.manifest.selected_observations == panel["n_trials"]
        and compiled.materialized.manifest.selected_subjects == panel["n_subjects"]
        and compiled.materialized.manifest.selected_sessions
        == panel["n_derived_paper_day_sessions"]
    )
    fold_parity = len(compiled.plan.folds) == len(legacy["folds"]) and all(
        len(fold.fit_rows) == expected["n_train_rows"]
        and len(fold.prediction_context_rows) == expected["n_prediction_context_rows"]
        and len(fold.scored_rows) == expected["n_test_rows"]
        for fold, expected in zip(compiled.plan.folds, legacy["folds"], strict=True)
    )
    models = _models()
    candidate_parity = tuple(candidate.name for candidate in compiled.protocol.candidates) == tuple(
        legacy["model_order"]
    ) == MODEL_ORDER and all(
        models[name].signature == legacy["models"][name]["model_signature"] for name in MODEL_ORDER
    )
    if run is None:
        score_parity = interval_parity = audit_parity = None
    else:
        results = {candidate.name: candidate for candidate in run.report.candidates}
        score_parity = all(
            abs(
                results[name].unit_balanced_log_loss
                - legacy["models"][name]["unit_balanced_log_loss"]
            )
            < 1e-12
            and abs(results[name].pooled_log_loss - legacy["models"][name]["pooled_log_loss"])
            < 1e-12
            for name in MODEL_ORDER
        )
        interval_parity = all(
            _interval_matches(
                results[name].unit_balanced_log_loss_interval,
                legacy["models"][name]["unit_balanced_log_loss_interval"],
            )
            for name in MODEL_ORDER
        )
        audit_parity = all(
            results[name].audit_status.value == legacy["models"][name]["audit_status"]
            for name in MODEL_ORDER
        )
    return CellProtocolParity(
        compiled.protocol.fingerprint,
        denominator_parity,
        fold_parity,
        candidate_parity,
        score_parity,
        interval_parity,
        audit_parity,
    )


def _interval_matches(observed: Any, expected: dict[str, Any]) -> bool:
    return observed is not None and all(
        abs(getattr(observed, name) - expected[name]) < 1e-12
        for name in ("estimate", "lower", "upper", "confidence_level")
    )


def _main() -> None:
    run = run_observed()
    parity_result = parity(run)
    manifest = run.compiled.materialized.manifest
    result = {
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
            "audit": [asdict(issue) for issue in run.compiled.plan.audit.issues],
            "folds": [
                {
                    "identifier": fold.identifier,
                    "fit_rows": len(fold.fit_rows),
                    "prediction_context_rows": len(fold.prediction_context_rows),
                    "scored_rows": len(fold.scored_rows),
                    "excluded_rows": len(fold.excluded_rows),
                }
                for fold in run.compiled.plan.folds
            ],
        },
        "evaluation": run.report.to_dict(retain_predictions=False),
        "parity": {**asdict(parity_result), "passed": parity_result.passed},
    }
    PROTOCOL_RESULT.write_text(render(result), encoding="utf-8")


if __name__ == "__main__":
    _main()
