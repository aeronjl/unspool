"""Tests for common execution under an audited protocol plan."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from test_compiler import (
    capabilities,
    frozen_nested_protocol,
    frozen_small_protocol,
    source_study,
)
from test_protocol import example_protocol

from behavio.comparison import ComparisonMultiplicity
from behavio.compiler import compile_execution_plan, materialize_protocol
from behavio.compose import smooth
from behavio.contracts.audit import FitDiagnostics
from behavio.evaluation import FoldStage
from behavio.models import (
    BernoulliHistoryGLM,
    FitResult,
    ModelCapabilities,
    Prediction,
    PredictionMode,
)
from behavio.protocol import (
    CandidateSpec,
    ProtocolState,
    ScoreMetric,
    Setting,
    WinnerPolicy,
)
from behavio.runner import (
    DeclarationCheck,
    ProtocolRunError,
    RankingStatus,
    run_nested_protocol,
    run_protocol,
    verify_candidate_declarations,
)
from behavio.study import Study
from behavio.validation import cohort_forward_session_splits


def compiled_small_protocol(candidates: tuple[CandidateSpec, ...] = ()):
    materialized = materialize_protocol(frozen_small_protocol(candidates), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    return compile_execution_plan(materialized, splits, capabilities=capabilities())


def declared_glm(name: str, **hyperparameters) -> CandidateSpec:
    """Declare a ``BernoulliHistoryGLM`` candidate exactly as it will be supplied."""

    return CandidateSpec(
        name=name,
        implementation="behavio.models.BernoulliHistoryGLM",
        hyperparameters=tuple(
            Setting(key, value) for key, value in sorted(hyperparameters.items())
        ),
        scored_columns=("choice",),
    )


def candidate_models():
    return {
        "static": BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=0.1),
        "smooth": BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=1.0),
    }


def test_runner_retains_fits_predictions_scores_calibration_and_comparisons() -> None:
    run = run_protocol(compiled_small_protocol(), candidate_models())

    assert run.protocol.state == ProtocolState.EVALUATED
    assert run.protocol.lifecycle[-1].artifact_fingerprint == run.report.fingerprint
    assert len(run.report.candidates) == 2
    assert len(run.report.paired_comparisons) == 1
    for candidate in run.report.candidates:
        assert candidate.eligible
        assert len(candidate.folds) == 1
        assert len(candidate.fold_predictions[0]) == 4
        assert len(candidate.unit_scores) == 2
        assert candidate.pooled_log_loss is not None
        assert candidate.unit_balanced_log_loss_interval is not None
        assert candidate.calibration.available
        assert candidate.calibration.n_observations == 4
        assert len(candidate.folds[0].fit_audit.issues) >= 0
    serialized = run.report.canonical_json()
    assert '"predictions"' in serialized
    assert '"covariance"' not in serialized
    assert "NaN" not in serialized


def test_declared_brier_score_controls_summaries_comparisons_and_ranking() -> None:
    protocol = frozen_small_protocol()
    protocol = replace(
        protocol,
        comparison=replace(
            protocol.comparison,
            metric=ScoreMetric.BRIER,
            winner_policy=WinnerPolicy.LOWEST_POINT_ESTIMATE,
        ),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())

    run = run_protocol(compiled, candidate_models())

    for candidate in run.report.candidates:
        assert candidate.score is not None
        assert candidate.score.metric == ScoreMetric.BRIER
        assert candidate.unit_balanced_brier_score is not None
        assert candidate.pooled_brier_score is not None
        assert candidate.unit_balanced_brier_interval is not None
        assert candidate.unit_balanced_log_loss is None
        assert candidate.pooled_log_loss is None
        assert candidate.unit_balanced_log_loss_interval is None
    left, right = run.report.candidates
    comparison = run.report.paired_comparisons[0]
    expected_difference = sum(
        left_score.brier_score - right_score.brier_score
        for left_score, right_score in zip(left.unit_scores, right.unit_scores, strict=True)
        if left_score.brier_score is not None and right_score.brier_score is not None
    ) / len(left.unit_scores)
    assert comparison.metric == ScoreMetric.BRIER
    assert comparison.left_minus_right.estimate == pytest.approx(expected_difference)
    assert (
        run.report.ranking.winner
        == min(
            run.report.candidates,
            key=lambda candidate: candidate.score.unit_balanced_score,  # type: ignore[union-attr]
        ).name
    )


def test_identical_candidates_retain_an_unresolved_ranking() -> None:
    settings = {"covariates": ("stimulus",), "choice_lags": 0, "l2": 0.5}
    model = BernoulliHistoryGLM(**settings)
    run = run_protocol(
        compiled_small_protocol(
            (declared_glm("static", **settings), declared_glm("smooth", **settings))
        ),
        {"static": model, "smooth": model},
    )

    comparison = run.report.paired_comparisons[0]
    assert comparison.left_minus_right.estimate == pytest.approx(0.0)
    assert comparison.left_minus_right.lower == pytest.approx(0.0)
    assert comparison.left_minus_right.upper == pytest.approx(0.0)
    assert run.report.ranking.status == RankingStatus.UNRESOLVED
    assert run.report.ranking.winner is None


class FailingFitModel:
    """A complete estimator whose optimizer always throws.

    It satisfies the whole ``BehaviourEstimator`` contract deliberately: the thing under
    test is that a *fold* failure is retained, not that a malformed object is rejected,
    and those are different findings.
    """

    model_name = "deliberate-failure"
    signature = "deliberate-failure[v1]"
    scored_columns = ("choice",)
    required_task_columns = ()
    supported_prediction_modes = (PredictionMode.FILTERED,)

    def fit(self, study):
        raise RuntimeError(f"deliberate failure for {len(study)} rows")

    def predict(self, study, fit, *, mode=PredictionMode.FILTERED):  # pragma: no cover
        raise RuntimeError("unreachable: fit always fails first")

    def pointwise_log_prob(self, study, fit, *, mode=PredictionMode.FILTERED):
        raise RuntimeError("unreachable: fit always fails first")  # pragma: no cover


class NotAnEstimator:
    """An object that does not satisfy the estimator contract at all."""

    model_name = "not-an-estimator"
    signature = "not-an-estimator[v1]"


def test_candidate_failure_is_retained_while_other_candidate_completes() -> None:
    run = run_protocol(
        compiled_small_protocol(
            (
                CandidateSpec(
                    name="static",
                    implementation=f"{FailingFitModel.__module__}.FailingFitModel",
                    hyperparameters=(),
                    scored_columns=("choice",),
                ),
                declared_glm("smooth", covariates=("stimulus",), choice_lags=0, l2=1.0),
            )
        ),
        {
            "static": FailingFitModel(),
            "smooth": candidate_models()["smooth"],
        },
    )

    failed, completed = run.report.candidates
    assert not failed.eligible
    assert failed.folds == ()
    assert len(failed.failures) == 1
    assert failed.failures[0].stage == FoldStage.FIT
    assert failed.failures[0].exception_type == "RuntimeError"
    assert "deliberate failure" in failed.failures[0].message
    assert completed.eligible
    assert run.report.ranking.status == RankingStatus.RESOLVED
    assert run.report.ranking.winner == "smooth"


def test_runner_rejects_registry_drift_from_frozen_candidates() -> None:
    with pytest.raises(ProtocolRunError, match="exactly match"):
        run_protocol(
            compiled_small_protocol(),
            {"static": candidate_models()["static"]},
        )


def test_runner_refuses_a_failed_pre_fit_audit() -> None:
    materialized = materialize_protocol(frozen_small_protocol(), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    failed = compile_execution_plan(materialized, splits, capabilities={})

    with pytest.raises(ProtocolRunError, match="audited execution plan"):
        run_protocol(failed, candidate_models())


def compiled_nested(study=None):
    materialized = materialize_protocol(frozen_nested_protocol(), study or source_study())
    outer = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    return compile_execution_plan(
        materialized,
        outer,
        capabilities=capabilities(),
        inner_splitter=lambda training, _fold: cohort_forward_session_splits(
            training,
            min_train_sessions=1,
        ),
    )


def test_nested_runner_selects_inside_training_then_scores_untouched_outer_rows() -> None:
    run = run_nested_protocol(compiled_nested(), candidate_models())

    assert run.protocol.state == ProtocolState.EVALUATED
    assert run.protocol.lifecycle[-1].artifact_fingerprint == run.report.fingerprint
    assert run.report.eligible
    assert len(run.report.folds) == 1
    fold = run.report.folds[0]
    assert fold.selected_candidate in {"static", "smooth"}
    assert tuple(candidate.name for candidate in fold.inner_candidates) == ("static", "smooth")
    assert all(candidate.eligible for candidate in fold.inner_candidates)
    assert fold.outer_result is not None
    assert len(fold.outer_result.fold_predictions[0]) == 4
    assert len(run.report.unit_scores) == 2
    assert sum(run.report.selection_counts.values()) == 1


def test_nested_selection_uses_its_declared_brier_score() -> None:
    protocol = frozen_nested_protocol()
    protocol = replace(
        protocol,
        selection=replace(protocol.selection, metric=ScoreMetric.BRIER),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
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

    run = run_nested_protocol(compiled, candidate_models())

    fold = run.report.folds[0]
    assert all(
        candidate.score is not None and candidate.score.metric == ScoreMetric.BRIER
        for candidate in fold.inner_candidates
    )
    assert (
        fold.selected_candidate
        == min(
            fold.inner_candidates,
            key=lambda candidate: candidate.score.unit_balanced_score,  # type: ignore[union-attr]
        ).name
    )
    assert run.report.score is not None
    assert run.report.score.metric == ScoreMetric.LOG_LOSS


def test_nested_selection_and_outer_comparison_use_their_own_aggregation_units() -> None:
    protocol = frozen_nested_protocol()
    protocol = replace(
        protocol,
        selection=replace(protocol.selection, aggregation_unit="session"),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
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

    run = run_nested_protocol(compiled, candidate_models())

    fold = run.report.folds[0]
    assert all(len(candidate.unit_scores) == 2 for candidate in fold.inner_candidates)
    assert {
        score.unit for candidate in fold.inner_candidates for score in candidate.unit_scores
    } == {
        "a-s1",
        "b-s1",
    }
    assert {score.unit for score in run.report.unit_scores} == {"a", "b"}


def test_outer_outcomes_cannot_change_inner_selection() -> None:
    baseline = run_nested_protocol(compiled_nested(), candidate_models())
    source = source_study()
    columns = {name: source[name].copy() for name in source.columns}
    outer_rows = columns["session_order"] == 2
    columns["choice"][outer_rows] = 1 - columns["choice"][outer_rows]
    changed = run_nested_protocol(
        compiled_nested(type(source).from_columns(columns)),
        candidate_models(),
    )

    assert changed.report.selected_candidates == baseline.report.selected_candidates
    assert changed.report.unit_balanced_log_loss != baseline.report.unit_balanced_log_loss


def test_flat_runner_refuses_nested_selection_protocol() -> None:
    with pytest.raises(ProtocolRunError, match="must use run_nested_protocol"):
        run_protocol(compiled_nested(), candidate_models())


class LookalikeGLM:
    """A non-dataclass estimator whose class name collides with a declared one."""

    model_name = "lookalike"
    signature = "lookalike[v1]"


def test_declared_model_is_verified_and_leaves_the_evaluation_untouched() -> None:
    compiled = compiled_small_protocol()
    baseline = run_protocol(compiled, candidate_models())

    verified = run_protocol(compiled, candidate_models())

    assert baseline.report.fingerprint == verified.report.fingerprint
    assert [item.candidate for item in verified.verification] == ["static", "smooth"]
    assert all(item.verified for item in verified.verification)
    assert all(not item.unverifiable and not item.contradictions for item in verified.verification)
    static = verified.verification[0]
    assert static.declared_implementation == "behavio.models.BernoulliHistoryGLM"
    assert static.observed_implementation == "behavio.models.glm.BernoulliHistoryGLM"
    assert static.model_name == "bernoulli-history-glm"
    assert {finding.subject for finding in static.findings} == {
        "implementation",
        "hyperparameter:covariates",
        "hyperparameter:choice_lags",
        "hyperparameter:l2",
    }


def test_a_contradicting_implementation_refuses_to_produce_evidence() -> None:
    models = candidate_models()
    models["smooth"] = smooth(
        BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=1.0),
        over="session_order",
    )

    with pytest.raises(ProtocolRunError, match="contradict the frozen candidate declaration"):
        run_protocol(compiled_small_protocol(), models)


def test_a_contradicting_hyperparameter_refuses_to_produce_evidence() -> None:
    models = candidate_models()
    models["static"] = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=0, l2=0.5)

    with pytest.raises(ProtocolRunError) as error:
        run_protocol(compiled_small_protocol(), models)

    message = str(error.value)
    assert "static.hyperparameter:l2" in message
    assert "declared=0.1" in message and "supplied=0.5" in message


def test_verification_separates_contradiction_from_unverifiability() -> None:
    protocol = frozen_small_protocol(
        (
            CandidateSpec(
                name="static",
                implementation="unimported.package.LookalikeGLM",
                hyperparameters=(Setting("l2", 0.1), Setting("tolerance", 1e-9)),
                scored_columns=("choice",),
            ),
            declared_glm("smooth", covariates=("stimulus",), choice_lags=0, l2=2.0),
        )
    )

    verification = verify_candidate_declarations(
        protocol, {"static": LookalikeGLM(), "smooth": candidate_models()["smooth"]}
    )

    static, smooth = verification
    assert [finding.status for finding in static.findings] == [DeclarationCheck.UNVERIFIABLE] * 3
    assert "not imported" in static.findings[0].detail
    assert "not a dataclass" in static.findings[1].detail
    assert [finding.status for finding in smooth.findings] == [
        DeclarationCheck.VERIFIED,
        DeclarationCheck.VERIFIED,
        DeclarationCheck.VERIFIED,
        DeclarationCheck.CONTRADICTED,
    ]


def test_a_setting_with_no_matching_field_is_recorded_rather_than_failed() -> None:
    protocol = frozen_small_protocol(
        (
            declared_glm("static", covariates=("stimulus",), choice_lags=0, l2=0.1),
            CandidateSpec(
                name="smooth",
                implementation="behavio.models.BernoulliHistoryGLM",
                hyperparameters=(Setting("l2", 1.0), Setting("optimizer", "irls")),
                scored_columns=("choice",),
            ),
        )
    )
    materialized = materialize_protocol(protocol, source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    compiled = compile_execution_plan(materialized, splits, capabilities=capabilities())

    run = run_protocol(compiled, candidate_models())

    unverifiable = run.verification[1].unverifiable
    assert [finding.subject for finding in unverifiable] == ["hyperparameter:optimizer"]
    assert "no field of that name" in unverifiable[0].detail
    assert run.verification[1].contradictions == ()
    assert not run.verification[1].verified


def test_nested_selection_verifies_the_same_frozen_declaration() -> None:
    models = candidate_models()
    models["static"] = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.1)

    with pytest.raises(ProtocolRunError, match="hyperparameter:choice_lags"):
        run_nested_protocol(compiled_nested(), models)


#: Per-subject score penalties that put the leading candidate on the exact boundary the
#: multiplicity declaration decides. Both contrasts against the leader invert to a
#: two-sided bootstrap probability just under 0.05, so both intervals exclude zero; the
#: third contrast, between two near-identical rivals, is nowhere near separating. The
#: Benjamini-Hochberg step-up over ``(0.037, 0.037, 0.740)`` accepts nothing -- the largest
#: probability sinks the whole family -- while the uncorrected reading accepts two. The
#: numbers are held fixed rather than simulated so the boundary cannot drift.
MARGINAL_PENALTIES = (
    -0.009816,
    -0.044661,
    0.029201,
    0.033149,
    0.003637,
    0.022593,
    0.060199,
    0.042011,
    0.032122,
    0.031565,
)


def marginal_study() -> Study:
    """Ten animals, three sessions each, with the columns the example protocol declares."""

    rows: list[dict[str, object]] = []
    source_row = 0
    for index in range(len(MARGINAL_PENALTIES)):
        subject = f"m{index:02d}"
        for session_order in range(3):
            for trial in range(2):
                rows.append(
                    {
                        "subject": subject,
                        "session": f"{subject}-s{session_order}",
                        "trial": trial,
                        "session_order": session_order,
                        "choice": (index + trial) % 2,
                        "stimulus": float(trial * 2 - 1),
                        "species": "mouse",
                        "source_asset": "asset-01",
                        "source_row": source_row,
                    }
                )
                source_row += 1
    return Study.from_records(rows)


class PrescribedScoreModel:
    """An estimator whose per-subject log loss is fixed by construction.

    The thing under test is the declared multiplicity adjustment, not an optimizer, so the
    fold loop, the equal-unit aggregation, the paired bootstrap, the step-up and the winner
    rule all run for real while the scores they consume are held exactly where the verdict
    turns.
    """

    required_task_columns = ()
    supported_prediction_modes = (PredictionMode.FILTERED,)
    scored_columns = ("choice",)

    def __init__(self, name: str, penalties: Mapping[str, float]) -> None:
        self.model_name = name
        self.signature = f"{name}[prescribed]"
        self._penalties = dict(penalties)

    def _log_probabilities(self, study) -> np.ndarray:
        return -np.asarray(
            [self._penalties[str(subject)] for subject in study["subject"]], dtype=float
        )

    def fit(self, study) -> FitResult:
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=("intercept",),
            estimates=np.array([0.0]),
            standard_errors=np.array([0.1]),
            covariance=np.array([[0.01]]),
            n_observations=len(study),
            diagnostics=FitDiagnostics(
                converged=True,
                optimizer="prescribed",
                status=0,
                message="prescribed",
                n_iterations=1,
                objective=1.0,
                gradient_norm=0.0,
                hessian_condition=1.0,
                boundary_estimate=False,
            ),
        )

    def predict(self, study, fit, *, mode=PredictionMode.FILTERED) -> Prediction:
        probability = np.exp(self._log_probabilities(study))
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability / (1.0 - probability)),
            mode=PredictionMode(mode),
        )

    def pointwise_log_prob(self, study, fit, *, mode=PredictionMode.FILTERED) -> np.ndarray:
        return self._log_probabilities(study)


def marginal_candidates() -> dict[str, PrescribedScoreModel]:
    subjects = [f"m{index:02d}" for index in range(len(MARGINAL_PENALTIES))]
    base = {subject: 0.60 + 0.01 * index for index, subject in enumerate(subjects)}
    rival = {
        subject: base[subject] + penalty
        for subject, penalty in zip(subjects, MARGINAL_PENALTIES, strict=True)
    }
    twin = {
        subject: rival[subject] + (0.0004 if index % 2 == 0 else -0.0004)
        for index, subject in enumerate(subjects)
    }
    return {
        "best": PrescribedScoreModel("best", base),
        "rival_a": PrescribedScoreModel("rival_a", rival),
        "rival_b": PrescribedScoreModel("rival_b", twin),
    }


def compiled_marginal_protocol(multiplicity: ComparisonMultiplicity):
    implementation = f"{PrescribedScoreModel.__module__}.PrescribedScoreModel"
    candidates = tuple(
        CandidateSpec(
            name=name,
            implementation=implementation,
            hyperparameters=(),
            scored_columns=("choice",),
        )
        for name in ("best", "rival_a", "rival_b")
    )
    protocol = example_protocol(with_recovery=False)
    protocol = replace(
        protocol,
        candidates=candidates,
        cohort=replace(
            protocol.cohort,
            expected_subjects=10,
            expected_sessions=30,
            expected_observations=60,
        ),
        panel=replace(protocol.panel, minimum_sessions=3),
        comparison=replace(
            protocol.comparison,
            multiplicity=multiplicity,
            reference_candidate=None,
        ),
    )
    materialized = materialize_protocol(protocol.freeze(), marginal_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    capability = ModelCapabilities(
        scored_columns=("choice",),
        prediction_modes=(PredictionMode.FILTERED,),
        can_simulate=False,
        can_recover_parameters=False,
    )
    return compile_execution_plan(
        materialized,
        splits,
        capabilities={name: capability for name in ("best", "rival_a", "rival_b")},
    )


def test_the_declared_multiplicity_decides_which_candidate_wins() -> None:
    """Same scores, same folds, same seed; only the frozen adjustment differs."""

    uncorrected = run_protocol(
        compiled_marginal_protocol(ComparisonMultiplicity.NONE), marginal_candidates()
    )
    corrected = run_protocol(
        compiled_marginal_protocol(ComparisonMultiplicity.BENJAMINI_HOCHBERG),
        marginal_candidates(),
    )

    def probabilities(run):
        return {
            (item.left_model, item.right_model): round(item.two_sided_probability, 3)
            for item in run.report.paired_comparisons
        }

    # The evidence itself is identical: adjustment reads the same numbers either way.
    assert probabilities(uncorrected) == probabilities(corrected)
    assert probabilities(uncorrected) == {
        ("best", "rival_a"): 0.037,
        ("best", "rival_b"): 0.037,
        ("rival_a", "rival_b"): 0.740,
    }
    assert uncorrected.report.ranking.family.n_separated == 2
    assert corrected.report.ranking.family.n_separated == 2

    assert uncorrected.report.ranking.status is RankingStatus.RESOLVED
    assert uncorrected.report.ranking.winner == "best"
    assert uncorrected.report.ranking.family.n_decisive == 2
    assert "uncorrected per-contrast reading" in uncorrected.report.ranking.reason

    assert corrected.report.ranking.status is RankingStatus.UNRESOLVED
    assert corrected.report.ranking.winner is None
    assert corrected.report.ranking.family.n_decisive == 0
    assert "benjamini-hochberg adjustment" in corrected.report.ranking.reason


def test_the_runner_records_the_adjustment_the_protocol_froze() -> None:
    for multiplicity in ComparisonMultiplicity:
        run = run_protocol(compiled_marginal_protocol(multiplicity), marginal_candidates())

        assert run.report.ranking.family.multiplicity is multiplicity
        assert run.report.to_dict()["ranking"]["family"]["multiplicity"] == multiplicity.value
        assert run.protocol.comparison.multiplicity is multiplicity
