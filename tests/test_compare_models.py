from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from behavio import (
    BernoulliHistoryGLM,
    ScoreMetric,
    Study,
    cohort_forward_session_splits,
    compare_models,
    nested_select_model,
)
from behavio.compare import DEFAULT_COMPARISON_METRICS, UndeclaredMetric
from behavio.compose import smooth
from behavio.contracts import (
    BehaviourEstimator,
    ConvergenceStatus,
    FitAuditStatus,
    FitResult,
    PredictionMode,
)
from behavio.evaluate import leave_one_lab_out_session_forecast_splits, leave_one_session_out_splits

KNOTS = (0.0, 2.0, 5.0)


def comparison_study() -> Study:
    generator = np.random.default_rng(711)
    subjects = ("short", "medium", "long")
    trials_per_session = {"short": 15, "medium": 25, "long": 40}
    columns: dict[str, list[object]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "session_order": [],
        "stimulus": [],
    }
    for subject in subjects:
        for session in range(6):
            n_trials = trials_per_session[subject]
            columns["subject"].extend([subject] * n_trials)
            columns["session"].extend([f"{subject}-{session}"] * n_trials)
            columns["trial"].extend(range(n_trials))
            columns["session_order"].extend([session] * n_trials)
            columns["stimulus"].extend(generator.normal(size=n_trials))
    design = Study(columns)
    generator_model = _smooth_glm()
    return generator_model.simulate(
        design,
        generator_model.parameters_from_paths(
            {
                "intercept": [-0.3, -0.1, 0.1],
                "stimulus": [0.4, 1.0, 1.8],
                "choice_lag_1": [0.2, 0.2, 0.2],
            }
        ),
        seed=712,
    )


def candidates() -> dict[str, object]:
    return {
        "static": BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02),
        "smooth": _smooth_glm(),
    }


def _smooth_glm():
    return smooth(
        BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02),
        over="session_order",
        knots=KNOTS,
        smoothness=3.0,
        shared_trajectory=True,
    )


def test_comparison_retains_matched_scores_fits_audits_and_provenance() -> None:
    study = comparison_study()
    splits = cohort_forward_session_splits(study, min_train_sessions=4)

    report = compare_models(
        candidates(),
        study,
        splits,
        bootstrap_resamples=200,
        bootstrap_seed=17,
    )

    assert report.model_order == ("static", "smooth")
    assert report.scored_columns == ("choice",)
    assert report.winner in report.model_order
    assert len(report.splits) == 2
    assert len(report.pairwise_comparisons) == 1
    static = report.result_for("static")
    smooth = report.result_for("smooth")
    assert static.aggregation_units == ("short", "medium", "long")
    assert len(static.evaluations) == 2
    assert len(static.audits) == 2
    assert static.n_scored_observations == 2 * (15 + 25 + 40)
    assert static.unit_balanced_log_loss_interval.estimate == pytest.approx(
        static.unit_balanced_log_loss
    )
    assert static.pooled_log_loss == pytest.approx(
        np.average(static.unit_log_losses, weights=[30, 50, 80])
    )
    comparison = report.comparison_for("static", "smooth")
    assert comparison.left_minus_right.estimate == pytest.approx(
        static.unit_balanced_log_loss - smooth.unit_balanced_log_loss
    )
    assert not static.unit_log_losses.flags.writeable

    payload = report.to_dict()
    assert payload["folds"][0]["scheme"] == "cohort-forward-session"
    assert payload["bootstrap"]["unit"] == "subject"
    assert payload["scored_columns"] == ["choice"]
    assert len(payload["models"]["static"]["fit_audits"]) == 2
    json.dumps(payload, allow_nan=False)


def test_comparison_bootstrap_is_reproducible_and_paired() -> None:
    study = comparison_study()
    splits = cohort_forward_session_splits(study, min_train_sessions=5)

    first = compare_models(candidates(), study, splits, bootstrap_resamples=100, bootstrap_seed=91)
    second = compare_models(candidates(), study, splits, bootstrap_resamples=100, bootstrap_seed=91)

    assert first.to_dict() == second.to_dict()


def test_comparison_serializes_population_future_session_provenance() -> None:
    study = comparison_study()
    columns = {name: study[name] for name in study.columns}
    columns["lab"] = np.asarray(
        ["short-lab" if subject == "short" else "other-lab" for subject in study["subject"]]
    )
    panel = Study(columns)
    splits = leave_one_lab_out_session_forecast_splits(panel, train_session_count=5)

    report = compare_models(
        candidates(),
        panel,
        splits,
        bootstrap_resamples=20,
        bootstrap_seed=18,
    )

    payload = report.to_dict()
    assert len(payload["folds"]) == 2
    assert payload["folds"][0]["scheme"] == "leave-one-lab-out-session-forecast"
    assert payload["folds"][0]["train_session_orders"] == list(range(5))
    assert payload["folds"][0]["test_session_orders"] == [5]
    assert payload["folds"][0]["n_prediction_context_rows"] == 0


def test_failed_audit_candidate_is_retained_but_ineligible_to_win() -> None:
    study = comparison_study()
    splits = cohort_forward_session_splits(study, min_train_sessions=5)
    models = candidates()
    models = {
        "forced_nonconvergence": BernoulliHistoryGLM(
            predictors=("stimulus",),
            choice_lags=1,
            l2=0.02,
            max_iterations=1,
        ),
        "eligible": models["static"],
    }

    report = compare_models(
        models,
        study,
        splits,
        bootstrap_resamples=50,
        bootstrap_seed=92,
    )

    assert report.result_for("forced_nonconvergence").audit_status.value == "fail"
    assert report.model_order == ("forced_nonconvergence", "eligible")
    assert report.eligible_model_order == ("eligible",)
    assert report.winner == "eligible"
    assert "forced_nonconvergence" in report.to_dict()["models"]


class SilentAboutConvergence:
    """A stand-in for a wrapped third-party fitter: it searches and reports no verdict.

    This is the state PyDDM 0.9 leaves a fit in, and the state any fitter with a private
    stopping rule leaves it in. The three older ``converged`` values each misdescribe it,
    and the one that used to be chosen in practice -- ``False`` -- made the audit ``FAIL``
    and evicted a perfectly usable candidate from the comparison.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def signature(self) -> str:
        return f"{self._inner.signature}+silent"

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return tuple(self._inner.scored_columns)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return tuple(self._inner.required_task_columns)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return tuple(self._inner.supported_prediction_modes)

    def fit(self, study: Study) -> FitResult:
        fitted = self._inner.fit(study)
        return replace(
            fitted,
            model_signature=self.signature,
            diagnostics=replace(
                fitted.diagnostics,
                converged=ConvergenceStatus.UNREPORTED,
                status=None,
                message="the fitter reported no convergence flag",
            ),
        )

    def predict(self, study: Study, fit: FitResult, *, mode: Any = PredictionMode.FILTERED) -> Any:
        return self._inner.predict(
            study, replace(fit, model_signature=self._inner.signature), mode=mode
        )

    def pointwise_log_prob(
        self, study: Study, fit: FitResult, *, mode: Any = PredictionMode.FILTERED
    ) -> Any:
        return self._inner.pointwise_log_prob(
            study, replace(fit, model_signature=self._inner.signature), mode=mode
        )


def test_a_candidate_that_reports_no_convergence_verdict_stays_eligible() -> None:
    """Absence of evidence is recorded as a warning, not converted into a failure.

    The candidate must remain comparable -- nobody measured a failure -- while the gap is
    visible in its audit rather than smoothed over.
    """

    study = comparison_study()
    splits = cohort_forward_session_splits(study, min_train_sessions=5)
    reference = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.02)
    silent = SilentAboutConvergence(BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1))
    assert isinstance(silent, BehaviourEstimator)

    report = compare_models(
        {"silent": silent, "eligible": reference},
        study,
        splits,
        bootstrap_resamples=50,
        bootstrap_seed=92,
    )

    result = report.result_for("silent")
    assert result.audit_status is FitAuditStatus.WARNING
    assert "silent" in report.eligible_model_order
    assert report.winner in {"silent", "eligible"}
    for audit in result.audits:
        assert audit.convergence is ConvergenceStatus.UNREPORTED
        assert "optimizer_convergence_unreported" in audit.issue_codes
        assert "optimizer_nonconvergence" not in audit.issue_codes
        assert not audit.numerical.failed_to_converge
    # The state reaches a portable record, not only the live object.
    record = report.to_dict()["models"]["silent"]
    assert record["audit_status"] == "warning"
    assert record["fit_audits"][0]["convergence"] == "unreported"
    assert record["fit_audits"][0]["numerical"]["converged"] == "unreported"


def test_nested_selection_never_uses_outer_test_outcomes() -> None:
    study = comparison_study()
    outer_split = cohort_forward_session_splits(study, min_train_sessions=5)[0]

    def inner_splitter(training: Study):
        return cohort_forward_session_splits(training, min_train_sessions=3)

    first = nested_select_model(
        candidates(),
        study,
        (outer_split,),
        inner_splitter,
        bootstrap_resamples=100,
        inner_bootstrap_resamples=100,
        bootstrap_seed=23,
    )
    columns = {name: study[name] for name in study.columns}
    changed_choice = np.array(study["choice"], copy=True)
    changed_choice[outer_split.test_indices] = 1 - changed_choice[outer_split.test_indices]
    columns["choice"] = changed_choice
    changed = nested_select_model(
        candidates(),
        Study(columns),
        (outer_split,),
        inner_splitter,
        bootstrap_resamples=100,
        inner_bootstrap_resamples=100,
        bootstrap_seed=23,
    )

    assert first.folds[0].selected_model == changed.folds[0].selected_model
    assert first.folds[0].inner_report.to_dict() == changed.folds[0].inner_report.to_dict()
    assert first.folds[0].outer_evaluation.fit.n_observations == 5 * (15 + 25 + 40)
    assert first.selection_counts[first.folds[0].selected_model] == 1
    assert first.audit_status.value in {"pass", "warning"}
    assert first.n_scored_observations == 15 + 25 + 40
    assert first.unit_balanced_log_loss != changed.unit_balanced_log_loss
    json.dumps(first.to_dict(), allow_nan=False)


def test_comparison_rejects_implicit_interpolation_and_invalid_contracts() -> None:
    study = comparison_study()
    with pytest.raises(ValueError, match="every split to be prospective"):
        compare_models(
            candidates(),
            study,
            leave_one_session_out_splits(study),
            bootstrap_resamples=10,
        )
    with pytest.raises(ValueError, match="aggregation column"):
        compare_models(
            candidates(),
            study,
            cohort_forward_session_splits(study, min_train_sessions=5),
            aggregation_column="lab",
            bootstrap_resamples=10,
        )
    with pytest.raises(ValueError, match="positive integer"):
        compare_models(
            candidates(),
            study,
            cohort_forward_session_splits(study, min_train_sessions=5),
            bootstrap_resamples=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="not among the scored columns"):
        compare_models(
            {"other_outcome": BernoulliHistoryGLM(predictors=("stimulus",), outcome="response")},
            Study(
                {
                    **{name: study[name] for name in study.columns},
                    "response": study["choice"],
                }
            ),
            cohort_forward_session_splits(study, min_train_sessions=5),
            bootstrap_resamples=10,
        )
    with pytest.raises(ValueError, match="same observed columns"):
        compare_models(
            {
                "choice": candidates()["static"],
                "response": BernoulliHistoryGLM(predictors=("stimulus",), outcome="response"),
            },
            Study(
                {
                    **{name: study[name] for name in study.columns},
                    "response": study["choice"],
                }
            ),
            cohort_forward_session_splits(study, min_train_sessions=5),
            bootstrap_resamples=10,
        )
    with pytest.raises(ValueError, match="produced no folds"):
        nested_select_model(
            candidates(),
            study,
            cohort_forward_session_splits(study, min_train_sessions=5),
            lambda _study: (),
            bootstrap_resamples=10,
        )


def test_declaring_the_default_metrics_is_the_comparison_that_declares_nothing() -> None:
    """A caller who names nothing gets the table this function has always produced.

    The stronger statement -- that the *serialized* default table is byte-identical to the
    one recorded before the metric set existed -- is made by the pinned fixture in
    ``tests/test_posterior_estimator_wiring.py``, which predates this declaration entirely.
    """

    study = comparison_study()
    splits = cohort_forward_session_splits(study, min_train_sessions=5)

    implicit = compare_models(candidates(), study, splits, bootstrap_resamples=100)
    explicit = compare_models(
        candidates(),
        study,
        splits,
        bootstrap_resamples=100,
        metrics=DEFAULT_COMPARISON_METRICS,
    )

    assert implicit.metrics == (ScoreMetric.LOG_LOSS, ScoreMetric.BRIER)
    assert implicit.ranked_by is ScoreMetric.LOG_LOSS
    assert implicit.to_dict() == explicit.to_dict()
    # The declaration is recoverable from the record, and a default record does not carry
    # a key that a record written before the declaration existed could not have carried.
    assert "declared_metrics" not in implicit.to_dict()


def test_a_brier_ranked_table_says_so_everywhere_the_verdict_appears() -> None:
    """Declaration order is the ranking rule, and the record is spelled with it."""

    study = comparison_study()
    splits = cohort_forward_session_splits(study, min_train_sessions=5)

    report = compare_models(
        candidates(),
        study,
        splits,
        bootstrap_resamples=100,
        metrics=(ScoreMetric.BRIER, ScoreMetric.LOG_LOSS),
    )

    assert report.ranked_by is ScoreMetric.BRIER
    assert report.winner == min(
        report.model_order,
        key=lambda name: report.result_for(name).unit_balanced_brier_score,
    )
    assert report.pairwise_comparisons[0].metric is ScoreMetric.BRIER
    payload = report.to_dict()
    assert payload["declared_metrics"] == ["brier", "log-loss"]
    assert payload["winner_policy"] == "lowest unit-balanced brier score among non-failed audits"
    assert payload["winner_by_unit_balanced_brier_score"] == report.winner
    assert "pairwise_brier_score_differences" in payload
    static = report.result_for("static")
    assert static.unit_balanced_brier_interval.estimate == pytest.approx(
        static.unit_balanced_brier_score
    )
    # Both columns are carried; only the ranked one is bootstrapped.
    assert set(payload["models"]["static"]["unit_scores"][0]) == {
        "unit",
        "brier_score",
        "log_loss",
    }
    with pytest.raises(UndeclaredMetric, match="holds no 'log-loss' interval"):
        _ = static.unit_balanced_log_loss_interval


def test_an_unsupported_or_malformed_metric_declaration_is_refused() -> None:
    study = comparison_study()
    splits = cohort_forward_session_splits(study, min_train_sessions=5)

    with pytest.raises(ValueError, match="at least one scoring rule"):
        compare_models(candidates(), study, splits, bootstrap_resamples=10, metrics=())
    with pytest.raises(ValueError, match="same scoring rule twice"):
        compare_models(
            candidates(),
            study,
            splits,
            bootstrap_resamples=10,
            metrics=(ScoreMetric.LOG_LOSS, ScoreMetric.LOG_LOSS),
        )
    with pytest.raises(TypeError, match="not one rule"):
        compare_models(
            candidates(), study, splits, bootstrap_resamples=10, metrics=ScoreMetric.LOG_LOSS
        )
    # The log column already scores every declared column jointly, so the protocol's name
    # for that same quantity is refused rather than admitted as a second heading.
    with pytest.raises(ValueError, match="already scores the complete observation"):
        compare_models(
            candidates(),
            study,
            splits,
            bootstrap_resamples=10,
            metrics=(ScoreMetric.JOINT_LOG_LOSS,),
        )


def test_nested_selection_carries_its_declared_metric_through_every_inner_report() -> None:
    study = comparison_study()

    nested = nested_select_model(
        candidates(),
        study,
        cohort_forward_session_splits(study, min_train_sessions=5),
        lambda training: cohort_forward_session_splits(training, min_train_sessions=3),
        bootstrap_resamples=100,
        inner_bootstrap_resamples=50,
        metrics=(ScoreMetric.LOG_LOSS,),
        backend="thread",
    )

    assert nested.metrics == (ScoreMetric.LOG_LOSS,)
    assert nested.ranked_by is ScoreMetric.LOG_LOSS
    assert all(fold.inner_report.metrics == (ScoreMetric.LOG_LOSS,) for fold in nested.folds)
    assert nested.to_dict()["declared_metrics"] == ["log-loss"]
    with pytest.raises(UndeclaredMetric, match="carries no 'brier' column"):
        _ = nested.pooled_brier_score
