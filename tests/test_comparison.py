from __future__ import annotations

import json

import numpy as np
import pytest

from unspool import (
    BernoulliHistoryGLM,
    SmoothBernoulliHistoryGLM,
    Study,
    cohort_forward_session_splits,
    compare_models,
    leave_one_lab_out_session_forecast_splits,
    leave_one_session_out_splits,
    nested_select_model,
)

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
    generator_model = SmoothBernoulliHistoryGLM(
        covariates=("stimulus",),
        choice_lags=1,
        knots=KNOTS,
        smoothness=3.0,
        l2=0.02,
        shared_trajectory=True,
    )
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
        "static": BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.02),
        "smooth": SmoothBernoulliHistoryGLM(
            covariates=("stimulus",),
            choice_lags=1,
            knots=KNOTS,
            smoothness=3.0,
            l2=0.02,
            shared_trajectory=True,
        ),
    }


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
            covariates=("stimulus",),
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
            {"other_outcome": BernoulliHistoryGLM(covariates=("stimulus",), outcome="response")},
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
                "response": BernoulliHistoryGLM(covariates=("stimulus",), outcome="response"),
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
