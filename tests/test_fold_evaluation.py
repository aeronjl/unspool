import numpy as np
import pytest

from behavio import BernoulliHistoryGLM, Study, evaluate_splits, forward_session_splits
from behavio.contracts.fold import EvaluationFold
from behavio.evaluate import (
    leave_one_lab_out_session_forecast_splits,
    leave_one_lab_out_splits,
    leave_one_session_out_splits,
    leave_one_subject_out_splits,
    within_session_rolling_splits,
)
from behavio.evaluate.folds import FoldFailurePolicy, FoldStage


def simulated_study() -> tuple[BernoulliHistoryGLM, Study]:
    generator = np.random.default_rng(11)
    n_sessions = 4
    n_trials = 80
    design = Study(
        {
            "subject": ["a"] * (n_sessions * n_trials),
            "session": [
                f"session-{session}" for session in range(n_sessions) for _ in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions,
            "session_order": [session for session in range(n_sessions) for _ in range(n_trials)],
            "stimulus": generator.normal(size=n_sessions * n_trials),
        }
    )
    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.1)
    study = model.simulate(
        design,
        {"intercept": -0.1, "stimulus": 1.0, "choice_lag_1": 0.4},
        seed=22,
    )
    return model, study


def test_evaluate_splits_fits_and_scores_each_prospective_origin() -> None:
    model, study = simulated_study()
    splits = forward_session_splits(study, min_train_sessions=2)

    evaluations = evaluate_splits(model, study, splits)

    assert len(evaluations) == 2
    assert evaluations[0].split.prospective
    assert evaluations[0].fit.n_observations == 160
    assert evaluations[0].prediction.probability.shape == (80,)
    assert evaluations[0].pointwise_log_probability.shape == (80,)
    assert np.isfinite(evaluations[0].mean_log_probability)
    assert evaluations[0].mean_log_loss == -evaluations[0].mean_log_probability


def test_nonprospective_evaluation_requires_explicit_acknowledgement() -> None:
    model, study = simulated_study()
    splits = leave_one_session_out_splits(study)

    with pytest.raises(ValueError, match="not prospective"):
        evaluate_splits(model, study, splits)

    evaluations = evaluate_splits(model, study, splits, require_prospective=False)
    assert len(evaluations) == 4
    assert all(not evaluation.split.prospective for evaluation in evaluations)


def test_within_session_evaluation_preserves_filtered_pre_origin_history() -> None:
    model, study = simulated_study()
    split = within_session_rolling_splits(
        study,
        min_train_sessions=2,
        min_train_trials=10,
        horizon=2,
        step=100,
    )[0]

    evaluation = evaluate_splits(model, study, [split])[0]

    assert evaluation.fit.n_observations == 170
    assert evaluation.prediction.probability.shape == (2,)
    session_rows = np.flatnonzero(study["session_order"] == split.origin_session_order)
    full_session = study.take(session_rows)
    full_prediction = model.predict(full_session, evaluation.fit)
    full_scores = model.pointwise_log_prob(full_session, evaluation.fit)
    np.testing.assert_allclose(
        evaluation.prediction.probability, full_prediction.probability[10:12]
    )
    np.testing.assert_allclose(evaluation.pointwise_log_probability, full_scores[10:12])

    reset_prediction = model.predict(study.take(split.test_indices), evaluation.fit)
    assert not np.isclose(
        evaluation.prediction.probability[0],
        reset_prediction.probability[0],
    )


def test_population_holdouts_fit_only_on_unseen_subjects_and_labs() -> None:
    generator = np.random.default_rng(31)
    subjects = ("a", "b", "c")
    labs = {"a": "north", "b": "north", "c": "south"}
    n_sessions = 2
    n_trials = 20
    design = Study(
        {
            "subject": [
                subject
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "session": [
                f"{subject}-{session}"
                for subject in subjects
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions * len(subjects),
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "lab": [
                labs[subject]
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "stimulus": generator.normal(size=len(subjects) * n_sessions * n_trials),
        }
    )
    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.1)
    study = model.simulate(
        design,
        {"intercept": -0.1, "stimulus": 1.0, "choice_lag_1": 0.4},
        seed=32,
    )

    subject_evaluation = evaluate_splits(model, study, leave_one_subject_out_splits(study))[0]
    assert subject_evaluation.fit.n_observations == 80
    assert subject_evaluation.prediction.probability.shape == (40,)
    assert subject_evaluation.split.test_subjects == ("a",)

    lab_evaluation = evaluate_splits(model, study, leave_one_lab_out_splits(study))[0]
    assert lab_evaluation.fit.n_observations == 40
    assert lab_evaluation.prediction.probability.shape == (80,)
    assert lab_evaluation.split.test_groups == ("north",)


def test_lab_session_forecasts_protect_both_population_and_future_boundaries() -> None:
    generator = np.random.default_rng(41)
    subjects = ("a", "b", "c", "d")
    labs = {"a": "north", "b": "north", "c": "south", "d": "south"}
    n_sessions = 6
    n_trials = 10
    design = Study(
        {
            "subject": [
                subject
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "session": [
                f"{subject}-{session}"
                for subject in subjects
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "trial": list(range(n_trials)) * n_sessions * len(subjects),
            "session_order": [
                session
                for _subject in subjects
                for session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "lab": [
                labs[subject]
                for subject in subjects
                for _session in range(n_sessions)
                for _trial in range(n_trials)
            ],
            "stimulus": generator.normal(size=len(subjects) * n_sessions * n_trials),
        }
    )
    model = BernoulliHistoryGLM(predictors=("stimulus",), choice_lags=1, l2=0.1)
    study = model.simulate(
        design,
        {"intercept": -0.1, "stimulus": 1.0, "choice_lag_1": 0.4},
        seed=42,
    )

    splits = leave_one_lab_out_session_forecast_splits(
        study,
        train_session_count=5,
    )

    assert len(splits) == 2
    assert all(split.prospective for split in splits)
    assert all(split.train_session_orders == tuple(range(5)) for split in splits)
    assert all(split.test_session_orders == (5,) for split in splits)
    assert all(set(split.train_subjects).isdisjoint(split.test_subjects) for split in splits)
    assert all(set(split.train_groups).isdisjoint(split.test_groups) for split in splits)
    assert sum(len(split.test_indices) for split in splits) == len(subjects) * n_trials
    evaluation = evaluate_splits(model, study, (splits[0],))[0]
    assert evaluation.fit.n_observations == 2 * 5 * n_trials
    assert evaluation.prediction.probability.shape == (2 * n_trials,)


def test_lab_session_forecast_rejects_unaligned_subject_clocks() -> None:
    unaligned = Study(
        {
            "subject": ["a"] * 3 + ["b"] * 3,
            "session": ["a-0", "a-1", "a-2", "b-0", "b-1", "b-2"],
            "trial": [0] * 6,
            "session_order": [0, 1, 2, 0, 1, 3],
            "lab": ["north"] * 3 + ["south"] * 3,
        }
    )

    with pytest.raises(ValueError, match="common aligned"):
        leave_one_lab_out_session_forecast_splits(
            unaligned,
            train_session_count=2,
            horizon=1,
        )


def test_a_retained_fold_failure_is_named_by_the_splitter_not_by_position() -> None:
    """The declared fold name is what a scientist reads back off a failure."""

    _model, study = simulated_study()
    splits = forward_session_splits(study, min_train_sessions=1)

    class RefusingModel(BernoulliHistoryGLM):
        def fit(self, study):  # type: ignore[override]
            raise RuntimeError("deliberate failure")

    result = evaluate_splits(
        RefusingModel(predictors=("stimulus",), choice_lags=1, l2=0.1),
        study,
        splits,
        on_failure=FoldFailurePolicy.RETAIN,
    )

    assert result.evaluations == ()
    assert [failure.fold for failure in result.failures] == [split.identifier for split in splits]
    assert [failure.fold for failure in result.failures] == [
        "forward-session/subject=a/forecast-sessions=1",
        "forward-session/subject=a/forecast-sessions=2",
        "forward-session/subject=a/forecast-sessions=3",
    ]
    assert all(failure.stage is FoldStage.FIT for failure in result.failures)
    # Every failure record is portable, so the name survives serialization too.
    assert [failure.to_dict()["fold"] for failure in result.failures] == [
        split.identifier for split in splits
    ]


def test_a_completed_fold_carries_the_splitter_name_it_was_scored_under() -> None:
    model, study = simulated_study()
    splits = forward_session_splits(study, min_train_sessions=2)

    evaluations = evaluate_splits(model, study, splits)

    assert [evaluation.identifier for evaluation in evaluations] == [
        split.identifier for split in splits
    ]


def test_a_split_that_does_not_name_itself_fails_the_fold_contract() -> None:
    """The old ``getattr`` fallback numbered such a split; the contract now refuses it."""

    model, study = simulated_study()
    split = forward_session_splits(study, min_train_sessions=2)[0]

    class UnnamedSplit:
        train_indices = split.train_indices
        test_indices = split.test_indices
        prediction_context_indices = split.prediction_context_indices
        scheme = "anonymous"
        prospective = True

    assert not isinstance(UnnamedSplit(), EvaluationFold)
    with pytest.raises(TypeError, match="declares no identifier"):
        evaluate_splits(model, study, (UnnamedSplit(),))


def test_two_folds_may_not_share_one_name() -> None:
    """A shared name would silently drop one fold from every record keyed by it."""

    model, study = simulated_study()
    split = forward_session_splits(study, min_train_sessions=2)[0]

    with pytest.raises(ValueError, match="share the identifier"):
        evaluate_splits(model, study, (split, split))
