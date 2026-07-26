import numpy as np
import pytest

from unspool import (
    BernoulliHistoryGLM,
    Study,
    evaluate_splits,
    forward_session_splits,
    leave_one_session_out_splits,
    within_session_rolling_splits,
)


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
    model = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=0.1)
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
