from dataclasses import replace

import numpy as np
import pytest
from test_compiler import capabilities, frozen_small_protocol, source_study
from test_runner import compiled_small_protocol

from behavio import (
    ChoiceSpec,
    MultinomialLogit,
    Study,
    cohort_forward_session_splits,
    compare_models,
    compile_execution_plan,
    evaluate_splits,
    forward_session_splits,
    run_parameter_recovery,
    run_protocol,
)
from behavio.design import DesignSpec, NumericTerm
from behavio.models import CategoricalPrediction, ModelDataError
from behavio.protocol import (
    CandidateSpec,
    ProtocolState,
    ScoreMetric,
    Setting,
    materialize_protocol,
)


def categorical_design(*, omissions: bool = False) -> Study:
    generator = np.random.default_rng(41)
    n_sessions = 4
    n_trials = 50
    available = np.empty(n_sessions * n_trials, dtype=object)
    available[:] = [
        ("left", "right") if trial % 7 == 0 else ("left", "right", "up")
        for _session in range(n_sessions)
        for trial in range(n_trials)
    ]
    columns = {
        "subject": ["a"] * (n_sessions * n_trials),
        "session": [f"session-{session}" for session in range(n_sessions) for _ in range(n_trials)],
        "trial": list(range(n_trials)) * n_sessions,
        "session_order": [session for session in range(n_sessions) for _ in range(n_trials)],
        "stimulus": generator.normal(size=n_sessions * n_trials),
        "available": available,
    }
    if omissions:
        columns["placeholder"] = np.zeros(n_sessions * n_trials)
    return Study(columns)


def categorical_model(*, omissions: bool = False, l2: float = 0.05) -> MultinomialLogit:
    return MultinomialLogit(
        choice=ChoiceSpec(
            options=("left", "right", "up"),
            omission_values=("omit",) if omissions else (),
            available_options_column="available",
        ),
        design=DesignSpec((NumericTerm("stimulus"),)),
        include_omission=omissions,
        l2=l2,
    )


def parameters(model: MultinomialLogit) -> dict[str, float]:
    values = {
        "category['right']::intercept": 0.3,
        "category['right']::stimulus": 1.1,
        "category['up']::intercept": -0.2,
        "category['up']::stimulus": -0.7,
    }
    if model.include_omission:
        values.update(
            {
                "category['omit']::intercept": -1.6,
                "category['omit']::stimulus": 0.15,
            }
        )
    assert set(values) == set(model.parameter_names)
    return values


def test_multinomial_logit_simulates_fits_predicts_and_scores_availability() -> None:
    model = categorical_model()
    assert model.required_task_columns == ("stimulus", "available")
    study = model.simulate(categorical_design(), parameters(model), seed=42)

    fit = model.fit(study)
    prediction = model.predict(study, fit)
    scores = model.pointwise_log_prob(study, fit)

    assert isinstance(prediction, CategoricalPrediction)
    assert prediction.categories == ("left", "right", "up")
    assert prediction.probability.shape == (len(study), 3)
    np.testing.assert_allclose(np.sum(prediction.probability, axis=1), 1.0)
    unavailable = np.asarray([len(options) == 2 for options in study["available"]])
    np.testing.assert_array_equal(prediction.probability[unavailable, 2], 0.0)
    assert scores.shape == (len(study),)
    assert np.all(np.isfinite(scores))
    assert fit.parameter_names == model.parameter_names
    assert fit.audit().status.value in {"pass", "warning"}


def test_omissions_are_a_modeled_category_not_silently_dropped() -> None:
    model = categorical_model(omissions=True)
    study = model.simulate(categorical_design(omissions=True), parameters(model), seed=7)
    fit = model.fit(study)
    prediction = model.predict(study, fit)
    codes = model.outcome_codes(study)

    assert prediction.categories == ("left", "right", "up", "omit")
    assert np.sum(codes == 3) > 0
    assert fit.n_observations == len(study)
    assert prediction.probability.shape == (len(study), 4)

    ordinary = replace(model, include_omission=False, omission_label=None)
    with pytest.raises(ModelDataError, match="include_omission"):
        ordinary.fit(study)


def test_categorical_predictions_participate_in_evaluation_comparison_and_recovery() -> None:
    model = categorical_model()
    design = categorical_design()
    study = model.simulate(design, parameters(model), seed=11)
    splits = forward_session_splits(study, min_train_sessions=2)

    evaluation = evaluate_splits(model, study, splits)[0]
    assert isinstance(evaluation.prediction, CategoricalPrediction)
    assert evaluation.outcome_codes is not None
    assert len(evaluation.outcome_codes) == len(evaluation.split.test_indices)

    report = compare_models(
        {"weak": replace(model, l2=1.0), "reference": model},
        study,
        splits,
        aggregation_column="session",
        outcome_column="choice",
        bootstrap_resamples=50,
    )
    assert all(0 <= result.pooled_brier_score <= 1 for result in report.model_results)

    recovery = run_parameter_recovery(
        model,
        design,
        parameter_sets=(parameters(model),),
        repeats=2,
        seed=14,
    )
    assert recovery.n_runs == 2


def multinomial_candidates() -> tuple[CandidateSpec, ...]:
    """Declare the categorical candidates the protocol runner is actually handed."""

    return tuple(
        CandidateSpec(
            name=name,
            implementation="behavio.models.MultinomialLogit",
            hyperparameters=(Setting("l2", l2),),
            scored_columns=("choice",),
        )
        for name, l2 in (("static", 0.1), ("smooth", 1.0))
    )


def test_protocol_runner_serializes_categorical_rows_and_scores_brier() -> None:
    compiled = compiled_small_protocol(multinomial_candidates())
    model = MultinomialLogit(
        choice=ChoiceSpec(options=(0, 1)),
        design=DesignSpec((NumericTerm("stimulus"),)),
        l2=0.1,
    )
    run = run_protocol(
        compiled,
        {"static": model, "smooth": replace(model, l2=1.0)},
    )

    point = run.report.candidates[0].predictions[0]
    assert point.is_categorical
    assert point.categories == (0, 1)
    assert point.observed_category_index in (0, 1)
    assert point.probability is None
    assert run.report.candidates[0].calibration.reason is not None
    assert "categorical" in run.report.candidates[0].calibration.reason
    assert "category_probabilities" in run.report.canonical_json()

    protocol = frozen_small_protocol(multinomial_candidates())
    protocol = replace(
        protocol,
        comparison=replace(protocol.comparison, metric=ScoreMetric.BRIER),
        state=ProtocolState.DRAFT,
        lifecycle=(),
    ).freeze()
    materialized = materialize_protocol(protocol, source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    brier_compiled = compile_execution_plan(
        materialized,
        splits,
        capabilities=capabilities(),
    )
    brier_run = run_protocol(
        brier_compiled,
        {"static": model, "smooth": replace(model, l2=1.0)},
    )
    assert all(
        candidate.score is not None and candidate.score.metric == ScoreMetric.BRIER
        for candidate in brier_run.report.candidates
    )
