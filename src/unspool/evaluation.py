"""Fold-aware evaluation for behavioural models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from unspool.models.base import (
    BehaviourEstimator,
    CategoricalBehaviourEstimator,
    CategoricalPrediction,
    FitResult,
    ModelPrediction,
    Prediction,
    PredictionMode,
    _protected_array,
    model_capabilities,
)
from unspool.study import Study
from unspool.validation import ValidationFold


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Fit, prediction, and pointwise score for one validation fold."""

    split: ValidationFold
    fit: FitResult
    prediction: ModelPrediction
    pointwise_log_probability: NDArray[np.float64]
    outcome_codes: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        scores = _protected_array(self.pointwise_log_probability, dtype=np.float64)
        if scores.ndim != 1 or scores.shape != (self.prediction.n_observations,):
            raise ValueError("pointwise scores must match the number of predictions")
        if not np.all(np.isfinite(scores)):
            raise ValueError("pointwise scores must be finite")
        codes = self.outcome_codes
        if isinstance(self.prediction, CategoricalPrediction):
            if codes is None:
                raise ValueError("categorical predictions require observed outcome codes")
            protected_codes = _protected_array(codes, dtype=np.int64)
            if protected_codes.shape != scores.shape or np.any(
                (protected_codes < 0) | (protected_codes >= len(self.prediction.categories))
            ):
                raise ValueError("outcome codes must identify one predicted category per row")
            object.__setattr__(self, "outcome_codes", protected_codes)
        elif codes is not None:
            raise ValueError("binary predictions must not attach categorical outcome codes")
        object.__setattr__(self, "pointwise_log_probability", scores)

    @property
    def mean_log_probability(self) -> float:
        return float(np.mean(self.pointwise_log_probability))

    @property
    def mean_log_loss(self) -> float:
        return -self.mean_log_probability

    @property
    def total_log_probability(self) -> float:
        return float(np.sum(self.pointwise_log_probability))


def evaluate_splits(
    model: BehaviourEstimator,
    study: Study,
    splits: Iterable[ValidationFold],
    *,
    mode: PredictionMode = PredictionMode.FILTERED,
    require_prospective: bool = True,
) -> tuple[FoldEvaluation, ...]:
    """Fit and score a model independently within each supplied fold.

    Prospective folds are required by default. Passing a non-prospective splitter therefore
    needs an explicit ``require_prospective=False`` acknowledgement. Prediction-context
    rows initialize filtered history but are removed from returned predictions and scores.
    """

    capabilities = model_capabilities(model)
    prediction_mode = PredictionMode(mode)
    if prediction_mode not in capabilities.prediction_modes:
        raise ValueError(
            f"model {model.model_name!r} does not support {prediction_mode.value!r} predictions"
        )
    missing = set(capabilities.scored_columns) - set(study.columns)
    if missing:
        raise ValueError(f"study is missing scored model columns: {sorted(missing)}")

    evaluations: list[FoldEvaluation] = []
    for split in splits:
        if require_prospective and not split.prospective:
            raise ValueError(
                f"split scheme {split.scheme!r} is not prospective; "
                "set require_prospective=False only for an intentional interpolation analysis"
            )
        _validate_positions(split.train_indices, len(study), "train_indices")
        _validate_positions(split.test_indices, len(study), "test_indices")
        _validate_positions(
            split.prediction_context_indices,
            len(study),
            "prediction_context_indices",
        )
        training = study.take(split.train_indices)
        fit = model.fit(training)
        if not isinstance(fit, FitResult):
            raise TypeError("model.fit must return a FitResult")
        if fit.model_name != model.model_name or fit.model_signature != model.signature:
            raise ValueError("fit result does not match the fitted estimator")
        if fit.n_observations != len(training):
            raise ValueError("fit result n_observations must equal the training-study length")
        prediction_rows = np.concatenate((split.prediction_context_indices, split.test_indices))
        prediction_study = study.take(prediction_rows)
        full_prediction = model.predict(prediction_study, fit, mode=prediction_mode)
        if not isinstance(full_prediction, (Prediction, CategoricalPrediction)):
            raise TypeError("model.predict must return Prediction or CategoricalPrediction")
        if full_prediction.n_observations != len(prediction_study):
            raise ValueError("model.predict must return one prediction per row")
        full_codes: NDArray[np.int64] | None = None
        if isinstance(full_prediction, CategoricalPrediction):
            if not isinstance(model, CategoricalBehaviourEstimator):
                raise TypeError(
                    "categorical predictions require categories and outcome_codes() on the model"
                )
            if tuple(model.categories) != full_prediction.categories:
                raise ValueError("model and prediction category coordinates differ")
            full_codes = np.asarray(model.outcome_codes(prediction_study), dtype=np.int64)
            if full_codes.shape != (len(prediction_study),):
                raise ValueError("outcome_codes must return one code per prediction row")
        full_scores = np.asarray(
            model.pointwise_log_prob(prediction_study, fit, mode=prediction_mode),
            dtype=np.float64,
        )
        if full_scores.shape != (len(prediction_study),):
            raise ValueError("pointwise_log_prob must return one score per prediction row")
        target = np.arange(
            len(split.prediction_context_indices),
            len(prediction_rows),
            dtype=np.intp,
        )
        prediction = full_prediction.take(target)
        scores = full_scores[target]
        evaluations.append(
            FoldEvaluation(
                split=split,
                fit=fit,
                prediction=prediction,
                pointwise_log_probability=scores,
                outcome_codes=None if full_codes is None else full_codes[target],
            )
        )
    return tuple(evaluations)


def _validate_positions(indices: NDArray[np.intp], length: int, name: str) -> None:
    if np.any(indices >= length):
        raise IndexError(f"{name} contains a row position outside the study")
