"""Fold-aware evaluation for behavioural models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from unspool.models.base import (
    BehaviourEstimator,
    FitResult,
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
    prediction: Prediction
    pointwise_log_probability: NDArray[np.float64]

    def __post_init__(self) -> None:
        scores = _protected_array(self.pointwise_log_probability, dtype=np.float64)
        if scores.ndim != 1 or scores.shape != self.prediction.probability.shape:
            raise ValueError("pointwise scores must match the number of predictions")
        if not np.all(np.isfinite(scores)):
            raise ValueError("pointwise scores must be finite")
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
        if not isinstance(full_prediction, Prediction):
            raise TypeError("model.predict must return a Prediction")
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
        prediction = Prediction(
            probability=full_prediction.probability[target],
            linear_predictor=full_prediction.linear_predictor[target],
            mode=full_prediction.mode,
        )
        scores = full_scores[target]
        evaluations.append(
            FoldEvaluation(
                split=split,
                fit=fit,
                prediction=prediction,
                pointwise_log_probability=scores,
            )
        )
    return tuple(evaluations)


def _validate_positions(indices: NDArray[np.intp], length: int, name: str) -> None:
    if np.any(indices >= length):
        raise IndexError(f"{name} contains a row position outside the study")
