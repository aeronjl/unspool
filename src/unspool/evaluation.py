"""Fold-aware evaluation for behavioural models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from unspool.models.base import (
    BehaviourModel,
    FitResult,
    Prediction,
    PredictionMode,
    _protected_array,
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
    model: BehaviourModel,
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
        prediction_rows = np.concatenate((split.prediction_context_indices, split.test_indices))
        prediction_study = study.take(prediction_rows)
        full_prediction = model.predict(prediction_study, fit, mode=mode)
        full_scores = model.pointwise_log_prob(prediction_study, fit, mode=mode)
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
