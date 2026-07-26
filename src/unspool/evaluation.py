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
from unspool.validation import ValidationSplit


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Fit, prediction, and pointwise score for one validation fold."""

    split: ValidationSplit
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
    splits: Iterable[ValidationSplit],
    *,
    mode: PredictionMode = PredictionMode.FILTERED,
    require_prospective: bool = True,
) -> tuple[FoldEvaluation, ...]:
    """Fit and score a model independently within each supplied fold.

    Prospective folds are required by default. Passing a non-prospective splitter therefore
    needs an explicit ``require_prospective=False`` acknowledgement.
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
        training = study.take(split.train_indices)
        testing = study.take(split.test_indices)
        fit = model.fit(training)
        prediction = model.predict(testing, fit, mode=mode)
        scores = model.pointwise_log_prob(testing, fit, mode=mode)
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
