"""Reference behavioural models and shared modelling contracts."""

from unspool.models.base import (
    BehaviourModel,
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from unspool.models.glm import (
    BernoulliHistoryGLM,
    CoefficientTrajectory,
    SmoothBernoulliHistoryGLM,
)

__all__ = [
    "BehaviourModel",
    "BernoulliHistoryGLM",
    "CoefficientTrajectory",
    "FitDiagnostics",
    "FitResult",
    "ModelDataError",
    "Prediction",
    "PredictionMode",
    "SmoothBernoulliHistoryGLM",
    "UnsupportedPredictionMode",
]
