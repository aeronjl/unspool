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
from unspool.models.glm_hmm import (
    BernoulliGLMHMM,
    FilteredStateProbabilities,
    GLMHMMFitResult,
    GLMHMMParameters,
    GLMHMMSimulation,
)

__all__ = [
    "BehaviourModel",
    "BernoulliGLMHMM",
    "BernoulliHistoryGLM",
    "CoefficientTrajectory",
    "FilteredStateProbabilities",
    "FitDiagnostics",
    "FitResult",
    "GLMHMMFitResult",
    "GLMHMMParameters",
    "GLMHMMSimulation",
    "ModelDataError",
    "Prediction",
    "PredictionMode",
    "SmoothBernoulliHistoryGLM",
    "UnsupportedPredictionMode",
]
