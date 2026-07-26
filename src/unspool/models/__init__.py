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
from unspool.models.hierarchical_glm import (
    HierarchicalBernoulliHistoryGLM,
    HierarchicalGLMFitResult,
    HierarchicalGLMSimulation,
)
from unspool.models.q_learning import (
    BinaryQLearning,
    QLearningFitResult,
    QLearningParameters,
    ValueTrajectory,
)

__all__ = [
    "BehaviourModel",
    "BernoulliGLMHMM",
    "BernoulliHistoryGLM",
    "BinaryQLearning",
    "CoefficientTrajectory",
    "FilteredStateProbabilities",
    "FitDiagnostics",
    "FitResult",
    "GLMHMMFitResult",
    "GLMHMMParameters",
    "GLMHMMSimulation",
    "HierarchicalBernoulliHistoryGLM",
    "HierarchicalGLMFitResult",
    "HierarchicalGLMSimulation",
    "ModelDataError",
    "Prediction",
    "PredictionMode",
    "QLearningFitResult",
    "QLearningParameters",
    "SmoothBernoulliHistoryGLM",
    "UnsupportedPredictionMode",
    "ValueTrajectory",
]
