"""Reference behavioural models and shared modelling contracts."""

from unspool.models.base import (
    BehaviourEstimator,
    BehaviourModel,
    FitDiagnostics,
    FitResult,
    GenerativeBehaviourModel,
    ModelCapabilities,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    model_capabilities,
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
from unspool.models.hierarchical_smooth_glm import (
    HierarchicalSmoothBernoulliHistoryGLM,
    HierarchicalSmoothGLMFitResult,
    HierarchicalSmoothGLMSimulation,
)
from unspool.models.q_learning import (
    BinaryQLearning,
    QLearningFitResult,
    QLearningParameters,
    ValueTrajectory,
)

__all__ = [
    "BehaviourEstimator",
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
    "GenerativeBehaviourModel",
    "HierarchicalBernoulliHistoryGLM",
    "HierarchicalGLMFitResult",
    "HierarchicalGLMSimulation",
    "HierarchicalSmoothBernoulliHistoryGLM",
    "HierarchicalSmoothGLMFitResult",
    "HierarchicalSmoothGLMSimulation",
    "ModelCapabilities",
    "ModelDataError",
    "Prediction",
    "PredictionMode",
    "QLearningFitResult",
    "QLearningParameters",
    "SmoothBernoulliHistoryGLM",
    "UnsupportedPredictionMode",
    "ValueTrajectory",
    "model_capabilities",
]
