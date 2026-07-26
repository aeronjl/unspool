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
from unspool.models.ddm import (
    DriftDiffusionFitResult,
    DriftDiffusionParameters,
    DriftDiffusionSimulation,
    UniformResponseTimeContaminant,
    WienerDriftDiffusion,
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
from unspool.models.hierarchical_smooth_ddm import (
    HierarchicalSmoothDriftDiffusionFitResult,
    HierarchicalSmoothDriftDiffusionSimulation,
    HierarchicalSmoothWienerDriftDiffusion,
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
from unspool.models.smooth_ddm import (
    DriftDiffusionTrajectory,
    SmoothDriftDiffusionFitResult,
    SmoothWienerDriftDiffusion,
)

__all__ = [
    "BehaviourEstimator",
    "BehaviourModel",
    "BernoulliGLMHMM",
    "BernoulliHistoryGLM",
    "BinaryQLearning",
    "CoefficientTrajectory",
    "DriftDiffusionFitResult",
    "DriftDiffusionParameters",
    "DriftDiffusionSimulation",
    "DriftDiffusionTrajectory",
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
    "HierarchicalSmoothDriftDiffusionFitResult",
    "HierarchicalSmoothDriftDiffusionSimulation",
    "HierarchicalSmoothGLMFitResult",
    "HierarchicalSmoothGLMSimulation",
    "HierarchicalSmoothWienerDriftDiffusion",
    "ModelCapabilities",
    "ModelDataError",
    "Prediction",
    "PredictionMode",
    "QLearningFitResult",
    "QLearningParameters",
    "SmoothBernoulliHistoryGLM",
    "SmoothDriftDiffusionFitResult",
    "SmoothWienerDriftDiffusion",
    "UniformResponseTimeContaminant",
    "UnsupportedPredictionMode",
    "ValueTrajectory",
    "WienerDriftDiffusion",
    "model_capabilities",
]
