"""Validation-first tools for modelling behaviour across learning."""

from unspool.clocks import (
    ClockedStudy,
    ClockKind,
    ClockScope,
    ClockSpec,
    ClockValidationError,
    session_order_clock,
    with_cumulative_trial_clock,
    with_elapsed_time_clock,
)
from unspool.evaluation import FoldEvaluation, evaluate_splits
from unspool.model_recovery import (
    ModelRecoveryMatrix,
    ModelRecoveryReport,
    ModelRecoveryScenario,
    run_model_recovery,
)
from unspool.models import (
    BehaviourModel,
    BernoulliHistoryGLM,
    CoefficientTrajectory,
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    SmoothBernoulliHistoryGLM,
    UnsupportedPredictionMode,
)
from unspool.recovery import (
    ParameterRecoveryReport,
    ParameterRecoverySummary,
    run_parameter_recovery,
)
from unspool.study import REQUIRED_COLUMNS, Study, StudyValidationError
from unspool.transforms import (
    FittedStudyTransform,
    FittedThresholdLandmarkClock,
    FoldTransformResult,
    LandmarkNotFoundError,
    StudyTransform,
    ThresholdLandmarkClock,
    TransformProvenance,
    fit_transform_split,
    fit_transform_splits,
)
from unspool.validation import (
    ValidationSplit,
    forward_session_splits,
    leave_one_session_out_splits,
)

__version__ = "0.1.0"

__all__ = [
    "REQUIRED_COLUMNS",
    "BehaviourModel",
    "BernoulliHistoryGLM",
    "ClockKind",
    "ClockScope",
    "ClockSpec",
    "ClockValidationError",
    "ClockedStudy",
    "CoefficientTrajectory",
    "FitDiagnostics",
    "FitResult",
    "FittedStudyTransform",
    "FittedThresholdLandmarkClock",
    "FoldEvaluation",
    "FoldTransformResult",
    "LandmarkNotFoundError",
    "ModelDataError",
    "ModelRecoveryMatrix",
    "ModelRecoveryReport",
    "ModelRecoveryScenario",
    "ParameterRecoveryReport",
    "ParameterRecoverySummary",
    "Prediction",
    "PredictionMode",
    "SmoothBernoulliHistoryGLM",
    "Study",
    "StudyTransform",
    "StudyValidationError",
    "ThresholdLandmarkClock",
    "TransformProvenance",
    "UnsupportedPredictionMode",
    "ValidationSplit",
    "__version__",
    "evaluate_splits",
    "fit_transform_split",
    "fit_transform_splits",
    "forward_session_splits",
    "leave_one_session_out_splits",
    "run_model_recovery",
    "run_parameter_recovery",
    "session_order_clock",
    "with_cumulative_trial_clock",
    "with_elapsed_time_clock",
]
