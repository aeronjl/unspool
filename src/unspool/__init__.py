"""Validation-first tools for modelling behaviour across learning."""

from unspool.evaluation import FoldEvaluation, evaluate_splits
from unspool.models import (
    BehaviourModel,
    BernoulliHistoryGLM,
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from unspool.recovery import (
    ParameterRecoveryReport,
    ParameterRecoverySummary,
    run_parameter_recovery,
)
from unspool.study import REQUIRED_COLUMNS, Study, StudyValidationError
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
    "FitDiagnostics",
    "FitResult",
    "FoldEvaluation",
    "ModelDataError",
    "ParameterRecoveryReport",
    "ParameterRecoverySummary",
    "Prediction",
    "PredictionMode",
    "Study",
    "StudyValidationError",
    "UnsupportedPredictionMode",
    "ValidationSplit",
    "__version__",
    "evaluate_splits",
    "forward_session_splits",
    "leave_one_session_out_splits",
    "run_parameter_recovery",
]
