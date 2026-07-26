"""Validation-first tools for modelling behaviour across learning."""

from unspool.study import REQUIRED_COLUMNS, Study, StudyValidationError
from unspool.validation import (
    ValidationSplit,
    forward_session_splits,
    leave_one_session_out_splits,
)

__version__ = "0.1.0"

__all__ = [
    "REQUIRED_COLUMNS",
    "Study",
    "StudyValidationError",
    "ValidationSplit",
    "__version__",
    "forward_session_splits",
    "leave_one_session_out_splits",
]
