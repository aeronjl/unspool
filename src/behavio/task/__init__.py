"""The behavioural task contract: what a trial means before any model is fitted.

:mod:`behavio.task.spec` declares the observed columns of a study -- the choice and its
alternatives, the reward, the response time -- and :mod:`behavio.task.response_times`
declares the units and origin those times are measured in.

The package sits below :mod:`behavio.models`: it is typed against the estimator contract in
:mod:`behavio.contracts.estimator`, not against any concrete model, so a model may require
task columns without the task layer knowing the catalogue exists.
"""

from behavio.task.response_times import ResponseTimes, ResponseTimeSpec, ResponseTimeUnit
from behavio.task.spec import (
    ChoiceData,
    ChoiceSpec,
    FittedModel,
    RewardSpec,
    TaskSpec,
    TaskValidation,
    TaskValidationError,
    fit_model,
)

__all__ = [
    "ChoiceData",
    "ChoiceSpec",
    "FittedModel",
    "ResponseTimeSpec",
    "ResponseTimeUnit",
    "ResponseTimes",
    "RewardSpec",
    "TaskSpec",
    "TaskValidation",
    "TaskValidationError",
    "fit_model",
]
