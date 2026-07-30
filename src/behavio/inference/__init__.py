"""Where estimates come from: the parameter space, and the optimizer that searches it.

:mod:`behavio.inference.parameters` declares a model's parameters -- their roles, bounds,
transforms and priors -- as a portable, serialisable object rather than as arguments buried
in an optimizer call. :mod:`behavio.inference.optimize` is the deterministic multistart
machinery that searches that space, and the backend contract a third-party optimizer
implements.

Both sit below :mod:`behavio.models`, which is why a model may own a parameter space without
this package knowing which models exist.
"""

from behavio.inference.optimize import (
    InferenceError,
    ObjectiveFunction,
    ObjectiveOutput,
    ObjectiveTarget,
    OptimizationAttempt,
    OptimizationBackend,
    OptimizationProblem,
    OptimizationRun,
    PriorMeasure,
    PyBADSMultistart,
    PyBADSUnavailableError,
    ScipyMultistart,
)
from behavio.inference.parameters import (
    PARAMETER_SPACE_SCHEMA,
    ParameterRole,
    ParameterSpace,
    ParameterSpaceError,
    ParameterSpaceProvider,
    ParameterSpec,
    ParameterTransform,
    PriorFamily,
    PriorSpec,
    parameter_space_from_dict,
    parameter_space_from_json,
)

__all__ = [
    "PARAMETER_SPACE_SCHEMA",
    "InferenceError",
    "ObjectiveFunction",
    "ObjectiveOutput",
    "ObjectiveTarget",
    "OptimizationAttempt",
    "OptimizationBackend",
    "OptimizationProblem",
    "OptimizationRun",
    "ParameterRole",
    "ParameterSpace",
    "ParameterSpaceError",
    "ParameterSpaceProvider",
    "ParameterSpec",
    "ParameterTransform",
    "PriorFamily",
    "PriorMeasure",
    "PriorSpec",
    "PyBADSMultistart",
    "PyBADSUnavailableError",
    "ScipyMultistart",
    "parameter_space_from_dict",
    "parameter_space_from_json",
]
