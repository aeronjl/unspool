"""Design-specific recovery evidence: can this design recover what it claims to measure?

Two questions, two modules, and the path now says which is which.
:mod:`behavio.recovery.parameters` simulates from known parameters and asks whether the
estimator gets them back, with coverage reported against the interval kind that produced
it. :mod:`behavio.recovery.models` simulates from one family and asks whether the selection
rule picks it out of a field of competitors.

A third recovery lives in :mod:`behavio.protocol.exact_recovery`: the same questions run
through a frozen compiled protocol rather than through an ad-hoc call.
"""

from behavio.recovery.models import (
    ModelRecoveryGridReport,
    ModelRecoveryGridSummary,
    ModelRecoveryMatrix,
    ModelRecoveryReport,
    ModelRecoveryScenario,
    ModelRecoveryScenarioMatrix,
    run_model_recovery,
    run_model_recovery_grid,
)
from behavio.recovery.parameters import (
    POSTERIOR_QUANTILE_INTERVAL,
    WALD_INTERVAL,
    ParameterRecoveryReport,
    ParameterRecoverySummary,
    run_parameter_recovery,
)

__all__ = [
    "POSTERIOR_QUANTILE_INTERVAL",
    "WALD_INTERVAL",
    "ModelRecoveryGridReport",
    "ModelRecoveryGridSummary",
    "ModelRecoveryMatrix",
    "ModelRecoveryReport",
    "ModelRecoveryScenario",
    "ModelRecoveryScenarioMatrix",
    "ParameterRecoveryReport",
    "ParameterRecoverySummary",
    "run_model_recovery",
    "run_model_recovery_grid",
    "run_parameter_recovery",
]
