"""Combinators that build models out of models.

Behavio used to have one dataclass per cell of a grid: a GLM, a smooth GLM, a hierarchical
GLM, a hierarchical smooth GLM, and the same again for drift diffusion, each
re-implementing ``simulate``, ``fit``, ``predict`` and ``pointwise_log_prob``. Eleven of
the twenty-four cells existed and cost four thousand lines; the two axes did not compose,
and the same word meant two different things in two families.

This package is the replacement. :func:`smooth` and :func:`hierarchical` take a model and
return a model, so the cells are expressions rather than classes::

    from behavio import BernoulliHistoryGLM
    from behavio.compose import hierarchical, smooth

    base = BernoulliHistoryGLM(covariates=("stimulus",))
    drifting = smooth(base, over="session_order", knots=(0.0, 4.0, 8.0))
    pooled = hierarchical(base, over="subject", parameters=("intercept",))
    both = hierarchical(drifting, over="subject", scale=0.5)

Everything they return satisfies the ordinary estimator contract, so ``evaluate_splits``,
``compare_models``, ``run_parameter_recovery``, ``fit_model`` and ``describe()`` work on the
result without knowing it was composed.

Order matters, and only one order is accepted: **hierarchy is the outer combinator.**
``hierarchical(smooth(model))`` is a smooth model whose paths vary by group;
``smooth(hierarchical(model))`` raises, because a hierarchical estimator reports the
population coordinate while fitting a joint one whose width depends on the number of groups
in the study, and nothing outside it can expand a coordinate of unknown width.

What a model must expose to be composable is
:class:`behavio.contracts.compose.PenalisedLinearEstimator`.
"""

from behavio.compose.formula import model_from_formula
from behavio.compose.hierarchy import (
    HierarchicalFitResult,
    HierarchicalModel,
    HierarchicalSimulation,
    UnseenGroupPrediction,
    hierarchical,
)
from behavio.compose.smoothness import SmoothModel, smooth
from behavio.compose.trajectory import CoefficientTrajectory

__all__ = [
    "CoefficientTrajectory",
    "HierarchicalFitResult",
    "HierarchicalModel",
    "HierarchicalSimulation",
    "SmoothModel",
    "UnseenGroupPrediction",
    "hierarchical",
    "model_from_formula",
    "smooth",
]
