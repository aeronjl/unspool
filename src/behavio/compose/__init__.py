"""Combinators that build models out of models.

Behavio used to have one dataclass per cell of a grid: a GLM, a smooth GLM, a hierarchical
GLM, a hierarchical smooth GLM, and the same again for drift diffusion, each
re-implementing ``simulate``, ``fit``, ``predict`` and ``pointwise_log_prob``. Eleven of
the twenty-four cells existed and cost four thousand lines; the two axes did not compose,
and the same word meant two different things in two families.

This package is the replacement. :func:`mix`, :func:`smooth` and :func:`hierarchical` take a
model and return a model, so the cells are expressions rather than classes::

    from behavio import BernoulliHistoryGLM
    from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth

    base = BernoulliHistoryGLM(predictors=("stimulus",))
    drifting = smooth(base, over="session_order", knots=(0.0, 4.0, 8.0))
    pooled = hierarchical(base, over="subject", parameters=("intercept",))
    lapsing = mix(base, UniformChoiceGuess())
    everything = hierarchical(smooth(lapsing, over="session_order", knots=(0.0, 4.0)))

Everything they return satisfies the ordinary estimator contract, so ``evaluate_splits``,
``compare_models``, ``run_parameter_recovery``, ``fit_model`` and ``describe()`` work on the
result without knowing it was composed.

Order matters, and exactly one order is accepted: **hierarchy outermost, mixture
innermost.** ``hierarchical(smooth(mix(model)))`` is the full stack, and every prefix of it
is a model.

``smooth(hierarchical(model))`` raises because a hierarchical estimator reports the
population coordinate while fitting a joint one whose width depends on the number of groups
in the study, and nothing outside it can expand a coordinate of unknown width. That
restriction is arithmetic. ``mix(smooth(model))`` raises for a different reason: it is
already expressible as ``smooth(mix(model), parameters=...)`` with the weight left
stationary, and a second spelling of a model that already has one is what this package
exists to remove. Each combinator wraps only things more primitive than itself, which is
what keeps the pass-through surface linear in the number of combinators rather than
quadratic.

What a model must expose to be composable is
:class:`behavio.contracts.compose.PenalisedLinearEstimator`; what a *simpler process* must
expose to be mixed with one is :class:`behavio.contracts.mixture.MixtureComponent`.
"""

from behavio.compose.components import (
    UniformCategoryGuess,
    UniformChoiceGuess,
    UniformResponseGuess,
)
from behavio.compose.formula import model_from_formula
from behavio.compose.hierarchy import (
    HierarchicalFitResult,
    HierarchicalModel,
    HierarchicalSimulation,
    UnseenGroupPrediction,
    hierarchical,
)
from behavio.compose.mixture import (
    MixtureLikelihood,
    MixtureModel,
    MixtureSimulation,
    mix,
)
from behavio.compose.smoothness import SmoothModel, smooth
from behavio.compose.trajectory import CoefficientTrajectory

__all__ = [
    "CoefficientTrajectory",
    "HierarchicalFitResult",
    "HierarchicalModel",
    "HierarchicalSimulation",
    "MixtureLikelihood",
    "MixtureModel",
    "MixtureSimulation",
    "SmoothModel",
    "UniformCategoryGuess",
    "UniformChoiceGuess",
    "UniformResponseGuess",
    "UnseenGroupPrediction",
    "hierarchical",
    "mix",
    "model_from_formula",
    "smooth",
]
