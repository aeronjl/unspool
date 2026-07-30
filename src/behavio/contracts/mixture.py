"""What a *simpler process* must expose for a model to be mixed with it.

Three unrelated mechanisms used to say one thing. A drift-diffusion model took a
``contaminant=`` constructor slot; a whole class existed whose only job was adding a
symmetric lapse to a logistic psychometric curve; a reinforcement-learning policy carried a
bounded lapse of its own. None of them composed with each other and none of them could be
applied to a model that had not anticipated them, so a lapse on a GLM, on a multinomial or
on a GLM-HMM could not be written at all.

They are one idea: **the observation came from the model, or from something simpler.**

.. math::

    p(y_r) = (1 - \\omega)\\, p_{\\text{model}}(y_r) + \\omega\\, p_{\\text{component}}(y_r)

A lapse is that mixture with a guessing process; a contaminant is that mixture with a
distribution over the scored outcome. The two differ in *which component* and in nothing
else, which is why :func:`behavio.compose.mix` takes a **component** rather than growing a
vocabulary of processes: "lapse" and "contaminant" are names for particular components, not
for particular combinators.

What a component is, and what it is not
--------------------------------------
A component is a **declared** distribution over exactly the columns the model scores. It
has no estimated parameters of its own: the only thing a mixture estimates is the weight
:math:`\\omega`. That line is deliberate and it is where the asymmetric psychophysical
two-gamma form falls on the far side. Writing

.. math::

    \\psi(x) = \\gamma + (1 - \\gamma - \\lambda) F(z)
            = (\\gamma + \\lambda)\\frac{\\gamma}{\\gamma + \\lambda}
            + \\bigl(1 - (\\gamma + \\lambda)\\bigr) F(z)

shows it *is* a mixture -- with weight :math:`\\gamma + \\lambda` and a Bernoulli guess of
probability :math:`\\gamma / (\\gamma + \\lambda)`. But two free rates are two estimated
numbers, and a component with an estimated parameter is a second model hiding inside the
first: it would need its own coordinate, its own box, its own group prior and its own place
in every combinator. :class:`~behavio.models.PsychometricFunction` already estimates the two
rates inside the link, which is where a *shape of the curve* belongs. A component with a
**declared** asymmetry -- ``probability=0.6`` -- is expressible here and costs nothing; an
estimated one is the link's business.

The five things a mixture needs from a component
------------------------------------------------
*A density on the same outcome.*
    :meth:`MixtureComponent.pointwise_log_density`, scoring the array
    :meth:`~behavio.contracts.compose.PenalisedLinearEstimator.outcomes` produced -- so a
    drift-diffusion component sees seconds because that is the model's outcome coordinate,
    and a multinomial component sees category codes for the same reason. It takes the study
    as well, because per-trial availability is a property of the trial and not of the
    outcome.

*A prediction.*
    :meth:`MixtureComponent.prediction_probability`. A mixture's predicted probability is
    the weighted average of two predictions, so the component has to have one, and it has
    to have the shape of the model's: one number per row for a binary or joint family, one
    per category for a multinomial. :attr:`MixtureComponent.prediction_width` declares which
    before any study exists, because a combinator names its predictor cells statically.

*A simulator.*
    :meth:`MixtureComponent.simulate_outcomes`, returning replacement values for the scored
    columns of the rows the mixture drew as component rows. Without it a mixture could be
    fitted and never recovered, and a weight that cannot be recovered is not a parameter,
    it is a knob.

*An identity.*
    :attr:`~MixtureComponent.component_name`, :attr:`~MixtureComponent.signature` and
    :attr:`~MixtureComponent.weight_name`. The first two put the component in the composed
    model's signature, so a fit can never be read without knowing what it was mixed with.
    The third names the weight in the *reported* coordinate: ``lapse_rate`` for a guessing
    process, ``contaminant_rate`` for a distribution over response times. The weight is one
    parameter under either name; the name is the component's because "what this rate is
    called" is a fact about the simpler process.

*A refusal.*
    :meth:`MixtureComponent.mixture_refusal`. Structural typing answers "does this object
    have these members", which is not the question: a uniform guess over three categories
    and a multinomial over four have identical member sets and cannot be mixed. A component
    is asked, in a sentence, why a particular model is not one it can score -- the same
    shape of answer ``penalised_linear_refusal`` gives in
    :mod:`behavio.contracts.compose`, and for the same reason.

Where the weight lives
----------------------
The weight is **estimated**; its ceiling is **declared**. ``mix(model, component,
weight_bounds=(0, 0.25))`` says the mixture may account for at most a quarter of the trials
and nothing about where in that range it sits, which is exactly what the deleted
``maximum_lapse`` arguments and the drift-diffusion ``probability_bounds`` each said in
their own dialect. The estimated coordinate is :data:`MIXTURE_LOGIT`, a single unbounded
number, and the reported weight is :func:`mixture_weight` of it; the reverse map is
:func:`mixture_logit`.

A logit rather than the weight itself, for three reasons that are all the same reason. A
penalised linear fit is unconstrained and a Bernoulli GLM's solver has no box to put a
bound in, so a natural-scale weight would have forced a box onto every coefficient of every
model that could be mixed. A group deviation on a rate near zero is not Gaussian and a
deviation on its log odds is, so ``hierarchical(mix(model))`` gets the right prior for free.
And a Wald interval formed on a logit and mapped back cannot leave ``[0, 1]``, which is the
same argument :class:`~behavio.contracts.natural.NaturalParameterisation` exists to make.

The price is that a saturated weight is a large coordinate rather than a coordinate at a
bound, and that is what :data:`MIXTURE_LOGIT_BOUND` is for: it is the magnitude at which a
mixture is reported as resting against its declared range, and it is the bound a *boxed*
family's coordinate gets so that its own solver still has a finite box to search.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from behavio.trials import Study

__all__ = [
    "MIXTURE_LOGIT",
    "MIXTURE_LOGIT_BOUND",
    "MixtureComponent",
    "mixture_logit",
    "mixture_weight",
    "require_mixture_component",
    "validate_weight_bounds",
]

MIXTURE_LOGIT = "mixture_logit"
"""The name of the one coordinate :func:`behavio.compose.mix` adds to a model."""

MIXTURE_LOGIT_BOUND = 12.0
"""Where a mixture logit stops being a number and starts being a saturation report.

``expit(12)`` is within four parts in a hundred thousand of one, so a weight whose logit
reaches this magnitude is at its declared range to any precision a behavioural study can
speak to. It is used twice: as the finite box a bounded family's own solver needs, and as
the threshold at which a fit is reported as a boundary estimate.
"""


@runtime_checkable
class MixtureComponent(Protocol):
    """A declared, parameter-free process a model may be mixed with."""

    @property
    def component_name(self) -> str:
        """A short name for the process, used in the composed model's name."""
        ...

    @property
    def signature(self) -> str:
        """A stable fingerprint of everything declared about this component."""
        ...

    @property
    def weight_name(self) -> str:
        """What the estimated mixing weight is called in the reported coordinate."""
        ...

    @property
    def scored_columns(self) -> tuple[str, ...]:
        """The study columns this component produces, which must be the model's."""
        ...

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        """The outcome shape this component scores, which must be the model's."""
        ...

    @property
    def prediction_width(self) -> int:
        """How many numbers this component's prediction has per row.

        ``1`` for a process predicting a single probability, and one per category for a
        process predicting a categorical distribution. Declared statically because a
        combinator names its predictor cells before it has seen a study.
        """
        ...

    def mixture_refusal(self, model: Any) -> str | None:
        """Return why this component cannot score ``model``'s outcome, or ``None``."""
        ...

    def pointwise_log_density(
        self, study: Study, outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return one log density per row, on the model's own outcome coordinate.

        ``-inf`` is a legitimate answer and means the observation lies outside this
        process's support -- a response faster than a uniform contaminant window allows.
        """
        ...

    def prediction_probability(self, study: Study) -> NDArray[np.float64]:
        """Return this process's predicted probability, shaped like the model's."""
        ...

    def simulate_outcomes(
        self,
        study: Study,
        rows: NDArray[np.intp],
        *,
        generator: np.random.Generator,
    ) -> Mapping[str, NDArray[Any]]:
        """Return replacement scored-column values for the rows drawn from this process."""
        ...


def require_mixture_component(component: Any, model: Any, *, combinator: str) -> MixtureComponent:
    """Return ``component`` if it may be mixed with ``model``, or raise.

    Three questions, in the order in which their answers are worth reading: has the object
    got the members at all, does it write the columns the model scores, and does it say --
    in its own sentence -- that this particular model is not one it can score. The third is
    a method rather than an inspection for the reason structural typing always fails here:
    a uniform guess over three categories has exactly the members of a uniform guess over
    four.
    """

    if not isinstance(component, MixtureComponent):
        raise TypeError(
            f"{combinator}() needs a component satisfying behavio.contracts.mixture."
            f"MixtureComponent; {type(component).__name__} does not"
        )
    scored = tuple(component.scored_columns)
    model_scored = tuple(getattr(model, "scored_columns", ()))
    if scored != model_scored:
        raise TypeError(
            f"{type(component).__name__} writes {list(scored)} but "
            f"{type(model).__name__} scores {list(model_scored)}, so they are not a "
            "mixture of one observation"
        )
    channels = tuple(component.outcome_channels)
    model_channels = tuple(getattr(model, "outcome_channels", ()))
    if channels != model_channels:
        raise TypeError(
            f"{type(component).__name__} scores outcome channels {list(channels)} but "
            f"{type(model).__name__} declares {list(model_channels)}"
        )
    width = component.prediction_width
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise TypeError("prediction_width must be a positive integer")
    refusal = component.mixture_refusal(model)
    if refusal:
        raise TypeError(
            f"{combinator}() cannot mix {type(model).__name__} with "
            f"{type(component).__name__}: {refusal}"
        )
    return component


def validate_weight_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    """Return the declared range a mixture weight is estimated inside, validated.

    The upper bound is strictly below one because a mixture in which the model explains
    nothing is not a mixture, and the lower bound is at or above zero because a negative
    weight is not a probability. Zero itself is admissible and is the useful default: it
    says the data may conclude that there is no second process, which is exactly what a
    lapse rate estimated at its floor is reporting.
    """

    values = tuple(float(value) for value in bounds)
    if len(values) != 2:
        raise ValueError("weight_bounds must contain a lower and an upper bound")
    lower, upper = values
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("weight_bounds must be finite")
    if not 0.0 <= lower < upper < 1.0:
        raise ValueError("weight_bounds must satisfy 0 <= lower < upper < 1")
    return lower, upper


def mixture_weight(
    coordinate: NDArray[np.float64] | float, bounds: tuple[float, float]
) -> NDArray[np.float64]:
    """Map a mixture logit onto the declared weight range."""

    lower, upper = bounds
    return np.asarray(lower + (upper - lower) * expit(coordinate), dtype=np.float64)


def mixture_logit(weight: float, bounds: tuple[float, float]) -> float:
    """Map a weight strictly inside the declared range back onto the estimated logit."""

    lower, upper = bounds
    value = float(weight)
    if not np.isfinite(value) or not lower < value < upper:
        raise ValueError(f"a mixture weight must lie strictly inside ({lower}, {upper})")
    return float(logit((value - lower) / (upper - lower)))
