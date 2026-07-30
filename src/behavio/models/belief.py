"""Normative belief updating: a Bayesian ideal observer and the Hierarchical Gaussian Filter.

Two families, one shape. A **perceptual model** turns a sequence of binary observations
:math:`u_{1:K}` into a belief :math:`\\hat{\\mu}_k = P(u_k = 1 \\mid u_{1:k-1})` held
*before* trial :math:`k` is seen; a **response model** turns that belief into the
probability of the subject's own binary response. The two are separate objects, separately
parameterised and separately named, because that is the field's own distinction and because
the same belief trajectory is routinely read out through different decision rules.

.. math::

    \\hat{\\mu}_k = \\text{perception}(u_{1:k-1}; \\vartheta), \\qquad
    P(y_k = 1) = \\sigma\\!\\left(\\eta(\\hat{\\mu}_k; \\zeta)\\right).

Both response models here are a logistic function of a linear predictor in the belief --
:class:`BeliefSoftmax` uses :math:`\\eta = \\beta(2\\hat{\\mu} - 1) + b`, and the unit-square
sigmoid of Mathys et al. (2014) is *exactly* :math:`\\eta = \\zeta\\,\\text{logit}(\\hat{\\mu})`
-- so the likelihood, its gradient, the solver and all thirteen composition members are
written once and a response rule costs one linear predictor and its two derivatives.

The observation is not the response
-----------------------------------
This is the structural fact that separates these models from the reinforcement-learning
families, and it decides what the combinators may do with them. A Q-learning agent's value
trace is written by **the action the agent took**, so the likelihood is a genuine recursion:
row :math:`k` has no density of its own. A normative observer's belief trajectory is written
by **the task's observations**, which are exogenous, so conditional on the parameters the
rows *are* independent and each one does have a density. That is why
:func:`~behavio.compose.mix` may put a lapse on top of an observer and may not put one on top
of an agent, and it is scientifically the right answer: a lapse on the response leaves the
perceptual model untouched, because the perceptual model never saw the response.

What survives is the other half of the block rule. A *perceptual* parameter that changed
part-way through a session would leave the belief trajectory unable to say which of its
values produced which part, so the perceptual coordinate must be constant within each
subject/session block even though every row is its own density. The two statements are
different and :attr:`~behavio.contracts.bounded.RowObjective.row_blocks` can only carry one
of them, so ``row_blocks`` reports the density statement -- ``arange(n_rows)`` -- and
:meth:`_BeliefRowObjective.value_and_gradient` enforces the coordinate statement itself,
against the recursion's own blocks, with the message that names the perceptual parameters.

Conventions this module fixes, because published implementations disagree
------------------------------------------------------------------------
The HGF's update equations are where implementations diverge, and a plausible-looking HGF
that is subtly wrong is worse than none. Every choice below is stated so it can be argued
with, and each is checked in ``tests/test_belief.py``.

*The binary first level contributes a variance, not a precision.* At level two,

.. math::

    \\pi_2^{(k)} = \\hat{\\pi}_2^{(k)}
      + \\hat{\\mu}_1^{(k)}\\bigl(1 - \\hat{\\mu}_1^{(k)}\\bigr),

which is :math:`\\hat{\\pi}_2 + 1/\\hat{\\pi}_1` with :math:`\\hat{\\pi}_1 = 1/(\\hat{\\mu}_1(1
- \\hat{\\mu}_1))`. Adding :math:`\\hat{\\pi}_1` itself -- the reading the notation invites --
inverts the update's dependence on how confident the observer already is, and still
converges, and still reports a number. This is the binary HGF's one genuinely surprising
line (Mathys et al. 2011, eq. 26; Mathys et al. 2014, eq. 12).

*Volatility couples through the exponent, on the previous trial's estimate.* The step
variance at level two is :math:`\\nu_2^{(k)} = \\exp(\\kappa_2\\mu_3^{(k-1)} + \\omega_2)`, not
:math:`\\kappa_2\\exp(\\mu_3) + \\omega_2` and not :math:`\\exp(\\mu_3)^{\\kappa_2}`, and it reads
the **posterior of the previous trial** rather than the current one, because the filter is
one-step-ahead everywhere.

*The third level's prediction errors.* The volatility prediction error is

.. math::

   \\delta_2^{(k)} = \\Bigl(\\sigma_2^{(k)}
     + \\bigl(\\mu_2^{(k)} - \\hat{\\mu}_2^{(k)}\\bigr)^2\\Bigr)\\,\\hat{\\pi}_2^{(k)} - 1,

formed from the **posterior** mean and variance at level two against the **prediction**
precision, and the volatility weight is :math:`w_2 = \\nu_2\\hat{\\pi}_2 \\in (0, 1)`. The
third level then updates as

.. math::

   \\pi_3^{(k)} &= \\hat{\\pi}_3^{(k)}
     + \\tfrac{1}{2}\\kappa_2^2 w_2\\bigl(w_2 + (2w_2 - 1)\\delta_2\\bigr), \\\\
   \\mu_3^{(k)} &= \\mu_3^{(k-1)} + \\tfrac{1}{2}\\,\\frac{\\kappa_2 w_2 \\delta_2}{\\pi_3^{(k)}}.

The :math:`\\tfrac{1}{2}` appears in both, the :math:`\\kappa_2` is squared in the precision
and linear in the mean, and the :math:`(2w_2 - 1)` factor is what can drive the precision
negative. See :class:`NegativePosteriorPrecision`.

*No drift, unit trial spacing, declared initial variances.* :math:`\\rho_2 = \\rho_3 = 0` and
:math:`t^{(k)} = 1`: this module fits a sequence of trials, not a set of timestamps, and a
drift term on an already weakly identified volatility is a coordinate the data cannot
locate. :math:`\\sigma_2^{(0)}`, :math:`\\sigma_3^{(0)}` and :math:`\\mu_3^{(0)}` are
**constructor arguments**, so they appear in the model signature and a fit can never be read
without them; only :math:`\\mu_2^{(0)}` is estimated, as ``initial_belief`` on the logit
scale, because it is the one initial condition a design can move.

*Estimated coordinates.* :math:`\\omega_2` and :math:`\\omega_3` are already logarithms of
variances, so they are estimated and reported unchanged; :math:`\\kappa_2 > 0` is estimated
as ``volatility_coupling_log``. The ideal observer's retention and prior mean are estimated
as logits and its prior strength as a logarithm. Every estimated coordinate is unconstrained,
which is what :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator` requires of a
model that is to be made hierarchical or smooth.

What reduces to what
--------------------
:func:`beta_bernoulli_beliefs` at ``retention=1`` is the exact Beta-Bernoulli posterior mean
:math:`(\\alpha_0 + n_1)/(\\nu + n)`, and at ``retention < 1`` it is *exactly* a
Rescorla-Wagner rule with a decay toward the prior mean,

.. math::

   \\mu_{k+1} = \\mu_k + \\frac{u_k - \\mu_k}{n_{k+1}}
     + \\frac{(1 - \\rho)\\nu\\,(m - \\mu_k)}{n_{k+1}}, \\qquad
   n_{k+1} = \\rho n_k + (1 - \\rho)\\nu + 1,

whose count recursion does not depend on the data at all and converges to
:math:`n^{*} = \\nu + 1/(1 - \\rho)`. The asymptotic learning rate is therefore the closed
form :math:`1/n^{*}`, which is what makes "a leaky ideal observer *is* a delta rule" a
statement with a number in it rather than an analogy.

The binary HGF's level-two update is likewise exactly a delta rule,
:math:`\\mu_2^{(k)} = \\mu_2^{(k-1)} + \\sigma_2^{(k)}\\,(u_k - \\hat{\\mu}_1^{(k)})`, whose
learning rate is the posterior variance. With the volatility held constant -- a two-level
filter, or :math:`\\kappa_2 = 0`, which makes the third level inert and the first two levels
*identical* to the two-level filter's -- that variance obeys a data-independent recursion
with the unique attracting fixed point :func:`hgf_fixed_learning_rate` returns, and the
filter is Rescorla-Wagner with that fixed rate. The honest qualification, which this module
states rather than glosses: the binary observation makes the fixed point depend on the
current belief through :math:`\\hat{\\mu}_1(1 - \\hat{\\mu}_1)`, so the reduction is exact
where the belief is stationary and approximate elsewhere. It is exactly Rescorla-Wagner for
*every* trajectory only in the continuous-input HGF, where the observation precision is a
constant. ``tests/test_belief.py`` asserts the exact fixed point by driving the filter with a
belief-neutral input, and bounds the departure on a real binary sequence.

Identifiability hazards, and where they are reported
----------------------------------------------------
Volatility and coupling trade off notoriously: a large :math:`\\kappa_2` with a small
:math:`\\omega_3` predicts nearly what a small :math:`\\kappa_2` with a large
:math:`\\omega_3` predicts, because both scale how much the third level is allowed to move.
Each hazard that is a fact about the *design* is a ``describe()`` finding, following the
precedent :func:`~behavio.compose.mix`, the psychometric family and the economic family set:

``stationary_observations``
    The observation sequence's rate does not change across the session, so there is no
    volatility for a third level to estimate. Reported against
    :class:`HierarchicalGaussianFilter` at three levels and against
    :class:`BetaBernoulliObserver`, whose leak is identified by exactly the same thing.
``short_blocks_for_volatility``
    A three-level filter estimating a volatility from a session too short for the
    volatility to have changed within it.
``degenerate_observations``
    A block whose observations are all identical: the belief saturates and the response
    parameter has one value of the predictor to work with.
``constant_response``
    Every response in the study is the same, so the response model's precision runs to its
    box whatever the belief did.

The post-fit half arrives through the machinery that already exists: a coordinate resting on
its box sets ``boundary_estimate``, and
:meth:`HierarchicalGaussianFilter.coupling_volatility_correlation` reads the Wald correlation
between :math:`\\kappa_2` and :math:`\\omega_3` straight off the fitted covariance, exactly as
the economic family's ``temperature_scale_correlation`` does.

Why this is a clean-room implementation
---------------------------------------
``pyhgf`` is not wrapped. Its licence is stated as GPL-3.0 on PyPI and differently in its
repository, and it pins ``jax<0.4.32``, which cannot share an environment with the other
extras this package offers. Everything here is written from the published update equations
cited below, and validated against closed forms rather than against another implementation.

References
----------
Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation
for individual learning under uncertainty. *Frontiers in Human Neuroscience, 5*, 39.

Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston,
K. J., & Stephan, K. E. (2014). Uncertainty in perception and the Hierarchical Gaussian
Filter. *Frontiers in Human Neuroscience, 8*, 825.

Behrens, T. E. J., Woolrich, M. W., Walton, M. E., & Rushworth, M. F. S. (2007). Learning
the value of information in an uncertain world. *Nature Neuroscience, 10*(9), 1214-1221.

Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning. In A. H. Black
& W. F. Prokasy (Eds.), *Classical Conditioning II* (pp. 64-99). Appleton-Century-Crofts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.contracts.bounded import (
    RowCoefficientDesign,
    block_constant_coordinates,
    validated_row_coefficients,
)
from behavio.contracts.compose import ridge_group_draw, ridge_group_penalty
from behavio.contracts.estimator import DerivedQuantity
from behavio.contracts.natural import natural_quantities
from behavio.models._kernels.bernoulli import ordered_session_indices
from behavio.models._kernels.curvature import (
    covariance_from_hessian,
    finite_difference_hessian,
    offset_steps,
)
from behavio.models._kernels.introspection import WARNING, Describable, ModelFinding
from behavio.models._kernels.rowfit import solve_row_coefficients
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.trials import REQUIRED_COLUMNS, Study

__all__ = [
    "BELIEF_FLOOR",
    "BELIEF_SENSITIVITY_FLOOR",
    "SENSITIVITY_PROBE",
    "VIOLATION_PENALTY",
    "BeliefFitResult",
    "BeliefResponse",
    "BeliefSoftmax",
    "BeliefTrajectory",
    "BetaBernoulliObserver",
    "BetaBernoulliParameters",
    "HierarchicalGaussianFilter",
    "HierarchicalGaussianFilterParameters",
    "NegativePosteriorPrecision",
    "UnitSquareSigmoid",
    "VolatilityStability",
    "beta_bernoulli_beliefs",
    "hgf_beliefs",
    "hgf_fixed_learning_rate",
]

#: Smallest belief a response model is evaluated at, so a saturated logit stays finite.
#:
#: The unit-square sigmoid reads ``logit(belief)``, which diverges at either end, and a
#: three-level filter can drive the belief to within floating-point distance of zero after a
#: long unbroken run. ``1e-6`` bounds the linear predictor at roughly 14 log-odds before the
#: response precision multiplies it, which is far outside anything a fitted precision reaches
#: and far inside where the arithmetic stops being meaningful.
BELIEF_FLOOR = 1e-6

#: Magnitude at which a finite coordinate is treated as resting on its declared box.
BOUNDARY_TOLERANCE = 1e-4

#: How far a perceptual coordinate is displaced when its identifiability is probed.
#:
#: One unit on the estimated scale, which is a factor of :math:`e` for a coupling strength, a
#: factor of :math:`e` in the step *variance* for a tonic volatility, and about a fifth of the
#: distance from a retention of 0.5 to one of 0.9. A parameter that a study's belief
#: trajectory barely notices moving that far is one that study cannot locate.
SENSITIVITY_PROBE = 1.0

#: Smallest belief-vector displacement a probe may cause before a parameter counts as seen.
#:
#: The statistic is the **Euclidean norm** of the change in the study's whole vector of
#: predicted beliefs, not the largest single change, and the difference matters: a parameter
#: that moves three trials a long way and four hundred not at all is not identified by four
#: hundred trials, and only a norm over the study says so. The norm also grows with the number
#: of trials that a parameter actually touches, which is the direction evidence grows in.
#:
#: A tenth of a probability, in that norm. A displacement of that size moves the response
#: model's linear predictor by a few tenths of a log-odds unit over the entire study -- about
#: what one extra correct response is worth -- so below it the study cannot tell the probed
#: parameter value from the unprobed one. Reported as a finding rather than an error: an
#: unidentified parameter is a fact about a design, and carrying one is sometimes the right
#: choice, but never an accidental one.
BELIEF_SENSITIVITY_FLOOR = 0.1

#: Per-row objective assigned where a filter's posterior precisions are not positive.
#:
#: Finite rather than infinite, and three orders of magnitude above anything attainable: a
#: row's Bernoulli loss is bounded by the largest linear predictor the boxes admit, which is
#: about 2100 for the widest response model here. An infinite objective is what a violated
#: region *deserves* and it is not what a boxed quasi-Newton solver can use -- see
#: :meth:`_BeliefObserver._rows_value_and_gradient` -- so the region is walled off at a height
#: no admissible fit can reach and both :meth:`_BeliefObserver.fit` and
#: :meth:`_BeliefObserver.fit_rows` refuse a result that ends up above the wall.
VIOLATION_PENALTY = 1e6

#: Largest exponent the volatility coupling is evaluated at before the filter is refused.
_EXPONENT_CEILING = 600.0


# --------------------------------------------------------------------------------------
# Failures the arithmetic can have, named
# --------------------------------------------------------------------------------------


class NegativePosteriorPrecision(ModelDataError):
    """The third level's posterior precision left the positive reals.

    A known and documented failure mode of the three-level HGF rather than a defect of this
    implementation: :math:`\\pi_3 = \\hat{\\pi}_3 + \\tfrac{1}{2}\\kappa_2^2 w_2(w_2 + (2w_2 -
    1)\\delta_2)` has a negative second term whenever :math:`w_2 < \\tfrac{1}{2}` and the
    volatility prediction error is large enough, and for admissible parameter values the sum
    can cross zero. Past that point the filter's Gaussian approximation has no posterior to
    describe and every number downstream of it -- a belief, a likelihood, a standard error --
    is arithmetic rather than inference.

    It is therefore raised rather than clipped. Inside :meth:`HierarchicalGaussianFilter.fit`
    the same condition scores :data:`VIOLATION_PENALTY` per row -- finite rather than
    infinite, for the reason :meth:`_BeliefObserver._rows_value_and_gradient` states -- so the
    optimizer walks away from the region instead of reporting a confident estimate from
    inside it; a fit is refused with
    this error only when *every* restart lands there, and a fit that succeeds carries the
    margin it succeeded by as the ``minimum_volatility_precision`` derived quantity.
    """


# --------------------------------------------------------------------------------------
# Trajectories, as records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BeliefTrajectory:
    """One observer's filtered belief and the states that produced it.

    ``belief`` is the one-step-ahead prediction :math:`\\hat{\\mu}_k = P(u_k = 1 \\mid
    u_{1:k-1})` held *before* trial ``k``'s observation is seen, in the study's own source
    row order. ``learning_rate`` is the step size the update multiplied the prediction error
    by, which is the quantity the Rescorla-Wagner comparison is about: the posterior variance
    :math:`\\sigma_2` for the filter and the reciprocal count :math:`1/n_{k+1}` for the ideal
    observer. ``states`` are each family's own post-update quantities, named by
    ``state_names`` so a reader never has to count columns.
    """

    belief: NDArray[np.float64]
    prediction_error: NDArray[np.float64]
    learning_rate: NDArray[np.float64]
    state_names: tuple[str, ...]
    states: NDArray[np.float64]

    def __post_init__(self) -> None:
        belief = protected_array(self.belief, dtype=np.float64)
        error = protected_array(self.prediction_error, dtype=np.float64)
        rate = protected_array(self.learning_rate, dtype=np.float64)
        names = tuple(self.state_names)
        states = protected_array(self.states, dtype=np.float64)
        if belief.ndim != 1:
            raise ValueError("belief must contain one value per trial")
        if error.shape != belief.shape or rate.shape != belief.shape:
            raise ValueError("trajectory vectors must contain one value per trial")
        if not names or len(set(names)) != len(names):
            raise ValueError("state names must be non-empty and unique")
        if states.shape != (len(belief), len(names)):
            raise ValueError("states must contain one named value per trial")
        if np.any((belief < 0.0) | (belief > 1.0)):
            raise ValueError("a belief must lie in the unit interval")
        object.__setattr__(self, "belief", belief)
        object.__setattr__(self, "prediction_error", error)
        object.__setattr__(self, "learning_rate", rate)
        object.__setattr__(self, "state_names", names)
        object.__setattr__(self, "states", states)

    def state(self, name: str) -> NDArray[np.float64]:
        """Return one named state's trajectory, or raise naming what is available."""

        if name not in self.state_names:
            raise KeyError(f"no state {name!r}; available: {list(self.state_names)}")
        return np.asarray(self.states[:, self.state_names.index(name)], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VolatilityStability:
    """Whether one parameter vector keeps a filter's posterior precisions positive.

    Reported rather than asserted, because the answer is a fact about a parameter value and a
    particular observation sequence and only the pair. ``first_violation`` is the source row
    index of the trial at which the recursion left the admissible region, or ``None``;
    ``minimum_precision`` is the smallest third-level posterior precision reached, which is
    the margin a successful fit succeeded by and the number that says how close to refusal it
    was. Both are ``inf``/``None`` for a family with no third level.
    """

    admissible: bool
    first_violation: int | None
    minimum_precision: float

    def __post_init__(self) -> None:
        if self.admissible != (self.first_violation is None):
            raise ValueError("an admissible trajectory has no violating trial, and conversely")


# --------------------------------------------------------------------------------------
# The response model, which is a declared component rather than a baked-in rule
# --------------------------------------------------------------------------------------


@runtime_checkable
class BeliefResponse(Protocol):
    """A decision rule mapping a belief to the log-odds of the observed response.

    The field's own perceptual/response split, made a member rather than an assumption. A
    response model owns its parameters, their unconstrained coordinates, its own box and its
    own starting values, and supplies one thing to the likelihood: a linear predictor and its
    derivative with respect to its coordinate and with respect to the belief. Everything else
    -- the Bernoulli likelihood, the optimizer, prediction, scoring, simulation and all
    thirteen composition members -- is written once against this protocol.

    A response *lapse* is not declared here. :func:`~behavio.compose.mix` already expresses
    it, as a mixture of this model's density with a
    :class:`~behavio.compose.UniformChoiceGuess`, and it applies to these families where it
    does not apply to the reinforcement-learning agents: an observer's belief trajectory is
    written by the task's observations and not by the subject's response, so a lapsed trial
    leaves the perceptual model exactly where it was.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The unconstrained coordinates this rule adds to the model's."""
        ...

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported names of the same parameters."""
        ...

    @property
    def signature(self) -> str:
        """A stable identity carried in the composed model's signature."""
        ...

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        """Map a natural mapping onto this rule's estimated coordinate."""
        ...

    def to_natural(self, vector: NDArray[np.float64]) -> tuple[float, ...]:
        """Map this rule's estimated coordinate onto its natural values, in order."""
        ...

    def jacobian(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``d natural / d estimated`` for this rule's own coordinate."""
        ...

    def box(self) -> NDArray[np.float64]:
        """Return the finite box this rule's coordinate is searched in."""
        ...

    def starts(self) -> tuple[NDArray[np.float64], ...]:
        """Return the deterministic starting vectors for this rule's coordinate."""
        ...

    def linear_predictor(
        self, belief: NDArray[np.float64], rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``eta``, ``d eta / d rows`` and ``d eta / d belief``, per row."""
        ...


@dataclass(frozen=True, slots=True)
class BeliefSoftmax:
    """Softmax over the two options' expected values, with an optional side bias.

    The belief assigns expected value :math:`\\hat{\\mu}` to responding one and
    :math:`1 - \\hat{\\mu}` to responding zero, so the value difference is
    :math:`2\\hat{\\mu} - 1` and

    .. math:: \\eta = \\beta\\,(2\\hat{\\mu} - 1) + b.

    This is the rule used when the response is a *choice between the two outcomes* and it is
    the one that makes :math:`\\beta` comparable with the inverse temperature of a
    reinforcement-learning policy or of a value-based choice model, because all three
    multiply a difference of expected values.
    """

    include_bias: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.include_bias, bool):
            raise ValueError("include_bias must be boolean")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = ["inverse_temperature_log"]
        if self.include_bias:
            names.append("choice_bias")
        return tuple(names)

    @property
    def natural_names(self) -> tuple[str, ...]:
        names = ["inverse_temperature"]
        if self.include_bias:
            names.append("choice_bias")
        return tuple(names)

    @property
    def signature(self) -> str:
        return f"belief_softmax(include_bias={self.include_bias!r})"

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        temperature = _require_positive(natural, "inverse_temperature")
        values = [float(np.log(temperature))]
        if self.include_bias:
            values.append(_require_finite(natural, "choice_bias"))
        return tuple(values)

    def to_natural(self, vector: NDArray[np.float64]) -> tuple[float, ...]:
        values = [float(np.exp(vector[0]))]
        if self.include_bias:
            values.append(float(vector[1]))
        return tuple(values)

    def jacobian(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        diagonal = [float(np.exp(vector[0]))]
        if self.include_bias:
            diagonal.append(1.0)
        return np.diag(np.asarray(diagonal, dtype=np.float64))

    def box(self) -> NDArray[np.float64]:
        bounds = [(-5.0, 5.0)]
        if self.include_bias:
            bounds.append((-10.0, 10.0))
        return np.asarray(bounds, dtype=np.float64)

    def starts(self) -> tuple[NDArray[np.float64], ...]:
        temperatures = (np.log(4.0), np.log(1.0), np.log(12.0))
        if not self.include_bias:
            return tuple(np.asarray([value], dtype=np.float64) for value in temperatures)
        return tuple(np.asarray([value, 0.0], dtype=np.float64) for value in temperatures)

    def linear_predictor(
        self, belief: NDArray[np.float64], rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the softmax log-odds and its two derivatives."""

        temperature = np.exp(rows[:, 0])
        difference = 2.0 * belief - 1.0
        linear = temperature * difference
        jacobian = np.empty_like(rows)
        jacobian[:, 0] = linear
        if self.include_bias:
            linear = linear + rows[:, 1]
            jacobian[:, 1] = 1.0
        return linear, jacobian, 2.0 * temperature


@dataclass(frozen=True, slots=True)
class UnitSquareSigmoid:
    """Mathys et al.'s (2014) unit-square sigmoid observation model.

    .. math::

        P(y = 1) = \\frac{\\hat{\\mu}^{\\zeta}}
                        {\\hat{\\mu}^{\\zeta} + (1 - \\hat{\\mu})^{\\zeta}}
                 = \\sigma\\!\\left(\\zeta\\,\\text{logit}(\\hat{\\mu})\\right),

    so it is a softmax on the belief's **log odds** rather than on its value difference, and
    the second form is how it is implemented here: the two are identical and the second has a
    linear predictor a Bernoulli likelihood can be written against directly. :math:`\\zeta = 1`
    is exact probability matching, :math:`\\zeta \\to \\infty` is deterministic maximising and
    :math:`\\zeta \\to 0` is a coin flip; the parameter is estimated as
    ``decision_noise_log`` because it is strictly positive and multiplicative.

    Where this differs from :class:`BeliefSoftmax` in practice: a belief of 0.99 is 4.6 log
    odds but only 0.98 value units, so the unit-square sigmoid keeps discriminating between
    confident beliefs where a softmax has already saturated. Which is right is a claim about
    the task, which is why neither is baked in.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("decision_noise_log",)

    @property
    def natural_names(self) -> tuple[str, ...]:
        return ("decision_noise",)

    @property
    def signature(self) -> str:
        return "unit_square_sigmoid"

    def encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        return (float(np.log(_require_positive(natural, "decision_noise"))),)

    def to_natural(self, vector: NDArray[np.float64]) -> tuple[float, ...]:
        return (float(np.exp(vector[0])),)

    def jacobian(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.diag(np.asarray([float(np.exp(vector[0]))], dtype=np.float64))

    def box(self) -> NDArray[np.float64]:
        return np.asarray([(-5.0, 5.0)], dtype=np.float64)

    def starts(self) -> tuple[NDArray[np.float64], ...]:
        return tuple(
            np.asarray([value], dtype=np.float64)
            for value in (np.log(1.0), np.log(3.0), np.log(0.4))
        )

    def linear_predictor(
        self, belief: NDArray[np.float64], rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the log-odds predictor and its two derivatives."""

        noise = np.exp(rows[:, 0])
        odds = np.log(belief) - np.log1p(-belief)
        linear = noise * odds
        jacobian = np.empty_like(rows)
        jacobian[:, 0] = linear
        return linear, jacobian, noise / (belief * (1.0 - belief))


# --------------------------------------------------------------------------------------
# The two published closed forms this module is validated against
# --------------------------------------------------------------------------------------


#: How many deterministic starting vectors each family declares per coordinate.
_RESTART_POINTS = 3


@dataclass(frozen=True, slots=True)
class _Coordinate:
    """One perceptual parameter: what it is called, where it lives and how it transforms.

    ``transform`` names the map from the estimated coordinate to the reported one, and it is
    always one of the three this package already uses -- ``identity`` for a parameter that is
    already unbounded, ``log`` for a positive one, ``logit`` for one in the unit interval.
    ``bounds`` and ``starts`` are on the **estimated** scale, because that is the scale the
    optimizer searches and the scale a Gaussian group deviation is placed on.
    """

    natural: str
    estimated: str
    transform: str
    bounds: tuple[float, float]
    starts: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.transform not in ("identity", "log", "logit"):
            raise ValueError("a perceptual transform must be identity, log or logit")
        if len(self.starts) != _RESTART_POINTS:
            raise ValueError(f"a perceptual coordinate needs {_RESTART_POINTS} starting values")

    def to_natural(self, value: float) -> float:
        """Map one estimated value onto the reported scale."""

        if self.transform == "identity":
            return float(value)
        if self.transform == "log":
            return float(np.exp(value))
        return float(expit(value))

    def encode(self, value: float) -> float:
        """Map one reported value back onto the estimated scale, or raise."""

        if self.transform == "identity":
            return float(value)
        if self.transform == "log":
            if value <= 0.0:
                raise ValueError(f"{self.natural} must be positive")
            return float(np.log(value))
        if not 0.0 < value < 1.0:
            raise ValueError(f"{self.natural} must lie strictly between zero and one")
        return float(np.log(value) - np.log1p(-value))

    def slope(self, value: float) -> float:
        """Return ``d natural / d estimated`` at one estimated value."""

        if self.transform == "identity":
            return 1.0
        if self.transform == "log":
            return float(np.exp(value))
        probability = float(expit(value))
        return probability * (1.0 - probability)


@dataclass(frozen=True, slots=True)
class _Path:
    """One block's filtered trajectory, before it is placed back into source row order."""

    belief: NDArray[np.float64]
    learning_rate: NDArray[np.float64]
    states: NDArray[np.float64]
    violation: int
    minimum_precision: float


def beta_bernoulli_beliefs(
    observations: Sequence[float] | NDArray[np.float64],
    *,
    retention: float = 1.0,
    prior_mean: float = 0.5,
    prior_strength: float = 2.0,
) -> BeliefTrajectory:
    """Return the leaky Beta-Bernoulli ideal observer's belief trajectory.

    The foundational normative model: a binary outcome with an unknown and possibly changing
    rate, tracked by a Beta posterior over that rate whose counts are discounted by
    ``retention`` :math:`\\rho` each trial. With :math:`\\alpha_0 = \\nu m` and
    :math:`\\beta_0 = \\nu (1 - m)` for prior strength :math:`\\nu` and prior mean :math:`m`,

    .. math::

        \\alpha_k = \\alpha_0 + \\sum_{s<k}\\rho^{\\,k-1-s} u_s, \\qquad
        \\beta_k  = \\beta_0  + \\sum_{s<k}\\rho^{\\,k-1-s} (1 - u_s),

    and the belief is the posterior mean :math:`\\alpha_k / (\\alpha_k + \\beta_k)`. At
    :math:`\\rho = 1` this is the **exact** Bayesian posterior mean for a static rate, which
    is the closed form ``tests/test_belief.py`` holds the implementation to; below one it is
    the exponential-forgetting approximation to a change-point process, with an effective
    memory of :math:`1/(1-\\rho)` trials, which is the standard workhorse (Behrens et al.
    2007 is the change-point formulation the leak approximates).

    ``retention`` may be exactly one here. The estimated coordinate is a logit and so cannot
    be, which is deliberate: a fit reports the largest retention its design can distinguish
    from lossless memory, not the claim that memory is lossless.

    ``observations`` may be any values in :math:`[0, 1]`; the model classes require
    :math:`\\{0, 1\\}`, but the update equations are defined for a fractional observation and
    a belief-neutral input is how the fixed point of the recursion is checked.
    """

    values = _validated_observations(observations)
    if not np.isfinite(retention) or not 0.0 < retention <= 1.0:
        raise ValueError("retention must be finite and lie in (0, 1]")
    if not np.isfinite(prior_mean) or not 0.0 < prior_mean < 1.0:
        raise ValueError("prior_mean must be finite and lie strictly in (0, 1)")
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be finite and positive")
    path = _beta_bernoulli_path(
        values.tolist(),
        retention=float(retention),
        prior_mean=float(prior_mean),
        prior_strength=float(prior_strength),
    )
    return BeliefTrajectory(
        belief=path.belief,
        prediction_error=values - path.belief,
        learning_rate=path.learning_rate,
        state_names=_BETA_BERNOULLI_STATES,
        states=path.states,
    )


def hgf_beliefs(
    observations: Sequence[float] | NDArray[np.float64],
    *,
    initial_belief: float = 0.0,
    tonic_volatility: float = -3.0,
    volatility_coupling: float | None = None,
    meta_volatility: float | None = None,
    initial_variance: float = 1.0,
    initial_meta_variance: float = 1.0,
    initial_meta_belief: float = 0.0,
) -> BeliefTrajectory:
    """Return the binary Hierarchical Gaussian Filter's belief trajectory.

    Two levels when ``volatility_coupling`` and ``meta_volatility`` are both ``None``, three
    when both are given. The update equations, and every convention in them that published
    implementations disagree about, are stated in this module's docstring; the natural-scale
    arguments here are the ones papers report, so a trajectory can be reproduced from a
    published table without constructing a model.

    Raises :class:`NegativePosteriorPrecision` when the third level's posterior precision
    leaves the positive reals, naming the trial. That is a property of the parameters and the
    sequence together, not a bug, and it is refused rather than clipped.
    """

    values = _validated_observations(observations)
    three = _validated_levels(volatility_coupling, meta_volatility)
    if not np.isfinite(initial_belief):
        raise ValueError("initial_belief must be finite")
    if not np.isfinite(tonic_volatility):
        raise ValueError("tonic_volatility must be finite")
    for value, label in (
        (initial_variance, "initial_variance"),
        (initial_meta_variance, "initial_meta_variance"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    if not np.isfinite(initial_meta_belief):
        raise ValueError("initial_meta_belief must be finite")
    path = _hgf_path(
        values.tolist(),
        initial_belief=float(initial_belief),
        tonic_volatility=float(tonic_volatility),
        volatility_coupling=None if not three else float(volatility_coupling or 0.0),
        meta_volatility=None if not three else float(meta_volatility or 0.0),
        initial_variance=float(initial_variance),
        initial_meta_variance=float(initial_meta_variance),
        initial_meta_belief=float(initial_meta_belief),
    )
    if path.violation >= 0:
        raise NegativePosteriorPrecision(
            "the three-level filter's posterior precision left the positive reals at trial "
            f"{path.violation}: these parameters and this observation sequence are outside "
            "the region where the filter's Gaussian approximation describes a posterior"
        )
    return BeliefTrajectory(
        belief=path.belief,
        prediction_error=values - path.belief,
        learning_rate=path.learning_rate,
        state_names=_HGF_STATES[3] if three else _HGF_STATES[2],
        states=path.states,
    )


def hgf_fixed_learning_rate(*, tonic_volatility: float, belief: float = 0.5) -> float:
    """Return the Rescorla-Wagner rate a constant-volatility binary HGF reduces to.

    With the volatility held constant the level-two step size obeys

    .. math:: \\sigma^{(k)} = \\Bigl(\\tfrac{1}{\\sigma^{(k-1)} + c} + v\\Bigr)^{-1},
        \\qquad c = e^{\\omega_2},\\ v = \\hat{\\mu}_1(1 - \\hat{\\mu}_1),

    a recursion in which no observation appears. Its unique positive fixed point solves
    :math:`v\\sigma^2 + vc\\sigma - c = 0`, so

    .. math:: \\sigma^{*} = \\frac{-vc + \\sqrt{v^2c^2 + 4vc}}{2v},

    and once :math:`\\sigma` has reached it the filter is *exactly*
    :math:`\\mu_2 \\mathrel{+}= \\sigma^{*}(u - \\hat{\\mu}_1)`: a delta rule with a fixed
    learning rate. The qualification the binary case forces, and the reason ``belief`` is an
    argument rather than an assumption, is that :math:`v` is the Bernoulli variance of the
    *current* belief, so the fixed point moves as the belief moves. It is a constant for
    every trajectory only when the observation precision is -- which is the continuous-input
    HGF, not this one.
    """

    if not np.isfinite(tonic_volatility):
        raise ValueError("tonic_volatility must be finite")
    if not np.isfinite(belief) or not 0.0 < belief < 1.0:
        raise ValueError("belief must be finite and lie strictly in (0, 1)")
    variance = float(belief) * (1.0 - float(belief))
    step = math.exp(float(tonic_volatility))
    product = variance * step
    return float((-product + math.sqrt(product * product + 4.0 * product)) / (2.0 * variance))


_BETA_BERNOULLI_STATES = ("alpha", "beta")
_HGF_STATES = {
    2: ("tendency", "tendency_precision"),
    3: ("tendency", "tendency_precision", "log_volatility", "volatility_precision"),
}


def _beta_bernoulli_path(
    observations: list[float], *, retention: float, prior_mean: float, prior_strength: float
) -> _Path:
    """Run the discounted-count recursion in Python floats, one trial at a time."""

    count = len(observations)
    belief = np.empty(count, dtype=np.float64)
    rate = np.empty(count, dtype=np.float64)
    states = np.empty((count, 2), dtype=np.float64)
    prior_alpha = prior_strength * prior_mean
    prior_beta = prior_strength - prior_alpha
    alpha = prior_alpha
    beta = prior_beta
    leak = 1.0 - retention
    for index in range(count):
        total = alpha + beta
        belief[index] = alpha / total
        observation = observations[index]
        alpha = retention * alpha + leak * prior_alpha + observation
        beta = retention * beta + leak * prior_beta + (1.0 - observation)
        rate[index] = 1.0 / (alpha + beta)
        states[index, 0] = alpha
        states[index, 1] = beta
    return _Path(
        belief=belief,
        learning_rate=rate,
        states=states,
        violation=-1,
        minimum_precision=math.inf,
    )


def _hgf_path(
    observations: list[float],
    *,
    initial_belief: float,
    tonic_volatility: float,
    volatility_coupling: float | None,
    meta_volatility: float | None,
    initial_variance: float,
    initial_meta_variance: float,
    initial_meta_belief: float,
) -> _Path:
    """Run the binary HGF forward, in Python floats, exactly as the module docstring states."""

    count = len(observations)
    three = volatility_coupling is not None
    width = 4 if three else 2
    belief = np.full(count, np.nan, dtype=np.float64)
    rate = np.full(count, np.nan, dtype=np.float64)
    states = np.full((count, width), np.nan, dtype=np.float64)
    tendency = initial_belief
    tendency_variance = initial_variance
    log_volatility = initial_meta_belief
    volatility_variance = initial_meta_variance
    coupling = float(volatility_coupling) if three else 0.0
    meta = float(meta_volatility) if three else 0.0
    minimum = math.inf
    violation = -1
    for index in range(count):
        exponent = coupling * log_volatility + tonic_volatility if three else tonic_volatility
        if not math.isfinite(exponent) or exponent > _EXPONENT_CEILING:
            violation = index
            break
        step_variance = math.exp(exponent)
        prediction_variance = tendency_variance + step_variance
        prediction_precision = 1.0 / prediction_variance
        expected = _sigmoid(tendency)
        belief[index] = expected
        error = observations[index] - expected
        posterior_precision = prediction_precision + expected * (1.0 - expected)
        step = 1.0 / posterior_precision
        updated_tendency = tendency + step * error
        rate[index] = step
        if three:
            change = updated_tendency - tendency
            volatility_error = (step + change * change) * prediction_precision - 1.0
            meta_prediction_precision = 1.0 / (volatility_variance + math.exp(meta))
            weight = step_variance * prediction_precision
            curvature = weight * (weight + (2.0 * weight - 1.0) * volatility_error)
            volatility_precision = meta_prediction_precision + 0.5 * coupling * coupling * curvature
            minimum = min(minimum, volatility_precision)
            if not math.isfinite(volatility_precision) or volatility_precision <= 0.0:
                violation = index
                break
            volatility_variance = 1.0 / volatility_precision
            log_volatility += 0.5 * volatility_variance * coupling * weight * volatility_error
            states[index, 2] = log_volatility
            states[index, 3] = volatility_precision
        tendency = updated_tendency
        tendency_variance = step
        states[index, 0] = tendency
        states[index, 1] = posterior_precision
        if not (math.isfinite(tendency) and math.isfinite(tendency_variance)):
            violation = index
            break
    return _Path(
        belief=belief,
        learning_rate=rate,
        states=states,
        violation=violation,
        minimum_precision=minimum,
    )


def _sigmoid(value: float) -> float:
    """Return the logistic function without overflowing on either tail."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value)) if value < _EXPONENT_CEILING else 1.0
    if value <= -_EXPONENT_CEILING:
        return 0.0
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


# --------------------------------------------------------------------------------------
# Natural-scale parameter records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BetaBernoulliParameters:
    """Natural-scale ideal-observer parameters in the units a report uses."""

    retention: float
    prior_mean: float
    prior_strength: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.retention) or not 0.0 < self.retention < 1.0:
            raise ValueError("retention must lie strictly between zero and one")
        if not np.isfinite(self.prior_mean) or not 0.0 < self.prior_mean < 1.0:
            raise ValueError("prior_mean must lie strictly between zero and one")
        if not np.isfinite(self.prior_strength) or self.prior_strength <= 0.0:
            raise ValueError("prior_strength must be finite and positive")

    @property
    def effective_memory(self) -> float:
        """The number of trials the discounted counts effectively average over."""

        return 1.0 / (1.0 - self.retention)

    @property
    def asymptotic_learning_rate(self) -> float:
        """The delta-rule rate this observer converges to, :math:`1/(\\nu + 1/(1-\\rho))`."""

        return 1.0 / (self.prior_strength + self.effective_memory)


@dataclass(frozen=True, slots=True)
class HierarchicalGaussianFilterParameters:
    """Natural-scale HGF parameters, in Mathys et al.'s own symbols.

    ``initial_belief`` is :math:`\\mu_2^{(0)}` on the **logit** scale, so
    ``expit(initial_belief)`` is the observer's initial probability that the observation is
    one. ``tonic_volatility`` is :math:`\\omega_2` and ``meta_volatility`` is
    :math:`\\omega_3`; both are log variances and so are reported unchanged.
    ``volatility_coupling`` is :math:`\\kappa_2 > 0`. The last two are absent for a two-level
    filter.
    """

    initial_belief: float
    tonic_volatility: float
    volatility_coupling: float | None = None
    meta_volatility: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_belief):
            raise ValueError("initial_belief must be finite")
        if not np.isfinite(self.tonic_volatility):
            raise ValueError("tonic_volatility must be finite")
        _validated_levels(self.volatility_coupling, self.meta_volatility)
        if self.volatility_coupling is not None and (
            not np.isfinite(self.volatility_coupling) or self.volatility_coupling <= 0.0
        ):
            raise ValueError("volatility_coupling must be finite and positive")
        if self.meta_volatility is not None and not np.isfinite(self.meta_volatility):
            raise ValueError("meta_volatility must be finite")

    @property
    def levels(self) -> int:
        """Two or three, whichever this parameter set describes."""

        return 3 if self.volatility_coupling is not None else 2


@dataclass(frozen=True, slots=True)
class BeliefFitResult(FitResult):
    """A belief-updating fit that retains every deterministic restart.

    The four restart fields are the :class:`~behavio.contracts.MultistartFit` contract, so
    :meth:`~behavio.contracts.FitResult.audit` derives a
    :class:`~behavio.contracts.RestartAudit` from this fit without either side knowing about
    the other. They matter more here than for a family with independent rows: a volatility
    likelihood is multi-modal in the way an inference problem about an inference problem
    usually is, and agreement between restarts is the only evidence about that which fitting
    itself produces.
    """

    restart_objectives: NDArray[np.float64]
    restart_converged: NDArray[np.bool_]
    restart_messages: tuple[str, ...]
    selected_restart: int

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        objectives = protected_array(self.restart_objectives, dtype=np.float64)
        converged = protected_array(self.restart_converged, dtype=np.bool_)
        messages = tuple(self.restart_messages)
        if objectives.ndim != 1 or converged.shape != objectives.shape:
            raise ValueError("restart diagnostics must have one value per restart")
        if len(messages) != len(objectives):
            raise ValueError("restart messages must have one value per restart")
        if not 0 <= self.selected_restart < len(objectives):
            raise ValueError("selected_restart must identify one restart")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)


# --------------------------------------------------------------------------------------
# Everything both families share
# --------------------------------------------------------------------------------------


class _BeliefObserver(Describable):
    """A perceptual belief trajectory read out through a declared response model.

    ``__slots__`` is empty so a frozen, slotted model dataclass can inherit this without
    gaining an instance dictionary, exactly as :class:`Describable` is inherited elsewhere.

    A family supplies four things and gets the rest:

    ``perception_specs``
        Every perceptual parameter it *could* estimate, as :class:`_Coordinate` records:
        the reported name, the estimated name, the transform between them, the box and the
        restarts. Which of them are actually estimated is decided by ``fixed_perception``.
    ``fixed_perception``
        The subset a particular instance **declares** rather than estimates, by name and
        natural value. Fixing a parameter removes its coordinate from the model entirely --
        from ``parameter_names``, from the box, from the restarts and from every combinator
        -- rather than pinning it inside a fit, and it puts the declared value in the model
        signature so a fit cannot be read without it.
    ``state_names`` and ``block_path(observations, vector)``
        The recursion itself: one block's observations and one estimated perceptual
        coordinate in, one :class:`_Path` out.
    ``design_findings(observations, blocks)``
        The family's own pre-fit statements about a design.

    Everything else -- the Bernoulli likelihood, its exact per-row gradient, the multi-start
    solver, prediction, scoring, simulation, the natural coordinate and all thirteen
    bounded-coordinate members -- is written here once and is family-independent.

    Why fixing is a first-class thing rather than a convenience. Both families have a
    parameter the literature routinely declares because most designs cannot locate it: the
    HGF's :math:`\\kappa_2`, which Mathys et al. and the TAPAS implementation fix at one, and
    the ideal observer's Beta prior, which is washed out after a few dozen trials. Offering
    them only as estimated coordinates would make the default model one whose parameters do
    not recover; offering them only as constants would make the hazard invisible. Declaring
    them by default and freeing them by argument makes the hazard a decision with a name.
    """

    __slots__ = ()

    # -- what a family supplies ----------------------------------------------------------

    @property
    def perception_specs(self) -> tuple[_Coordinate, ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def fixed_perception(self) -> Mapping[str, float]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def state_names(self) -> tuple[str, ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    def block_path(
        self, observations: list[float], vector: NDArray[np.float64]
    ) -> _Path:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- what the family gets from them --------------------------------------------------

    @property
    def free_perception(self) -> tuple[_Coordinate, ...]:
        """The perceptual parameters this instance estimates, in declaration order."""

        declared = self.fixed_perception
        return tuple(spec for spec in self.perception_specs if spec.natural not in declared)

    @property
    def perception_names(self) -> tuple[str, ...]:
        """The reported names of the estimated perceptual parameters."""

        return tuple(spec.natural for spec in self.free_perception)

    @property
    def perception_coordinates(self) -> tuple[str, ...]:
        """The estimated names of the estimated perceptual parameters."""

        return tuple(spec.estimated for spec in self.free_perception)

    def perception_box(self) -> NDArray[np.float64]:
        """Return the finite box the estimated perceptual coordinate is searched in."""

        bounds = [spec.bounds for spec in self.free_perception]
        return np.asarray(bounds, dtype=np.float64).reshape(len(bounds), 2)

    def perception_starts(self) -> tuple[NDArray[np.float64], ...]:
        """Return the deterministic starting vectors for the perceptual coordinate."""

        free = self.free_perception
        return tuple(
            np.asarray([spec.starts[index] for spec in free], dtype=np.float64)
            for index in range(_RESTART_POINTS)
        )

    def perception_to_natural(self, vector: NDArray[np.float64]) -> tuple[float, ...]:
        """Map the estimated perceptual coordinate onto its reported values."""

        return tuple(
            spec.to_natural(float(value))
            for spec, value in zip(self.free_perception, vector, strict=True)
        )

    def perception_encode(self, natural: Mapping[str, float]) -> tuple[float, ...]:
        """Map reported perceptual values back onto the estimated coordinate."""

        return tuple(
            spec.encode(_require_finite(natural, spec.natural)) for spec in self.free_perception
        )

    def perception_jacobian(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``d natural / d estimated`` for the perceptual coordinate, which is diagonal."""

        diagonal = [
            spec.slope(float(value))
            for spec, value in zip(self.free_perception, vector, strict=True)
        ]
        return np.diag(np.asarray(diagonal, dtype=np.float64)).reshape(len(diagonal), len(diagonal))

    def natural_perception(self, vector: NDArray[np.float64]) -> dict[str, float]:
        """Return every perceptual parameter's natural value, declared ones included.

        The one place the estimated and the declared halves of a perceptual model are put
        back together, and therefore the only thing :meth:`block_path` reads: a recursion
        should not have to know which of its parameters this instance decided to fit.
        """

        values = dict(self.fixed_perception)
        values.update(zip(self.perception_names, self.perception_to_natural(vector), strict=True))
        return values

    def design_findings(
        self, observations: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...]
    ) -> tuple[ModelFinding, ...]:
        """The family's own design-degeneracy findings."""

        del observations, blocks
        return ()

    # -- declarations a combinator reads -------------------------------------------------

    @property
    def penalised_linear_refusal(self) -> str:
        """Why the penalised-linear route cannot be applied to this family.

        Declared rather than left to structural typing, following the precedent
        :class:`~behavio.models.rl.BinaryRLAgent` and
        :class:`~behavio.models.economic.TemporalDiscounting` set, because "this object has no
        ``design_matrix``" is a fact about members and the reader wants a fact about the
        model.

        The obstacle is a *filter*, not a recursion in the likelihood. A belief trajectory is
        a nonlinear function of the whole earlier observation sequence, so there is no design
        matrix a combinator could widen -- but the observations are exogenous, so each row
        still has its own density and all three combinators reach this family through
        :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator`, the mixture included.
        """

        return (
            "a normative observer filters an observation sequence into a belief before any "
            "response is emitted, so the belief on a trial is a nonlinear function of every "
            "earlier observation and there is no design matrix and no linear predictor a "
            "combinator could widen; it composes through "
            "behavio.contracts.bounded.BoundedCoordinateEstimator instead"
        )

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        """One observed number per row -- a response -- so the scalar case, spelled out."""

        return ()

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.observation,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    # -- the estimated and reported coordinates ------------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The estimated coordinate: the perceptual parameters, then the response model's."""

        return (*self.perception_coordinates, *self.response.parameter_names)

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate, in the same order."""

        return (*self.perception_names, *self.response.natural_names)

    @property
    def n_perception(self) -> int:
        """How many leading columns of a coordinate belong to the perceptual model."""

        return len(self.perception_coordinates)

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Map one estimated coordinate onto the reported one."""

        vector = self._vector(estimates)
        width = self.n_perception
        values = (
            *self.perception_to_natural(vector[:width]),
            *self.response.to_natural(vector[width:]),
        )
        return MappingProxyType(dict(zip(self.natural_names, values, strict=True)))

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Map a complete natural mapping back onto the estimated coordinate."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        values = (*self.perception_encode(natural), *self.response.encode(natural))
        return MappingProxyType(dict(zip(self.parameter_names, values, strict=True)))

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return ``d natural / d estimated``, which is block diagonal by construction.

        The perceptual and response coordinates are reparameterised independently -- that is
        what makes them separate components -- so the delta method sees two blocks and no
        cross terms. Any correlation between a volatility and a decision noise is in the
        estimated covariance, where it belongs, not in this map.
        """

        vector = self._vector(estimates)
        width = self.n_perception
        jacobian = np.zeros((len(vector), len(vector)), dtype=np.float64)
        jacobian[:width, :width] = self.perception_jacobian(vector[:width])
        jacobian[width:, width:] = self.response.jacobian(vector[width:])
        return jacobian

    def parameters_from_components(self, **natural: float) -> Mapping[str, float]:
        """Validate and encode one exact natural-parameter mapping for this assembly."""

        if set(natural) != set(self.natural_names):
            raise ValueError(
                "natural parameters must match the assembled model exactly; "
                f"missing={sorted(set(self.natural_names) - set(natural))}, "
                f"extra={sorted(set(natural) - set(self.natural_names))}"
            )
        return self.from_natural(natural)

    def parameter_correlation(self, fit: FitResult, first: str, second: str) -> float:
        """Return the Wald correlation between two estimated coordinates of a fit.

        The post-fit face of every trade-off this family has. The value is read off the
        estimated covariance rather than recomputed, so it is the same curvature the standard
        errors came from, and a magnitude near one means the pair is only *jointly*
        identified and neither number should be reported alone.
        """

        self._validate_fit(fit)
        names = self.parameter_names
        for name in (first, second):
            if name not in names:
                raise ValueError(f"{name!r} is not an estimated coordinate of this model")
        covariance = np.asarray(fit.covariance, dtype=np.float64)
        row = names.index(first)
        column = names.index(second)
        product = float(covariance[row, row] * covariance[column, column])
        if not np.isfinite(product) or product <= 0.0:
            return float("nan")
        return float(covariance[row, column] / math.sqrt(product))

    # -- the filter, and its sensitivity to the perceptual coordinate --------------------

    def belief_trajectory(
        self, study: Study, parameters: Mapping[str, float] | FitResult
    ) -> BeliefTrajectory:
        """Return the filtered belief this study's observations imply, in source row order.

        Filtered, never smoothed: trial ``k``'s belief is formed from observations strictly
        before ``k``, which is what makes it a one-step-ahead prediction and what lets
        :meth:`pointwise_log_prob` score a response the belief has not seen.
        """

        vector = self._coordinate(parameters)[: self.n_perception]
        observations = self._observations(study)
        belief = np.full(len(study), np.nan, dtype=np.float64)
        rate = np.full(len(study), np.nan, dtype=np.float64)
        states = np.full((len(study), len(self.state_names)), np.nan, dtype=np.float64)
        minimum = math.inf
        for indices in ordered_session_indices(study):
            index = np.asarray(indices, dtype=np.intp)
            path = self.block_path(observations[index].tolist(), vector)
            minimum = min(minimum, path.minimum_precision)
            if path.violation >= 0:
                self._require_admissible(
                    VolatilityStability(
                        admissible=False,
                        first_violation=int(index[path.violation]),
                        minimum_precision=float(minimum),
                    )
                )
            belief[index] = path.belief
            rate[index] = path.learning_rate
            states[index] = path.states
        return BeliefTrajectory(
            belief=belief,
            prediction_error=observations - belief,
            learning_rate=rate,
            state_names=self.state_names,
            states=states,
        )

    def volatility_stability(
        self, study: Study, parameters: Mapping[str, float] | FitResult
    ) -> VolatilityStability:
        """Report whether one parameter vector keeps this study's precisions positive.

        The question :class:`NegativePosteriorPrecision` answers by raising, asked without
        raising, so a caller sweeping a grid of candidate volatilities can see where the
        admissible region ends rather than discovering it one exception at a time.
        """

        vector = self._coordinate(parameters)
        observations = self._observations(study)
        blocks = ordered_session_indices(study)
        rows = np.tile(vector[: self.n_perception], (len(study), 1))
        _, _, stability = self._filter(observations, blocks, rows, sensitivity=False)
        return stability

    def _filter(
        self,
        observations: NDArray[np.float64],
        blocks: tuple[tuple[int, ...], ...],
        perception: NDArray[np.float64],
        *,
        sensitivity: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], VolatilityStability]:
        """Return each row's belief, its derivative in the perceptual coordinate, and stability.

        The derivative is central-differenced through the recursion rather than propagated
        analytically, and it is differenced **once per block** while being read **per row**:
        one extra pass over a block gives every one of that block's rows its own
        :math:`\\partial\\hat{\\mu}_k/\\partial\\vartheta_j`, because the trajectory is what
        the pass returns. That is what keeps the gradient genuinely per-row, which
        :func:`~behavio.compose.mix` requires -- a mixture scales each row's gradient by that
        row's responsibility, so a gradient smeared evenly over a block would be wrong there
        and right nowhere.

        Analytic forward-mode differentiation would be faster and is not used. The filter's
        conventions are the thing this module is on the hook for; a hand-written Jacobian
        through them is a second place for a sign to be wrong, and it would be checked against
        the difference this computes anyway.
        """

        width = self.n_perception
        count = len(observations)
        belief = np.full(count, np.nan, dtype=np.float64)
        derivative = np.zeros((count, width), dtype=np.float64)
        first_violation: int | None = None
        minimum = math.inf
        for indices in blocks:
            index = np.asarray(indices, dtype=np.intp)
            vector = perception[index[0]]
            values = observations[index].tolist()
            path = self.block_path(values, vector)
            minimum = min(minimum, path.minimum_precision)
            if path.violation >= 0:
                if first_violation is None:
                    first_violation = int(index[path.violation])
                continue
            belief[index] = path.belief
            if not sensitivity:
                continue
            for column in range(width):
                derivative[index, column] = self._column_sensitivity(values, vector, path, column)
        stability = VolatilityStability(
            admissible=first_violation is None,
            first_violation=first_violation,
            minimum_precision=float(minimum),
        )
        return belief, derivative, stability

    def _column_sensitivity(
        self,
        observations: list[float],
        vector: NDArray[np.float64],
        path: _Path,
        column: int,
    ) -> NDArray[np.float64]:
        """Return one block's ``d belief / d vector[column]``, one value per trial.

        Central where both perturbations stay admissible, one-sided where only one does, and
        -- after one attempt at a ten-times smaller step -- zero where neither does. The last
        case is an admissible point sitting within a difference step of the boundary. Zero is
        the honest answer there: it stalls that coordinate rather than reporting a slope
        differenced across a region the model does not describe, and the margin that produced
        it reaches the fit as ``minimum_volatility_precision`` and as a boundary estimate.
        """

        base = 1e-5 * (1.0 + abs(float(vector[column])))
        for step in (base, 0.1 * base):
            upper = np.array(vector, dtype=np.float64)
            upper[column] += step
            lower = np.array(vector, dtype=np.float64)
            lower[column] -= step
            raised = self.block_path(observations, upper)
            lowered = self.block_path(observations, lower)
            if raised.violation < 0 and lowered.violation < 0:
                return (raised.belief - lowered.belief) / (2.0 * step)
            if raised.violation < 0:
                return (raised.belief - path.belief) / step
            if lowered.violation < 0:
                return (path.belief - lowered.belief) / step
        return np.zeros_like(path.belief)

    # -- likelihood ----------------------------------------------------------------------

    def response_probability(
        self, study: Study, parameters: Mapping[str, float] | FitResult
    ) -> NDArray[np.float64]:
        """Return ``P(response = 1)`` on every row of ``study``."""

        vector = self._coordinate(parameters)
        rows = np.tile(vector, (len(study), 1))
        linear, _, _, _, stability = self._predictor(study, rows)
        self._require_admissible(stability)
        return np.asarray(expit(linear), dtype=np.float64)

    def _predictor(
        self, study: Study, rows: NDArray[np.float64], *, sensitivity: bool = False
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        VolatilityStability,
    ]:
        """Return the linear predictor and everything its gradient needs, per row."""

        width = self.n_perception
        observations = self._observations(study)
        blocks = ordered_session_indices(study)
        belief, derivative, stability = self._filter(
            observations, blocks, rows[:, :width], sensitivity=sensitivity
        )
        if not stability.admissible:
            empty = np.zeros(len(study), dtype=np.float64)
            return empty, np.zeros_like(rows), empty, derivative, stability
        floored = np.clip(belief, BELIEF_FLOOR, 1.0 - BELIEF_FLOOR)
        linear, jacobian, slope = self.response.linear_predictor(floored, rows[:, width:])
        inside = (belief > BELIEF_FLOOR) & (belief < 1.0 - BELIEF_FLOOR)
        return linear, jacobian, np.where(inside, slope, 0.0), derivative, stability

    def _rows_value_and_gradient(
        self, study: Study, outcomes: NDArray[np.float64], rows: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in one coordinate per row.

        Exact in the response coordinate and exact in the chain rule; the only numerical
        step is the belief's own sensitivity, which :meth:`_filter` differences.

        An inadmissible parameter vector scores :data:`VIOLATION_PENALTY` per row rather than
        infinity, and that is not a detail. L-BFGS-B's first trial point is the full steepest
        descent step, which for a volatility likelihood routinely lands outside the region
        where the third level's precision is positive; an infinite value there produces a NaN
        in the line search's interpolation and the solver returns the starting vector while
        reporting convergence. A finite barrier several orders of magnitude above any
        attainable log likelihood makes the line search back out of the region instead, and
        makes "this optimum is inadmissible" a *readable* objective rather than a silent
        non-fit. :meth:`fit` and :meth:`fit_rows` both refuse a result that scores above it.
        """

        width = self.n_perception
        linear, jacobian, slope, derivative, stability = self._predictor(
            study, rows, sensitivity=True
        )
        if not stability.admissible:
            return VIOLATION_PENALTY * len(rows), np.zeros_like(rows)
        residual = expit(linear) - outcomes
        loss = float(np.sum(np.logaddexp(0.0, linear) - outcomes * linear))
        gradient = np.empty_like(rows)
        gradient[:, :width] = (residual * slope)[:, None] * derivative
        gradient[:, width:] = residual[:, None] * jacobian
        return loss, gradient

    def _objective(
        self, vector: NDArray[np.float64], study: Study, outcomes: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Score one shared coordinate, by broadcasting it over the rows."""

        rows = np.tile(np.asarray(vector, dtype=np.float64), (len(outcomes), 1))
        loss, gradient = self._rows_value_and_gradient(study, outcomes, rows)
        return loss, np.asarray(gradient.sum(axis=0), dtype=np.float64)

    # -- fit, predict, score, simulate ---------------------------------------------------

    def fit(self, study: Study) -> BeliefFitResult:
        """Fit the response likelihood by deterministic multi-start L-BFGS-B inside the box."""

        outcomes = self._outcomes(study)
        box = self.coordinate_box(study)
        bounds = [(float(low), float(high)) for low, high in box]
        starts = self.initial_points(study)
        results = [
            minimize(
                lambda vector: self._objective(vector, study, outcomes),
                start,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "gtol": self.tolerance,
                },
            )
            for start in starts
        ]
        objectives = np.asarray(
            [float(result.fun) if np.isfinite(result.fun) else np.inf for result in results],
            dtype=np.float64,
        )
        barrier = VIOLATION_PENALTY * len(study)
        admissible = np.flatnonzero(objectives < barrier).tolist()
        if not admissible:
            self._refuse_fit(study, results[int(np.argmin(objectives))].x)
        converged = [index for index in admissible if results[index].success]
        eligible = converged if converged else admissible
        selected = min(eligible, key=lambda index: float(objectives[index]))
        chosen = results[selected]
        estimates = np.asarray(chosen.x, dtype=np.float64)
        value, gradient = self._objective(estimates, study, outcomes)
        hessian = finite_difference_hessian(
            lambda vector: self._objective(vector, study, outcomes)[1],
            estimates,
            steps=offset_steps(estimates, scale=1e-5),
        )
        covariance = covariance_from_hessian(hessian)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        stability = self.volatility_stability(
            study, dict(zip(self.parameter_names, estimates, strict=True))
        )
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"L-BFGS-B ({len(starts)} deterministic restarts)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=float(np.linalg.cond(hessian)),
            boundary_estimate=_at_box(estimates, box) or self._near_refusal(stability),
        )
        return BeliefFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            derived=(
                *natural_quantities(self, estimates, covariance),
                *self._stability_quantities(stability),
            ),
            restart_objectives=objectives,
            restart_converged=np.asarray([bool(result.success) for result in results]),
            restart_messages=tuple(str(result.message) for result in results),
            selected_restart=selected,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return the filtered response probability under a fit."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        rows = np.tile(np.asarray(fit.estimates, dtype=np.float64), (len(study), 1))
        linear, _, _, _, stability = self._predictor(study, rows)
        self._require_admissible(stability)
        return Prediction(probability=expit(linear), linear_predictor=linear, mode=prediction_mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observed response under a fit."""

        outcomes = self._outcomes(study)
        linear = self.predict(study, fit, mode=mode).linear_predictor
        return protected_array(_bernoulli_scores(outcomes, linear), dtype=np.float64)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Draw a response on every row from the belief this design's observations imply.

        The observation column is **read**, never written. It is the task's own sequence, so
        a recovery experiment holds the environment fixed and varies only what the observer
        did with it, which is exactly the design-specific evidence a recovery claim is
        supposed to be about.
        """

        probability = self.response_probability(design, parameters)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = generator.binomial(1, probability).astype(np.int8)
        return Study(columns)

    # -- the bounded-coordinate composition contract --------------------------------------
    #
    # ``parameter_names`` is already the unconstrained coordinate, so hierarchy, smoothness
    # and mixture need nothing added to it. See ``behavio.contracts.bounded``.

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the observed response of each row, which is what a component scores."""

        return self._outcomes(study)

    def row_objective(self, study: Study) -> _BeliefRowObjective:
        """Return this study's negative log likelihood in one coordinate per row."""

        blocks = ordered_session_indices(study)
        reset_blocks = np.empty(len(study), dtype=np.intp)
        for block, indices in enumerate(blocks):
            reset_blocks[np.asarray(indices, dtype=np.intp)] = block
        return _BeliefRowObjective(
            model=self,
            study=study,
            outcomes=self._outcomes(study),
            reset_blocks=reset_blocks,
            n_rows=len(study),
        )

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the quadratic penalty on the coordinate, which is none.

        This is a maximum-likelihood fit inside a box; whatever regularisation it has comes
        from the box and from the transforms. Composition adds a group or roughness prior on
        top, which is the only place a penalty on this family has ever come from.
        """

        width = len(self.parameter_names)
        return np.zeros((width, width), dtype=np.float64)

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:
        """Return the finite box the transformed coordinate is searched in."""

        del study
        return np.vstack([self.perception_box(), self.response.box()])

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the deterministic restarts this model's own solver would use."""

        del study
        perception = self.perception_starts()
        response = self.response.starts()
        return tuple(
            np.concatenate([perception[index % len(perception)], response[index % len(response)]])
            for index in range(self.n_restarts)
        )

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Return the reported parameters one declared varying name stands for."""

        return (name,)

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the isotropic Gaussian precision on one group's deviation."""

        del columns
        return ridge_group_penalty(scales)

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw Gaussian deviations on the *transformed* coordinate, never the natural one.

        A subject whose tonic volatility is -6 and one whose is -3 differ by three log
        variance units, and that is already the scale on which a Gaussian deviation is
        admissible -- which is the whole reason :math:`\\omega` is estimated as itself. A
        coupling strength and a decision noise reach the same place through a logarithm.
        """

        del columns
        return ridge_group_draw(scales, groups=groups, generator=generator)

    def fit_rows(
        self,
        design: RowCoefficientDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a row-coefficient problem on this model's own optimizer settings.

        The composed route's half of the refusal :meth:`fit` makes on the single-level route.
        A combinator hands the joint coordinate to the same shared solver every
        bounded-coordinate family uses, and that solver has no way to know that this family's
        objective has an inadmissible region in it -- so the check is made here, on the way
        out, against the same wall :data:`VIOLATION_PENALTY` builds. A hierarchical fit whose
        optimum puts one subject's volatility outside the filter's assumptions is refused,
        not reported.
        """

        result = solve_row_coefficients(
            design,
            model_name=model_name,
            model_signature=model_signature,
            optimizer="L-BFGS-B",
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            boundary=self._row_boundary,
        )
        if result.diagnostics.objective >= VIOLATION_PENALTY * design.n_observations:
            raise NegativePosteriorPrecision(
                f"the composed {model_name} fit came to rest where the filter's posterior "
                "precisions are not positive, so no estimate is reported; the coordinate the "
                "combinator built cannot be scored by this model's own assumptions"
            )
        return result

    def _row_boundary(
        self, estimates: NDArray[np.float64], derived: NDArray[np.float64] | None
    ) -> bool:
        """This family's boundary convention on a composed estimate, which is the box.

        Every coordinate this family estimates has a declared finite box that says where the
        transform stops meaning anything -- a retention the design cannot distinguish from
        lossless, a decision noise that makes the response a step function of the belief.
        :func:`~behavio.models._kernels.rowfit.solve_row_coefficients` already reports a
        coordinate resting on that box, so saying so here is the answer rather than an
        omission. What it cannot see -- a third-level precision approaching zero -- is not a
        property of a coordinate at all, and is reported by ``fit`` as its own quantity.
        """

        del estimates, derived
        return False

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate responses with one parameter vector per row."""

        rows = self._row_coordinates(design, coefficients)
        linear, _, _, _, stability = self._predictor(design, rows)
        self._require_admissible(stability)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = generator.binomial(1, expit(linear)).astype(np.int8)
        return Study(columns)

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> Prediction:
        """Return the response probability under one parameter vector per row."""

        prediction_mode = self._prediction_mode(mode)
        rows = self._row_coordinates(study, coefficients)
        linear, _, _, _, stability = self._predictor(study, rows)
        self._require_admissible(stability)
        return Prediction(probability=expit(linear), linear_predictor=linear, mode=prediction_mode)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observation under one parameter vector per row."""

        rows = self._row_coordinates(study, coefficients)
        linear, _, _, _, stability = self._predictor(study, rows)
        self._require_admissible(stability)
        return protected_array(_bernoulli_scores(self._outcomes(study), linear), dtype=np.float64)

    def _row_coordinates(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Validate one coordinate per row, refusing a perceptual parameter that moves."""

        rows = validated_row_coefficients(
            coefficients,
            n_rows=len(study),
            n_parameters=len(self.parameter_names),
            what="row coefficients",
        )
        blocks = ordered_session_indices(study)
        reset_blocks = np.empty(len(study), dtype=np.intp)
        for block, indices in enumerate(blocks):
            reset_blocks[np.asarray(indices, dtype=np.intp)] = block
        block_constant_coordinates(
            rows[:, : self.n_perception], reset_blocks, what=self._perception_block_message
        )
        return rows

    @property
    def _perception_block_message(self) -> str:
        return f"{self.model_name}'s perceptual parameters"

    # -- findings ------------------------------------------------------------------------

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the designs in which this model's parameters are not located.

        Reached by ``describe(study)`` through
        :func:`~behavio.models._kernels.introspection.describe_model`, and carried through
        ``smooth()``, ``hierarchical()`` and ``mix()`` unchanged.
        """

        try:
            observations = self._observations(study)
            outcomes = self._outcomes(study)
            blocks = ordered_session_indices(study)
        except (ModelDataError, KeyError):
            return ()
        if not len(observations):
            return ()
        findings: list[ModelFinding] = []
        degenerate = [
            index
            for index, indices in enumerate(blocks)
            if len(set(observations[np.asarray(indices, dtype=np.intp)].tolist())) < 2
        ]
        if degenerate:
            findings.append(
                ModelFinding(
                    code="degenerate_observations",
                    severity=WARNING,
                    message=(
                        f"{len(degenerate)} of {len(blocks)} blocks have a constant "
                        f"observation column {self.observation!r}, so the belief saturates "
                        "and the response model has one value of its predictor to work with"
                    ),
                )
            )
        if len(set(outcomes.tolist())) < 2:
            findings.append(
                ModelFinding(
                    code="constant_response",
                    severity=WARNING,
                    message=(
                        f"every value of the response column {self.outcome!r} is the same, so "
                        "the response model's precision will be estimated at the edge of its "
                        "box whatever the belief trajectory did"
                    ),
                )
            )
        findings.extend(self.belief_sensitivity_findings(observations, blocks))
        findings.extend(self.design_findings(observations, blocks))
        return tuple(findings)

    def belief_sensitivity(
        self, observations: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...]
    ) -> Mapping[str, float]:
        """Return how far each estimated perceptual parameter can move this study's belief.

        A *measurement* rather than a heuristic, and the general form of every identifiability
        statement this module makes. The filter is run at the model's own first restart, then
        again with one coordinate displaced by :data:`SENSITIVITY_PROBE` in each direction, and
        what is reported is the Euclidean norm of the resulting change in the study's whole
        belief vector. A parameter whose probe moves that vector by less than
        :data:`BELIEF_SENSITIVITY_FLOOR` is one this study's responses cannot see, whatever
        the likelihood surface looks like, because the responses see the parameter only
        through the belief.

        It is design-specific by construction, which is the point: whether the HGF's
        :math:`\\omega_3` is identified is not a fact about the HGF, it is a fact about how
        much volatility a particular observation sequence contains. On the sequences most
        binary-outcome studies produce, with :math:`\\kappa_2` near one, the answer is that a
        factor of :math:`e` in the meta-volatility moves the belief by under a hundredth --
        which is a thing to know before fitting rather than after publishing.
        """

        free = self.free_perception
        if not free or not len(observations):
            return MappingProxyType({})
        start = self.perception_starts()[0]
        rows = np.tile(start, (len(observations), 1))
        base, _, stability = self._filter(observations, blocks, rows, sensitivity=False)
        if not stability.admissible:
            return MappingProxyType({})
        sensitivities: dict[str, float] = {}
        for column, spec in enumerate(free):
            displacement = 0.0
            for direction in (SENSITIVITY_PROBE, -SENSITIVITY_PROBE):
                probed = np.array(start, dtype=np.float64)
                probed[column] += direction
                belief, _, probe_stability = self._filter(
                    observations, blocks, np.tile(probed, (len(observations), 1)), sensitivity=False
                )
                if probe_stability.admissible:
                    difference = belief - base
                    displacement = max(displacement, float(np.sqrt(difference @ difference)))
            sensitivities[spec.natural] = displacement
        return MappingProxyType(sensitivities)

    def belief_sensitivity_findings(
        self, observations: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...]
    ) -> tuple[ModelFinding, ...]:
        """Report every estimated perceptual parameter this study's belief cannot see."""

        return tuple(
            ModelFinding(
                code="belief_insensitive_parameter",
                severity=WARNING,
                message=(
                    f"displacing {name} by {SENSITIVITY_PROBE:g} on its estimated scale moves "
                    f"this study's belief vector by {value:.4f} in norm, below the "
                    f"{BELIEF_SENSITIVITY_FLOOR:g} a response can be expected to reveal. The "
                    "responses see this parameter only through the belief, so its estimate "
                    "will be set by the restart and the box rather than by the data; declare "
                    "it, or use a design whose observations put it to work"
                ),
            )
            for name, value in self.belief_sensitivity(observations, blocks).items()
            if value < BELIEF_SENSITIVITY_FLOOR
        )

    # -- shared plumbing -------------------------------------------------------------------

    def _stability_quantities(self, stability: VolatilityStability) -> tuple[DerivedQuantity, ...]:
        """Report the margin an admissible fit succeeded by, when there is a third level."""

        if not math.isfinite(stability.minimum_precision):
            return ()
        return (
            DerivedQuantity(
                name="minimum_volatility_precision",
                value=float(stability.minimum_precision),
                standard_error=None,
                description=(
                    "smallest third-level posterior precision reached; the filter is refused "
                    "where this is not positive"
                ),
            ),
        )

    def _near_refusal(self, stability: VolatilityStability) -> bool:
        """Whether an admissible fit sits close enough to refusal to be flagged."""

        return bool(
            math.isfinite(stability.minimum_precision)
            and stability.minimum_precision <= self.minimum_precision_margin
        )

    def _refuse_fit(self, study: Study, estimates: NDArray[np.float64]) -> None:
        """Refuse a fit whose every restart came to rest outside the admissible region."""

        _, _, stability = self._filter(
            self._observations(study),
            ordered_session_indices(study),
            np.tile(np.asarray(estimates, dtype=np.float64)[: self.n_perception], (len(study), 1)),
            sensitivity=False,
        )
        raise NegativePosteriorPrecision(
            f"every {self.model_name} restart came to rest where the filter's posterior "
            f"precisions are not positive, the best of them first failing at row "
            f"{stability.first_violation}. No estimate is reported: a number obtained from "
            "outside the region the model describes is not a weaker answer than none, it is "
            "a different one"
        )

    def _require_admissible(self, stability: VolatilityStability) -> None:
        if stability.admissible:
            return
        raise NegativePosteriorPrecision(
            f"{self.model_name} left the region where its posterior precisions are positive "
            f"at row {stability.first_violation}: these parameters and this observation "
            "sequence are outside the filter's own assumptions"
        )

    def _vector(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        try:
            vector = np.asarray(estimates, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("estimates must contain finite numeric values") from None
        width = len(self.parameter_names)
        if vector.shape != (width,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"estimates must contain {width} finite values")
        return vector

    def _coordinate(self, parameters: Mapping[str, float] | FitResult) -> NDArray[np.float64]:
        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            return self._vector(parameters.estimates)
        if not isinstance(parameters, Mapping) or set(parameters) != set(self.parameter_names):
            raise ValueError("parameters must match the model exactly")
        try:
            values = [float(parameters[name]) for name in self.parameter_names]
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        return self._vector(np.asarray(values, dtype=np.float64))

    def _observations(self, study: Study) -> NDArray[np.float64]:
        values = _numeric_column(study, self.observation, role="task")
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ModelDataError(
                f"observation column {self.observation!r} must contain only zero and one"
            )
        return values

    def _outcomes(self, study: Study) -> NDArray[np.float64]:
        values = _numeric_column(study, self.outcome, role="outcome")
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ModelDataError(f"outcome column {self.outcome!r} must contain only zero and one")
        return values

    def _validate_fit(self, fit: FitResult) -> None:
        if (
            fit.model_name != self.model_name
            or fit.model_signature != self.signature
            or fit.parameter_names != self.parameter_names
        ):
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode is not PredictionMode.FILTERED:
            raise UnsupportedPredictionMode(
                f"{self.model_name} supports only filtered prediction, "
                f"not {prediction_mode.value!r}"
            )
        return prediction_mode


@dataclass(frozen=True, slots=True)
class _BeliefRowObjective:
    """The response likelihood as a function of one coordinate vector per row.

    This is the family where the two meanings of "block" come apart, and it is worth stating
    which one :attr:`row_blocks` reports and why.

    A **density** block is a set of rows whose outcomes cannot be scored separately. For a
    value-updating agent every session is one, because the value trace is written by the
    agent's own actions. Here there is no such set: the belief trajectory is written by the
    task's observations, which are exogenous, so conditional on the parameters row ``k``'s
    response has its own density and :func:`~behavio.compose.mix` may average it with a
    guess. ``row_blocks`` reports that, as ``arange(n_rows)``.

    A **coordinate** block is a set of rows that must share a parameter value. That does
    exist here, for the perceptual parameters only: a tonic volatility that changed part-way
    through a session would leave the belief trajectory unable to say which of its values
    produced which part. The check is made below, against the recursion's own subject/session
    blocks and against the perceptual columns only, so a response parameter or a mixture
    weight is free to vary from row to row while a volatility is not.
    """

    model: _BeliefObserver
    study: Study
    outcomes: NDArray[np.float64]
    reset_blocks: NDArray[np.intp]
    n_rows: int

    @property
    def n_parameters(self) -> int:
        """The width of one row's coordinate."""

        return len(self.model.parameter_names)

    @property
    def row_blocks(self) -> NDArray[np.intp]:
        """Every row has its own density, so every row is its own block."""

        return np.arange(self.n_rows, dtype=np.intp)

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the row coordinates."""

        values = validated_row_coefficients(
            rows, n_rows=self.n_rows, n_parameters=self.n_parameters, what="row coordinates"
        )
        block_constant_coordinates(
            values[:, : self.model.n_perception],
            self.reset_blocks,
            what=self.model._perception_block_message,
        )
        return self.model._rows_value_and_gradient(self.study, self.outcomes, values)


# --------------------------------------------------------------------------------------
# The Beta-Bernoulli ideal observer
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BetaBernoulliObserver(_BeliefObserver):
    """A leaky Beta-Bernoulli ideal observer of a binary outcome with a changing rate.

    The foundational normative model and the thing most papers mean by "the ideal observer":
    a conjugate Beta posterior over the outcome's rate whose counts are discounted by
    ``retention`` each trial, so the observer averages over an effective window of
    :math:`1/(1-\\rho)` trials rather than over the whole session. See
    :func:`beta_bernoulli_beliefs` for the equations, the exact static-rate limit and the
    exact Rescorla-Wagner form it takes below it.

    Three perceptual parameters -- ``retention`` :math:`\\rho`, ``prior_mean`` :math:`m` and
    ``prior_strength`` :math:`\\nu` -- of which only the first is estimated by default. The
    Beta prior is **declared** at :math:`\\text{Beta}(1, 1)`, the uniform prior of the
    textbook ideal observer, because a block of more than a few dozen trials has washed it out
    entirely: :math:`\\nu` and :math:`\\rho` both control how much one observation moves the
    belief, the asymptotic learning rate :math:`1/(\\nu + 1/(1-\\rho))` depends only on their
    sum, and a fit that estimates both reports two numbers for one. Pass ``prior_mean=None``
    or ``prior_strength=None`` to estimate them anyway -- which is the right thing to do for a
    design built to identify them, one with many short blocks -- and read
    :meth:`~_BeliefObserver.parameter_correlation` on the result.

    The estimated coordinates are ``retention_logit``, ``prior_mean_logit`` and
    ``prior_strength_log``, in that order, for whichever of the three are free. The response
    model is a separate declared component; the default softmax on the belief's value
    difference is the rule to use when the response is a bet on the next observation.
    """

    response: BeliefResponse = BeliefSoftmax()
    retention: float | None = None
    prior_mean: float | None = 0.5
    prior_strength: float | None = 2.0
    outcome: str = "choice"
    observation: str = "observation"
    n_restarts: int = 3
    max_iterations: int = 500
    tolerance: float = 1e-9
    minimum_precision_margin: float = 1e-3

    def __post_init__(self) -> None:
        _validate_columns(self.outcome, self.observation)
        if not isinstance(self.response, BeliefResponse):
            raise TypeError("response must satisfy behavio.models.belief.BeliefResponse")
        if self.retention is not None and not 0.0 < float(self.retention) < 1.0:
            raise ValueError("a declared retention must lie strictly between zero and one")
        if self.prior_mean is not None and not 0.0 < float(self.prior_mean) < 1.0:
            raise ValueError("a declared prior_mean must lie strictly between zero and one")
        if self.prior_strength is not None and float(self.prior_strength) <= 0.0:
            raise ValueError("a declared prior_strength must be positive")
        if not self.free_perception and not self.response.parameter_names:
            raise ValueError("a model with no estimated parameter cannot be fitted")
        _require_positive_integer(self.n_restarts, "n_restarts")
        _require_positive_integer(self.max_iterations, "max_iterations")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("perceptual and response components produce colliding names")

    @property
    def model_name(self) -> str:
        return "beta-bernoulli-observer"

    @property
    def signature(self) -> str:
        declared = ",".join(
            f"{name}={value:g}" for name, value in sorted(self.fixed_perception.items())
        )
        return (
            f"{self.model_name}[outcome={self.outcome};observation={self.observation};"
            f"response={self.response.signature};declared={declared or 'none'};"
            "reset=subject,session]"
        )

    @property
    def perception_specs(self) -> tuple[_Coordinate, ...]:
        """The three parameters this observer can estimate, in reporting order.

        ``retention_logit`` reaches 12, which is a retention of :math:`1 - 6\\times10^{-6}`:
        lossless memory to within anything a session of a few thousand trials could detect,
        and the honest way to report "this design cannot distinguish a leak from none" --
        as a coordinate resting on its box, which is what ``boundary_estimate`` is for.
        """

        return (
            _Coordinate("retention", "retention_logit", "logit", (-8.0, 12.0), (4.0, 2.0, 8.0)),
            _Coordinate("prior_mean", "prior_mean_logit", "logit", (-8.0, 8.0), (0.0, 0.0, 0.0)),
            _Coordinate(
                "prior_strength",
                "prior_strength_log",
                "log",
                (float(np.log(0.1)), float(np.log(200.0))),
                (float(np.log(2.0)), float(np.log(6.0)), float(np.log(1.0))),
            ),
        )

    @property
    def fixed_perception(self) -> Mapping[str, float]:
        """The perceptual parameters this instance declares rather than estimates."""

        declared = {
            "retention": self.retention,
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
        }
        return MappingProxyType(
            {name: float(value) for name, value in declared.items() if value is not None}
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return _BETA_BERNOULLI_STATES

    def block_path(self, observations: list[float], vector: NDArray[np.float64]) -> _Path:
        """Run the discounted-count recursion on one block's observations."""

        values = self.natural_perception(vector)
        return _beta_bernoulli_path(
            observations,
            retention=values["retention"],
            prior_mean=values["prior_mean"],
            prior_strength=values["prior_strength"],
        )

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> BetaBernoulliParameters:
        """Decode a coordinate into this observer's natural parameters, declared ones included."""

        values = self.natural_perception(self._coordinate(parameters)[: self.n_perception])
        return BetaBernoulliParameters(
            retention=values["retention"],
            prior_mean=values["prior_mean"],
            prior_strength=values["prior_strength"],
        )

    def design_findings(
        self, observations: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...]
    ) -> tuple[ModelFinding, ...]:
        """Report the designs in which the leak or the prior has nothing to be located by."""

        findings: list[ModelFinding] = []
        evidence = _rate_change_evidence(observations, blocks)
        if "retention" not in self.fixed_perception and (
            evidence is not None and evidence < _RATE_CHANGE_DEVIANCE
        ):
            findings.append(
                ModelFinding(
                    code="stationary_observations",
                    severity=WARNING,
                    message=(
                        f"the observation sequence is consistent with one constant rate per "
                        f"block ({evidence:.1f} standard deviations of homogeneity deviance, "
                        f"below the {_RATE_CHANGE_DEVIANCE:g} this treats as a rate change), "
                        "so there is no change for the leak to be identified by; retention "
                        "will be estimated at the lossless edge of its box and the fit is a "
                        "static Beta-Bernoulli observer under another name"
                    ),
                )
            )
        estimated_prior = [
            name for name in ("prior_mean", "prior_strength") if name not in self.fixed_perception
        ]
        if estimated_prior:
            shortest = min(len(indices) for indices in blocks)
            if shortest > _PRIOR_WASHOUT_TRIALS:
                findings.append(
                    ModelFinding(
                        code="washed_out_prior",
                        severity=WARNING,
                        message=(
                            f"{', '.join(estimated_prior)} are estimated but the shortest block "
                            f"has {shortest} trials, more than the {_PRIOR_WASHOUT_TRIALS} after "
                            "which the discounted counts have forgotten the prior entirely; the "
                            "prior is identified by the first few trials of each block and this "
                            "design has too few of them. Declare it, or shorten the blocks"
                        ),
                    )
                )
        return tuple(findings)


# --------------------------------------------------------------------------------------
# The Hierarchical Gaussian Filter
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HierarchicalGaussianFilter(_BeliefObserver):
    """The binary Hierarchical Gaussian Filter of Mathys et al. (2011, 2014).

    Two or three levels. A binary observation :math:`u`; a Gaussian belief about its tendency
    :math:`x_2` on the **logit** scale; and, at three levels, a Gaussian belief about that
    tendency's log volatility :math:`x_3`. The update equations and every convention this
    implementation fixes are in the module docstring, because that is where a reader
    comparing this against another implementation will look first, and
    :func:`hgf_beliefs` runs the same recursion from natural-scale arguments without a model.

    Coordinates: ``initial_belief`` (:math:`\\mu_2^{(0)}`, a logit) and ``tonic_volatility``
    (:math:`\\omega_2`) at two levels, plus ``volatility_coupling_log``
    (:math:`\\log\\kappa_2`) and ``meta_volatility`` (:math:`\\omega_3`) at three. Every one of
    them is a constructor argument as well: passing a number **declares** it and removes its
    coordinate from the model, passing ``None`` estimates it. The defaults follow Mathys et
    al. and the TAPAS implementation, and both of the declared ones are declared for a reason
    a design can overturn:

    :math:`\\mu_2^{(0)} = 0`
        An initial belief is identified by the first few trials of each block and by nothing
        else, so a study of one long block cannot locate it and a study of many short blocks
        can. ``initial_belief=None`` estimates it.
    :math:`\\kappa_2 = 1`
        The coupling and :math:`\\omega_3` set the third level's scale together, so most
        designs identify only their combination. ``volatility_coupling=None`` frees it and
        :meth:`coupling_volatility_correlation` reads the resulting Wald correlation off the
        fitted covariance.

    ``describe(study)`` measures the consequence rather than assuming it:
    :meth:`~_BeliefObserver.belief_sensitivity` displaces each estimated parameter and reports
    how far this study's belief vector moves, and a parameter the study cannot see is a
    finding before anything is fitted. On the observation sequences most binary-outcome
    studies produce that finding names :math:`\\omega_3`, which is a fact about how much
    volatility the sequence contains rather than about the filter.

    The response model is a separate declared component; :class:`UnitSquareSigmoid` is the
    rule Mathys et al. pair the filter with and is the one to use when the response is a bet
    on the next observation read off the belief's log odds.

    The other thing a user should expect to have to defend is the third level's posterior
    precision, which can go negative for admissible parameter values and is refused rather
    than clipped. See :class:`NegativePosteriorPrecision`.
    """

    levels: int = 3
    response: BeliefResponse = UnitSquareSigmoid()
    initial_belief: float | None = 0.0
    tonic_volatility: float | None = None
    volatility_coupling: float | None = 1.0
    meta_volatility: float | None = None
    outcome: str = "choice"
    observation: str = "observation"
    initial_variance: float = 1.0
    initial_meta_variance: float = 1.0
    initial_meta_belief: float = 0.0
    n_restarts: int = 3
    max_iterations: int = 500
    tolerance: float = 1e-9
    minimum_precision_margin: float = 1e-3
    minimum_volatility_block: int = 40

    def __post_init__(self) -> None:
        _validate_columns(self.outcome, self.observation)
        if isinstance(self.levels, bool) or self.levels not in (2, 3):
            raise ValueError("levels must be two or three")
        if not isinstance(self.response, BeliefResponse):
            raise TypeError("response must satisfy behavio.models.belief.BeliefResponse")
        if self.volatility_coupling is not None and float(self.volatility_coupling) <= 0.0:
            raise ValueError("a declared volatility_coupling must be positive")
        if self.levels == 2 and self.meta_volatility is not None:
            raise ValueError(
                "a two-level filter has no third level, so meta_volatility is neither "
                "estimated nor declared; it and volatility_coupling are ignored"
            )
        for value, label in (
            (self.initial_variance, "initial_variance"),
            (self.initial_meta_variance, "initial_meta_variance"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if not np.isfinite(self.initial_meta_belief):
            raise ValueError("initial_meta_belief must be finite")
        if not self.free_perception and not self.response.parameter_names:
            raise ValueError("a model with no estimated parameter cannot be fitted")
        _require_positive_integer(self.n_restarts, "n_restarts")
        _require_positive_integer(self.max_iterations, "max_iterations")
        _require_positive_integer(self.minimum_volatility_block, "minimum_volatility_block")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("perceptual and response components produce colliding names")

    @property
    def model_name(self) -> str:
        return f"hierarchical-gaussian-filter-{self.levels}"

    @property
    def signature(self) -> str:
        declared = ",".join(
            f"{name}={value:g}" for name, value in sorted(self.fixed_perception.items())
        )
        return (
            f"{self.model_name}[outcome={self.outcome};observation={self.observation};"
            f"response={self.response.signature};declared={declared or 'none'};"
            f"initial_variance={self.initial_variance};"
            f"initial_meta_variance={self.initial_meta_variance};"
            f"initial_meta_belief={self.initial_meta_belief};reset=subject,session]"
        )

    @property
    def three_level(self) -> bool:
        """Whether this filter carries a belief about its own volatility."""

        return self.levels == 3

    @property
    def perception_specs(self) -> tuple[_Coordinate, ...]:
        """The parameters this filter can estimate, in Mathys et al.'s own order.

        ``tonic_volatility`` runs from -12, where the step variance is :math:`6\\times10^{-6}`
        and the filter is an accumulating-precision Bayesian observer that stops learning, to
        4, where one trial's step variance is 55 log-odds units and the belief is whatever the
        last observation was. Neither end is distinguishable from its neighbourhood, which is
        what a box is for.
        """

        specs = [
            _Coordinate("initial_belief", "initial_belief", "identity", (-8.0, 8.0), (0.0,) * 3),
            _Coordinate(
                "tonic_volatility",
                "tonic_volatility",
                "identity",
                (-12.0, 4.0),
                (-3.0, -4.5, -2.0),
            ),
        ]
        if self.three_level:
            specs.append(
                _Coordinate(
                    "volatility_coupling",
                    "volatility_coupling_log",
                    "log",
                    (float(np.log(0.05)), float(np.log(5.0))),
                    (0.0, float(np.log(0.5)), float(np.log(1.4))),
                )
            )
            specs.append(
                _Coordinate(
                    "meta_volatility",
                    "meta_volatility",
                    "identity",
                    (-12.0, 2.0),
                    (-4.0, -6.0, -3.0),
                )
            )
        return tuple(specs)

    @property
    def fixed_perception(self) -> Mapping[str, float]:
        """The perceptual parameters this instance declares rather than estimates.

        ``volatility_coupling`` is declared at one by default, which is what Mathys et al. and
        the TAPAS implementation do and for the reason :meth:`design_findings` states: with
        both :math:`\\kappa_2` and :math:`\\omega_3` free the third level's scale is a ridge.
        Pass ``volatility_coupling=None`` to estimate it and read
        :meth:`coupling_volatility_correlation` on the result.
        """

        declared = {
            "initial_belief": self.initial_belief,
            "tonic_volatility": self.tonic_volatility,
        }
        if self.three_level:
            declared["volatility_coupling"] = self.volatility_coupling
            declared["meta_volatility"] = self.meta_volatility
        return MappingProxyType(
            {name: float(value) for name, value in declared.items() if value is not None}
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return _HGF_STATES[self.levels]

    def block_path(self, observations: list[float], vector: NDArray[np.float64]) -> _Path:
        """Run the filter forward on one block's observations."""

        values = self.natural_perception(vector)
        return _hgf_path(
            observations,
            initial_belief=values["initial_belief"],
            tonic_volatility=values["tonic_volatility"],
            volatility_coupling=values.get("volatility_coupling") if self.three_level else None,
            meta_volatility=values.get("meta_volatility") if self.three_level else None,
            initial_variance=float(self.initial_variance),
            initial_meta_variance=float(self.initial_meta_variance),
            initial_meta_belief=float(self.initial_meta_belief),
        )

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> HierarchicalGaussianFilterParameters:
        """Decode a coordinate into Mathys et al.'s own symbols, declared values included."""

        values = self.natural_perception(self._coordinate(parameters)[: self.n_perception])
        return HierarchicalGaussianFilterParameters(
            initial_belief=values["initial_belief"],
            tonic_volatility=values["tonic_volatility"],
            volatility_coupling=values.get("volatility_coupling"),
            meta_volatility=values.get("meta_volatility"),
        )

    def coupling_volatility_correlation(self, fit: FitResult) -> float:
        """Return the Wald correlation between the coupling and the meta-volatility.

        The named form of this family's headline identifiability hazard. Both parameters
        control how far the third level is allowed to move -- :math:`\\kappa_2` by scaling how
        much of that movement reaches level two, :math:`\\omega_3` by setting how much
        movement there is -- so a design that never puts them in tension estimates their
        product and reports two numbers for it. A magnitude near one means exactly that.
        """

        if not self.three_level:
            raise ValueError("a two-level filter has no coupling to correlate")
        return self.parameter_correlation(fit, "volatility_coupling_log", "meta_volatility")

    def design_findings(
        self, observations: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...]
    ) -> tuple[ModelFinding, ...]:
        """Report the designs in which a volatility cannot be located."""

        if not self.three_level:
            return ()
        findings: list[ModelFinding] = []
        shortest = min(len(indices) for indices in blocks)
        if shortest < self.minimum_volatility_block:
            findings.append(
                ModelFinding(
                    code="short_blocks_for_volatility",
                    severity=WARNING,
                    message=(
                        f"the shortest block has {shortest} trials, fewer than the declared "
                        f"{self.minimum_volatility_block} a volatility needs to have changed "
                        "within: a third level fitted here describes the filter's prior, not "
                        "the environment"
                    ),
                )
            )
        evidence = _rate_change_evidence(observations, blocks)
        if evidence is not None and evidence < _RATE_CHANGE_DEVIANCE:
            findings.append(
                ModelFinding(
                    code="stationary_observations",
                    severity=WARNING,
                    message=(
                        f"the observation sequence is consistent with one constant rate per "
                        f"block ({evidence:.1f} standard deviations of homogeneity deviance, "
                        f"below the {_RATE_CHANGE_DEVIANCE:g} this treats as a rate change), "
                        "so the environment's tendency never moves and the third level has "
                        "nothing to track; a volatility fitted here describes the filter's "
                        "prior rather than the environment"
                    ),
                )
            )
        free = {spec.natural for spec in self.free_perception}
        if {"volatility_coupling", "meta_volatility"} <= free:
            findings.append(
                ModelFinding(
                    code="coupled_volatility_scale",
                    severity=WARNING,
                    message=(
                        "volatility_coupling and meta_volatility are both estimated. They set "
                        "the third level's scale together -- kappa scales how much of the "
                        "level-three movement reaches level two, omega_3 sets how much "
                        "movement there is -- so most designs identify their combination and "
                        "report two numbers for it. Mathys et al. and TAPAS declare kappa at "
                        "one, which is this class's default; read "
                        "coupling_volatility_correlation on any fit that does not"
                    ),
                )
            )
        return tuple(findings)


# --------------------------------------------------------------------------------------
# Shared validation
# --------------------------------------------------------------------------------------

#: Standardised deviance above which an observation sequence's rate is taken to change.
#:
#: Three null standard deviations of the homogeneity deviance, so a sequence whose rate really
#: is constant produces this finding about once in a thousand designs. The threshold is on a
#: *standardised* statistic rather than on a raw rate difference because the alternative --
#: comparing window rates directly -- cannot tell a genuine change from the binomial noise of
#: a short window, and short windows are exactly what a volatile design has.
_RATE_CHANGE_DEVIANCE = 3.0

#: Window lengths a rate is estimated in when testing an observation sequence for homogeneity.
#:
#: Several, and the statistic is the largest evidence any of them finds, because one window
#: length cannot see every rate change: a design that reverses every sixteen trials looks
#: perfectly homogeneous in windows of twenty-five, since every such window straddles a
#: reversal and averages to the same thing. Taking the maximum over four scales multiplies the
#: null's false-positive rate by four, which at three standard deviations is still under one
#: design in two hundred.
_HOMOGENEITY_WINDOWS = (8, 16, 32, 64)

#: Block length past which a Beta prior no longer measurably influences the belief.
_PRIOR_WASHOUT_TRIALS = 60


def _rate_change_evidence(
    observations: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...]
) -> float | None:
    """Return the evidence that the observation rate changes within a block, standardised.

    A likelihood-ratio test of rate homogeneity, not a spread. Each block is cut into windows
    at each of the lengths :data:`_HOMOGENEITY_WINDOWS`, and the deviance between one Bernoulli
    rate per window and one rate per block is compared with the chi-square distribution it
    would have if the rate were constant: the value returned is the largest
    :math:`(D - \\nu)/\\sqrt{2\\nu}` over the scales, which is on the scale of standard
    deviations of that null whatever the block length and whatever the base rate.

    The reason it is a test rather than a difference of means. A design whose rate reverses
    every sixteen trials has *quarters* with nearly identical mean rates, so a spread across
    quarters calls the most volatile design imaginable stationary; and a genuinely stationary
    design cut into twenty-five-trial windows has a spread of a quarter from binomial noise
    alone, so the same statistic calls it volatile. The deviance is insensitive to both
    because it knows how much variation a constant rate produces.

    ``None`` when no block is long enough to hold two windows, which is a design this question
    cannot be asked of rather than an answer to it.
    """

    best: float | None = None
    for size in _HOMOGENEITY_WINDOWS:
        deviance = 0.0
        degrees = 0
        for indices in blocks:
            values = observations[np.asarray(indices, dtype=np.intp)]
            count = len(values) // size
            if count < 2:
                continue
            pooled = float(np.mean(values))
            for window in np.array_split(values[: count * size], count):
                deviance += _bernoulli_deviance(window, pooled)
            degrees += count - 1
        if degrees < 1:
            continue
        evidence = float((deviance - degrees) / math.sqrt(2.0 * degrees))
        best = evidence if best is None else max(best, evidence)
    return best


def _bernoulli_deviance(window: NDArray[np.float64], pooled: float) -> float:
    """Return twice the log-likelihood one window's own rate buys over the pooled one."""

    successes = float(np.sum(window))
    failures = float(len(window)) - successes
    rate = successes / len(window)
    total = 0.0
    if successes and pooled > 0.0:
        total += successes * math.log(rate / pooled)
    if failures and pooled < 1.0:
        total += failures * math.log((1.0 - rate) / (1.0 - pooled))
    return 2.0 * total


def _validated_observations(
    observations: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("observations must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("observations must be finite and lie in the unit interval")
    return values


def _validated_levels(coupling: float | None, meta: float | None) -> bool:
    if (coupling is None) != (meta is None):
        raise ValueError(
            "a three-level filter needs both volatility_coupling and meta_volatility, and a "
            "two-level filter needs neither"
        )
    return coupling is not None


def _validate_columns(outcome: str, observation: str) -> None:
    for value, label in ((outcome, "outcome"), (observation, "observation")):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty column name")
        if value in REQUIRED_COLUMNS:
            raise ValueError(f"{label} cannot replace a required Study column")
    if outcome == observation:
        raise ValueError("outcome and observation columns must be distinct")


def _numeric_column(study: Study, name: str, *, role: str) -> NDArray[np.float64]:
    if name not in study.columns:
        raise ModelDataError(f"study is missing {role} column {name!r}")
    try:
        values = np.asarray(study[name], dtype=np.float64)
    except (TypeError, ValueError):
        raise ModelDataError(f"{role} column {name!r} must be numeric") from None
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ModelDataError(f"{role} column {name!r} must be finite")
    return values


def _bernoulli_scores(
    outcomes: NDArray[np.float64], linear: NDArray[np.float64]
) -> NDArray[np.float64]:
    scores = outcomes * -np.logaddexp(0.0, -linear)
    scores += (1.0 - outcomes) * -np.logaddexp(0.0, linear)
    return np.asarray(scores, dtype=np.float64)


def _at_box(estimates: NDArray[np.float64], box: NDArray[np.float64]) -> bool:
    tolerances = BOUNDARY_TOLERANCE * np.maximum(1.0, box[:, 1] - box[:, 0])
    return bool(
        np.any(estimates - box[:, 0] <= tolerances) or np.any(box[:, 1] - estimates <= tolerances)
    )


def _require_finite(values: Mapping[str, float], name: str) -> float:
    try:
        value = float(values[name])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite numeric value") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    return value


def _require_positive(values: Mapping[str, float], name: str) -> float:
    value = _require_finite(values, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
