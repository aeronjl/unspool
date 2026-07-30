"""Economic and value-based choice: delay discounting and prospect theory.

Both families here are the same model with a different scalar utility in the middle of it.
A trial offers two options, each option has a subjective value :math:`V`, and the choice is

.. math::

    P(\\text{option 1}) = \\sigma\\!\\left(\\beta\\,(V_1 - V_0)\\right),

with an inverse temperature :math:`\\beta > 0`. Everything that is not :math:`V` -- the
softmax, the likelihood, the multi-start solver, the composition contract -- is written once
in :class:`_ValueBasedChoice` and shared, so a family costs one value function and its
derivative.

Conventions this module fixes, because published implementations disagree
------------------------------------------------------------------------
*Every coordinate is a logarithm.* A discount rate, a curvature exponent, a loss-aversion
coefficient and both probability-weighting parameters are all strictly positive and all
multiplicative in what they do, so all six are estimated as :math:`\\log` of themselves and
so is the inverse temperature. That is what makes a Wald interval on them honest (it cannot
cross zero), and it is what makes a Gaussian group deviation admissible, which is the whole
requirement :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator` imposes. The
estimated names therefore all end in ``_log`` and the reported ones do not.

*A curvature exponent is not bounded above by one.* Most reported gain exponents lie below
one, and several implementations enforce :math:`\\alpha \\in (0, 1)` by estimating a logit.
This module does not, because :math:`\\alpha > 1` is a convex value function and therefore
risk seeking for gains through curvature -- an empirical finding for individual subjects,
and one half of a prediction the fourfold pattern is supposed to be able to fail. A bound
that forbids an outcome is not a prior, it is a censored test.

*Probability weighting is Prelec (1998), two-parameter.* See
:func:`prelec_weight` for the argument and for what Tversky and Kahneman's (1992)
one-parameter form would change. One weighting function is estimated and used in both
domains; a separate :math:`(\\gamma^-, \\delta^-)` pair is not offered, and TK92's own
estimates (:math:`\\gamma^+ = 0.61`, :math:`\\gamma^- = 0.69`) are close enough that the
second pair usually buys correlated parameters rather than fit.

*Amounts are divided by a declared scale.* ``value_scale`` is a constructor argument, so it
appears in the model signature and a fit can never be read without it. It is **declared,
never learned**: derived from the study it would differ between training folds and the fitted
:math:`\\beta` would mean a different thing in each one. Its only job is units --
:math:`\\beta` is in reciprocal utility units, so a study in pennies and the same study in
pounds have :math:`\\beta` differing by a factor of :math:`100^{\\alpha}` and neither number
is comparable to a published one unless the scale is stated.

The identifiability hazard, and where it is reported
----------------------------------------------------
The inverse temperature and the value function's curvature trade off: a steeper utility with
a flatter softmax predicts nearly what a flatter utility with a sharper softmax predicts.
This is not a defect of the implementation, it is the shape of the likelihood, and there are
designs in which the trade-off is *exact* rather than merely awkward. Each of those is a
cheap pre-fit question about the design, and each is a ``describe()`` finding, following the
precedent :func:`~behavio.compose.mix` set for an unidentified mixture weight:

``unidentified_utility_curvature``
    Every non-zero outcome in the study has the same magnitude, so :math:`x^{\\alpha}` is one
    constant on every row and :math:`\\beta` absorbs it exactly.
``weakly_identified_utility_curvature``
    Two magnitudes. The exponent is identified only by how choices change between them.
``unidentified_loss_aversion``
    No trial places a gain against a loss. :math:`\\lambda` multiplies every loss value by a
    constant, so within a loss-only comparison it is :math:`\\beta` under another name.
``unidentified_probability_weighting``
    Every probability is zero or one, so :math:`w` is the identity on the observed support
    and neither weighting parameter does anything.
``unidentified_discount_rate``
    Both delays are equal on every trial, so the discount factor is a common factor of the
    value difference and, again, :math:`\\beta` absorbs it.
``value_scale_mismatch``
    The declared scale is more than two orders of magnitude away from the outcomes actually
    present, so the fitted inverse temperature will sit near the edge of its box.

None is an error. Each is a statement about the *design*, which only its author can fix, and
the post-fit half of the same hazard arrives through the diagnostics that already exist: a
coordinate resting on its box sets ``boundary_estimate``, and
:meth:`_ValueBasedChoice.temperature_scale_correlation` reads the Wald correlation between
the inverse temperature and a curvature coordinate straight off the fitted covariance.

References
----------
Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In M. L.
Commons, J. E. Mazur, J. A. Nevin, & H. Rachlin (Eds.), *Quantitative Analyses of Behavior:
The Effect of Delay and of Intervening Events on Reinforcement Value* (Vol. 5, pp. 55-73).
Erlbaum.

Kahneman, D., & Tversky, A. (1979). Prospect theory: an analysis of decision under risk.
*Econometrica, 47*(2), 263-291.

Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: cumulative representation
of uncertainty. *Journal of Risk and Uncertainty, 5*(4), 297-323.

Prelec, D. (1998). The probability weighting function. *Econometrica, 66*(3), 497-527.

Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting function.
*Cognitive Psychology, 38*(1), 129-166.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.contracts.bounded import RowCoefficientDesign, validated_row_coefficients
from behavio.contracts.compose import ridge_group_draw, ridge_group_penalty
from behavio.contracts.natural import natural_quantities
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
    "DiscountFunction",
    "DiscountingParameters",
    "ProspectTheory",
    "ProspectTheoryParameters",
    "TemporalDiscounting",
    "ValueBasedFitResult",
    "certainty_equivalent",
    "indifference_discount_rate",
    "prelec_weight",
]

#: Coordinate at which a positive parameter's logarithm is treated as resting on its bound.
BOUNDARY_TOLERANCE = 1e-4

#: Smallest probability the weighting function is evaluated at before it is treated as zero.
PROBABILITY_FLOOR = 1e-12

_TEMPERATURE_NAME = "inverse_temperature"
_TEMPERATURE_COORDINATE = "inverse_temperature_log"
_TEMPERATURE_BOX = (-12.0, 12.0)


# --------------------------------------------------------------------------------------
# The two published closed forms this module is validated against
# --------------------------------------------------------------------------------------


def indifference_discount_rate(
    sooner_amount: NDArray[np.float64] | float,
    sooner_delay: NDArray[np.float64] | float,
    later_amount: NDArray[np.float64] | float,
    later_delay: NDArray[np.float64] | float,
    *,
    discount: DiscountFunction | str = "hyperbolic",
) -> NDArray[np.float64]:
    """Return the rate at which a trial's two options have equal subjective value.

    A discounting model does not have to be fitted to be checked, because its indifference
    point is closed form. For Mazur's (1987) hyperbola
    :math:`V = A / (1 + k t)`, equating the two options gives

    .. math::

        k^{*} = \\frac{A_L - A_S}{A_S t_L - A_L t_S},

    and for the exponential :math:`V = A e^{-k t}`,
    :math:`k^{*} = \\log(A_L / A_S) / (t_L - t_S)`. At :math:`k = k^{*}` the model's choice
    probability is exactly one half **whatever the inverse temperature is**, which is the
    known answer :mod:`tests.test_economic` holds the implementation to.

    Rates that are not positive and finite -- a trial whose options cannot be equated by any
    rate, such as one where the later option is also the smaller -- are returned as ``nan``.
    """

    function = DiscountFunction(discount)
    sooner = np.asarray(sooner_amount, dtype=np.float64)
    later = np.asarray(later_amount, dtype=np.float64)
    early = np.asarray(sooner_delay, dtype=np.float64)
    late = np.asarray(later_delay, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if function is DiscountFunction.HYPERBOLIC:
            rate = (later - sooner) / (sooner * late - later * early)
        else:
            rate = np.log(later / sooner) / (late - early)
    rate = np.asarray(rate, dtype=np.float64)
    return np.where(np.isfinite(rate) & (rate > 0.0), rate, np.nan)


def prelec_weight(
    probability: NDArray[np.float64] | float,
    *,
    curvature: float,
    elevation: float = 1.0,
) -> NDArray[np.float64]:
    """Return Prelec's (1998) two-parameter probability weight
    :math:`w(p) = \\exp(-\\delta(-\\log p)^{\\gamma})`.

    Why this form rather than Tversky and Kahneman's (1992)
    :math:`w(p) = p^{\\gamma} / (p^{\\gamma} + (1-p)^{\\gamma})^{1/\\gamma}`:

    1. **It separates two things that are psychologically separate.** Gonzalez and Wu (1999)
       showed that the *curvature* of the weighting function (how strongly small
       probabilities are overweighted relative to large ones) and its *elevation* (overall
       optimism) vary independently across people. TK92's single :math:`\\gamma` moves both
       at once, so "nearly linear but pessimistic" is a subject it cannot describe.
       Prelec's :math:`\\gamma` is curvature and :math:`\\delta` is elevation.
    2. **It is monotone everywhere its parameters are admissible.** TK92's form is not
       increasing on :math:`(0, 1)` for :math:`\\gamma` below about 0.28, so an optimizer that
       wanders there is fitting something that is not a weighting function at all -- and it
       will converge, and report a number. Prelec's is strictly increasing for every
       :math:`\\gamma > 0, \\delta > 0`, so the box does not have to encode a constraint that
       came from the parameterisation rather than from the data.
    3. **Its fixed point is interpretable.** :math:`w(p) = p` at
       :math:`p = \\exp(-\\delta^{1/(1-\\gamma)})`, which is :math:`1/e \\approx 0.368` when
       :math:`\\delta = 1` -- close to the crossover repeatedly reported in the data.

    What TK92 would change: one parameter instead of two, so curvature and elevation would be
    locked together and the two-magnitude designs this module warns about would become
    *harder* rather than easier to fit; the box on :math:`\\gamma` would need a floor near
    0.28 to keep the function monotone; and the fixed point would move with :math:`\\gamma`
    instead of being read off :math:`\\delta`. Nothing else in this module would change:
    ``w`` and its two derivatives are the only thing the likelihood asks the weighting
    function for.

    Fixing ``elevation`` at one recovers Prelec's compound-invariant one-parameter form.
    """

    values = np.asarray(probability, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isfinite(curvature) or curvature <= 0:
        raise ValueError("curvature must be finite and positive")
    if not np.isfinite(elevation) or elevation <= 0:
        raise ValueError("elevation must be finite and positive")
    weight, _, _ = _weight_and_derivatives(values, np.float64(curvature), np.float64(elevation))
    return weight


def _weight_and_derivatives(
    probability: NDArray[np.float64],
    curvature: NDArray[np.float64] | np.float64,
    elevation: NDArray[np.float64] | np.float64,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``w(p)`` and its derivatives with respect to ``log gamma`` and ``log delta``.

    Both derivatives are taken on the estimated coordinate directly, because the chain rule
    collapses there: :math:`\\log w = -\\delta L^{\\gamma}` with :math:`L = -\\log p`, so
    :math:`\\partial w / \\partial \\log \\delta = w \\log w` and
    :math:`\\partial w / \\partial \\log \\gamma = w \\log w \\, \\gamma \\log L`. Both vanish
    at :math:`p = 1`, where :math:`\\log w = 0`, and at :math:`p = 0`, where :math:`w = 0`.
    """

    inside = probability > PROBABILITY_FLOOR
    safe = np.where(inside, np.maximum(probability, PROBABILITY_FLOOR), 1.0)
    logarithm = -np.log(safe)
    powered = np.power(logarithm, curvature)
    log_weight = -elevation * powered
    weight = np.where(inside, np.exp(log_weight), 0.0)
    elevation_derivative = np.where(inside, weight * log_weight, 0.0)
    positive = inside & (logarithm > 0.0)
    curvature_derivative = np.where(
        positive,
        weight * log_weight * curvature * np.log(np.where(positive, logarithm, 1.0)),
        0.0,
    )
    return weight, curvature_derivative, elevation_derivative


def certainty_equivalent(
    outcome: NDArray[np.float64] | float,
    probability: NDArray[np.float64] | float,
    parameters: ProspectTheoryParameters,
) -> NDArray[np.float64]:
    """Return the sure amount worth as much as the gamble ``(outcome, probability; 0)``.

    Closed form, because the value function is a power law and the zero branch contributes
    nothing: :math:`c = x\\, w(p)^{1/\\alpha}` for a gain and :math:`c = x\\, w(p)^{1/\\beta}`
    for a loss. Loss aversion cancels within a domain, which is exactly why
    :math:`\\lambda` is identified only by trials that place a gain against a loss.

    This is the quantity the risk literature reports and the quantity the **fourfold pattern**
    is a statement about. Risk seeking is :math:`c > px` in both domains, so with a gain the
    condition is :math:`w(p)^{1/\\alpha} > p` and with a loss it reverses. Under Tversky and
    Kahneman's (1992) median parameters the four cells come out risk averse for likely gains,
    risk seeking for unlikely gains, risk seeking for likely losses and risk averse for
    unlikely losses -- with no fitting involved, which is what makes it a check rather than a
    self-consistency statement.
    """

    if not isinstance(parameters, ProspectTheoryParameters):
        raise TypeError("parameters must be ProspectTheoryParameters")
    amounts = np.asarray(outcome, dtype=np.float64)
    weights = prelec_weight(
        probability,
        curvature=parameters.weighting_curvature,
        elevation=parameters.weighting_elevation,
    )
    exponent = np.where(amounts >= 0.0, parameters.gain_exponent, parameters.loss_exponent)
    return np.asarray(amounts * np.power(weights, 1.0 / exponent), dtype=np.float64)


# --------------------------------------------------------------------------------------
# Natural-scale parameter records
# --------------------------------------------------------------------------------------


class DiscountFunction(StrEnum):
    """The shape of the delay discount factor."""

    HYPERBOLIC = "hyperbolic"
    """Mazur (1987): :math:`D(t) = 1 / (1 + k t)`."""

    EXPONENTIAL = "exponential"
    """Constant-rate discounting: :math:`D(t) = e^{-k t}`."""


@dataclass(frozen=True, slots=True)
class DiscountingParameters:
    """Natural-scale delay-discounting parameters in the units a report uses."""

    discount_rate: float
    inverse_temperature: float

    def __post_init__(self) -> None:
        _require_positive(self.discount_rate, "discount_rate")
        _require_positive(self.inverse_temperature, "inverse_temperature")


@dataclass(frozen=True, slots=True)
class ProspectTheoryParameters:
    """Natural-scale prospect-theory parameters.

    ``gain_exponent`` and ``loss_exponent`` are the curvature of the value function in each
    domain, ``loss_aversion`` is the multiplier applied to the loss branch, and the two
    weighting parameters are Prelec's curvature and elevation. Every one of them is strictly
    positive and none of them is bounded above; see the module docstring for why.
    """

    gain_exponent: float
    loss_exponent: float
    loss_aversion: float
    weighting_curvature: float
    weighting_elevation: float
    inverse_temperature: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.gain_exponent, "gain_exponent"),
            (self.loss_exponent, "loss_exponent"),
            (self.loss_aversion, "loss_aversion"),
            (self.weighting_curvature, "weighting_curvature"),
            (self.weighting_elevation, "weighting_elevation"),
            (self.inverse_temperature, "inverse_temperature"),
        ):
            _require_positive(value, label)


@dataclass(frozen=True, slots=True)
class ValueBasedFitResult(FitResult):
    """A value-based choice fit that retains every deterministic restart.

    The four restart fields are the :class:`~behavio.contracts.MultistartFit` contract, so
    :meth:`~behavio.contracts.FitResult.audit` derives a
    :class:`~behavio.contracts.RestartAudit` from this fit without either side knowing about
    the other. They are retained for the same reason the other optimizer-backed families
    retain them: a multi-start solver's agreement between restarts is evidence that only
    fitting produced, and a utility-times-softmax likelihood is exactly the shape that has
    more than one basin when the design is thin.
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


class _ValueBasedChoice(Describable):
    """A binary softmax over two options' subjective values.

    ``__slots__`` is empty so a frozen, slotted model dataclass can inherit this without
    gaining an instance dictionary, exactly as :class:`Describable` is inherited elsewhere.

    A family supplies four things and gets the rest:

    ``utility_names`` / ``utility_coordinates``
        The natural and estimated names of the parameters its value function has, in order.
        Everything downstream appends the inverse temperature to them.
    ``read_options(study)``
        The arrays its value function reads, validated once.
    ``option_values(options, rows)``
        The two options' values and the derivative of each with respect to each **estimated**
        utility coordinate: ``(n, 2)`` and ``(n, 2, utility parameters)``.
    ``utility_box(options)`` and ``utility_starts(options)``
        Where the utility coordinate is searched, and from where.

    That split is the whole claim the audit made about this family being cheap: the softmax,
    the likelihood, its gradient, the solver, the row objective, prediction, scoring,
    simulation, the natural coordinate and all nine bounded-coordinate members are written
    here once and are family-independent.
    """

    __slots__ = ()

    @property
    def penalised_linear_refusal(self) -> str:
        """Why ``mix()`` and the penalised-linear route cannot be applied to this family.

        Declared rather than left to structural typing, following the precedent
        :class:`~behavio.models.rl.BinaryRLAgent` and
        :class:`~behavio.models.glm_hmm.BernoulliGLMHMM` set, because "this object has no
        ``design_matrix``" is a fact about members and the reader wants a fact about the
        model.

        The obstacle here is **not** the one the agents have. A value-based trial's rows *are*
        independent -- that is why ``smooth(model, over="trial")`` is admissible for this
        family and an error for a Q-learning agent. What is missing is a **linear predictor**:
        the utility is a power law and a hyperbola in the parameters, so there is no design
        matrix a combinator could widen and no linear predictor a mixture could average two
        densities over. A lapse on a discounting or prospect-theory model is a perfectly
        well-defined thing to want, and :func:`~behavio.compose.mix` cannot express it,
        because ``mix()`` is gated on the penalised-linear contract rather than on row
        independence. See ``docs/economic-choice.md``.
        """

        return (
            "a value-based choice model applies a nonlinear utility to each option before "
            "the softmax, so it has no design matrix and no linear predictor; its rows are "
            "independent, so it composes through "
            "behavio.contracts.bounded.BoundedCoordinateEstimator instead"
        )

    # -- what a family supplies ----------------------------------------------------------

    @property
    def utility_names(self) -> tuple[str, ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def utility_coordinates(self) -> tuple[str, ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    def read_options(self, study: Study) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def option_values(
        self, options: Any, rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def utility_box(self, options: Any) -> NDArray[np.float64]:  # pragma: no cover - abstract
        raise NotImplementedError

    def utility_starts(
        self, options: Any
    ) -> tuple[NDArray[np.float64], ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- the estimator surface -----------------------------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The estimated coordinate: the utility logarithms, then the temperature's."""

        return (*self.utility_coordinates, _TEMPERATURE_COORDINATE)

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate, which is the estimated one exponentiated."""

        return (*self.utility_names, _TEMPERATURE_NAME)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Exponentiate the estimated coordinate onto the reported one."""

        vector = self._vector(estimates)
        return MappingProxyType(
            {
                name: float(np.exp(value))
                for name, value in zip(self.natural_names, vector, strict=True)
            }
        )

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Take the logarithm of a complete natural mapping, refusing a non-positive value."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        values: dict[str, float] = {}
        for name, coordinate in zip(self.natural_names, self.parameter_names, strict=True):
            _require_positive(float(natural[name]), name)
            values[coordinate] = float(np.log(float(natural[name])))
        return MappingProxyType(values)

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return ``d natural / d estimated``, which is diagonal because every map is ``exp``.

        A model whose whole coordinate is one transform is the easy case of
        :class:`~behavio.contracts.natural.NaturalParameterisation`, and it is the case this
        family is in on purpose: the transform was chosen per parameter for its own reason
        and every one of those reasons came out the same.
        """

        return np.diag(np.exp(self._vector(estimates)))

    def temperature_scale_correlation(self, fit: FitResult, *, parameter: str) -> float:
        """Return the Wald correlation between ``parameter`` and the inverse temperature.

        The post-fit face of the hazard the module docstring names. A curvature coordinate
        and ``inverse_temperature_log`` are estimated from the same data and pull on the same
        prediction, so their estimates are correlated; a magnitude near one means the pair is
        only *jointly* identified and neither number should be reported alone. The value is
        read off the estimated covariance rather than recomputed, so it is the same curvature
        the standard errors came from.
        """

        self._validate_fit(fit)
        names = self.parameter_names
        if parameter not in names or parameter == _TEMPERATURE_COORDINATE:
            raise ValueError(f"{parameter!r} is not an estimated utility coordinate of this model")
        covariance = np.asarray(fit.covariance, dtype=np.float64)
        first = names.index(parameter)
        second = names.index(_TEMPERATURE_COORDINATE)
        scale = float(np.sqrt(covariance[first, first] * covariance[second, second]))
        if not np.isfinite(scale) or scale <= 0.0:
            return float("nan")
        return float(covariance[first, second] / scale)

    # -- likelihood ----------------------------------------------------------------------

    def choice_probability(
        self, study: Study, parameters: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """Return ``P(option 1)`` on every row of ``study``."""

        vector = self._coordinate(parameters)
        options = self.read_options(study)
        rows = np.tile(vector, (self._n_rows(options), 1))
        return expit(self._utilities(options, rows)[0])

    def subjective_values(
        self, study: Study, parameters: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """Return the ``(rows, 2)`` subjective value of each option, in utility units."""

        vector = self._coordinate(parameters)
        options = self.read_options(study)
        rows = np.tile(vector, (self._n_rows(options), 1))
        values, _ = self.option_values(options, rows[:, : len(self.utility_coordinates)])
        return np.asarray(values, dtype=np.float64)

    def _utilities(
        self, options: Any, rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the linear predictor, the value difference and the utility jacobian."""

        width = len(self.utility_coordinates)
        values, jacobian = self.option_values(options, rows[:, :width])
        temperature = np.exp(rows[:, width])
        difference = values[:, 1] - values[:, 0]
        return temperature * difference, difference, jacobian

    def _rows_value_and_gradient(
        self, options: Any, outcomes: NDArray[np.float64], rows: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in one coordinate per row.

        The only arithmetic in this module that a family does not own, and the reason a
        family owns so little: ``linear = beta * (V1 - V0)`` makes the temperature's
        derivative ``residual * linear`` whatever the value function was, and every utility
        coordinate's derivative ``residual * beta * (dV1 - dV0)``.
        """

        width = len(self.utility_coordinates)
        linear, _, jacobian = self._utilities(options, rows)
        residual = expit(linear) - outcomes
        loss = float(np.sum(np.logaddexp(0.0, linear) - outcomes * linear))
        gradient = np.zeros_like(rows)
        temperature = np.exp(rows[:, width])
        gradient[:, :width] = (residual * temperature)[:, None] * (jacobian[:, 1] - jacobian[:, 0])
        gradient[:, width] = residual * linear
        return loss, gradient

    def _objective(
        self, vector: NDArray[np.float64], options: Any, outcomes: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Score one shared coordinate, by broadcasting it over the rows.

        Deliberately not a second implementation. The psychometric family keeps a
        shared-coordinate objective beside its row objective because every published fit went
        through the first one and had to stay bit-identical; this family has no published fit
        to reproduce, so it has one likelihood and the shared-coordinate case is the
        row-coordinate case with a ``tile``.
        """

        rows = np.tile(np.asarray(vector, dtype=np.float64), (len(outcomes), 1))
        loss, gradient = self._rows_value_and_gradient(options, outcomes, rows)
        return loss, np.asarray(gradient.sum(axis=0), dtype=np.float64)

    # -- fit, predict, score, simulate ---------------------------------------------------

    def fit(self, study: Study) -> ValueBasedFitResult:
        """Fit the choice likelihood by deterministic multi-start L-BFGS-B inside the box."""

        options = self.read_options(study)
        outcomes = self._outcomes(study)
        box = self.coordinate_box(study)
        bounds = [(float(low), float(high)) for low, high in box]
        starts = self.initial_points(study)
        results = [
            minimize(
                lambda vector: self._objective(vector, options, outcomes),
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
        if not np.any(np.isfinite(objectives)):
            messages = "; ".join(str(result.message) for result in results)
            raise ModelDataError(
                f"all {self.model_name} restarts produced non-finite objectives: {messages}"
            )
        selected = int(np.argmin(objectives))
        chosen = results[selected]
        estimates = np.asarray(chosen.x, dtype=np.float64)
        value, gradient = self._objective(estimates, options, outcomes)
        hessian = finite_difference_hessian(
            lambda vector: self._objective(vector, options, outcomes)[1],
            estimates,
            steps=offset_steps(estimates, scale=1e-5),
        )
        covariance = covariance_from_hessian(hessian)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"L-BFGS-B ({len(starts)} deterministic restarts)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=float(np.linalg.cond(hessian)),
            boundary_estimate=_at_box(estimates, box),
        )
        return ValueBasedFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            derived=natural_quantities(self, estimates, covariance),
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
        """Return the softmax choice probability under a fit."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        estimates = np.asarray(fit.estimates, dtype=np.float64)
        options = self.read_options(study)
        rows = np.tile(estimates, (self._n_rows(options), 1))
        linear, _, _ = self._utilities(options, rows)
        return Prediction(probability=expit(linear), linear_predictor=linear, mode=prediction_mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observed choice under a fit."""

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
        """Draw a choice on every row from the softmax over the two options' values."""

        probability = self.choice_probability(design, parameters)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = generator.binomial(1, probability).astype(np.int8)
        return Study(columns)

    # -- the bounded-coordinate composition contract --------------------------------------
    #
    # ``parameter_names`` is already the unconstrained coordinate -- six logarithms and a
    # seventh -- so hierarchy and smoothness need nothing added to it. See
    # ``behavio.contracts.bounded``.

    def row_objective(self, study: Study) -> _ValueChoiceRowObjective:
        """Return this study's negative log likelihood in one coordinate per row."""

        return _ValueChoiceRowObjective(
            model=self,
            options=self.read_options(study),
            outcomes=self._outcomes(study),
            n_parameters=len(self.parameter_names),
        )

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the quadratic penalty on the coordinate, which is none.

        This is a maximum-likelihood fit inside a box; whatever regularisation it has comes
        from the box and the logarithms. Composition adds a group or roughness prior on top,
        which is the only place a penalty on this family has ever come from.
        """

        width = len(self.parameter_names)
        return np.zeros((width, width), dtype=np.float64)

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:
        """Return the finite box the transformed coordinate is searched in."""

        options = self.read_options(study)
        return np.vstack([self.utility_box(options), np.asarray([_TEMPERATURE_BOX])])

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the deterministic restarts this model's own solver would use."""

        options = self.read_options(study)
        utility = self.utility_starts(options)
        temperatures = (np.log(5.0), np.log(1.0), np.log(20.0), np.log(0.5))
        return tuple(
            np.concatenate([utility[index % len(utility)], [temperatures[index % 4]]])
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
        """Draw Gaussian deviations on the *logarithmic* coordinate, never the natural one.

        A subject whose discount rate is 0.01 per day and one whose rate is 0.1 per day differ
        by one log unit, not by 0.09 per day, and only the first statement stays positive for
        every draw. Nothing else in this module makes that claim, so it is made here.
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
        """Solve a row-coefficient problem on this model's own optimizer settings."""

        return solve_row_coefficients(
            design,
            model_name=model_name,
            model_signature=model_signature,
            optimizer="L-BFGS-B",
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            boundary=self._row_boundary,
        )

    def _row_boundary(
        self, estimates: NDArray[np.float64], derived: NDArray[np.float64] | None
    ) -> bool:
        """This family's boundary convention on a composed estimate, which is the box.

        Every parameter this family estimates is a logarithm, and every one of them has a
        declared finite box that says where the logarithm stops meaning anything -- a discount
        rate so small that no observed delay discounts, a temperature so large that the
        softmax is a step. :func:`~behavio.models._kernels.rowfit.solve_row_coefficients`
        already reports a coordinate resting on that box, so saying so here is the answer
        rather than an omission.
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
        """Generate choices with one parameter vector per row."""

        options = self.read_options(design)
        rows = self._row_coordinates(design, coefficients)
        linear, _, _ = self._utilities(options, rows)
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
        """Return the choice probability under one parameter vector per row."""

        prediction_mode = self._prediction_mode(mode)
        options = self.read_options(study)
        rows = self._row_coordinates(study, coefficients)
        linear, _, _ = self._utilities(options, rows)
        return Prediction(probability=expit(linear), linear_predictor=linear, mode=prediction_mode)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observation under one parameter vector per row."""

        options = self.read_options(study)
        rows = self._row_coordinates(study, coefficients)
        linear, _, _ = self._utilities(options, rows)
        return protected_array(_bernoulli_scores(self._outcomes(study), linear), dtype=np.float64)

    def _row_coordinates(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return validated_row_coefficients(
            coefficients,
            n_rows=len(study),
            n_parameters=len(self.parameter_names),
            what="row coefficients",
        )

    # -- findings ------------------------------------------------------------------------

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the designs in which this model's parameters trade off exactly.

        Reached by ``describe(study)`` through
        :func:`~behavio.models._kernels.introspection.describe_model`, and carried through
        ``smooth()`` and ``hierarchical()`` unchanged.
        """

        try:
            options = self.read_options(study)
        except ModelDataError:
            return ()
        if not self._n_rows(options):
            return ()
        return (*self._scale_findings(options), *self.design_findings(options))

    def design_findings(self, options: Any) -> tuple[ModelFinding, ...]:
        """The family's own design-degeneracy findings."""

        del options
        return ()

    def _scale_findings(self, options: Any) -> tuple[ModelFinding, ...]:
        magnitudes = np.abs(self.declared_amounts(options))
        present = magnitudes[magnitudes > 0.0]
        if not present.size:
            return ()
        ratio = float(np.max(present)) / self.value_scale
        if 0.01 <= ratio <= 100.0:
            return ()
        return (
            ModelFinding(
                code="value_scale_mismatch",
                severity=WARNING,
                message=(
                    f"the largest amount in this study is {ratio:g} times the declared "
                    f"value_scale of {self.value_scale:g}, so the inverse temperature has to "
                    "absorb the unit and will be estimated near the edge of its box; declare "
                    "value_scale in the amounts' own units instead"
                ),
            ),
        )

    def declared_amounts(self, options: Any) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    # -- shared plumbing -------------------------------------------------------------------

    @staticmethod
    def _n_rows(options: Any) -> int:
        return int(options.n_rows)

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
class _ValueChoiceRowObjective:
    """The softmax likelihood as a function of one coordinate vector per row.

    Every row is its own block, and that is the substantive difference between this family
    and the reinforcement-learning ones. A discounting or prospect-theory trial's likelihood
    reads that trial's amounts, delays or probabilities and nothing else that happened, so
    there is no value trace for a mid-block parameter change to make ambiguous, and
    ``smooth(model, over="trial")`` is admissible here where
    ``smooth(BinaryQLearning(), over="trial")`` is an error.
    """

    model: _ValueBasedChoice
    options: Any
    outcomes: NDArray[np.float64]
    n_parameters: int

    @property
    def n_rows(self) -> int:
        """The number of scored rows."""

        return len(self.outcomes)

    @property
    def row_blocks(self) -> NDArray[np.intp]:
        """Every row is independent, so every row is its own block."""

        return np.arange(len(self.outcomes), dtype=np.intp)

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the row coordinates."""

        coordinates = validated_row_coefficients(
            rows, n_rows=self.n_rows, n_parameters=self.n_parameters, what="row coordinates"
        )
        return self.model._rows_value_and_gradient(self.options, self.outcomes, coordinates)


# --------------------------------------------------------------------------------------
# Temporal discounting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DiscountingOptions:
    """One study's amounts and delays, validated once."""

    amounts: NDArray[np.float64]
    delays: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        return len(self.amounts)


@dataclass(frozen=True, slots=True)
class TemporalDiscounting(_ValueBasedChoice):
    """Binary choice between a smaller-sooner and a larger-later option.

    Each option's subjective value is its amount times a discount factor of its delay,

    .. math::

        V = \\frac{A}{S}\\,D(t), \\qquad
        D(t) = \\frac{1}{1 + k t} \\ \\text{(Mazur 1987)}
        \\quad\\text{or}\\quad D(t) = e^{-k t},

    and the choice is a softmax over :math:`V_1 - V_0`. Column ``outcome`` is one when the
    **second** option was chosen; the second option is conventionally the larger-later one,
    and nothing in the model requires it to be.

    The discount rate is estimated as ``discount_rate_log`` because it is strictly positive
    and because a Gaussian deviation on it -- which is what
    ``hierarchical(TemporalDiscounting(), over="subject")`` puts there -- has to live
    somewhere the parameter is unbounded. Its box is derived from the observed delays rather
    than declared: a rate so small that :math:`k t_{\\max} < 10^{-6}` is a subject who does not
    discount over any delay this study contains, and one so large that
    :math:`k t_{\\min} > 10^{6}` is a subject for whom every delayed reward is worthless.
    Neither end is distinguishable from its neighbourhood, which is what a box is for.

    Unlike prospect theory, this family has **no curvature**, so the notorious trade-off
    between the value function and the inverse temperature does not appear here as a
    non-identifiability: :math:`k` changes which option is preferred and :math:`\\beta` changes
    only how reliably. What does appear is the *units* half of the same thing --
    :math:`\\beta` multiplies a value difference in amount units, so it is not comparable
    across studies that used different amounts unless ``value_scale`` says so.
    """

    amount_columns: tuple[str, str] = ("sooner_amount", "later_amount")
    delay_columns: tuple[str, str] = ("sooner_delay", "later_delay")
    outcome: str = "choice"
    discount: DiscountFunction = DiscountFunction.HYPERBOLIC
    value_scale: float = 1.0
    n_restarts: int = 4
    max_iterations: int = 1_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        amounts = _validated_columns(self.amount_columns, "amount_columns")
        delays = _validated_columns(self.delay_columns, "delay_columns")
        _validated_columns((self.outcome,), "outcome")
        if len({*amounts, *delays, self.outcome}) != 5:
            raise ValueError("amount, delay and outcome column names must all be distinct")
        object.__setattr__(self, "amount_columns", amounts)
        object.__setattr__(self, "delay_columns", delays)
        object.__setattr__(self, "discount", DiscountFunction(self.discount))
        _require_positive(self.value_scale, "value_scale")
        _require_positive_integer(self.n_restarts, "n_restarts")
        _require_positive_integer(self.max_iterations, "max_iterations")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

    @property
    def model_name(self) -> str:
        return f"temporal-discounting-{self.discount.value}"

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[outcome={self.outcome};"
            f"amounts={','.join(self.amount_columns)};"
            f"delays={','.join(self.delay_columns)};value_scale={self.value_scale}]"
        )

    @property
    def utility_names(self) -> tuple[str, ...]:
        return ("discount_rate",)

    @property
    def utility_coordinates(self) -> tuple[str, ...]:
        return ("discount_rate_log",)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (*self.amount_columns, *self.delay_columns)

    def parameters_from_components(
        self, *, discount_rate: float, inverse_temperature: float
    ) -> Mapping[str, float]:
        """Validate and encode natural-scale parameters onto the estimated coordinate."""

        components = DiscountingParameters(discount_rate, inverse_temperature)
        return self.from_natural(
            {
                "discount_rate": components.discount_rate,
                "inverse_temperature": components.inverse_temperature,
            }
        )

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> DiscountingParameters:
        """Decode an estimated coordinate into a natural rate and temperature."""

        values = self.to_natural(self._coordinate(parameters))
        return DiscountingParameters(
            discount_rate=values["discount_rate"],
            inverse_temperature=values["inverse_temperature"],
        )

    def indifference_rates(self, study: Study) -> NDArray[np.float64]:
        """Return the discount rate that equates the two options, one value per row.

        The closed form of :func:`indifference_discount_rate` read off this model's own
        columns. It is what the fit's starting values are drawn from, and it is what makes a
        discounting design legible before anything is fitted: a design whose indifference
        rates all sit on one side of the plausible range cannot locate a rate inside it.
        """

        options = self.read_options(study)
        return indifference_discount_rate(
            options.amounts[:, 0] * self.value_scale,
            options.delays[:, 0],
            options.amounts[:, 1] * self.value_scale,
            options.delays[:, 1],
            discount=self.discount,
        )

    # -- what the shared machinery asks for ------------------------------------------------

    def read_options(self, study: Study) -> _DiscountingOptions:
        """Read and validate this study's amounts and delays."""

        amounts = np.column_stack(
            [_numeric_column(study, name, role="task") for name in self.amount_columns]
        )
        delays = np.column_stack(
            [_numeric_column(study, name, role="task") for name in self.delay_columns]
        )
        if np.any(delays < 0.0):
            raise ModelDataError("delays must be non-negative")
        return _DiscountingOptions(amounts=amounts / self.value_scale, delays=delays)

    def declared_amounts(self, options: _DiscountingOptions) -> NDArray[np.float64]:
        return options.amounts * self.value_scale

    def option_values(
        self, options: _DiscountingOptions, rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return both options' discounted values and ``dV / d log k``."""

        rate = np.exp(rows[:, 0])[:, None]
        scaled = rate * options.delays
        if self.discount is DiscountFunction.HYPERBOLIC:
            values = options.amounts / (1.0 + scaled)
            derivative = -values * scaled / (1.0 + scaled)
        else:
            values = options.amounts * np.exp(-scaled)
            derivative = -values * scaled
        return values, derivative[:, :, None]

    def utility_box(self, options: _DiscountingOptions) -> NDArray[np.float64]:
        """Return the box on ``discount_rate_log``, derived from the delays that were used."""

        positive = options.delays[options.delays > 0.0]
        if not positive.size:
            return np.asarray([[-20.0, 20.0]], dtype=np.float64)
        lower = float(np.log(1e-6) - np.log(float(np.max(positive))))
        upper = float(np.log(1e6) - np.log(float(np.min(positive))))
        return np.asarray([[lower, upper]], dtype=np.float64)

    def utility_starts(self, options: _DiscountingOptions) -> tuple[NDArray[np.float64], ...]:
        """Start from the design's own indifference rates, which are closed form."""

        rates = indifference_discount_rate(
            options.amounts[:, 0],
            options.delays[:, 0],
            options.amounts[:, 1],
            options.delays[:, 1],
            discount=self.discount,
        )
        usable = rates[np.isfinite(rates)]
        if not usable.size:
            quantiles = (0.01, 0.1, 0.001)
        else:
            logarithms = np.log(usable)
            quantiles = tuple(
                float(np.exp(np.quantile(logarithms, level))) for level in (0.5, 0.25, 0.75)
            )
        return tuple(np.asarray([float(np.log(rate))], dtype=np.float64) for rate in quantiles)

    def design_findings(self, options: _DiscountingOptions) -> tuple[ModelFinding, ...]:
        """Report a design in which the discount rate is a constant factor."""

        if not np.all(options.delays[:, 0] == options.delays[:, 1]):
            return ()
        return (
            ModelFinding(
                code="unidentified_discount_rate",
                severity=WARNING,
                message=(
                    "discount_rate is not identified by this design: both options carry the "
                    "same delay on every trial, so the discount factor is a common factor of "
                    "the value difference and the inverse temperature absorbs it exactly"
                ),
            ),
        )


# --------------------------------------------------------------------------------------
# Prospect theory
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ProspectOptions:
    """One study's two prospects per row, each as a ranked pair of outcomes.

    ``high`` and ``low`` are the two outcomes of an option with ``high >= low``, and
    ``probability`` is the probability of ``high``. Ranking here rather than in the
    likelihood is what lets the cumulative decision weights be written without a branch on
    which of a user's two columns happened to be larger.
    """

    high: NDArray[np.float64]
    low: NDArray[np.float64]
    probability: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        return len(self.high)


@dataclass(frozen=True, slots=True)
class ProspectTheory(_ValueBasedChoice):
    """Binary choice between two prospects under cumulative prospect theory.

    Each option is a two-outcome prospect :math:`(x, p; y, 1-p)`. Its value is

    .. math::

        V = \\pi_h\\, v(x_h) + \\pi_l\\, v(x_l), \\qquad
        v(z) = \\begin{cases} (z/S)^{\\alpha} & z \\ge 0 \\\\
        -\\lambda\\,(-z/S)^{\\beta} & z < 0, \\end{cases}

    with Prelec's :math:`w` supplying the decision weights and the choice a softmax over
    :math:`V_1 - V_0`. Column ``outcome`` is one when the **second** option was chosen.

    Decision weights are **cumulative** (Tversky & Kahneman 1992) rather than separable
    (Kahneman & Tversky 1979). For a two-outcome prospect that is three cases and no
    numerical work: both outcomes on the same side of zero puts weight :math:`w(q)` on the
    extreme one and :math:`1 - w(q)` on the other, so a sure thing is unweighted and no
    dominated prospect can be preferred; a mixed prospect puts :math:`w(p)` on the gain and
    :math:`w(1-p)` on the loss. The two theories agree exactly whenever one outcome is zero,
    which is the design most risk studies actually run, so the choice of cumulative weighting
    costs nothing there and buys correctness where it matters.

    A single weighting function is estimated and used in both domains. TK92 estimated
    :math:`\\gamma^+ = 0.61` and :math:`\\gamma^- = 0.69`; a second pair would double the
    weighting coordinate for a difference that a single subject's data will rarely resolve,
    and the pair that *is* estimated is already the one this model's ``describe()`` warns
    about when the design has fewer than two interior probabilities.

    ``other_outcome_columns`` is optional. Left as ``None`` every prospect is
    :math:`(x, p; 0, 1-p)`, which is the four-column gamble-or-sure-thing design; naming two
    columns admits genuinely two-outcome prospects, and **mixed gambles are the only design
    that identifies loss aversion from within one domain**, which is why the finding
    ``unidentified_loss_aversion`` exists.

    Any of the loss exponent, the loss aversion coefficient and the weighting elevation may
    be **fixed** at a declared value, in which case it leaves the estimated coordinate
    entirely rather than being estimated against a design that cannot see it. Fixing
    ``weighting_elevation`` at one recovers Prelec's one-parameter form. Tying two parameters
    together -- the common :math:`\\alpha = \\beta` convention -- is deliberately not offered:
    a tie is a constraint on the coordinate, and every combinator that widens this coordinate
    would have to be taught to respect it, whereas a declared value is simply a parameter
    that is not there.
    """

    outcome_columns: tuple[str, str] = ("option_0_outcome", "option_1_outcome")
    probability_columns: tuple[str, str] = ("option_0_probability", "option_1_probability")
    other_outcome_columns: tuple[str, str] | None = None
    outcome: str = "choice"
    value_scale: float = 1.0
    fixed_loss_exponent: float | None = None
    fixed_loss_aversion: float | None = None
    fixed_weighting_elevation: float | None = None
    n_restarts: int = 4
    max_iterations: int = 1_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        outcomes = _validated_columns(self.outcome_columns, "outcome_columns")
        probabilities = _validated_columns(self.probability_columns, "probability_columns")
        others: tuple[str, str] | None = None
        if self.other_outcome_columns is not None:
            others = _validated_columns(self.other_outcome_columns, "other_outcome_columns")
        _validated_columns((self.outcome,), "outcome")
        named = [*outcomes, *probabilities, *(others or ()), self.outcome]
        if len(set(named)) != len(named):
            raise ValueError("every declared column name must be distinct")
        object.__setattr__(self, "outcome_columns", outcomes)
        object.__setattr__(self, "probability_columns", probabilities)
        object.__setattr__(self, "other_outcome_columns", others)
        _require_positive(self.value_scale, "value_scale")
        for value, label in (
            (self.fixed_loss_exponent, "fixed_loss_exponent"),
            (self.fixed_loss_aversion, "fixed_loss_aversion"),
            (self.fixed_weighting_elevation, "fixed_weighting_elevation"),
        ):
            if value is not None:
                _require_positive(float(value), label)
        _require_positive_integer(self.n_restarts, "n_restarts")
        _require_positive_integer(self.max_iterations, "max_iterations")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

    @property
    def model_name(self) -> str:
        return "prospect-theory-prelec"

    @property
    def signature(self) -> str:
        others = (
            "zero" if self.other_outcome_columns is None else ",".join(self.other_outcome_columns)
        )
        fixed = ";".join(
            f"{label}={'estimated' if value is None else value}"
            for label, value in (
                ("loss_exponent", self.fixed_loss_exponent),
                ("loss_aversion", self.fixed_loss_aversion),
                ("weighting_elevation", self.fixed_weighting_elevation),
            )
        )
        return (
            f"{self.model_name}[outcome={self.outcome};"
            f"outcomes={','.join(self.outcome_columns)};"
            f"probabilities={','.join(self.probability_columns)};"
            f"other_outcomes={others};weighting=cumulative;"
            f"value_scale={self.value_scale};{fixed}]"
        )

    @property
    def fixed_values(self) -> tuple[float | None, ...]:
        """The declared value of each of the five utility parameters, or ``None``."""

        return (
            None,
            self.fixed_loss_exponent,
            self.fixed_loss_aversion,
            None,
            self.fixed_weighting_elevation,
        )

    @property
    def utility_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, fixed in zip(_PROSPECT_UTILITY_NAMES, self.fixed_values, strict=True)
            if fixed is None
        )

    @property
    def utility_coordinates(self) -> tuple[str, ...]:
        return tuple(f"{name}_log" for name in self.utility_names)

    @property
    def estimated_positions(self) -> tuple[int, ...]:
        """Which of the five utility parameters this configuration estimates."""

        return tuple(index for index, fixed in enumerate(self.fixed_values) if fixed is None)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (
            *self.outcome_columns,
            *self.probability_columns,
            *(self.other_outcome_columns or ()),
        )

    def parameters_from_components(
        self,
        *,
        gain_exponent: float,
        inverse_temperature: float,
        loss_exponent: float | None = None,
        loss_aversion: float | None = None,
        weighting_curvature: float = 1.0,
        weighting_elevation: float | None = None,
    ) -> Mapping[str, float]:
        """Validate and encode natural-scale parameters onto the estimated coordinate.

        A parameter this model declares fixed must be omitted here, or supplied equal to its
        declared value; one it estimates must be supplied.
        """

        supplied = (
            gain_exponent,
            loss_exponent,
            loss_aversion,
            weighting_curvature,
            weighting_elevation,
        )
        resolved = tuple(
            _resolve_declared(value, fixed, name)
            for value, fixed, name in zip(
                supplied, self.fixed_values, _PROSPECT_UTILITY_NAMES, strict=True
            )
        )
        ProspectTheoryParameters(*resolved, inverse_temperature)
        natural = {
            name: resolved[position]
            for position, name in enumerate(_PROSPECT_UTILITY_NAMES)
            if self.fixed_values[position] is None
        }
        natural[_TEMPERATURE_NAME] = float(inverse_temperature)
        return self.from_natural(natural)

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> ProspectTheoryParameters:
        """Decode an estimated coordinate into the complete natural parameter set.

        Complete, not partial: a fixed parameter is part of the model even though it is not
        part of the coordinate, so it appears here with the value that was declared.
        """

        vector = self._coordinate(parameters)
        natural = self._natural_utilities(vector[None, : len(self.utility_coordinates)])[0]
        return ProspectTheoryParameters(
            *(float(value) for value in natural),
            inverse_temperature=float(np.exp(vector[len(self.utility_coordinates)])),
        )

    def certainty_equivalent(
        self,
        outcome: NDArray[np.float64] | float,
        probability: NDArray[np.float64] | float,
        parameters: Mapping[str, float] | FitResult | ProspectTheoryParameters,
    ) -> NDArray[np.float64]:
        """Return the sure amount worth as much as ``(outcome, probability; 0)``."""

        components = (
            parameters
            if isinstance(parameters, ProspectTheoryParameters)
            else self.parameter_components(parameters)
        )
        return certainty_equivalent(outcome, probability, components)

    # -- what the shared machinery asks for ------------------------------------------------

    def read_options(self, study: Study) -> _ProspectOptions:
        """Read and validate this study's prospects, ranking each option's two outcomes."""

        outcomes = np.column_stack(
            [_numeric_column(study, name, role="task") for name in self.outcome_columns]
        )
        probabilities = np.column_stack(
            [_numeric_column(study, name, role="task") for name in self.probability_columns]
        )
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ModelDataError("option probabilities must lie in [0, 1]")
        if self.other_outcome_columns is None:
            others = np.zeros_like(outcomes)
        else:
            others = np.column_stack(
                [_numeric_column(study, name, role="task") for name in self.other_outcome_columns]
            )
        first_is_high = outcomes >= others
        high = np.where(first_is_high, outcomes, others) / self.value_scale
        low = np.where(first_is_high, others, outcomes) / self.value_scale
        probability = np.where(first_is_high, probabilities, 1.0 - probabilities)
        return _ProspectOptions(high=high, low=low, probability=probability)

    def declared_amounts(self, options: _ProspectOptions) -> NDArray[np.float64]:
        return np.concatenate([options.high.ravel(), options.low.ravel()]) * self.value_scale

    def _natural_utilities(self, rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Expand the estimated utility coordinates into all five natural parameters."""

        natural = np.empty((len(rows), len(_PROSPECT_UTILITY_NAMES)), dtype=np.float64)
        column = 0
        for position, fixed in enumerate(self.fixed_values):
            if fixed is None:
                natural[:, position] = np.exp(rows[:, column])
                column += 1
            else:
                natural[:, position] = float(fixed)
        return natural

    def option_values(
        self, options: _ProspectOptions, rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return both prospects' values and their derivatives on the estimated coordinate."""

        natural = self._natural_utilities(rows)
        alpha = natural[:, 0:1]
        loss_exponent = natural[:, 1:2]
        aversion = natural[:, 2:3]
        curvature = natural[:, 3:4]
        elevation = natural[:, 4:5]

        high_value, high_jacobian = _branch_value(options.high, alpha, loss_exponent, aversion)
        low_value, low_jacobian = _branch_value(options.low, alpha, loss_exponent, aversion)
        weights = _decision_weights(options, curvature, elevation)

        values = weights.high * high_value + weights.low * low_value
        full = np.zeros((*values.shape, len(_PROSPECT_UTILITY_NAMES)), dtype=np.float64)
        for position in range(3):
            full[:, :, position] = (
                weights.high * high_jacobian[position] + weights.low * low_jacobian[position]
            )
        full[:, :, 3] = weights.high_curvature * high_value + weights.low_curvature * low_value
        full[:, :, 4] = weights.high_elevation * high_value + weights.low_elevation * low_value
        return values, full[:, :, list(self.estimated_positions)]

    def utility_box(self, options: _ProspectOptions) -> NDArray[np.float64]:
        """Return the box on the estimated utility logarithms.

        Wide enough that no plausible subject reaches it and narrow enough that a coordinate
        resting on it is a statement: an exponent of :math:`e^{\\pm 4}` is a value function
        indistinguishable from a step or from indifference, a loss aversion of
        :math:`e^{4} \\approx 55` is not a number anybody has reported, and a weighting
        curvature outside :math:`e^{\\pm 3}` is a weighting function with no interior shape.
        """

        del options
        boxes = ((-4.0, 2.0), (-4.0, 2.0), (-4.0, 4.0), (-3.0, 3.0), (-3.0, 3.0))
        return np.asarray([boxes[position] for position in self.estimated_positions])

    def utility_starts(self, options: _ProspectOptions) -> tuple[NDArray[np.float64], ...]:
        """Start from Tversky and Kahneman's (1992) medians, and from three perturbations."""

        del options
        schedule = (
            (0.88, 0.88, 2.25, 0.65, 1.0),
            (0.50, 0.50, 1.00, 1.00, 1.0),
            (1.00, 1.00, 3.00, 0.50, 1.2),
            (0.70, 1.00, 1.50, 0.85, 0.8),
        )
        return tuple(
            np.asarray(
                [float(np.log(start[position])) for position in self.estimated_positions],
                dtype=np.float64,
            )
            for start in schedule
        )

    def design_findings(self, options: _ProspectOptions) -> tuple[ModelFinding, ...]:
        """Report the designs in which a prospect-theory parameter is a constant factor."""

        amounts = np.concatenate([options.high.ravel(), options.low.ravel()])
        findings: list[ModelFinding] = []
        findings.extend(self._curvature_findings(amounts))
        findings.extend(self._loss_findings(options, amounts))
        findings.extend(self._weighting_findings(options))
        return tuple(findings)

    def _curvature_findings(self, amounts: NDArray[np.float64]) -> list[ModelFinding]:
        findings: list[ModelFinding] = []
        for sign, name in ((1.0, "gain_exponent"), (-1.0, "loss_exponent")):
            if name not in self.utility_names:
                continue
            present = np.abs(amounts[np.sign(amounts) == sign])
            distinct = len(np.unique(np.round(present, 12)))
            if distinct == 1:
                findings.append(
                    ModelFinding(
                        code="unidentified_utility_curvature",
                        severity=WARNING,
                        message=(
                            f"{name} is not identified by this design: every non-zero outcome "
                            "on that side of zero has the same magnitude, so the exponent "
                            "enters every row as one constant factor and the inverse "
                            "temperature absorbs it exactly"
                        ),
                    )
                )
            elif distinct == 2:
                findings.append(
                    ModelFinding(
                        code="weakly_identified_utility_curvature",
                        severity=WARNING,
                        message=(
                            f"{name} is identified only by how choices change between two "
                            "outcome magnitudes, so it will trade off strongly against the "
                            "inverse temperature; check their correlation on the fit"
                        ),
                    )
                )
        return findings

    def _loss_findings(
        self, options: _ProspectOptions, amounts: NDArray[np.float64]
    ) -> list[ModelFinding]:
        names = [name for name in ("loss_exponent", "loss_aversion") if name in self.utility_names]
        if not names:
            return []
        if not np.any(amounts < 0.0):
            return [
                ModelFinding(
                    code="unobserved_loss_domain",
                    severity=WARNING,
                    message=(
                        f"{', '.join(names)} cannot be estimated from this design: it "
                        "contains no negative outcome, so the loss branch of the value "
                        "function is never evaluated and its coordinates have no gradient"
                    ),
                )
            ]
        if "loss_aversion" not in names:
            return []
        gains = (options.high > 0.0) | (options.low > 0.0)
        losses = (options.high < 0.0) | (options.low < 0.0)
        mixed_option = np.any(gains & losses, axis=1)
        cross_option = gains.any(axis=1) & losses.any(axis=1)
        if np.any(mixed_option | cross_option):
            return []
        return [
            ModelFinding(
                code="unidentified_loss_aversion",
                severity=WARNING,
                message=(
                    "loss_aversion is not identified by this design: no trial places a gain "
                    "against a loss, so lambda multiplies every value in the comparison by "
                    "one constant and the inverse temperature absorbs it exactly"
                ),
            )
        ]

    def _weighting_findings(self, options: _ProspectOptions) -> list[ModelFinding]:
        names = [
            name
            for name in ("weighting_curvature", "weighting_elevation")
            if name in self.utility_names
        ]
        if not names:
            return []
        probabilities = options.probability.ravel()
        interior = probabilities[(probabilities > 0.0) & (probabilities < 1.0)]
        distinct = len(np.unique(np.round(interior, 12)))
        if distinct >= len(names):
            return []
        return [
            ModelFinding(
                code="unidentified_probability_weighting",
                severity=WARNING,
                message=(
                    f"{', '.join(names)} cannot be separated by this design: it offers "
                    f"{distinct} probability value(s) strictly between zero and one, and a "
                    "weighting function evaluated at fewer points than it has parameters is "
                    "a lookup table the inverse temperature can rewrite"
                ),
            )
        ]


_PROSPECT_UTILITY_NAMES = (
    "gain_exponent",
    "loss_exponent",
    "loss_aversion",
    "weighting_curvature",
    "weighting_elevation",
)


@dataclass(frozen=True, slots=True)
class _DecisionWeights:
    """Cumulative decision weights on a ranked pair, and their weighting derivatives."""

    high: NDArray[np.float64]
    low: NDArray[np.float64]
    high_curvature: NDArray[np.float64]
    low_curvature: NDArray[np.float64]
    high_elevation: NDArray[np.float64]
    low_elevation: NDArray[np.float64]


def _decision_weights(
    options: _ProspectOptions,
    curvature: NDArray[np.float64],
    elevation: NDArray[np.float64],
) -> _DecisionWeights:
    """Return cumulative prospect theory's weights on a ranked two-outcome prospect.

    Three cases, and the reason each is what it is. Both outcomes weakly positive: the
    *better* one carries :math:`w(q)` and the other carries the complement, so a degenerate
    prospect is unweighted. Both weakly negative: rank runs from the worst outcome, so the
    *worse* one carries :math:`w(1-q)` and the other the complement. Mixed: the gain is
    weighted against the best outcome and the loss against the worst, so each carries its own
    :math:`w`, and the two need not sum to one -- which is exactly the subcertainty prospect
    theory was written to express.
    """

    high_weight, high_curvature, high_elevation = _weight_and_derivatives(
        options.probability, curvature, elevation
    )
    low_weight, low_curvature, low_elevation = _weight_and_derivatives(
        1.0 - options.probability, curvature, elevation
    )
    all_gain = options.low >= 0.0
    all_loss = ~all_gain & (options.high <= 0.0)
    return _DecisionWeights(
        high=np.where(all_loss, 1.0 - low_weight, high_weight),
        low=np.where(all_gain, 1.0 - high_weight, low_weight),
        high_curvature=np.where(all_loss, -low_curvature, high_curvature),
        low_curvature=np.where(all_gain, -high_curvature, low_curvature),
        high_elevation=np.where(all_loss, -low_elevation, high_elevation),
        low_elevation=np.where(all_gain, -high_elevation, low_elevation),
    )


def _branch_value(
    amounts: NDArray[np.float64],
    gain_exponent: NDArray[np.float64],
    loss_exponent: NDArray[np.float64],
    aversion: NDArray[np.float64],
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
    """Return ``v(x)`` and its derivatives on the three value-function log coordinates.

    Each derivative is taken on the logarithm directly, which is where every one of them
    simplifies: :math:`\\partial v / \\partial \\log \\alpha = v \\alpha \\log|x|` for a gain,
    and :math:`\\partial v / \\partial \\log \\lambda = v` for a loss, because
    :math:`\\lambda` multiplies the whole loss branch.
    """

    gain = amounts >= 0.0
    magnitude = np.abs(amounts)
    nonzero = magnitude > 0.0
    logarithm = np.where(nonzero, np.log(np.where(nonzero, magnitude, 1.0)), 0.0)
    gain_value = np.power(magnitude, gain_exponent)
    loss_value = -aversion * np.power(magnitude, loss_exponent)
    value = np.where(gain, gain_value, loss_value)
    return value, (
        np.where(gain, gain_value * gain_exponent * logarithm, 0.0),
        np.where(gain, 0.0, loss_value * loss_exponent * logarithm),
        np.where(gain, 0.0, loss_value),
    )


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def _bernoulli_scores(
    outcomes: NDArray[np.float64], linear: NDArray[np.float64]
) -> NDArray[np.float64]:
    scores = outcomes * -np.logaddexp(0.0, -linear)
    scores += (1.0 - outcomes) * -np.logaddexp(0.0, linear)
    return scores


def _at_box(estimates: NDArray[np.float64], box: NDArray[np.float64]) -> bool:
    tolerances = BOUNDARY_TOLERANCE * np.maximum(1.0, box[:, 1] - box[:, 0])
    return bool(
        np.any(estimates - box[:, 0] <= tolerances) or np.any(box[:, 1] - estimates <= tolerances)
    )


def _numeric_column(study: Study, name: str, *, role: str) -> NDArray[np.float64]:
    if name not in study.columns:
        raise ModelDataError(f"study is missing {role} column {name!r}")
    try:
        values = np.asarray(study[name], dtype=np.float64)
    except (TypeError, ValueError):
        raise ModelDataError(f"{role} column {name!r} must be numeric") from None
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ModelDataError(f"{role} column {name!r} must contain finite numbers")
    return values


def _validated_columns(names: Sequence[str], label: str) -> tuple[str, ...]:
    columns = tuple(names)
    if any(not isinstance(name, str) or not name for name in columns):
        raise ValueError(f"{label} must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{label} must contain distinct column names")
    if set(columns) & set(REQUIRED_COLUMNS):
        raise ValueError(f"{label} cannot replace a required Study column")
    return columns


def _require_positive(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{label} must be a positive real number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def _require_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _resolve_declared(supplied: float | None, fixed: float | None, label: str) -> float:
    if fixed is None:
        if supplied is None:
            raise ValueError(f"{label} is estimated by this model and must be supplied")
        return float(supplied)
    if supplied is not None and not np.isclose(float(supplied), float(fixed)):
        raise ValueError(f"{label} is fixed at {fixed} by this model")
    return float(fixed)
