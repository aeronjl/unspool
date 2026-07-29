"""Parametric psychometric functions with declared links and asymmetric guess/lapse rates.

The family implemented here is the one psychophysicists actually report:

.. math::

    \\psi(x) = \\gamma + (1 - \\gamma - \\lambda)\\, F\\!\\left(
        \\frac{t(x) - t(\\alpha)}{w} + z_{1/2} \\right)

with a *guess* rate :math:`\\gamma` and a *lapse* rate :math:`\\lambda` as two separate
bounded parameters (the "two-gamma" form), a threshold :math:`\\alpha` and a width
:math:`w` in stimulus units, and a declared link :math:`F`.

Conventions this module fixes, because published implementations disagree
------------------------------------------------------------------------
*Threshold.* :math:`\\alpha` is the stimulus level at which the **link reaches one half**,
so :math:`\\psi(\\alpha)` is exactly midway between :math:`\\gamma` and
:math:`1 - \\lambda`. That is what the constant :math:`z_{1/2}` is for: the logistic,
Gauss and erf links already satisfy :math:`F(0) = 1/2`, while the Gumbel link does not, so
its location is shifted by :math:`z_{1/2} = \\log\\log 2`. Wichmann and Hill (2001) write
the Weibull as :math:`1 - \\exp(-(x/\\alpha)^{\\beta})`, whose :math:`\\alpha` sits at
:math:`1 - e^{-1} \\approx 0.632`; the equivalent 50 %-referenced form used here is
:math:`1 - 2^{-(x/\\alpha)^{1/w}}`. Thresholds from the two conventions are not
interchangeable and the docstrings say which one produced a number.

*Stimulus scale.* The Weibull is the Gumbel on log stimulus, so :math:`t = \\log` for
:class:`PsychometricLink.WEIBULL` and :math:`t` is the identity for every other link. The
threshold is therefore always in stimulus units, while the Weibull's width is in
log-stimulus units and its shape parameter is :math:`1/w`.

*erf versus Gauss.* :math:`F_{\\mathrm{erf}}(z) = (\\operatorname{erf}(z) + 1)/2 =
\\Phi(z\\sqrt{2})`. The two links describe the same curves but their widths differ by
:math:`\\sqrt{2}`. Both are provided because the erf parameterisation is the one used by
the International Brain Laboratory's released ``erf_psycho_2gammas``; see
:func:`erf_two_gamma_probability`.

*Bounds.* ``maximum_guess`` and ``maximum_lapse`` are constructor arguments and appear in
the model signature, so a fit can never be read without its bounds. Either rate may
instead be **fixed** at a known value (``fixed_guess_rate=0.5`` for a symmetric two-
alternative task), in which case it leaves the parameter vector entirely rather than being
estimated at a boundary.

References
----------
Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I. Fitting, sampling,
and goodness of fit. *Perception & Psychophysics, 63*, 1293-1313.

Prins, N. (2012). The psychometric function: the lapse rate revisited. *Journal of
Vision, 12*(6):25.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import erf, expit, logit, ndtr, ndtri

from behavio._internal.arrays import protected_array
from behavio.contracts import (
    DerivedQuantity,
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models._kernels.curvature import covariance_from_hessian, finite_difference_hessian
from behavio.study import REQUIRED_COLUMNS, Study

#: Probability floor used when a curve is evaluated at a saturated guess or lapse rate.
PROBABILITY_FLOOR = 1e-12

#: Interval level carried by the derived quantities a fit publishes without being asked.
DEFAULT_INTERVAL_LEVEL = 0.95

_INV_SQRT_TWO_PI = float(1.0 / np.sqrt(2.0 * np.pi))
_GUMBEL_HALF = float(np.log(np.log(2.0)))


class PsychometricLink(StrEnum):
    """The sigmoid family used by a psychometric function."""

    LOGISTIC = "logistic"
    GAUSS = "gauss"
    ERF = "erf"
    GUMBEL = "gumbel"
    WEIBULL = "weibull"


@dataclass(frozen=True, slots=True)
class _Link:
    """One link: its cumulative form, its density, its half-point, and its stimulus scale."""

    name: PsychometricLink
    cumulative: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    density: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    half: float
    log_stimulus: bool


def _logistic_density(values: NDArray[np.float64]) -> NDArray[np.float64]:
    probability = expit(values)
    return probability * (1.0 - probability)


def _gauss_density(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return _INV_SQRT_TWO_PI * np.exp(-0.5 * np.square(values))


def _erf_cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (erf(values) + 1.0)


def _erf_density(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-np.square(values)) / np.sqrt(np.pi)


def _gumbel_cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return -np.expm1(-np.exp(np.clip(values, -700.0, 700.0)))


def _gumbel_density(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, -700.0, 700.0)
    return np.exp(clipped - np.exp(clipped))


_LINKS: Mapping[PsychometricLink, _Link] = MappingProxyType(
    {
        PsychometricLink.LOGISTIC: _Link(
            PsychometricLink.LOGISTIC, expit, _logistic_density, 0.0, False
        ),
        PsychometricLink.GAUSS: _Link(PsychometricLink.GAUSS, ndtr, _gauss_density, 0.0, False),
        PsychometricLink.ERF: _Link(
            PsychometricLink.ERF, _erf_cumulative, _erf_density, 0.0, False
        ),
        PsychometricLink.GUMBEL: _Link(
            PsychometricLink.GUMBEL, _gumbel_cumulative, _gumbel_density, _GUMBEL_HALF, False
        ),
        PsychometricLink.WEIBULL: _Link(
            PsychometricLink.WEIBULL, _gumbel_cumulative, _gumbel_density, _GUMBEL_HALF, True
        ),
    }
)


@dataclass(frozen=True, slots=True)
class PsychometricParameters:
    """Natural-scale psychometric parameters in the units a psychophysicist reports."""

    threshold: float
    width: float
    guess_rate: float
    lapse_rate: float

    def __post_init__(self) -> None:
        values = (self.threshold, self.width, self.guess_rate, self.lapse_rate)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("psychometric parameters must be finite")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.guess_rate < 0 or self.lapse_rate < 0:
            raise ValueError("guess_rate and lapse_rate must be non-negative")
        if self.guess_rate + self.lapse_rate >= 1:
            raise ValueError("guess_rate and lapse_rate must leave a positive dynamic range")


@dataclass(frozen=True, slots=True)
class PsychometricSummary:
    """Natural-scale estimates with Wald intervals built on each parameter's own scale.

    Each interval is formed on the coordinate that was estimated -- the log scale for the
    width and a positive threshold, the bounded logit scale for the two rates -- and then
    mapped back, so an interval can never leave the parameter's admissible range. A fixed
    rate reports a zero standard error and a degenerate interval, which is the honest
    description of a quantity that was declared rather than estimated.
    """

    link: PsychometricLink
    interval_level: float
    threshold: float
    width: float
    guess_rate: float
    lapse_rate: float
    threshold_standard_error: float
    width_standard_error: float
    guess_rate_standard_error: float
    lapse_rate_standard_error: float
    threshold_interval: tuple[float, float]
    width_interval: tuple[float, float]
    guess_rate_interval: tuple[float, float]
    lapse_rate_interval: tuple[float, float]
    slope_at_threshold: float
    guess_rate_is_fixed: bool
    lapse_rate_is_fixed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "link", PsychometricLink(self.link))
        if not 0 < self.interval_level < 1:
            raise ValueError("interval_level must lie strictly between zero and one")
        for lower, upper in (
            self.threshold_interval,
            self.width_interval,
            self.guess_rate_interval,
            self.lapse_rate_interval,
        ):
            if np.isfinite(lower) and np.isfinite(upper) and lower > upper:
                raise ValueError("interval bounds must be ordered")

    @property
    def parameters(self) -> PsychometricParameters:
        """The natural-scale point estimate on its own validated type."""

        return PsychometricParameters(
            threshold=self.threshold,
            width=self.width,
            guess_rate=self.guess_rate,
            lapse_rate=self.lapse_rate,
        )

    def derived_quantities(self) -> tuple[DerivedQuantity, ...]:
        """Return this summary as contract-level derived quantities.

        This is the natural coordinate a psychophysicist publishes, carried on the fit so
        that a consumer typed on plain :class:`~behavio.contracts.FitResult` -- model
        comparison, an evidence bundle, ``export_fit`` -- sees a threshold rather than a
        ``log_width``. The intervals are the summary's own, formed on the estimated
        coordinate and mapped back, so none of them can leave its parameter's range. A rate
        that was declared rather than estimated keeps its zero standard error and its
        degenerate interval, which is the honest description of a fixed quantity.
        """

        fixed_note = "declared rather than estimated"
        return (
            DerivedQuantity(
                "threshold",
                self.threshold,
                standard_error=self.threshold_standard_error,
                interval=self.threshold_interval,
                interval_level=self.interval_level,
                description="stimulus level at which the link reaches one half",
            ),
            DerivedQuantity(
                "width",
                self.width,
                standard_error=self.width_standard_error,
                interval=self.width_interval,
                interval_level=self.interval_level,
                description="curve width in stimulus units",
            ),
            DerivedQuantity(
                "guess_rate",
                self.guess_rate,
                standard_error=self.guess_rate_standard_error,
                interval=self.guess_rate_interval,
                interval_level=self.interval_level,
                description=fixed_note if self.guess_rate_is_fixed else "lower asymptote",
            ),
            DerivedQuantity(
                "lapse_rate",
                self.lapse_rate,
                standard_error=self.lapse_rate_standard_error,
                interval=self.lapse_rate_interval,
                interval_level=self.interval_level,
                description=fixed_note if self.lapse_rate_is_fixed else "upper asymptote gap",
            ),
            DerivedQuantity(
                "slope_at_threshold",
                self.slope_at_threshold,
                description="d psi / d x at the threshold, in probability per stimulus unit",
            ),
        )


@dataclass(frozen=True, slots=True)
class PsychometricFitResult(FitResult):
    """A psychometric fit that retains every deterministic restart.

    This subclass survives the removal of its detection-theory siblings, and it survives
    for one reason: the four restart fields are evidence that *only fitting produced*.
    They are the :class:`~behavio.contracts.MultistartFit` contract, so
    :meth:`~behavio.contracts.FitResult.audit` derives a
    :class:`~behavio.contracts.RestartAudit` from this fit without either side knowing
    about the other, and nothing recomputes them short of refitting. Six other models in
    the catalogue carry the same four fields for the same reason.

    Everything that was *not* such evidence has gone. ``link`` was a declared
    configuration already spelled out by ``model_name`` and ``model_signature``, and the
    typed ``threshold`` / ``width`` / ``guess_rate`` / ``lapse_rate`` readers only renamed
    entries of :attr:`~behavio.contracts.FitResult.derived`, where those quantities carry
    their own standard errors and intervals and where a consumer typed on plain
    ``FitResult`` can already see them.
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
        missing = sorted(
            {"threshold", "width", "guess_rate", "lapse_rate"} - set(self.derived_values)
        )
        if missing:
            raise ValueError(f"a psychometric fit must declare derived {missing}")
        values = self.derived_values
        PsychometricParameters(
            values["threshold"], values["width"], values["guess_rate"], values["lapse_rate"]
        )
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)


@dataclass(frozen=True, slots=True)
class PsychometricFunction:
    """A psychometric curve with a declared link and separate guess and lapse rates.

    The estimated coordinate is deliberately not the natural one. ``width`` is estimated as
    ``log_width`` and the two rates as bounded logits so the optimizer is unconstrained in
    the interior, while :meth:`summarize` maps the fit back onto thresholds, widths and
    rates with intervals. For a link on log stimulus (the Weibull) the location coordinate
    is ``log_threshold``; for every other link it is ``threshold`` itself, so the primary
    quantity a psychophysicist asks for carries its own standard error directly on the
    :class:`~behavio.contracts.FitResult`.
    """

    stimulus: str = "stimulus"
    outcome: str = "choice"
    link: PsychometricLink = PsychometricLink.LOGISTIC
    maximum_guess: float = 0.2
    maximum_lapse: float = 0.2
    fixed_guess_rate: float | None = None
    fixed_lapse_rate: float | None = None
    n_restarts: int = 4
    max_iterations: int = 1_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for value, label in ((self.stimulus, "stimulus"), (self.outcome, "outcome")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty column name")
            if value in REQUIRED_COLUMNS:
                raise ValueError(f"{label} cannot replace a required Study column")
        if self.stimulus == self.outcome:
            raise ValueError("stimulus and outcome columns must be distinct")
        object.__setattr__(self, "link", PsychometricLink(self.link))
        for value, label in (
            (self.maximum_guess, "maximum_guess"),
            (self.maximum_lapse, "maximum_lapse"),
        ):
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{label} must lie strictly between zero and one")
        for value, label in (
            (self.fixed_guess_rate, "fixed_guess_rate"),
            (self.fixed_lapse_rate, "fixed_lapse_rate"),
        ):
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{label} must be a real number or None")
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{label} must lie in [0, 1)")
        # ``maximum_guess`` and ``maximum_lapse`` bound only the rates that are estimated.
        # A rate declared fixed is a known property of the task -- 0.5 for a symmetric
        # two-alternative design -- and is not held to the estimation bound.
        if (
            self._rate_ceiling(self.fixed_guess_rate, self.maximum_guess)
            + self._rate_ceiling(self.fixed_lapse_rate, self.maximum_lapse)
            >= 1
        ):
            raise ValueError("the declared guess and lapse rates must leave a positive range")
        for value, label in (
            (self.n_restarts, "n_restarts"),
            (self.max_iterations, "max_iterations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

    @staticmethod
    def _rate_ceiling(fixed: float | None, maximum: float) -> float:
        return float(maximum if fixed is None else fixed)

    @property
    def model_name(self) -> str:
        return f"psychometric-{self.link.value}"

    @property
    def signature(self) -> str:
        guess = "estimated" if self.fixed_guess_rate is None else f"fixed={self.fixed_guess_rate}"
        lapse = "estimated" if self.fixed_lapse_rate is None else f"fixed={self.fixed_lapse_rate}"
        return (
            f"{self.model_name}[outcome={self.outcome};stimulus={self.stimulus};"
            f"guess={guess};maximum_guess={self.maximum_guess};"
            f"lapse={lapse};maximum_lapse={self.maximum_lapse}]"
        )

    @property
    def location_name(self) -> str:
        """Name of the estimated location coordinate, which is link dependent."""

        return "log_threshold" if _LINKS[self.link].log_stimulus else "threshold"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = [self.location_name, "log_width"]
        if self.fixed_guess_rate is None:
            names.append("guess_logit")
        if self.fixed_lapse_rate is None:
            names.append("lapse_logit")
        return tuple(names)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.stimulus,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate: threshold, width, and the estimated rates.

        A rate declared fixed is not part of this coordinate for the same reason it is not
        part of ``parameter_names``: it was not estimated, so there is nothing to report an
        uncertainty for and nothing for a Jacobian to differentiate.
        """

        names = ["threshold", "width"]
        if self.fixed_guess_rate is None:
            names.append("guess_rate")
        if self.fixed_lapse_rate is None:
            names.append("lapse_rate")
        return tuple(names)

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Decode the estimated coordinate into threshold, width, and estimated rates."""

        components = self._decode(self._natural_input(estimates))
        values = {"threshold": components.threshold, "width": components.width}
        if self.fixed_guess_rate is None:
            values["guess_rate"] = components.guess_rate
        if self.fixed_lapse_rate is None:
            values["lapse_rate"] = components.lapse_rate
        return MappingProxyType(values)

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Encode a complete natural mapping onto the estimated coordinate."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        return self.parameters_from_components(
            threshold=natural["threshold"],
            width=natural["width"],
            guess_rate=natural.get("guess_rate"),
            lapse_rate=natural.get("lapse_rate"),
        )

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return the derivative of the natural parameters at one estimated coordinate.

        The map is diagonal: each natural parameter is a monotone function of exactly one
        estimated coordinate. That is not true of every reparameterisation -- the detection-
        theory rating criteria are not -- which is why the contract passes a full Jacobian
        rather than a per-parameter derivative.
        """

        coordinate = self._natural_input(estimates)
        width = len(self.parameter_names)
        jacobian = np.zeros((len(self.natural_names), width), dtype=np.float64)
        log_stimulus = _LINKS[self.link].log_stimulus
        jacobian[0, 0] = float(np.exp(coordinate[0])) if log_stimulus else 1.0
        jacobian[1, 1] = float(np.exp(coordinate[1]))
        row = 2
        for fixed, maximum in (
            (self.fixed_guess_rate, self.maximum_guess),
            (self.fixed_lapse_rate, self.maximum_lapse),
        ):
            if fixed is not None:
                continue
            relative = float(expit(coordinate[row]))
            jacobian[row, row] = maximum * relative * (1.0 - relative)
            row += 1
        return jacobian

    def _natural_input(
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

    def parameters_from_components(
        self,
        *,
        threshold: float,
        width: float,
        guess_rate: float | None = None,
        lapse_rate: float | None = None,
    ) -> Mapping[str, float]:
        """Encode natural parameters onto the estimated coordinate.

        Keyword-argument façade over :meth:`from_natural`, which shares this encoding.

        A rate declared fixed on the model must be omitted here or supplied equal to its
        fixed value; an estimated rate must lie strictly inside its declared bound, because
        a logit coordinate cannot represent a rate sitting exactly on zero or the maximum.
        """

        guess = self._resolve_declared_rate(guess_rate, self.fixed_guess_rate, "guess_rate")
        lapse = self._resolve_declared_rate(lapse_rate, self.fixed_lapse_rate, "lapse_rate")
        components = PsychometricParameters(threshold, width, guess, lapse)
        if _LINKS[self.link].log_stimulus and components.threshold <= 0:
            raise ValueError(f"the {self.link.value} link requires a positive threshold")
        values: dict[str, float] = {
            self.location_name: (
                float(np.log(components.threshold))
                if _LINKS[self.link].log_stimulus
                else float(components.threshold)
            ),
            "log_width": float(np.log(components.width)),
        }
        if self.fixed_guess_rate is None:
            values["guess_logit"] = self._encode_rate(components.guess_rate, self.maximum_guess)
        if self.fixed_lapse_rate is None:
            values["lapse_logit"] = self._encode_rate(components.lapse_rate, self.maximum_lapse)
        return MappingProxyType(values)

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> PsychometricParameters:
        """Decode an estimated coordinate into natural threshold, width, and rates."""

        return self._decode(self._coordinate(parameters))

    def summarize(self, fit: FitResult, *, level: float = 0.95) -> PsychometricSummary:
        """Return natural-scale estimates with intervals at ``level`` for one fit.

        Standard errors are delta-method transformations of the estimated covariance;
        intervals are Wald intervals formed on the estimated coordinate and mapped back.
        """

        if not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        self._validate_fit(fit)
        return self._summarize(
            np.asarray(fit.estimates, dtype=np.float64),
            np.asarray(fit.standard_errors, dtype=np.float64),
            level=level,
        )

    def _summarize(
        self,
        coordinate: NDArray[np.float64],
        errors: NDArray[np.float64],
        *,
        level: float,
    ) -> PsychometricSummary:
        """Summarize one estimated coordinate and its standard errors.

        Split out of :meth:`summarize` so that :meth:`fit` can attach the same natural
        estimates and intervals to the fit result it is constructing, without either the
        summary or the fit becoming the other's prerequisite.
        """

        components = self._decode(coordinate)
        quantile = float(ndtri(0.5 + 0.5 * level))
        log_stimulus = _LINKS[self.link].log_stimulus
        location_error = float(errors[0])
        if log_stimulus:
            threshold_error = components.threshold * location_error
            threshold_interval = (
                float(np.exp(coordinate[0] - quantile * location_error)),
                float(np.exp(coordinate[0] + quantile * location_error)),
            )
        else:
            threshold_error = location_error
            threshold_interval = (
                float(coordinate[0] - quantile * location_error),
                float(coordinate[0] + quantile * location_error),
            )
        width_error = components.width * float(errors[1])
        width_interval = (
            float(np.exp(coordinate[1] - quantile * float(errors[1]))),
            float(np.exp(coordinate[1] + quantile * float(errors[1]))),
        )
        index = 2
        if self.fixed_guess_rate is None:
            guess_error, guess_interval = self._rate_uncertainty(
                coordinate[index], errors[index], self.maximum_guess, quantile
            )
            index += 1
        else:
            guess_error, guess_interval = 0.0, (components.guess_rate, components.guess_rate)
        if self.fixed_lapse_rate is None:
            lapse_error, lapse_interval = self._rate_uncertainty(
                coordinate[index], errors[index], self.maximum_lapse, quantile
            )
        else:
            lapse_error, lapse_interval = 0.0, (components.lapse_rate, components.lapse_rate)
        return PsychometricSummary(
            link=self.link,
            interval_level=float(level),
            threshold=components.threshold,
            width=components.width,
            guess_rate=components.guess_rate,
            lapse_rate=components.lapse_rate,
            threshold_standard_error=float(threshold_error),
            width_standard_error=float(width_error),
            guess_rate_standard_error=float(guess_error),
            lapse_rate_standard_error=float(lapse_error),
            threshold_interval=threshold_interval,
            width_interval=width_interval,
            guess_rate_interval=guess_interval,
            lapse_rate_interval=lapse_interval,
            slope_at_threshold=self.slope_at_threshold(components),
            guess_rate_is_fixed=self.fixed_guess_rate is not None,
            lapse_rate_is_fixed=self.fixed_lapse_rate is not None,
        )

    def slope_at_threshold(self, components: PsychometricParameters) -> float:
        """Return ``d psi / d x`` evaluated at the threshold, in probability per stimulus unit."""

        link = _LINKS[self.link]
        density = float(link.density(np.asarray([link.half], dtype=np.float64))[0])
        span = 1.0 - components.guess_rate - components.lapse_rate
        scale = components.width * (components.threshold if link.log_stimulus else 1.0)
        return float(span * density / scale)

    def probability(
        self,
        stimulus: NDArray[np.float64],
        components: PsychometricParameters,
    ) -> NDArray[np.float64]:
        """Return the curve evaluated at arbitrary stimulus levels."""

        values = self._transform(np.asarray(stimulus, dtype=np.float64))
        location = self._location(components.threshold)
        standardized = self._standardize(values, location, components.width)
        link = _LINKS[self.link]
        span = 1.0 - components.guess_rate - components.lapse_rate
        return components.guess_rate + span * link.cumulative(standardized)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        components = self.parameter_components(parameters)
        stimulus = self._stimulus(design)
        probability = np.clip(
            self.probability(stimulus, components), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR
        )
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = generator.binomial(1, probability).astype(np.int8)
        return Study(columns)

    def fit(self, study: Study) -> PsychometricFitResult:
        stimulus = self._stimulus(study)
        outcomes = self._outcomes(study)
        values = self._transform(stimulus)
        starts, bounds = self._start_schedule(values, outcomes)
        results = [
            minimize(
                lambda coordinate: self._objective(coordinate, values, outcomes),
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
        objectives = np.asarray([float(result.fun) for result in results], dtype=np.float64)
        selected = int(np.argmin(objectives))
        result = results[selected]
        estimates = np.asarray(result.x, dtype=np.float64)
        _, gradient = self._objective(estimates, values, outcomes)
        hessian = finite_difference_hessian(
            lambda coordinate: self._objective(coordinate, values, outcomes)[1], estimates
        )
        covariance = covariance_from_hessian(hessian)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        summary = self._summarize(estimates, standard_errors, level=DEFAULT_INTERVAL_LEVEL)
        at_bound = [
            abs(float(estimate) - float(low)) < 1e-6 or abs(float(estimate) - float(high)) < 1e-6
            for estimate, (low, high) in zip(estimates, bounds, strict=True)
        ]
        diagnostics = FitDiagnostics(
            converged=bool(result.success),
            optimizer="L-BFGS-B deterministic multistart",
            status=int(result.status),
            message=str(result.message),
            n_iterations=int(result.nit),
            objective=float(result.fun),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=float(np.linalg.cond(hessian)),
            boundary_estimate=bool(any(at_bound)),
        )
        return PsychometricFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            derived=summary.derived_quantities(),
            restart_objectives=objectives,
            restart_converged=np.asarray([bool(item.success) for item in results]),
            restart_messages=tuple(str(item.message) for item in results),
            selected_restart=selected,
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        self._prediction_mode(mode)
        components = self.parameter_components(fit)
        stimulus = self._stimulus(study)
        values = self._transform(stimulus)
        standardized = self._standardize(
            values, self._location(components.threshold), components.width
        )
        return Prediction(
            probability=self.probability(stimulus, components),
            linear_predictor=standardized,
            mode=PredictionMode.FILTERED,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        outcomes = self._outcomes(study)
        probability = np.clip(
            self.predict(study, fit, mode=mode).probability,
            PROBABILITY_FLOOR,
            1.0 - PROBABILITY_FLOOR,
        )
        scores = outcomes * np.log(probability) + (1.0 - outcomes) * np.log1p(-probability)
        return protected_array(scores, dtype=np.float64)

    def _objective(
        self,
        coordinate: NDArray[np.float64],
        values: NDArray[np.float64],
        outcomes: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        link = _LINKS[self.link]
        location = float(coordinate[0])
        width = float(np.exp(coordinate[1]))
        guess, guess_slope = self._rate(coordinate, 2, self.fixed_guess_rate, self.maximum_guess)
        offset = 2 + (self.fixed_guess_rate is None)
        lapse, lapse_slope = self._rate(
            coordinate, offset, self.fixed_lapse_rate, self.maximum_lapse
        )
        standardized = (values - location) / width + link.half
        cumulative = link.cumulative(standardized)
        density = link.density(standardized)
        span = 1.0 - guess - lapse
        probability = np.clip(guess + span * cumulative, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
        loss = -float(outcomes @ np.log(probability) + (1.0 - outcomes) @ np.log1p(-probability))
        score = (probability - outcomes) / (probability * (1.0 - probability))
        gradient = [
            float(np.sum(score * span * density * (-1.0 / width))),
            float(np.sum(score * span * density * -(standardized - link.half))),
        ]
        if self.fixed_guess_rate is None:
            gradient.append(float(np.sum(score * (1.0 - cumulative)) * guess_slope))
        if self.fixed_lapse_rate is None:
            gradient.append(float(np.sum(score * -cumulative) * lapse_slope))
        return loss, np.asarray(gradient, dtype=np.float64)

    def _rate(
        self,
        coordinate: NDArray[np.float64],
        index: int,
        fixed: float | None,
        maximum: float,
    ) -> tuple[float, float]:
        if fixed is not None:
            return float(fixed), 0.0
        relative = float(expit(coordinate[index]))
        return maximum * relative, maximum * relative * (1.0 - relative)

    def _start_schedule(
        self, values: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> tuple[list[NDArray[np.float64]], tuple[tuple[float, float], ...]]:
        span = float(np.max(values) - np.min(values))
        if not np.isfinite(span) or span <= 0:
            raise ModelDataError("a psychometric fit needs more than one stimulus level")
        crossing = self._empirical_crossing(values, outcomes)
        location_starts = (
            crossing,
            float(np.quantile(values, 0.25)),
            float(np.quantile(values, 0.75)),
        )
        width_starts = (0.25, 0.1, 0.5, 1.0)
        rate_starts = (0.05, 0.02, 0.15)
        starts: list[NDArray[np.float64]] = []
        for index in range(self.n_restarts):
            coordinate = [
                location_starts[index % len(location_starts)],
                float(np.log(span * width_starts[index % len(width_starts)])),
            ]
            relative = rate_starts[index % len(rate_starts)]
            if self.fixed_guess_rate is None:
                coordinate.append(float(logit(relative)))
            if self.fixed_lapse_rate is None:
                coordinate.append(float(logit(relative)))
            starts.append(np.asarray(coordinate, dtype=np.float64))
        bounds = [
            (float(np.min(values) - 2.0 * span), float(np.max(values) + 2.0 * span)),
            (float(np.log(span) - 8.0), float(np.log(span) + 4.0)),
        ]
        bounds.extend((-12.0, 12.0) for _ in self.parameter_names[2:])
        return starts, tuple(bounds)

    @staticmethod
    def _empirical_crossing(values: NDArray[np.float64], outcomes: NDArray[np.float64]) -> float:
        levels = np.unique(values)
        means = np.asarray([float(np.mean(outcomes[values == level])) for level in levels])
        if levels.size < 2 or means[0] == means[-1]:
            return float(np.median(values))
        increasing = means[-1] > means[0]
        ordered = means if increasing else means[::-1]
        grid = levels if increasing else levels[::-1]
        if ordered[0] >= 0.5:
            return float(grid[0])
        if ordered[-1] <= 0.5:
            return float(grid[-1])
        index = int(np.argmax(ordered >= 0.5))
        low, high = ordered[index - 1], ordered[index]
        if high == low:
            return float(grid[index])
        weight = (0.5 - low) / (high - low)
        return float(grid[index - 1] + weight * (grid[index] - grid[index - 1]))

    def _rate_uncertainty(
        self, coordinate: float, error: float, maximum: float, quantile: float
    ) -> tuple[float, tuple[float, float]]:
        relative = float(expit(coordinate))
        derivative = maximum * relative * (1.0 - relative)
        interval = (
            float(maximum * expit(coordinate - quantile * error)),
            float(maximum * expit(coordinate + quantile * error)),
        )
        return float(derivative * error), interval

    def _decode(self, coordinate: NDArray[np.float64]) -> PsychometricParameters:
        location = float(coordinate[0])
        threshold = float(np.exp(location)) if _LINKS[self.link].log_stimulus else location
        guess, _ = self._rate(coordinate, 2, self.fixed_guess_rate, self.maximum_guess)
        offset = 2 + (self.fixed_guess_rate is None)
        lapse, _ = self._rate(coordinate, offset, self.fixed_lapse_rate, self.maximum_lapse)
        return PsychometricParameters(
            threshold=threshold,
            width=float(np.exp(coordinate[1])),
            guess_rate=guess,
            lapse_rate=lapse,
        )

    def _coordinate(self, parameters: Mapping[str, float] | FitResult) -> NDArray[np.float64]:
        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            return np.asarray(parameters.estimates, dtype=np.float64)
        if set(parameters) != set(self.parameter_names):
            raise ValueError("parameters must match the model exactly")
        try:
            values = np.asarray(
                [parameters[name] for name in self.parameter_names], dtype=np.float64
            )
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must contain finite numeric values")
        return values

    @staticmethod
    def _resolve_declared_rate(supplied: float | None, fixed: float | None, label: str) -> float:
        if fixed is None:
            if supplied is None:
                raise ValueError(f"{label} is estimated by this model and must be supplied")
            return float(supplied)
        if supplied is not None and not np.isclose(float(supplied), float(fixed)):
            raise ValueError(f"{label} is fixed at {fixed} by this model")
        return float(fixed)

    @staticmethod
    def _encode_rate(rate: float, maximum: float) -> float:
        if not 0 < rate < maximum:
            raise ValueError(f"an estimated rate must lie strictly inside (0, {maximum})")
        return float(logit(rate / maximum))

    def _location(self, threshold: float) -> float:
        if _LINKS[self.link].log_stimulus:
            if threshold <= 0:
                raise ValueError(f"the {self.link.value} link requires a positive threshold")
            return float(np.log(threshold))
        return float(threshold)

    def _standardize(
        self, values: NDArray[np.float64], location: float, width: float
    ) -> NDArray[np.float64]:
        return (values - location) / width + _LINKS[self.link].half

    def _transform(self, stimulus: NDArray[np.float64]) -> NDArray[np.float64]:
        if not _LINKS[self.link].log_stimulus:
            return stimulus
        if np.any(stimulus <= 0):
            raise ModelDataError(f"the {self.link.value} link requires positive stimulus levels")
        return np.log(stimulus)

    def _stimulus(self, study: Study) -> NDArray[np.float64]:
        if self.stimulus not in study.columns:
            raise ModelDataError(f"study is missing stimulus column {self.stimulus!r}")
        try:
            values = np.asarray(study[self.stimulus], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError("stimulus values must be finite numbers") from None
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ModelDataError("stimulus values must be finite numbers")
        return values

    def _outcomes(self, study: Study) -> NDArray[np.float64]:
        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        try:
            outcomes = np.asarray(study[self.outcome], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError("outcomes must contain only zero and one") from None
        if outcomes.ndim != 1 or not np.all((outcomes == 0.0) | (outcomes == 1.0)):
            raise ModelDataError("outcomes must contain only zero and one")
        return outcomes

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


def erf_two_gamma_probability(
    stimulus: NDArray[np.float64] | float,
    *,
    bias: float,
    threshold: float,
    lapse_low: float,
    lapse_high: float,
) -> NDArray[np.float64]:
    """Return the released ``erf_psycho_2gammas`` curve on Behavio's own link machinery.

    The International Brain Laboratory's published psychometrics use

    .. math::

        \\psi(c) = \\lambda_{\\mathrm{low}} +
        (1 - \\lambda_{\\mathrm{low}} - \\lambda_{\\mathrm{high}})
        \\frac{\\operatorname{erf}\\!\\left((c - \\mathrm{bias}) / \\theta\\right) + 1}{2},

    which is exactly :class:`PsychometricFunction` under
    :attr:`PsychometricLink.ERF` with ``threshold=bias``, ``width=`` :math:`\\theta`,
    ``guess_rate=`` :math:`\\lambda_{\\mathrm{low}}` and ``lapse_rate=``
    :math:`\\lambda_{\\mathrm{high}}`. The two names for the location differ because the
    released code calls the erf's *scale* its "threshold"; in this package the threshold is
    the stimulus level at the curve's midpoint and the scale is the width.

    ``benchmarks/ibl2021_psychometrics`` keeps its own independent implementation of the
    same equation together with the released Nelder-Mead restart schedule and its penalty
    box. That module reproduces a *published pipeline*; this one is a reusable estimator.
    A committed test asserts the two agree to machine precision.
    """

    values = np.atleast_1d(np.asarray(stimulus, dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise ValueError("stimulus levels must be finite")
    components = PsychometricParameters(
        threshold=float(bias),
        width=float(threshold),
        guess_rate=float(lapse_low),
        lapse_rate=float(lapse_high),
    )
    model = PsychometricFunction(
        link=PsychometricLink.ERF,
        maximum_guess=0.49,
        maximum_lapse=0.49,
    )
    return model.probability(values, components)
