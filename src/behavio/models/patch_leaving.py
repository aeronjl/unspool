"""Patch leaving: a giving-up rule fitted as a hazard, read against the marginal value theorem.

Charnov's (1976) marginal value theorem says that an animal exploiting a depleting patch in an
environment whose patches are separated by a travel time :math:`\\tau` should leave when the
patch's instantaneous intake rate has fallen to the environment's long-run average rate,

.. math::

    g'(t^{*}) = R^{*} = \\max_{t} \\frac{g(t)}{\\tau + t},

which is a *normative* statement: it says what an optimal forager would do, not what a
likelihood is. This module keeps the two apart on purpose.

:func:`marginal_value_rate` and :func:`marginal_value_residence_time` are the theorem, closed
form, with no fitting anywhere near them. :class:`PatchLeaving` is a **behavioural** model of
when an animal actually goes, and it is fitted to residence times without assuming that the
threshold it estimates is the optimal one. The overstaying result -- animals reliably leave at
an intake rate *below* :math:`R^{*}` -- is then a number a fit reports
(``overstaying_ratio``), not an assumption baked into the model that reports it. A module that
fitted the MVT could not measure a departure from it.

The observable is a hazard, and that is a modelling claim
---------------------------------------------------------
The fitted object is a *leaving decision per unit residence time*. The animal is in the patch
and is, moment by moment, deciding whether to go; nothing about that is a choice between two
options, and nothing about it is a density that exists before the decision rule does. So the
primitive here is the instantaneous leaving rate :math:`\\lambda(t)` and the density is derived
from it -- which is the opposite of a response-time model, where a first-passage density is
the primitive and its hazard is a description computed afterwards. See
:mod:`behavio.models._kernels.hazard` for what the two therefore do and do not share: the
censoring arithmetic is common and is written once; the hazard *family* is not, and should not
be.

The leaving rule
----------------
The animal carries a giving-up rate :math:`\\theta`: the intake rate at which the patch stops
being worth staying in. It is not carried exactly. Writing the noise on the log scale, because
:math:`\\theta` is a rate and a rate's noise is multiplicative,

.. math::

    \\log \\Theta \\sim \\text{Logistic}(\\log\\theta,\\ s),
    \\qquad T = \\inf\\{t : g'(t) \\le \\Theta\\},

and because :math:`g'` is strictly decreasing this gives closed-form everything:

.. math::

    S(t) = \\frac{\\sigma(u(t))}{\\sigma(u(0))}, \\qquad
    u(t) = \\frac{\\log g'(t) - \\log\\theta}{s}, \\qquad
    \\lambda(t) = -u'(t)\\,\\bigl(1 - \\sigma(u(t))\\bigr).

The division by :math:`\\sigma(u(0))` is a conditioning, not a normalisation: an animal that
entered a patch and stayed in it for a positive time had a threshold below that patch's entry
rate, so the model conditions on that rather than placing an atom of probability at zero
residence. :math:`s` is dimensionless -- it is a Weber fraction on the intake rate -- and
:math:`\\lambda` rises towards :math:`-u'` as the patch depletes, which is the shape patch
residence data have.

As :math:`s \\to 0` the leaving time converges to the deterministic threshold crossing
:math:`g'(t) = \\theta`, so setting :math:`\\theta = R^{*}` recovers Charnov's :math:`t^{*}`
exactly. That is the check :mod:`tests.test_patch_leaving` holds the simulator to.

Censoring is real and is declared
---------------------------------
A session that ends while the animal is still in the patch is not a leaving time. Declare
``censoring_time_column`` -- the longest residence each row could have shown -- and such a row
is scored by :math:`\\log S(c)` rather than :math:`\\log f(t)`. Leave it undeclared and the
model asserts every duration ran to its event; ``describe()`` reports
``undeclared_censoring`` when the residence times pile up on a common maximum, which is what
that assertion looks like when it is false.

**One thing does not fit the prediction contract, and it is worth reporting rather than
working around.** ``predict`` returns a
:class:`~behavio.contracts.DensityPrediction` of the *leaving time*, which is the right
prediction for every row and is what the model actually claims. A censored row's *score* is a
survival probability, and no member of
:data:`~behavio.contracts.estimator.ModelPrediction` can carry "the probability the event is
still to come". So ``pointwise_log_prob`` and ``DensityPrediction.observed_log_density`` agree
on uncensored rows and deliberately disagree on censored ones. ``pointwise_log_prob`` is the
likelihood; the density is the prediction; a consumer that scores the density directly instead
of asking the model will misscore exactly the censored rows. ``describe()`` says so through
``heavy_censoring``.

``docs/decisions/0061-fit-patch-leaving-as-a-hazard-not-as-the-marginal-value-theorem.md``
records why the theorem is a benchmark rather than the likelihood, what a per-moment decision
noise would have confounded, and what a censored response-time model would and would not
reuse from here.

References
----------
Charnov, E. L. (1976). Optimal foraging, the marginal value theorem. *Theoretical Population
Biology, 9*(2), 129-136.

Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton University Press.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit, log_expit

from behavio._internal.arrays import protected_array
from behavio.contracts.estimator import (
    DensityPrediction,
    DerivedQuantity,
    FitResult,
    ModelDataError,
    PredictionMode,
)
from behavio.models._kernels.hazard import (
    CensoredDurations,
    censored_log_scores,
    censored_score_gradients,
    read_censoring,
    validated_durations,
)
from behavio.models._kernels.introspection import WARNING, ModelFinding
from behavio.models._kernels.logcoordinate import (
    LogCoordinateFitResult,
    LogCoordinateModel,
    numeric_column,
    positive_integer,
    require_positive,
    validated_columns,
)
from behavio.trials import Study

__all__ = [
    "GainFunction",
    "LogCoordinateFitResult",
    "PatchLeaving",
    "PatchLeavingParameters",
    "instantaneous_intake_rate",
    "marginal_value_rate",
    "marginal_value_residence_time",
    "patch_gain",
]

#: Survival probability at which a tabulated residence-time density is truncated.
_DENSITY_TAIL = 1e-4

_GIVING_UP_MARGIN = 6.0
_NOISE_BOX = (float(np.log(1e-3)), float(np.log(5.0)))


class GainFunction(StrEnum):
    """The shape of the cumulative gain an animal takes from one patch."""

    EXPONENTIAL = "exponential"
    """:math:`g(t) = A\\,(1 - e^{-\\rho t})`: the standard depleting-patch schedule.

    ``patch_decay`` is :math:`\\rho`, a rate with units of reciprocal time. The optimal
    residence time under this gain is **independent of** :math:`A`, because
    :math:`g'/g` does not contain it -- which is itself a discriminating prediction, and one
    the hyperbolic form does not make.
    """

    HYPERBOLIC = "hyperbolic"
    """:math:`g(t) = A\\,t / (t + h)`: Holling's disc equation, saturating in handling time.

    ``patch_decay`` is :math:`h`, a *time*, and the optimal residence time is exactly
    :math:`t^{*} = \\sqrt{h\\tau}` with :math:`R^{*} = A / (\\sqrt{h} + \\sqrt{\\tau})^2`. The
    two decay parameters are not interchangeable and neither are their fits.
    """


# --------------------------------------------------------------------------------------
# The closed forms this module is validated against
# --------------------------------------------------------------------------------------


def patch_gain(
    time: NDArray[np.float64] | float,
    *,
    patch_yield: NDArray[np.float64] | float,
    patch_decay: NDArray[np.float64] | float,
    gain: GainFunction | str = GainFunction.EXPONENTIAL,
) -> NDArray[np.float64]:
    """Return the cumulative gain :math:`g(t)` taken from a patch by time ``t``."""

    function = GainFunction(gain)
    elapsed = np.asarray(time, dtype=np.float64)
    amount = np.asarray(patch_yield, dtype=np.float64)
    decay = np.asarray(patch_decay, dtype=np.float64)
    if function is GainFunction.EXPONENTIAL:
        return np.asarray(amount * -np.expm1(-decay * elapsed), dtype=np.float64)
    return np.asarray(amount * elapsed / (elapsed + decay), dtype=np.float64)


def instantaneous_intake_rate(
    time: NDArray[np.float64] | float,
    *,
    patch_yield: NDArray[np.float64] | float,
    patch_decay: NDArray[np.float64] | float,
    gain: GainFunction | str = GainFunction.EXPONENTIAL,
) -> NDArray[np.float64]:
    """Return :math:`g'(t)`, the intake rate the marginal value theorem is a rule about."""

    function = GainFunction(gain)
    elapsed = np.asarray(time, dtype=np.float64)
    amount = np.asarray(patch_yield, dtype=np.float64)
    decay = np.asarray(patch_decay, dtype=np.float64)
    if function is GainFunction.EXPONENTIAL:
        return np.asarray(amount * decay * np.exp(-decay * elapsed), dtype=np.float64)
    return np.asarray(amount * decay / (elapsed + decay) ** 2, dtype=np.float64)


def _residence_at_rate(
    rate: NDArray[np.float64] | float,
    *,
    patch_yield: NDArray[np.float64],
    patch_decay: NDArray[np.float64],
    gain: GainFunction,
) -> NDArray[np.float64]:
    """Return the time at which each patch's intake rate has fallen to ``rate``."""

    threshold = np.asarray(rate, dtype=np.float64)
    entry = instantaneous_intake_rate(
        0.0, patch_yield=patch_yield, patch_decay=patch_decay, gain=gain
    )
    if gain is GainFunction.EXPONENTIAL:
        times = (np.log(entry) - np.log(threshold)) / patch_decay
    else:
        times = np.sqrt(patch_yield * patch_decay / threshold) - patch_decay
    return np.asarray(np.maximum(times, 0.0), dtype=np.float64)


def marginal_value_rate(
    patch_yield: Sequence[float] | NDArray[np.float64] | float,
    patch_decay: Sequence[float] | NDArray[np.float64] | float,
    travel_time: float,
    *,
    gain: GainFunction | str = GainFunction.EXPONENTIAL,
) -> float:
    """Return :math:`R^{*}`, the environment's optimal long-run intake rate.

    The marginal value theorem is a statement about an **environment**, not about a patch: one
    threshold rate applies to every patch type in it, and it is the fixed point of

    .. math::

        R = \\frac{\\sum_i p_i\\, g_i(t_i(R))}{\\tau + \\sum_i p_i\\, t_i(R)},
        \\qquad g_i'(t_i(R)) = R,

    with :math:`p_i` the frequency of each patch type. For a single patch type this collapses
    to the textbook tangent construction, and for the hyperbolic gain it has the exact
    solution :math:`R^{*} = A/(\\sqrt{h} + \\sqrt{\\tau})^2` -- which is what
    :mod:`tests.test_patch_leaving` checks the root finder against, rather than checking the
    root finder against itself.

    Patch types are taken with the frequencies in which they appear in the arrays given, so
    passing one row per observed patch weights the environment by what the animal actually
    met.
    """

    function = GainFunction(gain)
    amounts = np.atleast_1d(np.asarray(patch_yield, dtype=np.float64))
    decays = np.atleast_1d(np.asarray(patch_decay, dtype=np.float64))
    if amounts.shape != decays.shape:
        amounts, decays = np.broadcast_arrays(amounts, decays)
    if not amounts.size:
        raise ValueError("an environment must contain at least one patch")
    if np.any(amounts <= 0.0) or np.any(decays <= 0.0):
        raise ValueError("patch yields and decay parameters must be positive")
    travel = float(travel_time)
    if not np.isfinite(travel) or travel <= 0.0:
        raise ValueError("travel_time must be finite and positive")
    entry = instantaneous_intake_rate(0.0, patch_yield=amounts, patch_decay=decays, gain=function)
    upper = float(np.max(entry))

    def excess(rate: float) -> float:
        times = _residence_at_rate(rate, patch_yield=amounts, patch_decay=decays, gain=function)
        gains = patch_gain(times, patch_yield=amounts, patch_decay=decays, gain=function)
        return float(np.mean(gains)) / (travel + float(np.mean(times))) - rate

    lower = upper * 1e-12
    if excess(lower) <= 0.0:  # pragma: no cover - only for degenerate environments
        return float(lower)
    return float(brentq(excess, lower, upper * (1.0 - 1e-12), xtol=1e-14, rtol=1e-14))


def marginal_value_residence_time(
    patch_yield: Sequence[float] | NDArray[np.float64] | float,
    patch_decay: Sequence[float] | NDArray[np.float64] | float,
    travel_time: float,
    *,
    gain: GainFunction | str = GainFunction.EXPONENTIAL,
) -> NDArray[np.float64]:
    """Return :math:`t^{*}`, the optimal residence time in each patch of an environment.

    One value per patch given, because a heterogeneous environment has one *rate* threshold
    and therefore a different optimal residence time in each patch type. ``float`` inputs
    return a one-element array; a single-patch environment under the hyperbolic gain returns
    :math:`\\sqrt{h\\tau}` to machine precision.
    """

    function = GainFunction(gain)
    amounts = np.atleast_1d(np.asarray(patch_yield, dtype=np.float64))
    decays = np.atleast_1d(np.asarray(patch_decay, dtype=np.float64))
    amounts, decays = np.broadcast_arrays(amounts, decays)
    rate = marginal_value_rate(amounts, decays, travel_time, gain=function)
    return _residence_at_rate(rate, patch_yield=amounts, patch_decay=decays, gain=function)


# --------------------------------------------------------------------------------------
# Natural-scale parameter record
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatchLeavingParameters:
    """Natural-scale patch-leaving parameters in the units a report uses.

    ``giving_up_rate`` is in the intake units the study's ``patch_yield`` column is in,
    divided by time; ``decision_noise`` is dimensionless, and is a Weber fraction on that
    rate.
    """

    giving_up_rate: float
    decision_noise: float

    def __post_init__(self) -> None:
        require_positive(self.giving_up_rate, "giving_up_rate")
        require_positive(self.decision_noise, "decision_noise")


@dataclass(frozen=True, slots=True)
class _PatchProblem:
    """One study's patches, residence times and censoring, validated once."""

    log_entry_rate: NDArray[np.float64]
    log_observed_rate: NDArray[np.float64]
    log_decay_slope: NDArray[np.float64]
    durations: CensoredDurations

    @property
    def n_rows(self) -> int:
        return self.durations.n_rows


# --------------------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatchLeaving(LogCoordinateModel):
    """When an animal leaves a depleting patch, as a giving-up rate and a decision noise.

    Each row is one patch visit: the patch's ``patch_yield`` and ``patch_decay``, optionally
    the ``travel_time`` that separated it from the next one, and the observed residence time.
    The two estimated parameters are the intake rate at which the animal gives up and how
    noisily it applies that threshold; see the module docstring for the survival function they
    produce and for why the observable is a hazard.

    ``travel_time_column`` is optional and changes nothing about the likelihood. Declared, a
    fit gains three derived quantities -- ``marginal_value_rate``, ``optimal_residence_time``
    and ``overstaying_ratio`` -- which are Charnov's prediction and the fitted animal's
    departure from it. That is the whole design of this class: the theorem is a benchmark the
    fit is read against, never a constraint the fit is made to satisfy.

    ``censoring_time_column`` is optional and changes the likelihood a great deal; see the
    module docstring.

    Identifiability, before the fit
    -------------------------------
    ``unidentified_leaving_rule``
        Every patch in the study depletes identically. Then :math:`\\log g'(t)` is one fixed
        monotone function of :math:`t`, so "leave when the intake rate falls to
        :math:`\\theta`" and "leave after :math:`t_\\theta` seconds" make **identical**
        predictions about every residence time in the study. The content of the marginal value
        theorem is that one rate threshold governs patches of different richness and different
        depletion; a single-patch-type design cannot see that, however many visits it
        contains, and a fitted ``giving_up_rate`` from such a design is a residence time
        wearing a rate's units.
    ``undeclared_censoring`` / ``heavy_censoring`` / ``all_rows_censored``
        What the observation limit is doing to the fit, reported before it does it.
    ``heterogeneous_environment``
        More than one patch type is present, so ``optimal_residence_time`` is reported per
        type against one shared ``marginal_value_rate``, which is what the theorem says --
        stated because a reader expecting one optimal time would otherwise read the mean of
        several.
    """

    yield_column: str = "patch_yield"
    decay_column: str = "patch_decay"
    outcome: str = "residence_time"
    travel_time_column: str | None = None
    censoring_time_column: str | None = None
    gain: GainFunction = GainFunction.EXPONENTIAL
    grid_points: int = 257
    n_restarts: int = 4
    max_iterations: int = 1_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        declared = [self.yield_column, self.decay_column, self.outcome]
        if self.travel_time_column is not None:
            declared.append(self.travel_time_column)
        if self.censoring_time_column is not None:
            declared.append(self.censoring_time_column)
        validated_columns(declared, "declared columns")
        object.__setattr__(self, "gain", GainFunction(self.gain))
        positive_integer(self.grid_points, "grid_points")
        if self.grid_points < 8:
            raise ValueError("grid_points must be at least eight")
        positive_integer(self.n_restarts, "n_restarts")
        positive_integer(self.max_iterations, "max_iterations")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

    # -- identity -------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"patch-leaving-{self.gain.value}"

    @property
    def signature(self) -> str:
        travel = self.travel_time_column or "undeclared"
        censoring = self.censoring_time_column or "none"
        return (
            f"{self.model_name}[outcome={self.outcome};patch={self.yield_column},"
            f"{self.decay_column};travel={travel};censoring={censoring};"
            f"threshold=logistic;grid={self.grid_points}]"
        )

    @property
    def natural_names(self) -> tuple[str, ...]:
        return ("giving_up_rate", "decision_noise")

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        columns = [self.yield_column, self.decay_column]
        if self.travel_time_column is not None:
            columns.append(self.travel_time_column)
        if self.censoring_time_column is not None:
            columns.append(self.censoring_time_column)
        return tuple(columns)

    @property
    def density_outcome(self) -> str:
        """The continuous column :meth:`predict_density` tabulates a density over."""

        return self.outcome

    @property
    def penalised_linear_refusal(self) -> str:
        """Why the penalised-linear route cannot be applied to this family."""

        return (
            "a patch-leaving model divides a log intake rate by an estimated noise and reads "
            "the result through a survival function, so nothing in it is a design matrix "
            "times a coefficient; its rows are independent, so it composes through "
            "behavio.contracts.bounded.BoundedCoordinateEstimator instead"
        )

    # -- natural-coordinate conveniences ---------------------------------------------------

    def parameters_from_components(
        self, *, giving_up_rate: float, decision_noise: float
    ) -> Mapping[str, float]:
        """Validate and encode natural-scale parameters onto the estimated coordinate."""

        components = PatchLeavingParameters(giving_up_rate, decision_noise)
        return self.from_natural(
            {
                "giving_up_rate": components.giving_up_rate,
                "decision_noise": components.decision_noise,
            }
        )

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> PatchLeavingParameters:
        """Decode an estimated coordinate into a natural giving-up rate and noise."""

        values = self.to_natural(self.coordinate(parameters))
        return PatchLeavingParameters(
            giving_up_rate=float(values["giving_up_rate"]),
            decision_noise=float(values["decision_noise"]),
        )

    def deterministic_residence_time(
        self, study: Study, parameters: Mapping[str, float] | FitResult
    ) -> NDArray[np.float64]:
        """Return the noise-free leaving time of each row: where :math:`g'(t) = \\theta`.

        The quantity the marginal value theorem makes a prediction about, evaluated at this
        model's own threshold. With ``decision_noise`` small it is also the median residence
        time, which is how the simulator is checked against Charnov's closed form.
        """

        amounts, decays = self._patches(study)
        rate = float(self.parameter_components(parameters).giving_up_rate)
        return _residence_at_rate(rate, patch_yield=amounts, patch_decay=decays, gain=self.gain)

    def leaving_hazard(
        self, study: Study, parameters: Mapping[str, float] | FitResult, times: Any = None
    ) -> NDArray[np.float64]:
        """Return the instantaneous leaving rate of each row at ``times``.

        ``times`` defaults to each row's observed residence time. This is the model's
        primitive, exposed because a hazard is the quantity a foraging reader wants plotted
        and because deriving it from the density outside the model would be a second
        implementation of it.
        """

        problem = self.read_problem(study, require_outcome=times is None)
        elapsed = problem.durations.times if times is None else np.asarray(times, dtype=np.float64)
        amounts, decays = self._patches(study)
        rows = self.shared_rows(len(study), self.coordinate(parameters))
        log_rate = self._log_intake(elapsed, amounts, decays)
        slope = self._log_slope(elapsed, decays)
        noise = np.exp(rows[:, 1])
        deviate = (log_rate - rows[:, 0]) / noise
        return np.asarray((slope / noise) * (1.0 - expit(deviate)), dtype=np.float64)

    # -- reading a study -------------------------------------------------------------------

    def _patches(self, study: Study) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        amounts = numeric_column(study, self.yield_column, role="task")
        decays = numeric_column(study, self.decay_column, role="task")
        if np.any(amounts <= 0.0) or np.any(decays <= 0.0):
            raise ModelDataError("patch yields and decay parameters must be positive")
        return amounts, decays

    def _log_intake(
        self,
        elapsed: NDArray[np.float64],
        amounts: NDArray[np.float64],
        decays: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return :math:`\\log g'(t)`, which is where both gain functions stay well behaved."""

        if self.gain is GainFunction.EXPONENTIAL:
            return np.log(amounts * decays) - decays * elapsed
        return np.log(amounts * decays) - 2.0 * np.log(elapsed + decays)

    def _log_slope(
        self, elapsed: NDArray[np.float64], decays: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return :math:`-\\mathrm{d}\\log g' / \\mathrm{d}t`, which is positive throughout."""

        if self.gain is GainFunction.EXPONENTIAL:
            return np.asarray(np.zeros_like(elapsed) + decays, dtype=np.float64)
        return np.asarray(2.0 / (elapsed + decays), dtype=np.float64)

    def read_problem(self, study: Study, *, require_outcome: bool = True) -> _PatchProblem:
        """Read and validate this study's patches, residence times and observation limits."""

        amounts, decays = self._patches(study)
        if not require_outcome and self.outcome not in study.columns:
            times = np.zeros_like(amounts)
        else:
            times = validated_durations(
                numeric_column(study, self.outcome, role="outcome"),
                label=f"outcome column {self.outcome!r}",
            )
        limits = (
            None
            if self.censoring_time_column is None
            else numeric_column(study, self.censoring_time_column, role="task")
        )
        censored = read_censoring(
            times, limits, label=f"censoring column {self.censoring_time_column!r}"
        )
        zeros = np.zeros_like(times)
        return _PatchProblem(
            log_entry_rate=self._log_intake(zeros, amounts, decays),
            log_observed_rate=self._log_intake(times, amounts, decays),
            log_decay_slope=np.log(self._log_slope(times, decays)),
            durations=CensoredDurations(times=times, censored=censored),
        )

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the observed residence time of each row, in the outcome column's units."""

        return validated_durations(
            numeric_column(study, self.outcome, role="outcome"),
            label=f"outcome column {self.outcome!r}",
        )

    # -- the likelihood --------------------------------------------------------------------

    def _scores_and_gradients(
        self, problem: _PatchProblem, rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return each row's log likelihood and its gradient on the estimated coordinate.

        Every derivative is taken on the logarithms directly, which is where the survival
        function's two deviates simplify: with :math:`a = \\log\\theta` and
        :math:`b = \\log s`, :math:`\\partial u/\\partial a = -1/s` and
        :math:`\\partial u/\\partial b = -u` for both :math:`u(t)` and :math:`u(0)`.
        """

        noise = np.exp(rows[:, 1])
        deviate = (problem.log_observed_rate - rows[:, 0]) / noise
        entry = (problem.log_entry_rate - rows[:, 0]) / noise
        weight = expit(deviate)
        entry_weight = expit(entry)
        log_survival = log_expit(deviate) - log_expit(entry)
        log_density = problem.log_decay_slope - rows[:, 1] + log_expit(-deviate) + log_survival

        density_gradient = np.empty_like(rows)
        density_gradient[:, 0] = (2.0 * weight - entry_weight) / noise
        density_gradient[:, 1] = (
            -1.0 - deviate * (1.0 - 2.0 * weight) + entry * (1.0 - entry_weight)
        )
        survival_gradient = np.empty_like(rows)
        survival_gradient[:, 0] = (weight - entry_weight) / noise
        survival_gradient[:, 1] = -deviate * (1.0 - weight) + entry * (1.0 - entry_weight)

        censored = problem.durations.censored
        scores = censored_log_scores(
            log_density=log_density, log_survival=log_survival, censored=censored
        )
        gradients = censored_score_gradients(
            density_gradient=density_gradient,
            survival_gradient=survival_gradient,
            censored=censored,
        )
        return scores, gradients

    def row_value_and_gradient(
        self, problem: _PatchProblem, rows: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its analytic gradient, one row at a time."""

        scores, gradients = self._scores_and_gradients(problem, rows)
        return -float(np.sum(scores)), -gradients

    # -- the bounded-coordinate per-row members ---------------------------------------------

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> DensityPrediction:
        """Return the tabulated residence-time density under one coordinate per row.

        A density of the **leaving time**, on every row, censored or not: that is what the
        model predicts, and a censored row's observation is not a leaving time. The score of
        such a row is a survival probability and comes from ``pointwise_log_prob``; see the
        module docstring for why those two deliberately disagree.
        """

        prediction_mode = self.prediction_mode(mode)
        amounts, decays = self._patches(study)
        rows = self.row_coordinates(study, coefficients)
        noise = np.exp(rows[:, 1])
        entry = (self._log_intake(np.zeros_like(amounts), amounts, decays) - rows[:, 0]) / noise
        # The deviate at which the conditional survival has fallen to ``_DENSITY_TAIL``, taken
        # at the row whose entry conditioning is tightest so that no row's tail is cut short.
        quantile = _DENSITY_TAIL * float(np.min(expit(entry)))
        tail = float(np.log(quantile) - np.log1p(-quantile))
        tail_times = _residence_at_rate(
            np.exp(rows[:, 0] + noise * tail),
            patch_yield=amounts,
            patch_decay=decays,
            gain=self.gain,
        )
        observed = (
            numeric_column(study, self.outcome, role="outcome")
            if self.outcome in study.columns
            else np.zeros_like(amounts)
        )
        upper = max(float(np.max(tail_times)), float(np.max(observed)) * 1.05, 1e-6)
        grid = np.linspace(0.0, upper, self.grid_points)
        log_rate = self._log_intake(grid[None, :], amounts[:, None], decays[:, None])
        slope = self._log_slope(grid[None, :], decays[:, None])
        deviate = (log_rate - rows[:, 0][:, None]) / noise[:, None]
        density = (
            (slope / noise[:, None])
            * (1.0 - expit(deviate))
            * np.exp(log_expit(deviate) - log_expit(entry)[:, None])
        )
        return DensityPrediction(
            grid=grid, density=density, outcome=self.outcome, mode=prediction_mode
        )

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        """Return the predictive density, which is what ``predict`` already returns."""

        prediction_mode = self.prediction_mode(mode)
        self.validate_fit(fit)
        rows = self.shared_rows(len(study), np.asarray(fit.estimates, dtype=np.float64))
        return self.predict_rows(study, rows, mode=prediction_mode)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each residence time, by its density or by its survival if it was censored."""

        problem = self.read_problem(study)
        rows = self.row_coordinates(study, coefficients)
        scores, _ = self._scores_and_gradients(problem, rows)
        return protected_array(scores, dtype=np.float64)

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Draw a residence time on every row, truncating any that reaches its limit.

        The threshold is drawn once per patch visit from a logistic on the log intake rate,
        conditioned below that patch's entry rate -- which is the same conditioning the
        likelihood applies, drawn rather than integrated.
        """

        amounts, decays = self._patches(design)
        rows = self.row_coordinates(design, coefficients)
        noise = np.exp(rows[:, 1])
        entry = (self._log_intake(np.zeros_like(amounts), amounts, decays) - rows[:, 0]) / noise
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        ceiling = np.minimum(expit(entry), 1.0 - 1e-12)
        uniform = np.clip(generator.uniform(1e-300, ceiling), 1e-300, 1.0 - 1e-12)
        threshold = np.exp(rows[:, 0] + noise * (np.log(uniform) - np.log1p(-uniform)))
        times = _residence_at_rate(
            threshold, patch_yield=amounts, patch_decay=decays, gain=self.gain
        )
        if self.censoring_time_column is not None:
            limits = numeric_column(design, self.censoring_time_column, role="task")
            times = np.minimum(times, limits)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = np.asarray(times, dtype=np.float64)
        return Study(columns)

    # -- the box, the restarts and the derived quantities --------------------------------------

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:
        """Return the finite box the two logarithms are searched in.

        The threshold's box is derived from the study rather than declared, as
        :class:`~behavio.models.economic.TemporalDiscounting` derives its rate box from the
        delays that were used: a giving-up rate above every patch's entry rate is an animal
        that never enters, and one far below the rate any observed visit reached is an animal
        no residence time in this study can distinguish from a slightly lower one.
        """

        amounts, decays = self._patches(study)
        entry = self._log_intake(np.zeros_like(amounts), amounts, decays)
        upper = float(np.max(entry))
        if self.outcome in study.columns:
            observed = validated_durations(
                numeric_column(study, self.outcome, role="outcome"),
                label=f"outcome column {self.outcome!r}",
            )
            floor = float(np.min(self._log_intake(observed, amounts, decays)))
        else:
            floor = upper - 2.0 * _GIVING_UP_MARGIN
        lower = min(floor, upper - 1.0) - _GIVING_UP_MARGIN
        return np.asarray([[lower, upper], list(_NOISE_BOX)], dtype=np.float64)

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Start from the intake rate the median observed visit actually reached.

        Closed form and deterministic: a threshold model's median residence time *is* its
        threshold crossing, so the median observed visit's own intake rate is the design's own
        estimate of the giving-up rate, and the restarts vary the noise around it.
        """

        problem = self.read_problem(study)
        box = self.coordinate_box(study)
        rates = problem.log_observed_rate
        centre = float(np.median(rates)) if rates.size else float(np.mean(box[0]))
        spread = float(np.std(rates)) if rates.size else 0.5
        starts = [
            np.asarray([centre, float(np.log(max(scale * max(spread, 0.05), 1e-3)))])
            for scale in (1.0, 0.4, 2.5, 0.1)
        ]
        clipped = [np.clip(start, box[:, 0], box[:, 1]) for start in starts]
        return tuple(clipped[index % len(clipped)] for index in range(self.n_restarts))

    def derived_science(
        self,
        study: Study,
        estimates: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> tuple[DerivedQuantity, ...]:
        """Report Charnov's optimum beside the threshold that was actually fitted."""

        if self.travel_time_column is None:
            return ()
        amounts, decays = self._patches(study)
        travel = numeric_column(study, self.travel_time_column, role="task")
        if np.any(travel <= 0.0):
            raise ModelDataError("travel times must be positive")
        optimal_rate = marginal_value_rate(amounts, decays, float(np.mean(travel)), gain=self.gain)
        optimal_times = marginal_value_residence_time(
            amounts, decays, float(np.mean(travel)), gain=self.gain
        )
        fitted = float(np.exp(estimates[0]))
        ratio = optimal_rate / fitted
        return (
            DerivedQuantity(
                name="marginal_value_rate",
                value=float(optimal_rate),
                description=(
                    "Charnov (1976) optimal long-run intake rate for this environment; a "
                    "normative benchmark, not an estimate"
                ),
            ),
            DerivedQuantity(
                name="optimal_residence_time",
                value=float(np.mean(optimal_times)),
                description=(
                    "mean over rows of the residence time at which each patch's intake rate "
                    "reaches marginal_value_rate"
                ),
            ),
            DerivedQuantity(
                name="overstaying_ratio",
                value=float(ratio),
                standard_error=float(ratio * np.sqrt(max(float(covariance[0, 0]), 0.0))),
                description=(
                    "marginal_value_rate divided by the fitted giving_up_rate; above one is "
                    "an animal that leaves at a lower rate than optimal, which is to say it "
                    "overstays"
                ),
            ),
        )

    # -- findings ----------------------------------------------------------------------------

    def design_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report what this design can and cannot say about a rate threshold."""

        amounts, decays = self._patches(study)
        findings: list[ModelFinding] = []
        patches = np.unique(np.column_stack([amounts, decays]).round(12), axis=0)
        if len(patches) < 2:
            findings.append(
                ModelFinding(
                    code="unidentified_leaving_rule",
                    severity=WARNING,
                    message=(
                        "every patch in this study depletes identically, so the intake rate "
                        "is one fixed function of elapsed time and 'leave when the rate "
                        "falls to a threshold' predicts exactly what 'leave after a fixed "
                        "time' predicts; the marginal value theorem's content is that one "
                        "rate threshold governs different patches, and this design cannot "
                        "see it"
                    ),
                )
            )
        elif self.travel_time_column is not None:
            findings.append(
                ModelFinding(
                    code="heterogeneous_environment",
                    severity=WARNING,
                    message=(
                        f"this study contains {len(patches)} patch types, so the theorem "
                        "gives one marginal_value_rate for the environment and a different "
                        "optimal residence time in each type; optimal_residence_time is the "
                        "mean over rows and should be read per type"
                    ),
                )
            )
        findings.extend(self._censoring_findings(study))
        return tuple(findings)

    def _censoring_findings(self, study: Study) -> list[ModelFinding]:
        if self.outcome not in study.columns:
            return []
        times = validated_durations(
            numeric_column(study, self.outcome, role="outcome"),
            label=f"outcome column {self.outcome!r}",
        )
        if not times.size:
            return []
        if self.censoring_time_column is None:
            largest = float(np.max(times))
            tied = float(np.mean(np.isclose(times, largest, rtol=1e-9, atol=0.0)))
            if tied >= 0.1 and len(times) > 1:
                return [
                    ModelFinding(
                        code="undeclared_censoring",
                        severity=WARNING,
                        message=(
                            f"{tied:.0%} of residence times sit exactly on this study's "
                            f"longest one ({largest:g}), which is what right-censoring looks "
                            "like; this model declares none, so every one of them is being "
                            "scored as a completed departure. Declare "
                            "censoring_time_column, or state that the ties are real"
                        ),
                    )
                ]
            return []
        limits = numeric_column(study, self.censoring_time_column, role="task")
        censored = read_censoring(
            times, limits, label=f"censoring column {self.censoring_time_column!r}"
        )
        share = float(np.mean(censored))
        if share >= 1.0:
            return [
                ModelFinding(
                    code="all_rows_censored",
                    severity=WARNING,
                    message=(
                        "no visit in this study ended in a departure, so the likelihood is "
                        "made entirely of survival probabilities and the giving-up rate is "
                        "bounded rather than located"
                    ),
                )
            ]
        if share >= 0.25:
            return [
                ModelFinding(
                    code="heavy_censoring",
                    severity=WARNING,
                    message=(
                        f"{share:.0%} of visits were still in progress when observation "
                        "stopped. Those rows are scored by their survival function, which "
                        "the predicted density cannot express, so scoring "
                        "predict().observed_log_density() instead of pointwise_log_prob() "
                        "would misscore exactly this share of the study"
                    ),
                )
            ]
        return []
