"""Signal detection theory: yes/no, forced choice, unequal-variance ROC, and meta-d'.

This is a clean-room implementation from the published equations. Nothing here is derived
from ``psignifit``, ``metadpy``, ``psychofit`` or the released MATLAB toolboxes.

Conventions this module fixes, because published implementations disagree
------------------------------------------------------------------------
*Criterion sign and location.* The equal-variance model places noise at
:math:`-d'/2` and signal at :math:`+d'/2`, so that

.. math::

    H = \\Phi(d'/2 - c), \\quad F = \\Phi(-d'/2 - c), \\quad
    d' = z(H) - z(F), \\quad c = -\\tfrac{1}{2}\\left(z(H) + z(F)\\right).

A positive :math:`c` is therefore *conservative* (a bias against saying "yes"). The
relative criterion is :math:`c' = c/d'`.

*Bias in logs.* :math:`\\ln \\beta = c\\,d'` exactly. Because the literature reports
:math:`\\beta`, :math:`\\ln\\beta` and :math:`\\log_{10}\\beta` interchangeably and often
without saying which,:class:`SignalDetectionSummary` carries all three under unambiguous
names rather than one field called ``log_beta`` whose base the reader must guess.

*Extreme rates.* A hit rate of one or a false-alarm rate of zero makes :math:`d'`
infinite. Two standard repairs exist and they do not agree:
:attr:`RateCorrection.LOG_LINEAR` (Hautus, 1995) adds one half to all four cells of every
table, whether or not any rate is extreme, and so changes every estimate it touches;
:attr:`RateCorrection.ONE_OVER_2N` (Macmillan & Kaplan, 1985) replaces only a rate of
exactly zero or one by :math:`1/2N` or :math:`1 - 1/2N`, and so leaves non-extreme tables
alone. Neither is applied unless it is declared. :class:`CorrectedRates` records which
correction was declared *and* whether it actually changed anything, so a corrected rate can
never be read as an uncorrected one.

*Forced choice.* Two-alternative forced choice uses the exact relation
:math:`d' = \\sqrt{2}\\,z(P_c)`. That shortcut is wrong for :math:`m > 2`, which needs the
Green and Swets integral :math:`P_c = \\int \\varphi(x - d')\\,\\Phi(x)^{m-1}\\,dx` inverted
numerically. :func:`forced_choice_d_prime` dispatches on :math:`m` and the two agree
exactly at :math:`m = 2`.

*Unequal variance.* The z-ROC line is :math:`z(H) = a + s\\,z(F)` with slope
:math:`s = \\sigma_N/\\sigma_S`, so the signal standard deviation is :math:`1/s`. Then
:math:`d_a = \\sqrt{2}\\,a/\\sqrt{1 + s^2}` and the area under the binormal ROC is
:math:`A_z = \\Phi(d_a/\\sqrt{2})`. :func:`z_roc_summary` fits that line by ordinary least
squares on the observed operating points, which is the classical estimator;
:class:`UnequalVarianceSDT` fits the same model by maximum likelihood over the whole rating
table. They are different estimators of the same quantities and will not agree exactly.

*Meta-d'.* Meta-d' is the type-1 sensitivity that a type-1 observer would have needed in
order to produce the observed type-2 (confidence) data. It is fitted with type-1 d' and the
type-1 criterion held fixed: the criterion is held at a fixed *relative* position
:math:`c' = c/d'`, so the criterion used inside the meta model is
:math:`c_{\\mathrm{meta}} = \\mathrm{meta}\\text{-}d' \\times c/d'`. The likelihood is the
confidence distribution *conditional on the stimulus and the type-1 response*, which is what
makes the fit independent of type-1 performance. Type-2 criteria are parameterised as
positive gaps outward from the criterion, so they can never cross it and can never fall out
of order.

References
----------
Green, D. M., & Swets, J. A. (1966). *Signal Detection Theory and Psychophysics.* Wiley.

Grier, J. B. (1971). Nonparametric indexes for sensitivity and bias. *Psychological
Bulletin, 75*, 424-429.

Donaldson, W. (1992). Measuring recognition memory. *Journal of Experimental Psychology:
General, 121*, 275-277.

Macmillan, N. A., & Kaplan, H. L. (1985). Detection theory analysis of group data.
*Psychological Bulletin, 98*, 185-199.

Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on
estimated values of d'. *Behavior Research Methods, 27*, 46-51.

Macmillan, N. A., & Creelman, C. D. (2005). *Detection Theory: A User's Guide* (2nd ed.).
Erlbaum.

Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating
metacognitive sensitivity from confidence ratings. *Consciousness and Cognition, 21*,
422-430.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import ndtr, ndtri

from behavio._internal.arrays import protected_array
from behavio.contracts import (
    CategoricalPrediction,
    DerivedQuantity,
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    natural_quantities,
)
from behavio.models._kernels.curvature import (
    covariance_from_hessian,
    finite_difference_gradient,
    finite_difference_hessian,
)
from behavio.trials import REQUIRED_COLUMNS, Study

#: Probability floor used whenever a model cell would otherwise be exactly zero.
PROBABILITY_FLOOR = 1e-12

#: Widest sensitivity considered when inverting the m-alternative forced-choice integral.
MAXIMUM_FORCED_CHOICE_D_PRIME = 10.0

_LN_10 = float(np.log(10.0))


class RateCorrection(StrEnum):
    """A declared repair for hit and false-alarm rates of exactly zero or one."""

    #: No repair. An extreme rate yields an infinite ``d'``, which stays visible.
    NONE = "none"
    #: Hautus (1995): add 0.5 to every cell and 1 to every total, always.
    LOG_LINEAR = "log-linear"
    #: Macmillan and Kaplan (1985): replace only an extreme rate by ``1/2N`` or ``1-1/2N``.
    ONE_OVER_2N = "one-over-2n"


@dataclass(frozen=True, slots=True)
class DetectionCounts:
    """The four cells of a yes/no detection table."""

    hits: int
    misses: int
    false_alarms: int
    correct_rejections: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.hits, "hits"),
            (self.misses, "misses"),
            (self.false_alarms, "false_alarms"),
            (self.correct_rejections, "correct_rejections"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
            object.__setattr__(self, label, int(value))
        if self.hits + self.misses < 1 or self.false_alarms + self.correct_rejections < 1:
            raise ValueError("a detection table needs at least one signal and one noise trial")

    @property
    def n_signal(self) -> int:
        """Number of signal-present trials."""

        return self.hits + self.misses

    @property
    def n_noise(self) -> int:
        """Number of signal-absent trials."""

        return self.false_alarms + self.correct_rejections

    @classmethod
    def from_responses(
        cls,
        signal: Sequence[Any] | NDArray[Any],
        response: Sequence[Any] | NDArray[Any],
    ) -> DetectionCounts:
        """Tabulate zero/one signal and response vectors of equal length."""

        signals = _binary_vector(signal, "signal")
        responses = _binary_vector(response, "response")
        if signals.shape != responses.shape:
            raise ValueError("signal and response must have equal length")
        return cls(
            hits=int(np.count_nonzero((signals == 1) & (responses == 1))),
            misses=int(np.count_nonzero((signals == 1) & (responses == 0))),
            false_alarms=int(np.count_nonzero((signals == 0) & (responses == 1))),
            correct_rejections=int(np.count_nonzero((signals == 0) & (responses == 0))),
        )


@dataclass(frozen=True, slots=True)
class CorrectedRates:
    """Hit and false-alarm rates together with the correction that produced them."""

    hit_rate: float
    false_alarm_rate: float
    correction: RateCorrection
    correction_applied: bool
    n_signal: int
    n_noise: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "correction", RateCorrection(self.correction))
        for value, label in (
            (self.hit_rate, "hit_rate"),
            (self.false_alarm_rate, "false_alarm_rate"),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} must be a probability")
        if not isinstance(self.correction_applied, bool):
            raise ValueError("correction_applied must be boolean")
        if self.correction is RateCorrection.NONE and self.correction_applied:
            raise ValueError("an undeclared correction cannot have been applied")
        if self.n_signal < 1 or self.n_noise < 1:
            raise ValueError("rate denominators must be positive")

    @property
    def is_degenerate(self) -> bool:
        """Whether a rate still sits at exactly zero or one, making ``d'`` infinite."""

        return self.hit_rate in (0.0, 1.0) or self.false_alarm_rate in (0.0, 1.0)


def detection_rates(
    counts: DetectionCounts, *, correction: RateCorrection = RateCorrection.NONE
) -> CorrectedRates:
    """Return hit and false-alarm rates under an explicitly declared correction.

    ``correction`` is never inferred from the data. :attr:`RateCorrection.LOG_LINEAR`
    always reports ``correction_applied=True`` because it changes every table;
    :attr:`RateCorrection.ONE_OVER_2N` reports ``True`` only when a rate was actually
    extreme.
    """

    if not isinstance(counts, DetectionCounts):
        raise TypeError("counts must be a DetectionCounts")
    choice = RateCorrection(correction)
    n_signal, n_noise = counts.n_signal, counts.n_noise
    if choice is RateCorrection.LOG_LINEAR:
        return CorrectedRates(
            hit_rate=(counts.hits + 0.5) / (n_signal + 1.0),
            false_alarm_rate=(counts.false_alarms + 0.5) / (n_noise + 1.0),
            correction=choice,
            correction_applied=True,
            n_signal=n_signal,
            n_noise=n_noise,
        )
    hit_rate = counts.hits / n_signal
    false_alarm_rate = counts.false_alarms / n_noise
    if choice is RateCorrection.NONE:
        return CorrectedRates(
            hit_rate=hit_rate,
            false_alarm_rate=false_alarm_rate,
            correction=choice,
            correction_applied=False,
            n_signal=n_signal,
            n_noise=n_noise,
        )
    repaired_hit = _one_over_2n(hit_rate, n_signal)
    repaired_false_alarm = _one_over_2n(false_alarm_rate, n_noise)
    return CorrectedRates(
        hit_rate=repaired_hit,
        false_alarm_rate=repaired_false_alarm,
        correction=choice,
        correction_applied=bool(
            repaired_hit != hit_rate or repaired_false_alarm != false_alarm_rate
        ),
        n_signal=n_signal,
        n_noise=n_noise,
    )


@dataclass(frozen=True, slots=True)
class SignalDetectionSummary:
    """Equal-variance sensitivity and bias indices for one detection table.

    ``log_beta`` is the natural logarithm and equals ``criterion * d_prime`` exactly;
    ``log10_beta`` is the base-ten logarithm of the same quantity. Both are stored so that
    neither convention has to be guessed.
    """

    rates: CorrectedRates
    d_prime: float
    criterion: float
    relative_criterion: float
    beta: float
    log_beta: float
    log10_beta: float
    a_prime: float
    b_double_prime_d: float

    def __post_init__(self) -> None:
        if not isinstance(self.rates, CorrectedRates):
            raise TypeError("rates must be a CorrectedRates")

    @property
    def is_degenerate(self) -> bool:
        """Whether the summary rests on a rate of exactly zero or one."""

        return self.rates.is_degenerate


def equal_variance_summary(
    counts: DetectionCounts, *, correction: RateCorrection = RateCorrection.NONE
) -> SignalDetectionSummary:
    """Return equal-variance d', criterion, beta, and the nonparametric indices.

    ``A'`` follows Grier (1971) and ``B''_D`` follows Donaldson (1992). Both are computed
    from the same (possibly corrected) rates as the parametric indices, so a single
    :class:`CorrectedRates` describes the whole summary.
    """

    rates = detection_rates(counts, correction=correction)
    hit_rate, false_alarm_rate = rates.hit_rate, rates.false_alarm_rate
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z_hit = float(ndtri(hit_rate))
        z_false_alarm = float(ndtri(false_alarm_rate))
        d_prime = z_hit - z_false_alarm
        criterion = -0.5 * (z_hit + z_false_alarm)
        relative = criterion / d_prime if np.isfinite(d_prime) and d_prime != 0 else np.nan
        log_beta = criterion * d_prime
        beta = float(np.exp(log_beta)) if np.isfinite(log_beta) else float("nan")
    return SignalDetectionSummary(
        rates=rates,
        d_prime=float(d_prime),
        criterion=float(criterion),
        relative_criterion=float(relative),
        beta=beta,
        log_beta=float(log_beta),
        log10_beta=float(log_beta / _LN_10),
        a_prime=_a_prime(hit_rate, false_alarm_rate),
        b_double_prime_d=_b_double_prime_d(hit_rate, false_alarm_rate),
    )


def forced_choice_proportion_correct(d_prime: float, *, n_alternatives: int) -> float:
    """Return the unbiased proportion correct for ``m``-alternative forced choice.

    Uses :math:`P_c = \\Phi(d'/\\sqrt{2})` for two alternatives and the Green and Swets
    integral :math:`\\int \\varphi(x - d')\\,\\Phi(x)^{m-1}\\,dx` for more.
    """

    alternatives = _validate_alternatives(n_alternatives)
    value = float(d_prime)
    if not np.isfinite(value):
        raise ValueError("d_prime must be finite")
    if alternatives == 2:
        return float(ndtr(value / np.sqrt(2.0)))
    return _forced_choice_integral(value, alternatives)


def forced_choice_d_prime(proportion_correct: float, *, n_alternatives: int) -> float:
    """Return ``d'`` for an observed ``m``-alternative forced-choice proportion correct.

    Two alternatives use the exact relation :math:`d' = \\sqrt{2}\\,z(P_c)`. More than two
    inverts :func:`forced_choice_proportion_correct` numerically; the 2AFC shortcut is
    *not* a valid approximation there.
    """

    alternatives = _validate_alternatives(n_alternatives)
    proportion = float(proportion_correct)
    if not np.isfinite(proportion) or not 0 < proportion < 1:
        raise ValueError("proportion_correct must lie strictly between zero and one")
    if alternatives == 2:
        return float(np.sqrt(2.0) * ndtri(proportion))
    low, high = -MAXIMUM_FORCED_CHOICE_D_PRIME, MAXIMUM_FORCED_CHOICE_D_PRIME
    lowest = _forced_choice_integral(low, alternatives)
    highest = _forced_choice_integral(high, alternatives)
    if not lowest < proportion < highest:
        raise ValueError(
            f"proportion_correct {proportion} is outside the range reachable by "
            f"|d'| <= {MAXIMUM_FORCED_CHOICE_D_PRIME} with {alternatives} alternatives"
        )
    return float(
        brentq(
            lambda value: _forced_choice_integral(value, alternatives) - proportion,
            low,
            high,
            xtol=1e-12,
            rtol=1e-14,
        )
    )


@dataclass(frozen=True, slots=True)
class RatingCounts:
    """Confidence-rating counts for signal and noise trials on one ordered scale.

    Index zero is the rating most strongly indicating *noise* and the final index the
    rating most strongly indicating *signal*, so cumulating from the end produces ROC
    operating points in the usual order.
    """

    signal: tuple[int, ...]
    noise: tuple[int, ...]

    def __post_init__(self) -> None:
        signal = tuple(int(value) for value in self.signal)
        noise = tuple(int(value) for value in self.noise)
        if len(signal) < 2 or len(signal) != len(noise):
            raise ValueError("signal and noise counts must share at least two rating levels")
        if any(value < 0 for value in (*signal, *noise)):
            raise ValueError("rating counts must be non-negative")
        if sum(signal) < 1 or sum(noise) < 1:
            raise ValueError("rating counts need at least one signal and one noise trial")
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "noise", noise)

    @property
    def n_ratings(self) -> int:
        """Number of ordered rating levels."""

        return len(self.signal)

    @classmethod
    def from_responses(
        cls,
        signal: Sequence[Any] | NDArray[Any],
        rating: Sequence[Any] | NDArray[Any],
        *,
        levels: Sequence[Any],
    ) -> RatingCounts:
        """Tabulate ratings against a declared, ordered level vocabulary."""

        signals = _binary_vector(signal, "signal")
        ordered = tuple(levels)
        if len(ordered) < 2 or len(set(ordered)) != len(ordered):
            raise ValueError("levels must contain at least two distinct rating levels")
        ratings = np.asarray(rating)
        if ratings.shape != signals.shape:
            raise ValueError("signal and rating must have equal length")
        unknown = set(np.asarray(ratings).tolist()) - set(ordered)
        if unknown:
            raise ValueError(f"observed ratings outside the declared levels: {sorted(unknown)}")
        return cls(
            signal=tuple(
                int(np.count_nonzero((signals == 1) & (ratings == level))) for level in ordered
            ),
            noise=tuple(
                int(np.count_nonzero((signals == 0) & (ratings == level))) for level in ordered
            ),
        )


def roc_points(counts: RatingCounts) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(false_alarm_rates, hit_rates)`` including the ``(0, 0)`` and ``(1, 1)`` ends.

    Operating points are cumulated from the most signal-like rating downward, which is the
    standard rating-ROC construction.
    """

    if not isinstance(counts, RatingCounts):
        raise TypeError("counts must be a RatingCounts")
    signal = np.asarray(counts.signal, dtype=np.float64)
    noise = np.asarray(counts.noise, dtype=np.float64)
    hits = np.cumsum(signal[::-1])[::-1] / signal.sum()
    false_alarms = np.cumsum(noise[::-1])[::-1] / noise.sum()
    hit_rates = np.concatenate(([0.0], hits[::-1]))
    false_alarm_rates = np.concatenate(([0.0], false_alarms[::-1]))
    return (
        protected_array(false_alarm_rates, dtype=np.float64),
        protected_array(hit_rates, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class ZRocSummary:
    """A least-squares z-ROC fit and the unequal-variance indices it implies."""

    false_alarm_rates: NDArray[np.float64]
    hit_rates: NDArray[np.float64]
    intercept: float
    slope: float
    signal_sd: float
    d_a: float
    area_under_curve: float
    empirical_area: float
    correction: RateCorrection
    correction_applied: bool
    n_points: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "correction", RateCorrection(self.correction))
        object.__setattr__(
            self, "false_alarm_rates", protected_array(self.false_alarm_rates, dtype=np.float64)
        )
        object.__setattr__(self, "hit_rates", protected_array(self.hit_rates, dtype=np.float64))
        if self.n_points < 2:
            raise ValueError("a z-ROC line needs at least two interior operating points")


def z_roc_summary(
    counts: RatingCounts, *, correction: RateCorrection = RateCorrection.NONE
) -> ZRocSummary:
    """Fit ``z(H) = a + s z(F)`` by ordinary least squares over the interior ROC points.

    Each interior operating point is a two-by-two table in its own right, so the declared
    :class:`RateCorrection` is applied point by point. Points that remain degenerate are
    dropped from the regression and the retained count is reported as ``n_points``.
    """

    if not isinstance(counts, RatingCounts):
        raise TypeError("counts must be a RatingCounts")
    signal = np.asarray(counts.signal, dtype=np.float64)
    noise = np.asarray(counts.noise, dtype=np.float64)
    cumulative_hits = np.cumsum(signal[::-1])[::-1]
    cumulative_false_alarms = np.cumsum(noise[::-1])[::-1]
    n_signal, n_noise = int(signal.sum()), int(noise.sum())
    hit_rates: list[float] = []
    false_alarm_rates: list[float] = []
    applied = False
    for index in range(1, counts.n_ratings):
        table = DetectionCounts(
            hits=int(cumulative_hits[index]),
            misses=n_signal - int(cumulative_hits[index]),
            false_alarms=int(cumulative_false_alarms[index]),
            correct_rejections=n_noise - int(cumulative_false_alarms[index]),
        )
        rates = detection_rates(table, correction=correction)
        applied = applied or rates.correction_applied
        hit_rates.append(rates.hit_rate)
        false_alarm_rates.append(rates.false_alarm_rate)
    hits = np.asarray(hit_rates, dtype=np.float64)
    false_alarms = np.asarray(false_alarm_rates, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_hits = ndtri(hits)
        z_false_alarms = ndtri(false_alarms)
    usable = np.isfinite(z_hits) & np.isfinite(z_false_alarms)
    if int(np.count_nonzero(usable)) < 2:
        raise ModelDataError(
            "a z-ROC line needs at least two non-degenerate operating points; declare a "
            "RateCorrection or pool rating levels"
        )
    design = np.column_stack([np.ones(int(np.count_nonzero(usable))), z_false_alarms[usable]])
    coefficients, *_ = np.linalg.lstsq(design, z_hits[usable], rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])
    d_a = float(np.sqrt(2.0) * intercept / np.sqrt(1.0 + slope**2))
    curve_false_alarms, curve_hits = roc_points(counts)
    return ZRocSummary(
        false_alarm_rates=false_alarms,
        hit_rates=hits,
        intercept=intercept,
        slope=slope,
        signal_sd=float(1.0 / slope) if slope != 0 else float("inf"),
        d_a=d_a,
        area_under_curve=float(ndtr(d_a / np.sqrt(2.0))),
        empirical_area=float(np.trapezoid(curve_hits, curve_false_alarms)),
        correction=RateCorrection(correction),
        correction_applied=applied,
        n_points=int(np.count_nonzero(usable)),
    )


def _summary_quantities(summary: SignalDetectionSummary) -> tuple[DerivedQuantity, ...]:
    """Return the bias and nonparametric indices of one equal-variance summary.

    ``d_prime`` and ``criterion`` are absent because they *are* the estimated coordinate
    and already carry standard errors on the fit. Everything here is a function of them or
    of the rates, and none of these indices has an agreed sampling distribution, so each is
    reported without an uncertainty rather than with a fabricated one.
    """

    rates = summary.rates
    return (
        DerivedQuantity("hit_rate", rates.hit_rate, description="corrected hit rate"),
        DerivedQuantity(
            "false_alarm_rate", rates.false_alarm_rate, description="corrected false-alarm rate"
        ),
        DerivedQuantity(
            "relative_criterion", summary.relative_criterion, description="c divided by d'"
        ),
        DerivedQuantity("beta", summary.beta, description="likelihood ratio at the criterion"),
        DerivedQuantity("log_beta", summary.log_beta, description="natural log of beta"),
        DerivedQuantity("log10_beta", summary.log10_beta, description="base-ten log of beta"),
        DerivedQuantity("a_prime", summary.a_prime, description="Grier (1971) A'"),
        DerivedQuantity(
            "b_double_prime_d", summary.b_double_prime_d, description="Donaldson (1992) B''_D"
        ),
    )


@dataclass(frozen=True, slots=True)
class EqualVarianceSDT:
    """Equal-variance yes/no detection with a declared correction for extreme rates.

    The fit is the closed-form z-transform, which is the maximum-likelihood estimator when
    no correction is declared. Standard errors are the classical delta-method values
    :math:`\\operatorname{var}(z(p)) = p(1-p)/(N\\varphi(z(p))^2)`, propagated through
    :math:`d' = z(H) - z(F)` and :math:`c = -(z(H) + z(F))/2`; the two are correlated
    whenever the signal and noise trial counts or rates differ.
    """

    signal: str = "signal"
    response: str = "response"
    correction: RateCorrection = RateCorrection.NONE

    def __post_init__(self) -> None:
        _validate_column(self.signal, "signal")
        _validate_column(self.response, "response")
        if self.signal == self.response:
            raise ValueError("signal and response columns must be distinct")
        object.__setattr__(self, "correction", RateCorrection(self.correction))

    @property
    def model_name(self) -> str:
        return "equal-variance-sdt"

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[signal={self.signal};response={self.response};"
            f"correction={self.correction.value}]"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("d_prime", "criterion")

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.response,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.signal,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def counts(self, study: Study) -> DetectionCounts:
        """Tabulate the study into the four cells of a detection table."""

        return DetectionCounts.from_responses(
            _study_binary(study, self.signal, "signal"),
            _study_binary(study, self.response, "response"),
        )

    def summarize(self, study: Study) -> SignalDetectionSummary:
        """Return the closed-form summary under this model's declared correction.

        This is where the structured record lives. :meth:`fit` used to return a
        ``SignalDetectionFitResult`` carrying the same :class:`SignalDetectionSummary` as
        a field, which duplicated it: the summary is a deterministic function of the study
        and this model's declared correction, not of the fit, and a fit result exists to
        carry what only fitting produced. Its scalar indices still travel on the fit as
        :attr:`~behavio.contracts.FitResult.derived`.
        """

        return equal_variance_summary(self.counts(study), correction=self.correction)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        d_prime, criterion = self._parameter_vector(parameters)
        signals = _study_binary(design, self.signal, "signal")
        probability = self._yes_probability(signals, d_prime, criterion)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.response] = generator.binomial(1, probability).astype(np.int8)
        return Study(columns)

    def fit(self, study: Study) -> FitResult:
        counts = self.counts(study)
        summary = equal_variance_summary(counts, correction=self.correction)
        if summary.is_degenerate:
            raise ModelDataError(
                "an extreme hit or false-alarm rate makes d' infinite; declare "
                "RateCorrection.LOG_LINEAR (Hautus 1995, applied to every table) or "
                "RateCorrection.ONE_OVER_2N (Macmillan and Kaplan 1985, applied only to "
                "extreme rates) rather than leaving the choice implicit"
            )
        rates = summary.rates
        variance_hit = _z_variance(rates.hit_rate, rates.n_signal)
        variance_false_alarm = _z_variance(rates.false_alarm_rate, rates.n_noise)
        covariance = np.asarray(
            [
                [
                    variance_hit + variance_false_alarm,
                    0.5 * (variance_false_alarm - variance_hit),
                ],
                [
                    0.5 * (variance_false_alarm - variance_hit),
                    0.25 * (variance_hit + variance_false_alarm),
                ],
            ],
            dtype=np.float64,
        )
        estimates = np.asarray([summary.d_prime, summary.criterion], dtype=np.float64)
        signals = _study_binary(study, self.signal, "signal")
        responses = _study_binary(study, self.response, "response")
        probability = self._yes_probability(signals, summary.d_prime, summary.criterion)
        objective = -float(np.sum(_bernoulli_log_prob(responses, probability)))
        raw = detection_rates(counts, correction=RateCorrection.NONE)
        diagnostics = FitDiagnostics.closed_form(
            procedure="closed-form z-transform",
            message=(
                f"z-transform of {rates.n_signal} signal and {rates.n_noise} noise trials "
                f"under correction={rates.correction.value} "
                f"(applied={rates.correction_applied})"
            ),
            objective=objective,
            boundary_estimate=bool(raw.is_degenerate),
        )
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            derived=_summary_quantities(summary),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        _prediction_mode(self.model_name, mode)
        d_prime, criterion = self._fit_vector(fit)
        signals = _study_binary(study, self.signal, "signal")
        evidence = (2.0 * signals - 1.0) * d_prime / 2.0 - criterion
        return Prediction(
            probability=ndtr(evidence),
            linear_predictor=evidence,
            mode=PredictionMode.FILTERED,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        responses = _study_binary(study, self.response, "response")
        probability = self.predict(study, fit, mode=mode).probability
        return protected_array(_bernoulli_log_prob(responses, probability), dtype=np.float64)

    @staticmethod
    def _yes_probability(
        signals: NDArray[np.float64], d_prime: float, criterion: float
    ) -> NDArray[np.float64]:
        return ndtr((2.0 * signals - 1.0) * d_prime / 2.0 - criterion)

    def _parameter_vector(self, parameters: Mapping[str, float]) -> tuple[float, float]:
        values = _parameter_vector(parameters, self.parameter_names)
        return float(values[0]), float(values[1])

    def _fit_vector(self, fit: FitResult) -> tuple[float, float]:
        _validate_fit(self, fit)
        return float(fit.estimates[0]), float(fit.estimates[1])


@dataclass(frozen=True, slots=True)
class UnequalVarianceSDT:
    """Unequal-variance SDT fitted by maximum likelihood over a confidence-rating table.

    Noise evidence is standard normal by construction; signal evidence is
    :math:`N(\\mu, \\sigma)`. The ``K`` ordered ratings are produced by ``K-1`` strictly
    increasing criteria, parameterised as a first criterion plus positive gaps so the
    ordering can never be violated during optimization. Because the noise distribution
    fixes the scale, ``K`` must be at least three for the model to be identified.
    """

    signal: str = "signal"
    rating: str = "rating"
    ratings: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    n_restarts: int = 3
    max_iterations: int = 1_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        _validate_column(self.signal, "signal")
        _validate_column(self.rating, "rating")
        if self.signal == self.rating:
            raise ValueError("signal and rating columns must be distinct")
        levels = tuple(int(value) for value in self.ratings)
        if len(levels) < 3 or len(set(levels)) != len(levels):
            raise ValueError("unequal-variance SDT needs at least three distinct rating levels")
        if any(later <= earlier for earlier, later in pairwise(levels)):
            raise ValueError("ratings must be given in increasing order")
        for value, label in (
            (self.n_restarts, "n_restarts"),
            (self.max_iterations, "max_iterations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        object.__setattr__(self, "ratings", levels)

    @property
    def model_name(self) -> str:
        return "unequal-variance-sdt"

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[signal={self.signal};rating={self.rating};ratings={self.ratings}]"
        )

    @property
    def n_ratings(self) -> int:
        """Number of ordered rating levels the model represents."""

        return len(self.ratings)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        gaps = tuple(f"log_criterion_gap_{index}" for index in range(2, self.n_ratings))
        return ("signal_mean", "log_signal_sd", "criterion_1", *gaps)

    @property
    def categories(self) -> tuple[Any, ...]:
        return self.ratings

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.rating,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.signal,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate: a signal distribution and ordered criteria."""

        criteria = tuple(f"criterion_{index}" for index in range(1, self.n_ratings))
        return ("signal_mean", "signal_sd", *criteria)

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Decode the estimated coordinate into a signal distribution and criteria."""

        coordinate = _natural_input(estimates, len(self.parameter_names))
        criteria = self._criteria(coordinate)
        decoded = {
            "signal_mean": float(coordinate[0]),
            "signal_sd": float(np.exp(coordinate[1])),
        }
        for index, value in enumerate(criteria.tolist(), start=1):
            decoded[f"criterion_{index}"] = float(value)
        return MappingProxyType(decoded)

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Encode a signal distribution and ordered criteria onto the estimated coordinate."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        criteria = [natural[f"criterion_{index}"] for index in range(1, self.n_ratings)]
        return self.parameters_from_components(
            signal_mean=natural["signal_mean"],
            signal_sd=natural["signal_sd"],
            criteria=criteria,
        )

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return the derivative of the criteria and signal distribution.

        A criterion is the first criterion plus the cumulative sum of positive gaps, so
        criterion ``i`` depends on the first criterion and on every gap up to ``i``. Writing
        that dependence down is exactly what lets a delta-method standard error be reported
        on a criterion, which the optimizer never estimates directly.
        """

        coordinate = _natural_input(estimates, len(self.parameter_names))
        width = len(self.parameter_names)
        jacobian = np.zeros((len(self.natural_names), width), dtype=np.float64)
        jacobian[0, 0] = 1.0
        jacobian[1, 1] = float(np.exp(coordinate[1]))
        gaps = np.exp(np.asarray(coordinate[3:], dtype=np.float64))
        for row in range(self.n_ratings - 1):
            jacobian[2 + row, 2] = 1.0
            for gap_index in range(row):
                jacobian[2 + row, 3 + gap_index] = float(gaps[gap_index])
        return jacobian

    def parameters_from_components(
        self, *, signal_mean: float, signal_sd: float, criteria: Sequence[float]
    ) -> Mapping[str, float]:
        """Encode a signal distribution and ordered criteria onto the estimated coordinate.

        Keyword-argument façade over :meth:`from_natural`, which owns the encoding.
        """

        values = tuple(float(value) for value in criteria)
        if len(values) != self.n_ratings - 1:
            raise ValueError(f"criteria must contain {self.n_ratings - 1} values")
        if any(later <= earlier for earlier, later in pairwise(values)):
            raise ValueError("criteria must be strictly increasing")
        if not np.isfinite(signal_sd) or signal_sd <= 0:
            raise ValueError("signal_sd must be positive")
        encoded = {
            "signal_mean": float(signal_mean),
            "log_signal_sd": float(np.log(signal_sd)),
            "criterion_1": values[0],
        }
        for index in range(1, len(values)):
            encoded[f"log_criterion_gap_{index + 1}"] = float(
                np.log(values[index] - values[index - 1])
            )
        return MappingProxyType(encoded)

    def counts(self, study: Study) -> RatingCounts:
        """Tabulate the study into signal and noise counts per rating level."""

        return RatingCounts.from_responses(
            _study_binary(study, self.signal, "signal"),
            self._ratings(study),
            levels=self.ratings,
        )

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        """Return the zero-based rating index observed on every trial."""

        observed = self._ratings(study)
        index = np.searchsorted(np.asarray(self.ratings, dtype=np.int64), observed)
        return protected_array(index, dtype=np.int64)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        coordinate = _parameter_vector(parameters, self.parameter_names)
        signals = _study_binary(design, self.signal, "signal")
        probability = self._cell_probabilities(coordinate, signals)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        draws = _categorical_draws(generator, probability)
        columns = {name: design[name] for name in design.columns}
        columns[self.rating] = np.asarray(self.ratings, dtype=np.int64)[draws]
        return Study(columns)

    def fit(self, study: Study) -> FitResult:
        counts = self.counts(study)
        starts, bounds = self._start_schedule(counts)
        objective = self._objective_factory(counts)
        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=lambda values: finite_difference_gradient(objective, values),
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
        result = results[int(np.argmin(objectives))]
        estimates = np.asarray(result.x, dtype=np.float64)
        gradient = finite_difference_gradient(objective, estimates)
        hessian = finite_difference_hessian(
            lambda values: finite_difference_gradient(objective, values),
            estimates,
            relative_step=1e-3,
        )
        covariance = covariance_from_hessian(hessian)
        signal_mean = float(estimates[0])
        signal_sd = float(np.exp(estimates[1]))
        area = float(ndtr(signal_mean / np.sqrt(1.0 + signal_sd**2)))
        d_a = float(np.sqrt(2.0) * signal_mean / np.sqrt(1.0 + signal_sd**2))
        indices = (
            DerivedQuantity("d_a", d_a, description="sqrt(2) mu / sqrt(1 + sigma^2)"),
            DerivedQuantity(
                "z_roc_slope", float(1.0 / signal_sd), description="slope of the z-ROC line"
            ),
            DerivedQuantity(
                "area_under_curve", area, description="binormal ROC area Phi(d_a / sqrt(2))"
            ),
        )
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            covariance=covariance,
            n_observations=len(study),
            derived=(*natural_quantities(self, estimates, covariance), *indices),
            diagnostics=FitDiagnostics(
                converged=bool(result.success),
                optimizer="L-BFGS-B deterministic multistart",
                status=int(result.status),
                message=str(result.message),
                n_iterations=int(result.nit),
                objective=float(result.fun),
                gradient_norm=float(np.linalg.norm(gradient)),
                hessian_condition=float(np.linalg.cond(hessian)),
                boundary_estimate=bool(
                    any(
                        abs(value - low) < 1e-6 or abs(value - high) < 1e-6
                        for value, (low, high) in zip(estimates, bounds, strict=True)
                    )
                ),
            ),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> CategoricalPrediction:
        _prediction_mode(self.model_name, mode)
        _validate_fit(self, fit)
        signals = _study_binary(study, self.signal, "signal")
        probability = self._cell_probabilities(np.asarray(fit.estimates), signals)
        return CategoricalPrediction(
            probability=probability,
            linear_predictor=np.log(np.clip(probability, PROBABILITY_FLOOR, 1.0)),
            categories=self.categories,
            mode=PredictionMode.FILTERED,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        codes = self.outcome_codes(study)
        probability = self.predict(study, fit, mode=mode).probability
        selected = probability[np.arange(len(codes)), codes]
        return protected_array(np.log(np.clip(selected, PROBABILITY_FLOOR, 1.0)), dtype=np.float64)

    def _criteria(self, coordinate: NDArray[np.float64]) -> NDArray[np.float64]:
        gaps = np.exp(np.asarray(coordinate[3:], dtype=np.float64))
        return np.concatenate(([float(coordinate[2])], float(coordinate[2]) + np.cumsum(gaps)))

    def _rating_probabilities(
        self, coordinate: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        criteria = self._criteria(coordinate)
        signal_mean = float(coordinate[0])
        signal_sd = float(np.exp(coordinate[1]))
        boundaries = np.concatenate(([-np.inf], criteria, [np.inf]))
        noise = np.diff(ndtr(boundaries))
        signal = np.diff(ndtr((boundaries - signal_mean) / signal_sd))
        return (
            np.clip(noise, PROBABILITY_FLOOR, 1.0),
            np.clip(signal, PROBABILITY_FLOOR, 1.0),
        )

    def _cell_probabilities(
        self, coordinate: NDArray[np.float64], signals: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        noise, signal = self._rating_probabilities(coordinate)
        table = np.vstack([noise / noise.sum(), signal / signal.sum()])
        return table[signals.astype(np.int64), :]

    def _objective_factory(self, counts: RatingCounts):
        noise_counts = np.asarray(counts.noise, dtype=np.float64)
        signal_counts = np.asarray(counts.signal, dtype=np.float64)

        def objective(coordinate: NDArray[np.float64]) -> float:
            noise, signal = self._rating_probabilities(np.asarray(coordinate, dtype=np.float64))
            return -float(
                noise_counts @ np.log(noise / noise.sum())
                + signal_counts @ np.log(signal / signal.sum())
            )

        return objective

    def _start_schedule(
        self, counts: RatingCounts
    ) -> tuple[list[NDArray[np.float64]], tuple[tuple[float, float], ...]]:
        noise = np.asarray(counts.noise, dtype=np.float64)
        cumulative = np.cumsum(noise)[:-1] / noise.sum()
        criteria = ndtri(np.clip(cumulative, 0.02, 0.98))
        gaps = np.maximum(np.diff(criteria), 1e-2)
        schedule = ((1.0, 1.0), (0.5, 1.25), (1.5, 0.8), (0.0, 1.0))
        starts: list[NDArray[np.float64]] = []
        for index in range(self.n_restarts):
            mean_start, sd_start = schedule[index % len(schedule)]
            starts.append(
                np.concatenate(
                    (
                        [mean_start, float(np.log(sd_start)), float(criteria[0])],
                        np.log(gaps),
                    )
                )
            )
        bounds = [(-10.0, 10.0), (-3.0, 3.0), (-8.0, 8.0)]
        bounds.extend((-8.0, 4.0) for _ in range(self.n_ratings - 2))
        return starts, tuple(bounds)

    def _ratings(self, study: Study) -> NDArray[np.int64]:
        if self.rating not in study.columns:
            raise ModelDataError(f"study is missing rating column {self.rating!r}")
        try:
            values = np.asarray(study[self.rating], dtype=np.int64)
        except (TypeError, ValueError):
            raise ModelDataError("ratings must be integer confidence levels") from None
        unknown = sorted(set(values.tolist()) - set(self.ratings))
        if unknown:
            raise ModelDataError(f"observed ratings outside the declared levels: {unknown}")
        return values


@dataclass(frozen=True, slots=True)
class MetaSDT:
    """Meta-d' after Maniscalco and Lau (2012), with the type-1 fit held fixed.

    The scored observation is the *joint* response-and-confidence cell, so the model
    predicts a categorical distribution over ``2K`` cells named by the composite
    categories ``(response, confidence)`` -- ``(0, 3)`` is "responded no at confidence 3"
    -- under the declared factorisation ``("response", "confidence")``. A caller wanting
    the response margin calls
    :meth:`~behavio.contracts.CategoricalPrediction.marginal`; nothing has to parse a
    label. Its likelihood factorises exactly as the estimator does: the type-1
    response probability comes from the type-1 model :math:`(d', c)`, and the confidence
    distribution conditional on stimulus and response comes from the meta model. Fitting
    therefore proceeds in two stages -- a closed-form type-1 z-transform, then a
    constrained maximisation over meta-d' and the type-2 criteria -- and the reported
    covariance is block diagonal by construction, because the stages do not share
    information. That is a property of the published estimator, not an approximation
    introduced here.

    Unlike the released MATLAB implementation, no padding is added to zero confidence cells
    by default: a zero cell contributes nothing to the likelihood and needs no repair. The
    type-1 table is a different matter, and its treatment is governed by ``correction``.
    """

    signal: str = "signal"
    response: str = "response"
    confidence: str = "confidence"
    confidence_levels: tuple[int, ...] = (1, 2, 3, 4)
    correction: RateCorrection = RateCorrection.NONE
    maximum_meta_d_prime: float = 10.0
    n_restarts: int = 3
    max_iterations: int = 2_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for value, label in (
            (self.signal, "signal"),
            (self.response, "response"),
            (self.confidence, "confidence"),
        ):
            _validate_column(value, label)
        if len({self.signal, self.response, self.confidence}) != 3:
            raise ValueError("signal, response, and confidence columns must be distinct")
        levels = tuple(int(value) for value in self.confidence_levels)
        if len(levels) < 2 or len(set(levels)) != len(levels):
            raise ValueError("meta-d' needs at least two distinct confidence levels")
        if any(later <= earlier for earlier, later in pairwise(levels)):
            raise ValueError("confidence_levels must be given in increasing order")
        object.__setattr__(self, "confidence_levels", levels)
        object.__setattr__(self, "correction", RateCorrection(self.correction))
        if not np.isfinite(self.maximum_meta_d_prime) or self.maximum_meta_d_prime <= 0:
            raise ValueError("maximum_meta_d_prime must be finite and positive")
        for value, label in (
            (self.n_restarts, "n_restarts"),
            (self.max_iterations, "max_iterations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

    @property
    def model_name(self) -> str:
        return "meta-d-prime"

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[signal={self.signal};response={self.response};"
            f"confidence={self.confidence};levels={self.confidence_levels};"
            f"correction={self.correction.value}]"
        )

    @property
    def n_confidence(self) -> int:
        """Number of ordered confidence levels."""

        return len(self.confidence_levels)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        gaps = tuple(
            f"log_gap_{side}_{index}"
            for side in ("no", "yes")
            for index in range(1, self.n_confidence)
        )
        return ("d_prime", "criterion", "meta_d_prime", *gaps)

    @property
    def categories(self) -> tuple[Any, ...]:
        """The ``2K`` joint cells as ``(response, confidence)`` pairs.

        Ordered from most confident "no" through to most confident "yes", which is the
        order :meth:`outcome_codes` assigns and the order a type-2 ROC is read in.
        """

        no = tuple((0, level) for level in reversed(self.confidence_levels))
        yes = tuple((1, level) for level in self.confidence_levels)
        return (*no, *yes)

    @property
    def category_factors(self) -> tuple[str, ...]:
        """The factors a joint category names, which are the scored columns themselves."""

        return self.scored_columns

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.response, self.confidence)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.signal,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate: the type-1 anchor and ordered type-2 criteria."""

        criteria = tuple(
            f"type2_criterion_{side}_{index}"
            for side in ("no", "yes")
            for index in range(1, self.n_confidence)
        )
        return ("d_prime", "criterion", "meta_d_prime", *criteria)

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Decode the estimated coordinate into ordered type-2 criteria."""

        coordinate = _natural_input(estimates, len(self.parameter_names))
        criteria_no, criteria_yes = self._type2_criteria(coordinate[2:])
        decoded = {
            "d_prime": float(coordinate[0]),
            "criterion": float(coordinate[1]),
            "meta_d_prime": float(coordinate[2]),
        }
        for side, values in (("no", criteria_no), ("yes", criteria_yes)):
            for index, value in enumerate(values.tolist(), start=1):
                decoded[f"type2_criterion_{side}_{index}"] = float(value)
        return MappingProxyType(decoded)

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Encode a type-1 anchor and ordered type-2 criteria onto the estimated coordinate."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        sides = {
            side: [
                natural[f"type2_criterion_{side}_{index}"] for index in range(1, self.n_confidence)
            ]
            for side in ("no", "yes")
        }
        return self.parameters_from_components(
            d_prime=natural["d_prime"],
            criterion=natural["criterion"],
            meta_d_prime=natural["meta_d_prime"],
            type2_criteria_no=sides["no"],
            type2_criteria_yes=sides["yes"],
        )

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return the derivative of the type-2 criteria with respect to their log gaps.

        The type-1 anchor and meta-d' are estimated directly, so the first three rows are
        the identity. Each type-2 criterion is a signed cumulative sum of positive gaps, so
        it depends on every gap up to its own index on its own side and on nothing else --
        which is precisely why the criteria can never cross the type-1 criterion or fall
        out of order.
        """

        coordinate = _natural_input(estimates, len(self.parameter_names))
        width = self.n_confidence - 1
        jacobian = np.zeros((len(self.natural_names), len(self.parameter_names)), np.float64)
        for index in range(3):
            jacobian[index, index] = 1.0
        gaps = np.exp(np.asarray(coordinate[3:], dtype=np.float64))
        for side_index, sign in enumerate((-1.0, 1.0)):
            offset = side_index * width
            for row in range(width):
                for gap_index in range(row + 1):
                    jacobian[3 + offset + row, 3 + offset + gap_index] = sign * float(
                        gaps[offset + gap_index]
                    )
        return jacobian

    def parameters_from_components(
        self,
        *,
        d_prime: float,
        criterion: float,
        meta_d_prime: float,
        type2_criteria_no: Sequence[float],
        type2_criteria_yes: Sequence[float],
    ) -> Mapping[str, float]:
        """Encode a type-1 anchor and ordered type-2 criteria onto the estimated coordinate.

        Keyword-argument façade over :meth:`from_natural`, which owns the encoding. The
        type-2 criteria are given relative to the type-1 criterion: negative and
        decreasing for "no" responses, positive and increasing for "yes" responses.
        """

        if not np.isfinite(d_prime) or d_prime <= 0:
            raise ValueError("meta-d' is defined relative to a positive type-1 d'")
        if not 0 < meta_d_prime <= self.maximum_meta_d_prime:
            raise ValueError("meta_d_prime must lie in (0, maximum_meta_d_prime]")
        no = tuple(float(value) for value in type2_criteria_no)
        yes = tuple(float(value) for value in type2_criteria_yes)
        expected = self.n_confidence - 1
        if len(no) != expected or len(yes) != expected:
            raise ValueError(f"each response side needs {expected} type-2 criteria")
        encoded: dict[str, float] = {
            "d_prime": float(d_prime),
            "criterion": float(criterion),
            "meta_d_prime": float(meta_d_prime),
        }
        for side, values, sign in (("no", no, -1.0), ("yes", yes, 1.0)):
            scaled = [sign * value for value in values]
            if any(value <= 0 for value in scaled):
                raise ValueError(f"type-2 criteria for {side!r} responses are on the wrong side")
            if any(later <= earlier for earlier, later in pairwise(scaled)):
                raise ValueError(f"type-2 criteria for {side!r} responses must order outward")
            gaps = np.diff(np.concatenate(([0.0], scaled)))
            for index, gap in enumerate(gaps, start=1):
                encoded[f"log_gap_{side}_{index}"] = float(np.log(gap))
        return MappingProxyType(encoded)

    def cell_counts(self, study: Study) -> NDArray[np.float64]:
        """Return the ``2 x 2K`` table of stimulus by response-and-confidence counts."""

        signals = _study_binary(study, self.signal, "signal")
        codes = self.outcome_codes(study)
        table = np.zeros((2, 2 * self.n_confidence), dtype=np.float64)
        for stimulus in (0, 1):
            selected = codes[signals == stimulus]
            table[stimulus] = np.bincount(selected, minlength=2 * self.n_confidence)
        return table

    def type1_counts(self, study: Study) -> DetectionCounts:
        """Collapse confidence away and return the type-1 detection table."""

        return DetectionCounts.from_responses(
            _study_binary(study, self.signal, "signal"),
            _study_binary(study, self.response, "response"),
        )

    def type1_summary(self, study: Study) -> SignalDetectionSummary:
        """Return the type-1 anchor the meta fit holds fixed, with its corrected rates.

        This is where the structured :class:`CorrectedRates` record lives. :meth:`fit` used
        to carry a copy of it on a ``MetaSDTFitResult``, which duplicated a deterministic
        function of the study and this model's declared ``correction`` onto a result whose
        job is to carry what only fitting produced. The rates themselves still travel on
        the fit as :attr:`~behavio.contracts.FitResult.derived`, and whether the declared
        correction actually fired is recorded in the fit's diagnostic message.
        """

        return equal_variance_summary(self.type1_counts(study), correction=self.correction)

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        """Return the zero-based joint response-and-confidence cell for every trial."""

        responses = _study_binary(study, self.response, "response")
        confidence = self._confidence(study)
        index = np.searchsorted(np.asarray(self.confidence_levels, dtype=np.int64), confidence)
        codes = np.where(responses == 1.0, self.n_confidence + index, self.n_confidence - 1 - index)
        return protected_array(codes, dtype=np.int64)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        coordinate = _parameter_vector(parameters, self.parameter_names)
        signals = _study_binary(design, self.signal, "signal")
        table = self._cell_table(coordinate)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        draws = _categorical_draws(generator, table[signals.astype(np.int64), :])
        responses = (draws >= self.n_confidence).astype(np.int8)
        levels = np.asarray(self.confidence_levels, dtype=np.int64)
        confidence = np.where(
            draws >= self.n_confidence,
            levels[np.clip(draws - self.n_confidence, 0, self.n_confidence - 1)],
            levels[np.clip(self.n_confidence - 1 - draws, 0, self.n_confidence - 1)],
        )
        columns = {name: design[name] for name in design.columns}
        columns[self.response] = responses
        columns[self.confidence] = confidence.astype(np.int64)
        return Study(columns)

    def fit(self, study: Study) -> FitResult:
        type1 = self.type1_summary(study)
        if type1.is_degenerate or not np.isfinite(type1.d_prime):
            raise ModelDataError(
                "meta-d' needs a finite type-1 d'; declare a RateCorrection to repair an "
                "extreme hit or false-alarm rate"
            )
        if type1.d_prime <= 0:
            raise ModelDataError(
                f"meta-d' is defined relative to a positive type-1 d'; observed {type1.d_prime:.4f}"
            )
        counts = self.cell_counts(study)
        objective = self._meta_objective(counts, type1.d_prime, type1.criterion)
        starts, bounds = self._start_schedule(counts, type1.d_prime)
        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=lambda values: finite_difference_gradient(objective, values),
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
        result = results[int(np.argmin(objectives))]
        meta_coordinate = np.asarray(result.x, dtype=np.float64)
        gradient = finite_difference_gradient(objective, meta_coordinate)
        hessian = finite_difference_hessian(
            lambda values: finite_difference_gradient(objective, values),
            meta_coordinate,
            relative_step=1e-3,
        )
        meta_covariance = covariance_from_hessian(hessian)
        estimates = np.concatenate(([type1.d_prime, type1.criterion], meta_coordinate))
        variance_hit = _z_variance(type1.rates.hit_rate, type1.rates.n_signal)
        variance_false_alarm = _z_variance(type1.rates.false_alarm_rate, type1.rates.n_noise)
        covariance = np.zeros((len(estimates), len(estimates)), dtype=np.float64)
        covariance[0, 0] = variance_hit + variance_false_alarm
        covariance[1, 1] = 0.25 * (variance_hit + variance_false_alarm)
        covariance[0, 1] = covariance[1, 0] = 0.5 * (variance_false_alarm - variance_hit)
        covariance[2:, 2:] = meta_covariance
        meta_d_prime = float(meta_coordinate[0])
        joint = self._joint_objective(counts, type1.d_prime, type1.criterion, meta_coordinate)
        rates = type1.rates
        efficiency = (
            DerivedQuantity("hit_rate", rates.hit_rate, description="corrected type-1 hit rate"),
            DerivedQuantity(
                "false_alarm_rate",
                rates.false_alarm_rate,
                description="corrected type-1 false-alarm rate",
            ),
            DerivedQuantity(
                "relative_criterion",
                float(type1.relative_criterion),
                description="c divided by d', the position the meta fit holds fixed",
            ),
            DerivedQuantity(
                "m_ratio", float(meta_d_prime / type1.d_prime), description="meta-d' / d'"
            ),
            DerivedQuantity(
                "m_diff", float(meta_d_prime - type1.d_prime), description="meta-d' - d'"
            ),
        )
        return FitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            covariance=covariance,
            n_observations=len(study),
            diagnostics=FitDiagnostics(
                converged=bool(result.success),
                optimizer="closed-form type-1 z-transform, then L-BFGS-B multistart",
                status=int(result.status),
                message=(
                    f"{result.message} (type-1 table of {rates.n_signal} signal and "
                    f"{rates.n_noise} noise trials under correction="
                    f"{rates.correction.value}, applied={rates.correction_applied})"
                ),
                n_iterations=int(result.nit),
                objective=joint,
                gradient_norm=float(np.linalg.norm(gradient)),
                hessian_condition=float(np.linalg.cond(hessian)),
                boundary_estimate=bool(
                    any(
                        abs(value - low) < 1e-6 or abs(value - high) < 1e-6
                        for value, (low, high) in zip(meta_coordinate, bounds, strict=True)
                    )
                ),
            ),
            derived=(*natural_quantities(self, estimates, covariance), *efficiency),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> CategoricalPrediction:
        _prediction_mode(self.model_name, mode)
        _validate_fit(self, fit)
        signals = _study_binary(study, self.signal, "signal")
        table = self._cell_table(np.asarray(fit.estimates, dtype=np.float64))
        probability = table[signals.astype(np.int64), :]
        return CategoricalPrediction(
            probability=probability,
            linear_predictor=np.log(np.clip(probability, PROBABILITY_FLOOR, 1.0)),
            categories=self.categories,
            mode=PredictionMode.FILTERED,
            category_factors=self.category_factors,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        codes = self.outcome_codes(study)
        probability = self.predict(study, fit, mode=mode).probability
        selected = probability[np.arange(len(codes)), codes]
        return protected_array(np.log(np.clip(selected, PROBABILITY_FLOOR, 1.0)), dtype=np.float64)

    def _type2_criteria(
        self, meta_coordinate: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        width = self.n_confidence - 1
        no_gaps = np.exp(np.asarray(meta_coordinate[1 : 1 + width], dtype=np.float64))
        yes_gaps = np.exp(np.asarray(meta_coordinate[1 + width :], dtype=np.float64))
        return -np.cumsum(no_gaps), np.cumsum(yes_gaps)

    def _conditional_table(
        self, meta_coordinate: NDArray[np.float64], criterion_ratio: float
    ) -> NDArray[np.float64]:
        """Return ``P(cell | stimulus, response)`` for both stimuli under the meta model."""

        meta_d_prime = float(meta_coordinate[0])
        criteria_no, criteria_yes = self._type2_criteria(meta_coordinate)
        shift = meta_d_prime * criterion_ratio
        boundaries_no = np.concatenate(([-np.inf], criteria_no[::-1], [0.0]))
        boundaries_yes = np.concatenate(([0.0], criteria_yes, [np.inf]))
        table = np.zeros((2, 2 * self.n_confidence), dtype=np.float64)
        for stimulus in (0, 1):
            mean = (meta_d_prime / 2.0 if stimulus == 1 else -meta_d_prime / 2.0) - shift
            no_area = max(float(ndtr(-mean)), PROBABILITY_FLOOR)
            yes_area = max(1.0 - float(ndtr(-mean)), PROBABILITY_FLOOR)
            table[stimulus, : self.n_confidence] = np.diff(ndtr(boundaries_no - mean)) / no_area
            table[stimulus, self.n_confidence :] = np.diff(ndtr(boundaries_yes - mean)) / yes_area
        return np.clip(table, PROBABILITY_FLOOR, 1.0)

    def _cell_table(self, coordinate: NDArray[np.float64]) -> NDArray[np.float64]:
        d_prime = float(coordinate[0])
        criterion = float(coordinate[1])
        if not np.isfinite(d_prime) or d_prime <= 0:
            raise ValueError("meta-d' requires a positive type-1 d'")
        conditional = self._conditional_table(coordinate[2:], criterion / d_prime)
        table = np.empty_like(conditional)
        for stimulus in (0, 1):
            mean = d_prime / 2.0 if stimulus == 1 else -d_prime / 2.0
            yes = float(ndtr(mean - criterion))
            weights = np.concatenate(
                (
                    np.full(self.n_confidence, 1.0 - yes),
                    np.full(self.n_confidence, yes),
                )
            )
            row = conditional[stimulus] * weights
            table[stimulus] = row / row.sum()
        return table

    def _meta_objective(self, counts: NDArray[np.float64], d_prime: float, criterion: float):
        ratio = criterion / d_prime

        def objective(meta_coordinate: NDArray[np.float64]) -> float:
            conditional = self._conditional_table(
                np.asarray(meta_coordinate, dtype=np.float64), ratio
            )
            return -float(np.sum(counts * np.log(conditional)))

        return objective

    def _joint_objective(
        self,
        counts: NDArray[np.float64],
        d_prime: float,
        criterion: float,
        meta_coordinate: NDArray[np.float64],
    ) -> float:
        coordinate = np.concatenate(([d_prime, criterion], meta_coordinate))
        return -float(np.sum(counts * np.log(self._cell_table(coordinate))))

    def _start_schedule(
        self, counts: NDArray[np.float64], d_prime: float
    ) -> tuple[list[NDArray[np.float64]], tuple[tuple[float, float], ...]]:
        pooled = counts.sum(axis=0)
        no_counts = pooled[: self.n_confidence][::-1]
        yes_counts = pooled[self.n_confidence :]
        base = np.concatenate(
            (np.log(self._empirical_side(no_counts)), np.log(self._empirical_side(yes_counts)))
        )
        schedule = ((1.0, 1.0), (0.6, 1.3), (1.5, 0.7), (1.0, 1.6))
        starts: list[NDArray[np.float64]] = []
        for index in range(self.n_restarts):
            meta_scale, gap_scale = schedule[index % len(schedule)]
            meta_start = float(np.clip(d_prime * meta_scale, 1e-2, self.maximum_meta_d_prime))
            starts.append(np.concatenate(([meta_start], base + float(np.log(gap_scale)))))
        bounds = [(1e-3, float(self.maximum_meta_d_prime))]
        bounds.extend((-12.0, 6.0) for _ in range(2 * (self.n_confidence - 1)))
        return starts, tuple(bounds)

    def _empirical_side(self, counts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return positive type-2 gaps implied by one observed confidence distribution.

        Under a standard normal with the type-1 criterion at zero, a response region holds
        half the mass, so a cumulative confidence proportion ``q`` places the corresponding
        type-2 criterion at ``|z(0.5(1 + q))|`` away from the criterion. The magnitudes are
        differenced into the positive gaps the optimizer works in.
        """

        total = float(counts.sum())
        if total <= 0:
            return np.full(self.n_confidence - 1, 0.5, dtype=np.float64)
        cumulative = np.cumsum(counts)[:-1] / total
        quantiles = np.clip(cumulative, 0.05, 0.95)
        magnitudes = np.maximum.accumulate(np.abs(ndtri(0.5 * (1.0 + quantiles))))
        return np.maximum(np.diff(np.concatenate(([0.0], magnitudes))), 1e-2)

    def _confidence(self, study: Study) -> NDArray[np.int64]:
        if self.confidence not in study.columns:
            raise ModelDataError(f"study is missing confidence column {self.confidence!r}")
        try:
            values = np.asarray(study[self.confidence], dtype=np.int64)
        except (TypeError, ValueError):
            raise ModelDataError("confidence must contain integer levels") from None
        unknown = sorted(set(values.tolist()) - set(self.confidence_levels))
        if unknown:
            raise ModelDataError(f"observed confidence outside the declared levels: {unknown}")
        return values


def _natural_input(
    estimates: Sequence[float] | NDArray[np.floating[Any]], width: int
) -> NDArray[np.float64]:
    """Validate one estimated coordinate before mapping it onto the natural one."""

    try:
        vector = np.asarray(estimates, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("estimates must contain finite numeric values") from None
    if vector.shape != (width,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"estimates must contain {width} finite values")
    return vector


def _one_over_2n(rate: float, denominator: int) -> float:
    if rate == 0.0:
        return 1.0 / (2.0 * denominator)
    if rate == 1.0:
        return 1.0 - 1.0 / (2.0 * denominator)
    return rate


def _a_prime(hit_rate: float, false_alarm_rate: float) -> float:
    if hit_rate == false_alarm_rate:
        return 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        if hit_rate > false_alarm_rate:
            difference = hit_rate - false_alarm_rate
            denominator = 4.0 * hit_rate * (1.0 - false_alarm_rate)
            value = 0.5 + difference * (1.0 + difference) / denominator
        else:
            difference = false_alarm_rate - hit_rate
            denominator = 4.0 * false_alarm_rate * (1.0 - hit_rate)
            value = 0.5 - difference * (1.0 + difference) / denominator
    return float(value)


def _b_double_prime_d(hit_rate: float, false_alarm_rate: float) -> float:
    hit_spread = hit_rate * (1.0 - hit_rate)
    false_alarm_spread = false_alarm_rate * (1.0 - false_alarm_rate)
    total = hit_spread + false_alarm_spread
    if total == 0:
        return float("nan")
    difference = hit_spread - false_alarm_spread
    return float(difference / total if hit_rate >= false_alarm_rate else -difference / total)


def _z_variance(rate: float, denominator: int) -> float:
    density = float(np.exp(-0.5 * ndtri(rate) ** 2) / np.sqrt(2.0 * np.pi))
    if density <= 0:
        return float("inf")
    return float(rate * (1.0 - rate) / (denominator * density**2))


def _forced_choice_integral(d_prime: float, n_alternatives: int) -> float:
    def integrand(value: float) -> float:
        density = float(np.exp(-0.5 * (value - d_prime) ** 2) / np.sqrt(2.0 * np.pi))
        return density * float(ndtr(value)) ** (n_alternatives - 1)

    lower = min(-9.0, d_prime - 9.0)
    upper = max(9.0, d_prime + 9.0)
    value, _ = quad(integrand, lower, upper, limit=200)
    return float(np.clip(value, 0.0, 1.0))


def _validate_alternatives(n_alternatives: int) -> int:
    if isinstance(n_alternatives, bool) or not isinstance(n_alternatives, (int, np.integer)):
        raise ValueError("n_alternatives must be an integer")
    alternatives = int(n_alternatives)
    if alternatives < 2:
        raise ValueError("forced choice needs at least two alternatives")
    return alternatives


def _bernoulli_log_prob(
    outcomes: NDArray[np.float64], probability: NDArray[np.float64]
) -> NDArray[np.float64]:
    bounded = np.clip(probability, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    return outcomes * np.log(bounded) + (1.0 - outcomes) * np.log1p(-bounded)


def _categorical_draws(
    generator: np.random.Generator, probability: NDArray[np.float64]
) -> NDArray[np.int64]:
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.0
    draws = generator.random(probability.shape[0])[:, None]
    return np.argmax(draws < cumulative, axis=1).astype(np.int64)


def _binary_vector(values: Sequence[Any] | NDArray[Any], label: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain only zero and one") from None
    if array.ndim != 1 or not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{label} must contain only zero and one")
    return array


def _study_binary(study: Study, column: str, label: str) -> NDArray[np.float64]:
    if column not in study.columns:
        raise ModelDataError(f"study is missing {label} column {column!r}")
    try:
        return _binary_vector(study[column], label)
    except ValueError as error:
        raise ModelDataError(str(error)) from None


def _validate_column(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty column name")
    if value in REQUIRED_COLUMNS:
        raise ValueError(f"{label} cannot replace a required Study column")


def _parameter_vector(
    parameters: Mapping[str, float], names: tuple[str, ...]
) -> NDArray[np.float64]:
    if set(parameters) != set(names):
        raise ValueError("parameters must match the model exactly")
    try:
        values = np.asarray([parameters[name] for name in names], dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("parameters must contain finite numeric values") from None
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain finite numeric values")
    return values


def _validate_fit(model: Any, fit: FitResult) -> None:
    if (
        fit.model_name != model.model_name
        or fit.model_signature != model.signature
        or fit.parameter_names != model.parameter_names
    ):
        raise ValueError("fit result was produced by a different model specification")


def _prediction_mode(model_name: str, mode: PredictionMode) -> PredictionMode:
    prediction_mode = PredictionMode(mode)
    if prediction_mode is not PredictionMode.FILTERED:
        raise UnsupportedPredictionMode(
            f"{model_name} supports only filtered prediction, not {prediction_mode.value!r}"
        )
    return prediction_mode
