"""Scalar timing: duration reproduction and temporal bisection.

A note on the name, first
------------------------
This module is **not** ``behavio.observed.interval_policy``, which curates externally
discovered annotation bouts and has nothing to do with timing behaviour. "Interval" already
means two things in this package -- a confidence interval on
:class:`~behavio.contracts.estimator.DerivedQuantity`, and an annotated bout in
:mod:`behavio.observed` -- so nothing here is called one. The paradigm the literature calls
*interval reproduction* is :class:`DurationReproduction`, the theory is *scalar timing*, and
the thing an animal times is a **duration**.

The theory, and what makes a model one
--------------------------------------
Gibbon's (1977) scalar expectancy theory says that a timed duration is represented with
noise **proportional to the duration itself**. That single statement -- the *scalar property*
-- is the content of the theory: a clock whose error grows as a fixed number of milliseconds
is not a scalar clock, and the whole empirical case for the account is that timing
variability scales while absolute variability does not. So the memory both families here
share is

.. math::

    \\log \\hat{T} \\sim \\mathcal{N}\\!\\left(\\log(\\kappa\\,T^{\\beta}),\\ \\sigma^2\\right),
    \\qquad \\sigma^2 = \\log(1 + w^2),

for a clock rate :math:`\\kappa > 0`, a Weber fraction :math:`w > 0` and an optional
central-tendency exponent :math:`\\beta > 0`. The parameterisation is chosen so that the
**coefficient of variation of** :math:`\\hat{T}` **is exactly** :math:`w` **at every**
:math:`T`, which is the scalar property stated as an identity rather than as an approximation
that happens to hold for small :math:`w`.

*Why lognormal rather than Gibbon's Gaussian.* Gibbon writes the representation as a normal
variable with standard deviation proportional to its mean, which has the scalar property
exactly and also assigns positive probability to a negative duration. At :math:`w = 0.15`
that mass is :math:`10^{-11}` and nobody notices; at :math:`w = 0.5` it is 2.3 %, and an
optimizer exploring a box that reaches :math:`w = 3` will spend time in a region where the
model is not a model of durations at all. A lognormal is positive everywhere, has constant
coefficient of variation everywhere, and is linear in :math:`\\log T` -- which is what makes
the *same* memory produce a reproduction density and a bisection psychometric function
without a second set of assumptions. What it changes empirically is the shape of the
reproduction distribution's right tail, which is the one place the two forms disagree
measurably, and a study that can resolve that tail should say which it is fitting.

*What* :math:`\\beta` *is, and why it is fixed by default.* :math:`\\beta = 1` is pure scalar
timing: the reproduced duration is proportional to the target. :math:`\\beta < 1` is
Vierordt's law -- short durations over-reproduced, long ones under-reproduced -- which is
robust in humans and is a regression towards the centre of the tested range rather than a
property of the clock. It is available (``fixed_central_tendency=None`` estimates it) and it
is *fixed at one by default*, because a model whose mean is free is no longer making the
prediction the scalar property is a prediction about. It is also **not estimable from
bisection**: see :class:`TemporalBisection`.

The two paradigms and what each identifies
------------------------------------------
:class:`DurationReproduction` observes a **continuous duration** and identifies
:math:`\\kappa` from the location of the reproductions and :math:`w` from their spread.

:class:`TemporalBisection` observes a **binary report** -- was that probe more like the short
anchor or the long one? -- and identifies :math:`\\kappa` from where the psychometric
function crosses one half and :math:`w` from how steeply it does so.

The two are the same clock and the same coordinate. A study that runs both can ask whether
one :math:`w` describes them, which is scalar timing's strongest testable claim and one
neither paradigm can make alone.

The bisection decision rule is a declared modelling choice with an argument behind it;
``docs/decisions/0060-bisect-time-by-the-ratio-rule.md`` is that argument, including what the
alternatives change and why one anchor pair cannot settle it.

References
----------
Gibbon, J. (1977). Scalar expectancy theory and Weber's law in animal timing.
*Psychological Review, 84*(3), 279-325.

Church, R. M., & Deluty, M. Z. (1977). Bisection of temporal intervals. *Journal of
Experimental Psychology: Animal Behavior Processes, 3*(3), 216-228.

Gibbon, J. (1981). On the form and location of the psychometric bisection function for time.
*Journal of Mathematical Psychology, 24*(1), 58-87.

Wearden, J. H. (1991). Human performance on an analogue of an interval bisection task.
*Quarterly Journal of Experimental Psychology, 43B*(1), 59-81.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, ndtr

from behavio._internal.arrays import protected_array
from behavio.contracts.estimator import (
    DensityPrediction,
    DerivedQuantity,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
)
from behavio.models._kernels.hazard import validated_durations
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
    "BisectionParameters",
    "BisectionRule",
    "DurationReproduction",
    "LogCoordinateFitResult",
    "ReproductionParameters",
    "TemporalBisection",
    "bisection_threshold",
    "memory_log_sd",
    "weber_fraction_from_log_sd",
]

_LOG_TWO_PI = float(np.log(2.0 * np.pi))

#: How far into the tails a tabulated reproduction density is carried, in memory standard
#: deviations. Six is far enough that the truncated mass is below 1e-9 and near enough that
#: a shared grid over a wide range of targets still resolves each row's peak.
_DENSITY_SPAN = 6.0

_CLOCK_RATE_BOX = (float(np.log(0.05)), float(np.log(20.0)))
_WEBER_BOX = (float(np.log(1e-3)), float(np.log(3.0)))
_CENTRAL_TENDENCY_BOX = (float(np.log(0.1)), float(np.log(3.0)))


# --------------------------------------------------------------------------------------
# The closed forms both families are validated against
# --------------------------------------------------------------------------------------


def memory_log_sd(weber_fraction: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the memory's log standard deviation :math:`\\sqrt{\\log(1 + w^2)}`.

    This is the whole of the scalar property in one line. A lognormal with log standard
    deviation :math:`\\sigma` has coefficient of variation :math:`\\sqrt{e^{\\sigma^2} - 1}`
    **whatever its median is**, so inverting that relation is what makes ``weber_fraction``
    a number with the same meaning at every duration -- and what makes
    :func:`weber_fraction_from_log_sd` its exact inverse rather than a small-noise
    approximation to one.
    """

    values = np.asarray(weber_fraction, dtype=np.float64)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("weber_fraction must be finite and positive")
    return np.asarray(np.sqrt(np.log1p(values**2)), dtype=np.float64)


def weber_fraction_from_log_sd(log_sd: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the coefficient of variation a lognormal memory of this width produces."""

    values = np.asarray(log_sd, dtype=np.float64)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("log_sd must be finite and positive")
    return np.asarray(np.sqrt(np.expm1(values**2)), dtype=np.float64)


class BisectionRule(StrEnum):
    """Which of two remembered anchors a probe is judged closer to."""

    RATIO = "ratio"
    """Respond *long* when :math:`\\hat{T}/S > L/\\hat{T}`: the geometric mean."""

    DIFFERENCE = "difference"
    """Respond *long* when :math:`\\hat{T} - S > L - \\hat{T}`: the arithmetic mean."""


def bisection_threshold(
    short_anchor: float, long_anchor: float, *, rule: BisectionRule | str = BisectionRule.RATIO
) -> float:
    """Return the remembered duration a probe is compared against, in the anchors' own units.

    The **known answer** the bisection tests are held to, and the one number the two rules
    disagree about. The ratio rule compares a probe to :math:`\\sqrt{SL}` and the difference
    rule to :math:`(S + L)/2`; for :math:`S = 2` and :math:`L = 8` those are 4 and 5, and no
    amount of fitting turns one into the other.
    """

    require_positive(float(short_anchor), "short_anchor")
    require_positive(float(long_anchor), "long_anchor")
    if float(short_anchor) >= float(long_anchor):
        raise ValueError("short_anchor must be strictly below long_anchor")
    if BisectionRule(rule) is BisectionRule.RATIO:
        return float(np.sqrt(float(short_anchor) * float(long_anchor)))
    return float(0.5 * (float(short_anchor) + float(long_anchor)))


# --------------------------------------------------------------------------------------
# Natural-scale parameter records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReproductionParameters:
    """Natural-scale duration-reproduction parameters in the units a report uses."""

    clock_rate: float
    weber_fraction: float
    central_tendency: float = 1.0

    def __post_init__(self) -> None:
        require_positive(self.clock_rate, "clock_rate")
        require_positive(self.weber_fraction, "weber_fraction")
        require_positive(self.central_tendency, "central_tendency")


@dataclass(frozen=True, slots=True)
class BisectionParameters:
    """Natural-scale temporal-bisection parameters in the units a report uses."""

    clock_rate: float
    weber_fraction: float

    def __post_init__(self) -> None:
        require_positive(self.clock_rate, "clock_rate")
        require_positive(self.weber_fraction, "weber_fraction")


# --------------------------------------------------------------------------------------
# Duration reproduction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ReproductionProblem:
    """One study's log targets and log reproductions, validated once."""

    log_targets: NDArray[np.float64]
    log_reproductions: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        return len(self.log_targets)


@dataclass(frozen=True, slots=True)
class DurationReproduction(LogCoordinateModel):
    """Reproduce a target duration; the observation is a duration, not a choice.

    Each row presents a target :math:`T` and records a reproduction :math:`R`, and the model
    is the shared scalar memory read as a density on :math:`R`:

    .. math::

        \\log R \\sim \\mathcal{N}\\!\\left(\\log(\\kappa T^{\\beta}),\\ \\log(1 + w^2)\\right).

    ``clock_rate`` :math:`\\kappa` scales the **median** reproduction, so
    :math:`\\kappa = 1` is an animal whose typical reproduction is the target. Reporting the
    median rather than the mean is what keeps :math:`\\kappa` orthogonal to :math:`w`: the
    mean of a lognormal moves with its width, so a "mean reproduction" parameterisation would
    make a noisier animal look like a slower clock.

    ``weber_fraction`` :math:`w` is the coefficient of variation of the reproduction, exactly
    and at every target. ``fixed_central_tendency`` is one by default; see the module
    docstring for what setting it to ``None`` buys and costs.

    What comes back from ``predict``
    --------------------------------
    A :class:`~behavio.contracts.DensityPrediction` over the reproduction column, tabulated
    on one shared geometric grid -- a density, because the observation is continuous and
    :class:`~behavio.contracts.Prediction` and
    :class:`~behavio.contracts.CategoricalPrediction` both describe a discrete outcome.
    ``pointwise_log_prob`` returns the **analytic** log density rather than reading the grid,
    so a fold's score never depends on ``grid_points``; the grid is a tabulation of the same
    closed form, which is what makes the two agree to interpolation error rather than by
    construction.

    Identifiability, before the fit
    -------------------------------
    ``narrow_target_range``
        The tested durations span less than half an octave, so a scalar clock and a
        constant-variance clock predict nearly the same spreads and the *content* of the
        theory is not being tested. This is the finding that matters most: the scalar
        property is a statement about how variability changes with duration, and a
        single-duration design cannot see it however many trials it runs.
    ``unidentified_central_tendency`` / ``weakly_identified_central_tendency``
        With one target duration :math:`\\kappa T^{\\beta}` is one number and the exponent is
        the clock rate under another name; with two, it is identified only by how the two
        medians differ.
    """

    target_column: str = "target_duration"
    outcome: str = "reproduced_duration"
    fixed_central_tendency: float | None = 1.0
    grid_points: int = 257
    n_restarts: int = 4
    max_iterations: int = 1_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        columns = validated_columns((self.target_column, self.outcome), "declared columns")
        object.__setattr__(self, "target_column", columns[0])
        object.__setattr__(self, "outcome", columns[1])
        if self.fixed_central_tendency is not None:
            require_positive(float(self.fixed_central_tendency), "fixed_central_tendency")
            object.__setattr__(self, "fixed_central_tendency", float(self.fixed_central_tendency))
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
        return "duration-reproduction-scalar"

    @property
    def signature(self) -> str:
        exponent = (
            "estimated"
            if self.fixed_central_tendency is None
            else f"{self.fixed_central_tendency:g}"
        )
        return (
            f"{self.model_name}[outcome={self.outcome};target={self.target_column};"
            f"memory=lognormal;central_tendency={exponent};grid={self.grid_points}]"
        )

    @property
    def natural_names(self) -> tuple[str, ...]:
        if self.fixed_central_tendency is None:
            return ("clock_rate", "weber_fraction", "central_tendency")
        return ("clock_rate", "weber_fraction")

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.target_column,)

    @property
    def density_outcome(self) -> str:
        """The continuous column :meth:`predict_density` tabulates a density over."""

        return self.outcome

    @property
    def penalised_linear_refusal(self) -> str:
        """Why the penalised-linear route cannot be applied to this family.

        Declared rather than left to structural typing, following the precedent
        :class:`~behavio.models.rl.BinaryRLAgent` set. The obstacle is not a recursion --
        these rows are independent -- but the absence of a linear predictor: the Weber
        fraction enters the likelihood as the *scale* of a density and no design matrix has a
        column for a scale. All three combinators reach this family through
        :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator` instead.
        """

        return (
            "a scalar-timing model puts one of its parameters in the scale of a density "
            "rather than in a linear predictor, so it has no design matrix to widen; its "
            "rows are independent, so it composes through "
            "behavio.contracts.bounded.BoundedCoordinateEstimator instead"
        )

    # -- natural-coordinate conveniences ---------------------------------------------------

    def parameters_from_components(
        self,
        *,
        clock_rate: float,
        weber_fraction: float,
        central_tendency: float | None = None,
    ) -> Mapping[str, float]:
        """Validate and encode natural-scale parameters onto the estimated coordinate."""

        declared = self.fixed_central_tendency
        if declared is None:
            if central_tendency is None:
                raise ValueError("central_tendency is estimated by this model and must be supplied")
            exponent = float(central_tendency)
        else:
            if central_tendency is not None and not np.isclose(
                float(central_tendency), float(declared)
            ):
                raise ValueError(f"central_tendency is fixed at {declared} by this model")
            exponent = float(declared)
        components = ReproductionParameters(clock_rate, weber_fraction, exponent)
        natural = {
            "clock_rate": components.clock_rate,
            "weber_fraction": components.weber_fraction,
        }
        if declared is None:
            natural["central_tendency"] = components.central_tendency
        return self.from_natural(natural)

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> ReproductionParameters:
        """Decode an estimated coordinate into the complete natural parameter set."""

        values = self.to_natural(self.coordinate(parameters))
        exponent = (
            float(values["central_tendency"])
            if self.fixed_central_tendency is None
            else float(self.fixed_central_tendency)
        )
        return ReproductionParameters(
            clock_rate=float(values["clock_rate"]),
            weber_fraction=float(values["weber_fraction"]),
            central_tendency=exponent,
        )

    def median_reproduction(
        self, study: Study, parameters: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """Return :math:`\\kappa T^{\\beta}`, the median reproduction of every row."""

        problem = self.read_problem(study, require_outcome=False)
        rows = self.shared_rows(problem.n_rows, self.coordinate(parameters))
        location, _, _ = self._memory(problem, rows)
        return np.asarray(np.exp(location), dtype=np.float64)

    def coefficient_of_variation(self, parameters: Mapping[str, float] | FitResult) -> float:
        """Return the reproduction's coefficient of variation, which is one number.

        Not a function of the target, and that is the point: this method has no ``study``
        argument because under the scalar property there is nothing for it to depend on.
        """

        return float(self.parameter_components(parameters).weber_fraction)

    # -- reading a study -------------------------------------------------------------------

    def read_problem(self, study: Study, *, require_outcome: bool = True) -> _ReproductionProblem:
        """Read and validate this study's targets and reproductions."""

        targets = validated_durations(
            numeric_column(study, self.target_column, role="task"),
            label=f"target column {self.target_column!r}",
            strictly_positive=True,
        )
        if not require_outcome and self.outcome not in study.columns:
            return _ReproductionProblem(
                log_targets=np.log(targets),
                log_reproductions=np.zeros_like(targets),
            )
        reproductions = validated_durations(
            numeric_column(study, self.outcome, role="outcome"),
            label=f"outcome column {self.outcome!r}",
            strictly_positive=True,
        )
        return _ReproductionProblem(
            log_targets=np.log(targets), log_reproductions=np.log(reproductions)
        )

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the observed reproduction of each row, in the outcome column's own units."""

        return validated_durations(
            numeric_column(study, self.outcome, role="outcome"),
            label=f"outcome column {self.outcome!r}",
            strictly_positive=True,
        )

    # -- the likelihood --------------------------------------------------------------------

    def _memory(
        self, problem: _ReproductionProblem, rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return each row's memory location, log standard deviation, and exponent."""

        exponent = (
            np.exp(rows[:, 2])
            if self.fixed_central_tendency is None
            else np.full(len(rows), float(self.fixed_central_tendency))
        )
        location = rows[:, 0] + exponent * problem.log_targets
        squared = np.log1p(np.exp(2.0 * rows[:, 1]))
        return location, np.sqrt(squared), exponent

    @staticmethod
    def _log_density(
        problem: _ReproductionProblem,
        location: NDArray[np.float64],
        log_sd: NDArray[np.float64],
        deviate: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the lognormal log density of each observed reproduction.

        The ``-log R`` term is the Jacobian of the change of variable from ``log R`` to ``R``
        and is constant in the parameters, so it does not move the fit. It is kept because a
        pointwise score is compared *across models* -- a density that dropped its Jacobian
        would beat one that did not for a reason that is not about behaviour.
        """

        del location
        return -(problem.log_reproductions + np.log(log_sd) + 0.5 * _LOG_TWO_PI + 0.5 * deviate**2)

    def _row_scores(
        self, problem: _ReproductionProblem, rows: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        location, log_sd, _ = self._memory(problem, rows)
        deviate = (problem.log_reproductions - location) / log_sd
        return self._log_density(problem, location, log_sd, deviate)

    def row_value_and_gradient(
        self, problem: _ReproductionProblem, rows: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its analytic gradient, one row at a time.

        Every derivative is taken on the estimated coordinate directly, which is where they
        simplify. With :math:`b = \\log w` the width's derivative is
        :math:`\\partial\\sigma/\\partial b = w^2 / ((1 + w^2)\\sigma)`, because
        :math:`\\sigma^2 = \\log(1 + w^2)`; with :math:`c = \\log\\beta` the exponent enters
        the location as :math:`\\beta \\log T` and so does its derivative.
        """

        location, log_sd, exponent = self._memory(problem, rows)
        deviate = (problem.log_reproductions - location) / log_sd
        loss = -float(np.sum(self._log_density(problem, location, log_sd, deviate)))
        gradient = np.zeros_like(rows)
        location_term = -deviate / log_sd
        gradient[:, 0] = location_term
        squared_weber = np.exp(2.0 * rows[:, 1])
        gradient[:, 1] = (1.0 - deviate**2) * squared_weber / ((1.0 + squared_weber) * log_sd**2)
        if self.fixed_central_tendency is None:
            gradient[:, 2] = location_term * exponent * problem.log_targets
        return loss, gradient

    # -- the bounded-coordinate per-row members ---------------------------------------------

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> DensityPrediction:
        """Return the tabulated reproduction density under one coordinate per row."""

        prediction_mode = self.prediction_mode(mode)
        problem = self.read_problem(study, require_outcome=False)
        rows = self.row_coordinates(study, coefficients)
        location, log_sd, _ = self._memory(problem, rows)
        lower = float(np.min(location - _DENSITY_SPAN * log_sd))
        upper = float(np.max(location + _DENSITY_SPAN * log_sd))
        grid = np.exp(np.linspace(lower, upper, self.grid_points))
        log_grid = np.log(grid)[None, :]
        deviate = (log_grid - location[:, None]) / log_sd[:, None]
        density = np.exp(-0.5 * deviate**2 - 0.5 * _LOG_TWO_PI) / (grid[None, :] * log_sd[:, None])
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
        """Return the predictive density, which is what ``predict`` already returns.

        The two calls are the same object rather than two tabulations, which is the property
        :func:`behavio.adapters.estimator_conformance.check_behaviour_estimator` looks for:
        there are no two halves of the prediction to disagree with one another.
        """

        prediction_mode = self.prediction_mode(mode)
        self.validate_fit(fit)
        rows = self.shared_rows(len(study), np.asarray(fit.estimates, dtype=np.float64))
        return self.predict_rows(study, rows, mode=prediction_mode)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observed reproduction analytically under one coordinate per row."""

        problem = self.read_problem(study)
        rows = self.row_coordinates(study, coefficients)
        return protected_array(self._row_scores(problem, rows), dtype=np.float64)

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Draw a reproduction on every row from the scalar memory."""

        problem = self.read_problem(design, require_outcome=False)
        rows = self.row_coordinates(design, coefficients)
        location, log_sd, _ = self._memory(problem, rows)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = np.exp(generator.normal(location, log_sd))
        return Study(columns)

    # -- the box and the restarts ------------------------------------------------------------

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:
        """Return the declared finite box the logarithms are searched in.

        Declared rather than derived, unlike the discounting family's delay-derived box,
        because these bounds are statements about *timing* rather than about a design: a
        clock twenty times fast is not a clock, and a Weber fraction of three is an animal
        whose reproductions carry no information about the target.
        """

        del study
        boxes = [_CLOCK_RATE_BOX, _WEBER_BOX]
        if self.fixed_central_tendency is None:
            boxes.append(_CENTRAL_TENDENCY_BOX)
        return np.asarray(boxes, dtype=np.float64)

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Start from the study's own log-log regression, and from three perturbations of it.

        The location, the exponent and the residual spread of ``log R`` on ``log T`` are the
        three parameters, so ordinary least squares gives all of them in closed form; the
        restarts perturb the width, which is the coordinate whose likelihood is least convex.
        """

        problem = self.read_problem(study)
        box = self.coordinate_box(study)
        slope, intercept, spread = _log_log_fit(problem.log_targets, problem.log_reproductions)
        if self.fixed_central_tendency is not None:
            slope = float(self.fixed_central_tendency)
            residual = problem.log_reproductions - slope * problem.log_targets
            intercept = float(np.mean(residual))
            spread = float(np.std(residual))
        width = float(np.log(max(np.sqrt(np.expm1(max(spread, 1e-3) ** 2)), 1e-3)))
        starts: list[NDArray[np.float64]] = []
        for offset in (0.0, float(np.log(2.0)), float(np.log(0.5)), float(np.log(4.0))):
            vector = [intercept, width + offset]
            if self.fixed_central_tendency is None:
                vector.append(float(np.log(max(slope, 1e-3))))
            starts.append(np.clip(np.asarray(vector, dtype=np.float64), box[:, 0], box[:, 1]))
        return tuple(starts[index % len(starts)] for index in range(self.n_restarts))

    # -- findings ----------------------------------------------------------------------------

    def design_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the designs in which this model's parameters cannot be separated."""

        targets = numeric_column(study, self.target_column, role="task")
        positive = targets[targets > 0.0]
        if not positive.size:
            return ()
        distinct = len(np.unique(np.round(positive, 12)))
        findings: list[ModelFinding] = []
        span = float(np.max(positive)) / float(np.min(positive))
        if span < 1.5:
            findings.append(
                ModelFinding(
                    code="narrow_target_range",
                    severity=WARNING,
                    message=(
                        f"the tested durations span a factor of {span:g}, so a scalar clock "
                        "and a clock with constant absolute variability predict nearly the "
                        "same spreads here; the scalar property is a claim about how "
                        "variability changes with duration and this design cannot see it"
                    ),
                )
            )
        if self.fixed_central_tendency is None and distinct == 1:
            findings.append(
                ModelFinding(
                    code="unidentified_central_tendency",
                    severity=WARNING,
                    message=(
                        "central_tendency is not identified by this design: every trial uses "
                        "the same target duration, so the exponent and the clock rate enter "
                        "the median as one number and the clock rate absorbs it exactly"
                    ),
                )
            )
        elif self.fixed_central_tendency is None and distinct == 2:
            findings.append(
                ModelFinding(
                    code="weakly_identified_central_tendency",
                    severity=WARNING,
                    message=(
                        "central_tendency is identified only by how the medians differ "
                        "between two target durations, so it will trade off strongly against "
                        "the clock rate"
                    ),
                )
            )
        return tuple(findings)


# --------------------------------------------------------------------------------------
# Temporal bisection
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BisectionProblem:
    """One study's log probes and long/short reports, validated once."""

    log_probes: NDArray[np.float64]
    reports: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        return len(self.log_probes)


@dataclass(frozen=True, slots=True)
class TemporalBisection(LogCoordinateModel):
    """Report whether a probe duration was more like the short or the long anchor.

    Two anchors :math:`S < L` are trained and **declared**, never learned: they are a fact
    about the training procedure, so they appear in the model's signature and a fit can never
    be read without them, exactly as ``value_scale`` works for
    :class:`~behavio.models.economic.TemporalDiscounting`. Column ``outcome`` is one when the
    animal responded *long*.

    The probe's memory is the module's shared scalar memory, so with the ratio rule

    .. math::

        P(\\text{long} \\mid t)
        = \\Phi\\!\\left(\\frac{\\log(\\kappa t) - \\tfrac{1}{2}\\log(SL)}{\\sigma}\\right),
        \\qquad \\sigma = \\sqrt{\\log(1 + w^2)},

    which crosses one half at :math:`t = \\sqrt{SL}/\\kappa`. **An accurate clock bisects at
    the geometric mean of the anchors.** That is Church and Deluty's (1977) result and it is
    the known answer this family is validated against.

    Which rule, and why the ratio one
    ---------------------------------
    ``rule=BisectionRule.RATIO`` is the default, and the argument has two halves.

    *Empirical.* Church and Deluty (1977) trained rats on **four** anchor pairs -- 1 v 4, 2 v
    8, 3 v 12 and 4 v 16 seconds -- and found the bisection point at the geometric mean in
    every one. The arithmetic rule predicts 2.5, 5, 7.5 and 10 s where the geometric rule
    predicts 2, 4, 6 and 8, and it is *varying the pair* that makes the two accounts
    separable at all; see the identifiability note below for why a single pair does not.

    *Theoretical.* The clock's noise is multiplicative, so the only comparison that leaves
    every source of noise in the model multiplicative is a comparison of ratios. Here the
    anchors are treated as remembered exactly, and under that idealisation the two rules
    happen to differ in one number and nothing else -- both give a probit in :math:`\\log t`
    with the same slope, because the probit comes from the memory rather than from the rule.
    Admit noisy anchor memories, which is what any complete account has to do, and the
    idealisation breaks in the ratio rule's favour: the ratio comparison's anchor noise is
    additive on the log scale and therefore scalar, while the difference comparison's is
    additive on the linear scale and therefore is not, so the difference rule stops obeying
    Weber's law at exactly the point where it stops being a relabelling.

    ``rule=BisectionRule.DIFFERENCE`` is offered because the disagreement is real -- human
    bisection points often fall between the two means, and sometimes at the arithmetic one.
    **What it changes is exactly the comparison duration** :math:`(S+L)/2`. The likelihood
    surface is otherwise identical, so fitting the same study under both rules returns the
    same ``weber_fraction``, the same log likelihood, and a ``clock_rate`` differing by
    exactly :math:`2\\sqrt{SL}/(S+L)`. That is asserted in
    :mod:`tests.test_scalar_timing`, and it is the reason the rule is in the signature: a
    reported bisection point is meaningless without it.

    A third rule is sometimes named separately. Wearden's (1991) *similarity* rule --
    similarity of two durations being the ratio of the smaller to the larger -- **is** the
    ratio rule, and is reached by ``RATIO`` rather than by a third option.

    What bisection cannot identify
    ------------------------------
    **The rule itself, from one anchor pair.** :math:`\\kappa` multiplies the comparison
    duration, so ratio-with-:math:`\\kappa` and difference-with-:math:`\\kappa'` fit any
    single-pair study identically for :math:`\\kappa'/\\kappa = 2\\sqrt{SL}/(S+L)`. Since a
    model instance carries one anchor pair, **no single fit of this class can test the rule**.
    Testing it means fitting several pairs with the clock rate held common across them, which
    is Church and Deluty's design and is a comparison between models rather than a parameter
    inside one.

    A **central-tendency exponent**. Under this memory the decision variable is
    :math:`\\beta \\log t` scaled by :math:`1/\\sigma`, so :math:`\\beta` and the Weber
    fraction enter the slope as one number and are exactly confounded. That is why
    :class:`DurationReproduction` estimates an exponent and this family does not, and it is a
    reason to run both paradigms rather than a limitation of the implementation.

    A **response bias** separate from the clock. A subject who says *long* more often and a
    subject whose clock runs fast move the bisection point the same way, so ``clock_rate``
    here is the two of them together. Reproduction separates them; bisection alone cannot.
    """

    short_anchor: float = 2.0
    long_anchor: float = 8.0
    probe_column: str = "probe_duration"
    outcome: str = "choice"
    rule: BisectionRule = BisectionRule.RATIO
    n_restarts: int = 4
    max_iterations: int = 1_000
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        columns = validated_columns((self.probe_column, self.outcome), "declared columns")
        object.__setattr__(self, "probe_column", columns[0])
        object.__setattr__(self, "outcome", columns[1])
        object.__setattr__(self, "rule", BisectionRule(self.rule))
        require_positive(self.short_anchor, "short_anchor")
        require_positive(self.long_anchor, "long_anchor")
        if float(self.short_anchor) >= float(self.long_anchor):
            raise ValueError("short_anchor must be strictly below long_anchor")
        positive_integer(self.n_restarts, "n_restarts")
        positive_integer(self.max_iterations, "max_iterations")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

    # -- identity -------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"temporal-bisection-{self.rule.value}"

    @property
    def signature(self) -> str:
        return (
            f"{self.model_name}[outcome={self.outcome};probe={self.probe_column};"
            f"anchors={self.short_anchor:g},{self.long_anchor:g};memory=lognormal]"
        )

    @property
    def natural_names(self) -> tuple[str, ...]:
        return ("clock_rate", "weber_fraction")

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return (self.probe_column,)

    @property
    def comparison_duration(self) -> float:
        """The remembered duration a probe is judged against, under the declared rule."""

        return bisection_threshold(self.short_anchor, self.long_anchor, rule=self.rule)

    @property
    def penalised_linear_refusal(self) -> str:
        """Why the penalised-linear route cannot be applied to this family."""

        return (
            "a bisection model divides its linear term by an estimated memory width, so the "
            "coordinate does not enter a linear predictor and there is no design matrix to "
            "widen; its rows are independent, so it composes through "
            "behavio.contracts.bounded.BoundedCoordinateEstimator instead"
        )

    # -- natural-coordinate conveniences ---------------------------------------------------

    def parameters_from_components(
        self, *, clock_rate: float, weber_fraction: float
    ) -> Mapping[str, float]:
        """Validate and encode natural-scale parameters onto the estimated coordinate."""

        components = BisectionParameters(clock_rate, weber_fraction)
        return self.from_natural(
            {
                "clock_rate": components.clock_rate,
                "weber_fraction": components.weber_fraction,
            }
        )

    def parameter_components(
        self, parameters: Mapping[str, float] | FitResult
    ) -> BisectionParameters:
        """Decode an estimated coordinate into the natural clock rate and Weber fraction."""

        values = self.to_natural(self.coordinate(parameters))
        return BisectionParameters(
            clock_rate=float(values["clock_rate"]),
            weber_fraction=float(values["weber_fraction"]),
        )

    def bisection_point(self, parameters: Mapping[str, float] | FitResult) -> float:
        """Return the probe duration at which this model reports *long* half the time.

        :math:`\\sqrt{SL}/\\kappa` under the ratio rule and :math:`(S+L)/(2\\kappa)` under the
        difference rule. Closed form, so it is a check on a fit rather than a summary of one.
        """

        return float(self.comparison_duration / self.parameter_components(parameters).clock_rate)

    def report_probability(
        self, study: Study, parameters: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """Return :math:`P(\\text{long})` on every row of ``study``."""

        problem = self.read_problem(study, require_outcome=False)
        rows = self.shared_rows(problem.n_rows, self.coordinate(parameters))
        return np.asarray(ndtr(self._deviate(problem, rows)[0]), dtype=np.float64)

    # -- reading a study -------------------------------------------------------------------

    def read_problem(self, study: Study, *, require_outcome: bool = True) -> _BisectionProblem:
        """Read and validate this study's probe durations and long/short reports."""

        probes = validated_durations(
            numeric_column(study, self.probe_column, role="task"),
            label=f"probe column {self.probe_column!r}",
            strictly_positive=True,
        )
        if not require_outcome and self.outcome not in study.columns:
            reports = np.zeros_like(probes)
        else:
            reports = self.outcomes(study)
        return _BisectionProblem(log_probes=np.log(probes), reports=reports)

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the observed long/short report of each row, validated."""

        values = numeric_column(study, self.outcome, role="outcome")
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ModelDataError(f"outcome column {self.outcome!r} must contain only zero and one")
        return values

    # -- the likelihood --------------------------------------------------------------------

    def _deviate(
        self, problem: _BisectionProblem, rows: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the probit deviate and the memory's log standard deviation."""

        squared = np.log1p(np.exp(2.0 * rows[:, 1]))
        log_sd = np.sqrt(squared)
        margin = rows[:, 0] + problem.log_probes - float(np.log(self.comparison_duration))
        return margin / log_sd, log_sd

    def row_value_and_gradient(
        self, problem: _BisectionProblem, rows: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its analytic gradient.

        The inverse Mills ratio is formed as ``exp(log phi - log Phi)`` rather than as a
        quotient of two densities, because a probit fitted to a design with well-separated
        probes spends its time where ``Phi`` underflows and the quotient is ``0/0`` exactly
        where the gradient is largest.
        """

        deviate, log_sd = self._deviate(problem, rows)
        log_upper = log_ndtr(deviate)
        log_lower = log_ndtr(-deviate)
        loss = -float(np.sum(problem.reports * log_upper + (1.0 - problem.reports) * log_lower))
        log_density = -0.5 * deviate**2 - 0.5 * _LOG_TWO_PI
        score = problem.reports * np.exp(log_density - log_upper) - (
            1.0 - problem.reports
        ) * np.exp(log_density - log_lower)
        gradient = np.zeros_like(rows)
        gradient[:, 0] = -score / log_sd
        squared_weber = np.exp(2.0 * rows[:, 1])
        gradient[:, 1] = score * deviate * squared_weber / ((1.0 + squared_weber) * log_sd**2)
        return loss, gradient

    # -- the bounded-coordinate per-row members ---------------------------------------------

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> Prediction:
        """Return :math:`P(\\text{long})` under one coordinate per row."""

        prediction_mode = self.prediction_mode(mode)
        problem = self.read_problem(study, require_outcome=False)
        rows = self.row_coordinates(study, coefficients)
        deviate, _ = self._deviate(problem, rows)
        return Prediction(probability=ndtr(deviate), linear_predictor=deviate, mode=prediction_mode)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observed report under one coordinate per row."""

        problem = self.read_problem(study)
        rows = self.row_coordinates(study, coefficients)
        deviate, _ = self._deviate(problem, rows)
        scores = problem.reports * log_ndtr(deviate) + (1.0 - problem.reports) * log_ndtr(-deviate)
        return protected_array(scores, dtype=np.float64)

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Draw a long/short report on every row from the probit."""

        problem = self.read_problem(design, require_outcome=False)
        rows = self.row_coordinates(design, coefficients)
        deviate, _ = self._deviate(problem, rows)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = generator.binomial(1, ndtr(deviate)).astype(np.int8)
        return Study(columns)

    # -- the box, the restarts and the derived quantities --------------------------------------

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:
        """Return the declared finite box the two logarithms are searched in."""

        del study
        return np.asarray([_CLOCK_RATE_BOX, _WEBER_BOX], dtype=np.float64)

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Start from the probe at which the observed reports cross one half.

        A two-parameter probit has one location and one slope, and the design's own empirical
        crossing point locates the first without any fitting; the restarts vary the second,
        which is the coordinate a bisection likelihood is flat in when the probes are few.
        """

        problem = self.read_problem(study)
        box = self.coordinate_box(study)
        location = _empirical_crossing(problem.log_probes, problem.reports)
        clock = float(np.log(self.comparison_duration)) - location
        starts = [
            np.asarray([clock, float(np.log(width))], dtype=np.float64)
            for width in (0.2, 0.1, 0.5, 0.05)
        ]
        clipped = [np.clip(start, box[:, 0], box[:, 1]) for start in starts]
        return tuple(clipped[index % len(clipped)] for index in range(self.n_restarts))

    def derived_science(
        self,
        study: Study,
        estimates: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> tuple[DerivedQuantity, ...]:
        """Report the bisection point, which is what a bisection study publishes."""

        del study
        point = float(self.comparison_duration * np.exp(-float(estimates[0])))
        error = float(point * np.sqrt(max(float(covariance[0, 0]), 0.0)))
        return (
            DerivedQuantity(
                name="bisection_point",
                value=point,
                standard_error=error,
                description=(
                    f"probe duration reported long half the time, under the "
                    f"{self.rule.value} rule; compare with {self.comparison_duration:g}"
                ),
            ),
        )

    # -- findings ----------------------------------------------------------------------------

    def design_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the designs in which a bisection curve's location or slope is not located."""

        probes = numeric_column(study, self.probe_column, role="task")
        positive = probes[probes > 0.0]
        if not positive.size:
            return ()
        findings: list[ModelFinding] = []
        comparison = self.comparison_duration
        distinct = len(np.unique(np.round(positive, 12)))
        if distinct < 3:
            findings.append(
                ModelFinding(
                    code="too_few_probe_durations",
                    severity=WARNING,
                    message=(
                        f"this design offers {distinct} distinct probe duration(s); a "
                        "bisection curve has a location and a slope and cannot separate them "
                        "from fewer than three points"
                    ),
                )
            )
        if not (np.any(positive < comparison) and np.any(positive > comparison)):
            findings.append(
                ModelFinding(
                    code="probes_do_not_span_the_comparison",
                    severity=WARNING,
                    message=(
                        f"every probe falls on one side of {comparison:g}, the "
                        f"{self.rule.value} rule's comparison duration, so the bisection "
                        "point is an extrapolation rather than a measurement"
                    ),
                )
            )
        if np.any(positive < self.short_anchor) or np.any(positive > self.long_anchor):
            findings.append(
                ModelFinding(
                    code="probes_outside_the_anchors",
                    severity=WARNING,
                    message=(
                        "some probes fall outside the trained anchor range, where a "
                        "bisection model is extrapolating a rule the animal was never "
                        "trained to apply"
                    ),
                )
            )
        ratio = float(self.long_anchor) / float(self.short_anchor)
        if ratio < 1.5:
            geometric = bisection_threshold(self.short_anchor, self.long_anchor)
            arithmetic = bisection_threshold(
                self.short_anchor, self.long_anchor, rule=BisectionRule.DIFFERENCE
            )
            findings.append(
                ModelFinding(
                    code="narrow_anchor_ratio",
                    severity=WARNING,
                    message=(
                        f"the anchors differ by a factor of {ratio:g}, so the ratio and "
                        "difference rules place the comparison duration only "
                        f"{abs(geometric - arithmetic):g} apart; this design cannot tell the "
                        "two accounts of the decision rule apart"
                    ),
                )
            )
        return tuple(findings)


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def _log_log_fit(
    log_targets: NDArray[np.float64], log_outcomes: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Return the OLS slope, intercept and residual spread of ``log R`` on ``log T``."""

    centred = log_targets - float(np.mean(log_targets))
    variance = float(np.sum(centred**2))
    if variance <= 0.0:
        return 1.0, float(np.mean(log_outcomes - log_targets)), float(np.std(log_outcomes))
    slope = float(np.sum(centred * (log_outcomes - float(np.mean(log_outcomes)))) / variance)
    intercept = float(np.mean(log_outcomes) - slope * float(np.mean(log_targets)))
    residual = log_outcomes - intercept - slope * log_targets
    return slope, intercept, float(np.std(residual))


def _empirical_crossing(log_probes: NDArray[np.float64], reports: NDArray[np.float64]) -> float:
    """Return the log probe duration at which the observed report rate passes one half."""

    levels = np.unique(log_probes)
    if len(levels) < 2:
        return float(levels[0]) if len(levels) else 0.0
    rates = np.asarray(
        [float(np.mean(reports[log_probes == level])) for level in levels], dtype=np.float64
    )
    if np.all(rates >= 0.5) or np.all(rates <= 0.5):
        return float(np.mean(levels))
    below = int(np.max(np.flatnonzero(rates < 0.5)))
    above = int(np.min(np.flatnonzero(rates > 0.5)))
    if above <= below:
        return float(np.mean(levels))
    span = rates[above] - rates[below]
    weight = 0.0 if span <= 0.0 else (0.5 - rates[below]) / span
    return float(levels[below] + weight * (levels[above] - levels[below]))
