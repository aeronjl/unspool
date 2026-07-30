"""A Behavio estimator backed by PyDDM's drift-diffusion solver.

Behavio's own :class:`~behavio.models.ddm.WienerDriftDiffusion` hand-rolls the Wiener
first-passage density with a fixed twelve-term series and a hardcoded switch between the
small-time and large-time expansions at ``t < 0.15``. PyDDM computes the same density with
adaptive Navarro-Fuss term selection and published accuracy bounds, and it is MIT-licensed
and actively maintained. This module is the wrapper that lets a Behavio user fit the
PyDDM density through the whole falsification stack -- prospective splits, matched
comparison, parameter recovery, evidence bundles -- without leaving the ``Study`` contract.

What is wrapped, and what is not
--------------------------------
PyDDM's *solver* and its *fitter* are both wrapped: the likelihood is
:class:`pyddm.LossRobustLikelihood` (or :class:`pyddm.LossLikelihood`) over a
:class:`pyddm.Sample`, minimized by :func:`pyddm.fit_adjust_model`. What is added on top is
everything the Behavio contract needs and PyDDM does not produce:

- **A parameter-name map.** PyDDM names parameters after the objects that hold them
  (``B``, ``x0``, ``nondectime``); Behavio names them ``boundary``, ``starting_bias``,
  ``nondecision_time`` and ``drift.<feature>``. The map is exact and is
  :data:`PARAMETER_CORRESPONDENCE`, not a convention: recovery simulates from one name space
  and refits into the other, so a silent mismatch would make recovery report a clean
  diagonal for the wrong quantity. Boundary separation is *twice* PyDDM's ``B``, and the
  relative start ``w`` is ``(x0 + 1) / 2``; both conversions are checked against Behavio's
  own Wiener density in the test suite.
- **Per-trial interpolated log densities.** PyDDM returns a PDF tabulated on a time grid and
  its own loss reads it by rounding each response time to the nearest grid index, which
  makes a per-trial likelihood a function of the solver's step size. ``pointwise_log_prob``
  interpolates linearly instead, through
  :class:`~behavio.adapters.prediction.DensityPrediction`.
- **A covariance.** PyDDM reports a point estimate and a loss value and nothing about
  uncertainty. The wrapper differences the log-likelihood at the optimum to get an
  observed-information covariance, refuses rather than invents one when that matrix is not
  positive definite, and says which it did in the fit's diagnostics.
- **A deterministic seed.** Differential evolution is stochastic; the seed is a declared
  field and part of the signature, so the same configuration on the same study gives the
  same estimates.

Everything the wrapper predicts is filtered by construction. A diffusion's density for trial
*t* is a function of that trial's design row alone, so ``supported_prediction_modes`` is
``(FILTERED,)`` and
:func:`behavio.adapters.estimator_conformance.check_behaviour_estimator` verifies it
behaviourally rather than taking the declaration on trust.

Where PyDDM strains the contract
--------------------------------
Four places. Every one of them is reported by the fit rather than papered over, and none of
them was fixed by loosening what Behavio asks for.

*PyDDM solves a condition grid, not a trial.* A :class:`pyddm.Sample` groups trials into
unique combinations of its condition columns and solves once per combination. That is ideal
for a psychophysics design with a handful of contrast levels and wrong for a continuous
per-trial covariate, where the number of solves is the number of trials. ``describe()``
reports the count and ``fit`` refuses above :attr:`PyDDMDriftDiffusion.max_conditions`,
rather than appearing to work and taking an hour.

*Two of the five parameters enter PyDDM as grid indices, not as numbers.*
``OverlayNonDecision`` shifts the distribution by ``int(nondecision_time / dt)`` bins and
``ICPointRatio`` places the start on the ``dx`` lattice, both truncating rather than
rounding. The likelihood is therefore a step function of ``nondecision_time`` and of
``starting_bias``: each is identified only to within one lattice cell, truncation biases the
fitted value towards the top of its cell, and a numerical derivative taken inside one cell
measures exactly zero. The wrapper declares the time quantum as a derived quantity and takes
differences that span two cells; the residual bias is real, and Behavio's own
``run_parameter_recovery`` is what makes it visible.

*PyDDM 0.9 reports no convergence flag.* :func:`pyddm.fit_adjust_model` reads SciPy's
``message`` off ``result.__dict__``, which an ``OptimizeResult`` does not populate, so the
field is always empty for differential evolution and ``success`` is discarded entirely. The
wrapper therefore verifies convergence itself; see :meth:`PyDDMDriftDiffusion._local_curvature`
for what it can and cannot claim.

*The far tail of the analytic solution underflows to exactly zero.* PyDDM's grid holds
per-bin masses, and past a few seconds those round to zero even where the analytic density is
merely very small -- Behavio's own series still returns ``6e-10`` where PyDDM returns ``0``.
A single such trial would make a fold's score ``-inf``, so scores are floored at
:data:`~behavio.adapters.prediction.LOG_DENSITY_FLOOR`, the same floor Behavio's own Wiener
density uses, and the number of floored rows is retained on the fit.

One more thing is not a strain but is worth stating: ``pointwise_log_prob`` is *not* the
objective PyDDM minimised, because PyDDM reads its tabulated density by rounding each
response time to the nearest grid bin and the wrapper interpolates. The gap is reported as
``interpolation_gap`` on the fit and shrinks with ``time_step``.

Licence
-------
PyDDM is MIT, as Behavio is, so ``pip install 'behavio[pyddm]'`` changes no obligation.
``docs/foreign-models.md`` carries the matrix for the rest of this ecosystem, which is not
uniformly so permissive.

Import safety
-------------
This module is named ``pyddm`` inside ``behavio.foreign``. Python 3 resolves imports
absolutely, and nothing here writes ``import pyddm`` in any case: the dependency is reached
only through :func:`behavio.foreign._optional.require_pyddm`, so the two names cannot
collide and ``import behavio.foreign.pyddm`` works with the extra uninstalled.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from inspect import signature as _signature
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution as _differential_evolution

from behavio._internal.arrays import protected_array
from behavio.adapters.prediction import (
    LOG_DENSITY_FLOOR,
    DensityPrediction,
)
from behavio.contracts.audit import FitDiagnostics
from behavio.contracts.estimator import (
    DerivedQuantity,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.design.matrix import DesignSpec
from behavio.foreign._optional import PYDDM_SERIES, pyddm_version, require_pyddm
from behavio.models._kernels.design import build_matrix, resolve_design, validate_design_choice
from behavio.models._kernels.introspection import ERROR, WARNING, Describable, ModelFinding
from behavio.task.response_times import ResponseTimeSpec
from behavio.trials import REQUIRED_COLUMNS, Study

UPPER_CHOICE: Final = 1
"""The scored outcome value that means "the upper boundary was crossed"."""

LOWER_CHOICE: Final = 0
"""The scored outcome value that means "the lower boundary was crossed"."""

CHOICE_NAMES: Final = ("upper", "lower")
"""PyDDM's names for the two boundaries.

Deliberately not ``("correct", "error")``, which is PyDDM's default. Behavio's outcome
column is a *choice*, and whether a choice was correct is a fact about the task rather than
about the model; naming the boundaries after accuracy would silently reinterpret every study
whose outcome codes a side rather than a score.
"""

PARAMETER_CORRESPONDENCE: Final = MappingProxyType(
    {
        "boundary": "B (half-separation): boundary = 2 * B",
        "starting_bias": "x0 (signed ratio in [-1, 1]): starting_bias = (x0 + 1) / 2",
        "nondecision_time": "nondectime (seconds): equal",
        "drift.<feature>": "p<k> on design column <feature>: equal, diffusion scale fixed to 1",
    }
)
"""The exact correspondence between Behavio's parameter names and PyDDM's.

Recovery is only worth running when the simulator and the fitter mean the same thing by the
same name, and these four lines are the whole of what could go wrong. They are exported so a
reader can check the arithmetic without opening the source.
"""

_LOSSES: Final = ("robust-likelihood", "likelihood")
_FITTING_METHODS: Final = ("differential_evolution", "simplex")
_MASS_TOLERANCE: Final = 1e-3


@dataclass(frozen=True, slots=True)
class _LocalCurvature:
    """What differencing the likelihood at the reported optimum established, or did not."""

    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    gradient_norm: float | None
    converged: bool
    estimated: bool
    message: str


@dataclass(frozen=True, slots=True)
class PyDDMFitResult(FitResult):
    """A PyDDM fit with the foreign solver's own evidence retained.

    The common :class:`~behavio.contracts.FitResult` fields are the interoperability floor.
    What PyDDM knows and Behavio's fields have nowhere to put -- which loss was minimized,
    which solver produced the densities, how many condition grids were solved, how many rows
    the density floor caught -- stays here rather than being discarded or squeezed into the
    diagnostics message.

    ``interpolation_gap`` is the one number a reader should look at before treating this
    fit's pointwise scores as the objective that produced it. It is
    ``-sum(pointwise_log_prob) - diagnostics.objective``: how many nats separate Behavio's
    interpolated per-trial likelihood from PyDDM's nearest-grid-bin loss at the same
    parameters. It shrinks with ``time_step`` and is zero for no attainable configuration,
    because the two are different functions of the same density.
    """

    pyddm_version: str
    pyddm_loss: str
    pyddm_solver: str
    n_conditions: int
    likelihood_floor_count: int
    covariance_is_estimated: bool
    interpolation_gap: float

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        if not isinstance(self.pyddm_version, str) or not self.pyddm_version:
            raise ValueError("a PyDDM fit must record the version that produced it")
        if self.n_conditions < 1:
            raise ValueError("a PyDDM fit must solve at least one condition grid")
        if self.likelihood_floor_count < 0:
            raise ValueError("likelihood_floor_count must be non-negative")
        object.__setattr__(self, "interpolation_gap", float(self.interpolation_gap))


@dataclass(frozen=True, slots=True)
class PyDDMDriftDiffusion(Describable):
    """Two-boundary drift diffusion for binary choice and response time, solved by PyDDM.

    The configuration mirrors :class:`~behavio.models.ddm.WienerDriftDiffusion` field for
    field where the two models mean the same thing, so a study can be fitted by both and the
    fits compared without a translation step: same ``predictors``/``design``, same outcome and
    response-time columns, same parameter names, same natural-scale bounds. What differs is
    what does the arithmetic.

    ``time_step`` is PyDDM's ``dt`` and ``dx``, and ``max_time`` is its ``T_dur``. Both are
    solver settings that change the numbers, so both are in the signature. ``max_time`` must
    exceed every observed response time; PyDDM's loss refuses otherwise, and so does
    :meth:`describe`, before the fit starts.
    """

    predictors: tuple[str, ...] = ()
    outcome: str = "choice"
    response_time: ResponseTimeSpec = field(default_factory=ResponseTimeSpec)
    time_step: float = 0.005
    max_time: float = 4.0
    drift_bound: float = 12.0
    boundary_bounds: tuple[float, float] = (0.1, 5.0)
    starting_bias_bounds: tuple[float, float] = (0.02, 0.98)
    nondecision_time_bounds: tuple[float, float] | None = None
    minimum_decision_time: float = 1e-4
    contaminant_probability: float = 0.0
    loss: str = "robust-likelihood"
    fitting_method: str = "differential_evolution"
    population_size: int = 8
    max_iterations: int = 60
    tolerance: float = 0.01
    seed: int = 0
    max_conditions: int = 64
    convergence_tolerance: float = 0.05
    design: DesignSpec | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        predictors = tuple(self.predictors)
        validate_design_choice(self.design, predictors)
        if len(set(predictors)) != len(predictors):
            raise ValueError("predictors must be unique")
        if any(not isinstance(name, str) or not name for name in predictors):
            raise ValueError("predictor names must be non-empty strings")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if self.outcome in REQUIRED_COLUMNS or self.outcome in predictors:
            raise ValueError("outcome must be distinct from required and predictor columns")
        if not isinstance(self.response_time, ResponseTimeSpec):
            raise TypeError("response_time must be a ResponseTimeSpec")
        if self.response_time.column == self.outcome or self.response_time.column in predictors:
            raise ValueError("response-time, outcome, and predictor columns must be distinct")
        if self.loss not in _LOSSES:
            raise ValueError(f"loss must be one of {list(_LOSSES)}")
        if self.fitting_method not in _FITTING_METHODS:
            raise ValueError(f"fitting_method must be one of {list(_FITTING_METHODS)}")
        for value, name in (
            (self.time_step, "time_step"),
            (self.max_time, "max_time"),
            (self.drift_bound, "drift_bound"),
            (self.minimum_decision_time, "minimum_decision_time"),
            (self.tolerance, "tolerance"),
            (self.convergence_tolerance, "convergence_tolerance"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.time_step >= self.max_time:
            raise ValueError("time_step must be smaller than max_time")
        if not 0.0 <= self.contaminant_probability < 1.0:
            raise ValueError("contaminant_probability must lie in [0, 1)")
        for value, name in (
            (self.population_size, "population_size"),
            (self.max_iterations, "max_iterations"),
            (self.max_conditions, "max_conditions"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        boundary_bounds = _ordered_bounds(self.boundary_bounds, "boundary_bounds")
        starting_bounds = _ordered_bounds(self.starting_bias_bounds, "starting_bias_bounds")
        if boundary_bounds[0] <= 0:
            raise ValueError("boundary_bounds must be positive")
        if not 0 < starting_bounds[0] < starting_bounds[1] < 1:
            raise ValueError("starting_bias_bounds must lie strictly between zero and one")
        nondecision_bounds = self.nondecision_time_bounds
        if nondecision_bounds is not None:
            nondecision_bounds = _ordered_bounds(nondecision_bounds, "nondecision_time_bounds")
            if nondecision_bounds[0] < 0:
                raise ValueError("nondecision_time_bounds must be non-negative")
        object.__setattr__(self, "predictors", predictors)
        object.__setattr__(self, "boundary_bounds", boundary_bounds)
        object.__setattr__(self, "starting_bias_bounds", starting_bounds)
        object.__setattr__(self, "nondecision_time_bounds", nondecision_bounds)

    # -- identity ------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "pyddm-drift-diffusion"

    @property
    def signature(self) -> str:
        """A scientific fingerprint over a foreign configuration.

        Everything that changes the numbers is in it: which solver series produced the
        density, the grid it was tabulated on, the observation coding, the design, the
        natural-scale box, the contaminant mixture, and the entire fitting procedure
        including its seed. Two fits that share this string were produced by the same
        computation on the same coordinate; two that do not were not, whatever else they
        agree on.

        What is deliberately *out* of it: the exact installed PyDDM patch version, which is
        provenance and is recorded on the fit; and anything derived from the study, which
        would make one model have as many signatures as it has datasets.
        """

        predictors = ",".join(self.predictors)
        parts = [
            f"backend=pyddm:{PYDDM_SERIES}",
            f"outcome={self.outcome}",
            f"response_time={self.response_time.column}:{self.response_time.unit.value}",
            f"predictors={predictors}",
            "diffusion_scale=1",
            f"dt={self.time_step}",
            f"t_max={self.max_time}",
            f"contaminant={self.contaminant_probability}",
            f"boundary_bounds={self.boundary_bounds}",
            f"starting_bias_bounds={self.starting_bias_bounds}",
            f"drift_bound={self.drift_bound}",
            f"loss={self.loss}",
            f"fit={self.fitting_method}",
            f"population={self.population_size}",
            f"iterations={self.max_iterations}",
            f"tolerance={self.tolerance}",
            f"convergence_tolerance={self.convergence_tolerance}",
            f"seed={self.seed}",
        ]
        if self.design is not None:
            parts.append(f"design={self.design.signature}")
        if self.nondecision_time_bounds is not None:
            parts.append(f"nondecision_bounds={self.nondecision_time_bounds}")
        return f"{self.model_name}[" + ";".join(parts) + "]"

    @property
    def design_spec(self) -> DesignSpec:
        """The design generating the drift rate, declared or implied by ``predictors``."""

        return resolve_design(self.design, self.predictors)

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """Drift coefficients, one per design column, under the fixed ``drift.`` prefix.

        Identical to :attr:`behavio.models.ddm.WienerDriftDiffusion.coefficient_names` for
        the same design, which is what makes the two models directly comparable and makes a
        recovery study run against either simulator meaningful.
        """

        return tuple(f"drift.{name}" for name in self.design_spec.feature_names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (*self.coefficient_names, "boundary", "starting_bias", "nondecision_time")

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome, self.response_time.column)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        """Design columns that must be declared as task predictors."""

        return tuple(
            column
            for column in self.design_spec.required_columns
            if column not in self.scored_columns and column not in REQUIRED_COLUMNS
        )

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def density_outcome(self) -> str:
        """The continuous column :meth:`predict_density` tabulates a density over."""

        return self.response_time.column

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """The optimizer box, as configured. Reported by :meth:`describe`.

        ``nondecision_time`` appears only when it was declared: with no declared bound the
        fit derives one from the fastest observed response, exactly as Behavio's own
        drift-diffusion model does, so what is reported here is what the model declares
        rather than what a particular study will produce.
        """

        bounds: dict[str, tuple[float, float]] = {
            name: (-self.drift_bound, self.drift_bound) for name in self.coefficient_names
        }
        bounds["boundary"] = self.boundary_bounds
        bounds["starting_bias"] = self.starting_bias_bounds
        if self.nondecision_time_bounds is not None:
            bounds["nondecision_time"] = self.nondecision_time_bounds
        return bounds

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the two study-specific ways a PyDDM fit fails, before it is attempted."""

        findings: list[ModelFinding] = []
        if self.response_time.column in study.columns:
            try:
                seconds = self.response_time.read(study).seconds
            except ValueError as error:
                return (
                    ModelFinding(code="invalid_response_time", severity=ERROR, message=str(error)),
                )
            slowest = float(np.max(seconds))
            if slowest >= self.max_time:
                findings.append(
                    ModelFinding(
                        code="response_time_beyond_grid",
                        severity=ERROR,
                        message=(
                            f"the slowest response is {slowest:g}s and the solver grid stops "
                            f"at max_time={self.max_time:g}s; PyDDM cannot score a response "
                            "its grid does not reach"
                        ),
                    )
                )
            fastest = float(np.min(seconds))
            if self.nondecision_time_bounds is None and (fastest - self.minimum_decision_time <= 0):
                findings.append(
                    ModelFinding(
                        code="response_time_below_minimum_decision",
                        severity=ERROR,
                        message=(
                            f"the fastest response is {fastest:g}s, leaving no room for "
                            f"minimum_decision_time={self.minimum_decision_time:g}s"
                        ),
                    )
                )
        try:
            n_conditions = len(self._condition_rows(self._features(study))[0])
        except ModelDataError:
            return tuple(findings)
        if n_conditions > self.max_conditions:
            findings.append(
                ModelFinding(
                    code="too_many_condition_grids",
                    severity=ERROR,
                    message=(
                        f"this design has {n_conditions} distinct rows and PyDDM solves one "
                        f"grid per distinct row; max_conditions={self.max_conditions}. A "
                        "continuous per-trial covariate makes this the trial count -- bin it, "
                        "or raise max_conditions and accept the cost"
                    ),
                )
            )
        elif n_conditions > self.max_conditions // 2:
            findings.append(
                ModelFinding(
                    code="many_condition_grids",
                    severity=WARNING,
                    message=(
                        f"this design has {n_conditions} distinct rows; every likelihood "
                        "evaluation solves one first-passage grid for each of them"
                    ),
                )
            )
        return tuple(findings)

    # -- fitting -------------------------------------------------------------------------

    def fit(self, study: Study) -> PyDDMFitResult:
        """Minimize PyDDM's likelihood and return a Behavio fit with a covariance."""

        pyddm = require_pyddm()
        self.validate(study)
        features = self._features(study)
        conditions, row_index = self._condition_rows(features)
        outcomes = self._outcomes(study)
        seconds = self.response_time.read(study).seconds
        nondecision_bounds = self._nondecision_bounds(float(np.min(seconds)))
        bounds = self._search_box(nondecision_bounds)

        sample = self._sample(pyddm, features, outcomes, seconds)
        model = self._pyddm_model(pyddm, values=None, bounds=bounds)
        loss = (
            pyddm.LossRobustLikelihood if self.loss == "robust-likelihood" else pyddm.LossLikelihood
        )
        with _quiet_pyddm():
            pyddm.fit_adjust_model(
                sample,
                model,
                lossfunction=loss,
                fitting_method=self.fitting_method,
                fitparams=self._fit_parameters(),
                verbose=False,
            )
        native = dict(
            zip(
                model.get_model_parameter_names(),
                [float(value) for value in model.get_model_parameters()],
                strict=True,
            )
        )
        estimates = self._from_native(native)
        vector = np.asarray([estimates[name] for name in self.parameter_names], dtype=np.float64)

        scores = self._row_log_density(study, estimates, conditions, row_index)
        floor_count = int(np.sum(scores <= LOG_DENSITY_FLOOR))
        curvature = self._local_curvature(pyddm, sample, loss, vector, bounds)
        result = model.get_fit_result()
        pyddm_message = str(result.properties.get("mess", ""))
        return PyDDMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=vector,
            standard_errors=curvature.standard_errors,
            covariance=curvature.covariance,
            n_observations=len(study),
            diagnostics=FitDiagnostics(
                converged=curvature.converged,
                optimizer=f"pyddm:{self.fitting_method}",
                status=0 if curvature.converged else 1,
                message=(
                    f"{curvature.message} "
                    f"[pyddm {pyddm_version()} {pyddm_message or 'reported no message'}, "
                    f"loss {self.loss}, {len(conditions)} condition grids]"
                ),
                n_iterations=None,
                objective=float(result.value()),
                gradient_norm=curvature.gradient_norm,
                hessian_condition=_condition_number(curvature.covariance),
                boundary_estimate=_at_boundary(
                    vector, [bounds[name] for name in self.parameter_names]
                ),
            ),
            derived=(
                DerivedQuantity(
                    name="nondecision_time_grid_quantum",
                    value=self.time_step,
                    description=(
                        "PyDDM shifts the response-time distribution by "
                        "int(nondecision_time / dt) grid steps, truncating rather than "
                        "rounding, so the likelihood is piecewise constant in "
                        "nondecision_time and the parameter is identified only to within "
                        "one solver time step"
                    ),
                ),
            ),
            pyddm_version=pyddm_version(),
            pyddm_loss=str(result.loss),
            pyddm_solver=("analytical" if model.has_analytical_solution() else str(result.method)),
            n_conditions=len(conditions),
            likelihood_floor_count=floor_count,
            covariance_is_estimated=curvature.estimated,
            interpolation_gap=-float(np.sum(scores)) - float(result.value()),
        )

    # -- prediction and scoring ----------------------------------------------------------

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return the probability that the upper boundary is crossed on each trial.

        Derived by integrating the same defective densities :meth:`predict_density` returns
        and renormalising, so the choice half and the latency half of the prediction cannot
        disagree. Renormalisation matters only when the solver grid truncated mass, which
        :attr:`~behavio.adapters.prediction.DensityPrediction.total_mass` reports.
        """

        density = self.predict_density(study, fit, mode=mode)
        categorical = density.choice_prediction()
        probability = np.clip(
            np.asarray(categorical.probability[:, 1], dtype=np.float64),
            np.finfo(float).tiny,
            1.0 - np.finfo(float).eps,
        )
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=PredictionMode(mode),
        )

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        """Return the defective response-time density of each boundary, per trial.

        The two categories are :data:`LOWER_CHOICE` and :data:`UPPER_CHOICE` in that order,
        so ``density[:, 1, :]`` is the upper-boundary density and integrating it gives the
        upper-boundary probability.

        The grid is in **seconds**, whatever unit the response-time column declares, because
        that is the clock a diffusion is defined on and the clock ``pointwise_log_prob``
        scores in. Reporting the density in the column's own unit would make the wrapper's
        pointwise scores incomparable with
        :class:`~behavio.models.ddm.WienerDriftDiffusion`'s over a millisecond study, which
        is the one comparison this wrapper exists to make; a density and the score derived
        from it must be on one clock, and seconds is the clock Behavio already converts to.

        The returned array is ``(n_trials, 2, n_grid)``: with the default grid that is about
        13 kB per trial. Scoring goes through this object rather than around it, so the
        pointwise likelihood and the reported density can never drift apart.
        """

        self._require_mode(mode)
        values = self._parameters(fit)
        conditions, row_index = self._condition_rows(self._features(study))
        grid, table = self._density_table(values, conditions)
        return DensityPrediction(
            grid=grid,
            density=table[row_index],
            outcome=self.response_time.column,
            mode=PredictionMode(mode),
            categories=(LOWER_CHOICE, UPPER_CHOICE),
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Return the joint log density of each observed choice and response time.

        Floored at :data:`~behavio.adapters.prediction.LOG_DENSITY_FLOOR`. PyDDM's grid
        rounds the extreme tail of the analytic density to zero, and one such trial would
        otherwise make an entire fold's score ``-inf``; the count of floored rows is retained
        on the fit as ``likelihood_floor_count``.
        """

        self._require_mode(mode)
        values = self._parameters(fit)
        conditions, row_index = self._condition_rows(self._features(study))
        return self._row_log_density(study, values, conditions, row_index)

    # -- simulation ----------------------------------------------------------------------

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Sample choices and response times from PyDDM's own solved distribution.

        Inverse-transform sampling on the solved joint distribution over (boundary, time
        bin), with a uniform draw inside the sampled bin so the simulated times are
        continuous rather than a lattice. Two uniforms are drawn per design row, in source
        row order, from one seeded generator: the same seed always gives the same study, and
        no global stream is touched.

        This is a simulation *of the wrapped solver*, which is what recovery needs. A
        trajectory-level simulator would sample a different approximation of the same
        process and would make recovery a test of two discretisations rather than of the
        likelihood being fitted.
        """

        values = self._validated_parameters(parameters)
        conditions, row_index = self._condition_rows(self._features(design))
        grid, table = self._density_table(values, conditions)
        step = float(self.time_step)
        masses = table * step
        undecided = 1.0 - np.sum(masses, axis=(1, 2))
        if np.any(undecided > _MASS_TOLERANCE):
            worst = float(np.max(undecided))
            raise ModelDataError(
                f"{worst:.4f} of the probability mass has not terminated by "
                f"max_time={self.max_time:g}s under these parameters, so a simulated study "
                "would silently discard it; raise max_time"
            )
        flattened = masses.reshape(masses.shape[0], -1)
        cumulative = np.cumsum(flattened, axis=1)
        cumulative /= cumulative[:, -1:]

        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n_rows = len(design)
        draw = generator.random(n_rows)
        jitter = generator.random(n_rows)
        n_grid = grid.size
        choices = np.empty(n_rows, dtype=np.int8)
        seconds = np.empty(n_rows, dtype=np.float64)
        for position in range(len(conditions)):
            rows = np.flatnonzero(row_index == position)
            if not rows.size:
                continue
            cell = np.searchsorted(cumulative[position], draw[rows], side="left")
            cell = np.clip(cell, 0, flattened.shape[1] - 1)
            choices[rows] = np.where(cell // n_grid == 1, UPPER_CHOICE, LOWER_CHOICE)
            seconds[rows] = grid[cell % n_grid] + step * jitter[rows]
        seconds = np.maximum(seconds, np.nextafter(0.0, 1.0))

        columns: dict[str, Any] = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        columns[self.response_time.column] = seconds / self.response_time.unit.seconds_per_unit
        return Study(columns)

    # -- the PyDDM boundary --------------------------------------------------------------

    def _pyddm_model(
        self,
        pyddm: Any,
        *,
        values: Mapping[str, float] | None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> Any:
        """Build the PyDDM model, either at explicit values or with fittable ranges."""

        names = self.coefficient_names
        native = tuple(f"p{index}" for index in range(len(names)))
        columns = _condition_names(len(names))

        def get_drift(instance: Any, conditions: Mapping[str, float], **_: Any) -> float:
            return float(
                sum(
                    getattr(instance, parameter) * conditions[column]
                    for parameter, column in zip(native, columns, strict=True)
                )
            )

        drift_class = type(
            "BehavioDesignDrift",
            (pyddm.Drift,),
            {
                "name": "behavio-design-drift",
                "required_parameters": list(native),
                "required_conditions": list(columns),
                "get_drift": get_drift,
            },
        )
        if values is None:
            if bounds is None:
                raise ValueError("a fittable model needs a search box")
            settings: dict[str, Any] = {
                parameter: pyddm.Fittable(minval=bounds[name][0], maxval=bounds[name][1])
                for parameter, name in zip(native, names, strict=True)
            }
            half = _fittable_range(pyddm, bounds["boundary"], 0.5)
            start = _fittable_range(pyddm, bounds["starting_bias"], 2.0, offset=-1.0)
            nondecision = _fittable_range(pyddm, bounds["nondecision_time"], 1.0)
        else:
            settings = {
                parameter: float(values[name])
                for parameter, name in zip(native, names, strict=True)
            }
            half = float(values["boundary"]) / 2.0
            start = 2.0 * float(values["starting_bias"]) - 1.0
            nondecision = float(values["nondecision_time"])
        overlay: Any = pyddm.OverlayNonDecision(nondectime=nondecision)
        if self.contaminant_probability > 0.0:
            overlay = pyddm.OverlayChain(
                overlays=[
                    overlay,
                    pyddm.OverlayUniformMixture(umixturecoef=self.contaminant_probability),
                ]
            )
        return pyddm.Model(
            drift=drift_class(**settings),
            noise=pyddm.NoiseConstant(noise=1.0),
            bound=pyddm.BoundConstant(B=half),
            IC=pyddm.ICPointRatio(x0=start),
            overlay=overlay,
            dt=self.time_step,
            dx=self.time_step,
            T_dur=self.max_time,
            choice_names=CHOICE_NAMES,
        )

    def _sample(
        self,
        pyddm: Any,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        seconds: NDArray[np.float64],
    ) -> Any:
        """Build the PyDDM sample, which regroups trials by boundary and by condition.

        A :class:`pyddm.Sample` is not a table: it stores the upper-boundary times, the
        lower-boundary times, and a matching pair of arrays per condition. Nothing is ever
        read back out of it in row order -- the wrapper's own predictions and scores are
        computed from the ``Study`` directly -- so the regrouping cannot leak into a result.
        """

        upper = outcomes == UPPER_CHOICE
        empty = np.empty(0, dtype=np.float64)
        conditions = {
            name: (features[upper, index], features[~upper, index], empty)
            for index, name in enumerate(_condition_names(features.shape[1]))
        }
        return pyddm.Sample(
            choice_upper=seconds[upper].astype(np.float64),
            choice_lower=seconds[~upper].astype(np.float64),
            undecided=0,
            choice_names=CHOICE_NAMES,
            **conditions,
        )

    def _density_table(
        self,
        values: Mapping[str, float],
        conditions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve one first-passage grid per distinct design row.

        Returns the shared time grid in seconds and a ``(n_conditions, 2, n_grid)`` table of
        defective densities ordered ``(lower, upper)`` -- the order
        :data:`LOWER_CHOICE`, :data:`UPPER_CHOICE` declares, which is the reverse of
        :data:`CHOICE_NAMES`, PyDDM's own order.
        """

        pyddm = require_pyddm()
        model = self._pyddm_model(pyddm, values=values)
        names = _condition_names(conditions.shape[1])
        lower_name, upper_name = CHOICE_NAMES[1], CHOICE_NAMES[0]
        grid = np.asarray([], dtype=np.float64)
        table: list[NDArray[np.float64]] = []
        with _quiet_pyddm():
            for row in conditions:
                solution = model.solve(
                    dict(zip(names, (float(value) for value in row), strict=True))
                )
                if not grid.size:
                    grid = np.asarray(solution.t_domain, dtype=np.float64)
                table.append(
                    np.stack(
                        (
                            np.asarray(solution.pdf(lower_name), dtype=np.float64),
                            np.asarray(solution.pdf(upper_name), dtype=np.float64),
                        )
                    )
                )
        return grid, np.clip(np.asarray(table, dtype=np.float64), 0.0, None)

    def _row_log_density(
        self,
        study: Study,
        values: Mapping[str, float],
        conditions: NDArray[np.float64],
        row_index: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        grid, table = self._density_table(values, conditions)
        prediction = DensityPrediction(
            grid=grid,
            density=table[row_index],
            outcome=self.response_time.column,
            mode=PredictionMode.FILTERED,
            categories=(LOWER_CHOICE, UPPER_CHOICE),
        )
        outcomes = self._outcomes(study)
        seconds = self.response_time.read(study).seconds
        return prediction.observed_log_density(seconds, outcomes.astype(np.int64))

    def _local_curvature(
        self,
        pyddm: Any,
        sample: Any,
        loss: Any,
        vector: NDArray[np.float64],
        bounds: Mapping[str, tuple[float, float]],
    ) -> _LocalCurvature:
        """Difference the log-likelihood at the optimum, or refuse to.

        PyDDM reports a point estimate and a loss value. It reports no uncertainty, and --
        because :func:`pyddm.fit_adjust_model` reads SciPy's ``message`` off
        ``result.__dict__``, which an ``OptimizeResult`` does not populate -- it reports no
        convergence flag for differential evolution either. Both have to be recovered here,
        and both are recovered from the same central differences.

        *Convergence* is **coordinate-wise local optimality on the solver lattice**: no
        single-parameter move of one difference step in either direction improves PyDDM's
        loss by more than ``convergence_tolerance`` nats. That is a weaker claim than "the
        optimizer's stopping rule fired", and it is the strongest claim available. A
        gradient-based test is not: the likelihood is a *step function* of two of the five
        coordinates (see below), so its numerical gradient at any point is an artefact of
        where the reported estimate happens to sit inside a lattice cell, and a
        stationarity test on it reports non-convergence for a fit that no local move can
        improve. The probe evaluations are the ones the Hessian already needs, so the test
        is free.

        Everything here is differenced on **PyDDM's own loss**, not on the interpolated
        score ``pointwise_log_prob`` reports. The two differ -- PyDDM reads its tabulated
        density by rounding each response time to the nearest grid index -- and a covariance
        must be the curvature of the objective that actually produced the estimate, or the
        interval it implies is centred on a point that does not maximise it. The gap between
        the two objectives is itself reported, as ``interpolation_gap`` on the fit.

        *Step sizes* are per parameter and are not a matter of taste, because two of the
        five coordinates enter PyDDM as **grid indices** rather than as numbers:
        ``OverlayNonDecision`` shifts by ``int(nondecision_time / dt)`` bins and
        ``ICPointRatio`` places the start on the ``dx`` lattice. The likelihood is a step
        function of both, so a difference taken inside one cell measures exactly zero and a
        singular information matrix follows. The steps below span two cells of whichever
        lattice the coordinate lives on; the drift coefficients and the boundary enter the
        analytic solution continuously and take an ordinary relative step.

        When a coordinate is pinned so close to its bound that its step will not fit, the
        whole covariance is ``NaN`` rather than a number the wrapper does not believe.
        ``FitResult.audit()`` records that as a warning, not a failure, so the fit stays
        eligible for comparison while saying plainly that its interval is unknown.
        """

        size = len(vector)
        wanted = np.asarray(
            [self._difference_step(name, vector) for name in self.parameter_names],
            dtype=np.float64,
        )
        room = np.asarray(
            [
                0.9
                * min(
                    float(vector[index]) - bounds[name][0], bounds[name][1] - float(vector[index])
                )
                for index, name in enumerate(self.parameter_names)
            ],
            dtype=np.float64,
        )
        steps = np.minimum(wanted, np.maximum(room, 0.0))
        probed = steps > 0
        crosses_lattice = bool(np.all(probed) and np.all(steps >= wanted - 1e-15))

        function = loss(
            sample,
            required_conditions=list(_condition_names(len(self.coefficient_names))),
            T_dur=self.max_time,
            dt=self.time_step,
            nparams=size,
            samplesize=len(sample),
        )

        def objective(point: NDArray[np.float64]) -> float:
            values = dict(zip(self.parameter_names, point.tolist(), strict=True))
            with _quiet_pyddm():
                return -float(function.loss(self._pyddm_model(pyddm, values=values)))

        centre = objective(vector)
        neighbours: dict[int, tuple[float, float]] = {}
        for index in range(size):
            if not probed[index]:
                continue
            offset = np.zeros(size)
            offset[index] = steps[index]
            neighbours[index] = (objective(vector + offset), objective(vector - offset))
        if not neighbours:
            return _unknown_curvature(
                size,
                "no covariance and no convergence check: every parameter is pinned against "
                "its bound, so no admissible local move exists",
            )
        gain = max(max(high, low) for high, low in neighbours.values()) - centre
        converged = bool(np.isfinite(gain) and gain <= self.convergence_tolerance)
        verdict = (
            f"the best single-parameter move of one solver-lattice step changes PyDDM's "
            f"loss by {-gain:+.2e} nats (tolerance {self.convergence_tolerance:g})"
        )
        if not crosses_lattice:
            missing = [
                name
                for index, name in enumerate(self.parameter_names)
                if steps[index] < wanted[index] - 1e-15
            ]
            return _unknown_curvature(
                size,
                f"{verdict}; no covariance: {missing} sit too close to their bounds for a "
                "difference wide enough to cross PyDDM's solver lattice",
                converged=converged,
            )

        gradient = np.empty(size, dtype=np.float64)
        hessian = np.empty((size, size), dtype=np.float64)
        for i in range(size):
            high, low = neighbours[i]
            gradient[i] = (high - low) / (2.0 * steps[i])
            hessian[i, i] = (high - 2.0 * centre + low) / steps[i] ** 2
            forward = np.zeros(size)
            forward[i] = steps[i]
            for j in range(i + 1, size):
                right = np.zeros(size)
                right[j] = steps[j]
                value = (
                    objective(vector + forward + right)
                    - objective(vector + forward - right)
                    - objective(vector - forward + right)
                    + objective(vector - forward - right)
                ) / (4.0 * steps[i] * steps[j])
                hessian[i, j] = hessian[j, i] = value
        information = -hessian
        gradient_norm = float(np.linalg.norm(gradient))
        if not np.all(np.isfinite(information)) or not np.isfinite(gradient_norm):
            return _unknown_curvature(
                size,
                f"{verdict}; no covariance: the differenced loss is not finite",
                converged=converged,
            )
        if float(np.min(np.linalg.eigvalsh(information))) <= 0:
            return _unknown_curvature(
                size,
                f"{verdict}; no covariance: the observed information is not positive "
                "definite, so this is not an interior maximum",
                converged=converged,
                gradient_norm=gradient_norm,
            )
        covariance = np.linalg.inv(information)
        diagonal = np.diag(covariance)
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            return _unknown_curvature(
                size,
                f"{verdict}; no covariance: the inverted information has a non-positive variance",
                converged=converged,
                gradient_norm=gradient_norm,
            )
        return _LocalCurvature(
            covariance=protected_array(covariance, dtype=np.float64),
            standard_errors=protected_array(np.sqrt(diagonal), dtype=np.float64),
            gradient_norm=gradient_norm,
            converged=converged,
            estimated=True,
            message=verdict,
        )

    def _difference_step(self, name: str, vector: NDArray[np.float64]) -> float:
        """The finite-difference step for one coordinate, on its own lattice."""

        values = dict(zip(self.parameter_names, vector.tolist(), strict=True))
        if name == "nondecision_time":
            return 2.0 * self.time_step
        if name == "starting_bias":
            # x0 lives on the dx lattice over [-B, B]; one lattice cell is dx / B in x0,
            # hence dx / (2 * half-separation) = time_step / boundary in Behavio's w.
            return 2.0 * self.time_step / max(values["boundary"], self.boundary_bounds[0])
        return 1e-3 * max(1.0, abs(values[name]))

    # -- study reading -------------------------------------------------------------------

    def _features(self, study: Study) -> NDArray[np.float64]:
        design = self.design_spec
        for name in design.required_columns:
            if name not in study.columns:
                raise ModelDataError(f"study is missing predictor {name!r}")
        return np.asarray(build_matrix(design, study).values, dtype=np.float64)

    def _condition_rows(
        self, features: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Return the distinct design rows and, per trial, which one it is.

        This is where PyDDM's shape meets Behavio's. PyDDM solves per *condition
        combination*; Behavio hands it a per-trial design matrix. The distinct rows of that
        matrix are the smallest set of grids that scores every trial exactly.
        """

        conditions, row_index = np.unique(features, axis=0, return_inverse=True)
        return conditions, np.asarray(row_index, dtype=np.intp).reshape(-1)

    def _outcomes(self, study: Study) -> NDArray[np.float64]:
        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        try:
            outcomes = np.asarray(study[self.outcome], dtype=np.float64)
        except (TypeError, ValueError):
            raise ModelDataError(f"outcome column {self.outcome!r} must be numeric") from None
        if not np.all(np.isfinite(outcomes)) or not np.all((outcomes == 0) | (outcomes == 1)):
            raise ModelDataError(f"outcome column {self.outcome!r} must contain only zero and one")
        return outcomes

    # -- parameters ----------------------------------------------------------------------

    def parameters_from_components(
        self,
        *,
        drift: Mapping[str, float],
        boundary: float,
        starting_bias: float = 0.5,
        nondecision_time: float = 0.0,
    ) -> Mapping[str, float]:
        """Validate and pack natural-scale drift and timing parameters.

        Mirrors :meth:`behavio.models.ddm.WienerDriftDiffusion.parameters_from_components`
        exactly, so one parameter set can be handed to either model.
        """

        if set(drift) != set(self.coefficient_names):
            expected, observed = set(self.coefficient_names), set(drift)
            raise ValueError(
                "drift must match coefficient_names exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        values = {name: float(drift[name]) for name in self.coefficient_names}
        values["boundary"] = float(boundary)
        values["starting_bias"] = float(starting_bias)
        values["nondecision_time"] = float(nondecision_time)
        return MappingProxyType(self._validated_parameters(values))

    def _validated_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        expected, observed = set(self.parameter_names), set(parameters)
        if observed != expected:
            raise ValueError(
                "parameters must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            values = {name: float(parameters[name]) for name in self.parameter_names}
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        if not np.all(np.isfinite(list(values.values()))):
            raise ValueError("parameters must contain finite numeric values")
        if values["boundary"] <= 0:
            raise ValueError("boundary must be positive")
        if not 0 < values["starting_bias"] < 1:
            raise ValueError("starting_bias must lie strictly between zero and one")
        if values["nondecision_time"] < 0:
            raise ValueError("nondecision_time must be non-negative")
        return values

    def _parameters(self, fit: FitResult) -> dict[str, float]:
        if not isinstance(fit, FitResult):
            raise TypeError("fit must be a FitResult")
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result belongs to a different model specification")
        return self._validated_parameters(fit.parameters)

    def _from_native(self, native: Mapping[str, float]) -> dict[str, float]:
        """Map PyDDM's parameter names and scales onto Behavio's.

        The whole of :data:`PARAMETER_CORRESPONDENCE`, executed.
        """

        values = {
            name: float(native[f"p{index}"]) for index, name in enumerate(self.coefficient_names)
        }
        values["boundary"] = 2.0 * float(native["B"])
        values["starting_bias"] = (float(native["x0"]) + 1.0) / 2.0
        values["nondecision_time"] = float(native["nondectime"])
        return values

    def _nondecision_bounds(self, fastest: float) -> tuple[float, float]:
        if self.nondecision_time_bounds is not None:
            return self.nondecision_time_bounds
        highest = fastest - self.minimum_decision_time
        if highest <= 0:
            raise ModelDataError(
                "minimum response time is too short for the configured minimum_decision_time"
            )
        return 0.0, highest

    def _search_box(
        self, nondecision_bounds: tuple[float, float]
    ) -> dict[str, tuple[float, float]]:
        box = dict(self.parameter_bounds)
        box["nondecision_time"] = nondecision_bounds
        return box

    def _fit_parameters(self) -> dict[str, Any]:
        if self.fitting_method != "differential_evolution":
            return {}
        settings: dict[str, Any] = {
            "popsize": self.population_size,
            "maxiter": self.max_iterations,
            "tol": self.tolerance,
            "polish": True,
        }
        settings[_SEED_KEYWORD] = self.seed
        return settings

    def _require_mode(self, mode: PredictionMode) -> None:
        if PredictionMode(mode) not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "PyDDMDriftDiffusion supports only filtered prediction: a first-passage "
                "density for one trial is a function of that trial's design row, so there is "
                "no smoothed counterpart to report"
            )


def _condition_names(count: int) -> tuple[str, ...]:
    """PyDDM condition names for a design of ``count`` columns.

    Positional rather than derived from the design's feature names, because a Behavio
    feature name is free text -- ``stimulus:choice_lag_1``, ``contrast[left]`` -- and PyDDM
    passes conditions to a drift function as Python keyword arguments, which they are not
    valid as. The correspondence is the design's column order, which is fixed by
    :class:`~behavio.design.DesignSpec` and reported by ``describe()``.
    """

    return tuple(f"c{index}" for index in range(count))


def _fittable_range(pyddm: Any, bounds: tuple[float, float], scale: float, *, offset: float = 0.0):
    low = bounds[0] * scale + offset
    high = bounds[1] * scale + offset
    return pyddm.Fittable(minval=low, maxval=high)


_SEED_KEYWORD: Final = "rng" if "rng" in _signature(_differential_evolution).parameters else "seed"
"""Which keyword this SciPy spells differential evolution's deterministic stream as.

SciPy renamed ``seed`` to ``rng`` during its random-generator transition and Behavio
supports both of the SciPy versions that straddle it, so the keyword is resolved once from
the installed signature rather than pinned to one spelling.
"""


def _quiet_pyddm() -> Any:
    """Silence PyDDM's per-evaluation logging and its renormalisation warnings.

    PyDDM logs one line per likelihood evaluation at INFO and warns whenever a solved
    distribution is renormalised, which a differential-evolution fit triggers thousands of
    times while it explores. None of it is a finding about the fit; what *is* a finding --
    the loss, the message, the boundary contact, the floored rows -- is retained on the
    :class:`PyDDMFitResult`.
    """

    return _QuietPyDDM()


class _QuietPyDDM:
    __slots__ = ("_catcher", "_level", "_logger")

    def __enter__(self) -> None:
        self._logger = logging.getLogger("pyddm")
        self._level = self._logger.level
        self._logger.setLevel(logging.ERROR)
        self._catcher = warnings.catch_warnings()
        self._catcher.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exception: Any) -> None:
        self._catcher.__exit__(*exception)
        self._logger.setLevel(self._level)


def _unknown_curvature(
    size: int,
    message: str,
    *,
    converged: bool = False,
    gradient_norm: float | None = None,
) -> _LocalCurvature:
    return _LocalCurvature(
        covariance=protected_array(np.full((size, size), np.nan), dtype=np.float64),
        standard_errors=protected_array(np.full(size, np.nan), dtype=np.float64),
        gradient_norm=gradient_norm,
        converged=converged,
        estimated=False,
        message=message,
    )


def _condition_number(covariance: NDArray[np.float64]) -> float | None:
    if not np.all(np.isfinite(covariance)):
        return None
    try:
        return float(np.linalg.cond(covariance))
    except np.linalg.LinAlgError:  # pragma: no cover - a singular matrix is already NaN
        return None


def _at_boundary(vector: NDArray[np.float64], bounds: Sequence[tuple[float, float]]) -> bool:
    for value, (lower, upper) in zip(vector, bounds, strict=True):
        tolerance = 1e-4 * max(1.0, upper - lower)
        if value - lower <= tolerance or upper - value <= tolerance:
            return True
    return False


def _ordered_bounds(values: Sequence[float], name: str) -> tuple[float, float]:
    pair = tuple(float(value) for value in values)
    if len(pair) != 2 or not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
        raise ValueError(f"{name} must be a finite ordered pair")
    return pair[0], pair[1]


__all__ = [
    "CHOICE_NAMES",
    "LOWER_CHOICE",
    "PARAMETER_CORRESPONDENCE",
    "UPPER_CHOICE",
    "PyDDMDriftDiffusion",
    "PyDDMFitResult",
]
