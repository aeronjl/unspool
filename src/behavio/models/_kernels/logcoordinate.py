"""One estimator spine for models whose whole coordinate is a logarithm.

Three families in Behavio estimate nothing but strictly positive quantities: a clock rate, a
Weber fraction, a giving-up intake rate, a decision noise. Every one of them is estimated as
:math:`\\log` of itself, for the reason
:class:`~behavio.contracts.bounded.BoundedCoordinateEstimator` states in its own docstring --
a Gaussian group deviation and a Wald interval are both only honest where the parameter is
unbounded -- and once that is true of *every* coordinate of a model, the whole estimator
surface is the same code.

:class:`LogCoordinateModel` is that code. A family supplies six things:

``natural_names``
    The reported coordinate, in order. The estimated coordinate is these with ``_log``
    appended, and the natural map is :func:`numpy.exp` on every entry, so
    :class:`~behavio.contracts.natural.NaturalParameterisation` is satisfied here once.
``read_problem(study)``
    Everything one study's likelihood reads, validated once and carried as one object.
``row_value_and_gradient(problem, rows)``
    The negative log likelihood and its gradient in one coordinate vector per row. This is
    the *only* member that knows what the family is a model of.
``coordinate_box(study)`` and ``initial_points(study)``
    Where the coordinate is searched and from where.
``predict_rows`` / ``simulate_rows`` / ``pointwise_log_prob_rows``
    The three per-row members :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator`
    asks for, which differ by family because a reproduced duration, a binary report and a
    censored residence time are three different observations.

Everything else -- the multi-start solver, the numerical curvature, the natural
parameterisation, the row objective, the penalty, the group prior, ``fit_rows``, ``predict``,
``pointwise_log_prob`` and the shared refusals -- is written here and is family-independent.
:mod:`behavio.models.economic` makes the same split one level lower down, sharing a softmax
between two value functions; this module shares an *estimator* between three likelihoods that
have nothing else in common.

Why the row coordinate is the only likelihood member
----------------------------------------------------
``fit`` here is ``row_value_and_gradient`` evaluated at one coordinate tiled over every row,
exactly as :mod:`behavio.models.economic` does it, and for the same reason: a family whose
rows are independent has no second likelihood, so a shared-coordinate objective that was
written separately would be a second implementation of one thing. The consequence is that
every model on this spine composes under ``smooth()`` and ``hierarchical()`` without writing
anything a plain ``fit`` did not already need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from behavio._internal.arrays import protected_array
from behavio.contracts.audit import FitDiagnostics
from behavio.contracts.bounded import RowCoefficientDesign, validated_row_coefficients
from behavio.contracts.compose import ridge_group_draw, ridge_group_penalty
from behavio.contracts.estimator import (
    FitResult,
    ModelDataError,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.contracts.natural import natural_quantities
from behavio.models._kernels.curvature import (
    covariance_from_hessian,
    finite_difference_hessian,
    offset_steps,
)
from behavio.models._kernels.introspection import Describable, ModelFinding
from behavio.models._kernels.rowfit import solve_row_coefficients
from behavio.trials import REQUIRED_COLUMNS, Study

__all__ = [
    "BOUNDARY_TOLERANCE",
    "LogCoordinateFitResult",
    "LogCoordinateModel",
    "numeric_column",
    "positive_integer",
    "require_positive",
    "validated_columns",
]

#: Coordinate at which a positive parameter's logarithm is treated as resting on its bound.
BOUNDARY_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class LogCoordinateFitResult(FitResult):
    """A log-coordinate fit that retains every deterministic restart.

    The four restart fields are the :class:`~behavio.diagnostics.MultistartFit` contract, so
    :meth:`~behavio.contracts.FitResult.audit` derives a
    :class:`~behavio.diagnostics.RestartAudit` from this fit without either side knowing
    about the other. They are retained for the reason every other optimizer-backed family in
    the package retains them: agreement between restarts is evidence only fitting can
    produce, and a likelihood with a scale parameter in it has more than one basin whenever
    the design is thin.
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


class LogCoordinateModel(Describable):
    """A model every one of whose parameters is the logarithm of a positive quantity.

    ``__slots__`` is empty so that a frozen, slotted model dataclass can inherit this without
    gaining an instance dictionary, exactly as :class:`Describable` is inherited elsewhere.
    """

    __slots__ = ()

    # -- what a family supplies ----------------------------------------------------------

    @property
    def natural_names(self) -> tuple[str, ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    def read_problem(self, study: Study) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def row_value_and_gradient(
        self, problem: Any, rows: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    def initial_points(
        self, study: Study
    ) -> tuple[NDArray[np.float64], ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    def design_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """The family's own design-degeneracy findings, if it has any."""

        del study
        return ()

    # -- the estimator surface -----------------------------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The estimated coordinate: every reported name with ``_log`` appended."""

        return tuple(f"{name}_log" for name in self.natural_names)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        """One observed number per row, so the scalar case, spelled out.

        Declared rather than left absent because a mixture component is checked against it:
        a process that writes a joint choice and latency is not a mixture of one observation
        with a family that writes a single duration.
        """

        return ()

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Exponentiate the estimated coordinate onto the reported one."""

        vector = self.coordinate_vector(estimates)
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
            require_positive(float(natural[name]), name)
            values[coordinate] = float(np.log(float(natural[name])))
        return MappingProxyType(values)

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return ``d natural / d estimated``, which is diagonal because every map is ``exp``."""

        return np.diag(np.exp(self.coordinate_vector(estimates)))

    # -- fit -----------------------------------------------------------------------------

    def shared_value_and_gradient(
        self, problem: Any, vector: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Score one shared coordinate by broadcasting it over the rows.

        Deliberately not a second likelihood. These families score their rows independently,
        so the shared-coordinate case *is* the row-coordinate case with a ``tile``, and
        writing it twice would leave two things to keep in agreement.
        """

        rows = np.tile(np.asarray(vector, dtype=np.float64), (problem.n_rows, 1))
        loss, gradient = self.row_value_and_gradient(problem, rows)
        return loss, np.asarray(gradient.sum(axis=0), dtype=np.float64)

    def fit(self, study: Study) -> LogCoordinateFitResult:
        """Fit by deterministic multi-start L-BFGS-B inside the declared box."""

        problem = self.read_problem(study)
        box = self.coordinate_box(study)
        bounds = [(float(low), float(high)) for low, high in box]
        starts = self.initial_points(study)
        results = [
            minimize(
                lambda vector: self.shared_value_and_gradient(problem, vector),
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
        value, gradient = self.shared_value_and_gradient(problem, estimates)
        hessian = finite_difference_hessian(
            lambda vector: self.shared_value_and_gradient(problem, vector)[1],
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
            boundary_estimate=coordinate_at_box(estimates, box),
        )
        return LogCoordinateFitResult(
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
                *self.derived_science(study, estimates, covariance),
            ),
            restart_objectives=objectives,
            restart_converged=np.asarray([bool(result.success) for result in results]),
            restart_messages=tuple(str(result.message) for result in results),
            selected_restart=selected,
        )

    def derived_science(
        self,
        study: Study,
        estimates: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> tuple[Any, ...]:
        """Return the family's reportable functions of a fit beyond the natural coordinate.

        A bisection point and a marginal-value overstaying ratio are numbers a reader
        publishes and the optimizer never sees, which is exactly what
        :class:`~behavio.contracts.estimator.DerivedQuantity` is for. Empty by default.
        """

        del study, estimates, covariance
        return ()

    # -- predict and score, both through the per-row members ------------------------------

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Any:
        """Return the prediction implied by one fitted coordinate, shared across rows."""

        prediction_mode = self.prediction_mode(mode)
        self.validate_fit(fit)
        rows = self.shared_rows(len(study), np.asarray(fit.estimates, dtype=np.float64))
        return self.predict_rows(study, rows, mode=prediction_mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observation under one fitted coordinate."""

        self.prediction_mode(mode)
        self.validate_fit(fit)
        rows = self.shared_rows(len(study), np.asarray(fit.estimates, dtype=np.float64))
        return self.pointwise_log_prob_rows(study, rows)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate one observation per row from a single shared coordinate."""

        rows = self.shared_rows(len(design), self.coordinate(parameters))
        return self.simulate_rows(design, rows, seed=seed)

    # -- the bounded-coordinate composition contract --------------------------------------

    def row_objective(self, study: Study) -> _LogCoordinateRowObjective:
        """Return this study's negative log likelihood in one coordinate per row."""

        return _LogCoordinateRowObjective(
            model=self,
            problem=self.read_problem(study),
            n_parameters=len(self.parameter_names),
        )

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the quadratic penalty on the coordinate, which is none.

        These are maximum-likelihood fits inside a box; whatever regularisation they have
        comes from the box and from the logarithms. Composition adds a group or roughness
        prior on top, which is the only place a penalty on these families comes from.
        """

        width = len(self.parameter_names)
        return np.zeros((width, width), dtype=np.float64)

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

        A subject whose Weber fraction is 0.1 and one whose fraction is 0.3 differ by about
        one log unit, not by 0.2, and only the first statement stays positive for every draw.
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

        Every parameter here is a logarithm with a declared finite box saying where the
        logarithm stops meaning anything, and
        :func:`~behavio.models._kernels.rowfit.solve_row_coefficients` already reports a
        coordinate resting on that box. Saying so here is the answer rather than an omission.
        """

        del estimates, derived
        return False

    # -- findings ------------------------------------------------------------------------

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report this design's degeneracies, reached through ``describe(study)``.

        Guarded against a study the model cannot read at all: a missing column is already an
        error finding of its own, and a second traceback on top of it says nothing new.
        """

        try:
            return self.design_findings(study)
        except ModelDataError:
            return ()

    # -- shared plumbing -------------------------------------------------------------------

    def shared_rows(self, n_rows: int, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tile one coordinate over every row of a study."""

        return np.tile(self.coordinate_vector(vector), (int(n_rows), 1))

    def coordinate_vector(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Validate one estimated coordinate vector."""

        try:
            vector = np.asarray(estimates, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("estimates must contain finite numeric values") from None
        width = len(self.parameter_names)
        if vector.shape != (width,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"estimates must contain {width} finite values")
        return vector

    def coordinate(self, parameters: Mapping[str, float] | FitResult) -> NDArray[np.float64]:
        """Read an estimated coordinate out of a mapping or a fit."""

        if isinstance(parameters, FitResult):
            self.validate_fit(parameters)
            return self.coordinate_vector(parameters.estimates)
        if not isinstance(parameters, Mapping) or set(parameters) != set(self.parameter_names):
            raise ValueError("parameters must match the model exactly")
        try:
            values = [float(parameters[name]) for name in self.parameter_names]
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        return self.coordinate_vector(np.asarray(values, dtype=np.float64))

    def row_coordinates(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Validate one coordinate vector per study row."""

        return validated_row_coefficients(
            coefficients,
            n_rows=len(study),
            n_parameters=len(self.parameter_names),
            what="row coefficients",
        )

    def validate_fit(self, fit: FitResult) -> None:
        """Refuse a fit produced by a different model specification."""

        if (
            fit.model_name != self.model_name
            or fit.model_signature != self.signature
            or fit.parameter_names != self.parameter_names
        ):
            raise ValueError("fit result was produced by a different model specification")

    def prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        """Refuse every prediction mode but the filtered one."""

        prediction_mode = PredictionMode(mode)
        if prediction_mode is not PredictionMode.FILTERED:
            raise UnsupportedPredictionMode(
                f"{self.model_name} supports only filtered prediction, "
                f"not {prediction_mode.value!r}"
            )
        return prediction_mode


@dataclass(frozen=True, slots=True)
class _LogCoordinateRowObjective:
    """The likelihood as a function of one coordinate vector per row.

    Every row is its own block. That is a claim about these families rather than a
    convenience: a reproduced duration, a bisection report and a patch residence time each
    depend on that row's own task columns and on nothing any earlier row wrote, so there is
    no trace for a mid-block parameter change to make ambiguous and ``smooth(model,
    over="trial")`` is admissible where the same call on a value-updating agent is an error.
    """

    model: LogCoordinateModel
    problem: Any
    n_parameters: int

    @property
    def n_rows(self) -> int:
        """The number of scored rows."""

        return int(self.problem.n_rows)

    @property
    def row_blocks(self) -> NDArray[np.intp]:
        """Every row is independent, so every row is its own block."""

        return np.arange(self.n_rows, dtype=np.intp)

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the row coordinates."""

        coordinates = validated_row_coefficients(
            rows, n_rows=self.n_rows, n_parameters=self.n_parameters, what="row coordinates"
        )
        return self.model.row_value_and_gradient(self.problem, coordinates)


# --------------------------------------------------------------------------------------
# Small shared helpers, deliberately duplicated from behavio.models.economic
# --------------------------------------------------------------------------------------


def coordinate_at_box(estimates: NDArray[np.float64], box: NDArray[np.float64]) -> bool:
    """Whether any coordinate rests on its declared bound."""

    tolerances = BOUNDARY_TOLERANCE * np.maximum(1.0, box[:, 1] - box[:, 0])
    return bool(
        np.any(estimates - box[:, 0] <= tolerances) or np.any(box[:, 1] - estimates <= tolerances)
    )


def numeric_column(study: Study, name: str, *, role: str) -> NDArray[np.float64]:
    """Read one finite numeric study column, or raise a readable model data error."""

    if name not in study.columns:
        raise ModelDataError(f"study is missing {role} column {name!r}")
    try:
        values = np.asarray(study[name], dtype=np.float64)
    except (TypeError, ValueError):
        raise ModelDataError(f"{role} column {name!r} must be numeric") from None
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ModelDataError(f"{role} column {name!r} must contain finite numbers")
    return values


def validated_columns(names: Sequence[str], label: str) -> tuple[str, ...]:
    """Return declared column names, refusing duplicates and required-column collisions."""

    columns = tuple(names)
    if any(not isinstance(name, str) or not name for name in columns):
        raise ValueError(f"{label} must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{label} must contain distinct column names")
    if set(columns) & set(REQUIRED_COLUMNS):
        raise ValueError(f"{label} cannot replace a required Study column")
    return columns


def require_positive(value: float, label: str) -> None:
    """Raise unless ``value`` is a finite, strictly positive real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{label} must be a positive real number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def positive_integer(value: int, label: str) -> None:
    """Raise unless ``value`` is a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
