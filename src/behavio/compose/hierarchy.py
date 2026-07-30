"""``hierarchical(model)``: let declared parameters of a model vary by group."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from behavio._internal.arrays import protected_array
from behavio.contracts.compose import (
    GroupBlocks,
    LinearPredictorLikelihood,
    PenalisedDesign,
    PenalisedLinearEstimator,
    VaryingEffects,
    expand_group_design,
    expand_group_penalty,
    group_blocks,
    information_matrix,
    joint_parameter_names,
    linear_predictor,
    parameter_gradient,
    require_penalised_linear,
    validate_predictor_shape,
)
from behavio.contracts.estimator import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    ModelPrediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.models._kernels.curvature import relative_steps, value_difference_hessian
from behavio.models._kernels.introspection import Describable
from behavio.study import REQUIRED_COLUMNS, Study

__all__ = [
    "HierarchicalFitResult",
    "HierarchicalModel",
    "HierarchicalSimulation",
    "hierarchical",
]


def hierarchical(
    model: PenalisedLinearEstimator,
    *,
    over: str = "subject",
    parameters: Sequence[str] | None = None,
    scale: float = 0.5,
    parameter_scales: Mapping[str, float] | None = None,
    estimate_scale: bool = False,
    scale_bounds: tuple[float, float] = (0.05, 2.0),
) -> HierarchicalModel:
    """Return ``model`` with ``parameters`` free to vary by the ``over`` column.

    One joint maximum-a-posteriori fit estimates a population parameter vector and one
    shrunken deviation vector per training group. ``parameters=None`` lets every parameter
    vary, which is what the deleted ``HierarchicalBernoulliHistoryGLM`` did and all it could
    do; naming a subset is the point of this argument, and ``parameter_scales`` puts a
    different prior width on each named parameter.

    With ``estimate_scale=True`` the declared scales become an initial value for a bounded
    Laplace marginal-likelihood estimate of one common multiplier on them, and
    ``scale_bounds`` bounds the resulting scale of the first varying parameter.

    Predictions for a group that was not in training use the population parameters
    explicitly; :meth:`HierarchicalFitResult.group_was_fitted` says which is which.
    """

    require_penalised_linear(model, combinator="hierarchical")
    effects = VaryingEffects.declare(
        model.parameter_names,
        over=over,
        parameters=parameters,
        scale=scale,
        parameter_scales=parameter_scales,
    )
    return HierarchicalModel(
        model=model,
        effects=effects,
        estimate_scale=estimate_scale,
        scale_bounds=scale_bounds,
    )


# --------------------------------------------------------------------------------------
# What a hierarchical fit and a hierarchical simulation retain
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HierarchicalFitResult(FitResult):
    """Population parameters and shrunken group deviations from one joint MAP fit."""

    grouping: str
    groups: tuple[Any, ...]
    varying_parameters: tuple[str, ...]
    group_deviations: NDArray[np.float64]
    group_standard_errors: NDArray[np.float64]
    scales: NDArray[np.float64]
    scale_standard_error: float | None = None
    scale_bounds: tuple[float, float] | None = None
    scale_estimated: bool = False
    scale_at_boundary: bool = False
    unseen_group_policy: str = "population-plugin"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        groups = tuple(_scalar(group) for group in self.groups)
        varying = tuple(self.varying_parameters)
        deviations = protected_array(self.group_deviations, dtype=np.float64)
        standard_errors = protected_array(self.group_standard_errors, dtype=np.float64)
        scales = protected_array(self.scales, dtype=np.float64)
        expected = (len(groups), len(varying))
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("fit groups must be non-empty and unique")
        if not varying or len(set(varying)) != len(varying):
            raise ValueError("varying parameters must be non-empty and unique")
        if set(varying) - set(self.parameter_names):
            raise ValueError("varying parameters must be parameters of the fitted model")
        if deviations.shape != expected or standard_errors.shape != expected:
            raise ValueError("group estimates must have one row per fitted group")
        if not np.all(np.isfinite(deviations)) or not np.all(np.isfinite(standard_errors)):
            raise ValueError("group estimates and standard errors must be finite")
        if np.any(standard_errors < 0):
            raise ValueError("group standard errors must be non-negative")
        if scales.shape != (len(varying),) or not np.all(np.isfinite(scales)):
            raise ValueError("one scale is required per varying parameter")
        if np.any(scales <= 0):
            raise ValueError("varying-effect scales must be positive")
        if not isinstance(self.scale_estimated, bool):
            raise ValueError("scale_estimated must be boolean")
        if not isinstance(self.scale_at_boundary, bool):
            raise ValueError("scale_at_boundary must be boolean")
        if self.scale_estimated:
            if (
                self.scale_standard_error is None
                or not np.isfinite(self.scale_standard_error)
                or self.scale_standard_error < 0
            ):
                raise ValueError("an estimated scale requires a finite non-negative error")
            if self.scale_bounds is None:
                raise ValueError("an estimated scale requires the bounds it was found within")
        elif (
            self.scale_standard_error is not None
            or self.scale_bounds is not None
            or self.scale_at_boundary
        ):
            raise ValueError("fixed scales cannot retain estimation uncertainty")
        if self.unseen_group_policy != "population-plugin":
            raise ValueError("unseen_group_policy must be 'population-plugin'")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "varying_parameters", varying)
        object.__setattr__(self, "group_deviations", deviations)
        object.__setattr__(self, "group_standard_errors", standard_errors)
        object.__setattr__(self, "scales", scales)

    @property
    def group_parameters(self) -> NDArray[np.float64]:
        """Population-plus-deviation values of the varying parameters, by fitted group."""

        population = self.estimates[self._varying_positions]
        return protected_array(population[None, :] + self.group_deviations, dtype=np.float64)

    def parameters_for(self, group: Any) -> Mapping[str, float]:
        """Return one group's parameters, or the population plug-in for an unseen group."""

        values = np.array(self.estimates, dtype=np.float64)
        key = _scalar(group)
        if key in self.groups:
            values[self._varying_positions] += self.group_deviations[self.groups.index(key)]
        return MappingProxyType(dict(zip(self.parameter_names, values.tolist(), strict=True)))

    def group_was_fitted(self, group: Any) -> bool:
        """Report whether prediction can use an estimated deviation for ``group``."""

        return _scalar(group) in self.groups

    @property
    def scale_confidence_interval_95(self) -> tuple[float, float] | None:
        """Return a delta-method log-scale interval when a scale was estimated."""

        if self.scale_standard_error is None:
            return None
        scale = float(self.scales[0])
        log_standard_error = self.scale_standard_error / scale
        return (
            float(scale * np.exp(-1.96 * log_standard_error)),
            float(scale * np.exp(1.96 * log_standard_error)),
        )

    @property
    def _varying_positions(self) -> NDArray[np.intp]:
        selected = set(self.varying_parameters)
        return np.asarray(
            [index for index, name in enumerate(self.parameter_names) if name in selected],
            dtype=np.intp,
        )


@dataclass(frozen=True, slots=True)
class HierarchicalSimulation:
    """Observations paired with the group truth that generated them but is not observed."""

    study: Study
    grouping: str
    groups: tuple[Any, ...]
    parameter_names: tuple[str, ...]
    varying_parameters: tuple[str, ...]
    population_parameters: NDArray[np.float64]
    group_deviations: NDArray[np.float64]

    def __post_init__(self) -> None:
        groups = tuple(_scalar(group) for group in self.groups)
        names = tuple(self.parameter_names)
        varying = tuple(self.varying_parameters)
        population = protected_array(self.population_parameters, dtype=np.float64)
        deviations = protected_array(self.group_deviations, dtype=np.float64)
        if not names or len(set(names)) != len(names):
            raise ValueError("parameter names must be non-empty and unique")
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("simulation groups must be non-empty and unique")
        if population.shape != (len(names),):
            raise ValueError("population parameters must align with the parameter names")
        if deviations.shape != (len(groups), len(varying)):
            raise ValueError("group truth must align with groups and varying parameters")
        if not np.all(np.isfinite(population)) or not np.all(np.isfinite(deviations)):
            raise ValueError("simulation parameters must be finite")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "varying_parameters", varying)
        object.__setattr__(self, "population_parameters", population)
        object.__setattr__(self, "group_deviations", deviations)

    @property
    def group_parameters(self) -> NDArray[np.float64]:
        """Realised population-plus-deviation values of the varying parameters, by group."""

        selected = set(self.varying_parameters)
        positions = [index for index, name in enumerate(self.parameter_names) if name in selected]
        return protected_array(
            self.population_parameters[positions][None, :] + self.group_deviations,
            dtype=np.float64,
        )


# --------------------------------------------------------------------------------------
# The combinator
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HierarchicalModel(Describable):
    """A model whose declared parameters carry Gaussian deviations by group.

    Parameter naming is stable: the reported coordinate is the *population* one, identical
    to the wrapped model's own ``parameter_names``, so a hierarchical model simulates from
    the same named parameters as the model it wraps and a recovery study can compare them
    directly. Group deviations are not parameters of the reported coordinate; they are read
    off :class:`HierarchicalFitResult`, which names them by group label rather than by a
    flattened string.
    """

    model: PenalisedLinearEstimator
    effects: VaryingEffects
    estimate_scale: bool = False
    scale_bounds: tuple[float, float] = (0.05, 2.0)

    def __post_init__(self) -> None:
        if not isinstance(self.effects, VaryingEffects):
            raise TypeError("effects must be a VaryingEffects declaration")
        if not isinstance(self.estimate_scale, bool):
            raise ValueError("estimate_scale must be boolean")
        bounds = _validate_scale_bounds(self.scale_bounds)
        reference = float(self.effects.scales[0])
        if self.estimate_scale and not bounds[0] <= reference <= bounds[1]:
            raise ValueError("the initial scale must lie within scale_bounds")
        object.__setattr__(self, "scale_bounds", bounds)

    # -- identity ---------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"hierarchical-{self.model.model_name}"

    @property
    def signature(self) -> str:
        return (
            f"hierarchical[{self.effects.signature(self.parameter_names)};"
            f"estimate_scale={self.estimate_scale};"
            f"scale_bounds={self.scale_bounds[0]},{self.scale_bounds[1]}]"
            f"({self.model.signature})"
        )

    @property
    def varying_effects(self) -> VaryingEffects:
        """The declaration this model was built from.

        Its presence is also how :func:`behavio.compose.smooth` recognises that a model is
        already hierarchical and refuses to wrap it.
        """

        return self.effects

    @property
    def grouping(self) -> str:
        """The study column whose distinct values index the groups."""

        return self.effects.grouping

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.model.parameter_names)

    @property
    def varying_parameters(self) -> tuple[str, ...]:
        """The parameters that vary by group, in model order."""

        return self.effects.ordered_parameters(self.model.parameter_names)

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return tuple(getattr(self.model, "coefficient_names", self.model.parameter_names))

    @property
    def categories(self) -> tuple[Any, ...]:
        """The wrapped model's outcome coordinate, when it scores a categorical outcome.

        Absent, and raising :class:`AttributeError` rather than returning ``None``, when
        the wrapped model has none: that is what keeps ``hierarchical(glm)`` from
        structurally satisfying
        :class:`~behavio.contracts.estimator.CategoricalBehaviourEstimator`.
        """

        return tuple(self.model.categories)

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        """Return the wrapped model's observed category codes."""

        return self.model.outcome_codes(study)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return tuple(self.model.scored_columns)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        declared = tuple(self.model.required_task_columns)
        if self.grouping in REQUIRED_COLUMNS or self.grouping in declared:
            return declared
        return (*declared, self.grouping)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return tuple(self.model.supported_prediction_modes)

    @property
    def design_spec(self) -> Any:
        return getattr(self.model, "design_spec", None)

    @property
    def knots(self) -> tuple[float, ...] | None:
        """The wrapped model's knots, so pre-fit knot support is still reported."""

        return getattr(self.model, "knots", None)

    @property
    def time(self) -> str | None:
        """The wrapped model's clock, so pre-fit knot support is still reported."""

        return getattr(self.model, "time", None)

    @property
    def declared_priors(self) -> tuple[str, ...]:
        scales = ", ".join(
            f"{name} ~ Normal(0, {scale:.4g})"
            for name, scale in zip(
                self.effects.parameters, self.effects.scales.tolist(), strict=True
            )
        )
        estimated = (
            " (scale estimated by Laplace marginal likelihood)" if self.estimate_scale else ""
        )
        return (
            f"deviations by {self.grouping}: {scales}{estimated}",
            *getattr(self.model, "declared_priors", ()),
        )

    # -- fitting ------------------------------------------------------------------------

    def fit(self, study: Study) -> HierarchicalFitResult:
        """Jointly fit population parameters and shrunken group deviations."""

        blocks = group_blocks(study, self.grouping)
        if blocks.n_groups < 2:
            raise ModelDataError(
                f"hierarchical fitting requires at least two {self.grouping} groups"
            )
        design_matrix = validate_predictor_shape(self.model, self.model.design_matrix(study))
        outcomes = self.model.outcomes(study)
        offsets = self.model.predictor_offsets(study)
        if self.estimate_scale:
            return self._fit_estimated_scale(study, blocks, design_matrix, outcomes, offsets)
        return self._fit_fixed_scale(
            study, blocks, design_matrix, outcomes, offsets, self._scales()
        )

    def _fit_fixed_scale(
        self,
        study: Study,
        blocks: GroupBlocks,
        design_matrix: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        offsets: NDArray[np.float64] | None,
        scales: NDArray[np.float64],
    ) -> HierarchicalFitResult:
        columns = self._columns()
        width = len(columns)
        n_parameters = len(self.parameter_names)
        joint_fit = self.model.fit_penalised(
            PenalisedDesign(
                parameter_names=joint_parameter_names(
                    self.parameter_names, blocks, self.varying_parameters
                ),
                design_matrix=expand_group_design(design_matrix, blocks, columns),
                outcomes=outcomes,
                penalty_matrix=expand_group_penalty(
                    self.model.penalty_matrix(),
                    blocks,
                    self.model.group_penalty(columns, scales),
                ),
                likelihood=self.model.likelihood,
                offsets=offsets,
                derived_estimates=_effective_parameters(n_parameters, columns, blocks.n_groups),
            ),
            model_name=self.model_name,
            model_signature=self.signature,
        )
        population = joint_fit.estimates[:n_parameters]
        deviations = joint_fit.estimates[n_parameters:].reshape(blocks.n_groups, width)
        group_standard_errors = joint_fit.standard_errors[n_parameters:].reshape(
            blocks.n_groups, width
        )
        diagnostics = joint_fit.diagnostics
        return HierarchicalFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=population,
            standard_errors=joint_fit.standard_errors[:n_parameters],
            covariance=joint_fit.covariance[:n_parameters, :n_parameters],
            n_observations=len(study),
            diagnostics=diagnostics,
            grouping=self.grouping,
            groups=blocks.labels,
            varying_parameters=self.varying_parameters,
            group_deviations=deviations,
            group_standard_errors=group_standard_errors,
            scales=scales,
        )

    def _fit_estimated_scale(
        self,
        study: Study,
        blocks: GroupBlocks,
        design_matrix: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        offsets: NDArray[np.float64] | None,
    ) -> HierarchicalFitResult:
        columns = self._columns()
        n_parameters = len(self.parameter_names)
        declared = self._scales()
        reference = float(declared[0])
        penalty = self.model.penalty_matrix()
        likelihood = self.model.likelihood
        blocked = tuple(
            (
                design_matrix[blocks.row_block == block],
                outcomes[blocks.row_block == block],
                None if offsets is None else offsets[blocks.row_block == block],
            )
            for block in range(blocks.n_groups)
        )
        population_fit = self.model.fit_penalised(
            PenalisedDesign(
                parameter_names=self.parameter_names,
                design_matrix=design_matrix,
                outcomes=outcomes,
                penalty_matrix=penalty,
                likelihood=likelihood,
                offsets=offsets,
            ),
            model_name=self.model_name,
            model_signature=self.signature,
        )

        def scaled(log_scale: float) -> NDArray[np.float64]:
            return declared * (float(np.exp(log_scale)) / reference)

        def profile(parameters: NDArray[np.float64]) -> _LaplaceProfile:
            return _laplace_profile(
                blocked,
                np.asarray(parameters[:n_parameters], dtype=np.float64),
                columns=columns,
                group_penalty=self.model.group_penalty(columns, scaled(float(parameters[-1]))),
                likelihood=likelihood,
                population_penalty=penalty,
                max_iterations=self._max_iterations,
                tolerance=self._tolerance,
            )

        def profile_objective(parameters: NDArray[np.float64]) -> float:
            return profile(parameters).objective

        initial = np.concatenate([population_fit.estimates, np.asarray([np.log(reference)])])
        lower, upper = self.scale_bounds
        outer_bounds = [(None, None)] * n_parameters + [(np.log(lower), np.log(upper))]
        outer_fit = minimize(
            profile_objective,
            initial,
            method="L-BFGS-B",
            bounds=outer_bounds,
            options={
                "maxiter": self._max_iterations,
                "ftol": self._tolerance,
                "gtol": self._tolerance,
            },
        )
        optimizer = "L-BFGS-B with Laplace marginal likelihood"
        if not outer_fit.success:
            outer_fit = minimize(
                profile_objective,
                np.asarray(outer_fit.x, dtype=np.float64),
                method="Powell",
                bounds=outer_bounds,
                options={
                    "maxiter": self._max_iterations,
                    "ftol": self._tolerance,
                    "xtol": self._tolerance,
                },
            )
            optimizer = "Powell fallback with Laplace marginal likelihood"
        outer_point = np.asarray(outer_fit.x, dtype=np.float64)
        population = outer_point[:n_parameters]
        scales = scaled(float(outer_point[-1]))
        found = profile(outer_point)
        hessian = value_difference_hessian(
            profile_objective, outer_point, steps=relative_steps(outer_point, scale=1e-3)
        )
        gradient = _numerical_gradient(profile_objective, outer_point)
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        scale = float(scales[0])
        scale_standard_error = scale * standard_errors[-1]
        boundary_tolerance = 1e-5 * max(1.0, upper - lower)
        at_boundary = bool(
            scale - lower <= boundary_tolerance or upper - scale <= boundary_tolerance
        )
        if (scale - lower <= boundary_tolerance and gradient[-1] > 0) or (
            upper - scale <= boundary_tolerance and gradient[-1] < 0
        ):
            gradient[-1] = 0.0
        message = str(outer_fit.message)
        if not found.all_group_modes_converged:
            message = f"{message}; at least one conditional group mode did not converge"
        # Whether a coefficient is large enough to report is the wrapped model's
        # convention, and the only way to ask it is to let it fit: this path writes its own
        # objective, so unlike the fixed-scale path it never handed the model a problem to
        # solve. One joint solve at the settled scale, against a Laplace profile that has
        # already run an inner optimisation per group per outer step, is the cheap half.
        coefficients_at_boundary = self._fit_fixed_scale(
            study, blocks, design_matrix, outcomes, offsets, scales
        ).diagnostics.boundary_estimate
        diagnostics = FitDiagnostics(
            converged=bool(outer_fit.success and found.all_group_modes_converged),
            optimizer=optimizer,
            status=int(outer_fit.status),
            message=message,
            n_iterations=int(outer_fit.nit),
            objective=float(found.objective),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=float(np.linalg.cond(hessian)),
            boundary_estimate=at_boundary or bool(coefficients_at_boundary),
        )
        group_standard_errors = np.sqrt(
            np.maximum(np.diagonal(found.conditional_covariances, axis1=1, axis2=2), 0.0)
        )
        return HierarchicalFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=population,
            standard_errors=standard_errors[:n_parameters],
            covariance=covariance[:n_parameters, :n_parameters],
            n_observations=len(study),
            diagnostics=diagnostics,
            grouping=self.grouping,
            groups=blocks.labels,
            varying_parameters=self.varying_parameters,
            group_deviations=found.deviations,
            group_standard_errors=group_standard_errors,
            scales=scales,
            scale_standard_error=float(scale_standard_error),
            scale_bounds=self.scale_bounds,
            scale_estimated=True,
            scale_at_boundary=at_boundary,
        )

    # -- prediction and scoring ---------------------------------------------------------

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction:
        """Predict with fitted group deviations, or the population plug-in for unseen groups."""

        prediction_mode = self._prediction_mode(mode)
        hierarchical_fit = self._validated_fit(fit)
        design_matrix = self.model.design_matrix(study)
        coefficients = self._row_parameters(study, hierarchical_fit)
        predictor = (
            np.einsum("ij,ij->i", design_matrix, coefficients)
            if design_matrix.ndim == 2
            else np.einsum("rcp,rp->rc", design_matrix, coefficients, optimize=True)
        )
        offsets = self.model.predictor_offsets(study)
        if offsets is not None:
            predictor = predictor + offsets
        return self.model.likelihood.prediction(predictor, mode=prediction_mode)

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score observations under fitted or unseen-group plug-in parameters."""

        outcomes = self.model.outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        return self.model.likelihood.pointwise_log_prob(prediction.linear_predictor, outcomes)

    # -- simulation ----------------------------------------------------------------------

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate observations without exposing the realised group effects as data."""

        return self.simulate_with_effects(design, parameters, seed=seed).study

    def simulate_with_effects(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
        group_deviations: Mapping[Any, Sequence[float]] | None = None,
    ) -> HierarchicalSimulation:
        """Generate observations and retain the realised group truth separately."""

        population = self._parameter_vector(parameters)
        blocks = group_blocks(design, self.grouping)
        columns = self._columns()
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        if group_deviations is None:
            deviations = self.model.draw_group_deviations(
                columns,
                self._scales(),
                groups=blocks.n_groups,
                generator=generator,
            )
        else:
            deviations = self._declared_deviations(blocks, columns, group_deviations)
        rows = np.tile(population, (len(design), 1))
        rows[:, columns] += deviations[blocks.row_block]
        study = self.model.simulate_rows(design, rows, seed=generator)
        return HierarchicalSimulation(
            study=study,
            grouping=self.grouping,
            groups=blocks.labels,
            parameter_names=self.parameter_names,
            varying_parameters=self.varying_parameters,
            population_parameters=population,
            group_deviations=deviations,
        )

    # -- internals -------------------------------------------------------------------------

    @property
    def _max_iterations(self) -> int:
        return int(getattr(self.model, "max_iterations", 1_000))

    @property
    def _tolerance(self) -> float:
        return float(getattr(self.model, "tolerance", 1e-9))

    def _columns(self) -> NDArray[np.intp]:
        return self.effects.columns(self.parameter_names)

    def _scales(self) -> NDArray[np.float64]:
        return self.effects.ordered_scales(self.parameter_names)

    def _parameter_vector(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        expected = set(self.parameter_names)
        observed = set(parameters)
        if observed != expected:
            raise ValueError(
                "parameters must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        values = np.asarray([parameters[name] for name in self.parameter_names], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite")
        return values

    def _declared_deviations(
        self,
        blocks: GroupBlocks,
        columns: NDArray[np.intp],
        declared: Mapping[Any, Sequence[float]],
    ) -> NDArray[np.float64]:
        normalized = {_scalar(group): values for group, values in declared.items()}
        if set(normalized) != {_scalar(label) for label in blocks.labels}:
            raise ValueError("group_deviations must contain every design group exactly")
        deviations = np.empty((blocks.n_groups, len(columns)), dtype=np.float64)
        for position, label in enumerate(blocks.labels):
            values = np.asarray(normalized[_scalar(label)], dtype=np.float64)
            if values.shape != (len(columns),) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "each group deviation needs one finite value per varying parameter"
                )
            deviations[position] = values
        return deviations

    def _row_parameters(self, study: Study, fit: HierarchicalFitResult) -> NDArray[np.float64]:
        columns = self._columns()
        fitted = np.tile(fit.estimates, (len(fit.groups), 1))
        fitted[:, columns] += fit.group_deviations
        lookup = {group: position for position, group in enumerate(fit.groups)}
        if self.grouping not in study.columns:
            raise ModelDataError(f"study is missing grouping column {self.grouping!r}")
        return np.vstack(
            [
                fitted[lookup[key]] if key in lookup else fit.estimates
                for key in (_scalar(value) for value in study[self.grouping])
            ]
        )

    def _validated_fit(self, fit: FitResult) -> HierarchicalFitResult:
        if not isinstance(fit, HierarchicalFitResult):
            raise ValueError("fit result does not retain hierarchical group effects")
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result was produced by a different model specification")
        return fit

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                f"{self.model_name} does not support {prediction_mode.value!r} prediction"
            )
        return prediction_mode


# --------------------------------------------------------------------------------------
# The Laplace profile over a common scale multiplier
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LaplaceProfile:
    objective: float
    deviations: NDArray[np.float64]
    conditional_covariances: NDArray[np.float64]
    all_group_modes_converged: bool


def _laplace_profile(
    blocked: tuple[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None], ...
    ],
    population: NDArray[np.float64],
    *,
    columns: NDArray[np.intp],
    group_penalty: NDArray[np.float64],
    likelihood: LinearPredictorLikelihood,
    population_penalty: NDArray[np.float64],
    max_iterations: int,
    tolerance: float,
) -> _LaplaceProfile:
    """Profile out the group deviations at a fixed scale, family-independently.

    Everything family-specific reaches this function through ``likelihood``: the value, its
    gradient in the linear predictor, and the curvature that turns the group's design into
    an observed information matrix. That is the whole reason
    :class:`~behavio.contracts.compose.LinearPredictorLikelihood` is a separate protocol
    rather than an implementation detail of the estimator.

    Everything shape-specific reaches it through the three contractions in
    :mod:`behavio.contracts.compose`, so a per-category predictor is profiled by this same
    function: a group's rows are a group's rows whatever each of them predicts.
    """

    width = len(columns)
    n_parameters = len(population)
    deviations = np.zeros((len(blocked), width), dtype=np.float64)
    conditional_covariances = np.zeros((len(blocked), width, width), dtype=np.float64)
    sign, log_prior_determinant = np.linalg.slogdet(group_penalty)
    if sign <= 0:
        raise ValueError("the group penalty must be positive definite")
    objective = 0.5 * float(population @ population_penalty @ population)
    all_converged = True

    for index, (block_design, block_outcomes, block_offsets) in enumerate(blocked):
        restricted = block_design[..., columns]

        def conditional(
            deviation: NDArray[np.float64],
            design: NDArray[np.float64] = block_design,
            columns_: NDArray[np.intp] = columns,
            restricted_: NDArray[np.float64] = restricted,
            observed: NDArray[np.float64] = block_outcomes,
            block_offsets_: NDArray[np.float64] | None = block_offsets,
        ) -> tuple[float, NDArray[np.float64]]:
            offset = np.zeros(n_parameters, dtype=np.float64)
            offset[columns_] = deviation
            value, gradient = likelihood.value_and_gradient(
                linear_predictor(design, population + offset, block_offsets_), observed
            )
            value += 0.5 * float(deviation @ group_penalty @ deviation)
            return (
                value,
                parameter_gradient(restricted_, gradient) + group_penalty @ deviation,
            )

        mode = minimize(
            conditional,
            np.zeros(width, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iterations, "ftol": tolerance, "gtol": tolerance},
        )
        deviation = np.asarray(mode.x, dtype=np.float64)
        offset = np.zeros(n_parameters, dtype=np.float64)
        offset[columns] = deviation
        weights = likelihood.curvature(
            linear_predictor(block_design, population + offset, block_offsets)
        )
        conditional_hessian = information_matrix(restricted, weights) + group_penalty
        hessian_sign, log_determinant = np.linalg.slogdet(conditional_hessian)
        if hessian_sign <= 0:
            return _LaplaceProfile(
                objective=float("inf"),
                deviations=deviations,
                conditional_covariances=conditional_covariances,
                all_group_modes_converged=False,
            )
        deviations[index] = deviation
        conditional_covariances[index] = np.linalg.pinv(conditional_hessian, hermitian=True)
        conditional_value, _ = conditional(deviation)
        objective += conditional_value
        objective += 0.5 * (log_determinant - log_prior_determinant)
        all_converged = all_converged and bool(mode.success)

    return _LaplaceProfile(
        objective=float(objective),
        deviations=deviations,
        conditional_covariances=conditional_covariances,
        all_group_modes_converged=all_converged,
    )


def _effective_parameters(
    n_parameters: int, columns: NDArray[np.intp], n_groups: int
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return the map from a joint estimate to each group's population-plus-deviation.

    The quantity a hierarchical fit has to report a boundary on is not a coordinate of the
    vector the optimizer returns -- the joint coordinate holds a population value and a
    deviation from it, and neither of them is the number an animal's behaviour is generated
    by. This names it as a function of that vector so the wrapped model, which owns the
    convention for what "at a boundary" means, can apply it without a combinator ever
    quoting a threshold.
    """

    def effective(joint: NDArray[np.float64]) -> NDArray[np.float64]:
        population = joint[:n_parameters]
        deviations = joint[n_parameters:].reshape(n_groups, len(columns))
        return np.asarray(population[columns][None, :] + deviations, dtype=np.float64)

    return effective


def _numerical_gradient(
    objective: Callable[[NDArray[np.float64]], float],
    optimum: NDArray[np.float64],
) -> NDArray[np.float64]:
    steps = 1e-4 * np.maximum(1.0, np.abs(optimum))
    gradient = np.zeros(len(optimum), dtype=np.float64)
    for index, step in enumerate(steps):
        displacement = np.zeros(len(optimum), dtype=np.float64)
        displacement[index] = step
        gradient[index] = (
            objective(optimum + displacement) - objective(optimum - displacement)
        ) / (2.0 * step)
    return gradient


def _validate_scale_bounds(bounds: tuple[float, float] | None) -> tuple[float, float]:
    if bounds is None or not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError("scale_bounds must contain a lower and an upper value")
    try:
        lower, upper = (float(value) for value in bounds)
    except (TypeError, ValueError):
        raise ValueError("scale_bounds must contain finite numbers") from None
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise ValueError("scale_bounds must be finite, positive, and increasing")
    return lower, upper


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
