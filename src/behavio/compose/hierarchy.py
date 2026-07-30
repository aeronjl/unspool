"""``hierarchical(model)``: let declared parameters of a model vary by group."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from behavio._internal.arrays import protected_array
from behavio.contracts.compose import (
    GroupBlocks,
    GroupExpansion,
    LinearPredictorLikelihood,
    PenalisedDesign,
    PenalisedLinearEstimator,
    VaryingEffects,
    expand_group_box,
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
    "UnseenGroupPrediction",
    "hierarchical",
]

SCALE_ESTIMATORS = ("laplace-profile", "laplace-em")
"""The two ways a group scale may be estimated rather than declared."""

SCALE_UNCERTAINTY_POLICIES = MappingProxyType(
    {
        "local": "local-expected-prior-curvature",
        "observed": "louis-observed-information",
        "supplemented": "supplemented-laplace-em",
    }
)
"""How ``laplace-em`` reports uncertainty on the scales it estimated."""


def hierarchical(
    model: PenalisedLinearEstimator,
    *,
    over: str = "subject",
    parameters: Sequence[str] | None = None,
    scale: float = 0.5,
    parameter_scales: Mapping[str, float] | None = None,
    estimate_scale: bool = False,
    scale_estimator: str = "laplace-profile",
    scale_bounds: tuple[float, float] = (0.05, 2.0),
    scale_max_iterations: int = 12,
    scale_tolerance: float = 0.01,
    scale_uncertainty: str = "observed",
) -> HierarchicalModel:
    """Return ``model`` with ``parameters`` free to vary by the ``over`` column.

    One joint maximum-a-posteriori fit estimates a population parameter vector and one
    shrunken deviation vector per training group. ``parameters=None`` lets every parameter
    vary, which is what the deleted ``HierarchicalBernoulliHistoryGLM`` did and all it could
    do; naming a subset is the point of this argument, and ``parameter_scales`` puts a
    different prior width on each named parameter.

    A name may be either a parameter of ``model`` or something ``model`` says one parameter
    name stands for: for a smooth model, naming ``"boundary"`` names every knot of the
    boundary path, because a path varies by group as a whole path and one scale governs the
    whole of it. :meth:`~behavio.contracts.compose.PenalisedLinearEstimator.\
group_parameter_expansion` is what is asked.

    With ``estimate_scale=True`` the declared scales stop being fixed and become the
    starting point of one of two estimators. ``scale_estimator="laplace-profile"`` profiles
    the group deviations out at a fixed scale and optimises **one common multiplier** on the
    declared scales against the Laplace marginal likelihood; it is what the hand-written
    hierarchical GLM did. ``scale_estimator="laplace-em"`` alternates a joint MAP fit with a
    closed-form M-step and estimates **one scale per named parameter**; it is what the
    hand-written hierarchical smooth drift-diffusion family did, and it is the one to use
    when the parameters that vary are not commensurable. ``scale_uncertainty`` selects how
    the EM estimator reports uncertainty: ``"local"`` is uncorrected complete-data
    curvature, ``"observed"`` applies Louis' identity, and ``"supplemented"`` differences
    the EM map itself.

    Predictions for a group that was not in training use the population parameters
    explicitly; :meth:`HierarchicalFitResult.group_was_fitted` says which is which, and
    :meth:`HierarchicalModel.predict_new_groups` integrates the fitted random effect
    instead of plugging the population in.
    """

    require_penalised_linear(model, combinator="hierarchical")
    selected, scales = _resolve_declaration(model, parameters, parameter_scales)
    effects = VaryingEffects.declare(
        model.parameter_names,
        over=over,
        parameters=selected,
        scale=scale,
        parameter_scales=scales,
    )
    return HierarchicalModel(
        model=model,
        effects=effects,
        scale_parameters=_scale_parameter_names(model, parameters),
        estimate_scale=estimate_scale,
        scale_estimator=scale_estimator,
        scale_bounds=scale_bounds,
        scale_max_iterations=scale_max_iterations,
        scale_tolerance=scale_tolerance,
        scale_uncertainty=scale_uncertainty,
    )


def _scale_parameter_names(
    model: PenalisedLinearEstimator, parameters: Sequence[str] | None
) -> tuple[str, ...]:
    """Name the blocks that share one prior scale, in model order.

    One block per declared name, which for a smooth model is one per coefficient path and
    not one per knot: the whole point of naming ``"boundary"`` is that the boundary has *a*
    scale rather than one scale for each of the values its path happens to be stored as.
    """

    if parameters is None:
        return tuple(model.parameter_names)
    order = {name: index for index, name in enumerate(model.parameter_names)}
    declared = tuple(parameters)
    return tuple(
        sorted(declared, key=lambda name: order.get(model.group_parameter_expansion(name)[0], 0))
    )


def _resolve_declaration(
    model: PenalisedLinearEstimator,
    parameters: Sequence[str] | None,
    parameter_scales: Mapping[str, float] | None,
) -> tuple[tuple[str, ...] | None, Mapping[str, float] | None]:
    """Expand declared names into reported parameters, carrying each name's scale along."""

    if parameters is None:
        if parameter_scales is None:
            return None, None
        expanded_all: dict[str, float] = {}
        for name, value in parameter_scales.items():
            for reported in model.group_parameter_expansion(name):
                expanded_all[reported] = float(value)
        return None, expanded_all
    if isinstance(parameters, str):
        raise ValueError("parameters must be a sequence of parameter names")
    declared = tuple(parameters)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("varying parameters must be non-empty and unique")
    selected: list[str] = []
    scales: dict[str, float] = {}
    overrides = dict(parameter_scales or {})
    unknown = [name for name in overrides if name not in declared]
    if unknown:
        raise ValueError(f"parameter_scales names non-varying parameters: {sorted(unknown)}")
    for name in declared:
        reported = tuple(model.group_parameter_expansion(name))
        selected.extend(reported)
        if name in overrides:
            for entry in reported:
                scales[entry] = float(overrides[name])
    return tuple(selected), (scales or None)


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
    scale_parameters: tuple[str, ...] = ()
    scale_standard_errors: NDArray[np.float64] | None = None
    scale_local_standard_errors: NDArray[np.float64] | None = None
    scale_covariance: NDArray[np.float64] | None = None
    scale_em_rate_matrix: NDArray[np.float64] | None = None
    scale_bounds: tuple[float, float] | None = None
    scale_estimated: bool = False
    scales_at_boundary: NDArray[np.bool_] | None = None
    scale_estimation_iterations: int = 0
    scale_estimation_converged: bool = True
    scale_estimation_policy: str = "fixed"
    scale_uncertainty_policy: str = "fixed"
    unseen_group_policy: str = "population-plugin"

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        groups = tuple(_scalar(group) for group in self.groups)
        varying = tuple(self.varying_parameters)
        blocks = tuple(self.scale_parameters) or varying
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
        if scales.shape != (len(blocks),) or not np.all(np.isfinite(scales)):
            raise ValueError("one scale is required per scale parameter")
        if np.any(scales <= 0):
            raise ValueError("varying-effect scales must be positive")
        if not isinstance(self.scale_estimated, bool):
            raise ValueError("scale_estimated must be boolean")
        if not isinstance(self.scale_estimation_converged, bool):
            raise ValueError("scale_estimation_converged must be boolean")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "varying_parameters", varying)
        object.__setattr__(self, "scale_parameters", blocks)
        object.__setattr__(self, "group_deviations", deviations)
        object.__setattr__(self, "group_standard_errors", standard_errors)
        object.__setattr__(self, "scales", scales)
        self._validate_scale_estimation(scales)

    def _validate_scale_estimation(self, scales: NDArray[np.float64]) -> None:
        errors = _optional_array(self.scale_standard_errors, np.float64)
        local = _optional_array(self.scale_local_standard_errors, np.float64)
        covariance = _optional_array(self.scale_covariance, np.float64)
        rate = _optional_array(self.scale_em_rate_matrix, np.float64)
        at_boundary = _optional_array(self.scales_at_boundary, np.bool_)
        if self.scale_estimated:
            if self.scale_bounds is None:
                raise ValueError("an estimated scale requires the bounds it was found within")
            if errors is None or errors.shape != scales.shape or np.any(errors < 0):
                raise ValueError("an estimated scale requires finite non-negative errors")
            if not np.all(np.isfinite(errors)):
                raise ValueError("an estimated scale requires finite non-negative errors")
            if self.scale_estimation_policy not in ("laplace-profile", "laplace-em"):
                raise ValueError("an estimated scale must record which estimator found it")
            if covariance is not None and covariance.shape != (len(scales), len(scales)):
                raise ValueError("a scale covariance must be square over the scale parameters")
            if at_boundary is not None and at_boundary.shape != scales.shape:
                raise ValueError("scale boundary indicators must align with the scales")
            if local is not None and local.shape != scales.shape:
                raise ValueError("local scale errors must align with the scales")
        elif (
            errors is not None
            or local is not None
            or covariance is not None
            or rate is not None
            or at_boundary is not None
            or self.scale_bounds is not None
            or self.scale_estimation_iterations != 0
            or not self.scale_estimation_converged
            or self.scale_estimation_policy != "fixed"
            or self.scale_uncertainty_policy != "fixed"
        ):
            raise ValueError("fixed scales cannot retain estimation uncertainty")
        if self.unseen_group_policy != "population-plugin":
            raise ValueError("unseen_group_policy must be 'population-plugin'")
        object.__setattr__(self, "scale_standard_errors", errors)
        object.__setattr__(self, "scale_local_standard_errors", local)
        object.__setattr__(self, "scale_covariance", covariance)
        object.__setattr__(self, "scale_em_rate_matrix", rate)
        object.__setattr__(self, "scales_at_boundary", at_boundary)

    @property
    def group_parameters(self) -> NDArray[np.float64]:
        """Population-plus-deviation values of the varying parameters, by fitted group."""

        population = self.estimates[self._varying_positions]
        return protected_array(population[None, :] + self.group_deviations, dtype=np.float64)

    @property
    def group_parameter_vectors(self) -> NDArray[np.float64]:
        """Each fitted group's complete reported coordinate, one row per group.

        The whole vector and not only the varying part, so that a caller can hand it back
        to the wrapped model -- to a smooth model's ``knot_grid``, say -- without having to
        reassemble the parameters that did not vary.
        """

        vectors = np.tile(np.asarray(self.estimates, dtype=np.float64), (len(self.groups), 1))
        vectors[:, self._varying_positions] += self.group_deviations
        return protected_array(vectors, dtype=np.float64)

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
    def scale(self) -> float:
        """The first scale, which is the whole of it when one multiplier was estimated."""

        return float(self.scales[0])

    @property
    def scale_standard_error(self) -> float | None:
        """The first scale's standard error, for a fit that estimated one multiplier."""

        if self.scale_standard_errors is None:
            return None
        return float(self.scale_standard_errors[0])

    @property
    def scale_at_boundary(self) -> bool:
        """Whether any estimated scale met a declared bound."""

        if self.scales_at_boundary is None:
            return False
        return bool(np.any(self.scales_at_boundary))

    @property
    def scale_map(self) -> Mapping[str, float]:
        """The fitted or declared scale of each named scale parameter."""

        return MappingProxyType(dict(zip(self.scale_parameters, self.scales.tolist(), strict=True)))

    @property
    def scale_standard_error_map(self) -> Mapping[str, float] | None:
        """Approximate scale uncertainty, by scale parameter, when scales were estimated."""

        if self.scale_standard_errors is None:
            return None
        return MappingProxyType(
            dict(zip(self.scale_parameters, self.scale_standard_errors.tolist(), strict=True))
        )

    @property
    def scale_at_boundary_map(self) -> Mapping[str, bool] | None:
        """Whether each estimated scale met a declared bound, by scale parameter."""

        if self.scales_at_boundary is None:
            return None
        return MappingProxyType(
            dict(zip(self.scale_parameters, self.scales_at_boundary.tolist(), strict=True))
        )

    @property
    def scale_confidence_interval_95(self) -> tuple[float, float] | None:
        """Return a delta-method log-scale interval for the first scale."""

        intervals = self.scale_confidence_intervals_95
        if intervals is None:
            return None
        return intervals[self.scale_parameters[0]]

    @property
    def scale_confidence_intervals_95(self) -> Mapping[str, tuple[float, float]] | None:
        """Return delta-method log-scale intervals, clipped to the declared bounds."""

        if self.scale_standard_errors is None:
            return None
        return self._scale_intervals(self.scale_standard_errors)

    @property
    def scale_local_confidence_intervals_95(self) -> Mapping[str, tuple[float, float]] | None:
        """Return uncorrected complete-data curvature ranges, which are not calibrated.

        These ranges use expected-prior curvature alone and are systematically narrower
        than the observed information allows, so they undercover their nominal rate. They
        are retained as an optimization diagnostic and as the comparison baseline for the
        corrections reported by :attr:`scale_confidence_intervals_95`.
        """

        if self.scale_local_standard_errors is None:
            return None
        return self._scale_intervals(self.scale_local_standard_errors)

    @property
    def scale_em_spectral_radius(self) -> float | None:
        """Return the largest absolute EM-rate eigenvalue when supplemented."""

        if self.scale_em_rate_matrix is None:
            return None
        return float(np.max(np.abs(np.linalg.eigvals(self.scale_em_rate_matrix))))

    def _scale_intervals(
        self, standard_errors: NDArray[np.float64]
    ) -> Mapping[str, tuple[float, float]]:
        lower_bound, upper_bound = (
            self.scale_bounds if self.scale_bounds is not None else (0.0, np.inf)
        )
        intervals: dict[str, tuple[float, float]] = {}
        for parameter, scale, standard_error in zip(
            self.scale_parameters, self.scales, standard_errors, strict=True
        ):
            log_standard_error = float(standard_error / scale)
            log_scale = float(np.log(scale))
            intervals[parameter] = (
                float(
                    np.exp(
                        max(
                            np.log(lower_bound) if lower_bound > 0 else -np.inf,
                            log_scale - 1.96 * log_standard_error,
                        )
                    )
                ),
                float(np.exp(min(np.log(upper_bound), log_scale + 1.96 * log_standard_error))),
            )
        return MappingProxyType(intervals)

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

        return protected_array(
            self.population_parameters[self._varying_positions][None, :] + self.group_deviations,
            dtype=np.float64,
        )

    @property
    def group_parameter_vectors(self) -> NDArray[np.float64]:
        """Each group's complete realised coordinate, one row per group."""

        vectors = np.tile(
            np.asarray(self.population_parameters, dtype=np.float64), (len(self.groups), 1)
        )
        vectors[:, self._varying_positions] += self.group_deviations
        return protected_array(vectors, dtype=np.float64)

    @property
    def _varying_positions(self) -> list[int]:
        selected = set(self.varying_parameters)
        return [index for index, name in enumerate(self.parameter_names) if name in selected]


@dataclass(frozen=True, slots=True)
class UnseenGroupPrediction:
    """Empirical-Bayes random-effect prediction for entirely unseen groups.

    The population plug-in that :meth:`HierarchicalModel.predict` uses for an unseen group
    is the *mode* of that group's prior, which is not the same as its predictive
    distribution: a new animal is not an average animal. This integrates the fitted random
    effect by Monte Carlo instead, and retains the diagnostics that say whether the
    integration was adequate.
    """

    groups: tuple[Any, ...]
    row_group_indices: NDArray[np.intp]
    probability: NDArray[np.float64]
    draw_probabilities: NDArray[np.float64]
    draw_pointwise_log_probability: NDArray[np.float64]
    pointwise_marginal_log_probability: NDArray[np.float64]
    group_joint_log_probability: NDArray[np.float64]
    group_effective_draws: NDArray[np.float64]
    group_log_probability_mcse: NDArray[np.float64]
    random_effect_draws: NDArray[np.float64]
    varying_parameters: tuple[str, ...]
    scales: NDArray[np.float64]
    seed: int
    policy: str = "empirical-bayes-random-effect-monte-carlo"

    def __post_init__(self) -> None:
        groups = tuple(_scalar(group) for group in self.groups)
        row_groups = protected_array(self.row_group_indices, dtype=np.intp)
        probability = protected_array(self.probability, dtype=np.float64)
        draw_probabilities = protected_array(self.draw_probabilities, dtype=np.float64)
        draw_pointwise = protected_array(self.draw_pointwise_log_probability, dtype=np.float64)
        pointwise = protected_array(self.pointwise_marginal_log_probability, dtype=np.float64)
        joint = protected_array(self.group_joint_log_probability, dtype=np.float64)
        effective_draws = protected_array(self.group_effective_draws, dtype=np.float64)
        monte_carlo_error = protected_array(self.group_log_probability_mcse, dtype=np.float64)
        draws = protected_array(self.random_effect_draws, dtype=np.float64)
        parameters = tuple(self.varying_parameters)
        scales = protected_array(self.scales, dtype=np.float64)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("predictive groups must be non-empty and unique")
        if row_groups.ndim != 1 or np.any((row_groups < 0) | (row_groups >= len(groups))):
            raise ValueError("row group indices must reference predictive groups")
        n_draws = draw_probabilities.shape[0] if draw_probabilities.ndim == 2 else -1
        if n_draws < 1 or draw_probabilities.shape[1:] != probability.shape:
            raise ValueError("draw probabilities must align with predicted rows")
        if draw_pointwise.shape != draw_probabilities.shape:
            raise ValueError("draw log probabilities must align with predicted rows")
        if probability.shape != row_groups.shape or pointwise.shape != probability.shape:
            raise ValueError("predictive row summaries must have one value per row")
        if joint.shape != (len(groups),):
            raise ValueError("group joint scores must have one value per group")
        if effective_draws.shape != joint.shape or monte_carlo_error.shape != joint.shape:
            raise ValueError("Monte Carlo diagnostics must have one value per group")
        if draws.shape != (n_draws, len(groups), len(parameters)):
            raise ValueError("random-effect draws must align with draws, groups, and parameters")
        if scales.shape != (len(parameters),) and scales.ndim != 1:
            raise ValueError("predictive scales must be one-dimensional")
        if not np.all(np.isfinite(draw_probabilities)) or np.any(
            (draw_probabilities < 0) | (draw_probabilities > 1)
        ):
            raise ValueError("draw probabilities must be finite values between zero and one")
        if (
            not np.all(np.isfinite(probability))
            or not np.all(np.isfinite(draw_pointwise))
            or not np.all(np.isfinite(pointwise))
        ):
            raise ValueError("predictive row summaries must be finite")
        if not np.all(np.isfinite(joint)) or not np.all(np.isfinite(draws)):
            raise ValueError("predictive scores and random effects must be finite")
        if (
            not np.all(np.isfinite(effective_draws))
            or np.any((effective_draws < 1.0) | (effective_draws > n_draws))
            or not np.all(np.isfinite(monte_carlo_error))
            or np.any(monte_carlo_error < 0)
        ):
            raise ValueError("Monte Carlo diagnostics must be finite and admissible")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("predictive seed must be a non-negative integer")
        if self.policy != "empirical-bayes-random-effect-monte-carlo":
            raise ValueError("unsupported unseen-group prediction policy")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "row_group_indices", row_groups)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "draw_probabilities", draw_probabilities)
        object.__setattr__(self, "draw_pointwise_log_probability", draw_pointwise)
        object.__setattr__(self, "pointwise_marginal_log_probability", pointwise)
        object.__setattr__(self, "group_joint_log_probability", joint)
        object.__setattr__(self, "group_effective_draws", effective_draws)
        object.__setattr__(self, "group_log_probability_mcse", monte_carlo_error)
        object.__setattr__(self, "random_effect_draws", draws)
        object.__setattr__(self, "varying_parameters", parameters)
        object.__setattr__(self, "scales", scales)

    @property
    def n_draws(self) -> int:
        """The number of random-effect draws the marginal was averaged over."""

        return len(self.draw_probabilities)

    @property
    def group_joint_log_probability_map(self) -> Mapping[Any, float]:
        """Each unseen group's integrated joint score, by group label."""

        return MappingProxyType(
            dict(zip(self.groups, self.group_joint_log_probability.tolist(), strict=True))
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
    scale_parameters: tuple[str, ...] = ()
    estimate_scale: bool = False
    scale_estimator: str = "laplace-profile"
    scale_bounds: tuple[float, float] = (0.05, 2.0)
    scale_max_iterations: int = 12
    scale_tolerance: float = 0.01
    scale_uncertainty: str = "observed"

    def __post_init__(self) -> None:
        if not isinstance(self.effects, VaryingEffects):
            raise TypeError("effects must be a VaryingEffects declaration")
        if not isinstance(self.estimate_scale, bool):
            raise ValueError("estimate_scale must be boolean")
        if self.scale_estimator not in SCALE_ESTIMATORS:
            raise ValueError(f"scale_estimator must be one of {SCALE_ESTIMATORS}")
        if self.scale_uncertainty not in SCALE_UNCERTAINTY_POLICIES:
            raise ValueError("scale_uncertainty must be 'local', 'observed', or 'supplemented'")
        if self.scale_uncertainty == "supplemented" and not (
            self.estimate_scale and self.scale_estimator == "laplace-em"
        ):
            raise ValueError(
                "supplemented scale uncertainty requires estimate_scale=True with "
                "scale_estimator='laplace-em'"
            )
        if (
            isinstance(self.scale_max_iterations, bool)
            or not isinstance(self.scale_max_iterations, int)
            or self.scale_max_iterations < 1
        ):
            raise ValueError("scale_max_iterations must be a positive integer")
        if not np.isfinite(self.scale_tolerance) or self.scale_tolerance <= 0:
            raise ValueError("scale_tolerance must be finite and positive")
        bounds = _validate_scale_bounds(self.scale_bounds)
        blocks = tuple(self.scale_parameters) or tuple(self.effects.parameters)
        if self.estimate_scale and not all(
            bounds[0] <= scale <= bounds[1] for scale in self._block_scales(blocks)
        ):
            raise ValueError("the initial scale must lie within scale_bounds")
        object.__setattr__(self, "scale_bounds", bounds)
        object.__setattr__(self, "scale_parameters", blocks)

    # -- identity ---------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"hierarchical-{self.model.model_name}"

    @property
    def signature(self) -> str:
        estimator = (
            f";scale_estimator={self.scale_estimator};"
            f"scale_max_iterations={self.scale_max_iterations};"
            f"scale_tolerance={self.scale_tolerance};"
            f"scale_uncertainty={self.scale_uncertainty}"
            if self.estimate_scale and self.scale_estimator == "laplace-em"
            else ""
        )
        return (
            f"hierarchical[{self.effects.signature(self.parameter_names)};"
            f"estimate_scale={self.estimate_scale}{estimator};"
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
                self.scale_parameters, self._block_scales(self.scale_parameters), strict=True
            )
        )
        estimated = f" (scale estimated by {self.scale_estimator})" if self.estimate_scale else ""
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
        if self.estimate_scale and self.scale_estimator == "laplace-em":
            return self._fit_estimated_scales_em(study, blocks, design_matrix, outcomes, offsets)
        if self.estimate_scale:
            return self._fit_estimated_scale(study, blocks, design_matrix, outcomes, offsets)
        return self._fit_fixed_scale(
            study, blocks, design_matrix, outcomes, offsets, self._scales()
        )

    def _joint_design(
        self,
        study: Study,
        blocks: GroupBlocks,
        design_matrix: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        offsets: NDArray[np.float64] | None,
        scales: NDArray[np.float64],
    ) -> PenalisedDesign:
        columns = self._columns()
        n_parameters = len(self.parameter_names)
        box = expand_group_box(self.model.coordinate_box(study), blocks, columns)
        starts = tuple(
            np.concatenate([point, np.zeros(blocks.n_groups * len(columns), dtype=np.float64)])
            for point in self.model.initial_points(study)
        )
        return PenalisedDesign(
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
            box=box,
            initial_points=starts,
            expansion=GroupExpansion(
                n_population=n_parameters, n_groups=blocks.n_groups, columns=columns
            ),
            derived_estimates=_effective_parameters(n_parameters, columns, blocks.n_groups),
        )

    def _fit_fixed_scale(
        self,
        study: Study,
        blocks: GroupBlocks,
        design_matrix: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        offsets: NDArray[np.float64] | None,
        scales: NDArray[np.float64],
        *,
        block_scales: NDArray[np.float64] | None = None,
        estimation: Mapping[str, Any] | None = None,
    ) -> HierarchicalFitResult:
        columns = self._columns()
        width = len(columns)
        n_parameters = len(self.parameter_names)
        joint_fit = self.model.fit_penalised(
            self._joint_design(study, blocks, design_matrix, outcomes, offsets, scales),
            model_name=self.model_name,
            model_signature=self.signature,
        )
        population = joint_fit.estimates[:n_parameters]
        deviations = joint_fit.estimates[n_parameters:].reshape(blocks.n_groups, width)
        group_standard_errors = joint_fit.standard_errors[n_parameters:].reshape(
            blocks.n_groups, width
        )
        reported = (
            self._block_scales(self.scale_parameters) if block_scales is None else block_scales
        )
        return HierarchicalFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=population,
            standard_errors=joint_fit.standard_errors[:n_parameters],
            covariance=joint_fit.covariance[:n_parameters, :n_parameters],
            n_observations=len(study),
            diagnostics=joint_fit.diagnostics,
            grouping=self.grouping,
            groups=blocks.labels,
            varying_parameters=self.varying_parameters,
            group_deviations=deviations,
            group_standard_errors=group_standard_errors,
            scales=np.asarray(reported, dtype=np.float64),
            scale_parameters=self.scale_parameters,
            **dict(estimation or {}),
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
        blocks_of_scales = self._scale_blocks()
        reported = np.asarray([float(scales[block[0]]) for block in blocks_of_scales])
        reported_errors = np.asarray(
            [
                float(scale_standard_error) * float(scales[block[0]]) / scale
                for block in blocks_of_scales
            ]
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
            scales=reported,
            scale_parameters=self.scale_parameters,
            scale_standard_errors=reported_errors,
            scale_bounds=self.scale_bounds,
            scale_estimated=True,
            scales_at_boundary=np.full(len(reported), at_boundary, dtype=np.bool_),
            scale_estimation_policy="laplace-profile",
            scale_uncertainty_policy="laplace-profile-hessian",
        )

    def _fit_estimated_scales_em(
        self,
        study: Study,
        blocks: GroupBlocks,
        design_matrix: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        offsets: NDArray[np.float64] | None,
    ) -> HierarchicalFitResult:
        """Alternate a joint MAP fit with a closed-form M-step, one scale per named block.

        The E-step is the joint fit the fixed-scale path already runs: its group deviations
        are the conditional modes and its
        :attr:`~behavio.contracts.compose.PenalisedFitResult.conditional_group_covariances`
        are the conditional variances, which together are the second moment the M-step
        needs. Nothing here is family-specific -- the prior enters only through
        :meth:`~behavio.contracts.compose.PenalisedLinearEstimator.group_penalty`, and the
        one thing this estimator assumes about it is that a scale enters as an isotropic
        ridge on its own block, which is what makes ``d precision / d log scale`` equal to
        ``-2 / scale**2`` there and zero everywhere else.
        """

        scale_blocks = self._scale_blocks()
        log_bounds = (float(np.log(self.scale_bounds[0])), float(np.log(self.scale_bounds[1])))
        scales = self._block_scales(self.scale_parameters)

        def update(
            block_scales: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            joint = self.model.fit_penalised(
                self._joint_design(
                    study,
                    blocks,
                    design_matrix,
                    outcomes,
                    offsets,
                    self._broadcast_scales(block_scales),
                ),
                model_name=self.model_name,
                model_signature=self.signature,
            )
            n_parameters = len(self.parameter_names)
            width = len(self._columns())
            deviations = joint.estimates[n_parameters:].reshape(blocks.n_groups, width)
            covariances = getattr(joint, "conditional_group_covariances", None)
            if covariances is None:
                raise ModelDataError(
                    f"{type(self.model).__name__} does not report conditional group "
                    "covariances, which scale_estimator='laplace-em' needs"
                )
            updated = np.empty_like(block_scales)
            for index, block in enumerate(scale_blocks):
                found = minimize_scalar(
                    _expected_group_prior_objective,
                    args=(deviations[:, block], covariances[:, block][:, :, block], self, block),
                    bounds=log_bounds,
                    method="bounded",
                    options={"xatol": self.scale_tolerance / 10.0},
                )
                updated[index] = float(np.exp(found.x))
            return updated, deviations, np.asarray(covariances, dtype=np.float64)

        iterations = 0
        converged = False
        deviations = np.zeros((blocks.n_groups, len(self._columns())), dtype=np.float64)
        covariances = np.zeros((blocks.n_groups, deviations.shape[1], deviations.shape[1]))
        for iteration in range(1, self.scale_max_iterations + 1):
            iterations = iteration
            updated, deviations, covariances = update(scales)
            change = float(np.max(np.abs(np.log(updated) - np.log(scales))))
            scales = updated
            if change <= self.scale_tolerance:
                converged = True
                break
        at_boundary = _scales_at_bounds(
            scales, self.scale_bounds, tolerance=max(self.scale_tolerance, 1e-4)
        )

        final = self._fit_fixed_scale(
            study,
            blocks,
            design_matrix,
            outcomes,
            offsets,
            self._broadcast_scales(scales),
        )
        deviations = np.asarray(final.group_deviations, dtype=np.float64)
        covariances = self._conditional_covariances(
            study, blocks, design_matrix, outcomes, offsets, scales
        )
        local = np.asarray(
            [
                _scale_curvature_standard_error(
                    float(scale),
                    deviations[:, block],
                    covariances[:, block][:, :, block],
                    self,
                    block,
                )
                for scale, block in zip(scales, scale_blocks, strict=True)
            ]
        )
        local_log_covariance = np.diag((local / scales) ** 2)
        rate_matrix: NDArray[np.float64] | None = None
        if self.scale_uncertainty == "supplemented":
            if not converged:
                raise ModelDataError("supplemented scale uncertainty requires EM convergence")
            rate_matrix = _em_rate_matrix(scales, update, log_bounds, self.scale_tolerance)
            if np.max(np.abs(np.linalg.eigvals(rate_matrix))) >= 1.0:
                raise ModelDataError(
                    "supplemented scale uncertainty requires a stable EM rate matrix"
                )
            complete = np.linalg.pinv(local_log_covariance, hermitian=True)
            information = (np.eye(len(scales)) - rate_matrix) @ complete
            information = 0.5 * (information + information.T)
            if np.min(np.linalg.eigvalsh(information)) <= 0:
                raise ModelDataError("supplemented scale information is not positive definite")
            log_covariance = np.linalg.pinv(information, hermitian=True)
        elif self.scale_uncertainty == "observed":
            information = _louis_scale_information(
                scales, deviations, covariances, self, scale_blocks
            )
            if np.min(np.linalg.eigvalsh(information)) <= 0:
                raise ModelDataError("observed scale information is not positive definite")
            log_covariance = np.linalg.pinv(information, hermitian=True)
        else:
            log_covariance = local_log_covariance
        covariance = np.diag(scales) @ log_covariance @ np.diag(scales)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return self._fit_fixed_scale(
            study,
            blocks,
            design_matrix,
            outcomes,
            offsets,
            self._broadcast_scales(scales),
            block_scales=scales,
            estimation={
                "scale_standard_errors": standard_errors,
                "scale_local_standard_errors": local,
                "scale_covariance": covariance,
                "scale_em_rate_matrix": rate_matrix,
                "scale_bounds": self.scale_bounds,
                "scale_estimated": True,
                "scales_at_boundary": at_boundary,
                "scale_estimation_iterations": iterations,
                "scale_estimation_converged": converged,
                "scale_estimation_policy": "laplace-em",
                "scale_uncertainty_policy": SCALE_UNCERTAINTY_POLICIES[self.scale_uncertainty],
            },
        )

    def _conditional_covariances(
        self,
        study: Study,
        blocks: GroupBlocks,
        design_matrix: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        offsets: NDArray[np.float64] | None,
        scales: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        joint = self.model.fit_penalised(
            self._joint_design(
                study, blocks, design_matrix, outcomes, offsets, self._broadcast_scales(scales)
            ),
            model_name=self.model_name,
            model_signature=self.signature,
        )
        covariances = getattr(joint, "conditional_group_covariances", None)
        if covariances is None:
            raise ModelDataError(
                f"{type(self.model).__name__} does not report conditional group covariances"
            )
        return np.asarray(covariances, dtype=np.float64)

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
        coefficients = self._row_parameters(study, hierarchical_fit)
        return self.model.likelihood.prediction(
            self._row_predictor(study, coefficients), mode=prediction_mode
        )

    def _row_predictor(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        design_matrix = self.model.design_matrix(study)
        predictor = (
            np.einsum("ij,ij->i", design_matrix, coefficients)
            if design_matrix.ndim == 2
            else np.einsum("rcp,rp->rc", design_matrix, coefficients, optimize=True)
        )
        offsets = self.model.predictor_offsets(study)
        if offsets is not None:
            predictor = predictor + offsets
        return predictor

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score observations under fitted or unseen-group plug-in parameters.

        The likelihood is scored on the model's own linear predictor rather than on the one
        a :class:`~behavio.contracts.estimator.Prediction` carries. Those are the same array
        for a family whose prediction *is* its predictor, and they are not for a
        drift-diffusion family, whose four predictor cells produce one choice probability.
        """

        self._prediction_mode(mode)
        hierarchical_fit = self._validated_fit(fit)
        coefficients = self._row_parameters(study, hierarchical_fit)
        return self.model.likelihood.pointwise_log_prob(
            self._row_predictor(study, coefficients), self.model.outcomes(study)
        )

    def predict_new_groups(
        self,
        study: Study,
        fit: FitResult,
        *,
        n_draws: int,
        seed: int,
    ) -> UnseenGroupPrediction:
        """Integrate the fitted random effect over entirely unseen groups."""

        hierarchical_fit = self._validated_fit(fit)
        if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 2:
            raise ValueError("n_draws must be an integer of at least two")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("predictive seed must be a non-negative integer")
        blocks = group_blocks(study, self.grouping)
        groups = tuple(_scalar(label) for label in blocks.labels)
        if set(groups) & set(hierarchical_fit.groups):
            raise ValueError("predict_new_groups requires entirely unseen groups")
        columns = self._columns()
        scales = self._broadcast_scales(np.asarray(hierarchical_fit.scales, dtype=np.float64))
        outcomes = self.model.outcomes(study)
        generator = np.random.default_rng(seed)
        n_rows = len(study)
        probabilities = np.empty((n_draws, n_rows), dtype=np.float64)
        log_probabilities = np.empty_like(probabilities)
        deviations = np.empty((n_draws, blocks.n_groups, len(columns)), dtype=np.float64)
        for draw in range(n_draws):
            deviations[draw] = self.model.draw_group_deviations(
                columns, scales, groups=blocks.n_groups, generator=generator
            )
            rows = np.tile(hierarchical_fit.estimates, (n_rows, 1))
            rows[:, columns] += deviations[draw][blocks.row_block]
            predictor = self._row_predictor(study, rows)
            prediction = self.model.likelihood.prediction(predictor, mode=PredictionMode.FILTERED)
            probabilities[draw] = np.asarray(prediction.probability, dtype=np.float64)
            log_probabilities[draw] = self.model.likelihood.pointwise_log_prob(predictor, outcomes)
        log_draws = np.log(float(n_draws))
        joint = np.empty(blocks.n_groups, dtype=np.float64)
        effective_draws = np.empty(blocks.n_groups, dtype=np.float64)
        monte_carlo_error = np.empty(blocks.n_groups, dtype=np.float64)
        for group in range(blocks.n_groups):
            log_weights = np.sum(log_probabilities[:, blocks.row_block == group], axis=1)
            log_weight_sum = float(logsumexp(log_weights))
            joint[group] = log_weight_sum - log_draws
            normalized = np.exp(log_weights - log_weight_sum)
            effective_draws[group] = 1.0 / float(np.sum(normalized**2))
            shifted = np.exp(log_weights - float(np.max(log_weights)))
            monte_carlo_error[group] = float(
                np.std(shifted, ddof=1) / np.sqrt(n_draws) / np.mean(shifted)
            )
        return UnseenGroupPrediction(
            groups=groups,
            row_group_indices=blocks.row_block,
            probability=np.mean(probabilities, axis=0),
            draw_probabilities=probabilities,
            draw_pointwise_log_probability=log_probabilities,
            pointwise_marginal_log_probability=logsumexp(log_probabilities, axis=0) - log_draws,
            group_joint_log_probability=joint,
            group_effective_draws=effective_draws,
            group_log_probability_mcse=monte_carlo_error,
            random_effect_draws=deviations,
            varying_parameters=self.varying_parameters,
            scales=scales,
            seed=seed,
        )

    # -- reading paths back --------------------------------------------------------------

    def coefficient_trajectory(
        self, fit: FitResult, *, times: Sequence[float] | None = None
    ) -> Any:
        """Evaluate the fitted **population** paths of a wrapped smooth model."""

        hierarchical_fit = self._validated_fit(fit)
        return self._smooth_model().trajectory_from_knots(hierarchical_fit.estimates, times=times)

    def group_trajectory(
        self, fit: FitResult, group: Any, *, times: Sequence[float] | None = None
    ) -> Any:
        """Evaluate one group's fitted paths, or the population plug-in for an unseen one."""

        hierarchical_fit = self._validated_fit(fit)
        values = np.array(hierarchical_fit.estimates, dtype=np.float64)
        key = _scalar(group)
        if key in hierarchical_fit.groups:
            position = hierarchical_fit.groups.index(key)
            values[self._columns()] += hierarchical_fit.group_deviations[position]
        return self._smooth_model().trajectory_from_knots(values, times=times)

    def parameters_from_paths(self, paths: Mapping[str, Any]) -> Mapping[str, float]:
        """Pack population paths, delegating to the wrapped smooth model that owns them."""

        return self._smooth_model().parameters_from_paths(paths)

    def knot_grid(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one coordinate vector as the wrapped smooth model's knot grid."""

        return self._smooth_model().knot_grid(values)

    def _smooth_model(self) -> Any:
        model = self.model
        if not hasattr(model, "trajectory_from_knots"):
            raise TypeError(
                f"{type(model).__name__} has no coefficient paths; "
                "trajectories exist for hierarchical(smooth(model))"
            )
        return model

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

    def _scale_blocks(self) -> tuple[NDArray[np.intp], ...]:
        """Positions within one group's deviation vector that share each declared scale."""

        varying = self.varying_parameters
        position = {name: index for index, name in enumerate(varying)}
        blocks: list[NDArray[np.intp]] = []
        for name in self.scale_parameters:
            reported = self.model.group_parameter_expansion(name)
            blocks.append(
                np.asarray(
                    [position[entry] for entry in reported if entry in position], dtype=np.intp
                )
            )
        if sum(len(block) for block in blocks) != len(varying):
            raise ValueError("scale parameters must partition the varying parameters")
        return tuple(blocks)

    def _block_scales(self, blocks: Sequence[str]) -> NDArray[np.float64]:
        """One scale per declared block, read off the per-parameter declaration."""

        by_name = dict(zip(self.effects.parameters, self.effects.scales.tolist(), strict=True))
        values = []
        for name in blocks:
            reported = self.model.group_parameter_expansion(name)
            values.append(float(by_name[next(entry for entry in reported if entry in by_name)]))
        return np.asarray(values, dtype=np.float64)

    def _broadcast_scales(self, block_scales: NDArray[np.float64]) -> NDArray[np.float64]:
        """Spread one scale per block across the parameters of that block, in model order."""

        scales = np.empty(len(self.varying_parameters), dtype=np.float64)
        for value, block in zip(block_scales, self._scale_blocks(), strict=True):
            scales[block] = float(value)
        return scales

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
            values = np.asarray(normalized[_scalar(label)], dtype=np.float64).ravel()
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
            linear_predictor(block_design, population + offset, block_offsets), block_outcomes
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


# --------------------------------------------------------------------------------------
# The Laplace-EM estimate of one scale per named block
# --------------------------------------------------------------------------------------


def _block_precision(
    model: HierarchicalModel, block: NDArray[np.intp], scale: float
) -> NDArray[np.float64]:
    """The deviation prior's precision on one scale block, at one scale."""

    columns = model._columns()[block]
    return model.model.group_penalty(columns, np.full(len(block), float(scale)))


def _expected_group_prior_objective(
    log_scale: float,
    deviations: NDArray[np.float64],
    covariances: NDArray[np.float64],
    model: HierarchicalModel,
    block: NDArray[np.intp],
) -> float:
    """Expected normalized Gaussian-prior loss for one block's scale.

    This is the M-step: the complete-data prior term with the deviations replaced by their
    conditional second moment, which is the conditional mode outer itself plus the
    conditional covariance.
    """

    scale = float(np.exp(log_scale))
    precision = _block_precision(model, block, scale)
    sign, log_determinant = np.linalg.slogdet(precision)
    if sign <= 0 or not np.isfinite(log_determinant):
        return float(np.finfo(np.float64).max / 1e100)
    quadratic = 0.0
    for deviation, covariance in zip(deviations, covariances, strict=True):
        second_moment = covariance + np.outer(deviation, deviation)
        quadratic += float(np.trace(precision @ second_moment))
    return 0.5 * (quadratic - len(deviations) * float(log_determinant))


def _scale_curvature_standard_error(
    scale: float,
    deviations: NDArray[np.float64],
    covariances: NDArray[np.float64],
    model: HierarchicalModel,
    block: NDArray[np.intp],
) -> float:
    """Return local expected-prior curvature uncertainty on the natural scale."""

    center = float(np.log(scale))
    step = 1e-3

    def objective(value: float) -> float:
        return _expected_group_prior_objective(value, deviations, covariances, model, block)

    curvature = (
        objective(center + step) - 2.0 * objective(center) + objective(center - step)
    ) / step**2
    return float(scale / np.sqrt(max(float(curvature), np.finfo(np.float64).eps)))


def _louis_scale_information(
    scales: NDArray[np.float64],
    deviations: NDArray[np.float64],
    covariances: NDArray[np.float64],
    model: HierarchicalModel,
    blocks: tuple[NDArray[np.intp], ...],
) -> NDArray[np.float64]:
    """Return observed log-scale information as complete minus missing information.

    The expected-prior curvature the M-step minimizes is complete-data information, which
    is never smaller than the observed information the marginal likelihood carries. Louis'
    identity subtracts the conditional variance of the complete-data score, which for a
    Gaussian deviation prior is a closed form in the conditional means and covariances.
    """

    n_parameters = len(scales)
    n_groups = len(covariances)
    precisions = np.asarray(scales, dtype=np.float64) ** -2.0
    complete = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    for index, (precision, block) in enumerate(zip(precisions, blocks, strict=True)):
        prior_covariance = np.linalg.inv(_block_precision(model, block, float(scales[index])))
        second_moment = float(np.sum(deviations[:, block] ** 2)) + sum(
            float(np.trace(covariance[np.ix_(block, block)])) for covariance in covariances
        )
        complete[index, index] = 2.0 * precision * (
            second_moment - n_groups * float(np.trace(prior_covariance))
        ) + 2.0 * n_groups * precision**2 * float(np.sum(prior_covariance * prior_covariance))
    missing = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    for row in range(n_parameters):
        for column in range(row + 1):
            total = 0.0
            for group, covariance in enumerate(covariances):
                cross = covariance[np.ix_(blocks[row], blocks[column])]
                left = deviations[group, blocks[row]]
                right = deviations[group, blocks[column]]
                total += 2.0 * float(np.sum(cross * cross)) + 4.0 * float(left @ cross @ right)
            value = precisions[row] * precisions[column] * total
            missing[row, column] = missing[column, row] = value
    information = complete - missing
    return 0.5 * (information + information.T)


def _em_rate_matrix(
    scales: NDArray[np.float64],
    update: Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], Any, Any]],
    log_bounds: tuple[float, float],
    tolerance: float,
) -> NDArray[np.float64]:
    """Difference the EM map itself, which is the supplemented correction's rate matrix."""

    center = np.log(scales)
    step = max(0.02, tolerance / 2.0)
    rate = np.empty((len(scales), len(scales)), dtype=np.float64)
    for index in range(len(scales)):
        lower_point = center.copy()
        upper_point = center.copy()
        lower_point[index] = max(lower_point[index] - step, log_bounds[0])
        upper_point[index] = min(upper_point[index] + step, log_bounds[1])
        lower_update, _, _ = update(np.exp(lower_point))
        upper_update, _, _ = update(np.exp(upper_point))
        rate[:, index] = (np.log(upper_update) - np.log(lower_update)) / (
            upper_point[index] - lower_point[index]
        )
    return rate


def _scales_at_bounds(
    scales: NDArray[np.float64], bounds: tuple[float, float], *, tolerance: float
) -> NDArray[np.bool_]:
    """Mark variance components within a log-scale tolerance of either bound."""

    lower, upper = bounds
    log_scales = np.log(scales)
    return np.asarray(
        (np.abs(log_scales - np.log(lower)) <= tolerance)
        | (np.abs(log_scales - np.log(upper)) <= tolerance),
        dtype=np.bool_,
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


def _optional_array(values: Any, dtype: Any) -> Any:
    return None if values is None else protected_array(values, dtype=dtype)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
