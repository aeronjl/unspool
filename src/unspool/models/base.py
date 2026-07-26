"""Shared contracts for behavioural models and their outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from unspool.study import Study

if TYPE_CHECKING:
    from unspool.diagnostics import FitAudit, FitAuditPolicy


class PredictionMode(StrEnum):
    """The information set used to construct a prediction."""

    FILTERED = "filtered"
    SMOOTHED = "smoothed"


class ModelDataError(ValueError):
    """Raised when a study cannot be interpreted by a model."""


class UnsupportedPredictionMode(ValueError):
    """Raised when a model cannot provide the requested prediction mode."""


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Validated description of what one behavioural estimator exposes.

    ``scored_columns`` names the complete observation used by each pointwise likelihood.
    It is deliberately distinct from the binary choice probability returned by
    :class:`Prediction`: a future reaction-time model may predict choice while scoring the
    joint choice and response-time observation.
    """

    scored_columns: tuple[str, ...]
    prediction_modes: tuple[PredictionMode, ...]
    can_simulate: bool
    can_recover_parameters: bool

    def __post_init__(self) -> None:
        if isinstance(self.scored_columns, str):
            raise ValueError("scored_columns must be a tuple of column names")
        columns = tuple(self.scored_columns)
        modes = tuple(PredictionMode(mode) for mode in self.prediction_modes)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("scored_columns must be non-empty and unique")
        if any(not isinstance(column, str) or not column for column in columns):
            raise ValueError("scored_columns must contain non-empty strings")
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("prediction_modes must be non-empty and unique")
        if not isinstance(self.can_simulate, bool) or not isinstance(
            self.can_recover_parameters, bool
        ):
            raise ValueError("capability flags must be boolean")
        if self.can_recover_parameters and not self.can_simulate:
            raise ValueError("parameter recovery requires simulation")
        object.__setattr__(self, "scored_columns", columns)
        object.__setattr__(self, "prediction_modes", modes)


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    """Optimizer and numerical diagnostics that remain attached to a fit."""

    converged: bool
    optimizer: str
    status: int
    message: str
    n_iterations: int
    objective: float
    gradient_norm: float
    hessian_condition: float
    boundary_estimate: bool


@dataclass(frozen=True, slots=True)
class FitResult:
    """Immutable parameter estimates and diagnostics for one fitted model."""

    model_name: str
    model_signature: str
    parameter_names: tuple[str, ...]
    estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    covariance: NDArray[np.float64]
    n_observations: int
    diagnostics: FitDiagnostics

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("parameter_names must be non-empty and unique")
        estimates = _protected_array(self.estimates, dtype=np.float64)
        standard_errors = _protected_array(self.standard_errors, dtype=np.float64)
        covariance = _protected_array(self.covariance, dtype=np.float64)
        if estimates.ndim != 1 or estimates.shape != (len(names),):
            raise ValueError("estimates must contain one value per parameter")
        if standard_errors.shape != estimates.shape:
            raise ValueError("standard_errors must contain one value per parameter")
        if covariance.shape != (len(names), len(names)):
            raise ValueError("covariance must be square with one row per parameter")
        if self.n_observations < 1:
            raise ValueError("n_observations must be positive")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "covariance", covariance)

    @property
    def parameters(self) -> Mapping[str, float]:
        """Estimated parameters keyed by their stable public names."""

        return MappingProxyType(
            dict(zip(self.parameter_names, self.estimates.tolist(), strict=True))
        )

    @property
    def standard_error_map(self) -> Mapping[str, float]:
        """Approximate standard errors keyed by parameter name."""

        return MappingProxyType(
            dict(zip(self.parameter_names, self.standard_errors.tolist(), strict=True))
        )

    def audit(self, *, policy: FitAuditPolicy | None = None) -> FitAudit:
        """Normalize all available diagnostics without removing their raw evidence."""

        from unspool.diagnostics import audit_fit

        return audit_fit(self, policy=policy)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Point predictions with an explicit temporal information mode."""

    probability: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]
    mode: PredictionMode

    def __post_init__(self) -> None:
        probability = _protected_array(self.probability, dtype=np.float64)
        linear_predictor = _protected_array(self.linear_predictor, dtype=np.float64)
        mode = PredictionMode(self.mode)
        if probability.ndim != 1 or linear_predictor.shape != probability.shape:
            raise ValueError("prediction arrays must be one-dimensional and equally sized")
        if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
            raise ValueError("probabilities must be finite values between zero and one")
        if not np.all(np.isfinite(linear_predictor)):
            raise ValueError("linear predictors must be finite")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "linear_predictor", linear_predictor)
        object.__setattr__(self, "mode", mode)


@runtime_checkable
class BehaviourEstimator(Protocol):
    """Minimum fitting, prediction, and pointwise-scoring contract."""

    @property
    def model_name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    @property
    def scored_columns(self) -> tuple[str, ...]: ...

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]: ...

    def fit(self, study: Study) -> FitResult: ...

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction: ...

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]: ...


@runtime_checkable
class GenerativeBehaviourModel(BehaviourEstimator, Protocol):
    """An estimator with named parameters and a matching simulator."""

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study: ...


@runtime_checkable
class BehaviourModel(GenerativeBehaviourModel, Protocol):
    """Backward-compatible name for Unspool's full generative model contract."""


def model_capabilities(model: BehaviourEstimator) -> ModelCapabilities:
    """Validate and return the capabilities advertised by an estimator.

    Runtime-checkable protocols establish method presence. This function additionally
    validates the semantic metadata on which evaluation and recovery rely.
    """

    if not isinstance(model, BehaviourEstimator):
        raise TypeError("model must satisfy the BehaviourEstimator contract")
    if not isinstance(model.model_name, str) or not model.model_name:
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(model.signature, str) or not model.signature:
        raise ValueError("signature must be a non-empty string")
    generative = isinstance(model, GenerativeBehaviourModel)
    if generative:
        if isinstance(model.parameter_names, str):
            raise ValueError("parameter_names must be a tuple of names")
        names = tuple(model.parameter_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("parameter_names must be non-empty and unique")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("parameter_names must contain non-empty strings")
    return ModelCapabilities(
        scored_columns=tuple(model.scored_columns),
        prediction_modes=tuple(model.supported_prediction_modes),
        can_simulate=generative,
        can_recover_parameters=generative,
    )


def _protected_array(
    values: Sequence[Any] | NDArray[Any], *, dtype: np.dtype[Any] | type[Any]
) -> NDArray[Any]:
    owner = np.array(values, dtype=dtype, copy=True)
    owner.setflags(write=False)
    view = owner.view()
    view.setflags(write=False)
    return view
