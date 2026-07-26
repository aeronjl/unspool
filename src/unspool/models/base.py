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
class BehaviourModel(Protocol):
    """Minimum generative, fitting, prediction, and scoring model contract."""

    @property
    def model_name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]: ...

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study: ...

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


def _protected_array(
    values: Sequence[Any] | NDArray[Any], *, dtype: np.dtype[Any] | type[Any]
) -> NDArray[Any]:
    owner = np.array(values, dtype=dtype, copy=True)
    owner.setflags(write=False)
    view = owner.view()
    view.setflags(write=False)
    return view
