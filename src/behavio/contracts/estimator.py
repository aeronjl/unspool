"""The frequentist estimator contract: fits, predictions, and the model protocols.

Every name here used to live in ``behavio.models.base``, which now re-exports them.

This module also owns the inversion point that breaks the old
``behavio.diagnostics`` <-> ``behavio.models.base`` cycle. ``FitResult.audit()`` is public
(see ``README.md``), but ``behavio.contracts`` must stay a leaf, so it cannot import
``behavio.diagnostics``. Instead this module declares the :class:`FitAuditor` protocol and
a single-slot registry; ``behavio.diagnostics`` registers ``audit_fit`` at import time and
``FitResult.audit()`` dispatches through the registry. There is therefore no module-level
cycle and no function-local import whose only purpose is to dodge one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.audit import FitAudit, FitAuditPolicy, FitDiagnostics
from behavio.study import Study


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
    It is deliberately distinct from the choice probabilities returned by
    :class:`Prediction` or :class:`CategoricalPrediction`: a reaction-time model may
    predict choice while scoring the joint choice and response-time observation.
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
        estimates = protected_array(self.estimates, dtype=np.float64)
        standard_errors = protected_array(self.standard_errors, dtype=np.float64)
        covariance = protected_array(self.covariance, dtype=np.float64)
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

        return fit_auditor()(self, policy=policy)


@runtime_checkable
class FitAuditor(Protocol):
    """Callable that normalizes one :class:`FitResult` into a :class:`FitAudit`."""

    def __call__(self, fit: FitResult, *, policy: FitAuditPolicy | None = None) -> FitAudit: ...


_FIT_AUDITOR: FitAuditor | None = None


def register_fit_auditor(auditor: FitAuditor) -> None:
    """Install the implementation backing :meth:`FitResult.audit`.

    ``behavio.diagnostics`` calls this at import time. Importing any Behavio submodule
    executes ``behavio/__init__.py`` first, which imports ``behavio.diagnostics``, so the
    auditor is always installed before user code can reach a :class:`FitResult`.
    """

    if not callable(auditor):
        raise TypeError("auditor must be callable")
    global _FIT_AUDITOR
    _FIT_AUDITOR = auditor


def fit_auditor() -> FitAuditor:
    """Return the registered fit auditor."""

    if _FIT_AUDITOR is None:
        raise RuntimeError(
            "no fit auditor is registered; import behavio.diagnostics to install the default"
        )
    return _FIT_AUDITOR


@dataclass(frozen=True, slots=True)
class Prediction:
    """Point predictions with an explicit temporal information mode."""

    probability: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]
    mode: PredictionMode

    def __post_init__(self) -> None:
        probability = protected_array(self.probability, dtype=np.float64)
        linear_predictor = protected_array(self.linear_predictor, dtype=np.float64)
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

    @property
    def n_observations(self) -> int:
        """Number of trial rows represented by the prediction."""

        return len(self.probability)

    def take(self, indices: Sequence[int] | NDArray[np.integer[Any]]) -> Prediction:
        """Return a protected row subset without changing prediction semantics."""

        return Prediction(
            probability=self.probability[indices],
            linear_predictor=self.linear_predictor[indices],
            mode=self.mode,
        )


@dataclass(frozen=True, slots=True)
class CategoricalPrediction:
    """Probabilities on one explicit categorical outcome coordinate.

    Rows index trials and columns index ``categories``. Impossible actions may have
    probability zero and a ``-inf`` linear predictor, which is required for tasks with
    trial-specific option availability.
    """

    probability: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]
    categories: tuple[Any, ...]
    mode: PredictionMode

    def __post_init__(self) -> None:
        probability = protected_array(self.probability, dtype=np.float64)
        linear_predictor = protected_array(self.linear_predictor, dtype=np.float64)
        categories = tuple(_prediction_category(value) for value in self.categories)
        mode = PredictionMode(self.mode)
        if probability.ndim != 2 or probability.shape[1] < 2:
            raise ValueError("categorical probabilities must have at least two columns")
        if linear_predictor.shape != probability.shape:
            raise ValueError("categorical predictors and probabilities must be equally sized")
        if len(categories) != probability.shape[1]:
            raise ValueError("categories must name every probability column")
        keys = tuple((type(value), value) for value in categories)
        try:
            unique = set(keys)
        except TypeError:
            raise ValueError("prediction categories must be scalar and hashable") from None
        if len(unique) != len(categories):
            raise ValueError("prediction categories must be unique")
        if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
            raise ValueError("categorical probabilities must be finite values in [0, 1]")
        if not np.allclose(np.sum(probability, axis=1), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("categorical probability rows must sum to one")
        if np.any(np.isnan(linear_predictor)) or np.any(np.isposinf(linear_predictor)):
            raise ValueError("categorical predictors may contain finite values or -inf")
        if np.any(np.all(np.isneginf(linear_predictor), axis=1)):
            raise ValueError("every prediction row must contain an available category")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "linear_predictor", linear_predictor)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "mode", mode)

    @property
    def n_observations(self) -> int:
        """Number of trial rows represented by the prediction."""

        return self.probability.shape[0]

    def take(self, indices: Sequence[int] | NDArray[np.integer[Any]]) -> CategoricalPrediction:
        """Return a protected row subset on the same category coordinate."""

        return CategoricalPrediction(
            probability=self.probability[indices],
            linear_predictor=self.linear_predictor[indices],
            categories=self.categories,
            mode=self.mode,
        )


ModelPrediction = Prediction | CategoricalPrediction


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
    ) -> ModelPrediction: ...

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]: ...


@runtime_checkable
class CategoricalBehaviourEstimator(BehaviourEstimator, Protocol):
    """An estimator whose scored choice is represented by stable category codes."""

    @property
    def categories(self) -> tuple[Any, ...]: ...

    def outcome_codes(self, study: Study) -> NDArray[np.int64]: ...


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
    """Backward-compatible name for Behavio's full generative model contract."""


def model_capabilities(model: BehaviourEstimator) -> ModelCapabilities:
    """Validate and return the capabilities advertised by an estimator.

    Runtime-checkable protocols establish method presence. This function additionally
    validates the semantic metadata on which evaluation and recovery rely.
    """

    if not isinstance(model, BehaviourEstimator):
        raise TypeError("model must satisfy the BehaviourEstimator contract")
    validate_model_identity(model)
    generative = isinstance(model, GenerativeBehaviourModel)
    if generative:
        validate_parameter_names(model.parameter_names)
    return ModelCapabilities(
        scored_columns=tuple(model.scored_columns),
        prediction_modes=tuple(model.supported_prediction_modes),
        can_simulate=generative,
        can_recover_parameters=generative,
    )


def validate_model_identity(model: Any) -> None:
    """Check that a model advertises a non-empty name and configuration signature."""

    if not isinstance(model.model_name, str) or not model.model_name:
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(model.signature, str) or not model.signature:
        raise ValueError("signature must be a non-empty string")


def validate_parameter_names(names: Any) -> tuple[str, ...]:
    """Check and return a non-empty tuple of unique, non-empty parameter names."""

    if isinstance(names, str):
        raise ValueError("parameter_names must be a tuple of names")
    parameter_names = tuple(names)
    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must be non-empty and unique")
    if any(not isinstance(name, str) or not name for name in parameter_names):
        raise ValueError("parameter_names must contain non-empty strings")
    return parameter_names


def _prediction_category(value: Any) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    if scalar is None or isinstance(scalar, (str, bool, int)):
        return scalar
    if isinstance(scalar, float) and np.isfinite(scalar):
        return scalar
    raise ValueError(f"prediction category must be a finite scalar: {scalar!r}")
