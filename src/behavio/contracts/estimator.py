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
from dataclasses import dataclass, field
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

    ``required_task_columns`` names the predictive context the model consumes but does not
    score -- the stimulus a psychometric curve is a function of, the reward a learner
    updates on. Every column here must carry a declared task role, which
    :meth:`behavio.task.TaskSpec.validate_model` enforces. It defaults to ``()`` so that
    the *empty* declaration can be written by omission; the declaration itself is not
    optional, because :class:`BehaviourEstimator` requires it. A model that reads nothing
    but its scored column answers ``()``, which is an answer, not a refusal to answer.
    """

    scored_columns: tuple[str, ...]
    prediction_modes: tuple[PredictionMode, ...]
    can_simulate: bool
    can_recover_parameters: bool
    required_task_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.scored_columns, str):
            raise ValueError("scored_columns must be a tuple of column names")
        columns = tuple(self.scored_columns)
        modes = tuple(PredictionMode(mode) for mode in self.prediction_modes)
        required = validate_required_task_columns(self.required_task_columns)
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
        overlap = sorted(set(required) & set(columns))
        if overlap:
            raise ValueError(f"required_task_columns must not repeat a scored column: {overlap}")
        object.__setattr__(self, "scored_columns", columns)
        object.__setattr__(self, "prediction_modes", modes)
        object.__setattr__(self, "required_task_columns", required)


@dataclass(frozen=True, slots=True)
class DerivedQuantity:
    """One reportable function of a fit's estimates.

    ``parameters`` and ``standard_error_map`` are keyed on the *optimizer* coordinate,
    which is unconstrained by design and is therefore frequently not the quantity anyone
    publishes: a psychophysicist asks for a threshold, not a ``log_width``, and a
    metacognition experiment reports ``m_ratio``, which is not estimated at all. A derived
    quantity carries that number with whatever uncertainty the model can honestly attach
    to it.

    ``standard_error`` is normally a delta-method propagation of the estimated covariance
    and ``interval`` is normally formed on the coordinate that was estimated and then
    mapped back, so that it can never leave the quantity's admissible range. Both are
    optional: ``None`` means *this model does not claim an uncertainty for this quantity*,
    which is distinct from claiming a non-finite one. ``interval_level`` is required
    whenever ``interval`` is present, because an interval without its level is unreadable.
    """

    name: str
    value: float
    standard_error: float | None = None
    interval: tuple[float, float] | None = None
    interval_level: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a derived quantity needs a non-empty name")
        if not isinstance(self.description, str):
            raise ValueError("a derived quantity description must be a string")
        object.__setattr__(self, "value", float(self.value))
        if self.standard_error is not None:
            error = float(self.standard_error)
            if error < 0:
                raise ValueError("a derived standard error must be non-negative")
            object.__setattr__(self, "standard_error", error)
        if self.interval is None:
            if self.interval_level is not None:
                raise ValueError("interval_level requires an interval")
            return
        bounds = tuple(float(value) for value in self.interval)
        if len(bounds) != 2:
            raise ValueError("a derived interval must be a lower and an upper bound")
        if np.isfinite(bounds[0]) and np.isfinite(bounds[1]) and bounds[0] > bounds[1]:
            raise ValueError("derived interval bounds must be ordered")
        if self.interval_level is None or not 0 < float(self.interval_level) < 1:
            raise ValueError("an interval requires a level strictly between zero and one")
        object.__setattr__(self, "interval", bounds)
        object.__setattr__(self, "interval_level", float(self.interval_level))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-shaped record of the quantity and its declared uncertainty."""

        return {
            "name": self.name,
            "value": self.value,
            "standard_error": self.standard_error,
            "interval": None if self.interval is None else list(self.interval),
            "interval_level": self.interval_level,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class FitResult:
    """Immutable parameter estimates and diagnostics for one fitted model.

    ``derived`` is the optional, default-empty place for functions of the estimates that a
    scientist would publish but the optimizer never sees. It is keyword-only so that a
    model-specific subclass can keep adding required fields after it, and it defaults to
    ``()`` so that a model that declares nothing behaves exactly as it did before the
    field existed. A derived name may coincide with a parameter name -- a psychometric
    threshold is estimated directly under four of the five links -- in which case the two
    describe the same number on the same scale.
    """

    model_name: str
    model_signature: str
    parameter_names: tuple[str, ...]
    estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    covariance: NDArray[np.float64]
    n_observations: int
    diagnostics: FitDiagnostics
    derived: tuple[DerivedQuantity, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        derived = tuple(self.derived)
        if any(not isinstance(quantity, DerivedQuantity) for quantity in derived):
            raise ValueError("derived must contain DerivedQuantity values")
        derived_names = [quantity.name for quantity in derived]
        if len(set(derived_names)) != len(derived_names):
            raise ValueError("derived quantity names must be unique within one fit")
        object.__setattr__(self, "derived", derived)
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

    @property
    def derived_quantities(self) -> Mapping[str, DerivedQuantity]:
        """Every declared derived quantity keyed by its name, in declaration order."""

        return MappingProxyType({quantity.name: quantity for quantity in self.derived})

    @property
    def derived_values(self) -> Mapping[str, float]:
        """Point values of the declared derived quantities, keyed by name."""

        return MappingProxyType({quantity.name: quantity.value for quantity in self.derived})

    def derived_value(self, name: str) -> float:
        """Return one declared derived quantity's value or raise a readable error."""

        try:
            return self.derived_values[name]
        except KeyError:
            raise KeyError(
                f"{self.model_name!r} declares no derived quantity {name!r}; "
                f"available: {sorted(self.derived_values)}"
            ) from None

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

    Composite categories
    --------------------
    A category is either a scalar or a **tuple of scalars**, and a tuple category is how a
    model that scores a joint observation names its cells. Meta-d' scores the joint
    ``(response, confidence)`` outcome; before tuples were admitted it had to encode a
    cell as the string ``"no-3"``, which every consumer wanting the response margin had to
    parse. ``scored_columns=("response", "confidence")`` already said the observation was
    joint, so only the labels were stringly typed.

    ``category_factors`` names the tuple positions and is required exactly when the
    categories are tuples: an unlabelled tuple is no more readable than the string it
    replaces. Its names normally match the model's ``scored_columns``. With it,
    :meth:`marginal` marginalises over the other factors without anyone parsing a label.
    """

    probability: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]
    categories: tuple[Any, ...]
    mode: PredictionMode
    category_factors: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        probability = protected_array(self.probability, dtype=np.float64)
        linear_predictor = protected_array(self.linear_predictor, dtype=np.float64)
        categories = tuple(_prediction_category(value) for value in self.categories)
        factors = _category_factors(categories, self.category_factors)
        mode = PredictionMode(self.mode)
        if probability.ndim != 2 or probability.shape[1] < 2:
            raise ValueError("categorical probabilities must have at least two columns")
        if linear_predictor.shape != probability.shape:
            raise ValueError("categorical predictors and probabilities must be equally sized")
        if len(categories) != probability.shape[1]:
            raise ValueError("categories must name every probability column")
        keys = tuple(_category_key(value) for value in categories)
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
        object.__setattr__(self, "category_factors", factors)
        object.__setattr__(self, "mode", mode)

    @property
    def n_observations(self) -> int:
        """Number of trial rows represented by the prediction."""

        return self.probability.shape[0]

    @property
    def is_composite(self) -> bool:
        """Whether each category names a cell of a declared factorisation."""

        return self.category_factors is not None

    def factor_levels(self, factor: str) -> tuple[Any, ...]:
        """Return one declared factor's distinct levels in first-appearance order."""

        position = self._factor_position(factor)
        levels: list[Any] = []
        seen: set[Any] = set()
        for category in self.categories:
            level = category[position]
            key = _category_key(level)
            if key not in seen:
                seen.add(key)
                levels.append(level)
        return tuple(levels)

    def marginal(self, factor: str) -> CategoricalPrediction:
        """Sum out every factor but ``factor`` and return the marginal prediction.

        This is the operation a caller of a joint-scoring model actually wants -- "what
        does this model say about the response, ignoring confidence?" -- and it is exact,
        because the cells of one row partition that row's probability. The marginal's
        linear predictor is the log of the marginal probability, ``-inf`` for a level the
        row cannot reach; it is not a sum of the joint linear predictors, which are not
        additive in probability.
        """

        levels = self.factor_levels(factor)
        if len(levels) < 2:
            raise ValueError(
                f"factor {factor!r} has a single level, so its marginal is not a "
                "categorical prediction"
            )
        position = self._factor_position(factor)
        index = {_category_key(level): column for column, level in enumerate(levels)}
        probability = np.zeros((self.n_observations, len(levels)), dtype=np.float64)
        for column, category in enumerate(self.categories):
            probability[:, index[_category_key(category[position])]] += self.probability[:, column]
        with np.errstate(divide="ignore"):
            linear_predictor = np.log(probability)
        return CategoricalPrediction(
            probability=probability,
            linear_predictor=linear_predictor,
            categories=levels,
            mode=self.mode,
        )

    def take(self, indices: Sequence[int] | NDArray[np.integer[Any]]) -> CategoricalPrediction:
        """Return a protected row subset on the same category coordinate."""

        return CategoricalPrediction(
            probability=self.probability[indices],
            linear_predictor=self.linear_predictor[indices],
            categories=self.categories,
            mode=self.mode,
            category_factors=self.category_factors,
        )

    def _factor_position(self, factor: str) -> int:
        if self.category_factors is None:
            raise ValueError(
                "this prediction declares scalar categories, so it has no factors to "
                "marginalise over"
            )
        try:
            return self.category_factors.index(factor)
        except ValueError:
            raise KeyError(
                f"unknown category factor {factor!r}; declared: {list(self.category_factors)}"
            ) from None


ModelPrediction = Prediction | CategoricalPrediction


@runtime_checkable
class BehaviourEstimator(Protocol):
    """Minimum fitting, prediction, and pointwise-scoring contract.

    ``required_task_columns`` is a member of this protocol rather than of a side protocol.
    "What columns does this model need?" is the first question anyone asks of an estimator,
    and every estimator can answer it: a model whose likelihood reads nothing but its
    scored column answers ``()``. It used to live on a separate ``TaskColumnEstimator``
    only to avoid evicting models that had not yet written the declaration down, and that
    reason has expired.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    @property
    def scored_columns(self) -> tuple[str, ...]: ...

    @property
    def required_task_columns(self) -> tuple[str, ...]: ...

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
        required_task_columns=model_task_columns(model),
    )


def model_task_columns(model: Any) -> tuple[str, ...]:
    """Return the validated predictive context an estimator declares.

    The declaration is checked here rather than only at task-validation time, so a
    malformed one is a model defect that surfaces from :func:`model_capabilities`.

    A missing declaration still yields ``()`` rather than raising, because
    :class:`behavio.contracts.posterior.PosteriorBehaviourEstimator` does not yet require
    it and :func:`behavio.contracts.posterior.posterior_model_capabilities` routes through
    here. For the frequentist contract the tolerance is unreachable: an estimator without
    ``required_task_columns`` is not a :class:`BehaviourEstimator`, and
    :func:`model_capabilities` rejects it before reaching this call.
    """

    declared = getattr(model, "required_task_columns", None)
    if declared is None:
        return ()
    return validate_required_task_columns(declared)


def validate_required_task_columns(columns: Any) -> tuple[str, ...]:
    """Check and return a possibly empty tuple of unique, non-empty column names."""

    if isinstance(columns, str):
        raise ValueError("required_task_columns must be a tuple of column names")
    required = tuple(columns)
    if len(set(required)) != len(required):
        raise ValueError("required_task_columns must be unique")
    if any(not isinstance(column, str) or not column for column in required):
        raise ValueError("required_task_columns must contain non-empty strings")
    return required


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
    """Normalize one category label, which is a scalar or a tuple of scalars."""

    if isinstance(value, tuple):
        if not value:
            raise ValueError("a composite prediction category must name at least one factor")
        return tuple(_prediction_scalar(item) for item in value)
    return _prediction_scalar(value)


def _prediction_scalar(value: Any) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    if scalar is None or isinstance(scalar, (str, bool, int)):
        return scalar
    if isinstance(scalar, float) and np.isfinite(scalar):
        return scalar
    raise ValueError(f"prediction category must be a finite scalar: {scalar!r}")


def _category_key(value: Any) -> Any:
    """Return a hashable identity that keeps ``0``, ``0.0``, ``False`` and ``"0"`` apart.

    Recursing into tuples is what keeps the distinction inside a composite category, where
    ``(0, 1)`` and ``(False, 1)`` are equal and equally hashed as plain tuples.
    """

    if isinstance(value, tuple):
        return (tuple, tuple(_category_key(item) for item in value))
    return (type(value), value)


def _category_factors(categories: tuple[Any, ...], factors: Any) -> tuple[str, ...] | None:
    """Validate a declared factorisation against the categories it is meant to name."""

    composite = [isinstance(category, tuple) for category in categories]
    if any(composite) and not all(composite):
        raise ValueError("prediction categories must be either all scalar or all composite")
    if not any(composite):
        if factors is not None:
            raise ValueError("category_factors requires composite (tuple) categories")
        return None
    arities = {len(category) for category in categories}
    if len(arities) != 1:
        raise ValueError("composite prediction categories must all name the same factors")
    if factors is None:
        raise ValueError("composite prediction categories require declared category_factors")
    if isinstance(factors, str):
        raise ValueError("category_factors must be a tuple of factor names")
    names = tuple(factors)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("category_factors must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("category_factors must be unique")
    if len(names) != arities.pop():
        raise ValueError("category_factors must name every position of a composite category")
    return names
