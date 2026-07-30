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
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.audit import FitAudit, FitAuditPolicy, FitDiagnostics
from behavio.trials import Study


class PredictionMode(StrEnum):
    """The information set used to construct a prediction."""

    FILTERED = "filtered"
    SMOOTHED = "smoothed"


LOG_DENSITY_FLOOR: Final = float(np.log(np.finfo(np.float64).tiny))
"""Smallest log density a pointwise score reports.

A density read off a numerical grid underflows to exactly zero far into the tail even where
the analytic density is merely very small, so ``log(0)`` is a statement about the grid rather
than about the model. Flooring keeps a single tail trial from making an entire fold's score
``-inf``, and the floor is a declared constant rather than a magic number inside one model:
``behavio.models._kernels.wiener`` and ``behavio.compose.mixture`` both read it from here.
"""

_MASS_TOLERANCE: Final = 1e-3
"""How far above one a density row's integrated mass may sit before it is rejected.

Loose on purpose. The trapezoid rule overshoots a convex density, so a legitimate
tabulation on a coarse grid integrates to slightly more than one, and rejecting that would
reject every real solver output. The check exists to catch a density that is wrong *by
construction* -- per-bin masses passed as densities, a missing Jacobian after a unit change,
a normalisation applied twice -- and those are wrong by orders of magnitude, not by 1e-4.
"""


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

    ``is_sampled`` says which of the two estimator contracts produced this record: an
    optimizer-fitted :class:`BehaviourEstimator` or a
    :class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator`. It is a capability
    rather than a detail because it decides *how a candidate is driven* -- ``fit`` versus
    ``sample``, a :class:`FitResult` versus a
    :class:`~behavio.posterior.PosteriorResult`, an optimizer audit versus a convergence
    audit -- and a frozen protocol declares it per candidate.

    ``can_bind_design`` says the model is generative **relative to a design**: it cannot
    name its scalar parameters in the abstract, but ``bind(design)`` returns a model that
    can. A mixed-effects parameter vector is the standard case -- how many coordinates
    ``(1|subject)`` has and which columns ``C(condition)`` yields are facts about the data,
    not about the specification -- so such a model reports ``can_simulate=False`` and
    ``can_bind_design=True``. Recovery is still available to it, which is why
    ``can_recover_parameters`` may be true without ``can_simulate``; ``AGENTS.md`` already
    treats recovery as design-specific evidence, so a design is exactly what such a model
    was waiting for.
    """

    scored_columns: tuple[str, ...]
    prediction_modes: tuple[PredictionMode, ...]
    can_simulate: bool
    can_recover_parameters: bool
    required_task_columns: tuple[str, ...] = ()
    is_sampled: bool = False
    can_bind_design: bool = False

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
        if not all(
            isinstance(flag, bool)
            for flag in (
                self.can_simulate,
                self.can_recover_parameters,
                self.is_sampled,
                self.can_bind_design,
            )
        ):
            raise ValueError("capability flags must be boolean")
        if self.can_recover_parameters and not (self.can_simulate or self.can_bind_design):
            raise ValueError(
                "parameter recovery requires simulation, either directly or after binding "
                "the model to a design"
            )
        if self.can_simulate and self.can_bind_design:
            raise ValueError(
                "a model is generative either in the abstract or relative to a design, not "
                "both; binding a model that already names its parameters would give one "
                "simulator two vocabularies"
            )
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


@dataclass(frozen=True, slots=True)
class DensityPrediction:
    """A predictive density for one continuous outcome, on an explicit grid.

    :class:`Prediction` is a probability plus a linear predictor and
    :class:`CategoricalPrediction` is a simplex over named categories. Both describe a
    *discrete* outcome. Nothing else in the contract described a response time, a
    continuous confidence report, or the finishing-time distribution of a race between
    accumulators, so a model that predicts one had exactly two choices: throw the
    continuous prediction away and report choice probabilities alone, or return it through
    a private side channel no Behavio consumer could read. The first is what the contract
    silently encouraged, and it discards the half of the prediction that distinguishes a
    response-time model from a logistic regression -- which is to say, the half a
    falsification layer most wants to test.

    ``grid`` is the strictly increasing outcome coordinate the density is tabulated on, in
    the units of the scored column. ``density`` is ``(n_trials, n_grid)`` for an unlabelled
    continuous outcome and ``(n_trials, n_categories, n_grid)`` when the density is
    *defective* across ``categories``:

    - an unlabelled continuous outcome (a confidence rating) is a density that integrates
      to one;
    - a two-boundary diffusion is two defective densities whose masses are the choice
      probabilities and whose total integral is one;
    - an *n*-accumulator race is *n* defective densities on the same coordinate.

    Integrated mass is checked, not assumed: a row's total mass may fall short of one,
    because a finite grid truncates a distribution with unbounded support and because a
    model may leave probability undecided, but it may never exceed one. :attr:`total_mass`
    reports what each row actually integrates to, so a caller can see truncation rather than
    discover it as a bias.

    How a consumer scores one
    -------------------------
    The three facts a caller needs -- what the choice probabilities are, what the density
    is at an observed value, and how much mass the grid failed to cover -- are methods
    rather than conventions, so no two consumers can derive them differently.
    :meth:`observed_log_density` is the pointwise log score, and it is exactly what a
    proper log-scoring rule means for a joint discrete/continuous observation.
    :meth:`choice_prediction` is the discrete margin, and it is the only part of a density
    that a *probability* scoring rule such as the Brier score can be applied to; see
    :func:`behavio.compare.compare_models` for what that means for a comparison table.
    """

    grid: NDArray[np.float64]
    density: NDArray[np.float64]
    outcome: str
    mode: PredictionMode
    categories: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        grid = protected_array(self.grid, dtype=np.float64)
        density = protected_array(self.density, dtype=np.float64)
        mode = PredictionMode(self.mode)
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("a density prediction must name the outcome column it predicts")
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("the outcome grid must be one-dimensional with at least two points")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("the outcome grid must be finite and strictly increasing")
        categories = (
            None
            if self.categories is None
            else tuple(_prediction_category(value) for value in self.categories)
        )
        if categories is None:
            if density.ndim != 2:
                raise ValueError("an unlabelled density must be (n_trials, n_grid)")
        else:
            if len(categories) < 2:
                raise ValueError("a defective density must name at least two categories")
            keys = [_category_key(value) for value in categories]
            if len(set(keys)) != len(keys):
                raise ValueError("density categories must be unique")
            if density.ndim != 3 or density.shape[1] != len(categories):
                raise ValueError("a defective density must be (n_trials, n_categories, n_grid)")
        if density.shape[-1] != grid.size:
            raise ValueError("the density's last axis must match the outcome grid")
        if not density.shape[0]:
            raise ValueError("a density prediction must cover at least one trial")
        if not np.all(np.isfinite(density)) or np.any(density < 0):
            raise ValueError("densities must be finite and non-negative")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "mode", mode)
        mass = self.total_mass
        if np.any(mass > 1.0 + _MASS_TOLERANCE):
            worst = float(np.max(mass))
            raise ValueError(f"a predictive density may not integrate above one; observed {worst}")
        if np.any(mass <= 0.0):
            raise ValueError("every trial's predictive density must carry positive mass")

    @property
    def n_observations(self) -> int:
        """Number of trial rows represented by the prediction."""

        return int(self.density.shape[0])

    @property
    def n_grid(self) -> int:
        """Number of points the density is tabulated on."""

        return int(self.grid.size)

    @property
    def is_defective(self) -> bool:
        """Whether the density is split across named categories that share the grid."""

        return self.categories is not None

    @property
    def category_mass(self) -> NDArray[np.float64]:
        """Integrated mass of each category, ``(n_trials, n_categories)``.

        For a two-boundary diffusion these are the choice probabilities, up to whatever mass
        the grid truncated away.
        """

        if self.categories is None:
            raise ValueError("this prediction declares no categories, so it has no category mass")
        return protected_array(np.trapezoid(self.density, self.grid, axis=-1), dtype=np.float64)

    @property
    def total_mass(self) -> NDArray[np.float64]:
        """Total integrated mass of each trial's prediction, ``(n_trials,)``.

        One minus this is the probability the grid does not account for: truncated tail,
        undecided mass, or both. It is reported rather than normalised away.
        """

        integral = np.trapezoid(self.density, self.grid, axis=-1)
        if self.categories is not None:
            integral = np.sum(integral, axis=-1)
        return protected_array(integral, dtype=np.float64)

    def density_at(
        self, values: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Linearly interpolate the density at one outcome value per trial.

        Returns ``(n_trials,)`` for an unlabelled density and ``(n_trials, n_categories)``
        for a defective one. Values outside the grid evaluate to zero rather than to the
        nearest endpoint, because a grid that does not reach an observation carries no
        information about it and clamping would invent some.

        Interpolation, not nearest-bin lookup, is the point. A package that returns a
        tabulated PDF is usually scored by rounding each observation to its grid index,
        which makes the reported per-trial likelihood a function of the solver's step size.
        """

        observed = np.asarray(values, dtype=np.float64)
        if observed.ndim != 1 or observed.size != self.n_observations:
            raise ValueError("values must contain one outcome per predicted trial")
        inside = np.isfinite(observed) & (observed >= self.grid[0]) & (observed <= self.grid[-1])
        upper = np.clip(np.searchsorted(self.grid, observed, side="left"), 1, self.n_grid - 1)
        lower = upper - 1
        span = self.grid[upper] - self.grid[lower]
        weight = np.where(span > 0, (observed - self.grid[lower]) / span, 0.0)
        weight = np.clip(weight, 0.0, 1.0)
        if self.categories is None:
            left = self.density[np.arange(self.n_observations), lower]
            right = self.density[np.arange(self.n_observations), upper]
            interpolated = left + weight * (right - left)
            return protected_array(np.where(inside, interpolated, 0.0), dtype=np.float64)
        rows = np.arange(self.n_observations)[:, None]
        columns = np.arange(len(self.categories))[None, :]
        left = self.density[rows, columns, lower[:, None]]
        right = self.density[rows, columns, upper[:, None]]
        interpolated = left + weight[:, None] * (right - left)
        return protected_array(np.where(inside[:, None], interpolated, 0.0), dtype=np.float64)

    def observed_log_density(
        self,
        values: Sequence[float] | NDArray[np.floating[Any]],
        categories: Sequence[Any] | NDArray[Any] | None = None,
        *,
        floor: float = LOG_DENSITY_FLOOR,
    ) -> NDArray[np.float64]:
        """Return one floored log density per trial for the observation it actually made.

        This is the pointwise score of a joint discrete/continuous observation: for a
        two-boundary diffusion, the log of the defective density of the *observed* boundary
        at the *observed* time. ``categories`` is required exactly when the prediction is
        defective, and names the observed category of each row.
        """

        if self.categories is None:
            if categories is not None:
                raise ValueError("this prediction declares no categories to select")
            density = self.density_at(values)
        else:
            if categories is None:
                raise ValueError("a defective density needs the observed category of each row")
            observed = list(categories)
            if len(observed) != self.n_observations:
                raise ValueError("categories must contain one observed category per trial")
            index = {_category_key(value): column for column, value in enumerate(self.categories)}
            try:
                columns = np.asarray(
                    [index[_category_key(value)] for value in observed], dtype=np.intp
                )
            except KeyError as error:
                raise ValueError(
                    f"observed category {error.args[0][1]!r} is not one this prediction "
                    f"declares: {list(self.categories)}"
                ) from None
            density = self.density_at(values)[np.arange(self.n_observations), columns]
        floored = float(floor)
        with np.errstate(divide="ignore"):
            scores = np.where(density > 0.0, np.log(density), floored)
        return protected_array(np.maximum(scores, floored), dtype=np.float64)

    def category_codes(self, categories: Sequence[Any] | NDArray[Any]) -> NDArray[np.int64]:
        """Return the column index of each row's observed category.

        These are the ``outcome_codes`` a fold retains beside a defective density, and the
        index :meth:`choice_prediction` must be scored against. Deriving them here rather
        than in each consumer is what keeps a fold's codes and the density's own column
        order from drifting apart.
        """

        if self.categories is None:
            raise ValueError("this prediction declares no categories to code against")
        observed = list(categories)
        if len(observed) != self.n_observations:
            raise ValueError("categories must contain one observed category per trial")
        index = {_category_key(value): column for column, value in enumerate(self.categories)}
        try:
            return protected_array(
                [index[_category_key(value)] for value in observed], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(
                f"observed category {error.args[0][1]!r} is not one this prediction "
                f"declares: {list(self.categories)}"
            ) from None

    def choice_prediction(self) -> CategoricalPrediction:
        """Marginalise the grid away and return the categorical prediction it implies.

        Rows are renormalised, because a :class:`CategoricalPrediction` must sum to one and
        a truncated grid does not. The renormalisation is exact only when
        :attr:`total_mass` is one; read that array before treating the result as the model's
        unconditional choice probabilities.
        """

        mass = np.asarray(self.category_mass, dtype=np.float64)
        total = np.sum(mass, axis=1, keepdims=True)
        probability = mass / total
        with np.errstate(divide="ignore"):
            linear_predictor = np.log(probability)
        return CategoricalPrediction(
            probability=probability,
            linear_predictor=linear_predictor,
            categories=self.categories or (),
            mode=self.mode,
        )

    def expected_outcome(self) -> NDArray[np.float64]:
        """Mass-weighted mean outcome per trial, conditional on the grid's support."""

        weighted = self.density * self.grid
        integral = np.trapezoid(weighted, self.grid, axis=-1)
        if self.categories is not None:
            integral = np.sum(integral, axis=-1)
        return protected_array(integral / self.total_mass, dtype=np.float64)

    def take(self, indices: Sequence[int] | NDArray[np.integer[Any]]) -> DensityPrediction:
        """Return a protected row subset on the same grid and category coordinate."""

        return DensityPrediction(
            grid=self.grid,
            density=self.density[indices],
            outcome=self.outcome,
            mode=self.mode,
            categories=self.categories,
        )


ModelPrediction = Prediction | CategoricalPrediction | DensityPrediction
"""Every prediction shape a Behavio consumer reads back off ``predict()``.

The three are not a hierarchy. :class:`Prediction` is one probability per row,
:class:`CategoricalPrediction` is a simplex over named categories, and
:class:`DensityPrediction` is a density over a continuous outcome that may additionally be
defective across categories. Every consumer that slices a prediction -- most of all
:func:`behavio.evaluate.evaluate_splits`, which keeps only a fold's scored rows -- goes
through ``take``, which all three implement with the same meaning.
"""


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
class DensityBehaviourEstimator(BehaviourEstimator, Protocol):
    """An estimator that predicts a density for a continuous scored outcome.

    A model may satisfy this and still return a :class:`Prediction` from ``predict`` -- the
    protocol only promises that a density is *available*, not that it is the headline
    prediction. A model that returns the :class:`DensityPrediction` from ``predict`` itself,
    as :class:`behavio.foreign.pyddm.PyDDMDriftDiffusion` does, is the case the evaluation
    and comparison layers were widened for: the density then reaches every fold, every
    score and every report rather than being reachable only by a caller who knows to ask
    for it.

    A model implementing both halves cannot quietly disagree with itself:
    :func:`behavio.adapters.estimator_conformance.check_behaviour_estimator` integrates the
    density and compares it against the choice probabilities.
    """

    @property
    def density_outcome(self) -> str: ...

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction: ...


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
    bindable = not generative and _binds_a_design(model)
    if generative:
        validate_parameter_names(model.parameter_names)
    return ModelCapabilities(
        scored_columns=tuple(model.scored_columns),
        prediction_modes=tuple(model.supported_prediction_modes),
        can_simulate=generative,
        can_recover_parameters=generative or bindable,
        required_task_columns=model_task_columns(model),
        can_bind_design=bindable,
    )


def _binds_a_design(model: Any) -> bool:
    """Whether the model is generative only relative to a design.

    The protocol itself is :class:`behavio.contracts.posterior.DesignGenerativeBehaviourModel`,
    which cannot be named here: this module is the one ``behavio.contracts.posterior``
    imports. A ``bind`` method is exactly the structural claim that protocol makes, so it is
    read directly, and the *result* is checked against the generative contracts by
    :func:`behavio.contracts.posterior.bind_to_design` rather than assumed here.
    """

    return callable(getattr(model, "bind", None))


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

    A NumPy scalar is unwrapped first. Declared categories are normalised on construction
    and are therefore always Python scalars, but *observed* categories arrive straight out
    of a study column as ``np.int64``; keying on the type would make every observation
    unmatchable against the category it names.
    """

    if isinstance(value, np.generic):
        value = value.item()
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
