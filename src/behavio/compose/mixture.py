"""``mix(model, component)``: let a simpler process account for some of the trials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from behavio._internal.arrays import protected_array
from behavio.contracts.bounded import (
    BOUNDED_COORDINATE,
    BoundedCoordinateEstimator,
    RowCoefficientDesign,
    RowObjective,
    validated_row_coefficients,
)
from behavio.contracts.compose import (
    LinearPredictorLikelihood,
    PenalisedDesign,
    PenalisedLinearEstimator,
    linear_predictor,
    validate_predictor_shape,
)
from behavio.contracts.estimator import (
    LOG_DENSITY_FLOOR,
    DensityPrediction,
    FitResult,
    ModelDataError,
    ModelPrediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.contracts.mixture import (
    MIXTURE_LOGIT,
    MIXTURE_LOGIT_BOUND,
    MixtureComponent,
    mixture_logit,
    mixture_weight,
    require_independent_rows,
    require_mixable,
    require_mixture_component,
    validate_weight_bounds,
)
from behavio.contracts.natural import natural_quantities
from behavio.models._kernels.introspection import WARNING, Describable, ModelFinding
from behavio.trials import Study

__all__ = [
    "MixtureLikelihood",
    "MixtureModel",
    "MixtureRowModel",
    "MixtureSimulation",
    "blended_density",
    "mix",
]

LOGIT_STARTS = (-5.0, -2.0, 1.0)
"""Deterministic mixture-logit restarts, in the order the deleted lapse model used them.

A mixture likelihood is not convex in the weight and the model's parameters jointly -- a
shallow slope with a large weight and a steep slope with a small one are two hills -- so a
mixture is a multi-start problem even when the model it wraps is not. These three are the
starts the deleted ``LapsePsychometric`` searched, and at ``weight_bounds=(0, 0.2)`` they
place the weight at 0.13 %, 2.4 % and 14.6 % of the trials.
"""


def mix(
    model: PenalisedLinearEstimator | BoundedCoordinateEstimator,
    component: MixtureComponent,
    *,
    weight_bounds: tuple[float, float] = (0.0, 0.25),
    n_restarts: int = 3,
) -> MixtureModel | MixtureRowModel:
    """Return ``model`` mixed with ``component``, estimating the mixing weight.

    Each row's density becomes ``(1 - w) * model + w * component``, with ``w`` estimated as
    one extra coordinate, :data:`~behavio.contracts.mixture.MIXTURE_LOGIT`. ``weight_bounds``
    **declares** the range the weight is estimated inside, and is the general form of the
    ``maximum_lapse`` and ``probability_bounds`` arguments three separate mechanisms used to
    carry; ``(0.0, 0.25)`` says a mixture may account for at most a quarter of the trials.
    Everything about the component is declared and nothing about it is estimated, so a
    mixture adds exactly one parameter whatever it is mixed with.

    The condition is **row independence** and nothing else, so both estimator contracts are
    open: a model whose likelihood reads a study through a penalised linear predictor is
    mixed by adding one cell to that predictor, and a model whose likelihood is a
    :class:`~behavio.contracts.bounded.RowObjective` with one block per row is mixed by
    adding one column to the row coordinate. What is refused is a likelihood that recurses,
    and the refusal is the model's own sentence -- see
    :func:`~behavio.contracts.mixture.require_mixable`.

    ``mix`` is the **innermost** combinator. ``smooth(mix(model))`` is a mixed model whose
    parameters -- the weight among them, if it is named -- follow paths;
    ``hierarchical(mix(model), parameters=("mixture_logit",))`` is a lapse rate that varies
    by subject. The reverse orders are refused: a mixture wrapping a smooth model is
    ``smooth(mix(model), parameters=...)`` with the weight left stationary, so admitting it
    would be a second spelling of a model that is already expressible, and every combinator
    wrapping only things more primitive than itself is what keeps the pass-through surface
    linear rather than quadratic.
    """

    if hasattr(model, "varying_effects"):
        raise TypeError(
            "mix is the innermost combinator: write hierarchical(mix(model)) rather than "
            "mix(hierarchical(model)). A hierarchical estimator reports the population "
            "coordinate while fitting a joint one whose width depends on how many groups "
            "the study has, so it cannot be widened again from outside"
        )
    if hasattr(model, "trajectory_from_knots"):
        raise TypeError(
            "mix is the innermost combinator: write smooth(mix(model)) rather than "
            "mix(smooth(model)). Naming only the wrapped model's parameters in smooth() "
            "leaves the mixture weight stationary, which is the model mix(smooth(model)) "
            "would have been"
        )
    if hasattr(model, "mixture_component"):
        raise TypeError(
            "a model is mixed with one component: mix(mix(model, a), b) estimates a weight "
            "on a weight, which is a reparameterisation of a three-way mixture and not one "
            "anybody reports. Declare the combined process as a single component"
        )
    route = require_mixable(model, combinator="mix")
    require_mixture_component(component, model, combinator="mix")
    build = MixtureRowModel if route == BOUNDED_COORDINATE else MixtureModel
    return build(
        model=model,
        component=component,
        weight_bounds=tuple(weight_bounds),
        n_restarts=n_restarts,
    )


# --------------------------------------------------------------------------------------
# The mixed likelihood
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixtureLikelihood:
    """Two densities per row, combined by a weight that is one of the predictor cells.

    Every quantity this needs arrives through the linear predictor, which is what makes a
    mixture composable rather than a special case inside one family. The wrapped model's
    own cells come first; then the mixture logit, which is the only cell a parameter
    multiplies; then the component's log density and its predicted probability, which are
    carried as **offsets** -- terms on the predictor that no parameter multiplies. That is
    the same channel per-trial option availability already travels down, widened by one
    observation: a component's log density is a function of the outcome as well as of the
    trial, and the predictor is the only per-row channel a combinator preserves when it
    slices a study into groups or multiplies a design by a temporal basis.

    Nothing here is family-specific. The wrapped likelihood supplies its own per-row log
    density, its own gradient in its own cells and its own curvature; the mixture arithmetic
    -- responsibilities, and the chain rule through the logit -- is the same for a logistic
    GLM, a softmax and a first-passage density.
    """

    model_likelihood: LinearPredictorLikelihood
    n_model_cells: int
    scalar_model: bool
    prediction_width: int
    weight_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        if self.n_model_cells < 1 or self.prediction_width < 1:
            raise ValueError("a mixture needs at least one model cell and one prediction cell")
        object.__setattr__(self, "weight_bounds", validate_weight_bounds(self.weight_bounds))

    @property
    def n_cells(self) -> int:
        """The total width of one row's mixed linear predictor."""

        return self.n_model_cells + 2 + self.prediction_width

    @property
    def logit_cell(self) -> int:
        """Which cell carries the estimated mixture logit."""

        return self.n_model_cells

    @property
    def density_cell(self) -> int:
        """Which cell carries the component's log density of the observed outcome."""

        return self.n_model_cells + 1

    def model_cells(self, cells: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the wrapped model's own linear predictor, in its own shape."""

        block = np.asarray(cells, dtype=np.float64)[:, : self.n_model_cells]
        return block[:, 0] if self.scalar_model else block

    def weights(self, cells: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return each row's mixing weight on the declared range."""

        return mixture_weight(
            np.asarray(cells, dtype=np.float64)[:, self.logit_cell], self.weight_bounds
        )

    def prediction(
        self, linear_predictor: NDArray[np.float64], *, mode: PredictionMode
    ) -> ModelPrediction:
        """Return the weighted average of the two processes' predictions."""

        cells = np.asarray(linear_predictor, dtype=np.float64)
        model_prediction = self.model_likelihood.prediction(self.model_cells(cells), mode=mode)
        return blended_prediction(
            model_prediction, self.weights(cells), cells[:, self.density_cell + 1 :]
        )

    def pointwise_log_prob(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the mixed log density of each observation."""

        return self._parts(linear_predictor, outcomes).log_density

    def log_density(
        self, cells: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the mixed log density, under the name a Wiener solver asks for."""

        return self.pointwise_log_prob(cells, outcomes)

    def negative_log_likelihood(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> float:
        """Return the summed negative log density.

        A mixture is never inadmissible in the way a bare first-passage density is: a
        response too fast for the model to have produced is exactly what the component is
        there to explain, so the floored value the wrapped density returns is combined
        with a finite component density rather than short-circuiting the objective.
        """

        return float(-np.sum(self.pointwise_log_prob(linear_predictor, outcomes)))

    def value_and_gradient(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the mixed cells.

        The gradient in the wrapped model's cells is its own gradient scaled by the
        posterior probability that the row came from the model, which is the responsibility
        an EM step would compute and is here simply the derivative.
        """

        parts = self._parts(linear_predictor, outcomes)
        cells = np.asarray(linear_predictor, dtype=np.float64)
        gradient = np.zeros_like(cells)
        model_gradient = self._model_gradient(cells, outcomes)
        if self.scalar_model:
            gradient[:, 0] = parts.model_responsibility * model_gradient
        else:
            gradient[:, : self.n_model_cells] = parts.model_responsibility[:, None] * model_gradient
        gradient[:, self.logit_cell] = -(parts.upper - parts.lower) * parts.weight_slope
        gradient[:, self.density_cell] = -parts.responsibility
        return float(-np.sum(parts.log_density)), gradient

    def curvature(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return each row's observed information block in the mixed cells.

        Observed rather than expected, and a matrix rather than a number even when the
        wrapped family's is a number: mixing couples the weight to every cell the model
        has, so a mixed predictor's cells are never independent.
        """

        parts = self._parts(linear_predictor, outcomes)
        cells = np.asarray(linear_predictor, dtype=np.float64)
        n_rows, n_cells = cells.shape
        blocks = np.zeros((n_rows, n_cells, n_cells), dtype=np.float64)
        width = self.n_model_cells
        model_curvature = np.asarray(
            self.model_likelihood.curvature(self.model_cells(cells), outcomes), dtype=np.float64
        )
        gradient = self._model_gradient(cells, outcomes)
        if self.scalar_model:
            model_curvature = model_curvature.reshape(n_rows, 1, 1)
            gradient = gradient.reshape(n_rows, 1)
        model_responsibility = parts.model_responsibility
        product = parts.responsibility * model_responsibility
        blocks[:, :width, :width] = (
            model_responsibility[:, None, None] * model_curvature
            - product[:, None, None] * gradient[:, :, None] * gradient[:, None, :]
        )
        # d(responsibility)/d(weight), which is what couples every model cell to the logit.
        drift = parts.upper - parts.responsibility * (parts.upper - parts.lower)
        cross = -drift[:, None] * gradient * parts.weight_slope[:, None]
        blocks[:, :width, self.logit_cell] = cross
        blocks[:, self.logit_cell, :width] = cross
        density_cross = -product[:, None] * gradient
        blocks[:, :width, self.density_cell] = density_cross
        blocks[:, self.density_cell, :width] = density_cross
        difference = parts.upper - parts.lower
        blocks[:, self.logit_cell, self.logit_cell] = (
            difference**2 * parts.weight_slope**2 - difference * parts.weight_bend
        )
        logit_density = -drift * parts.weight_slope
        blocks[:, self.logit_cell, self.density_cell] = logit_density
        blocks[:, self.density_cell, self.logit_cell] = logit_density
        blocks[:, self.density_cell, self.density_cell] = -product
        return blocks

    def responsibilities(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the posterior probability that each row came from the component."""

        return self._parts(linear_predictor, outcomes).responsibility

    def _model_gradient(
        self, cells: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the wrapped likelihood's own per-row gradient in its own cells.

        Asked for separately, and only by the two members that need it, because a
        first-passage density's gradient costs two evaluations of the density per cell and
        the objective a bounded solver differences numerically never wants any of them.
        """

        _, gradient = self.model_likelihood.value_and_gradient(self.model_cells(cells), outcomes)
        return np.asarray(gradient, dtype=np.float64)

    def _parts(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> _MixtureParts:
        cells = np.asarray(linear_predictor, dtype=np.float64)
        if cells.ndim != 2 or cells.shape[1] != self.n_cells:
            raise ValueError(f"a mixed linear predictor has {self.n_cells} cells per row")
        observed = np.asarray(outcomes, dtype=np.float64)
        model_density = np.asarray(
            self.model_likelihood.pointwise_log_prob(self.model_cells(cells), observed),
            dtype=np.float64,
        )
        return mixture_parts(
            model_density,
            cells[:, self.density_cell],
            cells[:, self.logit_cell],
            self.weight_bounds,
        )


@dataclass(frozen=True, slots=True)
class _MixtureParts:
    """Everything both the gradient and the curvature of a mixed row are built from.

    ``upper`` and ``lower`` are ``exp(component - mixed)`` and ``exp(model - mixed)``, which
    are the responsibilities divided by their own weights. Carrying them rather than the
    responsibilities is what keeps a weight resting at exactly zero or exactly one finite:
    the quotient that would be ``0 / 0`` there is a single exponential of a difference of
    logs here.
    """

    log_density: NDArray[np.float64]
    responsibility: NDArray[np.float64]
    model_responsibility: NDArray[np.float64]
    upper: NDArray[np.float64]
    lower: NDArray[np.float64]
    weight_slope: NDArray[np.float64]
    weight_bend: NDArray[np.float64]


def mixture_parts(
    model_density: NDArray[np.float64],
    component_density: NDArray[np.float64],
    logit: NDArray[np.float64],
    bounds: tuple[float, float],
) -> _MixtureParts:
    """Combine two per-row log densities at a per-row weight logit.

    The whole of a mixture's arithmetic, and the reason there is one of it rather than two.
    Three arrays of one number per row go in -- the model's log density, the component's, and
    the coordinate the weight is a function of -- and everything a gradient or a curvature
    needs comes out. Nothing here knows whether the logit arrived as a cell of a linear
    predictor or as a column of a row coordinate, which is exactly the property that lets the
    two estimator contracts share a mixture instead of each growing one.
    """

    lower_bound, upper_bound = bounds
    relative = expit(logit)
    weight = lower_bound + (upper_bound - lower_bound) * relative
    slope = (upper_bound - lower_bound) * relative * (1.0 - relative)
    bend = slope * (1.0 - 2.0 * relative)
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = weight > 0.0
        log_weight = np.where(positive, np.log(np.where(positive, weight, 1.0)), -np.inf)
        log_complement = np.log1p(-weight)
        model_term = log_complement + model_density
        component_term = log_weight + component_density
        raw = np.logaddexp(model_term, component_term)
        usable = np.isfinite(raw)
        floored = np.where(usable, raw, LOG_DENSITY_FLOOR)
        upper = np.where(
            usable & np.isfinite(component_density),
            np.exp(np.minimum(component_density - floored, 700.0)),
            0.0,
        )
        lower = np.where(
            usable & np.isfinite(model_density),
            np.exp(np.minimum(model_density - floored, 700.0)),
            0.0,
        )
    return _MixtureParts(
        log_density=floored,
        responsibility=weight * upper,
        model_responsibility=(1.0 - weight) * lower,
        upper=upper,
        lower=lower,
        weight_slope=slope,
        weight_bend=bend,
    )


def blended_prediction(
    model_prediction: ModelPrediction,
    weight: NDArray[np.float64],
    component: NDArray[np.float64],
) -> ModelPrediction:
    """Return the weighted average of two point predictions, keeping the model's own type.

    A mixture's predicted probability is an average of two probabilities, which is the one
    quantity of a mixture that is not a density and therefore not a function of the two log
    densities :func:`mixture_parts` combines. It is written once here for the same reason.
    """

    if isinstance(model_prediction, DensityPrediction):
        raise TypeError(
            "a tabulated density is averaged by blended_density(), not here: a component's "
            "contribution to a continuous prediction is a density on the model's own grid "
            "rather than one number per row"
        )
    probability = np.asarray(model_prediction.probability, dtype=np.float64)
    if probability.ndim == 1:
        blended = (1.0 - weight) * probability + weight * component[:, 0]
        floor = float(np.finfo(np.float64).tiny)
        clipped = np.clip(blended, floor, 1.0 - float(np.finfo(np.float64).eps))
        return replace(
            model_prediction,
            probability=blended,
            linear_predictor=np.log(clipped) - np.log1p(-clipped),
        )
    blended = (1.0 - weight)[:, None] * probability + weight[:, None] * component
    with np.errstate(divide="ignore"):
        logits = np.log(blended)
    return replace(model_prediction, probability=blended, linear_predictor=logits)


def blended_density(
    model_prediction: DensityPrediction,
    weight: NDArray[np.float64],
    component: NDArray[np.float64],
) -> DensityPrediction:
    """Return the weighted average of two densities tabulated on one shared grid.

    The continuous twin of :func:`blended_prediction`, and the reason it is a second function
    rather than a branch inside the first: a probability prediction is one number per row and
    a component supplies it without knowing what the model predicted, while a density is
    tabulated on a grid the *model* chose, so a component's half has to be evaluated after
    the model's prediction exists rather than before it.

    Nothing about the arithmetic is new -- :math:`(1 - \\omega) f_{\\text{model}} + \\omega
    f_{\\text{comp}}` at every grid point is the same statement
    :class:`MixtureLikelihood` makes at every observation. Two densities each integrating to
    at most one average to at most one, so the mixed tabulation carries whatever truncation
    the model's own grid had and no more, which is what
    :attr:`~behavio.contracts.estimator.DensityPrediction.total_mass` goes on reporting.
    """

    if model_prediction.is_defective:
        raise ValueError(
            "a mixture over a density split across categories needs a component that writes "
            "the same categories, which is a joint observation rather than a bare continuous "
            "one; declare a component that scores both channels"
        )
    density = np.asarray(model_prediction.density, dtype=np.float64)
    values = np.asarray(component, dtype=np.float64)
    if values.shape != density.shape:
        raise ValueError("a component's density must be tabulated on the model's own grid")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("a component's density must be finite and non-negative")
    mixed = (1.0 - weight)[:, None] * density + weight[:, None] * values
    return replace(model_prediction, density=mixed)


def prediction_values(prediction: ModelPrediction) -> NDArray[np.float64]:
    """Return whatever a prediction says about a row, for the one question asked of it.

    Only ever asked "does this differ between two rows", which a probability, a simplex and a
    tabulated density each answer in their own numbers. Written once so that a mixture's
    identifiability check does not have to grow a branch per prediction type.
    """

    if isinstance(prediction, DensityPrediction):
        return np.asarray(prediction.density, dtype=np.float64)
    return np.asarray(prediction.probability, dtype=np.float64)


# --------------------------------------------------------------------------------------
# What a mixed simulation retains
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixtureSimulation:
    """Observations paired with the process that produced each of them but is not observed."""

    study: Study
    from_component: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not isinstance(self.study, Study):
            raise TypeError("study must be a Study")
        indicators = protected_array(self.from_component, dtype=np.bool_)
        if indicators.shape != (len(self.study),):
            raise ValueError("from_component must contain one indicator per study row")
        object.__setattr__(self, "from_component", indicators)

    @property
    def n_from_component(self) -> int:
        """How many rows the component, rather than the model, generated."""

        return int(np.sum(self.from_component))


# --------------------------------------------------------------------------------------
# Everything a mixture is, whichever contract it reaches the model's density through
# --------------------------------------------------------------------------------------


class _MixedModel(Describable):
    """The two-thirds of a mixture that does not depend on how the density is reached.

    Naming, the declared weight range, the component's own log density, the penalty, the
    box, the restarts, the group prior, the simulator and the reported coordinate are the
    same statements whether the weight arrived as a cell of a linear predictor or as a column
    of a row coordinate. Only the objective differs, which is the same division
    :class:`~behavio.contracts.compose.PenalisedLinearEstimator` and
    :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator` already make.

    A mixin rather than a dataclass base: both subclasses declare the same four fields, and
    a frozen slotted dataclass inheriting fields from another one is a subtlety that buys
    nothing here. ``__slots__`` is empty for the reason :class:`Describable`'s is.
    """

    __slots__ = ()

    model: Any
    component: MixtureComponent
    weight_bounds: tuple[float, float]
    n_restarts: int

    def _validate_mixture(self) -> tuple[float, float]:
        """Return the validated weight range, refusing a coordinate the weight would shadow.

        A component's ``weight_name`` is checked against the wrapped model's **reported**
        names as well as its estimated ones, because the reported coordinate is what a
        mixture appends to and two ``lapse_rate`` entries are not a coordinate. That is the
        refusal :class:`~behavio.models.PsychometricFunction` gets, and it is the right one:
        a curve that already estimates a lapse inside its link has nothing left for a
        symmetric mixture weight to be identified against.
        """

        bounds = validate_weight_bounds(self.weight_bounds)
        if (
            isinstance(self.n_restarts, bool)
            or not isinstance(self.n_restarts, int)
            or self.n_restarts < 1
        ):
            raise ValueError("n_restarts must be a positive integer")
        inner = tuple(self.model.parameter_names)
        if MIXTURE_LOGIT in inner:
            raise ValueError(f"the wrapped model already has a parameter named {MIXTURE_LOGIT!r}")
        reported = (*inner, *getattr(self.model, "natural_names", ()))
        if self.component.weight_name in reported:
            raise ValueError(
                f"the wrapped model already has a parameter named "
                f"{self.component.weight_name!r}, which is what this component calls its weight"
            )
        return bounds

    # -- identity ---------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"{self.component.component_name}-{self.model.model_name}"

    @property
    def signature(self) -> str:
        lower, upper = self.weight_bounds
        return (
            f"mix[component={self.component.signature};weight={self.component.weight_name};"
            f"weight_bounds={lower:g},{upper:g};n_restarts={self.n_restarts}]"
            f"({self.model.signature})"
        )

    @property
    def mixture_component(self) -> MixtureComponent:
        """The declared process this model is mixed with.

        Its presence is also how :func:`mix` recognises that a model is already mixed and
        refuses to mix it again.
        """

        return self.component

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (*self.model.parameter_names, MIXTURE_LOGIT)

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """The estimated coordinate, under the name model introspection looks for."""

        return self.parameter_names

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return tuple(self.model.scored_columns)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return tuple(self.model.required_task_columns)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return tuple(self.model.supported_prediction_modes)

    @property
    def design_spec(self) -> Any:
        """The wrapped model's design: a mixture adds a parameter, not a column."""

        return getattr(self.model, "design_spec", None)

    @property
    def declared_priors(self) -> tuple[str, ...]:
        lower, upper = self.weight_bounds
        return (
            f"mixture with {self.component.signature}: {self.component.weight_name} estimated "
            f"in [{lower:g}, {upper:g}] as an unpenalised logit",
            *getattr(self.model, "declared_priors", ()),
        )

    @property
    def categories(self) -> tuple[Any, ...]:
        """The wrapped model's outcome coordinate, when it scores a categorical outcome."""

        return tuple(self.model.categories)

    @property
    def outcome_channels(self) -> tuple[str, ...]:
        """The wrapped model's channels: a mixture cannot change what is observed."""

        return tuple(self.model.outcome_channels)

    @property
    def score_metrics(self) -> tuple[Any, ...]:
        """The wrapped model's scoring rules, when it declares any.

        Forwarded rather than restated because mixing does not change what kind of prediction
        a model makes: a mixture over a family that predicts an unlabelled density over a
        continuous outcome still has no discrete margin for a probability scoring rule to
        read. Absent when the wrapped model declares nothing, so that a consumer's default
        stays the default.
        """

        declared = getattr(self.model, "score_metrics", None)
        if declared is None:
            raise AttributeError(
                f"{type(self.model).__name__} declares no score_metrics, so a mixture over "
                "it declares none either"
            )
        return tuple(declared)

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        """Return the wrapped model's observed category codes."""

        return self.model.outcome_codes(study)

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped model's scored observation, unchanged."""

        return self.model.outcomes(study)

    # -- what the component contributes, and what the weight costs ----------------------

    def component_log_density(self, study: Study) -> NDArray[np.float64]:
        """Return the component's log density of each observed outcome.

        ``-inf`` on every row when the study carries no observations at all, which is what
        a prediction-only study is: a process that produced nothing cannot be responsible
        for anything, and no scoring path can run without the outcome anyway.
        """

        if any(column not in study.columns for column in self.scored_columns):
            return np.full(len(study), -np.inf, dtype=np.float64)
        density = np.asarray(
            self.component.pointwise_log_density(study, self.model.outcomes(study)),
            dtype=np.float64,
        )
        if density.shape != (len(study),):
            raise ValueError("a component must return one log density per study row")
        if np.any(np.isnan(density)) or np.any(np.isposinf(density)):
            raise ValueError("component log densities must be finite or -inf")
        return density

    def component_density_on_grid(
        self, study: Study, grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the component's density at every point of the model's own outcome grid.

        A continuous prediction is a tabulation, so a mixture's predicted density needs the
        component's density at values *nobody observed*. No new member is asked for: a
        component's :meth:`~behavio.contracts.mixture.MixtureComponent.pointwise_log_density`
        is already a function of the study and of an outcome per row, so evaluating it with
        one grid point broadcast across the rows is exactly the question, asked once per grid
        point. That is why the loop is over the grid rather than over the rows -- the grid is
        a few hundred points and the rows may be a few hundred thousand.

        The component sees each grid point as the outcome it would be asked about, which is
        what makes a censoring-aware component answer with a density here and with a survival
        probability where the observation actually sat on a limit.
        """

        points = np.asarray(grid, dtype=np.float64)
        n_rows = len(study)
        log_density = np.empty((n_rows, points.size), dtype=np.float64)
        for index, value in enumerate(points):
            column = np.asarray(
                self.component.pointwise_log_density(
                    study, np.full(n_rows, float(value), dtype=np.float64)
                ),
                dtype=np.float64,
            )
            if column.shape != (n_rows,):
                raise ValueError("a component must return one log density per study row")
            log_density[:, index] = column
        if np.any(np.isnan(log_density)) or np.any(np.isposinf(log_density)):
            raise ValueError("component log densities must be finite or -inf")
        return np.asarray(np.exp(log_density), dtype=np.float64)

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the wrapped penalty with an unpenalised row and column for the weight.

        A mixture weight carries no prior. A ridge on a logit is a statement that the
        mixture is probably a particular size, which is exactly the declaration
        ``weight_bounds`` already makes explicitly and which nobody would want made twice.
        """

        inner = self.model.penalty_matrix()
        size = inner.shape[0] + 1
        penalty = np.zeros((size, size), dtype=np.float64)
        penalty[:-1, :-1] = inner
        return penalty

    def coordinate_box(self, study: Study) -> NDArray[np.float64] | None:
        """Return the wrapped box with a finite logit range, or ``None`` for an open one.

        A model estimated without bounds stays without bounds: an unpenalised logit whose
        weight has saturated has a vanishing gradient, so a quasi-Newton search stops on
        its own and the coordinate it stops at is what
        :data:`~behavio.contracts.mixture.MIXTURE_LOGIT_BOUND` is compared against. A model
        whose own solver needs a box gets one for the logit too, because a box with one
        infinite entry is not a box. A bounded-coordinate model always needs one, so on that
        route the second branch is the only one taken.
        """

        box = self.model.coordinate_box(study)
        if box is None:
            return None
        return np.vstack([box, np.asarray([[-MIXTURE_LOGIT_BOUND, MIXTURE_LOGIT_BOUND]])])

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the wrapped starts crossed with a schedule of mixture-weight starts."""

        inner = self.model.initial_points(study)
        count = max(len(inner), self.n_restarts)
        return tuple(
            np.concatenate([inner[index % len(inner)], [LOGIT_STARTS[index % len(LOGIT_STARTS)]]])
            for index in range(count)
        )

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Return what one declared varying name stands for in the mixed coordinate."""

        if name == MIXTURE_LOGIT:
            return (MIXTURE_LOGIT,)
        return tuple(self.model.group_parameter_expansion(name))

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the wrapped prior on the wrapped columns, and a ridge on the weight."""

        selected = np.asarray(columns, dtype=np.intp)
        scale_values = np.asarray(scales, dtype=np.float64)
        n_parameters = len(self.model.parameter_names)
        inner = selected[selected < n_parameters]
        width = len(selected)
        penalty = np.zeros((width, width), dtype=np.float64)
        if len(inner):
            penalty[: len(inner), : len(inner)] = self.model.group_penalty(
                inner, scale_values[: len(inner)]
            )
        if len(inner) < width:
            penalty[len(inner) :, len(inner) :] = np.diag(1.0 / scale_values[len(inner) :] ** 2)
        return penalty

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw wrapped deviations for the wrapped columns and Gaussian ones for the weight."""

        selected = np.asarray(columns, dtype=np.intp)
        scale_values = np.asarray(scales, dtype=np.float64)
        n_parameters = len(self.model.parameter_names)
        inner = selected[selected < n_parameters]
        blocks = []
        if len(inner):
            blocks.append(
                self.model.draw_group_deviations(
                    inner, scale_values[: len(inner)], groups=groups, generator=generator
                )
            )
        if len(inner) < len(selected):
            blocks.append(
                generator.normal(
                    0.0, scale_values[len(inner) :], size=(groups, len(selected) - len(inner))
                )
            )
        return np.asarray(np.hstack(blocks), dtype=np.float64)

    # -- simulation ---------------------------------------------------------------------

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate observations given one mixed parameter vector per row."""

        return self.simulate_rows_with_component(design, coefficients, seed=seed).study

    def simulate_rows_with_component(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> MixtureSimulation:
        """Generate observations and retain which process produced each of them."""

        values = np.asarray(coefficients, dtype=np.float64)
        n_parameters = len(self.model.parameter_names)
        if values.shape != (len(design), n_parameters + 1):
            raise ValueError("simulate_rows needs one value per mixed parameter per study row")
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        study = self.model.simulate_rows(design, values[:, :n_parameters], seed=generator)
        weight = mixture_weight(values[:, n_parameters], self.weight_bounds)
        from_component = generator.binomial(1, weight).astype(np.bool_)
        rows = np.flatnonzero(from_component).astype(np.intp)
        if not len(rows):
            return MixtureSimulation(study=study, from_component=from_component)
        replacements = self.component.simulate_outcomes(study, rows, generator=generator)
        columns = {name: np.array(study[name], copy=True) for name in study.columns}
        for column, drawn in replacements.items():
            if column not in columns:
                raise ValueError(f"a component may only write scored columns, not {column!r}")
            columns[column][rows] = drawn
        return MixtureSimulation(study=Study(columns), from_component=from_component)

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate observations without exposing which process produced each of them."""

        return self.simulate_with_component(design, parameters, seed=seed).study

    def simulate_with_component(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> MixtureSimulation:
        """Generate observations and retain the unobserved process indicator separately."""

        values = self.parameter_vector(parameters)
        rows = np.broadcast_to(values, (len(design), len(values)))
        return self.simulate_rows_with_component(design, rows, seed=seed)

    # -- the reported weight -------------------------------------------------------------

    def weight(self, fit: FitResult) -> float:
        """Return the fitted mixing weight on its own scale."""

        self._validate_fit(fit)
        return float(self.to_natural(fit.estimates)[self.component.weight_name])

    def parameters_from_weight(
        self, parameters: Mapping[str, float], weight: float
    ) -> Mapping[str, float]:
        """Pack wrapped parameters and a natural weight onto the estimated coordinate."""

        return self.from_natural({**dict(parameters), self.component.weight_name: float(weight)})

    def parameter_vector(self, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        """Validate a named parameter mapping and return it in model order."""

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

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:  # pragma: no cover - each route declares its own
        raise NotImplementedError

    def from_natural(
        self, natural: Mapping[str, float]
    ) -> Mapping[str, float]:  # pragma: no cover - each route declares its own
        raise NotImplementedError

    # -- what is wrong with fitting this here -------------------------------------------

    def wrapped_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Forward whatever the wrapped model has to say about this study.

        A combinator that reports only its own findings swallows the wrapped model's, which
        is the same defect ``smooth()`` records in its own ``additional_findings``. A
        discounting model's ``unidentified_discount_rate`` is not less true for the model
        having been mixed with a lapse.
        """

        declared = getattr(self.model, "additional_findings", None)
        return () if declared is None else tuple(declared(study))

    def unreachable_component_finding(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report a component that gives zero density to everything that was observed.

        A component that could not have produced any observation cannot have been
        responsible for one, so the weight is identified only through its cost and will rest
        on the floor of its declared range. The statement is about the component and the
        outcomes and about nothing else, so it means the same thing on both routes and is
        computed the same way on both.
        """

        if any(column not in study.columns for column in self.scored_columns):
            return ()
        try:
            density = self.component_log_density(study)
        except (ModelDataError, ValueError):
            return ()
        if not len(density) or np.any(np.isfinite(density)):
            return ()
        return (
            ModelFinding(
                code="unreachable_mixture_component",
                severity=WARNING,
                message=(
                    f"{self.component.signature} gives zero density to every observed "
                    f"outcome, so {self.component.weight_name} can only be estimated at "
                    "the floor of its declared range"
                ),
            ),
        )

    # -- internals ---------------------------------------------------------------------

    def _natural_input(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        try:
            vector = np.asarray(estimates, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("estimates must contain finite numeric values") from None
        width = len(self.parameter_names)
        if vector.shape != (width,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"estimates must contain {width} finite values")
        return vector

    def _validate_fit(self, fit: FitResult) -> None:
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result was produced by a different model specification")

    def _prediction_mode(self, mode: PredictionMode) -> PredictionMode:
        prediction_mode = PredictionMode(mode)
        if prediction_mode not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                f"{self.model_name} does not support {prediction_mode.value!r} prediction"
            )
        return prediction_mode


# --------------------------------------------------------------------------------------
# The combinator, over a linear predictor
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixtureModel(_MixedModel):
    """A model whose observations may instead have come from a declared simpler process.

    Parameter naming is stable and mechanical: the wrapped model keeps every name it had
    and one coordinate is appended, :data:`~behavio.contracts.mixture.MIXTURE_LOGIT`. The
    reported coordinate replaces that logit with the component's own
    :attr:`~behavio.contracts.mixture.MixtureComponent.weight_name` -- ``lapse_rate`` for a
    guessing process, ``contaminant_rate`` for a distribution over response times -- and a
    fit carries it as a derived quantity with a delta-method standard error.

    This is the penalised-linear route. A model whose likelihood is a
    :class:`~behavio.contracts.bounded.RowObjective` is mixed by :class:`MixtureRowModel`
    instead, and the two differ in one thing: where the weight is carried.
    """

    model: PenalisedLinearEstimator
    component: MixtureComponent
    weight_bounds: tuple[float, float] = (0.0, 0.25)
    n_restarts: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_bounds", self._validate_mixture())

    # -- identity ---------------------------------------------------------------------

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate: the wrapped names plus the weight on its own scale.

        The wrapped names verbatim, because
        :class:`~behavio.contracts.compose.PenalisedLinearEstimator` declares that its
        ``parameter_names`` is "the coordinate the model is estimated, reported and simulated
        in". The bounded-coordinate contract declares the opposite, which is why
        :class:`MixtureRowModel` delegates instead.
        """

        return (*self.model.parameter_names, self.component.weight_name)

    # -- the shape of a mixed problem ---------------------------------------------------

    @property
    def n_model_cells(self) -> int:
        """How many cells the wrapped model's own linear predictor occupies."""

        return max(1, len(self.model.predictor_cells))

    @property
    def scalar_model(self) -> bool:
        """Whether the wrapped model's predictor is one number per row."""

        return not self.model.predictor_cells

    @property
    def predictor_cells(self) -> tuple[str, ...]:
        """The wrapped cells, then the weight, then what the component contributes."""

        model_cells = tuple(self.model.predictor_cells) or ("linear_predictor",)
        prediction = tuple(
            f"component_probability[{index}]" for index in range(self.component.prediction_width)
        )
        return (*model_cells, "mixture_weight", "component_log_density", *prediction)

    @property
    def likelihood(self) -> MixtureLikelihood:
        """The two-component observation model this estimator's cells feed."""

        return MixtureLikelihood(
            model_likelihood=self.model.likelihood,
            n_model_cells=self.n_model_cells,
            scalar_model=self.scalar_model,
            prediction_width=self.component.prediction_width,
            weight_bounds=self.weight_bounds,
        )

    def design_matrix(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped design in its own cells, plus one intercept for the weight."""

        features = validate_predictor_shape(self.model, self.model.design_matrix(study))
        n_parameters = len(self.model.parameter_names)
        design = np.zeros(
            (len(study), len(self.predictor_cells), n_parameters + 1), dtype=np.float64
        )
        if self.scalar_model:
            design[:, 0, :n_parameters] = features
        else:
            design[:, : self.n_model_cells, :n_parameters] = features
        design[:, self.n_model_cells, n_parameters] = 1.0
        return design

    def predictor_offsets(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped offsets, plus what the component contributes per row.

        Two of the offsets are unusual and deliberately so. The component's log density is
        a function of the observation as well as of the trial, which is wider than an
        offset had needed to be; and its predicted probability is not a term of any
        likelihood at all, only of a prediction. Both travel here because the linear
        predictor is the one per-row channel every combinator preserves -- an outer
        combinator slices it by group and multiplies it by a temporal basis without ever
        having to know that either of these is what it is carrying.
        """

        offsets = np.zeros((len(study), len(self.predictor_cells)), dtype=np.float64)
        inner = self.model.predictor_offsets(study)
        if inner is not None:
            values = np.asarray(inner, dtype=np.float64)
            if self.scalar_model:
                offsets[:, 0] = values
            else:
                offsets[:, : self.n_model_cells] = values
        offsets[:, self.n_model_cells + 1] = self.component_log_density(study)
        probability = np.asarray(self.component.prediction_probability(study), dtype=np.float64)
        offsets[:, self.n_model_cells + 2 :] = probability.reshape(
            len(study), self.component.prediction_width
        )
        return offsets

    def fit_penalised(
        self,
        design: PenalisedDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a mixed penalised problem with the wrapped model's own optimizer settings."""

        return self.model.fit_penalised(
            design, model_name=model_name, model_signature=model_signature
        )

    # -- the estimator contract --------------------------------------------------------

    def fit(self, study: Study) -> FitResult:
        """Fit the wrapped model and the mixing weight jointly."""

        result = self.fit_penalised(
            PenalisedDesign(
                parameter_names=self.parameter_names,
                design_matrix=self.design_matrix(study),
                outcomes=self.outcomes(study),
                penalty_matrix=self.penalty_matrix(),
                likelihood=self.likelihood,
                offsets=self.predictor_offsets(study),
                box=self.coordinate_box(study),
                initial_points=self.initial_points(study),
            ),
            model_name=self.model_name,
            model_signature=self.signature,
        )
        return replace(
            result,
            derived=(
                *result.derived,
                *natural_quantities(self, result.estimates, result.covariance),
            ),
            diagnostics=replace(
                result.diagnostics,
                boundary_estimate=bool(
                    result.diagnostics.boundary_estimate
                    or abs(float(result.estimates[-1])) >= MIXTURE_LOGIT_BOUND
                ),
            ),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction:
        """Return the weighted average of the model's and the component's predictions."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        return self.likelihood.prediction(self.row_predictor(study, fit), mode=prediction_mode)

    def row_predictor(self, study: Study, fit: FitResult) -> NDArray[np.float64]:
        """Return the mixed linear predictor of each row under a fitted coordinate."""

        self._validate_fit(fit)
        return linear_predictor(
            self.design_matrix(study), fit.estimates, self.predictor_offsets(study)
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observation under the fitted mixture."""

        self._prediction_mode(mode)
        return self.likelihood.pointwise_log_prob(
            self.row_predictor(study, fit), self.outcomes(study)
        )

    def component_responsibility(self, study: Study, fit: FitResult) -> NDArray[np.float64]:
        """Return the fitted posterior probability that each row came from the component.

        A mixture assigns responsibility, not membership: a fast response is *evidence for*
        the contaminant, not a contaminant. This is the quantity a diagnostic plot should
        show and the one an exclusion rule, if anybody insists on writing one, should be
        built from.
        """

        return protected_array(
            self.likelihood.responsibilities(self.row_predictor(study, fit), self.outcomes(study)),
            dtype=np.float64,
        )

    # -- the reported coordinate --------------------------------------------------------

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Replace the mixture logit with the weight it stands for."""

        values = self._natural_input(estimates)
        natural = dict(zip(self.model.parameter_names, values[:-1].tolist(), strict=True))
        natural[self.component.weight_name] = float(
            mixture_weight(float(values[-1]), self.weight_bounds)
        )
        return MappingProxyType(natural)

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Encode a complete natural mapping onto the estimated coordinate."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        values = {name: float(natural[name]) for name in self.model.parameter_names}
        values[MIXTURE_LOGIT] = mixture_logit(
            natural[self.component.weight_name], self.weight_bounds
        )
        return MappingProxyType(values)

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return the derivative of the reported coordinate, which is diagonal."""

        values = self._natural_input(estimates)
        size = len(values)
        jacobian = np.eye(size, dtype=np.float64)
        lower, upper = self.weight_bounds
        relative = float(expit(values[-1]))
        jacobian[-1, -1] = (upper - lower) * relative * (1.0 - relative)
        return jacobian

    # -- what is wrong with fitting this here -------------------------------------------

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the two ways a declared mixture is not identified by a declared design.

        A mixture weight is estimated from the *shape* of the model's prediction, not from
        its level: a guessing process pulls every prediction towards the same place, so a
        model whose prediction is the same on every row cannot be told apart from a smaller
        version of itself mixed with more guessing. That is the classic trade-off between a
        lapse rate and a shallow slope taken to its limit, and at the limit it is exact
        rather than merely awkward -- which is why it is worth reporting before a fit
        returns a confident number for both. On this route the question is *exact*: a design
        matrix that does not distinguish two rows gives them the same prediction at every
        coordinate, not merely at the ones a search would visit.

        The second failure is the mirror image, and it is
        :meth:`~_MixedModel.unreachable_component_finding`: a component that gives zero
        density to every observation cannot have produced any of them.
        """

        try:
            design = np.asarray(self.model.design_matrix(study), dtype=np.float64)
        except (ModelDataError, ValueError):
            return ()
        findings: list[ModelFinding] = []
        if len(study) and not _varies_across_rows(design):
            findings.append(
                ModelFinding(
                    code="unidentified_mixture",
                    severity=WARNING,
                    message=(
                        f"{self.component.weight_name} is not identified by this design: the "
                        "model predicts the same thing on every row, so any weight can be "
                        "traded against the model's own parameters without changing the fit"
                    ),
                )
            )
        findings.extend(self.unreachable_component_finding(study))
        return (*findings, *self.wrapped_findings(study))


# --------------------------------------------------------------------------------------
# The combinator, over a row objective
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixtureRowModel(_MixedModel):
    """A mixture over a model whose likelihood is one density per row, not one predictor.

    Everything :class:`MixtureModel` says about naming, the declared weight range and the
    reported weight holds here word for word. What differs is the channel the weight travels
    down, and it differs because there is no linear predictor to put a cell on.

    On the penalised-linear route the mixture logit is a **cell of the linear predictor**,
    which is what makes a mixture survive being sliced by group or multiplied by a temporal
    basis: an outer combinator rewrites the coordinate and the predictor follows. A row
    objective has no predictor, but it has the thing the predictor was standing in for --
    **one coordinate vector per row** -- so the mixture logit is one extra *column* of it.
    That is the same property in the other contract's vocabulary, and it is why
    ``hierarchical(mix(model), parameters=("mixture_logit",))`` is a per-subject lapse rate
    and ``smooth(mix(model), parameters=("mixture_logit",))`` is a drifting one, with no
    combinator changed and none of them aware that a mixture is what they are widening.

    The component's log density has nowhere to ride, because there is no offset either. It
    does not need one: it is a function of the study and of the observed outcome and of no
    parameter at all, so :class:`_MixtureRowObjective` computes it once when the objective is
    built. An offset was always the *representation* of "a per-row term no parameter
    multiplies"; here that term is simply held.
    """

    model: BoundedCoordinateEstimator
    component: MixtureComponent
    weight_bounds: tuple[float, float] = (0.0, 0.25)
    n_restarts: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_bounds", self._validate_mixture())

    # -- the shape of a mixed problem ---------------------------------------------------

    @property
    def n_model_parameters(self) -> int:
        """How many columns of a row's coordinate belong to the wrapped model."""

        return len(self.model.parameter_names)

    def row_objective(self, study: Study) -> _MixtureRowObjective:
        """Return this study's mixed negative log likelihood, in one coordinate per row.

        The row-independence check is here rather than in :func:`mix` because this is the
        first point at which a study exists, and ``row_blocks`` is a statement about a study.
        A model that declared nothing and turns out to recurse is refused here.
        """

        inner = self.model.row_objective(study)
        require_independent_rows(inner, model_name=self.model.model_name, combinator="mix")
        return _MixtureRowObjective(
            model=self,
            study=study,
            inner=inner,
            component_density=self.component_log_density(study),
        )

    def fit_rows(
        self,
        design: RowCoefficientDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a mixed row-coefficient problem with the wrapped model's own solver."""

        return self.model.fit_rows(design, model_name=model_name, model_signature=model_signature)

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> ModelPrediction:
        """Return the weighted average of the two processes' predictions, per row.

        Two shapes of prediction and therefore two averages. A model that predicts a
        probability is blended against the one number the component predicts; a model that
        predicts a **tabulated density** -- which is what a family whose outcome is a
        duration returns -- is blended against the component's density on that model's own
        grid. Both are the same weighted average of what the two processes say about a row;
        they differ only in how many numbers a row's prediction is.
        """

        values = self._row_coordinates(study, coefficients)
        width = self.n_model_parameters
        model_prediction = self.model.predict_rows(study, values[:, :width], mode=mode)
        weight = mixture_weight(values[:, width], self.weight_bounds)
        if isinstance(model_prediction, DensityPrediction):
            return blended_density(
                model_prediction,
                weight,
                self.component_density_on_grid(study, model_prediction.grid),
            )
        probability = np.asarray(
            self.component.prediction_probability(study), dtype=np.float64
        ).reshape(len(study), self.component.prediction_width)
        return blended_prediction(model_prediction, weight, probability)

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the mixed log density of each observation, per row coordinate."""

        return self._row_parts(study, self._row_coordinates(study, coefficients)).log_density

    # -- the estimator contract --------------------------------------------------------

    def fit(self, study: Study) -> FitResult:
        """Fit the wrapped model and the mixing weight jointly, on the model's own solver."""

        objective = self.row_objective(study)
        result = self.fit_rows(
            RowCoefficientDesign(
                parameter_names=self.parameter_names,
                objective=objective,
                expand=lambda joint: np.tile(joint, (objective.n_rows, 1)),
                contract=lambda gradient: np.asarray(gradient, dtype=np.float64).sum(axis=0),
                penalty_matrix=self.penalty_matrix(),
                box=self.coordinate_box(study),
                initial_points=self.initial_points(study),
            ),
            model_name=self.model_name,
            model_signature=self.signature,
        )
        return replace(
            result,
            derived=(
                *result.derived,
                *natural_quantities(self, result.estimates, result.covariance),
            ),
            diagnostics=replace(
                result.diagnostics,
                boundary_estimate=bool(
                    result.diagnostics.boundary_estimate
                    or abs(float(result.estimates[-1])) >= MIXTURE_LOGIT_BOUND
                ),
            ),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction:
        """Return the weighted average of the model's and the component's predictions."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        return self.predict_rows(study, self._fitted_rows(study, fit), mode=prediction_mode)

    @property
    def density_outcome(self) -> str:
        """The continuous column a mixed density prediction tabulates, when there is one.

        Raises rather than returning a placeholder when the wrapped model predicts no
        continuous outcome, because that is what makes ``isinstance(model,
        DensityBehaviourEstimator)`` answer the question it is asked: a mixture over a
        logistic GLM has no density to name, and saying so by absence is how every other
        model on that protocol says it.
        """

        declared = getattr(self.model, "density_outcome", None)
        if declared is None:
            raise AttributeError(
                f"{type(self.model).__name__} predicts no continuous outcome, so a mixture "
                "over it tabulates no density either"
            )
        return str(declared)

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        """Return the mixed predictive density, which is what ``predict`` already returns.

        One object rather than two tabulations, exactly as the wrapped families arrange it:
        there are no two halves of a mixed prediction to disagree with one another.
        """

        declared = self.density_outcome
        prediction = self.predict(study, fit, mode=mode)
        if not isinstance(prediction, DensityPrediction) or prediction.outcome != declared:
            raise TypeError(
                f"{self.model.model_name} declares a density over {declared!r} but predicted "
                f"a {type(prediction).__name__}"
            )
        return prediction

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score each observation under the fitted mixture."""

        self._prediction_mode(mode)
        self._validate_fit(fit)
        return self.pointwise_log_prob_rows(study, self._fitted_rows(study, fit))

    def component_responsibility(self, study: Study, fit: FitResult) -> NDArray[np.float64]:
        """Return the fitted posterior probability that each row came from the component.

        A mixture assigns responsibility, not membership: an implausible choice is *evidence
        for* a lapse, not a lapse.
        """

        self._validate_fit(fit)
        parts = self._row_parts(study, self._fitted_rows(study, fit))
        return protected_array(parts.responsibility, dtype=np.float64)

    # -- the reported coordinate --------------------------------------------------------

    @property
    def natural_names(self) -> tuple[str, ...]:
        """The wrapped model's *reported* names, plus the weight on its own scale.

        Delegated rather than copied, and that is the one place the two routes genuinely
        disagree. A bounded-coordinate model's ``parameter_names`` is by contract an
        unconstrained transform of what it reports -- ``discount_rate_log``, not
        ``discount_rate`` -- so appending a weight to the *estimated* names would report a
        logarithm under the name of the thing it is the logarithm of.
        """

        return (*self._wrapped_natural_names, self.component.weight_name)

    def to_natural(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> Mapping[str, float]:
        """Map the wrapped coordinate through its own transform and the logit through ours."""

        values = self._natural_input(estimates)
        declared = getattr(self.model, "to_natural", None)
        inner = (
            dict(declared(values[:-1]))
            if declared is not None
            else dict(zip(self.model.parameter_names, values[:-1].tolist(), strict=True))
        )
        inner[self.component.weight_name] = float(
            mixture_weight(float(values[-1]), self.weight_bounds)
        )
        return MappingProxyType(inner)

    def from_natural(self, natural: Mapping[str, float]) -> Mapping[str, float]:
        """Encode a complete natural mapping onto the estimated coordinate."""

        if not isinstance(natural, Mapping) or set(natural) != set(self.natural_names):
            raise ValueError("natural parameters must match the model exactly")
        wrapped = {name: natural[name] for name in natural if name != self.component.weight_name}
        declared = getattr(self.model, "from_natural", None)
        values = (
            dict(declared(wrapped))
            if declared is not None
            else {name: float(value) for name, value in wrapped.items()}
        )
        values[MIXTURE_LOGIT] = mixture_logit(
            natural[self.component.weight_name], self.weight_bounds
        )
        return MappingProxyType(values)

    def natural_jacobian(
        self, estimates: Sequence[float] | NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Return the wrapped model's own Jacobian block, with the weight's below it."""

        values = self._natural_input(estimates)
        declared = getattr(self.model, "natural_jacobian", None)
        block = (
            np.asarray(declared(values[:-1]), dtype=np.float64)
            if declared is not None
            else np.eye(len(values) - 1, dtype=np.float64)
        )
        jacobian = np.zeros((block.shape[0] + 1, len(values)), dtype=np.float64)
        jacobian[: block.shape[0], : block.shape[1]] = block
        lower, upper = self.weight_bounds
        relative = float(expit(values[-1]))
        jacobian[-1, -1] = (upper - lower) * relative * (1.0 - relative)
        return jacobian

    # -- what is wrong with fitting this here -------------------------------------------

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the mixture hazards this route can see, and no more than it can see.

        ``unreachable_mixture_component`` carries over exactly: it is a statement about the
        component and the observed outcomes, and neither of those knows which contract the
        model satisfies.

        ``unidentified_mixture`` carries over in **meaning** and not in proof. On the
        penalised route it is read off the design matrix, so "the model predicts the same
        thing on every row" holds at every coordinate. There is no design matrix here, and
        the model's prediction is a nonlinear function of the coordinate, so what is checked
        instead is the prediction at each of the deterministic restarts the model's own
        solver would use. A design that does not distinguish two rows gives them the same
        prediction at every one of those, so the hazard is still caught; a coordinate at
        which a varying design happens to look flat would be reported without being
        degenerate, which is why the message says where it looked.
        """

        findings = list(self.unreachable_component_finding(study))
        try:
            starts = self.model.initial_points(study)
        except (ModelDataError, ValueError):
            return (*findings, *self.wrapped_findings(study))
        if len(study) and starts and self._prediction_is_flat_at(study, starts):
            findings.append(
                ModelFinding(
                    code="unidentified_mixture",
                    severity=WARNING,
                    message=(
                        f"{self.component.weight_name} is not identified by this design: at "
                        f"every coordinate {self.model.model_name} would start its own search "
                        "from, it predicts the same thing on every row, so any weight can be "
                        "traded against the model's own parameters without changing the fit"
                    ),
                )
            )
        return (*findings, *self.wrapped_findings(study))

    # -- internals ---------------------------------------------------------------------

    @property
    def _wrapped_natural_names(self) -> tuple[str, ...]:
        return tuple(getattr(self.model, "natural_names", None) or self.model.parameter_names)

    def _fitted_rows(self, study: Study, fit: FitResult) -> NDArray[np.float64]:
        """Repeat one fitted coordinate over the study's rows."""

        return np.tile(np.asarray(fit.estimates, dtype=np.float64), (len(study), 1))

    def _row_coordinates(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return validated_row_coefficients(
            coefficients,
            n_rows=len(study),
            n_parameters=len(self.parameter_names),
            what="row coefficients",
        )

    def _row_parts(self, study: Study, values: NDArray[np.float64]) -> _MixtureParts:
        width = self.n_model_parameters
        model_density = np.asarray(
            self.model.pointwise_log_prob_rows(study, values[:, :width]), dtype=np.float64
        )
        return mixture_parts(
            model_density,
            self.component_log_density(study),
            values[:, width],
            self.weight_bounds,
        )

    def _prediction_is_flat_at(self, study: Study, starts: tuple[NDArray[np.float64], ...]) -> bool:
        for point in starts:
            rows = np.tile(np.asarray(point, dtype=np.float64), (len(study), 1))
            try:
                prediction = self.model.predict_rows(
                    study, rows, mode=self.supported_prediction_modes[0]
                )
            except (ModelDataError, ValueError, UnsupportedPredictionMode, IndexError):
                return False
            values = prediction_values(prediction)
            if _varies_across_rows(values.reshape(len(study), -1)):
                return False
        return True


@dataclass(frozen=True, slots=True)
class _MixtureRowObjective:
    """Two densities per row, combined by a weight that is one column of the coordinate.

    The row-objective twin of :class:`MixtureLikelihood`, and one member narrower because a
    :class:`~behavio.contracts.bounded.RowObjective` is one member wide: a solver on this
    route differences the curvature it needs out of the analytic gradient, so there is no
    second derivative to write.

    Two evaluations of the wrapped model go into one of these. ``pointwise_log_prob_rows``
    supplies the per-row log density the average is taken over, and ``value_and_gradient``
    supplies the per-row gradient the responsibility scales -- and the second is a per-row
    gradient rather than a block one precisely because :func:`mix` admits only models whose
    rows are their own blocks.
    """

    model: MixtureRowModel
    study: Study
    inner: RowObjective
    component_density: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        """The number of scored rows."""

        return int(self.inner.n_rows)

    @property
    def n_parameters(self) -> int:
        """The width of one row's mixed coordinate."""

        return len(self.model.parameter_names)

    @property
    def row_blocks(self) -> NDArray[np.intp]:
        """Every row is its own block, which :func:`mix` refused to proceed without."""

        return np.arange(self.n_rows, dtype=np.intp)

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the mixed negative log likelihood and its gradient in the mixed rows.

        Analytic throughout, and the same two lines the penalised route's gradient is: the
        wrapped model's own gradient scaled by the posterior probability that the row came
        from the model, and the weight's own derivative through the logit. Neither is a
        finite difference, and a mixture of two differentiable densities has no reason to
        become one.
        """

        values = validated_row_coefficients(
            rows, n_rows=self.n_rows, n_parameters=self.n_parameters, what="row coordinates"
        )
        width = self.model.n_model_parameters
        inner_rows = values[:, :width]
        model_density = np.asarray(
            self.model.model.pointwise_log_prob_rows(self.study, inner_rows), dtype=np.float64
        )
        _, inner_gradient = self.inner.value_and_gradient(inner_rows)
        parts = mixture_parts(
            model_density, self.component_density, values[:, width], self.model.weight_bounds
        )
        gradient = np.zeros_like(values)
        gradient[:, :width] = parts.model_responsibility[:, None] * np.asarray(
            inner_gradient, dtype=np.float64
        )
        gradient[:, width] = -(parts.upper - parts.lower) * parts.weight_slope
        return float(-np.sum(parts.log_density)), gradient


def _varies_across_rows(design: NDArray[np.float64]) -> bool:
    """Whether any column of a design distinguishes one row from another."""

    flattened = design.reshape(design.shape[0], -1)
    spread = np.max(flattened, axis=0) - np.min(flattened, axis=0)
    return bool(np.any(np.abs(spread) > 1e-12))
