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
from behavio.contracts.compose import (
    LinearPredictorLikelihood,
    PenalisedDesign,
    PenalisedLinearEstimator,
    linear_predictor,
    require_penalised_linear,
    validate_predictor_shape,
)
from behavio.contracts.estimator import (
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
    require_mixture_component,
    validate_weight_bounds,
)
from behavio.contracts.natural import natural_quantities
from behavio.models._kernels.introspection import WARNING, Describable, ModelFinding
from behavio.trials import Study

__all__ = [
    "MixtureLikelihood",
    "MixtureModel",
    "MixtureSimulation",
    "mix",
]

LOG_DENSITY_FLOOR = float(np.log(np.finfo(np.float64).tiny))
"""The floor a mixture log density is held at when neither process could have produced a row."""

LOGIT_STARTS = (-5.0, -2.0, 1.0)
"""Deterministic mixture-logit restarts, in the order the deleted lapse model used them.

A mixture likelihood is not convex in the weight and the model's parameters jointly -- a
shallow slope with a large weight and a steep slope with a small one are two hills -- so a
mixture is a multi-start problem even when the model it wraps is not. These three are the
starts the deleted ``LapsePsychometric`` searched, and at ``weight_bounds=(0, 0.2)`` they
place the weight at 0.13 %, 2.4 % and 14.6 % of the trials.
"""


def mix(
    model: PenalisedLinearEstimator,
    component: MixtureComponent,
    *,
    weight_bounds: tuple[float, float] = (0.0, 0.25),
    n_restarts: int = 3,
) -> MixtureModel:
    """Return ``model`` mixed with ``component``, estimating the mixing weight.

    Each row's density becomes ``(1 - w) * model + w * component``, with ``w`` estimated as
    one extra coordinate, :data:`~behavio.contracts.mixture.MIXTURE_LOGIT`. ``weight_bounds``
    **declares** the range the weight is estimated inside, and is the general form of the
    ``maximum_lapse`` and ``probability_bounds`` arguments three separate mechanisms used to
    carry; ``(0.0, 0.25)`` says a mixture may account for at most a quarter of the trials.
    Everything about the component is declared and nothing about it is estimated, so a
    mixture adds exactly one parameter whatever it is mixed with.

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
    require_penalised_linear(model, combinator="mix")
    require_mixture_component(component, model, combinator="mix")
    return MixtureModel(
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
        probability = np.asarray(model_prediction.probability, dtype=np.float64)
        weight = self.weights(cells)
        component = cells[:, self.density_cell + 1 :]
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
        component_density = cells[:, self.density_cell]
        logit = cells[:, self.logit_cell]
        lower_bound, upper_bound = self.weight_bounds
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
# The combinator
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixtureModel(Describable):
    """A model whose observations may instead have come from a declared simpler process.

    Parameter naming is stable and mechanical: the wrapped model keeps every name it had
    and one coordinate is appended, :data:`~behavio.contracts.mixture.MIXTURE_LOGIT`. The
    reported coordinate replaces that logit with the component's own
    :attr:`~behavio.contracts.mixture.MixtureComponent.weight_name` -- ``lapse_rate`` for a
    guessing process, ``contaminant_rate`` for a distribution over response times -- and a
    fit carries it as a derived quantity with a delta-method standard error.
    """

    model: PenalisedLinearEstimator
    component: MixtureComponent
    weight_bounds: tuple[float, float] = (0.0, 0.25)
    n_restarts: int = 3

    def __post_init__(self) -> None:
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
        if self.component.weight_name in inner:
            raise ValueError(
                f"the wrapped model already has a parameter named "
                f"{self.component.weight_name!r}, which is what this component calls its weight"
            )
        object.__setattr__(self, "weight_bounds", bounds)

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
    def natural_names(self) -> tuple[str, ...]:
        """The reported coordinate: the wrapped names plus the weight on its own scale."""

        return (*self.model.parameter_names, self.component.weight_name)

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

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        """Return the wrapped model's observed category codes."""

        return self.model.outcome_codes(study)

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
    def outcome_channels(self) -> tuple[str, ...]:
        """The wrapped model's channels: a mixture cannot change what is observed."""

        return tuple(self.model.outcome_channels)

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

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the wrapped model's scored observation, unchanged."""

        return self.model.outcomes(study)

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
        infinite entry is not a box.
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

    # -- what is wrong with fitting this here -------------------------------------------

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the two ways a declared mixture is not identified by a declared design.

        A mixture weight is estimated from the *shape* of the model's prediction, not from
        its level: a guessing process pulls every prediction towards the same place, so a
        model whose prediction is the same on every row cannot be told apart from a smaller
        version of itself mixed with more guessing. That is the classic trade-off between a
        lapse rate and a shallow slope taken to its limit, and at the limit it is exact
        rather than merely awkward -- which is why it is worth reporting before a fit
        returns a confident number for both.

        The second failure is the mirror image: a component that gives zero density to
        every observation cannot have produced any of them, so the weight is identified
        only through its cost and will rest on the floor of its declared range.
        """

        findings: list[ModelFinding] = []
        try:
            design = np.asarray(self.model.design_matrix(study), dtype=np.float64)
        except (ModelDataError, ValueError):
            return ()
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
        if all(column in study.columns for column in self.scored_columns):
            density = self.component_log_density(study)
            if len(density) and not np.any(np.isfinite(density)):
                findings.append(
                    ModelFinding(
                        code="unreachable_mixture_component",
                        severity=WARNING,
                        message=(
                            f"{self.component.signature} gives zero density to every observed "
                            f"outcome, so {self.component.weight_name} can only be estimated at "
                            "the floor of its declared range"
                        ),
                    )
                )
        return tuple(findings)

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


def _varies_across_rows(design: NDArray[np.float64]) -> bool:
    """Whether any column of a design distinguishes one row from another."""

    flattened = design.reshape(design.shape[0], -1)
    spread = np.max(flattened, axis=0) - np.min(flattened, axis=0)
    return bool(np.any(np.abs(spread) > 1e-12))
