"""What a model must expose for hierarchy and smoothness to be applied generically.

Hierarchy and time-variation are not wrappers. Applying either to a model changes its
parameter vector, its objective, its covariance structure and its simulator at once, so a
combinator cannot be written against :class:`~behavio.contracts.estimator.BehaviourEstimator`
alone: that contract says a model can ``fit``, not *what* it fits. This module names the
extra surface, and it is deliberately narrow -- it describes exactly one family of models,
the ones whose likelihood sees a study only through a **quadratically penalised linear
predictor**. Every generalized linear family in Behavio is one of those; the drift-diffusion
families are not, and the contract says so by refusing them rather than by pretending.

Cells: how wide one row's linear predictor is
---------------------------------------------
A Bernoulli GLM gives each row one number. A multinomial logit gives each row one number
per category, and its likelihood is no less linear for it: what changed is the *shape* of
what the design produces, not the fact that a quadratic penalty is applied to coefficients
that enter linearly. :attr:`PenalisedLinearEstimator.predictor_cells` names that shape.
``()`` is the scalar case and every array keeps the shape it always had; a non-empty tuple
names one **cell** per column of a ``(rows, cells)`` linear predictor, produced by a
``(rows, cells, parameters)`` design.

The alternative was a second contract for categorical models, which would have recreated
the variant explosion this package exists to remove: the multinomial's per-category
structure is expressed in its *parameter names* already, and it needed the *predictor* to
be widened, not the coordinate. Nothing about hierarchy or smoothness is per-family, so
nothing about them is per-shape either: :func:`expand_group_design` gained one ellipsis
and :func:`expand_group_penalty` gained nothing at all.

Support: which cells exist on which rows
----------------------------------------
:meth:`PenalisedLinearEstimator.predictor_offsets` is the one thing a likelihood may read
from a study that is not the design and not the outcome. It is an additive term on the
linear predictor that no parameter multiplies, and ``-inf`` in it declares a cell **outside
the support** on that row -- a task option that was not offered on that trial. Availability
has to reach the likelihood somehow, and an offset is the representation that makes it
family-independent: a combinator carries offsets through untouched, so a group deviation on
an unavailable category contributes exactly zero gradient and zero curvature and is left to
its prior, which is the modelling answer as well as the arithmetic one.

Five ingredients, and which member carries each
-----------------------------------------------

*A penalised objective the combinator can add to.*
    :meth:`PenalisedLinearEstimator.design_matrix`,
    :meth:`~PenalisedLinearEstimator.penalty_matrix`,
    :meth:`~PenalisedLinearEstimator.outcomes`,
    :meth:`~PenalisedLinearEstimator.predictor_offsets`,
    :attr:`~PenalisedLinearEstimator.likelihood` and
    :meth:`~PenalisedLinearEstimator.fit_penalised`. The first five *are* the objective
    -- ``likelihood(X @ theta + offset, y) + 0.5 theta' P theta`` -- and the last is the
    model's own solver, so that a composed fit is bit-for-bit the fit the model would have
    run itself on the expanded problem rather than a re-implementation of it.

*A declaration of which parameters may vary and over what.*
    :class:`VaryingEffects`, resolved against
    :attr:`~PenalisedLinearEstimator.parameter_names`. This is the shape the drift-diffusion
    families already use -- named parameters plus per-parameter scales -- rather than the
    single shared scalar the GLM used to have.

*A per-group block structure.*
    :class:`GroupBlocks` and :func:`group_blocks` on the data side;
    :meth:`~PenalisedLinearEstimator.group_penalty` on the model side. The second is the
    part that is easy to get wrong: a group's deviation is *the same kind of object* as the
    population parameter, so a model whose parameters have internal shape -- a smooth
    coefficient path, whose knots are a path and not a bag of numbers -- must be able to
    give that shape to the deviation too. A combinator that assumed an isotropic ridge
    would silently fit a different, worse model.

*A way to expand and contract the parameter vector.*
    :func:`expand_group_design` and :func:`expand_group_penalty` expand; the combinator
    contracts by slicing the joint estimate. :meth:`~PenalisedLinearEstimator.simulate_rows`
    is the same operation for the simulator: it takes one coefficient vector **per row** on
    the model's own parameter coordinate, which is the single representation that both
    hierarchy (coefficients differ by group) and smoothness (coefficients differ by clock
    value) collapse to.

*A route for the simulator to draw group-level effects.*
    :meth:`~PenalisedLinearEstimator.draw_group_deviations`. It belongs to the model and
    not to the combinator because the correct draw is the prior implied by
    :meth:`~PenalisedLinearEstimator.group_penalty`, which only the model knows the
    structure of.

Composition order
-----------------
Hierarchy is the **outer** combinator. ``hierarchical(smooth(model))`` is a smooth model
made hierarchical; ``smooth(hierarchical(model))`` is refused, because a hierarchical
estimator reports the population coordinate while fitting a joint one that grows with the
number of groups in the study, and an outer combinator cannot expand a coordinate whose
width it does not know until it sees the data. The refusal is a real restriction and is
documented as one; it is not a limitation of this contract so much as a statement that
"which parameters vary by group" is the last question to ask about a model, not the first.

Saying no to the contract
-------------------------
Structural typing answers "does this object have these members", which is not the question.
:class:`~behavio.models.BernoulliGLMHMM` subclasses a GLM for its per-state emissions and so
*inherits* every member below, while its likelihood is a mixture over latent states: its row
scores do not factorise as ``f(eta_r, y_r)``, and no arrangement of members can be inspected
to discover that. A model in that position declares ``penalised_linear_refusal``, a sentence
saying why the mathematics says no, and :func:`require_penalised_linear` -- the single check
both combinators run -- reports it. The declaration is part of this contract's vocabulary
rather than an exception raised from inside a numerical member that a combinator happened
to call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.estimator import (
    FitResult,
    ModelDataError,
    ModelPrediction,
    PredictionMode,
)
from behavio.study import Study

__all__ = [
    "GroupBlocks",
    "LinearPredictorLikelihood",
    "PenalisedDesign",
    "PenalisedLinearEstimator",
    "VaryingEffects",
    "expand_group_design",
    "expand_group_penalty",
    "group_blocks",
    "information_matrix",
    "joint_parameter_names",
    "linear_predictor",
    "parameter_gradient",
    "require_penalised_linear",
    "ridge_group_penalty",
    "validate_predictor_shape",
]


# --------------------------------------------------------------------------------------
# The observation model, seen through one linear predictor per row
# --------------------------------------------------------------------------------------


@runtime_checkable
class LinearPredictorLikelihood(Protocol):
    """A likelihood that reads a study only through the linear predictor of each row.

    Separating this from the estimator is what lets a combinator write a *new* objective --
    the Laplace profile over a group scale, for instance -- without knowing which family it
    is profiling. The four members are the four things any such objective needs: a point
    prediction, a pointwise score, a value with its gradient in the linear predictor, and
    the curvature that turns a design matrix into an observed information matrix.

    A row's linear predictor is one number for a scalar-predictor family and one number per
    cell for a family that declares :attr:`PenalisedLinearEstimator.predictor_cells`, so
    ``linear_predictor`` is ``(rows,)`` or ``(rows, cells)`` and every member follows it:
    :meth:`value_and_gradient` returns a gradient of the same shape, and :meth:`curvature`
    returns ``(rows,)`` second derivatives or ``(rows, cells, cells)`` Hessian blocks.
    :func:`parameter_gradient` and :func:`information_matrix` are the two shape-aware
    contractions that carry either back to the parameter coordinate.
    """

    def prediction(
        self, linear_predictor: NDArray[np.float64], *, mode: PredictionMode
    ) -> ModelPrediction:
        """Return the model's point prediction for these linear predictors."""
        ...

    def pointwise_log_prob(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return one log probability per observation."""
        ...

    def value_and_gradient(
        self, linear_predictor: NDArray[np.float64], outcomes: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the linear predictor."""
        ...

    def curvature(self, linear_predictor: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the per-row second derivative of the negative log likelihood."""
        ...


# --------------------------------------------------------------------------------------
# The penalised problem
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PenalisedDesign:
    """One model's fit reduced to the pieces a combinator can rewrite.

    A combinator builds a wider :class:`PenalisedDesign` from a narrower one and hands it
    back to :meth:`PenalisedLinearEstimator.fit_penalised`, so the arithmetic that solves
    the expanded problem is the model's own.

    ``design_matrix`` is ``(rows, parameters)`` for a scalar predictor and
    ``(rows, cells, parameters)`` for a per-cell one. ``offsets`` matches the linear
    predictor's shape and may contain ``-inf`` to place a cell outside the support of a
    row.

    ``derived_estimates`` is where reporting policy that the joint coordinate cannot
    express is handed back to the model. A hierarchical fit has to know whether
    *population plus deviation* is at a boundary, and that number is not a coordinate of
    the vector the optimizer returns; the combinator supplies it as a function of that
    vector and the model's own solver applies its own boundary convention to the result.
    This is why the contract itself no longer carries a ``boundary_threshold``: every
    number in a :class:`~behavio.contracts.estimator.FitDiagnostics` is now produced by
    the model that owns the convention behind it.
    """

    parameter_names: tuple[str, ...]
    design_matrix: NDArray[np.float64]
    outcomes: NDArray[np.float64]
    penalty_matrix: NDArray[np.float64]
    likelihood: LinearPredictorLikelihood
    offsets: NDArray[np.float64] | None = None
    derived_estimates: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = field(
        default=None, compare=False
    )

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        design = np.asarray(self.design_matrix, dtype=np.float64)
        outcomes = np.asarray(self.outcomes, dtype=np.float64)
        penalty = np.asarray(self.penalty_matrix, dtype=np.float64)
        if not names or len(set(names)) != len(names):
            raise ValueError("penalised design parameter names must be non-empty and unique")
        if design.ndim not in (2, 3) or design.shape[-1] != len(names):
            raise ValueError("the design matrix must have one column per parameter")
        if design.ndim == 3 and design.shape[1] < 1:
            raise ValueError("a per-cell design needs at least one predictor cell")
        if outcomes.ndim != 1 or outcomes.shape[0] != design.shape[0]:
            raise ValueError("outcomes must have one value per design row")
        if penalty.shape != (len(names), len(names)):
            raise ValueError("the penalty must be square with one row per parameter")
        if not isinstance(self.likelihood, LinearPredictorLikelihood):
            raise TypeError("likelihood must satisfy LinearPredictorLikelihood")
        if self.derived_estimates is not None and not callable(self.derived_estimates):
            raise TypeError("derived_estimates must be a function of the estimate vector")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "design_matrix", design)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "penalty_matrix", penalty)
        if self.offsets is None:
            return
        offsets = np.asarray(self.offsets, dtype=np.float64)
        if offsets.shape != self.predictor_shape:
            raise ValueError("offsets must have the shape of the linear predictor")
        if np.any(np.isnan(offsets)) or np.any(np.isposinf(offsets)):
            raise ValueError("offsets may contain finite values or -inf")
        object.__setattr__(self, "offsets", offsets)

    @property
    def n_observations(self) -> int:
        """The number of scored rows."""

        return int(self.design_matrix.shape[0])

    @property
    def n_cells(self) -> int | None:
        """The width of one row's linear predictor, or ``None`` when it is a scalar."""

        return None if self.design_matrix.ndim == 2 else int(self.design_matrix.shape[1])

    @property
    def predictor_shape(self) -> tuple[int, ...]:
        """The shape of this problem's linear predictor."""

        return self.design_matrix.shape[:-1]

    def linear_predictor(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return this design's linear predictor at ``coefficients``, offsets included."""

        return linear_predictor(self.design_matrix, coefficients, self.offsets)


# --------------------------------------------------------------------------------------
# The three shape-aware contractions between a coordinate and a linear predictor
# --------------------------------------------------------------------------------------


def linear_predictor(
    design_matrix: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    offsets: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return ``X theta + offset``, whatever shape one row's predictor has.

    The scalar branch is written as the plain matrix product it has always been rather
    than as a one-cell case of the general one, because a fit that was published before
    predictor cells existed must still be reproducible to the last bit.
    """

    values = (
        design_matrix @ coefficients
        if design_matrix.ndim == 2
        else np.einsum("rcp,p->rc", design_matrix, coefficients)
    )
    return values if offsets is None else values + offsets


def parameter_gradient(
    design_matrix: NDArray[np.float64], gradient: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pull a gradient in the linear predictor back to the parameter coordinate."""

    if design_matrix.ndim == 2:
        return np.asarray(design_matrix.T @ gradient, dtype=np.float64)
    return np.asarray(
        np.einsum("rcp,rc->p", design_matrix, gradient, optimize=True), dtype=np.float64
    )


def information_matrix(
    design_matrix: NDArray[np.float64], curvature: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Turn per-row likelihood curvature into observed information on the coordinate."""

    if design_matrix.ndim == 2:
        return np.asarray(design_matrix.T @ (curvature[:, None] * design_matrix), dtype=np.float64)
    return np.asarray(
        np.einsum("rcp,rcd,rdq->pq", design_matrix, curvature, design_matrix, optimize=True),
        dtype=np.float64,
    )


@runtime_checkable
class PenalisedLinearEstimator(Protocol):
    """A model whose fit is a quadratically penalised linear-predictor problem.

    This is the contract ``behavio.compose.smooth`` and ``behavio.compose.hierarchical``
    are written against. It is checkable at runtime, so a combinator refuses a model that
    does not satisfy it instead of failing somewhere inside the optimizer.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The coordinate the model is estimated, reported and simulated in."""
        ...

    @property
    def predictor_cells(self) -> tuple[str, ...]:
        """Names for the columns of one row's linear predictor, or ``()`` if it is scalar.

        A multinomial logit names its categories here; a Bernoulli GLM declares ``()`` and
        every array it exchanges with a combinator keeps the shape it always had. The
        names are not parameters -- they say what the *predictor* is a predictor of.
        """
        ...

    @property
    def likelihood(self) -> LinearPredictorLikelihood:
        """The observation model this estimator's linear predictor feeds."""
        ...

    def outcomes(self, study: Study) -> NDArray[np.float64]:
        """Return the scored observation, validated, as one value per row."""
        ...

    def design_matrix(self, study: Study) -> NDArray[np.float64]:
        """Return the array whose product with the coordinate is the linear predictor.

        ``(rows, parameters)`` when :attr:`predictor_cells` is empty, and
        ``(rows, cells, parameters)`` when it is not.
        """
        ...

    def predictor_offsets(self, study: Study) -> NDArray[np.float64] | None:
        """Return an additive term on the linear predictor, or ``None`` if there is none.

        ``-inf`` declares a cell outside the support of a row, which is how per-trial
        option availability reaches a likelihood without any combinator having to know
        that availability is what it is looking at.
        """
        ...

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the quadratic penalty on the parameter coordinate."""
        ...

    def fit_penalised(
        self,
        design: PenalisedDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a penalised problem on this model's own optimizer settings."""
        ...

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate observations given one parameter vector per row."""
        ...

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the penalty on one group's deviation copy of ``columns``.

        The default is an isotropic ridge, :func:`ridge_group_penalty`. A model whose
        parameters have internal shape overrides it so that a deviation inherits that
        shape.
        """
        ...

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw ``(groups, len(columns))`` deviations from the prior ``group_penalty`` implies."""
        ...


def require_penalised_linear(model: Any, *, combinator: str) -> PenalisedLinearEstimator:
    """Return ``model`` if a combinator may rewrite its penalised problem, or raise.

    Three questions, in the order in which their answers are worth reading. First, does the
    model *declare* that it is not one of these -- the ``penalised_linear_refusal``
    attribute, an opt-out for a model that inherits every member below from a parent whose
    likelihood it does not share. Second, does it have the members at all. Third, is the
    coordinate it reports the coordinate it estimates, which a penalty of the wrong width
    disproves.

    The refusal is an attribute rather than a protocol member because a contract cannot
    require a model to announce that it does not satisfy it; and it is checked before the
    structural test because the structural test would say yes.
    """

    refusal = getattr(model, "penalised_linear_refusal", None)
    if refusal:
        raise TypeError(f"{combinator}() cannot be applied to {type(model).__name__}: {refusal}")
    if not isinstance(model, PenalisedLinearEstimator):
        raise TypeError(
            f"{combinator}() needs a model satisfying behavio.contracts.compose."
            f"PenalisedLinearEstimator; {type(model).__name__} does not"
        )
    penalty = model.penalty_matrix()
    if penalty.shape != (len(model.parameter_names),) * 2:
        raise TypeError(
            f"{type(model).__name__} declares {len(model.parameter_names)} parameters but a "
            f"{penalty.shape} penalty, so its estimated coordinate is not the one it reports"
        )
    cells = tuple(model.predictor_cells)
    if any(not isinstance(cell, str) or not cell for cell in cells) or len(set(cells)) != len(
        cells
    ):
        raise TypeError("predictor_cells must be unique non-empty names, or empty for a scalar")
    if len(cells) == 1:
        raise TypeError(
            "a linear predictor of one cell is the scalar case: declare predictor_cells=()"
        )
    return model


def validate_predictor_shape(
    model: PenalisedLinearEstimator, design_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``design_matrix`` if it has the shape ``model.predictor_cells`` promises.

    The declaration would be decorative if nothing ever read it. This is what reads it, at
    the point where a combinator first has a study in hand, so a family whose design and
    whose declaration disagree is caught by name rather than by a broadcasting error six
    frames further in.
    """

    cells = tuple(model.predictor_cells)
    expected = 3 if cells else 2
    if design_matrix.ndim != expected:
        raise TypeError(
            f"{type(model).__name__} declares {len(cells)} predictor cells, so its design "
            f"must be {expected}-dimensional; it built a {design_matrix.ndim}-dimensional one"
        )
    if cells and design_matrix.shape[1] != len(cells):
        raise TypeError(
            f"{type(model).__name__} declares predictor cells {cells} but built a design "
            f"with {design_matrix.shape[1]} of them"
        )
    return design_matrix


def ridge_group_penalty(scales: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the isotropic Gaussian precision ``diag(1 / scales**2)``.

    This is the correct group penalty for any model whose parameters are exchangeable
    numbers rather than samples of one structured object, which is every model in Behavio
    except the smooth ones.
    """

    return np.diag(1.0 / np.asarray(scales, dtype=np.float64) ** 2)


def ridge_group_draw(
    scales: NDArray[np.float64], *, groups: int, generator: np.random.Generator
) -> NDArray[np.float64]:
    """Draw independent Gaussian deviations matching :func:`ridge_group_penalty`.

    A uniform scale is passed to the generator as a scalar rather than as a broadcast
    array, because those are two different code paths in NumPy and only the scalar one
    reproduces the draw sequence the hand-written hierarchical GLM published.
    """

    scale_array = np.asarray(scales, dtype=np.float64)
    uniform = bool(np.all(scale_array == scale_array[0]))
    scale: Any = float(scale_array[0]) if uniform else scale_array
    return np.asarray(
        generator.normal(loc=0.0, scale=scale, size=(groups, len(scale_array))),
        dtype=np.float64,
    )


# --------------------------------------------------------------------------------------
# Which parameters vary, and over what
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VaryingEffects:
    """Which of a model's parameters vary by group, and how far they may vary.

    The GLM used to expose one ``subject_scale`` scalar shared by every coefficient, so
    there was no way to say "the bias varies between animals but the stimulus sensitivity
    does not", which is the first thing anyone wants to say. The drift-diffusion families
    already took named parameters plus per-parameter scales, and this is that shape,
    generalised off ``subject``: ``grouping`` is any study column.
    """

    grouping: str
    parameters: tuple[str, ...]
    scales: NDArray[np.float64]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        scales = protected_array(self.scales, dtype=np.float64)
        if not isinstance(self.grouping, str) or not self.grouping:
            raise ValueError("grouping must be a non-empty study column name")
        if not parameters or len(set(parameters)) != len(parameters):
            raise ValueError("varying parameters must be non-empty and unique")
        if scales.shape != (len(parameters),):
            raise ValueError("varying effects need one scale per varying parameter")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("varying-effect scales must be finite and positive")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "scales", scales)

    @classmethod
    def declare(
        cls,
        parameter_names: Sequence[str],
        *,
        over: str,
        parameters: Sequence[str] | None = None,
        scale: float = 0.5,
        parameter_scales: Mapping[str, float] | None = None,
    ) -> VaryingEffects:
        """Resolve a user's declaration against a model's parameter names.

        ``parameters=None`` means every parameter varies, which is what the hand-written
        hierarchical GLM did and the only behaviour it could express. ``scale`` is the
        default prior standard deviation and ``parameter_scales`` overrides it by name.
        """

        available = tuple(parameter_names)
        if parameters is None:
            selected = available
        else:
            if isinstance(parameters, str):
                raise ValueError("parameters must be a sequence of parameter names")
            selected = tuple(parameters)
            unknown = [name for name in selected if name not in available]
            if unknown:
                raise ValueError(
                    f"varying parameters are not parameters of this model: {sorted(unknown)}; "
                    f"available: {list(available)}"
                )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        overrides = dict(parameter_scales or {})
        unknown_scales = [name for name in overrides if name not in selected]
        if unknown_scales:
            raise ValueError(
                f"parameter_scales names non-varying parameters: {sorted(unknown_scales)}"
            )
        scales = np.asarray(
            [float(overrides.get(name, scale)) for name in selected], dtype=np.float64
        )
        return cls(grouping=over, parameters=selected, scales=scales)

    def columns(self, parameter_names: Sequence[str]) -> NDArray[np.intp]:
        """Return the positions of the varying parameters, in model order.

        Model order, not declaration order: the joint design's group block is a slice of
        the model's own design matrix, so the two have to agree column for column.
        """

        available = tuple(parameter_names)
        selected = set(self.parameters)
        return np.asarray(
            [index for index, name in enumerate(available) if name in selected],
            dtype=np.intp,
        )

    def ordered_scales(self, parameter_names: Sequence[str]) -> NDArray[np.float64]:
        """Return the scales in model order, matching :meth:`columns`."""

        by_name = dict(zip(self.parameters, self.scales.tolist(), strict=True))
        return np.asarray(
            [by_name[name] for name in parameter_names if name in by_name], dtype=np.float64
        )

    def ordered_parameters(self, parameter_names: Sequence[str]) -> tuple[str, ...]:
        """Return the varying parameter names in model order."""

        selected = set(self.parameters)
        return tuple(name for name in parameter_names if name in selected)

    def signature(self, parameter_names: Sequence[str]) -> str:
        """A stable fingerprint of the declaration, for model signatures.

        ``varying=all`` and a single ``scale=`` are written where they are exact, because a
        signature is read by people and the wrapped model's own signature -- which sits in
        the same string -- already says what "all" is. Anything less than all parameters,
        or any scale that differs between them, is written out in full.
        """

        available = tuple(parameter_names)
        varying = (
            "all"
            if self.ordered_parameters(available) == available
            else (",".join(self.parameters))
        )
        scales = self.scales.tolist()
        if len(set(scales)) == 1:
            return f"over={self.grouping};varying={varying};scale={scales[0]:g}"
        rendered = ",".join(
            f"{name}:{scale:g}" for name, scale in zip(self.parameters, scales, strict=True)
        )
        return f"over={self.grouping};varying={varying};scales={rendered}"


# --------------------------------------------------------------------------------------
# Group structure on the data side
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupBlocks:
    """The realised grouping of one study's rows: the labels and each row's block."""

    grouping: str
    labels: tuple[Any, ...]
    row_block: NDArray[np.intp]

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        row_block = protected_array(self.row_block, dtype=np.intp)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("group labels must be non-empty and unique")
        if row_block.ndim != 1:
            raise ValueError("row_block must be one block index per study row")
        if len(row_block) and (row_block.min() < 0 or row_block.max() >= len(labels)):
            raise ValueError("row_block must index a declared group label")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "row_block", row_block)

    @property
    def n_groups(self) -> int:
        """The number of distinct groups."""

        return len(self.labels)


def group_blocks(study: Study, grouping: str) -> GroupBlocks:
    """Return the block structure a grouping column induces on a study's rows.

    Labels appear in first-appearance order, which for ``grouping="subject"`` is exactly
    ``Study.subjects``, so a hierarchical fit's group order is the study's subject order.
    """

    if grouping not in study.columns:
        raise ModelDataError(f"study is missing grouping column {grouping!r}")
    values = [_scalar(value) for value in study[grouping]]
    labels: list[Any] = []
    positions: dict[Any, int] = {}
    for value in values:
        key = _label_key(value)
        if key not in positions:
            positions[key] = len(labels)
            labels.append(value)
    row_block = np.asarray([positions[_label_key(value)] for value in values], dtype=np.intp)
    return GroupBlocks(grouping=grouping, labels=tuple(labels), row_block=row_block)


# --------------------------------------------------------------------------------------
# Expanding a penalised problem over groups
# --------------------------------------------------------------------------------------


def expand_group_design(
    design_matrix: NDArray[np.float64],
    blocks: GroupBlocks,
    columns: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Return ``[X | one zero-padded copy of X[..., columns] per group]``.

    The population block keeps every column; each group block keeps only the columns whose
    parameters were declared varying, which is what makes "only the bias varies" a
    structurally smaller problem rather than a prior trick.

    Widening the linear predictor cost this function one ellipsis. Grouping partitions
    *rows*, and a cell axis sits between the row axis and the coordinate axis it never
    touches, so a per-category design is expanded by exactly the same block copy as a
    scalar one -- and every category of a row lands in the same group, which is the
    statement that a subject is a subject on every cell of the trial.
    """

    n_parameters = design_matrix.shape[-1]
    width = len(columns)
    joint = np.zeros(
        (*design_matrix.shape[:-1], n_parameters + blocks.n_groups * width), dtype=np.float64
    )
    joint[..., :n_parameters] = design_matrix
    restricted = design_matrix[..., columns]
    for row, block in enumerate(blocks.row_block):
        start = n_parameters + int(block) * width
        joint[row, ..., start : start + width] = restricted[row]
    return joint


def expand_group_penalty(
    penalty_matrix: NDArray[np.float64],
    blocks: GroupBlocks,
    group_penalty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the block-diagonal penalty pairing :func:`expand_group_design`.

    Unchanged by predictor cells, and necessarily so: a penalty lives on the parameter
    coordinate, which is where a cell axis is not.
    """

    n_parameters = penalty_matrix.shape[0]
    width = group_penalty.shape[0]
    size = n_parameters + blocks.n_groups * width
    joint = np.zeros((size, size), dtype=np.float64)
    joint[:n_parameters, :n_parameters] = penalty_matrix
    for block in range(blocks.n_groups):
        start = n_parameters + block * width
        joint[start : start + width, start : start + width] = group_penalty
    return joint


def joint_parameter_names(
    parameter_names: Sequence[str],
    blocks: GroupBlocks,
    varying: Sequence[str],
) -> tuple[str, ...]:
    """Name the joint coordinate ``expand_group_design`` builds.

    ``population.intercept`` and ``subject[0].intercept`` are the names the hand-written
    hierarchical GLM used for the same columns, and they are kept so that a joint fit read
    out of a debugger reads the same way it always did.
    """

    return tuple(
        [f"population.{name}" for name in parameter_names]
        + [
            f"{blocks.grouping}[{block}].{name}"
            for block in range(blocks.n_groups)
            for name in varying
        ]
    )


def _label_key(value: Any) -> Any:
    return (type(value), value)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
