"""What a *bounded-coordinate* model must expose for hierarchy and smoothness.

:class:`~behavio.contracts.compose.PenalisedLinearEstimator` describes one family of models:
the ones whose likelihood sees a study only through a quadratically penalised **linear
predictor**. Three families in Behavio are not in it and never will be. A Q-learning agent's
likelihood is a recursion over trials; a psychometric curve's is a nonlinear link with two
bounded rates in it; neither has a design matrix a combinator could widen.

They are still composable, and the reason is already written down elsewhere in the package.
:meth:`~behavio.contracts.compose.PenalisedLinearEstimator.simulate_rows` takes **one
coefficient vector per row** and says, in its docstring, that this is "the single
representation that both hierarchy (coefficients differ by group) and smoothness
(coefficients differ by clock value) collapse to". That is true of the *likelihood* as well
as of the simulator, and it is the whole content of this module: a model that can score a
study given one coordinate vector per row can be made hierarchical or smooth, whether or not
those coordinates enter through a linear predictor.

The smaller core the two contracts share
----------------------------------------
Eight of this protocol's members are also members of ``PenalisedLinearEstimator``, with the
same names and the same meanings: ``parameter_names``, ``penalty_matrix``,
``coordinate_box``, ``initial_points``, ``group_parameter_expansion``, ``group_penalty``,
``draw_group_deviations`` and ``simulate_rows``. A ninth is the model's own solver under a
different name -- ``fit_rows`` for ``fit_penalised`` -- and it is there for the same reason:
a composed fit should be the fit the model would have run itself on a wider problem rather
than a re-implementation of it. Everything hierarchy and smoothness do to a coordinate --
select columns, add a Gaussian block, repeat a value over knots, expand a box, name a joint
vector -- is shared verbatim, because none of it ever looks at a likelihood.

What differs is exactly the objective, and it differs in one member.
``PenalisedLinearEstimator`` supplies ``design_matrix`` + ``likelihood``, from which a
combinator forms ``likelihood(X theta + offset, y)``. A bounded-coordinate model supplies
:meth:`BoundedCoordinateEstimator.row_objective`, from which a combinator forms
``objective(rows)`` directly. Both combinators therefore keep one implementation of
*everything except which of two problem objects they hand to which of two solvers*, and both
solvers are still the model's own.

Why the coordinate must be unconstrained
----------------------------------------
Hierarchy puts a **Gaussian** deviation on a group's copy of a parameter. A learning rate
lives in ``(0, 1)`` and a lapse rate in ``(0, maximum_lapse)``; a Gaussian deviation on
either is wrong in a way no amount of shrinkage repairs, because it puts mass outside the
parameter's own range. The repair is not to truncate the prior but to place it where the
parameter is unbounded, which every one of these models already does: ``learning_rate_logit``,
``inverse_temperature_log``, ``log_width`` and ``lapse_logit`` *are* ``parameter_names``, and
the natural coordinate is reached through
:class:`~behavio.contracts.natural.NaturalParameterisation` or a
:class:`~behavio.inference.parameters.ParameterSpace`. So this contract asks for nothing new:
it asks that the coordinate the combinator sees be the one the optimizer already searched.

``coordinate_box`` is where a model *says* that, and it is why the member is required here
rather than optional as it is on the penalised-linear side, where an unconstrained log-odds
coordinate answers ``None``. A finite box on the transformed coordinate is a model stating
where its transform was applied and how far out the data can no longer tell one value from
another; the composed solver reports a coordinate resting on it as a boundary estimate,
which is how a group deviation that has saturated a rate becomes visible.

Rows that share a coordinate
----------------------------
A psychometric curve scores each row independently, so every row may carry its own
coordinate. A value-updating agent may not: a trial's choice probability depends on a value
trace every earlier trial in its session wrote to, so "the learning rate on trial 40" is only
meaningful relative to a block over which the recursion runs. :attr:`RowObjective.row_blocks`
names those blocks, and a row objective refuses coordinates that are not constant within one.
That refusal is what makes ``smooth(BinaryQLearning(), over="cumulative_trial")`` an error
rather than a number: the clock has to be constant within a reset block, because between two
trials of one session there is no point at which the agent's parameters could change without
the value trace being ambiguous about which of them produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from behavio.contracts.compose import GroupExpansion, require_penalised_linear
from behavio.contracts.estimator import FitResult, ModelPrediction, PredictionMode
from behavio.trials import Study

__all__ = [
    "BOUNDED_COORDINATE",
    "PENALISED_LINEAR",
    "BoundedCoordinateEstimator",
    "RowCoefficientDesign",
    "RowObjective",
    "block_constant_coordinates",
    "contract_group_rows",
    "expand_group_rows",
    "group_blocks_respect",
    "require_bounded_coordinate",
    "require_composable",
    "uses_row_coefficients",
    "validated_row_coefficients",
]

PENALISED_LINEAR = "penalised-linear"
"""The composition route through a design matrix and a linear-predictor likelihood."""

BOUNDED_COORDINATE = "bounded-coordinate"
"""The composition route through one unconstrained coordinate vector per row."""


# --------------------------------------------------------------------------------------
# The objective, seen as a function of one coordinate vector per row
# --------------------------------------------------------------------------------------


@runtime_checkable
class RowObjective(Protocol):
    """A negative log likelihood in one coordinate vector per study row.

    This is the bounded-coordinate counterpart of
    :class:`~behavio.contracts.compose.LinearPredictorLikelihood`, and it is deliberately
    one method wide. A combinator's whole contribution is a *linear* map from the joint
    coordinate it invented to the ``(rows, parameters)`` array a model can score, so the
    chain rule is a contraction over rows and nothing else: no combinator ever needs a
    second derivative, and no model ever needs to know which combinator called it.
    """

    @property
    def n_rows(self) -> int:
        """The number of scored rows, which is ``len(study)``."""
        ...

    @property
    def n_parameters(self) -> int:
        """The width of one row's coordinate, matching the model's ``parameter_names``."""
        ...

    @property
    def row_blocks(self) -> NDArray[np.intp]:
        """One block index per row, within which the coordinate must be constant.

        ``arange(n_rows)`` for a model whose rows are independent. A model with a recursion
        answers with the blocks the recursion runs over -- a subject's session, for the
        agents here -- because a parameter that changed inside one would leave the value
        trace ambiguous about which value of it produced which part of the trace.
        """
        ...

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in ``rows``.

        ``rows`` is ``(n_rows, n_parameters)`` and the gradient has the same shape. Where a
        block spans several rows only the *sum* of the block's gradient rows is defined --
        the rows share one coordinate, so nothing distinguishes them -- and an
        implementation may distribute the block's gradient over its rows however it likes.
        Every map a combinator builds is constant within a block, so every contraction sums
        over one.
        """
        ...


# --------------------------------------------------------------------------------------
# The problem a combinator hands back to the model's own solver
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowCoefficientDesign:
    """One model's fit reduced to the pieces a combinator can rewrite.

    The bounded-coordinate twin of :class:`~behavio.contracts.compose.PenalisedDesign`, and
    it carries the same five things for the same five reasons: an objective, a quadratic
    penalty, a box, the starting vectors a multi-start solver would have used, and -- when a
    combinator invented group blocks -- the :class:`~behavio.contracts.compose.GroupExpansion`
    that says how the joint coordinate was built and the ``derived_estimates`` map that names
    population-plus-deviation, which is not a coordinate of the vector the optimizer returns.

    ``expand`` and ``contract`` are the combinator's linear map and its transpose:
    ``expand(joint)`` is the ``(rows, model parameters)`` array the objective scores, and
    ``contract(row_gradient)`` pulls a gradient in that array back to the joint coordinate.
    Keeping them as functions rather than as a matrix is what lets one hierarchical fit over
    forty subjects avoid ever forming a ``(rows, joint)`` array that is mostly zeros.
    """

    parameter_names: tuple[str, ...]
    objective: RowObjective
    expand: Callable[[NDArray[np.float64]], NDArray[np.float64]] = field(compare=False)
    contract: Callable[[NDArray[np.float64]], NDArray[np.float64]] = field(compare=False)
    penalty_matrix: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    box: NDArray[np.float64] | None = None
    initial_points: tuple[NDArray[np.float64], ...] | None = field(default=None, compare=False)
    expansion: GroupExpansion | None = None
    derived_estimates: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = field(
        default=None, compare=False
    )

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("row-coefficient parameter names must be non-empty and unique")
        if not isinstance(self.objective, RowObjective):
            raise TypeError("objective must satisfy behavio.contracts.bounded.RowObjective")
        if not callable(self.expand) or not callable(self.contract):
            raise TypeError("expand and contract must be callables on the joint coordinate")
        penalty = np.asarray(self.penalty_matrix, dtype=np.float64)
        if penalty.shape != (len(names), len(names)):
            raise ValueError("the penalty must be square with one row per parameter")
        if not np.all(np.isfinite(penalty)):
            raise ValueError("the penalty must be finite")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "penalty_matrix", penalty)
        if self.box is not None:
            box = np.asarray(self.box, dtype=np.float64)
            if box.shape != (len(names), 2):
                raise ValueError("the box must contain a lower and an upper bound per parameter")
            if not np.all(np.isfinite(box)) or np.any(box[:, 1] <= box[:, 0]):
                raise ValueError("box bounds must be finite and increasing")
            object.__setattr__(self, "box", box)
        if self.initial_points is not None:
            points = tuple(np.asarray(point, dtype=np.float64) for point in self.initial_points)
            if not points:
                raise ValueError("initial_points must contain at least one starting vector")
            if any(point.shape != (len(names),) for point in points):
                raise ValueError("every initial point must have one value per parameter")
            if any(not np.all(np.isfinite(point)) for point in points):
                raise ValueError("initial points must be finite")
            object.__setattr__(self, "initial_points", points)
        if self.expansion is not None and self.expansion.n_population > len(names):
            raise ValueError("a group expansion must fit inside the joint coordinate")
        if self.derived_estimates is not None and not callable(self.derived_estimates):
            raise TypeError("derived_estimates must be a function of the estimate vector")

    @property
    def n_observations(self) -> int:
        """The number of scored rows."""

        return int(self.objective.n_rows)

    def value_and_gradient(self, joint: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the penalised objective and its gradient on the joint coordinate.

        A solver may use this and nothing else. It is a method rather than a closure built
        by the combinator so that the arithmetic joining the likelihood to the penalty lives
        beside the contract that declares both, and so that a solver reading a
        :class:`RowCoefficientDesign` out of a debugger can evaluate it.
        """

        vector = np.asarray(joint, dtype=np.float64)
        value, row_gradient = self.objective.value_and_gradient(self.expand(vector))
        penalty_term = np.asarray(self.penalty_matrix @ vector, dtype=np.float64)
        gradient = np.asarray(self.contract(row_gradient), dtype=np.float64) + penalty_term
        return float(value) + 0.5 * float(vector @ penalty_term), gradient


# --------------------------------------------------------------------------------------
# The estimator contract
# --------------------------------------------------------------------------------------


@runtime_checkable
class BoundedCoordinateEstimator(Protocol):
    """A model scored from one unconstrained coordinate vector per row.

    Checkable at runtime, so a combinator refuses a model that does not satisfy it instead
    of failing somewhere inside an optimizer. The members that also appear on
    :class:`~behavio.contracts.compose.PenalisedLinearEstimator` mean exactly what they mean
    there and are documented there.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The **unconstrained** coordinate the model is estimated and composed in."""
        ...

    def row_objective(self, study: Study) -> RowObjective:
        """Return this study's negative log likelihood in one coordinate vector per row."""
        ...

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return the quadratic penalty on the parameter coordinate."""
        ...

    def coordinate_box(self, study: Study) -> NDArray[np.float64] | None:
        """Return the finite ``(parameters, 2)`` box the transformed coordinate is searched in."""
        ...

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the deterministic starting vectors this model's solver would use."""
        ...

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Return the reported parameters one declared varying name stands for."""
        ...

    def fit_rows(
        self,
        design: RowCoefficientDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a row-coefficient problem on this model's own optimizer settings."""
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

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> ModelPrediction:
        """Return the point prediction implied by one parameter vector per row.

        A penalised linear model needs no such member because its prediction is a function
        of a linear predictor a combinator can form itself. Nothing outside a recursion can
        form a Q-learning agent's choice probability, so the model is asked.
        """
        ...

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observation under one parameter vector per row."""
        ...

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the Gaussian precision on one group's deviation copy of ``columns``."""
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


def require_bounded_coordinate(model: Any, *, combinator: str) -> BoundedCoordinateEstimator:
    """Return ``model`` if a combinator may rewrite its row-coefficient problem, or raise.

    Three questions, as with :func:`~behavio.contracts.compose.require_penalised_linear`,
    and in the order in which their answers are worth reading. Does the model *declare* that
    it is not one of these; does it have the members; and is the coordinate it reports the
    coordinate it estimates, which a penalty of the wrong width disproves.

    What it cannot check is that the coordinate is genuinely unconstrained, because that is a
    claim about a transform rather than about a member. ``coordinate_box(study)`` is where a
    model makes that claim, and the study it needs is not available here.
    """

    refusal = getattr(model, "bounded_coordinate_refusal", None)
    if refusal:
        raise TypeError(f"{combinator}() cannot be applied to {type(model).__name__}: {refusal}")
    if not isinstance(model, BoundedCoordinateEstimator):
        raise TypeError(
            f"{combinator}() needs a model satisfying behavio.contracts.bounded."
            f"BoundedCoordinateEstimator; {type(model).__name__} does not"
        )
    penalty = model.penalty_matrix()
    if penalty.shape != (len(model.parameter_names),) * 2:
        raise TypeError(
            f"{type(model).__name__} declares {len(model.parameter_names)} parameters but a "
            f"{penalty.shape} penalty, so its estimated coordinate is not the one it reports"
        )
    return model


def require_composable(model: Any, *, combinator: str) -> str:
    """Return which contract ``model`` is composed through, or raise naming both.

    The single check ``smooth()`` and ``hierarchical()`` run. It returns a route rather than
    a model because the two routes differ in exactly one thing -- which problem object goes
    to which solver -- and everything a combinator does on either side of that is the same
    code operating on the same coordinate.

    ``mix()`` deliberately does *not* call this. A mixture combines two densities *given a
    linear predictor*, so it is a penalised-linear operation and stays one; a value-updating
    agent's lapse belongs on its policy, inside the recursion, where it can mix the emitted
    action while leaving the value update to see the action that was taken.
    """

    if uses_row_coefficients(model):
        require_bounded_coordinate(model, combinator=combinator)
        return BOUNDED_COORDINATE
    require_penalised_linear(model, combinator=combinator)
    return PENALISED_LINEAR


def uses_row_coefficients(model: Any) -> bool:
    """Whether a composable model is composed through rows rather than a design matrix.

    Presence of ``likelihood`` is the discriminator rather than an ``isinstance`` call
    against either protocol, and it has to be. A runtime-checkable protocol resolves its
    members with :func:`inspect.getattr_static`, which finds the *descriptor* without
    running it, so a ``SmoothModel`` wrapping a Q-learning agent passes
    ``isinstance(..., PenalisedLinearEstimator)`` on the strength of a ``likelihood``
    property that raises the moment anything reads it. ``hasattr`` reads it, which is what
    makes this the question that decides which branch works -- and it is cheap enough to sit
    inside an EM loop.
    """

    return not hasattr(model, "likelihood") and hasattr(model, "row_objective")


# --------------------------------------------------------------------------------------
# The two maps hierarchy builds, and the block rule both respect
# --------------------------------------------------------------------------------------


def expand_group_rows(
    joint: NDArray[np.float64],
    *,
    n_population: int,
    columns: NDArray[np.intp],
    row_block: NDArray[np.intp],
    n_groups: int,
) -> NDArray[np.float64]:
    """Return each row's population-plus-its-group's-deviation coordinate.

    The row-coefficient counterpart of
    :func:`~behavio.contracts.compose.expand_group_design`, and the same statement about the
    data: grouping partitions rows, every row of a group gets that group's deviation, and a
    parameter not declared varying gets the population value on every row.
    """

    vector = np.asarray(joint, dtype=np.float64)
    deviations = vector[n_population:].reshape(n_groups, len(columns))
    rows = np.tile(vector[:n_population], (len(row_block), 1))
    rows[:, columns] += deviations[row_block]
    return rows


def contract_group_rows(
    row_gradient: NDArray[np.float64],
    *,
    n_population: int,
    columns: NDArray[np.intp],
    row_block: NDArray[np.intp],
    n_groups: int,
) -> NDArray[np.float64]:
    """Pull a gradient in the row coordinates back to the joint coordinate."""

    gradient = np.asarray(row_gradient, dtype=np.float64)
    joint = np.zeros(n_population + n_groups * len(columns), dtype=np.float64)
    joint[:n_population] = gradient.sum(axis=0)
    deviations = np.zeros((n_groups, len(columns)), dtype=np.float64)
    np.add.at(deviations, row_block, gradient[:, columns])
    joint[n_population:] = deviations.reshape(-1)
    return joint


def block_constant_coordinates(
    rows: NDArray[np.float64],
    blocks: NDArray[np.intp],
    *,
    what: str,
) -> NDArray[np.float64]:
    """Return one coordinate vector per block, refusing coordinates that vary inside one.

    The refusal is the point. A combinator that produced coordinates varying within a
    recursion's block has built a model whose value trace cannot say which parameter value
    produced it, and the honest answer is that the composition is not defined rather than a
    number obtained by evaluating the recursion in some arbitrary order.
    """

    values = np.asarray(rows, dtype=np.float64)
    order = np.asarray(blocks, dtype=np.intp)
    if not len(order):
        return np.zeros((0, values.shape[1]), dtype=np.float64)
    n_blocks = int(order.max()) + 1
    representative = np.zeros((n_blocks, values.shape[1]), dtype=np.float64)
    # A later row overwrites an earlier one of the same block, so the representative is the
    # block's last row; the equality check below is what makes "which row" immaterial.
    representative[order] = values
    if not np.array_equal(values, representative[order]):
        raise ValueError(
            f"{what} must be constant within each block of the recursion this model runs "
            "over: a value trace cannot say which of two parameter values produced it"
        )
    return representative


def group_blocks_respect(
    row_block: NDArray[np.intp],
    objective_blocks: NDArray[np.intp],
    *,
    grouping: str,
) -> None:
    """Raise unless every block of the model's recursion lies inside one group.

    A subject's session lies inside that subject, so ``over="subject"`` is admissible for an
    agent that resets by subject and session. ``over="reward_size"`` is not, and the failure
    is worth a sentence rather than a silent average.
    """

    groups = np.asarray(row_block, dtype=np.intp)
    blocks = np.asarray(objective_blocks, dtype=np.intp)
    if not len(blocks):
        return
    order = np.argsort(blocks, kind="stable")
    sorted_blocks = blocks[order]
    boundaries = np.flatnonzero(np.diff(sorted_blocks)) + 1
    for chunk in np.split(order, boundaries):
        if len(np.unique(groups[chunk])) > 1:
            raise ValueError(
                f"grouping by {grouping!r} splits a block this model's likelihood recurses "
                "over, so a group's parameters would have to change part-way through one "
                "value trace; group by a column that is constant within a session"
            )


def validated_row_coefficients(
    coefficients: NDArray[np.float64], *, n_rows: int, n_parameters: int, what: str
) -> NDArray[np.float64]:
    """Return ``coefficients`` as a finite ``(n_rows, n_parameters)`` array, or raise."""

    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (n_rows, n_parameters):
        raise ValueError(f"{what} needs one finite value per parameter per study row")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{what} needs one finite value per parameter per study row")
    return values
