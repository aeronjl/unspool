"""Stationary and covariate-dependent Bernoulli GLM-HMMs with label diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit, logsumexp

from behavio._internal.arrays import protected_array
from behavio.contracts.bounded import (
    RowCoefficientDesign,
    block_constant_coordinates,
    validated_row_coefficients,
)
from behavio.contracts.compose import ridge_group_draw, ridge_group_penalty
from behavio.design.matrix import DesignSpec
from behavio.models._kernels.bernoulli import ordered_session_indices
from behavio.models._kernels.curvature import finite_difference_hessian, offset_steps
from behavio.models._kernels.design import build_matrix, resolve_design, validate_design_choice
from behavio.models._kernels.rowfit import solve_row_coefficients
from behavio.models.base import (
    FitDiagnostics,
    FitResult,
    ModelDataError,
    Prediction,
    PredictionMode,
)
from behavio.models.glm import BernoulliHistoryGLM
from behavio.models.state_alignment import LatentStateAlignment, align_latent_states
from behavio.trials import REQUIRED_COLUMNS, Study


@dataclass(frozen=True, slots=True)
class GLMHMMParameters:
    """Natural-scale components of a Bernoulli GLM-HMM parameter vector."""

    initial_probabilities: NDArray[np.float64]
    transition_matrix: NDArray[np.float64]
    emission_coefficients: NDArray[np.float64]
    coefficient_names: tuple[str, ...]
    transition_effects: NDArray[np.float64] | None = None
    transition_predictor_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        initial = protected_array(self.initial_probabilities, dtype=np.float64)
        transition = protected_array(self.transition_matrix, dtype=np.float64)
        emissions = protected_array(self.emission_coefficients, dtype=np.float64)
        names = tuple(self.coefficient_names)
        transition_names = tuple(self.transition_predictor_names)
        if initial.ndim != 1 or len(initial) < 2:
            raise ValueError("initial_probabilities must contain at least two states")
        n_states = len(initial)
        if transition.shape != (n_states, n_states):
            raise ValueError("transition_matrix must have one row and column per state")
        if emissions.shape != (n_states, len(names)) or not names:
            raise ValueError("emission_coefficients must have one named row per state")
        if len(set(names)) != len(names):
            raise ValueError("coefficient_names must be unique")
        if not np.all(np.isfinite(initial)) or np.any(initial <= 0):
            raise ValueError("initial probabilities must be finite and strictly positive")
        if not np.isclose(initial.sum(), 1.0, atol=1e-8):
            raise ValueError("initial probabilities must sum to one")
        if not np.all(np.isfinite(transition)) or np.any(transition <= 0):
            raise ValueError("transition probabilities must be finite and strictly positive")
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("every transition row must sum to one")
        if not np.all(np.isfinite(emissions)):
            raise ValueError("emission coefficients must be finite")
        if len(set(transition_names)) != len(transition_names) or any(
            not isinstance(name, str) or not name for name in transition_names
        ):
            raise ValueError("transition predictor names must be unique non-empty strings")
        if self.transition_effects is None:
            effects = np.zeros((n_states, n_states, len(transition_names)), dtype=np.float64)
        else:
            effects = protected_array(self.transition_effects, dtype=np.float64)
        expected_effects = (n_states, n_states, len(transition_names))
        if effects.shape != expected_effects or not np.all(np.isfinite(effects)):
            raise ValueError(
                "transition_effects must contain one finite destination-logit effect per "
                "source state and transition predictor"
            )
        if not np.allclose(effects.sum(axis=1), 0.0, atol=1e-8):
            raise ValueError(
                "transition effects must sum to zero over destination states; this is the "
                "identified centred-logit representation"
            )
        object.__setattr__(self, "initial_probabilities", initial)
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "emission_coefficients", emissions)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "transition_effects", protected_array(effects, dtype=np.float64))
        object.__setattr__(self, "transition_predictor_names", transition_names)

    @property
    def n_states(self) -> int:
        return len(self.initial_probabilities)

    def transition_probabilities(self, covariates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one transition matrix per row of a transition-covariate design.

        Row ``t`` describes the transition *into* trial ``t``. Session-opening rows have a
        matrix too for a rectangular return value, but filtering correctly ignores it and
        resets to :attr:`initial_probabilities`.
        """

        values = np.asarray(covariates, dtype=np.float64)
        expected = len(self.transition_predictor_names)
        if values.ndim != 2 or values.shape[1] != expected or not np.all(np.isfinite(values)):
            raise ValueError(
                "transition covariates must be a finite matrix with one column per predictor"
            )
        baseline = np.log(np.asarray(self.transition_matrix, dtype=np.float64))
        logits = baseline[None, :, :] + np.einsum(
            "tp,ijp->tij", values, np.asarray(self.transition_effects), optimize=True
        )
        logits -= logsumexp(logits, axis=2, keepdims=True)
        return protected_array(np.exp(logits), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GLMHMMSimulation:
    """A simulated observed study paired with its unexposed latent-state truth."""

    study: Study
    states: NDArray[np.int64]
    n_states: int

    def __post_init__(self) -> None:
        states = protected_array(self.states, dtype=np.int64)
        if (
            isinstance(self.n_states, bool)
            or not isinstance(self.n_states, int)
            or self.n_states < 2
        ):
            raise ValueError("n_states must be an integer of at least two")
        if states.shape != (len(self.study),) or np.any((states < 0) | (states >= self.n_states)):
            raise ValueError("states must contain one valid label per trial")
        object.__setattr__(self, "states", states)


@dataclass(frozen=True, slots=True)
class FilteredStateProbabilities:
    """Predictive and outcome-updated state probabilities in source row order."""

    predictive: NDArray[np.float64]
    filtered: NDArray[np.float64]

    def __post_init__(self) -> None:
        predictive = protected_array(self.predictive, dtype=np.float64)
        filtered = protected_array(self.filtered, dtype=np.float64)
        if predictive.ndim != 2 or filtered.shape != predictive.shape:
            raise ValueError("state-probability arrays must be equally sized matrices")
        if predictive.shape[1] < 2:
            raise ValueError("state probabilities must contain at least two states")
        for name, values in (("predictive", predictive), ("filtered", filtered)):
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{name} state probabilities must be finite and non-negative")
            if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
                raise ValueError(f"{name} state probabilities must sum to one")
        object.__setattr__(self, "predictive", predictive)
        object.__setattr__(self, "filtered", filtered)


@dataclass(frozen=True, slots=True)
class GroupLabelAgreement:
    """Whether each group's fitted states still mean what the population's states mean.

    A hierarchical GLM-HMM identifies labels by *anchoring*: group :math:`g`'s emissions are
    the population's plus a shrunken deviation, so relabelling one group is not a symmetry
    of the joint objective -- it costs whatever the deviation costs. That argument says the
    label-consistent solution is the global optimum; it does not say a local optimizer found
    it, and it does not say the anchor was strong enough to be worth trusting. This is the
    check.

    For each group the Hungarian matching between that group's emission rows and the
    population's is computed on Euclidean distance. ``permutations[g][k]`` is the population
    state group ``g``'s state ``k`` is closest to; ``aligned[g]`` is whether that matching is
    the identity, which is the only case in which "this animal's state 1 is biased" is a
    statement about the same state the population's coordinate names. ``margins[g]`` is how
    much worse the best *other* matching is, in the same distance units: a margin near zero
    is a group whose states are not separated enough for the anchor to have bitten.
    """

    groups: tuple[Any, ...]
    permutations: tuple[tuple[int, ...], ...]
    distances: NDArray[np.float64]
    margins: NDArray[np.float64]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        permutations = tuple(tuple(int(state) for state in row) for row in self.permutations)
        distances = protected_array(self.distances, dtype=np.float64)
        margins = protected_array(self.margins, dtype=np.float64)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("label agreement groups must be non-empty and unique")
        if len(permutations) != len(groups):
            raise ValueError("one permutation is required per group")
        n_states = len(permutations[0]) if permutations else 0
        if n_states < 2 or any(sorted(row) != list(range(n_states)) for row in permutations):
            raise ValueError("each permutation must permute at least two latent states")
        if distances.shape != (len(groups),) or margins.shape != (len(groups),):
            raise ValueError("one distance and one margin is required per group")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0):
            raise ValueError("label agreement distances must be finite and non-negative")
        if not np.all(np.isfinite(margins)) or np.any(margins < 0):
            raise ValueError("label agreement margins must be finite and non-negative")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "permutations", permutations)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "margins", margins)

    @property
    def aligned(self) -> tuple[bool, ...]:
        """Whether each group's closest matching to the population is the identity."""

        return tuple(row == tuple(range(len(row))) for row in self.permutations)

    @property
    def all_aligned(self) -> bool:
        """Whether every group's states carry the population's meaning."""

        return all(self.aligned)

    @property
    def relabelled_groups(self) -> tuple[Any, ...]:
        """The groups whose deviation is a relabelling rather than a difference."""

        return tuple(
            group for group, aligned in zip(self.groups, self.aligned, strict=True) if not aligned
        )


@dataclass(frozen=True, slots=True)
class _PosteriorStatistics:
    log_likelihood: float
    state_probabilities: NDArray[np.float64]
    initial_counts: NDArray[np.float64]
    transition_counts: NDArray[np.float64]
    transition_expectations: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GLMHMMFitResult(FitResult):
    """A fitted GLM-HMM with restart, occupancy, and label-order diagnostics."""

    restart_objectives: NDArray[np.float64]
    restart_converged: NDArray[np.bool_]
    restart_messages: tuple[str, ...]
    selected_restart: int
    canonical_permutation: tuple[int, ...]
    state_occupancy: NDArray[np.float64]
    state_separation: float
    label_order_gap: float
    label_ambiguous: bool
    low_occupancy: bool

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        objectives = protected_array(self.restart_objectives, dtype=np.float64)
        converged = protected_array(self.restart_converged, dtype=np.bool_)
        messages = tuple(self.restart_messages)
        permutation = tuple(self.canonical_permutation)
        occupancy = protected_array(self.state_occupancy, dtype=np.float64)
        if objectives.ndim != 1 or converged.shape != objectives.shape:
            raise ValueError("restart diagnostics must have one value per restart")
        if len(messages) != len(objectives) or np.any(np.isnan(objectives)):
            raise ValueError("restart messages and non-NaN objectives must align")
        if not 0 <= self.selected_restart < len(objectives):
            raise ValueError("selected_restart must identify one restart")
        if sorted(permutation) != list(range(len(permutation))) or len(permutation) < 2:
            raise ValueError("canonical_permutation must permute every latent state")
        if occupancy.shape != (len(permutation),) or np.any(occupancy < 0):
            raise ValueError("state_occupancy must contain one non-negative value per state")
        if not np.isclose(occupancy.sum(), 1.0, atol=1e-8):
            raise ValueError("state occupancy must sum to one")
        if not np.isfinite(self.state_separation) or self.state_separation < 0:
            raise ValueError("state_separation must be finite and non-negative")
        if not np.isfinite(self.label_order_gap) or self.label_order_gap < 0:
            raise ValueError("label_order_gap must be finite and non-negative")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "restart_converged", converged)
        object.__setattr__(self, "restart_messages", messages)
        object.__setattr__(self, "canonical_permutation", permutation)
        object.__setattr__(self, "state_occupancy", occupancy)


@dataclass(frozen=True, slots=True)
class BernoulliGLMHMM(BernoulliHistoryGLM):
    """A stationary or covariate-dependent HMM with Bernoulli GLM emissions.

    The initial distribution resets at each subject/session boundary. Transition
    probabilities are stationary when ``transition_predictors`` and ``transition_design``
    are absent. Otherwise each row of the transition matrix is a multinomial logistic
    regression on the declared exogenous design, evaluated with trial ``t``'s covariates for
    the transition from ``t - 1`` into ``t``. Latent labels are canonicalized by increasing
    values of ``label_by``, with the complete emission vector used only as a deterministic
    tie-breaker.

    Dynamic transitions use an orthonormal isometric-log-ratio (ILR) coordinate. Unlike a
    reference-category logit, an isotropic Gaussian penalty in this coordinate is invariant
    to relabelling the destination states. That is what makes ``hierarchical(...,
    parameters=("transition",))`` a well-defined random-effects model rather than a prior
    on an arbitrary chart.
    """

    n_states: int = 2
    n_restarts: int = 5
    random_seed: int = 0
    label_by: str = "intercept"
    label_tolerance: float = 1e-3
    state_occupancy_warning: float = 0.01
    probability_warning_threshold: float = 1e-4
    stickiness: float = 0.0
    transition_predictors: tuple[str, ...] = ()
    transition_l2: float = 1.0
    transition_design: DesignSpec | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        BernoulliHistoryGLM.__post_init__(self)
        transition_predictors = tuple(self.transition_predictors)
        validate_design_choice(self.transition_design, transition_predictors)
        if len(set(transition_predictors)) != len(transition_predictors) or any(
            not isinstance(name, str) or not name for name in transition_predictors
        ):
            raise ValueError("transition predictors must be unique non-empty strings")
        if self.outcome in transition_predictors:
            raise ValueError("the outcome cannot also be a transition predictor")
        if self.transition_design is not None and self.transition_design.intercept:
            raise ValueError(
                "transition_design must use intercept=False: the baseline transition matrix "
                "is already the transition intercept"
            )
        if not np.isfinite(self.transition_l2) or self.transition_l2 < 0:
            raise ValueError("transition_l2 must be finite and non-negative")
        if (
            isinstance(self.n_states, bool)
            or not isinstance(self.n_states, int)
            or self.n_states < 2
        ):
            raise ValueError("n_states must be an integer of at least two")
        if (
            isinstance(self.n_restarts, bool)
            or not isinstance(self.n_restarts, int)
            or self.n_restarts < 1
        ):
            raise ValueError("n_restarts must be a positive integer")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if self.label_by not in self.coefficient_names:
            raise ValueError(
                f"label_by must name an emission coefficient: {self.coefficient_names}"
            )
        if not np.isfinite(self.label_tolerance) or self.label_tolerance < 0:
            raise ValueError("label_tolerance must be finite and non-negative")
        if not 0 < self.state_occupancy_warning < 1:
            raise ValueError("state_occupancy_warning must lie strictly between zero and one")
        if not 0 < self.probability_warning_threshold < 0.5:
            raise ValueError("probability_warning_threshold must lie strictly between zero and 0.5")
        if not np.isfinite(self.stickiness) or self.stickiness < 0:
            raise ValueError("stickiness must be finite and non-negative")
        if self.is_dynamic and self.stickiness:
            raise ValueError(
                "stickiness is a Dirichlet pseudo-count on one stationary transition matrix; "
                "it is not a prior on covariate-dependent transition probabilities"
            )
        object.__setattr__(self, "transition_predictors", transition_predictors)

    @property
    def is_dynamic(self) -> bool:
        """Whether transitions depend on a declared exogenous design."""

        return bool(self.transition_predictors or self.transition_design is not None)

    @property
    def transition_covariate_design(self) -> DesignSpec | None:
        """The no-intercept design whose rows alter transition logits."""

        if not self.is_dynamic:
            return None
        return resolve_design(
            self.transition_design,
            self.transition_predictors,
            intercept=False,
        )

    @property
    def transition_coefficient_names(self) -> tuple[str, ...]:
        """Names of the columns that enter the transition multinomial logits."""

        design = self.transition_covariate_design
        return () if design is None else design.feature_names

    @property
    def model_name(self) -> str:
        return "dynamic-bernoulli-glm-hmm" if self.is_dynamic else "bernoulli-glm-hmm"

    @property
    def signature(self) -> str:
        predictors = ",".join(self.predictors)
        transition = ""
        if self.is_dynamic:
            design = self.transition_covariate_design
            assert design is not None
            shorthand = (
                f"predictors={','.join(self.transition_predictors)}"
                if self.transition_design is None
                else f"design={design.signature}"
            )
            transition = f";transition_{shorthand};transition_l2={self.transition_l2}"
        return (
            f"{self.model_name}[states={self.n_states};outcome={self.outcome};"
            f"predictors={predictors};choice_lags={self.choice_lags};"
            f"label_by={self.label_by};l2={self.l2};"
            f"stickiness={self.stickiness}{transition}{self._design_signature}]"
        )

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        """Columns used by either the emission or transition design."""

        columns = list(BernoulliHistoryGLM.required_task_columns.fget(self))
        design = self.transition_covariate_design
        if design is not None:
            columns.extend(
                name
                for name in design.required_columns
                if name != self.outcome and name not in REQUIRED_COLUMNS and name not in columns
            )
        return tuple(columns)

    @property
    def declared_priors(self) -> tuple[str, ...]:
        """Human-readable penalties and chart-free transition declarations."""

        declared = list(BernoulliHistoryGLM.declared_priors.fget(self))
        if self.is_dynamic and self.transition_l2:
            declared.append(
                "ridge on every transition-covariate ILR coefficient: "
                f"Normal(0, {1.0 / self.transition_l2**0.5:.4g}) "
                f"(transition_l2={self.transition_l2})"
            )
        if self.stickiness:
            declared.append(
                f"sticky Dirichlet self-transition pseudo-count (stickiness={self.stickiness})"
            )
        return tuple(declared)

    @property
    def likelihood(self) -> Any:
        """Decline the inherited Bernoulli likelihood, which is not this model's.

        :class:`BernoulliHistoryGLM` answers this with the Bernoulli density of one row
        given one linear predictor. That density is a *part* of a GLM-HMM -- it is the
        emission of one state -- but it is not the model's likelihood, which marginalises
        over a latent path and factorises over sessions rather than over rows. Inheriting a
        working answer to the wrong question is exactly the hazard the composition contracts
        exist to remove, so the answer is withdrawn.

        Withdrawing it is also load-bearing.
        :func:`behavio.contracts.bounded.uses_row_coefficients` asks whether a composable
        model has a ``likelihood`` in order to decide which of the two contracts it is
        composed through, and it asks with :func:`hasattr` precisely so that a property that
        raises counts as "no". This is that property.
        """

        raise AttributeError(
            f"{type(self).__name__} has no linear-predictor likelihood: its density "
            "marginalises over a latent state path. Compose it through "
            "behavio.contracts.bounded.BoundedCoordinateEstimator, whose row_objective() "
            "is the session-blocked forward recursion this model actually scores."
        )

    @property
    def penalised_linear_refusal(self) -> str:
        """Decline the penalised-linear contract this family inherits but cannot honour.

        :class:`BernoulliGLMHMM` extends :class:`BernoulliHistoryGLM` for its per-state
        emissions, which means it inherits every member
        :class:`behavio.contracts.compose.PenalisedLinearEstimator` asks for, and would
        satisfy any widening of them: a ``(rows, states)`` linear predictor is exactly what
        the emissions produce. What it cannot honour is the part of that contract which is
        not a shape. A penalised linear model's log likelihood is a sum of independent row
        scores ``f(eta_r, y_r)``; a GLM-HMM's is a forward recursion in which row ``r``'s
        contribution depends on every row before it.

        That is a refusal of the **penalised-linear** contract. It is not a refusal of
        hierarchy: row independence is what
        :attr:`~behavio.contracts.bounded.RowObjective.row_blocks` exists to relax, and a
        GLM-HMM's recursion runs over one subject's session, which lies inside a subject.
        ``hierarchical(model, over="subject", parameters=<emission coefficients>)`` therefore
        composes through :class:`~behavio.contracts.bounded.BoundedCoordinateEstimator`;
        what it may vary is limited by :meth:`varying_parameter_refusal`. ``mix()`` is
        refused separately and for its own reason, by :attr:`independent_rows_refusal`.

        No arrangement of members can be inspected to discover any of that, which is why
        this is a sentence rather than a signature.
        """

        return (
            "a GLM-HMM is a latent-state mixture, not a penalised linear model: its row "
            "scores come from a forward recursion over a whole session rather than from "
            "one linear predictor per row. Compose hierarchy through "
            "behavio.contracts.bounded.BoundedCoordinateEstimator instead, which is "
            "written against the blocks a recursion runs over"
        )

    @property
    def independent_rows_refusal(self) -> str:
        """Why a mixture may not be applied to this model, on either contract.

        ``mix()`` is gated on row independence rather than on a linear predictor, so
        widening the combinator did not open this cell and could not have. A GLM-HMM's row
        scores come out of a forward recursion over a whole session, so there is no per-row
        density for a second one to be averaged with.

        The refusal is a modelling statement as much as an arithmetic one. A lapse on a
        GLM-HMM is a lapse on the *emission*, inside the recursion. Averaged in from outside,
        over the marginal one-step-ahead prediction, the weight would be free to absorb the
        state switching it is supposed to be distinguished from -- which is the opposite of
        what a lapse competitor is for.
        """

        return (
            "a GLM-HMM is a latent-state mixture whose rows are not independent: its row "
            "scores come from a forward recursion over a whole session rather than from "
            "one density per row, and a lapse on a GLM-HMM is a lapse on the state's "
            "emission, inside that recursion, not a second density averaged with the "
            "marginal one from outside it. Mixing from outside would let the weight absorb "
            "the state switching it is supposed to be distinguished from"
        )

    def varying_parameter_refusal(
        self, parameters: Sequence[str] | None, *, combinator: str
    ) -> str:
        """Say which of this model's parameters may carry a group effect or a path.

        Two of the three answers are refusals, and both are about the difference between a
        coordinate and the thing it is a chart for.

        **Stationary transitions and the initial distribution are refused.** The legacy
        stationary transition matrix and the initial simplex use reference-category logits.
        An isotropic Gaussian there is a prior on an arbitrary chart. A dynamic model instead
        stores both transition intercepts and slopes in an orthonormal ILR coordinate. Naming
        ``"transition"`` then varies the complete transition regression under one isotropic,
        relabelling-invariant scale. Individual source, destination, or contrast coordinates
        remain inseparable because selecting one would break that invariance.

        **A path is refused outright.** ``smooth()`` would make a state's emissions a
        function of clock time, and this model's states are labelled by *ordering* one
        emission coefficient. An ordering of paths is only a permutation if the paths do not
        cross; where two states' ``label_by`` paths cross there is no permutation that
        canonicalises the fit, and "state 0" names one behaviour early in training and
        another late. That is not a numerical difficulty -- the fit would converge and
        report knots -- it is the fit meaning two incompatible things at once. A drifting
        GLM-HMM is a real and useful model; it needs a labelling rule defined on paths, and
        a report of where paths cross, before anything it estimates can be read.

        **Emission coefficients are admitted, by coefficient and never by state.** Naming
        ``"intercept"`` varies ``state[k].intercept`` for every ``k`` under one scale, and
        naming ``state[0].intercept`` alone is refused: a group's deviation has to be the
        same kind of object as the population parameter, and the population parameter here
        is permutation-equivariant. If one state's copy could carry a deviation the others
        could not, the joint objective would stop being invariant under simultaneous
        relabelling and the canonical ordering this model reports would no longer be a
        symmetry of the thing it canonicalises.
        """

        if combinator == "smooth":
            return (
                "a GLM-HMM's latent labels are an ordering of one emission coefficient, and "
                "an ordering of coefficient *paths* is only a permutation where the paths do "
                "not cross; a smooth GLM-HMM therefore has no canonical labelling and its "
                "'state 0' would name different behaviour at the two ends of the clock. Fit "
                "separate GLM-HMMs per training stage, or compare this model against a "
                "smooth GLM, until a labelling rule defined on paths exists"
            )
        if parameters is None:
            transition_note = (
                "Its dynamic transition regression is eligible under the alias 'transition', "
                "but its initial distribution remains a reference-category logit. "
                if self.is_dynamic
                else (
                    "Its transition and initial coordinates are reference-category logits, "
                    "and an isotropic Gaussian is a prior on that chart rather than on the "
                    "simplex it charts. "
                )
            )
            return (
                "a GLM-HMM cannot let every parameter vary by group. "
                f"{transition_note}Name the complete exchangeable blocks that vary, for "
                f"example parameters={self.coefficient_names!r}"
            )
        declared = tuple(parameters)
        emissions = set(self.coefficient_names)
        transition_aliases = {"transition"} if self.is_dynamic else set()
        offenders = [
            name for name in declared if name not in emissions and name not in transition_aliases
        ]
        if not offenders:
            return ""
        per_state = [name for name in offenders if name.startswith("state[")]
        if per_state:
            return (
                f"{sorted(per_state)} names one state's copy of an emission coefficient. A "
                "GLM-HMM's coordinate is permutation-equivariant, so a coefficient varies by "
                "group for every state at once or not at all: name the bare coefficient "
                f"instead, one of {list(self.coefficient_names)}"
            )
        if self.is_dynamic:
            return (
                f"{sorted(offenders)} is not a complete exchangeable parameter block. Name "
                f"emission coefficients from {list(self.coefficient_names)}, or name "
                "'transition' to vary every source-state ILR intercept and covariate effect "
                "together. One source, destination, or contrast cannot vary alone because "
                "that selection is not closed under latent-state relabelling; the initial "
                f"simplex is still reference-coded ({self._reference_note()})"
            )
        return (
            f"{sorted(offenders)} is not an emission coefficient of this model. Only "
            f"{list(self.coefficient_names)} may carry group deviations: the initial and "
            "transition coordinates are reference-category logits on a simplex "
            f"({self._reference_note()}), where an isotropic Gaussian is a prior on the "
            "chart and not on the transition matrix, and the reference state is chosen by "
            "label canonicalisation rather than by you. Use stickiness= for population-level "
            "persistence, or declare transition predictors to fit chart-free transition "
            "random effects"
        )

    def _reference_note(self) -> str:
        return f"state {self.n_states - 1} is the reference"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        emissions = tuple(
            f"state[{state}].{coefficient}"
            for state in range(self.n_states)
            for coefficient in self.coefficient_names
        )
        initial = tuple(
            f"initial_logit[state={state}|reference={self.n_states - 1}]"
            for state in range(self.n_states - 1)
        )
        if not self.is_dynamic:
            transitions = tuple(
                f"transition_logit[from={source},to={destination}|reference={self.n_states - 1}]"
                for source in range(self.n_states)
                for destination in range(self.n_states - 1)
            )
            return (*emissions, *initial, *transitions)
        transition_intercepts = tuple(
            f"transition_ilr[from={source},contrast={contrast}]"
            for source in range(self.n_states)
            for contrast in range(self.n_states - 1)
        )
        transition_effects = tuple(
            f"transition_ilr_coefficient[from={source},contrast={contrast}].{predictor}"
            for source in range(self.n_states)
            for contrast in range(self.n_states - 1)
            for predictor in self.transition_coefficient_names
        )
        return (*emissions, *initial, *transition_intercepts, *transition_effects)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return (PredictionMode.FILTERED,)

    def parameters_from_components(
        self,
        *,
        initial_probabilities: Sequence[float],
        transition_matrix: Sequence[Sequence[float]],
        emissions: Mapping[str, Sequence[float]],
        transition_coefficients: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> Mapping[str, float]:
        """Validate, canonicalize, and pack natural-scale model components.

        Dynamic transition effects are supplied as one ``(source, destination)`` matrix per
        transition-design column. Each source row must sum to zero, the identified centred
        multinomial-logit representation. A positive value makes that destination more
        likely relative to the geometric mean of all destinations as the predictor grows.
        """

        if set(emissions) != set(self.coefficient_names):
            expected = set(self.coefficient_names)
            observed = set(emissions)
            raise ValueError(
                "emissions must match model coefficients exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        emission_matrix = np.column_stack(
            [np.asarray(emissions[name], dtype=np.float64) for name in self.coefficient_names]
        )
        transition_names = self.transition_coefficient_names
        supplied = {} if transition_coefficients is None else dict(transition_coefficients)
        if set(supplied) != set(transition_names) and (supplied or transition_names):
            expected = set(transition_names)
            observed = set(supplied)
            raise ValueError(
                "transition_coefficients must match the transition design exactly; "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
        if transition_names:
            transition_effects = np.stack(
                [np.asarray(supplied[name], dtype=np.float64) for name in transition_names],
                axis=2,
            )
        else:
            transition_effects = np.zeros((self.n_states, self.n_states, 0), dtype=np.float64)
        components = GLMHMMParameters(
            initial_probabilities=np.asarray(initial_probabilities, dtype=np.float64),
            transition_matrix=np.asarray(transition_matrix, dtype=np.float64),
            emission_coefficients=emission_matrix,
            coefficient_names=self.coefficient_names,
            transition_effects=transition_effects,
            transition_predictor_names=transition_names,
        )
        if components.n_states != self.n_states:
            raise ValueError(f"components must contain exactly {self.n_states} states")
        canonical, _ = self._canonicalize_components(components)
        values = self._pack_components(canonical)
        return MappingProxyType(dict(zip(self.parameter_names, values.tolist(), strict=True)))

    def parameter_components(
        self,
        parameters: Mapping[str, float] | FitResult,
    ) -> GLMHMMParameters:
        """Decode optimizer coordinates into probabilities and emission coefficients."""

        if isinstance(parameters, FitResult):
            self._validate_fit(parameters)
            vector = parameters.estimates
        else:
            expected = set(self.parameter_names)
            observed = set(parameters)
            if observed != expected:
                raise ValueError(
                    "parameters must match the model exactly; "
                    f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
                )
            try:
                vector = np.asarray(
                    [parameters[name] for name in self.parameter_names], dtype=np.float64
                )
            except (TypeError, ValueError):
                raise ValueError("parameters must contain finite numeric values") from None
            if not np.all(np.isfinite(vector)):
                raise ValueError("parameters must contain finite numeric values")
        return self._unpack_components(vector)

    def transition_design_matrix(self, study: Study) -> NDArray[np.float64]:
        """Return exogenous transition covariates in source row order.

        Trial ``t``'s row governs ``P(z_t | z_{t-1})``. The first row of every session is
        intentionally retained but unused because the state distribution resets there.
        Learned scaling, categorical levels, and other fitted landmarks must therefore be
        declared in a training-fitted :class:`~behavio.design.DesignSpec`, exactly as for an
        emission design.
        """

        design = self.transition_covariate_design
        if design is None:
            return np.empty((len(study), 0), dtype=np.float64)
        return np.asarray(build_matrix(design, study).values, dtype=np.float64)

    def transition_probabilities(
        self,
        study: Study,
        parameters: Mapping[str, float] | FitResult,
    ) -> NDArray[np.float64]:
        """Return the fitted or declared transition matrix on every source row.

        This is the reportable natural-scale estimand for a non-homogeneous HMM. It includes
        session-opening rows for alignment with the study, although those rows are reset to
        the initial distribution and do not contribute a transition to the likelihood.
        """

        components = self.parameter_components(parameters)
        return components.transition_probabilities(self.transition_design_matrix(study))

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices without exposing latent-state truth as an observed column."""

        return self.simulate_with_states(design, parameters, seed=seed).study

    def simulate_with_states(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> GLMHMMSimulation:
        """Generate choices and return latent-state truth in a separate result."""

        components = self.parameter_components(parameters)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices, states = self._generate(design, lambda index: components, generator)

        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return GLMHMMSimulation(study=Study(columns), states=states, n_states=self.n_states)

    def fit(self, study: Study) -> GLMHMMFitResult:
        """Fit by deterministic multi-start maximum penalized likelihood."""

        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        transition_features = self.transition_design_matrix(study)
        sessions = ordered_session_indices(study)

        def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            return self._objective_gradient(
                vector,
                features,
                outcomes,
                sessions,
                transition_features=transition_features,
            )

        starts = self._initial_points(features, outcomes)
        bounds = [(-30.0, 30.0)] * len(self.parameter_names)
        results = [
            minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "gtol": self.tolerance,
                },
            )
            for start in starts
        ]
        restart_objectives = np.asarray(
            [float(result.fun) if np.isfinite(result.fun) else np.inf for result in results]
        )
        finite = [index for index, value in enumerate(restart_objectives) if np.isfinite(value)]
        if not finite:
            messages = "; ".join(str(result.message) for result in results)
            raise ModelDataError(f"all GLM-HMM restarts produced non-finite objectives: {messages}")
        successful = [index for index in finite if results[index].success]
        eligible = successful if successful else finite
        selected = min(eligible, key=lambda index: float(restart_objectives[index]))
        raw = np.asarray(results[selected].x, dtype=np.float64)
        canonical, permutation = self._canonicalize_components(self._unpack_components(raw))
        estimates = self._pack_components(canonical)
        value, gradient = objective(estimates)
        hessian = finite_difference_hessian(
            lambda vector: objective(vector)[1],
            estimates,
            steps=offset_steps(estimates, scale=1e-5),
        )
        condition = float(np.linalg.cond(hessian))
        covariance = np.linalg.pinv(hessian, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        gamma = self._posterior_state_probabilities(
            estimates,
            features,
            transition_features,
            outcomes,
            sessions,
        )
        occupancy = np.mean(gamma, axis=0)
        separation = _minimum_pairwise_distance(canonical.emission_coefficients)
        label_values = canonical.emission_coefficients[
            :, self.coefficient_names.index(self.label_by)
        ]
        label_gap = float(np.min(np.diff(label_values)))
        transition_probabilities = canonical.transition_probabilities(transition_features)
        probability_values = np.concatenate(
            (canonical.initial_probabilities, transition_probabilities.ravel())
        )
        boundary = bool(
            np.any(np.abs(canonical.emission_coefficients) >= self.coefficient_warning_threshold)
            or np.any(probability_values <= self.probability_warning_threshold)
            or np.any(probability_values >= 1.0 - self.probability_warning_threshold)
        )
        chosen = results[selected]
        diagnostics = FitDiagnostics(
            converged=bool(chosen.success),
            optimizer=f"L-BFGS-B ({self.n_restarts} deterministic restarts)",
            status=int(chosen.status),
            message=str(chosen.message),
            n_iterations=int(chosen.nit),
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
            hessian_condition=condition,
            boundary_estimate=boundary,
        )
        return GLMHMMFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=estimates,
            standard_errors=standard_errors,
            covariance=covariance,
            n_observations=len(study),
            diagnostics=diagnostics,
            restart_objectives=restart_objectives,
            restart_converged=np.asarray([result.success for result in results]),
            restart_messages=tuple(str(result.message) for result in results),
            selected_restart=selected,
            canonical_permutation=permutation,
            state_occupancy=occupancy,
            state_separation=separation,
            label_order_gap=label_gap,
            label_ambiguous=bool(label_gap <= self.label_tolerance),
            low_occupancy=bool(np.any(occupancy < self.state_occupancy_warning)),
        )

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> Prediction:
        """Return one-step-ahead choice probabilities under filtered latent state."""

        prediction_mode = self._prediction_mode(mode)
        self._validate_fit(fit)
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        transition_features = self.transition_design_matrix(study)
        components = self.parameter_components(fit)
        state_probabilities = self._filtered_state_probabilities(
            features,
            transition_features,
            outcomes,
            ordered_session_indices(study),
            components,
        )
        emission_probabilities = expit(features @ components.emission_coefficients.T)
        probability = np.sum(state_probabilities.predictive * emission_probabilities, axis=1)
        probability = np.clip(probability, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=prediction_mode,
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Score observed choices under sequential filtered predictions."""

        outcomes = self.outcomes(study)
        prediction = self.predict(study, fit, mode=mode)
        scores = outcomes * np.log(prediction.probability)
        scores += (1.0 - outcomes) * np.log1p(-prediction.probability)
        return protected_array(scores, dtype=np.float64)

    def state_probabilities(
        self,
        study: Study,
        fit: FitResult,
    ) -> FilteredStateProbabilities:
        """Return predictive and outcome-updated state probabilities."""

        self._validate_fit(fit)
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        transition_features = self.transition_design_matrix(study)
        return self._filtered_state_probabilities(
            features,
            transition_features,
            outcomes,
            ordered_session_indices(study),
            self.parameter_components(fit),
        )

    def state_recovery(
        self,
        simulation: GLMHMMSimulation,
        fit: FitResult,
        *,
        ambiguity_tolerance: float = 0.05,
    ) -> LatentStateAlignment:
        """Align outcome-filtered state probabilities to separately retained truth."""

        if not isinstance(simulation, GLMHMMSimulation):
            raise TypeError("simulation must be a GLMHMMSimulation")
        if simulation.n_states != self.n_states:
            raise ValueError("simulation and model must contain the same number of states")
        filtered = self.state_probabilities(simulation.study, fit).filtered
        return align_latent_states(
            simulation.states,
            filtered,
            ambiguity_tolerance=ambiguity_tolerance,
        )

    # -- the bounded-coordinate composition contract --------------------------------------
    #
    # ``parameter_names`` is already an unconstrained coordinate -- emission coefficients on
    # the log-odds scale, and reference-category logits for the two simplexes -- so hierarchy
    # needs nothing added to it. What it needs is the likelihood as a function of one such
    # coordinate *per row*, and the blocks that coordinate must be constant within. Those
    # blocks are the sessions the forward recursion runs over, which is exactly the relaxation
    # ``row_blocks`` was written for. See ``behavio.contracts.bounded``.

    @property
    def bounded_coordinate_refusal(self) -> str:
        """Decline composition while a sticky transition prior is in force.

        ``stickiness`` adds ``-kappa sum_k log A[k,k]`` to the objective. That term is
        neither a per-row score nor a quadratic penalty, which are the only two things the
        bounded-coordinate contract can carry: ``RowObjective`` is a likelihood and
        ``penalty_matrix`` is a quadratic form. Folding it into the row objective would
        apply it once per session rather than once per model, which is a different and
        stronger prior on studies with more sessions.

        The restriction costs little, because transitions are pooled in a composed fit
        anyway (see :meth:`varying_parameter_refusal`) and the population transition matrix
        is therefore estimated from every subject at once.
        """

        if not self.stickiness:
            return ""
        return (
            "a sticky Dirichlet transition prior adds -kappa*sum_k log A[k,k] to the "
            "objective, which is neither a per-row score nor a quadratic penalty, so a "
            "combinator would have to apply it once per session block instead of once per "
            "model. Compose the stickiness=0 model, or keep the sticky prior and fit each "
            "subject separately"
        )

    def row_objective(self, study: Study) -> _GLMHMMRowObjective:
        """Return this study's negative log likelihood in one coordinate per row."""

        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        transition_features = self.transition_design_matrix(study)
        sessions = ordered_session_indices(study)
        row_blocks = np.empty(len(study), dtype=np.intp)
        blocks = []
        for block, indices in enumerate(sessions):
            index = np.asarray(indices, dtype=np.intp)
            row_blocks[index] = block
            blocks.append((index, features[index], transition_features[index], outcomes[index]))
        return _GLMHMMRowObjective(
            model=self,
            blocks=tuple(blocks),
            row_blocks=row_blocks,
            n_rows=len(study),
        )

    def penalty_matrix(self) -> NDArray[np.float64]:
        """Return emission ridge plus dynamic-transition slope ridge.

        The inherited GLM answers "every coordinate but the first", which is the right
        statement about a coefficient vector and the wrong one about this coordinate: it
        would ridge one state's intercept, every other state's intercept, and both
        simplexes' logits. Stationary transition logits and dynamic ILR intercepts remain
        unpenalised. Dynamic covariate effects are genuine regression coefficients, and are
        ridge-regularised in the orthonormal log-ratio coordinate by ``transition_l2``.
        """

        penalty = np.zeros(len(self.parameter_names), dtype=np.float64)
        n_coefficients = len(self.coefficient_names)
        for state in range(self.n_states):
            start = state * n_coefficients
            penalty[start + 1 : start + n_coefficients] = self.l2
        if self.is_dynamic:
            slope_start = (
                self.n_states * n_coefficients
                + (self.n_states - 1)
                + self.n_states * (self.n_states - 1)
            )
            penalty[slope_start:] = self.transition_l2
        return np.diag(penalty)

    def coordinate_box(self, study: Study) -> NDArray[np.float64]:
        """Return the finite box :meth:`fit` already searches this coordinate in."""

        del study
        return np.tile(np.asarray([-30.0, 30.0]), (len(self.parameter_names), 1))

    def initial_points(self, study: Study) -> tuple[NDArray[np.float64], ...]:
        """Return the deterministic restarts this model's own solver would use.

        The origin is not among them and must not be: at the origin every state has the same
        emissions, which is a saddle of the marginal likelihood rather than a starting point.
        The restarts separate the states along ``label_by`` in canonical order, which is also
        what makes a composed fit start from a labelling and not from a coin flip.
        """

        return self._initial_points(self.design_matrix(study), self.outcomes(study))

    def group_parameter_expansion(self, name: str) -> tuple[str, ...]:
        """Expand an emission coefficient or the complete dynamic transition regression.

        This is the same statement a smooth model makes about a coefficient path: a group's
        deviation is the same kind of object as the population parameter, and the population
        parameter here is a coefficient *for all states*, because the coordinate is
        permutation-equivariant. Naming one state's copy is refused in
        :meth:`varying_parameter_refusal`; naming the coefficient names all of them, under
        one prior scale.
        """

        if name in self.coefficient_names:
            return tuple(f"state[{state}].{name}" for state in range(self.n_states))
        if name == "transition" and self.is_dynamic:
            return tuple(
                parameter
                for parameter in self.parameter_names
                if parameter.startswith("transition_ilr[")
                or parameter.startswith("transition_ilr_coefficient[")
            )
        return (name,)

    def group_penalty(
        self, columns: NDArray[np.intp], scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return isotropic Gaussian precision in emission or orthonormal ILR coordinates."""

        del columns
        return ridge_group_penalty(scales)

    def draw_group_deviations(
        self,
        columns: NDArray[np.intp],
        scales: NDArray[np.float64],
        *,
        groups: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw independent Gaussian deviations on the log-odds emission coordinate."""

        del columns
        return ridge_group_draw(scales, groups=groups, generator=generator)

    def fit_rows(
        self,
        design: RowCoefficientDesign,
        *,
        model_name: str,
        model_signature: str,
    ) -> FitResult:
        """Solve a row-coefficient problem, then put the joint solution in label order.

        The second half is what makes a composed GLM-HMM reportable. The joint objective is
        invariant under relabelling the population states *and* every group's deviation
        block together -- that is the one label symmetry hierarchy does not break, and it is
        the same symmetry :meth:`fit` resolves by sorting states along ``label_by``. Here it
        is resolved the same way and applied to the whole joint vector at once, which is
        possible because relabelling is a *linear* map on this coordinate:
        :meth:`relabelling_map` writes it down, so the covariance is carried along exactly
        rather than approximated.

        The symmetry hierarchy *does* break -- relabelling one group on its own -- is the
        reason the deviations mean anything, and it is checked after the fact by
        :meth:`group_label_agreement` rather than assumed here.
        """

        fit = solve_row_coefficients(
            design,
            model_name=model_name,
            model_signature=model_signature,
            optimizer="L-BFGS-B",
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            boundary=self._row_boundary,
        )
        return self._relabelled_fit(fit, design)

    def simulate_rows(
        self,
        design: Study,
        coefficients: NDArray[np.float64],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Generate choices given one parameter vector per row, constant within a session."""

        rows = validated_row_coefficients(
            coefficients,
            n_rows=len(design),
            n_parameters=len(self.parameter_names),
            what="simulate_rows",
        )
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        choices, _ = self._generate(
            design,
            lambda index: self._session_components(rows, index, "simulate_rows coefficients"),
            generator,
        )
        columns = {name: design[name] for name in design.columns}
        columns[self.outcome] = choices
        return Study(columns)

    def predict_rows(
        self,
        study: Study,
        coefficients: NDArray[np.float64],
        *,
        mode: PredictionMode,
    ) -> Prediction:
        """Return one-step-ahead filtered choice probabilities under per-row coordinates."""

        prediction_mode = self._prediction_mode(mode)
        probability = self._row_choice_probability(study, coefficients)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(probability) - np.log1p(-probability),
            mode=prediction_mode,
        )

    def pointwise_log_prob_rows(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score each observed choice under per-row coordinates and a filtered state."""

        outcomes = self.outcomes(study)
        probability = self._row_choice_probability(study, coefficients)
        scores = outcomes * np.log(probability)
        scores += (1.0 - outcomes) * np.log1p(-probability)
        return protected_array(scores, dtype=np.float64)

    def group_label_agreement(self, fit: Any) -> GroupLabelAgreement:
        """Check that each group's fitted states still name the population's states.

        Takes any object exposing ``groups`` and ``group_parameter_vectors`` -- which is
        what a :class:`behavio.compose.HierarchicalFitResult` exposes -- and reports the
        Hungarian matching between each group's emission rows and the population's.

        This is the diagnostic that separates a hierarchical GLM-HMM from a well-typed
        guess. Anchoring makes label consistency the *global* optimum of the joint
        objective; it does not make it the point a multi-start local optimizer reached, and
        it says nothing at all when a group's own states are barely separated. A group that
        comes back permuted has a deviation that is a relabelling rather than a difference,
        and reading it as "this animal is more biased in state 1" is exactly the confident
        nonsense a latent-state model is capable of.

        It is deliberately *not* :func:`~behavio.models.state_alignment.align_latent_states`.
        That function aligns inferred state posteriors against simulated ground truth and so
        cannot run on data; this one compares two fitted parameter vectors and runs on
        anything.
        """

        groups = tuple(_scalar(group) for group in fit.groups)
        vectors = np.asarray(fit.group_parameter_vectors, dtype=np.float64)
        population = self._unpack_components(
            np.asarray(fit.estimates, dtype=np.float64)
        ).emission_coefficients
        if vectors.ndim != 2 or vectors.shape != (len(groups), len(self.parameter_names)):
            raise ValueError("group_parameter_vectors must hold one full coordinate per group")
        permutations: list[tuple[int, ...]] = []
        distances = np.empty(len(groups), dtype=np.float64)
        margins = np.empty(len(groups), dtype=np.float64)
        for index, vector in enumerate(vectors):
            emissions = self._unpack_components(vector).emission_coefficients
            cost = np.linalg.norm(emissions[:, None, :] - population[None, :, :], axis=2)
            rows, matched = linear_sum_assignment(cost)
            mapping = np.empty(self.n_states, dtype=np.int64)
            mapping[rows] = matched
            best = float(cost[np.arange(self.n_states), mapping].sum())
            permutations.append(tuple(int(state) for state in mapping))
            distances[index] = best
            margins[index] = max(0.0, _runner_up_cost(cost, mapping) - best)
        return GroupLabelAgreement(
            groups=groups,
            permutations=tuple(permutations),
            distances=distances,
            margins=margins,
        )

    def relabelling_map(self, permutation: Sequence[int]) -> NDArray[np.float64]:
        """Write relabelling the states as a matrix on this model's own coordinate.

        Emissions permute rows, which is an index permutation. The two simplexes do not:
        their coordinates are differences of log probabilities against a reference state,
        so relabelling both permutes *and* re-references them. It stays linear --
        ``l'[k] = u[p(k)] - u[p(K-1)]`` where ``u`` is the log-probability vector the
        reference logits chart -- which is the whole reason the covariance can be carried
        through a canonicalisation instead of being recomputed.
        """

        order = [int(state) for state in permutation]
        if sorted(order) != list(range(self.n_states)):
            raise ValueError("a relabelling must permute every latent state")
        n_coefficients = len(self.coefficient_names)
        n_parameters = len(self.parameter_names)
        reference = self.n_states - 1
        emission_end = self.n_states * n_coefficients
        initial_end = emission_end + reference
        matrix = np.zeros((n_parameters, n_parameters), dtype=np.float64)
        for state in range(self.n_states):
            for coefficient in range(n_coefficients):
                row = state * n_coefficients + coefficient
                matrix[row, order[state] * n_coefficients + coefficient] = 1.0
        for state in range(reference):
            row = emission_end + state
            if order[state] < reference:
                matrix[row, emission_end + order[state]] += 1.0
            if order[reference] < reference:
                matrix[row, emission_end + order[reference]] -= 1.0
        if not self.is_dynamic:
            for source in range(self.n_states):
                for destination in range(reference):
                    row = initial_end + source * reference + destination
                    base = initial_end + order[source] * reference
                    if order[destination] < reference:
                        matrix[row, base + order[destination]] += 1.0
                    if order[reference] < reference:
                        matrix[row, base + order[reference]] -= 1.0
            return matrix

        destination_map = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        destination_map[np.arange(self.n_states), np.asarray(order, dtype=np.intp)] = 1.0
        basis = _ilr_basis(self.n_states)
        rotation = basis.T @ destination_map @ basis
        transition_start = initial_end
        for source in range(self.n_states):
            old_source = order[source]
            rows = slice(
                transition_start + source * reference,
                transition_start + (source + 1) * reference,
            )
            columns = slice(
                transition_start + old_source * reference,
                transition_start + (old_source + 1) * reference,
            )
            matrix[rows, columns] = rotation

        n_predictors = len(self.transition_coefficient_names)
        slope_start = transition_start + self.n_states * reference
        for source in range(self.n_states):
            old_source = order[source]
            for predictor in range(n_predictors):
                for new_contrast in range(reference):
                    row = (
                        slope_start + (source * reference + new_contrast) * n_predictors + predictor
                    )
                    for old_contrast in range(reference):
                        column = (
                            slope_start
                            + (old_source * reference + old_contrast) * n_predictors
                            + predictor
                        )
                        matrix[row, column] = rotation[new_contrast, old_contrast]
        return matrix

    # -- internals of the composition contract --------------------------------------------

    def _row_choice_probability(
        self, study: Study, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Run the one-step-ahead filter session by session under that session's coordinate."""

        rows = validated_row_coefficients(
            coefficients,
            n_rows=len(study),
            n_parameters=len(self.parameter_names),
            what="row coefficients",
        )
        outcomes = self.outcomes(study)
        features = self.design_matrix(study)
        transition_features = self.transition_design_matrix(study)
        probability = np.empty(len(study), dtype=np.float64)
        for session_indices in ordered_session_indices(study):
            index = np.asarray(session_indices, dtype=np.intp)
            components = self._session_components(rows, index, "row coefficients")
            linear = features[index] @ components.emission_coefficients.T
            emission_log = outcomes[index, None] * -np.logaddexp(0.0, -linear) + (
                1.0 - outcomes[index, None]
            ) * -np.logaddexp(0.0, linear)
            emission_probability = expit(linear)
            transition = components.transition_probabilities(transition_features[index])
            prior = np.asarray(components.initial_probabilities)
            for position, row in enumerate(index):
                probability[row] = float(prior @ emission_probability[position])
                log_weight = np.log(prior) + emission_log[position]
                posterior = np.exp(log_weight - logsumexp(log_weight))
                if position + 1 < len(index):
                    prior = posterior @ transition[position + 1]
        return np.clip(probability, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)

    def _session_components(
        self, rows: NDArray[np.float64], index: NDArray[np.intp], what: str
    ) -> GLMHMMParameters:
        block = rows[index]
        if not np.array_equal(block, np.tile(block[0], (len(index), 1))):
            raise ValueError(
                f"{what} must be constant within a session: a forward recursion cannot say "
                "which of two parameter values produced which part of one state path"
            )
        return self._unpack_components(block[0])

    def _generate(
        self,
        design: Study,
        components_for: Callable[[NDArray[np.intp]], GLMHMMParameters],
        generator: np.random.Generator,
    ) -> tuple[NDArray[np.int8], NDArray[np.int64]]:
        """Simulate choices and latent states, one set of components per session.

        One simulator for both callers. ``simulate_with_states`` passes the study's single
        decoded component set and ``simulate_rows`` decodes each session's own, which is the
        only difference between "this model" and "this model made hierarchical": the draw
        sequence, the history recursion and the state chain are identical, and a fit
        published before this contract existed is still reproduced draw for draw.
        """

        predictors = self._predictor_matrix(design)
        transition_features = self.transition_design_matrix(design)
        choices = np.zeros(len(design), dtype=np.int8)
        states = np.zeros(len(design), dtype=np.int64)
        predictor_end = 1 + len(self.predictors)
        for session_indices in ordered_session_indices(design):
            index = np.asarray(session_indices, dtype=np.intp)
            components = components_for(index)
            transition = components.transition_probabilities(transition_features[index])
            generated_history: list[float] = []
            state = int(generator.choice(self.n_states, p=components.initial_probabilities))
            for position, row in enumerate(session_indices):
                if position:
                    state = int(generator.choice(self.n_states, p=transition[position, state]))
                features = np.empty(len(self.coefficient_names), dtype=np.float64)
                features[0] = 1.0
                features[1:predictor_end] = predictors[row]
                for lag in range(1, self.choice_lags + 1):
                    features[predictor_end + lag - 1] = (
                        generated_history[-lag] if len(generated_history) >= lag else 0.0
                    )
                probability = expit(float(features @ components.emission_coefficients[state]))
                choice = int(generator.binomial(1, probability))
                choices[row] = choice
                states[row] = state
                generated_history.append(2.0 * choice - 1.0)
        return choices, states

    def _row_boundary(
        self,
        estimates: NDArray[np.float64],
        derived: NDArray[np.float64] | None,
    ) -> bool:
        """Apply this model's own boundary convention to a composed estimate.

        The population block decodes to a complete set of components and gets the same test
        :meth:`fit` applies. Every *derived* value -- population plus one group's deviation
        -- is an emission coefficient and nothing else, because
        :meth:`varying_parameter_refusal` admits nothing else, so the coefficient magnitude
        test applies to it directly without having to be told which coordinates it holds.
        """

        n_parameters = len(self.parameter_names)
        vector = np.asarray(estimates, dtype=np.float64)[:n_parameters]
        if len(vector) == n_parameters:
            try:
                components = self._unpack_components(vector)
            except ValueError:
                return True
            probabilities = np.concatenate(
                (components.initial_probabilities, components.transition_matrix.ravel())
            )
            emissions = np.abs(components.emission_coefficients)
            if (
                np.any(emissions >= self.coefficient_warning_threshold)
                or np.any(probabilities <= self.probability_warning_threshold)
                or np.any(probabilities >= 1.0 - self.probability_warning_threshold)
            ):
                return True
        if derived is None:
            return False
        values = np.asarray(derived, dtype=np.float64)
        return bool(np.any(np.abs(values) >= self.coefficient_warning_threshold))

    def _joint_relabelling_map(
        self, permutation: Sequence[int], design: RowCoefficientDesign
    ) -> NDArray[np.float64]:
        """Lift :meth:`relabelling_map` onto the joint coordinate a combinator built."""

        population = self.relabelling_map(permutation)
        n_parameters = len(self.parameter_names)
        expansion = design.expansion
        if expansion is None:
            if len(design.parameter_names) != n_parameters:
                raise ValueError(
                    "a composed GLM-HMM fit must report either this model's coordinate or a "
                    "group expansion of it; canonical state order is undefined otherwise"
                )
            return population
        columns = np.asarray(expansion.columns, dtype=np.intp)
        block = population[np.ix_(columns, columns)]
        outside = np.delete(np.arange(n_parameters), columns)
        if outside.size and np.any(population[np.ix_(columns, outside)]):
            raise ValueError(
                "the varying parameters of a GLM-HMM must be closed under relabelling: "
                "a coefficient varies by group for every state at once or not at all"
            )
        size = len(design.parameter_names)
        matrix = np.zeros((size, size), dtype=np.float64)
        matrix[:n_parameters, :n_parameters] = population
        for group in range(expansion.n_groups):
            span = expansion.group_slice(group)
            matrix[span, span] = block
        return matrix

    def _relabelled_fit(self, fit: FitResult, design: RowCoefficientDesign) -> FitResult:
        n_parameters = len(self.parameter_names)
        estimates = np.asarray(fit.estimates, dtype=np.float64)
        _, permutation = self._canonicalize_components(
            self._unpack_components(estimates[:n_parameters])
        )
        if permutation == tuple(range(self.n_states)):
            return fit
        matrix = self._joint_relabelling_map(permutation, design)
        covariance = matrix @ np.asarray(fit.covariance, dtype=np.float64) @ matrix.T
        blocks = getattr(fit, "conditional_group_covariances", None)
        expansion = design.expansion
        if blocks is not None and expansion is not None:
            columns = np.asarray(expansion.columns, dtype=np.intp)
            block = self.relabelling_map(permutation)[np.ix_(columns, columns)]
            blocks = np.einsum("ij,gjk,lk->gil", block, np.asarray(blocks), block, optimize=True)
        return replace(
            fit,
            estimates=matrix @ estimates,
            standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            covariance=covariance,
            **({} if blocks is None else {"conditional_group_covariances": blocks}),
        )

    def _initial_points(
        self,
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        penalty = np.zeros(features.shape[1], dtype=np.float64)
        penalty[1:] = self.l2

        def glm_objective(coefficients: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            linear = features @ coefficients
            loss = np.logaddexp(0.0, linear).sum() - outcomes @ linear
            loss += 0.5 * float(np.sum(penalty * coefficients**2))
            gradient = features.T @ (expit(linear) - outcomes) + penalty * coefficients
            return float(loss), gradient

        base_result = minimize(
            glm_objective,
            np.zeros(features.shape[1]),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        base = np.asarray(base_result.x, dtype=np.float64)
        persistent = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(persistent, 0.9)
        transition_logits = (
            _encode_probability_rows(persistent).ravel()
            if not self.is_dynamic
            else _encode_ilr_rows(persistent).ravel()
        )
        transition_slopes = np.zeros(
            self.n_states * (self.n_states - 1) * len(self.transition_coefficient_names),
            dtype=np.float64,
        )
        generator = np.random.default_rng(self.random_seed)
        label_index = self.coefficient_names.index(self.label_by)
        starts: list[NDArray[np.float64]] = []
        for restart in range(self.n_restarts):
            emissions = np.tile(base, (self.n_states, 1))
            emissions[:, label_index] += np.linspace(-0.75, 0.75, self.n_states)
            scale = 0.05 if restart == 0 else 0.35
            emissions += generator.normal(0.0, scale, emissions.shape)
            initial_logits = generator.normal(0.0, scale, self.n_states - 1)
            transitions = transition_logits + generator.normal(
                0.0, scale, self.n_states * (self.n_states - 1)
            )
            slopes = transition_slopes + generator.normal(
                0.0, scale * 0.25, transition_slopes.shape
            )
            starts.append(np.concatenate((emissions.ravel(), initial_logits, transitions, slopes)))
        return tuple(starts)

    def _likelihood_gradient(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        transition_features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative marginal log likelihood and its gradient, with no priors.

        Split out of :meth:`_objective_gradient` because the bounded-coordinate contract
        asks for a likelihood and carries priors separately: a ridge belongs in
        :meth:`penalty_matrix`, where a combinator can widen it, and a term that is neither
        per-row nor quadratic belongs nowhere, which is what
        :attr:`bounded_coordinate_refusal` says about ``stickiness``.
        """

        components = self._unpack_components(vector)
        emissions = components.emission_coefficients
        initial = components.initial_probabilities
        transition = components.transition_probabilities(transition_features)
        linear = features @ emissions.T
        emission_log_probability = outcomes[:, None] * -np.logaddexp(0.0, -linear) + (
            1.0 - outcomes[:, None]
        ) * -np.logaddexp(0.0, linear)
        posterior = _forward_backward(
            np.log(initial),
            np.log(transition),
            emission_log_probability,
            sessions,
        )
        gamma = posterior.state_probabilities

        emission_gradient = np.empty_like(emissions)
        probabilities = expit(linear)
        for state in range(self.n_states):
            emission_gradient[state] = features.T @ (
                gamma[:, state] * (probabilities[:, state] - outcomes)
            )
        initial_gradient = len(sessions) * initial[:-1] - posterior.initial_counts[:-1]
        if not self.is_dynamic:
            stationary = components.transition_matrix
            departures = posterior.transition_counts.sum(axis=1)
            transition_gradient = (
                departures[:, None] * stationary[:, :-1] - posterior.transition_counts[:, :-1]
            )
            transition_blocks = (transition_gradient.ravel(),)
        else:
            basis = _ilr_basis(self.n_states)
            expected = np.asarray(posterior.transition_expectations)
            departures = expected.sum(axis=2)
            residual = departures[:, :, None] * transition - expected
            contrast_residual = np.einsum("tij,jc->tic", residual, basis, optimize=True)
            intercept_gradient = contrast_residual.sum(axis=0)
            slope_gradient = np.einsum(
                "tic,tp->icp", contrast_residual, transition_features, optimize=True
            )
            transition_blocks = (intercept_gradient.ravel(), slope_gradient.ravel())
        gradient = np.concatenate((emission_gradient.ravel(), initial_gradient, *transition_blocks))
        return -posterior.log_likelihood, gradient

    def _objective_gradient(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        transition_features: NDArray[np.float64] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        if transition_features is None:
            transition_features = np.empty((len(outcomes), 0), dtype=np.float64)
        loss, gradient = self._likelihood_gradient(
            vector, features, transition_features, outcomes, sessions
        )
        components = self._unpack_components(vector)
        emissions = components.emission_coefficients
        transition = components.transition_matrix
        n_coefficients = len(self.coefficient_names)
        penalty = np.zeros_like(emissions)
        penalty[:, 1:] = self.l2 * emissions[:, 1:]
        loss = loss + 0.5 * self.l2 * float(np.sum(emissions[:, 1:] ** 2))
        gradient[: self.n_states * n_coefficients] += penalty.ravel()
        if self.is_dynamic and self.transition_l2:
            n_slopes = self.n_states * (self.n_states - 1) * len(self.transition_coefficient_names)
            slopes = np.asarray(vector[-n_slopes:], dtype=np.float64)
            loss += 0.5 * self.transition_l2 * float(slopes @ slopes)
            gradient[-n_slopes:] += self.transition_l2 * slopes
        if self.stickiness:
            # A sticky Dirichlet prior adds kappa pseudo-counts to self-transitions.
            # Constants independent of the parameters are omitted from the MAP objective.
            loss -= self.stickiness * float(np.sum(np.log(np.diag(transition))))
            sticky_gradient = self.stickiness * transition[:, :-1]
            for state in range(self.n_states - 1):
                sticky_gradient[state, state] -= self.stickiness
            gradient[self.n_states * n_coefficients + self.n_states - 1 :] += (
                sticky_gradient.ravel()
            )
        return float(loss), gradient

    def _posterior_state_probabilities(
        self,
        vector: NDArray[np.float64],
        features: NDArray[np.float64],
        transition_features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
    ) -> NDArray[np.float64]:
        components = self._unpack_components(vector)
        linear = features @ components.emission_coefficients.T
        emission = outcomes[:, None] * -np.logaddexp(0.0, -linear) + (
            1.0 - outcomes[:, None]
        ) * -np.logaddexp(0.0, linear)
        log_initial = np.log(components.initial_probabilities)
        log_transition = np.log(components.transition_probabilities(transition_features))
        return _forward_backward(
            log_initial,
            log_transition,
            emission,
            sessions,
        ).state_probabilities

    def _filtered_state_probabilities(
        self,
        features: NDArray[np.float64],
        transition_features: NDArray[np.float64],
        outcomes: NDArray[np.float64],
        sessions: tuple[tuple[int, ...], ...],
        components: GLMHMMParameters,
    ) -> FilteredStateProbabilities:
        linear = features @ components.emission_coefficients.T
        emission_log = outcomes[:, None] * -np.logaddexp(0.0, -linear) + (
            1.0 - outcomes[:, None]
        ) * -np.logaddexp(0.0, linear)
        predictive = np.empty((len(outcomes), self.n_states), dtype=np.float64)
        filtered = np.empty_like(predictive)
        transition = components.transition_probabilities(transition_features)
        for session_indices in sessions:
            prior = np.asarray(components.initial_probabilities)
            for position, index in enumerate(session_indices):
                predictive[index] = prior
                log_weight = np.log(prior) + emission_log[index]
                posterior = np.exp(log_weight - logsumexp(log_weight))
                filtered[index] = posterior
                if position + 1 < len(session_indices):
                    prior = posterior @ transition[session_indices[position + 1]]
        return FilteredStateProbabilities(predictive=predictive, filtered=filtered)

    def _canonicalize_components(
        self,
        components: GLMHMMParameters,
    ) -> tuple[GLMHMMParameters, tuple[int, ...]]:
        label_index = self.coefficient_names.index(self.label_by)
        other = tuple(index for index in range(len(self.coefficient_names)) if index != label_index)
        permutation = tuple(
            sorted(
                range(self.n_states),
                key=lambda state: (
                    float(components.emission_coefficients[state, label_index]),
                    *(float(components.emission_coefficients[state, index]) for index in other),
                ),
            )
        )
        indices = np.asarray(permutation, dtype=np.intp)
        canonical = GLMHMMParameters(
            initial_probabilities=components.initial_probabilities[indices],
            transition_matrix=components.transition_matrix[np.ix_(indices, indices)],
            emission_coefficients=components.emission_coefficients[indices],
            coefficient_names=self.coefficient_names,
            transition_effects=np.asarray(components.transition_effects)[indices][:, indices, :],
            transition_predictor_names=components.transition_predictor_names,
        )
        return canonical, permutation

    def _pack_components(self, components: GLMHMMParameters) -> NDArray[np.float64]:
        transition_intercepts = (
            _encode_probability_rows(components.transition_matrix)
            if not self.is_dynamic
            else _encode_ilr_rows(components.transition_matrix)
        )
        blocks = [
            components.emission_coefficients.ravel(),
            _encode_probability_vector(components.initial_probabilities),
            transition_intercepts.ravel(),
        ]
        if self.is_dynamic:
            effects = np.asarray(components.transition_effects, dtype=np.float64)
            blocks.append(
                np.einsum("jc,ijp->icp", _ilr_basis(self.n_states), effects, optimize=True).ravel()
            )
        return np.concatenate(blocks)

    def _unpack_components(self, vector: Sequence[float]) -> GLMHMMParameters:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(self.parameter_names),) or not np.all(np.isfinite(values)):
            raise ValueError("parameter vector has the wrong shape or contains non-finite values")
        n_coefficients = len(self.coefficient_names)
        emission_end = self.n_states * n_coefficients
        initial_end = emission_end + self.n_states - 1
        emissions = values[:emission_end].reshape(self.n_states, n_coefficients)
        initial = _decode_reference_logits(values[emission_end:initial_end])
        transition_end = initial_end + self.n_states * (self.n_states - 1)
        transition_logits = values[initial_end:transition_end].reshape(
            self.n_states, self.n_states - 1
        )
        if not self.is_dynamic:
            transition = np.vstack([_decode_reference_logits(row) for row in transition_logits])
            effects = np.zeros((self.n_states, self.n_states, 0), dtype=np.float64)
        else:
            transition = _decode_ilr_rows(transition_logits)
            coordinates = values[transition_end:].reshape(
                self.n_states,
                self.n_states - 1,
                len(self.transition_coefficient_names),
            )
            effects = np.einsum(
                "jc,icp->ijp", _ilr_basis(self.n_states), coordinates, optimize=True
            )
        return GLMHMMParameters(
            initial_probabilities=initial,
            transition_matrix=transition,
            emission_coefficients=emissions,
            coefficient_names=self.coefficient_names,
            transition_effects=effects,
            transition_predictor_names=self.transition_coefficient_names,
        )


@dataclass(frozen=True, slots=True)
class _GLMHMMRowObjective:
    """The marginal negative log likelihood as a function of one coordinate per trial.

    Session-blocked rather than trial-separable, and the block structure *is* the model. A
    GLM-HMM's likelihood is a forward recursion in which a trial's contribution depends on
    every earlier trial of its session, so no per-row score exists; what does exist is one
    score per session, exact for the one coordinate that session's rows share. That is
    precisely the relaxation :attr:`~behavio.contracts.bounded.RowObjective.row_blocks`
    declares, and it is why ``hierarchical(model, over="subject")`` is expressible while
    ``over`` anything that cuts a session is refused by
    :func:`~behavio.contracts.bounded.group_blocks_respect`.

    The gradient is the model's own analytic one -- forward-backward state and transition
    expectations, the same arithmetic :meth:`BernoulliGLMHMM.fit` differentiates -- evaluated
    on one session at a time and spread evenly over that session's rows. Only the block sum
    is defined, and every map a combinator builds is constant within a block, so how it is
    spread cannot be observed.
    """

    model: BernoulliGLMHMM
    blocks: tuple[
        tuple[
            NDArray[np.intp],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        ...,
    ]
    row_blocks: NDArray[np.intp]
    n_rows: int

    @property
    def n_parameters(self) -> int:
        """The width of one row's coordinate."""

        return len(self.model.parameter_names)

    def value_and_gradient(self, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log likelihood and its gradient in the row coordinates."""

        values = validated_row_coefficients(
            rows, n_rows=self.n_rows, n_parameters=self.n_parameters, what="row coordinates"
        )
        block_constant_coordinates(values, self.row_blocks, what="a GLM-HMM's parameters")
        total = 0.0
        gradient = np.zeros_like(values)
        for index, features, transition_features, outcomes in self.blocks:
            whole = (tuple(range(len(index))),)
            value, block_gradient = self.model._likelihood_gradient(
                values[index[0]], features, transition_features, outcomes, whole
            )
            total += value
            gradient[index] = block_gradient / len(index)
        return float(total), gradient


def _runner_up_cost(cost: NDArray[np.float64], mapping: NDArray[np.int64]) -> float:
    """The best assignment cost that gives up at least one edge of the winning one."""

    alternatives: list[float] = []
    for row, column in enumerate(mapping):
        constrained = cost.copy()
        constrained[row, column] = np.inf
        rows, columns = linear_sum_assignment(constrained)
        alternatives.append(float(constrained[rows, columns].sum()))
    finite = [value for value in alternatives if np.isfinite(value)]
    return min(finite) if finite else float(cost[np.arange(len(mapping)), mapping].sum())


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _decode_reference_logits(logits: Sequence[float]) -> NDArray[np.float64]:
    coordinates = np.append(np.asarray(logits, dtype=np.float64), 0.0)
    return np.exp(coordinates - logsumexp(coordinates))


def _encode_probability_vector(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(probabilities[:-1]) - np.log(probabilities[-1])


def _encode_probability_rows(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(probabilities[:, :-1]) - np.log(probabilities[:, -1:])


def _ilr_basis(n_states: int) -> NDArray[np.float64]:
    """Return a deterministic orthonormal basis of the simplex's log-contrast space.

    This is the Helmert sub-matrix, oriented only to give parameter names a stable order.
    Its columns are orthonormal and orthogonal to the all-ones vector, so relabelling states
    rotates coordinates without changing Euclidean lengths or isotropic Gaussian priors.
    """

    basis = np.zeros((n_states, n_states - 1), dtype=np.float64)
    for contrast in range(n_states - 1):
        denominator = np.sqrt((contrast + 1) * (contrast + 2))
        basis[: contrast + 1, contrast] = 1.0 / denominator
        basis[contrast + 1, contrast] = -(contrast + 1) / denominator
    return basis


def _encode_ilr_rows(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=np.float64)
    return np.log(values) @ _ilr_basis(values.shape[1])


def _decode_ilr_rows(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(coordinates, dtype=np.float64)
    logits = values @ _ilr_basis(values.shape[1] + 1).T
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def _minimum_pairwise_distance(values: NDArray[np.float64]) -> float:
    return float(
        min(
            np.linalg.norm(values[left] - values[right])
            for left, right in combinations(range(len(values)), 2)
        )
    )


def _forward_backward(
    log_initial: NDArray[np.float64],
    log_transition: NDArray[np.float64],
    emission_log_probability: NDArray[np.float64],
    sessions: tuple[tuple[int, ...], ...],
) -> _PosteriorStatistics:
    n_states = len(log_initial)
    gamma = np.empty_like(emission_log_probability)
    initial_counts = np.zeros(n_states, dtype=np.float64)
    transition_counts = np.zeros((n_states, n_states), dtype=np.float64)
    transition_expectations = np.zeros(
        (len(emission_log_probability), n_states, n_states), dtype=np.float64
    )
    transitions = np.asarray(log_transition, dtype=np.float64)
    if transitions.ndim not in (2, 3):
        raise ValueError("log_transition must be one matrix or one matrix per trial")
    if transitions.ndim == 2 and transitions.shape != (n_states, n_states):
        raise ValueError("a stationary transition matrix must have one row and column per state")
    if transitions.ndim == 3 and transitions.shape != (
        len(emission_log_probability),
        n_states,
        n_states,
    ):
        raise ValueError("dynamic transitions must contain one square matrix per trial")
    log_likelihood = 0.0
    for session_indices in sessions:
        indices = np.asarray(session_indices, dtype=np.intp)
        emission = emission_log_probability[indices]
        n_trials = len(indices)
        alpha = np.empty((n_trials, n_states), dtype=np.float64)
        beta = np.zeros_like(alpha)
        alpha[0] = log_initial + emission[0]
        for trial in range(1, n_trials):
            transition = transitions if transitions.ndim == 2 else transitions[indices[trial]]
            alpha[trial] = emission[trial] + logsumexp(
                alpha[trial - 1, :, None] + transition,
                axis=0,
            )
        session_likelihood = float(logsumexp(alpha[-1]))
        log_likelihood += session_likelihood
        for trial in range(n_trials - 2, -1, -1):
            transition = transitions if transitions.ndim == 2 else transitions[indices[trial + 1]]
            beta[trial] = logsumexp(
                transition + emission[trial + 1][None, :] + beta[trial + 1][None, :],
                axis=1,
            )
        session_gamma = np.exp(alpha + beta - session_likelihood)
        gamma[indices] = session_gamma
        initial_counts += session_gamma[0]
        for trial in range(1, n_trials):
            transition = transitions if transitions.ndim == 2 else transitions[indices[trial]]
            expected = np.exp(
                alpha[trial - 1, :, None]
                + transition
                + emission[trial][None, :]
                + beta[trial][None, :]
                - session_likelihood
            )
            transition_counts += expected
            transition_expectations[indices[trial]] = expected
    return _PosteriorStatistics(
        log_likelihood=log_likelihood,
        state_probabilities=gamma,
        initial_counts=initial_counts,
        transition_counts=transition_counts,
        transition_expectations=transition_expectations,
    )
