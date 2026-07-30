"""A Behavio estimator backed by dynamax's switching linear autoregression.

Behavio's only latent-state model is :class:`~behavio.models.glm_hmm.BernoulliGLMHMM`: one
state count, Bernoulli emissions, stationary transitions, a *discrete* observation. Nothing
in the package describes a continuous behavioural time series -- running speed, pupil
diameter, licking rate, a kinematic component of a pose -- as a switch between regimes, and
nothing describes a regime that has its own *dynamics* rather than its own mean. dynamax
does, it is MIT, it is maintained, and its inference is exact. This module is the wrapper
that lets a Behavio user fit it through the whole falsification stack.

Which family, and why this one
------------------------------
A Gaussian-emission HMM was the obvious on-ramp and is *included* -- ``num_lags=0`` is
exactly that model -- but it is not what earns the wrap. The default is ``num_lags=1``: a
**switching linear autoregression**, where each latent state carries its own autoregressive
coefficients, its own offset and its own innovation variance,

    y_t = b_k + sum_l W_{k,l} y_{t-l} + e_t,   e_t ~ N(0, s_k),   k = z_t.

Three reasons that is the family to wrap rather than the plain HMM.

*It is the one nothing else here can express.* Behavio already models history dependence in
choice, through ``lag()`` and ``kernel()`` in its own formula language. It models regime
switching, through the GLM-HMM. It has never been able to write down a *regime that is
itself a dynamical system*, and that is the standard description of continuous behaviour --
a mouse that is running has a different autocorrelation from one that is grooming, not
merely a different mean speed. A Gaussian HMM forced to describe that data answers with
extra states whose only job is to tile the autocorrelation, which is the failure mode the
switching autoregression exists to avoid.

*The nesting is the falsification.* ``num_lags=0`` is the same model with every ``W`` fixed
at zero, its parameter names are a strict subset, and both are fitted by the same code
through the same contract. ``compare_models({"ar": ..., "hmm": ...}, study, splits)`` is
therefore a targeted competitor rather than a different package's answer -- which is what
``AGENTS.md`` demands before a latent state is interpreted at all.

*It is where the sequence layout stops being bookkeeping.* See below: for a plain HMM only
the state chain crosses a session boundary; for an autoregression the *emissions* do, and a
wrapper that flattens the study loses the distinction silently.

What is wrapped, and what is added
----------------------------------
dynamax's E-step, its M-step, its forward filter, its backward smoother, its Viterbi decoder
and its sampler are all wrapped and none of them is reimplemented. dynamax has no
``fit(data) -> result`` object at all -- it is ``hmm.initialize(key)``, then a parameter
pytree, then more pytrees -- so everything the Behavio contract needs is the wrapper's:

- **A ragged EM loop, because dynamax's cannot be one.** See "Padding" below. This is the
  one place the wrapper owns control flow rather than arithmetic, and it reproduces
  :meth:`dynamax.ssm.SSM.fit_em` to floating-point equality on the equal-length case the
  two agree about, which the test suite asserts.
- **A covariance, which EM does not hand you.** dynamax reports a parameter pytree and a
  log-joint trace. The wrapper differentiates the objective EM maximised, twice, with jax,
  and delta-methods the result onto the natural coordinates the estimates are reported in.
  See :meth:`DynamaxSwitchingAutoregression._curvature`.
- **A stationarity check, because EM never fails.** EM increases its objective at every
  iteration and stops when it is told to, so "it finished" says nothing. The exact gradient
  of the log joint at the reported estimate does, and it is free once the Hessian is taken.
- **A canonical state order and a report of how identified it is.** See "Label switching".
- **A filtered prediction and a smoothed description, kept apart.** See below; this is the
  point of the wrap.
- **Behavio's parameter names.** dynamax spells a state's offset ``means`` under
  :class:`~dynamax.hidden_markov_model.GaussianHMM` and ``biases`` under
  :class:`~dynamax.hidden_markov_model.LinearAutoregressiveHMM`, and stores the emission
  variance as a one-by-one covariance matrix. :data:`PARAMETER_CORRESPONDENCE` is the map,
  and it is exact.

Filtered prediction versus smoothed description
-----------------------------------------------
This is the first model in Behavio that legitimately declares
:attr:`~behavio.contracts.PredictionMode.SMOOTHED`, and getting the distinction right is
the reason to wrap a state-space package at all. ``AGENTS.md`` requires the two to be
distinguished; until now every model in the package declared ``FILTERED`` and the
requirement was satisfied by there being nothing to confuse it with.

- :attr:`~behavio.contracts.PredictionMode.FILTERED` mixes the emission densities under
  ``predicted_probs``, dynamax's ``p(z_t | y_{1:t-1})``. The resulting row density is
  ``p(y_t | y_{1:t-1})``, the genuine one-step-ahead predictive: it uses no observation at
  or after *t*, it is what a held-out score should mean, and its logs sum **exactly** to
  the marginal log likelihood dynamax's own filter reports. The test suite asserts that
  identity to 1e-9 rather than describing it.
- :attr:`~behavio.contracts.PredictionMode.SMOOTHED` mixes the same emission densities
  under ``smoothed_probs``, ``p(z_t | y_{1:T})``. That is a *description of the recorded
  session*, conditioned on the whole sequence including trial *t* itself and everything
  after it. It is the right object for "which regime was the animal in at trial 40" and the
  wrong object for every score, which is why it is never the default and why
  :meth:`~DynamaxSwitchingAutoregression.pointwise_log_prob` says so in its own docstring.
- ``predicted_probs`` rather than ``filtered_probs`` is what the filtered mode uses, and
  the difference matters. ``filtered_probs`` is ``p(z_t | y_{1:t})``, which is admissible
  under Behavio's definition of filtering -- it reads nothing after *t* -- but using it to
  predict ``y_t`` conditions the prediction on the observation being predicted. It would
  pass the conformance harness and be worthless, so the wrapper does not use it for
  prediction; it is reported separately by
  :meth:`~DynamaxSwitchingAutoregression.state_probabilities`, where it is the correct
  answer to a different question.

:func:`behavio.adapters.check_behaviour_estimator` decides all of this behaviourally rather
than reading the declaration: it relabels the second half of every trial sequence, holds the
fit fixed, and requires the filtered output on the first half to be unchanged and the
smoothed output on the same rows to move. Both directions fire for this model, which is the
first time the check has had a model that could fail it either way.

Shape: why ``sequence_layout``, and why not a padded tensor
------------------------------------------------------------
A :class:`~behavio.trials.Study` is a flat columnar table in source row order; dynamax wants
per-sequence arrays. :func:`behavio.trials.sequence_layout` is built from
:meth:`~behavio.trials.Study.chronological_indices`, so it cannot disagree with the
package's own chronology, and ``layout.join(layout.split(v)) == v`` exactly. Everything this
wrapper hands dynamax comes from ``split`` and everything it hands back comes from ``join``.

The round-trip invariant is necessary and it is not sufficient, and it is worth being
precise about which failures it catches. It catches a *misalignment*: a prediction
concatenated in sequence order and returned as though the study had been sorted, which is
silently wrong whenever the source table is not already chronological. It does not catch a
*contamination*, because a contaminated block has exactly the right length and joins back
perfectly. Two contaminations are live here and both are handled outside the invariant:

*Autoregressive inputs must be built per sequence.* ``hmm.compute_inputs`` lags an array,
and lagging the flat study would make the first trial of one session a function of the last
trial of the previous one -- across a night, or across animals. The wrapper calls it on each
block, so each sequence's first ``num_lags`` rows are conditioned on dynamax's own zero
history. This is the concrete reason the autoregressive family is where the layout earns its
place: for ``num_lags=0`` only the state chain resets at a boundary, and a wrapper that got
it wrong would be wrong more quietly.

*Padding is not safe here, so there is none.* :meth:`dynamax.ssm.SSM.fit_em` vmaps its
E-step over a ``(n_sequences, num_timesteps, emission_dim)`` batch and takes no mask, so
zero-padding ragged sequences to a common length feeds the forward-backward pass invented
observations at an invented emission value and returns their sufficient statistics to the
M-step. It is not a small effect and it is measured rather than feared: on the test suite's
four sessions of 40, 33, 26 and 19 trials, zero-padding moves a fitted emission offset by
**0.79** against the same data fitted ragged, on states whose true offsets are one apart.
Behavioural sessions are ragged essentially always. The wrapper therefore partitions the layout by
length, vmaps dynamax's own ``e_step`` within each partition, concatenates the sufficient
statistics -- which are sums over time and so are length-independent for every component
here -- and calls dynamax's own ``m_step`` once. That is ``fit_em``'s loop with its batching
replaced, and on an equal-length batch the two agree to floating-point equality.

Label switching
---------------
A hidden Markov model's states are unidentified up to permutation: relabelling them and
permuting the parameters accordingly gives exactly the same likelihood. The package has
already worked out that there are two different answers to this and that they are not
interchangeable. :func:`~behavio.models.align_latent_states` aligns *inferred posteriors
against simulated truth* and therefore cannot run on data. The hierarchical GLM-HMM keeps
labels identified *during* a joint fit, by anchoring each group's emissions to the
population's, and then reports whether the anchor bit.

Neither applies to a single-group EM fit of a foreign model, so this wrapper does the third
thing, which is what :class:`~behavio.models.glm_hmm.BernoulliGLMHMM` itself does for a
single fit: **canonicalise afterwards, and report how identified the canonical order is.**
States are sorted by increasing emission bias, ties broken by the rest of the emission row;
the permutation is applied to the initial distribution, to both axes of the transition
matrix and to every emission parameter, and it is recorded on the fit as
``canonical_permutation``. Because the log joint is exactly invariant under simultaneous
relabelling, the covariance is computed *after* the permutation rather than permuted, which
needs no relabelling map at all -- the GLM-HMM needs one because its coordinates are
reference-category logits and relabelling re-references them; a probability simplex reported
in full has no reference to move.

What canonicalisation does not do is make the order *meaningful*, and the fit says so:
``label_order_gap`` is the smallest distance between adjacent canonical biases and
``label_ambiguous`` is true when it falls below ``label_tolerance``. Two states with
indistinguishable biases have an ordering decided by numerical noise, and reading "state 0"
as a behaviour across two fits of the same animal is then exactly the confident nonsense a
latent-state model invites. ``state_occupancy`` and ``low_occupancy`` report the other half
of the same problem: a state nothing is assigned to has no identified parameters at all.

Where dynamax strains the contract
----------------------------------
Five places. None was fixed by loosening what Behavio asks for.

*jax is 32-bit by default, and this wrapper turns that off process-wide.* A forward-backward
pass in float32 accumulates log probabilities badly over a few hundred trials, and an
observed-information Hessian is not meaningfully computable in it. jax 0.11 removed the
``enable_x64`` context manager, so there is no scoped form of the switch;
:func:`behavio.foreign._optional.require_dynamax` sets ``jax_enable_x64`` globally and says
so. Nothing else in Behavio uses jax, so the only code affected is the caller's.

*One emission column, because Behavio has no shape for more.* dynamax's Gaussian and
autoregressive families are multivariate, and a switching *vector* autoregression is the
model most of this literature actually fits. :class:`~behavio.contracts.DensityPrediction`
tabulates a density over one continuous coordinate, so a two-dimensional emission has
nothing to be returned as. Fitting and pointwise scoring would both work unchanged; only
``predict`` has nowhere to go. That is a gap in Behavio's prediction vocabulary, named here
and in ``docs/foreign-models.md`` rather than papered over by reporting one margin of a
joint density and calling it the prediction.

*The predictive density is unbounded and the grid is not.* ``predict`` tabulates on a grid
fixed at fit time -- derived from the training range and the fitted variances, and retained
on the fit -- because a grid derived from the study being predicted would make an early
row's reported density a function of later rows, which is the leak the conformance harness
exists to catch. A held-out row outside that range therefore loses tail mass, which
:attr:`~behavio.contracts.DensityPrediction.total_mass` reports per row rather than
normalising away, and ``grid_truncation`` summarises on the fit.
:meth:`~DynamaxSwitchingAutoregression.pointwise_log_prob` is computed in closed form and
never off the grid, so no score is a function of the tabulation; ``grid_log_density_gap``
reports how far the tabulation is from the closed form on the training rows, which is the
same number PyDDM's wrapper calls ``interpolation_gap`` and is here a property of the report
rather than of the score.

*EM cannot fail, so "converged" had to be measured.* :meth:`dynamax.ssm.SSM.fit_em` runs a
fixed number of iterations and monotonically increases its objective; there is no stopping
rule to have fired and no status to read. The wrapper therefore reports **exact
stationarity**: the gradient of the log joint with respect to dynamax's own unconstrained
parameters, at the estimate, must have norm at or below ``gradient_tolerance``. That is a
stronger claim than any optimizer flag and it costs nothing, because the same
differentiation produces the Hessian. An under-iterated fit fails it, and the fit's audit
fails with it.

*Observed information is reported on the coordinates that were estimated, and the reported
covariance is singular by construction.* The natural parameter vector is
over-parameterised -- ``K`` initial probabilities that sum to one, ``K`` transition rows that
each sum to one -- so the delta-method covariance on it has rank at most that of the
unconstrained problem and is exactly singular along the sum-to-one directions. That is
correct rather than a defect: the variance of a quantity that cannot move is zero. It does
mean the reported ``hessian_condition`` is the condition number of the **unconstrained
observed information**, the matrix actually inverted, and not of the natural covariance,
whose condition number would be an artefact of the constraint and would trip an
ill-conditioning warning on every healthy fit.

Licence, and the jax conflict
-----------------------------
dynamax is MIT, as Behavio is. Its dependency closure is permissive -- jax and jaxlib
Apache-2.0, optax Apache-2.0, ``tfp-nightly`` Apache-2.0, scikit-learn BSD-3-Clause,
``jaxtyping`` MIT, ``fastprogress`` Apache-2.0 -- so ``pip install 'behavio[dynamax]'``
changes no obligation. Two things about that closure are worth knowing before installing it
and are recorded in ``docs/foreign-models.md``: dynamax depends on ``tfp-nightly`` rather
than a released TensorFlow Probability, so the extra pulls a dated nightly build whose
reproducibility rests on ``uv.lock``; and it brings a modern jax, which cannot share an
environment with ``pyhgf``'s ``jax<0.4.32`` pin at any version of either.

Import safety
-------------
This module is named ``dynamax`` inside ``behavio.foreign``. Python 3 resolves imports
absolutely, and nothing here writes ``import dynamax`` at module scope in any case: the
dependency is reached only through
:func:`behavio.foreign._optional.require_dynamax`, so the two names cannot collide and
``import behavio.foreign.dynamax`` works with the extra uninstalled.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.audit import FitDiagnostics
from behavio.contracts.estimator import (
    LOG_DENSITY_FLOOR,
    DensityPrediction,
    DerivedQuantity,
    FitResult,
    ModelDataError,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.foreign._optional import dynamax_version, jax_version, require_dynamax
from behavio.foreign._shared import (
    ForeignCurvature,
    condition_number,
    quiet_foreign_package,
    unknown_curvature,
)
from behavio.models._kernels.introspection import ERROR, WARNING, Describable, ModelFinding
from behavio.trials import (
    REQUIRED_COLUMNS,
    SequenceGrouping,
    SequenceLayout,
    Study,
    sequence_layout,
)

PARAMETER_CORRESPONDENCE: Final = MappingProxyType(
    {
        "initial[k]": "ParamsStandardHMMInitialState.probs[k]: equal",
        "transition[j->k]": "ParamsStandardHMMTransitions.transition_matrix[j, k]: equal",
        "state[k].bias": (
            "emissions.biases[k, 0] with lags, emissions.means[k, 0] without: equal, and with "
            "no lags it is the state's mean rather than an offset"
        ),
        "state[k].lag<l>": "emissions.weights[k, 0, l - 1]: equal",
        "state[k].variance": "emissions.covs[k, 0, 0]: equal, a one-by-one covariance matrix",
    }
)
"""The exact correspondence between Behavio's parameter names and dynamax's pytree paths.

Recovery is only worth running when the simulator and the fitter mean the same thing by the
same name. Two of these five would be easy to get wrong and are the reason the map is
published rather than assumed: dynamax spells the emission offset ``means`` for the plain
Gaussian family and ``biases`` for the autoregressive one, and it stores a scalar emission
variance as a one-by-one covariance *matrix*, so a wrapper that read ``covs`` as a standard
deviation would produce a clean-looking recovery diagonal for the wrong quantity.
"""

INITIALISATIONS: Final = ("kmeans", "prior")
"""How EM's starting point is chosen, and why it is part of the model rather than a detail.

EM finds a local optimum of a multimodal objective, so the starting point is part of the
procedure that produced the estimate and is in the signature. ``kmeans`` clusters the
training emissions and is dynamax's data-driven initialiser; it is fitted *inside* whatever
fold it is handed, which is what ``AGENTS.md`` requires of learned preprocessing. ``prior``
draws from the model's own prior and reads no data at all, which is slower to converge and
is the honest choice when a fold is too small for clustering to mean anything.
"""

_MAX_GRID_POINTS: Final = 20_001
"""Ceiling on the tabulated density's resolution.

The grid must resolve the *narrowest* fitted state, so a fit in which one state's variance
has collapsed towards zero would ask for an unboundedly fine grid. That is a statement about
the fit rather than about the tabulation, so the grid stops here and
:meth:`DynamaxSwitchingAutoregression.predict_density` refuses with the state named instead
of returning a density that integrates to more than one.
"""

_MASS_CEILING: Final = 1.0 + 1e-4
_PROBABILITY_WARNING: Final = 1e-4


@dataclass(frozen=True, slots=True)
class SwitchingStateProbabilities:
    """Three state posteriors over the same rows, in the study's own source order.

    Kept apart because they answer three different questions and are routinely confused.
    ``predictive`` is ``p(z_t | y_{1:t-1})`` and is what a one-step-ahead prediction of
    ``y_t`` may condition on. ``filtered`` is ``p(z_t | y_{1:t})``: still filtered by
    Behavio's definition, since it reads nothing after *t*, but it has already seen the
    observation, so predicting ``y_t`` from it is circular. ``smoothed`` is
    ``p(z_t | y_{1:T})`` and is the best description of what the animal was doing, available
    only after the session ended.
    """

    predictive: NDArray[np.float64]
    filtered: NDArray[np.float64]
    smoothed: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = {
            "predictive": protected_array(self.predictive, dtype=np.float64),
            "filtered": protected_array(self.filtered, dtype=np.float64),
            "smoothed": protected_array(self.smoothed, dtype=np.float64),
        }
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("the three state posteriors must cover the same rows and states")
        shape = shapes.pop()
        if len(shape) != 2 or shape[1] < 2:
            raise ValueError("state posteriors must be (n_rows, n_states) with at least two states")
        for name, array in arrays.items():
            if not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"{name} state probabilities must be finite and non-negative")
            object.__setattr__(self, name, array)

    @property
    def n_states(self) -> int:
        """How many latent states the posteriors are over."""

        return int(self.predictive.shape[1])


@dataclass(frozen=True, slots=True)
class DynamaxFitResult(FitResult):
    """A dynamax fit with the foreign package's own evidence, and the wrapper's, retained.

    The common :class:`~behavio.contracts.FitResult` fields are the interoperability floor.
    What an EM fit of a latent-state model knows and those fields have nowhere to put stays
    here: which restart won and what the others reached, how far the last EM step still
    moved the objective, which permutation put the states in canonical order and how
    separated that order is, how much of the data each state accounts for, and the grid
    ``predict`` will tabulate on -- which is retained precisely so that ``predict`` is a
    function of the fit and not of the study it is scoring.
    """

    dynamax_version: str
    jax_version: str
    n_sequences: int
    restart_objectives: NDArray[np.float64]
    selected_restart: int
    final_em_increment: float
    canonical_permutation: tuple[int, ...]
    state_occupancy: NDArray[np.float64]
    label_order_gap: float
    label_ambiguous: bool
    low_occupancy: bool
    covariance_is_estimated: bool
    outcome_grid: NDArray[np.float64]
    grid_truncation: float
    grid_log_density_gap: float

    def __post_init__(self) -> None:
        FitResult.__post_init__(self)
        objectives = protected_array(self.restart_objectives, dtype=np.float64)
        occupancy = protected_array(self.state_occupancy, dtype=np.float64)
        grid = protected_array(self.outcome_grid, dtype=np.float64)
        permutation = tuple(int(state) for state in self.canonical_permutation)
        if not isinstance(self.dynamax_version, str) or not self.dynamax_version:
            raise ValueError("a dynamax fit must record the version that produced it")
        if objectives.ndim != 1 or not objectives.size:
            raise ValueError("restart objectives must contain one value per restart")
        if not 0 <= self.selected_restart < objectives.size:
            raise ValueError("selected_restart must identify one restart")
        if self.n_sequences < 1:
            raise ValueError("a dynamax fit must cover at least one trial sequence")
        if sorted(permutation) != list(range(len(permutation))) or len(permutation) < 2:
            raise ValueError("canonical_permutation must permute every latent state")
        if occupancy.shape != (len(permutation),) or np.any(occupancy < 0):
            raise ValueError("state_occupancy must hold one non-negative value per state")
        if not np.isclose(float(occupancy.sum()), 1.0, atol=1e-6):
            raise ValueError("state occupancy must sum to one")
        if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError("the outcome grid must be increasing with at least two points")
        if not np.isfinite(self.label_order_gap) or self.label_order_gap < 0:
            raise ValueError("label_order_gap must be finite and non-negative")
        object.__setattr__(self, "restart_objectives", objectives)
        object.__setattr__(self, "state_occupancy", occupancy)
        object.__setattr__(self, "outcome_grid", grid)
        object.__setattr__(self, "canonical_permutation", permutation)
        object.__setattr__(self, "final_em_increment", float(self.final_em_increment))
        object.__setattr__(self, "grid_truncation", float(self.grid_truncation))
        object.__setattr__(self, "grid_log_density_gap", float(self.grid_log_density_gap))


@dataclass(frozen=True, slots=True)
class _Components:
    """One parameter set in Behavio's natural coordinates, unpacked from the flat vector."""

    initial: NDArray[np.float64]
    transitions: NDArray[np.float64]
    biases: NDArray[np.float64]
    weights: NDArray[np.float64]
    variances: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DynamaxSwitchingAutoregression(Describable):
    """A hidden Markov model with state-specific Gaussian autoregressive emissions.

    ``outcome`` is the one continuous ``Study`` column the model describes; it has no
    default, because a continuous behavioural measurement has no canonical name and guessing
    one would be a modelling decision hidden in a signature.

    ``num_lags`` is the whole difference between the two models this class covers.
    ``num_lags=0`` is a Gaussian-emission HMM -- each state is a mean and a variance, and
    ``state[k].bias`` *is* that mean. ``num_lags>=1`` is a switching linear autoregression:
    each state additionally carries its own autoregressive coefficients, so a regime is a
    dynamical system rather than a level. The parameter names of the former are a strict
    subset of the latter's, which is what makes the pair a nested comparison rather than two
    unrelated candidates.

    ``sequence_grouping`` decides what the state chain resets at:
    :attr:`~behavio.trials.SequenceGrouping.SESSION` restarts the initial distribution and
    the autoregressive history at every session boundary, which is what a recording implies;
    :attr:`~behavio.trials.SequenceGrouping.SUBJECT` carries both across a subject's
    sessions in ``session_order``, which is the right choice only when the latent state
    genuinely persists between recordings.

    Everything that changes the numbers is a field and is in the :attr:`signature`,
    including the entire fitting procedure -- how EM was started, how many restarts, how
    many iterations, the seed -- and the tabulation ``predict`` reports on.
    """

    outcome: str
    n_states: int = 2
    num_lags: int = 1
    sequence_grouping: SequenceGrouping = SequenceGrouping.SESSION
    initialisation: str = "kmeans"
    n_restarts: int = 3
    em_iterations: int = 100
    random_seed: int = 0
    transition_stickiness: float = 0.0
    initial_concentration: float = 1.1
    transition_concentration: float = 1.1
    gradient_tolerance: float = 1e-4
    label_tolerance: float = 1e-3
    state_occupancy_warning: float = 0.01
    grid_points: int = 512
    grid_points_per_sd: float = 8.0
    grid_padding: float = 6.0

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ValueError("outcome must be a non-empty column name")
        if self.outcome in REQUIRED_COLUMNS:
            raise ValueError("outcome must not be one of the study's identity columns")
        if isinstance(self.n_states, bool) or not isinstance(self.n_states, int):
            raise ValueError("n_states must be an integer")
        if self.n_states < 2:
            raise ValueError("n_states must be at least two; one state is not a switching model")
        if isinstance(self.num_lags, bool) or not isinstance(self.num_lags, int):
            raise ValueError("num_lags must be an integer")
        if self.num_lags < 0:
            raise ValueError("num_lags must be non-negative; zero is a Gaussian-emission HMM")
        if self.initialisation not in INITIALISATIONS:
            raise ValueError(f"initialisation must be one of {list(INITIALISATIONS)}")
        for value, name in (
            (self.n_restarts, "n_restarts"),
            (self.em_iterations, "em_iterations"),
            (self.grid_points, "grid_points"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.grid_points < 8:
            raise ValueError("grid_points must be at least eight for a usable tabulation")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        for value, name in (
            (self.transition_stickiness, "transition_stickiness"),
            (self.label_tolerance, "label_tolerance"),
            (self.state_occupancy_warning, "state_occupancy_warning"),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for value, name in (
            (self.initial_concentration, "initial_concentration"),
            (self.transition_concentration, "transition_concentration"),
            (self.gradient_tolerance, "gradient_tolerance"),
            (self.grid_points_per_sd, "grid_points_per_sd"),
            (self.grid_padding, "grid_padding"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "sequence_grouping", SequenceGrouping(self.sequence_grouping))

    # -- identity ------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "dynamax-switching-autoregression"

    @property
    def signature(self) -> str:
        """A scientific fingerprint over a foreign latent-state configuration.

        Everything that changes the numbers is in it: the scored column, the state count and
        the lag order, what counts as one sequence, the priors on the two simplexes, the
        entire EM procedure including its initialiser and seed, the stationarity tolerance
        the convergence verdict is decided by, the label tolerance the canonical order is
        judged against, and the tabulation ``predict`` reports on.

        What is deliberately *out* of it: the installed dynamax and jax versions. The
        argument is :attr:`behavio.foreign.bambi.BambiRegression.signature`'s rather than
        :attr:`behavio.foreign.pyddm.PyDDMDriftDiffusion.signature`'s, and it turns on the
        same distinction. PyDDM's series is in its fingerprint because a first-passage
        density is *computed by a truncated series* whose term selection differs between
        releases, so the same parameters give different numbers. Forward filtering and
        backward smoothing are exact arithmetic on an exactly specified likelihood. What a
        dynamax release genuinely can change is the default initialisation of EM, and that is
        a declared field here rather than a library default. The exact versions are
        provenance and are carried on the fit as ``dynamax_version`` and ``jax_version``.
        """

        parts = [
            "backend=dynamax.em",
            f"outcome={self.outcome}",
            f"states={self.n_states}",
            f"lags={self.num_lags}",
            f"grouping={self.sequence_grouping.value}",
            f"init={self.initialisation}",
            f"restarts={self.n_restarts}",
            f"iterations={self.em_iterations}",
            f"seed={self.random_seed}",
            f"stickiness={self.transition_stickiness}",
            f"initial_concentration={self.initial_concentration}",
            f"transition_concentration={self.transition_concentration}",
            f"gradient_tolerance={self.gradient_tolerance}",
            f"label_tolerance={self.label_tolerance}",
            f"grid={self.grid_points}:{self.grid_points_per_sd}:{self.grid_padding}",
        ]
        return f"{self.model_name}[" + ";".join(parts) + "]"

    @property
    def state_names(self) -> tuple[str, ...]:
        """Canonical state labels, in the increasing-bias order the fit reports them in."""

        return tuple(f"state[{index}]" for index in range(self.n_states))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Every scalar parameter, in the order the estimate vector packs them.

        Deliberately the *full* simplexes rather than a reference-category chart. A
        reference-category logit is a smaller and better-conditioned coordinate, and
        :class:`~behavio.models.glm_hmm.BernoulliGLMHMM` uses one, but its reference state is
        chosen by label canonicalisation and so is a function of the data. Reporting the
        probabilities themselves means a reader of ``transition[0->1]`` is reading a
        transition probability rather than a contrast against whichever state happened to
        sort last, at the cost of a covariance that is singular along the sum-to-one
        directions -- which the module docstring states and which is the correct variance for
        a quantity that cannot move.
        """

        names = [f"initial[{state}]" for state in range(self.n_states)]
        names.extend(
            f"transition[{source}->{target}]"
            for source in range(self.n_states)
            for target in range(self.n_states)
        )
        names.extend(f"state[{state}].bias" for state in range(self.n_states))
        names.extend(
            f"state[{state}].lag{lag}"
            for state in range(self.n_states)
            for lag in range(1, self.num_lags + 1)
        )
        names.extend(f"state[{state}].variance" for state in range(self.n_states))
        return tuple(names)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        """None. An autoregression is a function of its own past and of nothing else.

        A stated empty declaration rather than an omission: the model reads the outcome
        column and the four identity columns that fix chronology, and no task variable at
        all. Adding a stimulus would make it an input-driven switching regression, which is
        a different dynamax family and a different wrapper.
        """

        return ()

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        """Both, and this is the first model in Behavio for which that is true.

        See the module docstring. ``FILTERED`` is the one-step-ahead predictive density
        under ``p(z_t | y_{1:t-1})``; ``SMOOTHED`` is the same emission mixture under
        ``p(z_t | y_{1:T})`` and is a description of a recorded session rather than a
        prediction of it. :func:`behavio.adapters.check_behaviour_estimator` verifies both
        claims behaviourally instead of taking either on trust.
        """

        return (PredictionMode.FILTERED, PredictionMode.SMOOTHED)

    @property
    def density_outcome(self) -> str:
        """The continuous column :meth:`predict_density` tabulates a density over."""

        return self.outcome

    @property
    def declared_priors(self) -> tuple[str, ...]:
        """dynamax's own conjugate priors, as :meth:`describe` reports them."""

        lines = [
            f"initial distribution: Dirichlet(concentration={self.initial_concentration:g})",
            f"transition rows: Dirichlet(concentration={self.transition_concentration:g}"
            + (
                f", diagonal stickiness={self.transition_stickiness:g})"
                if self.transition_stickiness
                else ")"
            ),
        ]
        if self.num_lags:
            lines.append("emissions: none; dynamax places no prior on an autoregressive state")
        else:
            lines.append("emissions: dynamax's normal-inverse-Wishart default")
        return tuple(lines)

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report the study-specific ways an EM fit of this model goes wrong, before it runs."""

        findings: list[ModelFinding] = []
        if self.outcome not in study.columns:
            return (
                ModelFinding(
                    code="missing_outcome_column",
                    severity=ERROR,
                    message=f"the study has no outcome column {self.outcome!r}",
                ),
            )
        try:
            values = np.asarray(study[self.outcome], dtype=np.float64)
        except (TypeError, ValueError):
            return (
                ModelFinding(
                    code="non_numeric_outcome",
                    severity=ERROR,
                    message=(
                        f"outcome {self.outcome!r} must be a continuous numeric column; a "
                        "Gaussian emission has no meaning for a categorical one"
                    ),
                ),
            )
        if not np.all(np.isfinite(values)):
            findings.append(
                ModelFinding(
                    code="nonfinite_outcome",
                    severity=ERROR,
                    message=(
                        f"outcome {self.outcome!r} contains non-finite values; dynamax has no "
                        "missing-observation mask, so they cannot be marginalised out"
                    ),
                )
            )
            return tuple(findings)
        spread = float(np.std(values))
        if spread <= 0:
            findings.append(
                ModelFinding(
                    code="constant_outcome",
                    severity=ERROR,
                    message=(
                        f"outcome {self.outcome!r} is constant, so every state's emission "
                        "variance collapses to zero and the likelihood is unbounded"
                    ),
                )
            )
            return tuple(findings)
        layout = sequence_layout(study, grouping=self.sequence_grouping)
        shortest = min(layout.lengths)
        if self.num_lags and shortest <= self.num_lags:
            findings.append(
                ModelFinding(
                    code="sequence_shorter_than_lag_order",
                    severity=WARNING,
                    message=(
                        f"the shortest {self.sequence_grouping.value} has {shortest} trials and "
                        f"num_lags={self.num_lags}, so every one of its emissions is conditioned "
                        "on dynamax's zero history rather than on observed trials"
                    ),
                )
            )
        n_parameters = len(self.parameter_names)
        if len(study) < 10 * n_parameters:
            findings.append(
                ModelFinding(
                    code="few_trials_per_parameter",
                    severity=WARNING,
                    message=(
                        f"{len(study)} trials for {n_parameters} parameters; a switching "
                        "autoregression needs enough dwell time in each state for its own "
                        "dynamics to be identified, and EM will happily report a local optimum"
                    ),
                )
            )
        if self.initialisation == "kmeans" and len(np.unique(values)) < self.n_states:
            findings.append(
                ModelFinding(
                    code="fewer_distinct_values_than_states",
                    severity=ERROR,
                    message=(
                        f"outcome {self.outcome!r} takes {len(np.unique(values))} distinct values "
                        f"and n_states={self.n_states}; k-means cannot seed a state per cluster"
                    ),
                )
            )
        if layout.n_sequences == 1 and self.sequence_grouping is SequenceGrouping.SESSION:
            findings.append(
                ModelFinding(
                    code="single_trial_sequence",
                    severity=WARNING,
                    message=(
                        "this study is one contiguous sequence, so the initial distribution is "
                        "estimated from a single observation and is prior-dominated"
                    ),
                )
            )
        return tuple(findings)

    # -- fitting -------------------------------------------------------------------------

    def fit(self, study: Study) -> DynamaxFitResult:
        """Run dynamax's EM over the study's sequences and return a Behavio fit.

        Restarts are deterministic: ``random_seed`` seeds one jax key per restart, each
        restart runs dynamax's own initialiser and then ``em_iterations`` of the ragged EM
        loop described in the module docstring, and the restart whose final log joint is
        highest wins. Ties are broken by restart index, so two runs of the same
        configuration on the same study give bit-identical estimates.

        The states of the winning restart are then permuted into canonical order, and the
        covariance and the stationarity check are computed *there* -- at the point the fit
        actually reports -- rather than being computed and then relabelled.
        """

        backend = _Backend(self)
        self.validate(study)
        layout = sequence_layout(study, grouping=self.sequence_grouping)
        blocks = backend.blocks(study, layout)

        objectives: list[float] = []
        fitted = backend.expectation_maximization(blocks, 0)
        selected = 0
        objectives.append(float(fitted[1][-1]))
        for restart in range(1, self.n_restarts):
            candidate = backend.expectation_maximization(blocks, restart)
            objectives.append(float(candidate[1][-1]))
            if objectives[-1] > objectives[selected]:
                fitted, selected = candidate, restart
        params, trace = fitted

        components, permutation = backend.canonical(params)
        params = backend.parameters(components)
        vector = _pack(components)
        curvature, condition = backend.curvature(blocks, components, len(vector))

        posteriors = backend.state_posteriors(blocks, params, layout)
        occupancy = np.mean(posteriors.smoothed, axis=0)
        observed = np.asarray(study[self.outcome], dtype=np.float64)
        means = backend.conditional_means(observed, layout, components)
        grid = backend.grid(observed, components)
        exact = _log_mixture_density(observed, means, components.variances, posteriors.predictive)
        density = backend.tabulate(means, components, posteriors.predictive, grid)
        mass = np.trapezoid(density, grid, axis=-1)
        interpolated = np.log(
            np.clip(_interpolate(grid, density, observed), np.finfo(np.float64).tiny, None)
        )
        biases = np.sort(components.biases)
        label_gap = float(np.min(np.diff(biases)))
        probabilities = np.concatenate((components.initial, components.transitions.ravel()))
        increment = float(trace[-1] - trace[-2]) if trace.size > 1 else float("nan")
        return DynamaxFitResult(
            model_name=self.model_name,
            model_signature=self.signature,
            parameter_names=self.parameter_names,
            estimates=protected_array(vector, dtype=np.float64),
            standard_errors=curvature.standard_errors,
            covariance=curvature.covariance,
            n_observations=len(study),
            diagnostics=FitDiagnostics(
                converged=curvature.converged,
                optimizer=(
                    f"dynamax:em ({self.n_restarts} restarts x {self.em_iterations} iterations, "
                    f"{self.initialisation} initialisation)"
                ),
                status=curvature.status,
                message=(
                    f"{curvature.message} [dynamax {dynamax_version()} on jax {jax_version()}, "
                    f"{layout.n_sequences} {self.sequence_grouping.value} sequences, restart "
                    f"{selected} of {self.n_restarts}, last EM step {increment:+.3e} nats]"
                ),
                n_iterations=self.em_iterations,
                objective=-float(trace[-1]),
                gradient_norm=curvature.gradient_norm,
                hessian_condition=condition,
                boundary_estimate=bool(
                    np.any(probabilities <= _PROBABILITY_WARNING)
                    or np.any(probabilities >= 1.0 - _PROBABILITY_WARNING)
                    or np.any(components.variances <= 1e-8)
                ),
            ),
            derived=(
                DerivedQuantity(
                    name="state_dwell_time",
                    value=float(
                        np.mean(1.0 / np.clip(1.0 - np.diag(components.transitions), 1e-12, None))
                    ),
                    description=(
                        "mean expected dwell time in trials, averaged over states: "
                        "1 / (1 - A[k, k]). A geometric dwell time is what a stationary "
                        "transition matrix implies, and is the assumption a semi-Markov model "
                        "would relax"
                    ),
                ),
            ),
            dynamax_version=dynamax_version(),
            jax_version=jax_version(),
            n_sequences=layout.n_sequences,
            restart_objectives=protected_array(objectives, dtype=np.float64),
            selected_restart=selected,
            final_em_increment=increment,
            canonical_permutation=permutation,
            state_occupancy=protected_array(occupancy, dtype=np.float64),
            label_order_gap=label_gap,
            label_ambiguous=bool(label_gap <= self.label_tolerance),
            low_occupancy=bool(np.any(occupancy < self.state_occupancy_warning)),
            covariance_is_estimated=curvature.estimated,
            outcome_grid=grid,
            grid_truncation=float(np.max(1.0 - mass)),
            grid_log_density_gap=float(np.sum(interpolated) - float(np.sum(exact))),
        )

    # -- prediction and scoring ----------------------------------------------------------

    def predict(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        """Return the model's whole prediction, which is a density over ``outcome``.

        This *is* :meth:`predict_density`. A switching autoregression predicts a
        distribution over a continuous measurement, and there is no discrete margin to
        report instead.
        """

        return self.predict_density(study, fit, mode=mode)

    def predict_density(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> DensityPrediction:
        """Tabulate the per-row emission mixture on the grid the fit fixed.

        Under :attr:`~behavio.contracts.PredictionMode.FILTERED` the mixing weights are
        ``p(z_t | y_{1:t-1})``, so the tabulated curve is the one-step-ahead predictive
        density of trial *t*. Under :attr:`~behavio.contracts.PredictionMode.SMOOTHED` they
        are ``p(z_t | y_{1:T})``, so it is a description of the recorded sequence and is not
        a prediction of anything.

        The grid comes from ``fit`` and never from ``study``. That is not tidiness: a grid
        derived from the rows being predicted would make an early row's reported density a
        function of later rows, which is exactly the leak
        :func:`behavio.adapters.check_behaviour_estimator` relabels the future to detect,
        and it would make a filtered prediction fail the check for a reason that has nothing
        to do with filtering.
        """

        self._require_mode(mode)
        components = self._components(fit)
        grid = _fit_grid(fit)
        backend = _Backend(self)
        layout = sequence_layout(study, grouping=self.sequence_grouping)
        weights = backend.mixing_weights(study, layout, components, mode)
        means = backend.conditional_means(
            np.asarray(study[self.outcome], dtype=np.float64), layout, components
        )
        density = backend.tabulate(means, components, weights, grid)
        mass = np.trapezoid(density, grid, axis=-1)
        if np.any(mass > _MASS_CEILING) or np.any(mass <= 0.0):
            raise ModelDataError(
                f"the fitted density grid, {grid[0]:.4g} to {grid[-1]:.4g} over {grid.size} "
                f"points, cannot tabulate these rows: integrated mass runs from "
                f"{float(np.min(mass)):.4g} to {float(np.max(mass)):.4g}. A mass above one "
                "means a fitted state's variance is too small for the grid to resolve "
                f"(smallest variance {float(np.min(components.variances)):.3g}); a mass of "
                "zero means the row lies entirely outside the range the training fold "
                "covered. pointwise_log_prob is computed in closed form and is unaffected"
            )
        return DensityPrediction(
            grid=grid,
            density=density,
            outcome=self.outcome,
            mode=PredictionMode(mode),
        )

    def pointwise_log_prob(
        self,
        study: Study,
        fit: FitResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Return one log density per trial, in closed form and never off the grid.

        Under the default filtered mode this is ``log p(y_t | y_{1:t-1})``, and its sum over
        a sequence is *exactly* the marginal log likelihood dynamax's own forward filter
        reports for that sequence -- the chain rule, not an approximation of it. That makes
        the pointwise score and the objective EM maximised the same quantity decomposed two
        ways, which is a checkable claim and is checked.

        Under ``SMOOTHED`` it is the same mixture reweighted by ``p(z_t | y_{1:T})``, and it
        is **not a held-out score**: it conditions on ``y_t`` itself and on every trial after
        it, so it is larger than the filtered score by construction and is not comparable
        with any other model's likelihood. ``evaluate_splits`` and ``compare_models`` both
        default to filtered and neither will reach this branch unless a caller asks for it
        explicitly.
        """

        self._require_mode(mode)
        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        components = self._components(fit)
        backend = _Backend(self)
        layout = sequence_layout(study, grouping=self.sequence_grouping)
        weights = backend.mixing_weights(study, layout, components, mode)
        observed = np.asarray(study[self.outcome], dtype=np.float64)
        means = backend.conditional_means(observed, layout, components)
        return _log_mixture_density(observed, means, components.variances, weights)

    def state_probabilities(self, study: Study, fit: FitResult) -> SwitchingStateProbabilities:
        """Return the predictive, filtered and smoothed state posteriors, in source order.

        The three are computed once and returned together precisely because they are
        routinely confused; see :class:`SwitchingStateProbabilities`. All three are written
        back onto the study's own rows through :meth:`~behavio.trials.SequenceLayout.join`,
        so a caller can put them beside any other column of the study without re-deriving an
        order.
        """

        components = self._components(fit)
        backend = _Backend(self)
        layout = sequence_layout(study, grouping=self.sequence_grouping)
        blocks = backend.blocks(study, layout)
        return backend.state_posteriors(blocks, backend.parameters(components), layout)

    def most_likely_states(self, study: Study, fit: FitResult) -> NDArray[np.int64]:
        """Return dynamax's Viterbi path per row, in source order and canonical labels.

        A *smoothed description*, unambiguously: the most likely state sequence is a
        function of the whole recording, so trial 3's label depends on trial 300. It is
        reported because it is what a reader of a switching model wants to see, and it is
        named a description rather than a prediction because nothing about it is available
        before a session ends.
        """

        components = self._components(fit)
        backend = _Backend(self)
        layout = sequence_layout(study, grouping=self.sequence_grouping)
        blocks = backend.blocks(study, layout)
        return backend.viterbi(blocks, backend.parameters(components), layout)

    # -- simulation ----------------------------------------------------------------------

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Sample one emission per design row from dynamax's own generative model.

        Each trial sequence of the design is sampled independently by
        :meth:`dynamax.hidden_markov_model.HMM.sample`, from its own key derived from
        ``seed``, and the blocks are written back onto the design's rows with
        :meth:`~behavio.trials.SequenceLayout.join`. Nothing about the state chain, the
        autoregression or the Gaussian variate is implemented here, so parameter recovery
        tests the inference rather than two implementations of one likelihood.

        A sequence's autoregressive history starts at dynamax's zeros, which is the same
        convention :meth:`fit` uses when it builds the inputs, so a study simulated by this
        method and then refitted is not fighting a boundary convention it did not use.
        """

        if not isinstance(design, Study):
            raise TypeError("design must be a Study")
        components = self._validated_components(parameters)
        backend = _Backend(self)
        layout = sequence_layout(design, grouping=self.sequence_grouping)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        entropy = int(generator.integers(0, 2**31 - 1))
        blocks = backend.sample(layout, components, entropy)
        columns: dict[str, Any] = {name: design[name] for name in design.columns}
        columns[self.outcome] = np.asarray(layout.join(blocks), dtype=np.float64)
        return Study(columns)

    # -- parameters ----------------------------------------------------------------------

    def parameters_from_components(
        self,
        *,
        initial: Sequence[float],
        transitions: Sequence[Sequence[float]],
        biases: Sequence[float],
        variances: Sequence[float],
        weights: Sequence[Sequence[float]] | None = None,
    ) -> Mapping[str, float]:
        """Validate and pack a natural-scale parameter set, keyed by :attr:`parameter_names`.

        ``weights`` is one row of ``num_lags`` autoregressive coefficients per state, and is
        forbidden when ``num_lags`` is zero. Writing a switching autoregression's truth out
        by hand is error-prone in exactly the way a recovery study cannot afford, so this
        checks the simplexes and the positivity before anything is simulated.
        """

        if self.num_lags == 0:
            if weights is not None:
                raise ValueError("weights applies only when num_lags is at least one")
            weight_array = np.zeros((self.n_states, 0), dtype=np.float64)
        else:
            if weights is None:
                raise ValueError(f"num_lags={self.num_lags} needs one weight row per state")
            weight_array = np.asarray(weights, dtype=np.float64)
        components = _Components(
            initial=np.asarray(initial, dtype=np.float64),
            transitions=np.asarray(transitions, dtype=np.float64),
            biases=np.asarray(biases, dtype=np.float64),
            weights=weight_array,
            variances=np.asarray(variances, dtype=np.float64),
        )
        self._require_shapes(components)
        vector = _pack(components)
        return MappingProxyType(
            dict(zip(self.parameter_names, (float(value) for value in vector), strict=True))
        )

    # -- internals -----------------------------------------------------------------------

    def _components(self, fit: FitResult) -> _Components:
        if not isinstance(fit, FitResult):
            raise TypeError("fit must be a FitResult")
        if fit.model_signature != self.signature or fit.parameter_names != self.parameter_names:
            raise ValueError("fit result belongs to a different model specification")
        return self._validated_components(fit.parameters)

    def _validated_components(self, parameters: Mapping[str, float]) -> _Components:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        expected, observed = set(self.parameter_names), set(parameters)
        if observed != expected:
            raise ValueError(
                "parameters must match the model exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            vector = np.asarray(
                [float(parameters[name]) for name in self.parameter_names], dtype=np.float64
            )
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        components = self._unpack(vector)
        self._require_shapes(components)
        return components

    def _unpack(self, vector: NDArray[np.float64]) -> _Components:
        states, lags = self.n_states, self.num_lags
        cursor = states
        initial = vector[:cursor]
        transitions = vector[cursor : cursor + states * states].reshape(states, states)
        cursor += states * states
        biases = vector[cursor : cursor + states]
        cursor += states
        weights = vector[cursor : cursor + states * lags].reshape(states, lags)
        cursor += states * lags
        variances = vector[cursor : cursor + states]
        return _Components(
            initial=initial,
            transitions=transitions,
            biases=biases,
            weights=weights,
            variances=variances,
        )

    def _require_shapes(self, components: _Components) -> None:
        states, lags = self.n_states, self.num_lags
        if components.initial.shape != (states,):
            raise ValueError(f"the initial distribution needs one probability per state ({states})")
        if components.transitions.shape != (states, states):
            raise ValueError("the transition matrix needs one row and column per state")
        if components.biases.shape != (states,) or components.variances.shape != (states,):
            raise ValueError("biases and variances need one value per state")
        if components.weights.shape != (states, lags):
            raise ValueError(f"weights must be ({states}, {lags}) for num_lags={lags}")
        values = np.concatenate(
            (
                components.initial,
                components.transitions.ravel(),
                components.biases,
                components.weights.ravel(),
                components.variances,
            )
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must contain finite numeric values")
        if np.any(components.initial < 0) or not np.isclose(float(components.initial.sum()), 1.0):
            raise ValueError("the initial distribution must be non-negative and sum to one")
        rows = components.transitions
        if np.any(rows < 0) or not np.allclose(rows.sum(axis=1), 1.0):
            raise ValueError("every transition row must be non-negative and sum to one")
        if np.any(components.variances <= 0):
            raise ValueError("every emission variance must be strictly positive")

    def _require_mode(self, mode: PredictionMode) -> None:
        if PredictionMode(mode) not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                f"{type(self).__name__} supports "
                f"{[value.value for value in self.supported_prediction_modes]}"
            )


# ------------------------------------------------------------------------------------------
# The dynamax boundary. Everything that touches jax lives below this line, is reached only
# through require_dynamax(), and is constructed per call so that the module imports with the
# extra uninstalled.
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SequenceBlock:
    """One group of equal-length sequences, stacked for a single vmapped E-step."""

    positions: tuple[int, ...]
    emissions: Any
    inputs: Any


class _Backend:
    """The dynamax model object and every operation the wrapper drives it with."""

    __slots__ = ("_flatten", "_hmm", "_jax", "_jnp", "_jr", "_model", "_params_module", "_props")

    def __init__(self, model: DynamaxSwitchingAutoregression) -> None:
        require_dynamax()
        self._model = model
        self._jax = importlib.import_module("jax")
        self._jnp = importlib.import_module("jax.numpy")
        self._jr = importlib.import_module("jax.random")
        self._flatten = importlib.import_module("jax.flatten_util")
        self._params_module = importlib.import_module("dynamax.parameters")
        families = importlib.import_module("dynamax.hidden_markov_model")
        with _quiet_dynamax():
            if model.num_lags:
                self._hmm = families.LinearAutoregressiveHMM(
                    model.n_states,
                    1,
                    num_lags=model.num_lags,
                    initial_probs_concentration=model.initial_concentration,
                    transition_matrix_concentration=model.transition_concentration,
                    transition_matrix_stickiness=model.transition_stickiness,
                )
            else:
                self._hmm = families.GaussianHMM(
                    model.n_states,
                    1,
                    initial_probs_concentration=model.initial_concentration,
                    transition_matrix_concentration=model.transition_concentration,
                    transition_matrix_stickiness=model.transition_stickiness,
                )
            self._props = self._hmm.initialize(self._jr.PRNGKey(0), method="prior")[1]

    # -- shape -------------------------------------------------------------------------

    def blocks(self, study: Study, layout: SequenceLayout) -> tuple[_SequenceBlock, ...]:
        """Partition the layout's sequences by length and stack each group for vmap.

        This is where a padded ``(n_sequences, max_length, 1)`` tensor would go, and does
        not: :meth:`dynamax.ssm.SSM.fit_em` takes no mask, so padded rows enter the
        forward-backward pass as real observations. Partitioning by length instead keeps
        dynamax's vmapped E-step wherever the lengths happen to agree -- which for a
        fixed-length task is everywhere -- and costs one extra traced shape otherwise.
        """

        if self._model.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self._model.outcome!r}")
        observed = np.asarray(study[self._model.outcome], dtype=np.float64)
        if not np.all(np.isfinite(observed)):
            raise ModelDataError(
                f"outcome column {self._model.outcome!r} contains non-finite values"
            )
        groups: dict[int, list[int]] = {}
        for position, sequence in enumerate(layout.sequences):
            groups.setdefault(len(sequence), []).append(position)
        pieces = layout.split(observed)
        blocks: list[_SequenceBlock] = []
        for length in sorted(groups):
            positions = tuple(groups[length])
            emissions = self._jnp.asarray(
                np.stack([pieces[position].reshape(length, 1) for position in positions])
            )
            blocks.append(
                _SequenceBlock(
                    positions=positions,
                    emissions=emissions,
                    inputs=self._inputs(emissions),
                )
            )
        return tuple(blocks)

    def _inputs(self, emissions: Any) -> Any:
        """Lagged emissions, built **per sequence** so no session reads the previous one."""

        if not self._model.num_lags:
            return None
        return self._jax.vmap(self._hmm.compute_inputs)(emissions)

    # -- fitting -------------------------------------------------------------------------

    def expectation_maximization(
        self, blocks: Sequence[_SequenceBlock], restart: int
    ) -> tuple[Any, NDArray[np.float64]]:
        """Run dynamax's own E and M steps over ragged sequences, and return the log-joint trace.

        Identical to :meth:`dynamax.ssm.SSM.fit_em` except that the batch is a partition by
        length rather than one padded array: the same ``e_step``, the same ``m_step``, the
        same ``log_prior``, the same jitted iteration. The sufficient statistics every
        component here collects are sums over time -- occupancy, transition counts, weighted
        emission moments -- so concatenating them across blocks of different lengths is
        exactly what a single vmapped pass over equal-length sequences would have produced.
        """

        jax, jnp = self._jax, self._jnp
        params = self._start(blocks, restart)
        props = self._props

        def em_step(params: Any, m_step_state: Any) -> tuple[Any, Any, Any]:
            collected: list[Any] = []
            total = jnp.asarray(0.0)
            for block in blocks:
                if block.inputs is None:
                    stats, lls = jax.vmap(lambda e: self._hmm.e_step(params, e))(block.emissions)
                else:
                    stats, lls = jax.vmap(lambda e, u: self._hmm.e_step(params, e, u))(
                        block.emissions, block.inputs
                    )
                collected.append(stats)
                total = total + jnp.sum(lls)
            batch = (
                collected[0]
                if len(collected) == 1
                else jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *collected)
            )
            objective = self._hmm.log_prior(params) + total
            updated, m_step_state = self._hmm.m_step(params, props, batch, m_step_state)
            return updated, m_step_state, objective

        compiled = jax.jit(em_step)
        m_step_state = self._hmm.initialize_m_step_state(params, props)
        trace: list[float] = []
        with _quiet_dynamax():
            for _ in range(self._model.em_iterations):
                params, m_step_state, objective = compiled(params, m_step_state)
                trace.append(float(objective))
            trace.append(float(self._log_joint(params, blocks)))
        return params, np.asarray(trace, dtype=np.float64)

    def _start(self, blocks: Sequence[_SequenceBlock], restart: int) -> Any:
        """Ask dynamax for a starting point, from a key derived from the declared seed."""

        key = self._jr.fold_in(self._jr.PRNGKey(self._model.random_seed), restart)
        keywords: dict[str, Any] = {"method": self._model.initialisation}
        if self._model.initialisation == "kmeans":
            keywords["emissions"] = self._jnp.concatenate(
                [block.emissions.reshape(-1, 1) for block in blocks]
            )
        with _quiet_dynamax():
            return self._hmm.initialize(key, **keywords)[0]

    def _log_joint(self, params: Any, blocks: Sequence[_SequenceBlock]) -> Any:
        total = self._hmm.log_prior(params)
        for block in blocks:
            if block.inputs is None:
                marginals = self._jax.vmap(lambda e: self._hmm.marginal_log_prob(params, e))(
                    block.emissions
                )
            else:
                marginals = self._jax.vmap(lambda e, u: self._hmm.marginal_log_prob(params, e, u))(
                    block.emissions, block.inputs
                )
            total = total + self._jnp.sum(marginals)
        return total

    # -- parameters ----------------------------------------------------------------------

    def parameters(self, components: _Components) -> Any:
        """Build a dynamax parameter pytree from Behavio's natural coordinates."""

        jnp = self._jnp
        keywords: dict[str, Any] = {
            "initial_probs": jnp.asarray(components.initial),
            "transition_matrix": jnp.asarray(components.transitions),
            "emission_covariances": jnp.asarray(components.variances).reshape(-1, 1, 1),
        }
        if self._model.num_lags:
            keywords["emission_biases"] = jnp.asarray(components.biases).reshape(-1, 1)
            keywords["emission_weights"] = jnp.asarray(components.weights).reshape(
                self._model.n_states, 1, self._model.num_lags
            )
        else:
            keywords["emission_means"] = jnp.asarray(components.biases).reshape(-1, 1)
        with _quiet_dynamax():
            return self._hmm.initialize(self._jr.PRNGKey(0), method="prior", **keywords)[0]

    def components(self, params: Any) -> _Components:
        """Read a dynamax parameter pytree back into Behavio's natural coordinates."""

        emissions = params.emissions
        offsets = getattr(emissions, "biases", None)
        if offsets is None:
            offsets = emissions.means
        weights = getattr(emissions, "weights", None)
        return _Components(
            initial=np.asarray(params.initial.probs, dtype=np.float64),
            transitions=np.asarray(params.transitions.transition_matrix, dtype=np.float64),
            biases=np.asarray(offsets, dtype=np.float64).reshape(-1),
            weights=(
                np.zeros((self._model.n_states, 0), dtype=np.float64)
                if weights is None
                else np.asarray(weights, dtype=np.float64).reshape(
                    self._model.n_states, self._model.num_lags
                )
            ),
            variances=np.asarray(emissions.covs, dtype=np.float64).reshape(-1),
        )

    def canonical(self, params: Any) -> tuple[_Components, tuple[int, ...]]:
        """Sort the states into increasing-bias order and permute every parameter with them."""

        raw = self.components(params)
        order = sorted(
            range(self._model.n_states),
            key=lambda state: (
                float(raw.biases[state]),
                *(float(value) for value in raw.weights[state]),
                float(raw.variances[state]),
            ),
        )
        index = np.asarray(order, dtype=np.intp)
        canonical = _Components(
            initial=raw.initial[index],
            transitions=raw.transitions[np.ix_(index, index)],
            biases=raw.biases[index],
            weights=raw.weights[index],
            variances=raw.variances[index],
        )
        return canonical, tuple(int(state) for state in order)

    # -- curvature -----------------------------------------------------------------------

    def curvature(
        self,
        blocks: Sequence[_SequenceBlock],
        components: _Components,
        size: int,
    ) -> tuple[ForeignCurvature, float | None]:
        """Differentiate the objective EM maximised, twice, and report what that establishes.

        EM hands back a parameter pytree and nothing else: no uncertainty, and no convergence
        flag, because a fixed number of monotone iterations has no stopping rule to have
        fired. Both are recovered here, from the same two derivatives of the *same* objective
        dynamax's own M-step ascends -- ``log_prior(theta) + sum over sequences of the
        marginal log likelihood`` -- because a covariance must be the curvature of the
        function that produced the estimate or the interval it implies is centred on a point
        that does not maximise it.

        Everything is differentiated in **dynamax's own unconstrained coordinates**, reached
        through its ``to_unconstrained``/``from_unconstrained`` bijectors. That matters twice
        over. A simplex has no interior derivative in its natural coordinates, so a Hessian
        taken there is singular by construction and says nothing; and the unconstrained chart
        is minimal, so the observed information is the full-rank matrix an inverse is
        meaningful for. The covariance is then carried onto the reported natural coordinates
        by the delta method, with the Jacobian of the constraining map taken by the same
        automatic differentiation rather than by finite differences.

        *Convergence* is exact stationarity: the gradient norm at the reported estimate,
        against ``gradient_tolerance``. That is a stronger claim than PyDDM's differenced
        coordinate-wise probe and much stronger than "the optimizer said so", and unlike both
        it is free -- the Hessian pass computes the gradient on the way.

        When the observed information is not positive definite the covariance is all-``NaN``
        with a message, exactly as :class:`~behavio.foreign.pyddm.PyDDMDriftDiffusion` does.
        For an HMM that is a real and frequent state rather than a corner case: a state with
        no occupancy, a transition probability driven to zero, or an under-iterated EM run
        all produce one, and reporting a number there would be reporting a curvature that is
        not one.

        The second return value is the condition number of the **unconstrained observed
        information**, which is the matrix actually inverted. PyDDM's wrapper reports the
        condition number of its covariance instead, and the difference is not a style
        choice: the natural covariance here is singular along the sum-to-one directions of
        two simplexes, so its condition number would be an artefact of the constraint and
        would raise an ill-conditioning warning on every healthy fit.
        """

        jax, jnp = self._jax, self._jnp
        params = self.parameters(components)
        unconstrained = self._params_module.to_unconstrained(params, self._props)
        flat, unravel = self._flatten.ravel_pytree(unconstrained)

        def objective(vector: Any) -> Any:
            return self._log_joint(
                self._params_module.from_unconstrained(unravel(vector), self._props), blocks
            )

        def natural(vector: Any) -> Any:
            restored = self._params_module.from_unconstrained(unravel(vector), self._props)
            emissions = restored.emissions
            offsets = getattr(emissions, "biases", None)
            if offsets is None:
                offsets = emissions.means
            pieces = [
                restored.initial.probs,
                restored.transitions.transition_matrix.ravel(),
                jnp.ravel(offsets),
            ]
            if self._model.num_lags:
                pieces.append(jnp.ravel(emissions.weights))
            pieces.append(jnp.diagonal(emissions.covs, axis1=1, axis2=2).ravel())
            return jnp.concatenate(pieces)

        with _quiet_dynamax():
            gradient = np.asarray(jax.grad(objective)(flat), dtype=np.float64)
            hessian = np.asarray(jax.hessian(objective)(flat), dtype=np.float64)
            jacobian = np.asarray(jax.jacobian(natural)(flat), dtype=np.float64)

        norm = float(np.linalg.norm(gradient)) if np.all(np.isfinite(gradient)) else float("inf")
        converged = bool(norm <= self._model.gradient_tolerance)
        verdict = (
            f"the exact gradient of the log joint at the reported estimate has norm "
            f"{norm:.3e} (tolerance {self._model.gradient_tolerance:g})"
        )
        information = -hessian
        if not np.all(np.isfinite(information)):
            return (
                unknown_curvature(
                    size,
                    f"{verdict}; no covariance: the differentiated log joint is not finite",
                    converged=converged,
                    gradient_norm=norm if np.isfinite(norm) else None,
                ),
                None,
            )
        condition = condition_number(information)
        smallest = float(np.min(np.linalg.eigvalsh(0.5 * (information + information.T))))
        if smallest <= 0:
            return (
                unknown_curvature(
                    size,
                    f"{verdict}; no covariance: the observed information is not positive "
                    f"definite (smallest eigenvalue {smallest:.3e}), so this is not an "
                    "interior maximum -- an unoccupied state or a transition probability at "
                    "zero is the usual reason",
                    converged=converged,
                    gradient_norm=norm,
                ),
                condition,
            )
        covariance = jacobian @ np.linalg.inv(information) @ jacobian.T
        covariance = 0.5 * (covariance + covariance.T)
        diagonal = np.diag(covariance)
        # A coordinate the constraints pin -- the last free direction of a simplex -- has
        # variance exactly zero, so rounding can put its delta-method diagonal a few ulps
        # below it. That is not a failure; a diagonal that is negative by more than roundoff
        # is, and the two are distinguished rather than both clipped or both refused.
        tolerance = 1e-10 * max(float(np.max(np.abs(diagonal))), 1.0)
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal < -tolerance):
            return (
                unknown_curvature(
                    size,
                    f"{verdict}; no covariance: the delta-method variance of a reported "
                    "coordinate is negative or non-finite",
                    converged=converged,
                    gradient_norm=norm,
                ),
                condition,
            )
        return (
            ForeignCurvature(
                covariance=protected_array(covariance, dtype=np.float64),
                standard_errors=protected_array(
                    np.sqrt(np.clip(diagonal, 0.0, None)), dtype=np.float64
                ),
                gradient_norm=norm,
                converged=converged,
                estimated=True,
                message=verdict,
            ),
            condition,
        )

    # -- inference -----------------------------------------------------------------------

    def state_posteriors(
        self,
        blocks: Sequence[_SequenceBlock],
        params: Any,
        layout: SequenceLayout,
    ) -> SwitchingStateProbabilities:
        """Run dynamax's smoother once per sequence and join all three posteriors back."""

        jax = self._jax
        predictive: list[Any] = [None] * layout.n_sequences
        filtered: list[Any] = [None] * layout.n_sequences
        smoothed: list[Any] = [None] * layout.n_sequences
        with _quiet_dynamax():
            for block in blocks:
                if block.inputs is None:
                    posterior = jax.vmap(lambda e: self._hmm.smoother(params, e))(block.emissions)
                else:
                    posterior = jax.vmap(lambda e, u: self._hmm.smoother(params, e, u))(
                        block.emissions, block.inputs
                    )
                for row, position in enumerate(block.positions):
                    predictive[position] = np.asarray(posterior.predicted_probs[row])
                    filtered[position] = np.asarray(posterior.filtered_probs[row])
                    smoothed[position] = np.asarray(posterior.smoothed_probs[row])
        return SwitchingStateProbabilities(
            predictive=np.asarray(layout.join(predictive), dtype=np.float64),
            filtered=np.asarray(layout.join(filtered), dtype=np.float64),
            smoothed=np.asarray(layout.join(smoothed), dtype=np.float64),
        )

    def mixing_weights(
        self,
        study: Study,
        layout: SequenceLayout,
        components: _Components,
        mode: PredictionMode,
    ) -> NDArray[np.float64]:
        """The per-row state weights the requested mode mixes the emission densities under."""

        posteriors = self.state_posteriors(
            self.blocks(study, layout), self.parameters(components), layout
        )
        if PredictionMode(mode) is PredictionMode.SMOOTHED:
            return posteriors.smoothed
        return posteriors.predictive

    def viterbi(
        self,
        blocks: Sequence[_SequenceBlock],
        params: Any,
        layout: SequenceLayout,
    ) -> NDArray[np.int64]:
        """Run dynamax's Viterbi decoder once per sequence and join the paths back."""

        jax = self._jax
        paths: list[Any] = [None] * layout.n_sequences
        with _quiet_dynamax():
            for block in blocks:
                if block.inputs is None:
                    decoded = jax.vmap(lambda e: self._hmm.most_likely_states(params, e))(
                        block.emissions
                    )
                else:
                    decoded = jax.vmap(lambda e, u: self._hmm.most_likely_states(params, e, u))(
                        block.emissions, block.inputs
                    )
                for row, position in enumerate(block.positions):
                    paths[position] = np.asarray(decoded[row], dtype=np.int64)
        return protected_array(layout.join(paths), dtype=np.int64)

    def sample(
        self,
        layout: SequenceLayout,
        components: _Components,
        entropy: int,
    ) -> list[NDArray[np.float64]]:
        """Draw one emission block per sequence from dynamax's own generative model."""

        params = self.parameters(components)
        keys = self._jr.split(self._jr.PRNGKey(entropy), layout.n_sequences)
        blocks: list[NDArray[np.float64]] = []
        with _quiet_dynamax():
            for position, sequence in enumerate(layout.sequences):
                _, emissions = self._hmm.sample(params, keys[position], len(sequence))
                blocks.append(np.asarray(emissions, dtype=np.float64).reshape(-1))
        return blocks

    # -- densities -----------------------------------------------------------------------

    def conditional_means(
        self,
        observed: NDArray[np.float64],
        layout: SequenceLayout,
        components: _Components,
    ) -> NDArray[np.float64]:
        """Return each row's per-state conditional mean, ``(n_rows, n_states)``.

        For a plain Gaussian HMM that is the state's mean, repeated. For an autoregression it
        is ``b_k + sum_l W_{k,l} y_{t-l}`` with the lags taken **within** the row's own
        sequence, exactly as :meth:`_Backend.blocks` builds the inputs the fit saw; the
        first ``num_lags`` rows of every sequence use dynamax's zero history rather than the
        previous session's last trials.
        """

        rows, states = observed.size, self._model.n_states
        if not self._model.num_lags:
            return np.broadcast_to(components.biases[None, :], (rows, states)).copy()
        lagged = np.zeros((rows, self._model.num_lags), dtype=np.float64)
        for sequence in layout.sequences:
            indices = np.asarray(sequence.indices, dtype=np.intp)
            block = observed[indices]
            for lag in range(1, self._model.num_lags + 1):
                if lag < block.size:
                    lagged[indices[lag:], lag - 1] = block[:-lag]
        return components.biases[None, :] + lagged @ components.weights.T

    def grid(
        self,
        observed: NDArray[np.float64],
        components: _Components,
    ) -> NDArray[np.float64]:
        """Fix the tabulation grid from the training rows and the fitted variances.

        Spanned by the training range extended by ``grid_padding`` of the widest fitted
        state, and resolved finely enough for the *narrowest*: a mixture is only tabulated
        correctly when its sharpest component is, and a grid coarse relative to one state's
        standard deviation integrates that state's mass to something other than its weight.
        The result is retained on the fit, which is what makes ``predict`` a function of the
        fit rather than of the study.
        """

        deviations = np.sqrt(components.variances)
        widest, narrowest = float(np.max(deviations)), float(np.min(deviations))
        low = float(np.min(observed)) - self._model.grid_padding * widest
        high = float(np.max(observed)) + self._model.grid_padding * widest
        span = high - low
        wanted = int(np.ceil(span / max(narrowest / self._model.grid_points_per_sd, 1e-12))) + 1
        points = int(np.clip(max(wanted, self._model.grid_points), 8, _MAX_GRID_POINTS))
        return protected_array(np.linspace(low, high, points), dtype=np.float64)

    def tabulate(
        self,
        means: NDArray[np.float64],
        components: _Components,
        weights: NDArray[np.float64],
        grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the per-row emission mixture on the grid, ``(n_rows, n_grid)``."""

        variances = components.variances[None, None, :]
        offsets = grid[None, :, None] - means[:, None, :]
        curves = np.exp(-0.5 * offsets**2 / variances) / np.sqrt(2.0 * np.pi * variances)
        return protected_array(np.einsum("tgk,tk->tg", curves, weights), dtype=np.float64)


def _log_mixture_density(
    observed: NDArray[np.float64],
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The log of the per-row emission mixture at each observed value, in closed form.

    Never read off the tabulated grid, so no score in this package is a function of a
    tabulation's step size. The floor is the one
    :data:`~behavio.contracts.LOG_DENSITY_FLOOR` names, so a single row far into a fitted
    state's tail cannot make a whole fold's score ``-inf``.
    """

    spread = variances[None, :]
    exponent = -0.5 * (observed[:, None] - means) ** 2 / spread
    normaliser = -0.5 * np.log(2.0 * np.pi * spread)
    density = np.sum(weights * np.exp(normaliser + exponent), axis=1)
    smallest = np.finfo(np.float64).tiny
    return protected_array(
        np.maximum(np.log(np.clip(density, smallest, None)), LOG_DENSITY_FLOOR),
        dtype=np.float64,
    )


def _interpolate(
    grid: NDArray[np.float64],
    density: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Read the tabulated density at one observed value per row, linearly."""

    upper = np.clip(np.searchsorted(grid, observed, side="left"), 1, grid.size - 1)
    lower = upper - 1
    span = grid[upper] - grid[lower]
    weight = np.clip(np.where(span > 0, (observed - grid[lower]) / span, 0.0), 0.0, 1.0)
    rows = np.arange(observed.size)
    left, right = density[rows, lower], density[rows, upper]
    inside = (observed >= grid[0]) & (observed <= grid[-1])
    return np.where(inside, left + weight * (right - left), 0.0)


def _pack(components: _Components) -> NDArray[np.float64]:
    return np.concatenate(
        (
            components.initial,
            components.transitions.ravel(),
            components.biases,
            components.weights.ravel(),
            components.variances,
        )
    )


def _fit_grid(fit: FitResult) -> NDArray[np.float64]:
    grid = getattr(fit, "outcome_grid", None)
    if grid is None:
        raise ValueError(
            "this fit carries no tabulation grid, so it was not produced by "
            "DynamaxSwitchingAutoregression.fit; a density grid derived from the study being "
            "predicted would make an early row's density a function of later rows"
        )
    return np.asarray(grid, dtype=np.float64)


def _quiet_dynamax() -> Any:
    """Silence jax's and dynamax's advisories for the duration of one call.

    jax warns about donated buffers and about x64 promotion, ``jaxtyping`` warns about
    annotations it cannot resolve, and ``tfp`` emits deprecation notices from inside the
    distributions it builds. None of it is a finding about the fit; what *is* a finding --
    the gradient norm, the observed information, the restart trace, the occupancy, the label
    order gap -- is computed here and retained on the :class:`DynamaxFitResult`.
    """

    return quiet_foreign_package(
        "jax",
        "jax._src",
        "absl",
        "dynamax",
        categories=(RuntimeWarning, UserWarning, FutureWarning, DeprecationWarning),
    )


__all__ = [
    "INITIALISATIONS",
    "PARAMETER_CORRESPONDENCE",
    "DynamaxFitResult",
    "DynamaxSwitchingAutoregression",
    "SwitchingStateProbabilities",
]
