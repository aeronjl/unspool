"""A Behavio *sampled* estimator backed by Bambi's PyMC-compiled regression models.

The PyDDM wrapper in :mod:`behavio.foreign.pyddm` answered "can a foreign optimizer satisfy
:class:`~behavio.contracts.BehaviourEstimator`?". This one answers the other half of the
question, and it is the half that had never been asked of anything outside this package:
:class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator`, its point-summary
projection, the per-fold convergence gate, blocked PSIS-LOO and
:func:`~behavio.posterior.compare_posterior_models` were all built and tested against
first-party code. Bambi is posterior-native, returns an ArviZ container, and is therefore the
cheapest honest test of whether that contract is a contract or a description of Behavio's own
habits.

What is wrapped
---------------
Bambi's *model construction* and its *inference* both are. A formula and a family become a
PyMC graph; NUTS samples it; :meth:`bambi.Model.compute_log_likelihood` and
:meth:`bambi.Model.predict` evaluate that same graph on rows the sampler never saw. What is
added on top is everything the Behavio contract needs and Bambi does not produce:

- **A labelled, backend-neutral posterior.** :meth:`BambiRegression.sample` returns a
  :class:`~behavio.posterior.PosteriorResult`, not an ``InferenceData``, so the audit, the
  projection, LOO and the evidence bundle all read one shape.
- **A grouping variable, so blocked LOO can run.** See below; this is the single most
  concrete gap Bambi leaves.
- **The training rows, retained on the posterior.** Bambi's ``predict`` needs the model
  object that was fitted, and a fold hands ``predict`` a ``PosteriorResult`` and nothing
  else. The columns the formula reads are therefore retained in ``constant_data`` and the
  Bambi model is rebuilt from them, which is what makes out-of-fold prediction reproduce the
  in-fold model exactly rather than re-deriving a design from the test rows.
- **A held-out score that integrates the posterior.** ``pointwise_log_prob`` is
  ``log((1/S) sum_s p(y_i | theta_s))`` over the per-draw response parameters Bambi's own
  inverse link produces, never the density at the posterior mean.
- **A refusal to predict groups the fit never saw**, with the levels named, instead of
  Bambi's ``ValueError`` from inside ``predict_group_specific``.

Why Bambi and not another PyMC front end
----------------------------------------
Not for the formula: Behavio has its own now (:mod:`behavio.design.formula`). For three
things Behavio genuinely lacks and is not going to grow: **crossed and nested random
effects** (``(1|subject) + (1|stimulus_id)``, ``(1|subject/session)``), where
:func:`behavio.compose.hierarchical` pools over exactly one grouping; **splines and other
stateful basis transforms** (``bs(x, df=5)``, ``scale(x)``), fitted on the training frame and
*reused* on new rows by ``formulae``'s stored transform state, which is the information
boundary ``AGENTS.md`` demands and which this wrapper verifies behaviourally rather than
trusting; and a **multi-alternative choice family** with a proper categorical likelihood.

The formula collision, and why nothing is translated
----------------------------------------------------
Behavio's formula desugars to a :class:`~behavio.design.DesignSpec`. Bambi's desugars to a
``formulae`` design. :attr:`BambiRegression.formula` is **Bambi's own string, passed through
verbatim**, and Behavio's formula language does not apply to it. That is a deliberate refusal
to build a translator, for three reasons that are not symmetrical.

*Downward, a translation loses the reason to be here.* A ``DesignSpec`` is one fixed matrix
with no varying-effect representation -- its own module says so -- so ``(1|subject)``,
``(1|subject/session)`` and ``bs(x, df=5)`` have no image in it. Translating Bambi's formula
into Behavio's would drop precisely the terms this wrapper exists for.

*Upward, the two languages disagree about what a row is.* Behavio's ``lag()`` and
``kernel()`` are **sequence-aware**: they read trial order and reset at subject and session
boundaries. ``formulae`` has no notion of trial order at all; its transforms are functions of
a column, not of a sequence. ``lag(choice, 1)`` therefore has no Bambi formula that means the
same thing -- only a *materialised column*, which is a different object, computed by Behavio,
that a Bambi formula then refers to by name. That is not a translation between formulas and
pretending otherwise would be the lossy step.

*And a partial translator is worse than none.* It would accept the intersection silently and
fail -- or, worse, quietly mean something else -- outside it, with no way for a reader of a
signature to tell which had happened. So: to fit history-dependent features with this
wrapper, add them to the ``Study`` as columns **inside the training fold** and name them in
the Bambi formula. The wrapper will not do it for you, because doing it for you would mean
computing them across a fold boundary.

Priors: a narrow reconciliation, and an explicit delegation
-----------------------------------------------------------
:attr:`BambiRegression.priors` accepts Behavio's own
:class:`~behavio.inference.PriorSpec` -- ``normal``, ``half-normal``, ``beta``, ``uniform`` --
keyed by Bambi term name, and translates each to the identically parameterised
:class:`bambi.Prior`. Those four map exactly, with no reparameterisation, so nothing is
approximated.

Everything else is **delegated to Bambi and stated rather than wrapped**:

- A prior on a *group-specific* term is refused. Bambi spells one as a ``Normal`` whose
  ``sigma`` is itself a ``Prior``; a flat ``PriorSpec`` can only express a ``Normal`` with a
  *fixed* sigma, which is not a weaker hierarchical prior but a different model with no
  pooling at all. Refusing is the only honest option a scalar prior language has here.
- A term with no declared ``PriorSpec`` keeps Bambi's default, which under
  :attr:`BambiRegression.auto_scale` is **a function of the training data**. That is a real
  strain and it is reported, not smoothed over: see below.
- ``bambi.Prior`` objects are not accepted. They are mutable, they carry no JSON-safe form,
  and they would therefore be absent from the model's signature and from any evidence bundle
  built around it -- a fit whose prior cannot be written down is not replayable.

Where Bambi strains the contract
--------------------------------
Six places. Each is reported by the posterior, refused before sampling, or routed around in
a way the test suite checks.

*Bambi retains no grouping variable, so blocked LOO cannot run on its output.*
:func:`behavio.posterior.psis_loo` reaches a blocked estimand through
``block="trial_subject"``, which it resolves against ``constant_data`` or ``observed_data``.
Bambi's ``InferenceData`` has no ``constant_data`` group at all: it carries ``posterior``,
``sample_stats``, ``observed_data`` and ``log_likelihood`` and nothing that says which
subject a row belongs to. Left alone, every Bambi posterior would silently be scoreable only
at leave-one-*trial*-out -- the estimand ``behavio.posterior.loo`` opens by calling the wrong
one for a multi-subject design. The wrapper therefore writes ``trial_subject``,
``trial_session``, ``trial_in_session`` and ``trial_session_order`` into ``constant_data``
over the scored dimension, exactly as
:class:`~behavio.pymc_backend.PyMCHierarchicalGLMBackend` does, and
:attr:`BLOCKING_VARIABLES` names them. ``psis_loo(result, block="trial_subject")`` then works
with no ``block_values`` argument.

*Bambi's observation dimension is ``__obs__``.* Every other labelled array in Behavio calls
it ``trial``. The wrapper renames it on import, so a Pareto-k target reads
``choice[trial=17]`` and a blocked result's coordinate is a subject rather than a dunder. The
rename is a relabelling of one dimension and touches no value.

*Default priors are a function of the training fold.* With ``auto_scale=True`` -- Bambi's
default, and this wrapper's -- the prior scales are computed from the data handed to
``bambi.Model``. Inside :func:`~behavio.evaluate.evaluate_splits` that means each fold fits a
slightly different model, and the signature cannot say so because it is computed without a
study. Two things follow, and both are implemented rather than documented away: ``auto_scale``
is a declared field of the signature, so a run that turned it off is distinguishable from one
that did not; and the *realised* prior specification Bambi actually built is recorded on every
posterior as the ``prior_specification`` attribute, so the fold's model is recoverable from
its own evidence. ``describe()`` reports the same thing as a warning finding.

*A random effect can only be predicted for groups the fit saw.* A prospective
session-forward splitter holds subjects fixed and sessions varying, so ``(1|subject)``
predicts out of fold and ``(1|session)`` cannot: the test fold's sessions are new levels and
their offsets were never estimated. Bambi will draw them from the hyperprior if asked
(``sample_new_groups=True``), which is a legitimate and quite different predictive claim.
The wrapper's default is to refuse, naming the unseen levels and the grouping factor, because
silently sampling a new group's effect turns a held-out prediction into a prior predictive
one and nothing downstream would know. One consequence worth stating separately:
:func:`~behavio.evaluate.forward_session_splits` builds one fold *per subject*, so a
``(1|subject)`` model fitted inside one of its folds has a single group level and an
unidentified variance. :func:`~behavio.evaluate.cohort_forward_session_splits` is the
splitter that matches a subject-level random effect, and ``describe()`` reports the one-level
case as a warning before any sampler starts.

*Bambi 0.19 cannot compute an out-of-sample log likelihood for the categorical family.*
``bambi.Model.compute_log_likelihood(idata, data=new_rows)`` raises a PyTensor broadcast
error there, while the in-sample group computed during ``fit`` is fine. It is an upstream
defect and not something a wrapper should paper over by re-deriving the categorical pmf under
a different name -- so the wrapper does not use that method at all. It reads the observed
entry of the draw-averaged response simplex instead, which for a finite-support family *is*
the pointwise predictive density, is one graph evaluation rather than two, and makes
``predict`` and ``pointwise_log_prob`` literally the same computation. That the result is
Bambi's own likelihood is then a checkable claim: the test suite compares it, per row,
against the ``log_likelihood`` group Bambi computed during sampling.

*A sampled model's parameter vector is a function of its data.* ``parameter_names`` and
``posterior_parameter_labels`` -- the two members
:class:`~behavio.contracts.posterior.GenerativePosteriorBehaviourModel` needs for parameter
recovery -- are study-independent properties. For a mixed-effects model they cannot be: how
many coordinates ``1|subject`` has, which columns ``C(condition)`` produces, how many basis
columns ``bs(x, df=5)`` implies are all facts about the data. :class:`BambiRegression`
therefore does *not* claim the generative contract. :meth:`BambiRegression.bind` returns a
:class:`DesignBoundBambiRegression`, which does, because a design is exactly what it was
missing -- and ``AGENTS.md`` already says recovery is design-specific evidence, so binding a
recovery model to its design states something true rather than working around something
false.

Simulation is Bambi's own
-------------------------
:meth:`DesignBoundBambiRegression.simulate` does not reimplement a likelihood. It writes the
supplied parameter values into a one-draw posterior with the model's own dimensions and calls
``bambi.Model.predict(kind="response")``, so the observation is drawn by the family Bambi
declared, through PyMC's own random generator, seeded. A hand-rolled Bernoulli draw would
have been three lines and would have made recovery a test of two implementations of the same
family rather than of the inference.

Licence
-------
Bambi is MIT, as Behavio is. Its transitive dependencies were read from published metadata
and are permissive throughout: PyMC and PyTensor Apache-2.0/BSD-3-Clause, ``formulae`` MIT,
pandas BSD-3-Clause, matplotlib PSF-based, seaborn BSD-3-Clause, ``sparse`` BSD-3-Clause,
``graphviz`` MIT, ``arviz-plots``/``arviz-base``/``arviz-stats`` Apache-2.0. So
``pip install 'behavio[bambi]'`` changes no obligation. ``docs/foreign-models.md`` carries the
matrix.

Python floor
------------
Bambi 0.18 raised its floor to Python 3.12 and moved to PyMC 6; Behavio still supports 3.11.
The extra therefore splits exactly as Behavio's existing ``bayesian`` and ``probabilistic``
extras already do -- Bambi 0.17.x with PyMC 5 below 3.12, Bambi 0.19.x with PyMC 6 at or
above it -- rather than becoming a marker that installs nothing on the interpreter floor the
package claims to support.

Import safety
-------------
This module is named ``bambi`` inside ``behavio.foreign``. Python 3 resolves imports
absolutely, and nothing here writes ``import bambi`` in any case: the dependency is reached
only through :func:`behavio.foreign._optional.require_bambi`, so the two names cannot collide
and ``import behavio.foreign.bambi`` works with the extra uninstalled.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.contracts.estimator import (
    CategoricalPrediction,
    FitResult,
    ModelDataError,
    ModelPrediction,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
)
from behavio.contracts.posterior import PosteriorCentre, posterior_point_summary
from behavio.foreign._optional import bambi_version, require_bambi
from behavio.inference.parameters import PriorFamily, PriorSpec
from behavio.models._kernels.introspection import ERROR, WARNING, Describable, ModelFinding
from behavio.posterior.result import (
    SAMPLE_DIMS,
    PosteriorError,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    posterior_result_from_arviz,
)
from behavio.trials import REQUIRED_COLUMNS, Study

TRIAL_DIM: Final = "trial"
"""What this wrapper calls Bambi's observation dimension.

Bambi names it ``__obs__``. Every other labelled array in Behavio -- the PyMC backend's
posterior, a Pareto-k target, a blocked LOO coordinate -- names it ``trial``, and a
double-underscore name reads as private in the one place a user is most likely to look.
"""

BLOCKING_VARIABLES: Final = (
    "trial_subject",
    "trial_session",
    "trial_in_session",
    "trial_session_order",
)
"""The grouping columns retained in ``constant_data`` so blocked PSIS-LOO can reach them.

Bambi retains none of them. Without these, ``psis_loo(result, block="trial_subject")`` has
nothing to resolve and the only available estimand is leave-one-trial-out, which is the wrong
question for every multi-subject design. Identical to the set
:class:`~behavio.pymc_backend.PyMCHierarchicalGLMBackend` retains, so a Bambi posterior and a
first-party one block the same way.
"""

SUPPORTED_FAMILIES: Final = ("bernoulli", "categorical")
"""The Bambi families this wrapper predicts through, and why it is not every family.

A Behavio ``predict`` must return one of three shapes.
:class:`~behavio.contracts.Prediction` is a binary probability, which ``bernoulli`` is;
:class:`~behavio.contracts.CategoricalPrediction` is a simplex over named categories, which
``categorical`` is. The third, :class:`~behavio.contracts.DensityPrediction`, is a density on
a continuous grid, and reaching it from Bambi's ``gaussian``, ``t``, ``gamma`` or ``wald``
would mean tabulating each family's density in this module -- re-hand-rolling exactly what
wrapping a maintained package is supposed to stop. ``poisson`` and ``negativebinomial`` have
no shape at all: ``ModelPrediction`` names nothing for an unbounded count, and truncating one
to a categorical simplex would be a modelling decision disguised as a data structure.

Both gaps are gaps in Behavio, not in Bambi, and both are named in ``docs/foreign-models.md``
rather than papered over with an approximation.
"""

_LINKS: Final = MappingProxyType(
    {
        "bernoulli": ("logit", "probit", "cloglog", "identity", "log"),
        "categorical": ("softmax",),
    }
)

_PRIOR_CORRESPONDENCE: Final = MappingProxyType(
    {
        PriorFamily.NORMAL: ("Normal", ("location", "mu"), ("scale", "sigma")),
        PriorFamily.HALF_NORMAL: ("HalfNormal", ("scale", "sigma")),
        PriorFamily.BETA: ("Beta", ("alpha", "alpha"), ("beta", "beta")),
        PriorFamily.UNIFORM: ("Uniform", ("lower", "lower"), ("upper", "upper")),
    }
)
"""Behavio's four scalar prior families and the identically parameterised Bambi ones.

Exact, not approximate: each pair is ``(behavio argument, bambi keyword)`` and no
reparameterisation happens between them. Exported through :func:`prior_correspondence` so a
reader can check the mapping without opening the source.
"""

_PROBABILITY_FLOOR: Final = 1e-12
"""Clip applied only when inverting a probability to a linear predictor.

The probabilities themselves are reported as computed. A draw-averaged probability of
exactly zero or one has an infinite logit, and :class:`~behavio.contracts.Prediction`
requires a finite linear predictor; clipping the *reported logit* says "beyond this the
linear predictor is not resolvable from these draws", which is true, while clipping the
probability would silently move the number a score is computed from.
"""


def prior_correspondence() -> Mapping[str, str]:
    """Return the exact Behavio-to-Bambi prior mapping, one readable line per family."""

    lines = {}
    for family, entry in _PRIOR_CORRESPONDENCE.items():
        distribution, *pairs = entry
        arguments = ", ".join(f"{ours}->{theirs}" for ours, theirs in pairs)
        lines[str(family.value)] = f"bambi.Prior({distribution!r}, {arguments})"
    return MappingProxyType(lines)


@dataclass(frozen=True, slots=True)
class BambiRegression(Describable):
    """A Bambi regression sampled with PyMC NUTS, behind Behavio's sampled contract.

    ``formula`` is **Bambi's** formula string and is passed through verbatim; Behavio's own
    formula language does not apply to it, and nothing here translates between the two. See
    the module docstring for the argument.

    Everything that changes the numbers is a field, so the model is reproducible from its
    :attr:`signature` plus the study: the formula, the family and its link, the declared
    priors, whether Bambi is allowed to scale the undeclared ones from the data, the
    centring of the group-specific parameterisation, and the entire sampler configuration
    including its seed.

    ``categories`` is required for the ``categorical`` family and forbidden otherwise. The
    outcome's level set is part of the model specification rather than a fact about a fold:
    a training fold that happened to lose a level would otherwise fit a different model
    under the same signature. Bambi orders levels by their sorted string form, so the
    declared tuple must already be in that order and construction refuses one that is not.
    """

    formula: str
    family: str = "bernoulli"
    link: str | None = None
    categories: tuple[Any, ...] | None = None
    priors: Mapping[str, PriorSpec] | None = None
    auto_scale: bool = True
    noncentered: bool = True
    draws: int = 1_000
    tune: int = 1_000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    seed: int = 0
    sample_new_groups: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ValueError("formula must be a non-empty Bambi formula string")
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"family must be one of {list(SUPPORTED_FAMILIES)}; Behavio has no prediction "
                "shape for the others (see SUPPORTED_FAMILIES)"
            )
        if self.link is not None and self.link not in _LINKS[self.family]:
            raise ValueError(
                f"link for the {self.family} family must be one of {list(_LINKS[self.family])}"
            )
        categories = None if self.categories is None else tuple(self.categories)
        if self.family == "categorical":
            if categories is None or len(categories) < 2:
                raise ValueError(
                    "the categorical family requires at least two declared categories; the "
                    "outcome's level set is part of the model, not of a fold"
                )
            if len(set(map(str, categories))) != len(categories):
                raise ValueError("declared categories must be distinct under their string form")
            if list(categories) != sorted(categories, key=str):
                raise ValueError(
                    "Bambi orders a categorical response's levels by their sorted string form, "
                    f"so declare them in that order: {sorted(categories, key=str)!r}"
                )
        elif categories is not None:
            raise ValueError("categories applies only to the categorical family")
        priors = self._validated_priors()
        for value, name in (
            (self.draws, "draws"),
            (self.tune, "tune"),
            (self.chains, "chains"),
            (self.cores, "cores"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.chains < 2:
            raise ValueError("chains must be at least two for cross-chain diagnostics")
        if self.cores > self.chains:
            raise ValueError("cores cannot exceed chains")
        if not np.isfinite(self.target_accept) or not 0.5 <= self.target_accept < 1.0:
            raise ValueError("target_accept must be finite and in [0.5, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for value, name in (
            (self.auto_scale, "auto_scale"),
            (self.noncentered, "noncentered"),
            (self.sample_new_groups, "sample_new_groups"),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(self, "formula", self.formula.strip())
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "priors", priors)

    def _validated_priors(self) -> Mapping[str, PriorSpec] | None:
        if self.priors is None:
            return None
        if not isinstance(self.priors, Mapping):
            raise TypeError("priors must be a mapping of Bambi term name to PriorSpec")
        validated: dict[str, PriorSpec] = {}
        for name in sorted(self.priors):
            value = self.priors[name]
            if not isinstance(name, str) or not name:
                raise ValueError("prior term names must be non-empty strings")
            if not isinstance(value, PriorSpec):
                raise TypeError(
                    f"prior for {name!r} must be a behavio PriorSpec; a bambi.Prior carries no "
                    "JSON-safe form and so could not enter the model's signature"
                )
            if "|" in name:
                raise ValueError(
                    f"prior for the group-specific term {name!r} is refused: Bambi spells one as "
                    "a Normal whose sigma is itself a Prior, and a scalar PriorSpec can only "
                    "express a Normal with a fixed sigma -- which is a model with no pooling, "
                    "not a weaker hierarchical prior. Leave it to Bambi's own default"
                )
            validated[name] = value
        return MappingProxyType(validated)

    # -- identity ------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "bambi-regression"

    @property
    def signature(self) -> str:
        """A scientific fingerprint over a foreign sampled configuration.

        What is in it: the formula verbatim, the family and link, the declared categories,
        every declared prior on its natural coordinate, whether Bambi may scale the
        undeclared priors from data, the group-specific parameterisation, the whole sampler
        configuration including its seed, and whether unseen groups may be predicted.

        What is deliberately **out** of it, and why this differs from
        :attr:`behavio.foreign.pyddm.PyDDMDriftDiffusion.signature`: the installed Bambi and
        PyMC versions. PyDDM's series is in its signature because a first-passage solver *is*
        the model -- change the numerics and the density changes under the same parameters.
        A sampler is not the model; it is a Monte Carlo approximation to a target that the
        formula, family and priors already define, and whose adequacy is measured rather than
        assumed, by :func:`~behavio.posterior.audit_posterior` on every fold. Writing the
        sampler's version into the fingerprint would make two fits of the same target
        incomparable because one machine had a patch release, while the thing that genuinely
        can differ between Bambi series -- the *default* priors -- is recorded exactly, as the
        realised ``prior_specification`` on each posterior. The exact versions are provenance
        and are carried as ``inference_library_version``.
        """

        parts = [
            "backend=bambi.pymc-nuts",
            f"formula={self.formula}",
            f"family={self.family}",
            f"link={self.link or 'default'}",
            f"auto_scale={self.auto_scale}",
            f"noncentered={self.noncentered}",
            f"draws={self.draws}",
            f"tune={self.tune}",
            f"chains={self.chains}",
            f"target_accept={self.target_accept}",
            f"seed={self.seed}",
            f"sample_new_groups={self.sample_new_groups}",
        ]
        if self.categories is not None:
            parts.append("categories=" + ",".join(str(value) for value in self.categories))
        if self.priors:
            declared = ";".join(
                f"{name}:{spec.family.value}("
                + ",".join(f"{key}={spec.arguments[key]:.17g}" for key in sorted(spec.arguments))
                + ")"
                for name, spec in self.priors.items()
            )
            parts.append(f"priors={declared}")
        return f"{self.model_name}[" + "|".join(parts) + "]"

    @property
    def outcome(self) -> str:
        """The response column the Bambi formula names, read from the formula alone."""

        return _response_name(self.formula)

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return (self.outcome,)

    @property
    def formula_columns(self) -> tuple[str, ...]:
        """Every ``Study`` column the Bambi formula reads, response first.

        Read from ``formulae``'s parse of the string alone, so it is available before any
        study exists, and it includes the grouping factors of group-specific terms and the
        four required identity columns when the formula names them -- ``(1|subject)`` does.
        This is the exact set retained on the posterior, which is what lets ``predict``
        rebuild the fitted model rather than re-derive a design from the rows it is scoring.
        """

        others = sorted(_formula_variables(self.formula) - {self.outcome})
        return (self.outcome, *others)

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        """The formula's columns that the ``Study`` contract does not already mandate.

        The four identity columns are excluded because every study has them; this is the
        set a task specification has to supply, which is what the contract asks for.
        :attr:`formula_columns` is the complete list.
        """

        return tuple(
            name
            for name in self.formula_columns
            if name != self.outcome and name not in REQUIRED_COLUMNS
        )

    @property
    def grouping_factors(self) -> tuple[str, ...]:
        """The factor columns of every group-specific term, in formula order."""

        return _grouping_factors(self.formula)

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        """Filtered only.

        A Bambi row's linear predictor is a function of that row's own design values and of
        parameters estimated from the training fold. Nothing in a ``formulae`` design reads a
        later row -- its stateful transforms carry their training state forward rather than
        re-estimating -- so there is no smoothed counterpart to report, and
        :func:`behavio.adapters.check_behaviour_estimator` verifies that behaviourally
        instead of taking the declaration on trust.
        """

        return (PredictionMode.FILTERED,)

    @property
    def declared_priors(self) -> tuple[str, ...]:
        """Human-readable priors for :meth:`describe`, and what is left to Bambi."""

        lines = [
            f"{name}: {spec.family.value}("
            + ", ".join(f"{key}={spec.arguments[key]:g}" for key in sorted(spec.arguments))
            + ")"
            for name, spec in (self.priors or {}).items()
        ]
        undeclared = (
            "every other term: Bambi's default, scaled from the training data (auto_scale=True)"
            if self.auto_scale
            else "every other term: Bambi's default, unscaled (auto_scale=False)"
        )
        return (*lines, undeclared)

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        """Report what will go wrong before a sampler is started."""

        findings: list[ModelFinding] = []
        if self.auto_scale:
            findings.append(
                ModelFinding(
                    code="data_dependent_default_priors",
                    severity=WARNING,
                    message=(
                        "auto_scale=True lets Bambi compute the scale of every undeclared "
                        "prior from the data it is handed, so each fold of a prospective "
                        "evaluation fits a slightly different model. The realised priors are "
                        "recorded on every posterior as prior_specification; set "
                        "auto_scale=False for one model across folds"
                    ),
                )
            )
        missing = [name for name in self.required_task_columns if name not in study.columns]
        if self.outcome not in study.columns:
            missing.append(self.outcome)
        if missing:
            findings.append(
                ModelFinding(
                    code="missing_formula_column",
                    severity=ERROR,
                    message=f"the formula reads columns the study does not have: {sorted(missing)}",
                )
            )
            return tuple(findings)
        if self.family == "categorical":
            observed = {str(value) for value in np.asarray(study[self.outcome])}
            declared = {str(value) for value in self.categories or ()}
            if not observed <= declared:
                findings.append(
                    ModelFinding(
                        code="undeclared_outcome_level",
                        severity=ERROR,
                        message=(
                            f"outcome {self.outcome!r} contains levels the model does not "
                            f"declare: {sorted(observed - declared)}"
                        ),
                    )
                )
            elif observed != declared:
                findings.append(
                    ModelFinding(
                        code="unobserved_outcome_level",
                        severity=WARNING,
                        message=(
                            f"declared levels {sorted(declared - observed)} do not occur in this "
                            "study; Bambi builds the response coordinate from the rows it sees, "
                            "so sampling will refuse rather than fit an unidentified level"
                        ),
                    )
                )
        else:
            values = np.asarray(study[self.outcome])
            distinct = np.unique(values)
            if not np.all(np.isin(distinct, (0, 1))):
                findings.append(
                    ModelFinding(
                        code="non_binary_outcome",
                        severity=ERROR,
                        message=(
                            f"the bernoulli family needs outcome {self.outcome!r} to contain "
                            f"only zero and one; found {list(distinct)[:6]}"
                        ),
                    )
                )
            elif distinct.size < 2:
                findings.append(
                    ModelFinding(
                        code="single_outcome_level",
                        severity=WARNING,
                        message=(
                            f"outcome {self.outcome!r} takes one value in this study; the "
                            "posterior will be prior-dominated and separated"
                        ),
                    )
                )
        for factor in self.grouping_factors:
            levels = np.unique(np.asarray(study[factor]))
            if levels.size < 2:
                findings.append(
                    ModelFinding(
                        code="single_group_level",
                        severity=WARNING,
                        message=(
                            f"group-specific term over {factor!r} has {levels.size} level in "
                            "this study, so its variance is not identified"
                        ),
                    )
                )
        return tuple(findings)

    # -- inference -----------------------------------------------------------------------

    def sample(self, study: Study) -> PosteriorResult:
        """Sample the posterior with PyMC NUTS and return Behavio's labelled result.

        The returned :class:`~behavio.posterior.PosteriorResult` carries everything the rest
        of the package needs from it and Bambi does not supply: the training rows the
        formula read, the four grouping columns :data:`BLOCKING_VARIABLES` names, the
        realised prior specification, and the observation dimension renamed to
        :data:`TRIAL_DIM`.
        """

        bambi = require_bambi()
        self.validate(study)
        frame = self._frame(study, outcome=True)
        model = self._model(bambi, frame)
        with _quiet_bambi():
            model.build()
            inference = model.fit(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=self.target_accept,
                random_seed=self._sampling_seeds(),
                progressbar=False,
                idata_kwargs={"log_likelihood": True},
            )
        result = posterior_result_from_arviz(
            inference,
            model_name=self.model_name,
            model_signature=self.signature,
            inference_library="Bambi",
            inference_library_version=_backend_version(),
            parameter_names=_sampled_parameter_names(inference),
        )
        result = _rename_observation_dim(result)
        if "log_likelihood" not in result.group_names:
            raise PosteriorError(
                "Bambi returned no pointwise log likelihood, so this posterior could not be "
                "scored or blocked; the wrapper asks for it explicitly"
            )
        return _with_behavio_evidence(
            result,
            study=study,
            frame=frame,
            attrs=self._attributes(model, study),
        )

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult:
        """Project the posterior to the frequentist contract, measuring nothing new.

        The reference projection, unchanged:
        :func:`~behavio.contracts.posterior.posterior_point_summary` reduces the draws to a
        centre, a posterior standard deviation and a posterior covariance and leaves every
        optimizer-shaped diagnostic absent. It is used verbatim rather than specialised,
        which is the point -- a foreign posterior either passes through it or the projection
        was never general.
        """

        self._require_posterior(posterior)
        return posterior_point_summary(posterior, converged=converged, centre=centre)

    # -- prediction and scoring ----------------------------------------------------------

    def predict(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction:
        """Return the draw-averaged predictive probability of each row's outcome.

        The average is taken over draws on the **probability** scale, which is the posterior
        predictive probability and is exactly the margin implied by
        :meth:`pointwise_log_prob`; averaging the linear predictor instead would give a
        different, smaller-variance number that no score in this package uses.

        The Bambi model is rebuilt from the training rows retained on ``posterior``, never
        from ``study``. That is what makes the design, the stateful transforms and the group
        coordinate identical to the ones the sampler saw.
        """

        self._require_mode(mode)
        probability = self._response_probability(study, posterior)
        if self.family == "categorical":
            return CategoricalPrediction(
                probability=probability,
                linear_predictor=np.log(np.clip(probability, _PROBABILITY_FLOOR, None)),
                categories=tuple(self.categories or ()),
                mode=PredictionMode(mode),
            )
        clipped = np.clip(probability, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
        return Prediction(
            probability=probability,
            linear_predictor=np.log(clipped / (1.0 - clipped)),
            mode=PredictionMode(mode),
        )

    def pointwise_log_prob(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        """Return the log pointwise predictive density of each row, over the whole posterior.

        ``log((1/S) sum_s p(y_i | theta_s))``, and deliberately not
        ``log p(y_i | mean(theta))``, which by Jensen is a different and generally larger
        quantity that would not be comparable with any other Bayesian score in this package.

        For a family with finite support the per-draw density *is* the response parameter:
        ``p(y_i | theta_s)`` is entry ``y_i`` of the simplex Bambi's inverse link produces
        for draw ``s``. This method therefore averages that simplex over draws and reads the
        observed entry, which is the same number
        :func:`~behavio.contracts.posterior.posterior_log_predictive_density` would return
        from per-draw log densities -- exactly, not approximately, because averaging on the
        probability scale is what both do.

        Two things follow, and both are why this route was chosen over a second call to
        :meth:`bambi.Model.compute_log_likelihood`. It costs one evaluation of Bambi's graph
        rather than two; and :meth:`predict` and this method are then *the same computation*,
        so the discrete prediction and the score it implies cannot drift apart. That the
        result really is Bambi's own likelihood is a checkable claim rather than an assertion:
        the test suite compares it against the ``log_likelihood`` group Bambi itself computed
        during sampling, on the training rows, for every supported family.

        (Bambi 0.19's ``compute_log_likelihood`` also happens to raise a PyTensor broadcast
        error for the ``categorical`` family on out-of-sample rows, which would have blocked
        that route for half the supported families in any case.)
        """

        self._require_mode(mode)
        if self.outcome not in study.columns:
            raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
        probability = self._response_probability(study, posterior)
        if self.family == "categorical":
            codes = self.outcome_codes(study)
            observed = probability[np.arange(len(study)), codes]
        else:
            outcomes = np.asarray(study[self.outcome], dtype=np.float64)
            observed = np.where(outcomes == 1, probability, 1.0 - probability)
        return protected_array(
            np.log(np.clip(observed, _PROBABILITY_FLOOR, None)), dtype=np.float64
        )

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        """Return each row's observed level as an index into :attr:`categories`."""

        if self.family != "categorical":
            raise ModelDataError("outcome codes are declared only for the categorical family")
        declared = {str(value): index for index, value in enumerate(self.categories or ())}
        observed = [str(value) for value in np.asarray(study[self.outcome])]
        unknown = sorted({value for value in observed if value not in declared})
        if unknown:
            raise ModelDataError(f"outcome column contains undeclared levels {unknown}")
        return protected_array([declared[value] for value in observed], dtype=np.int64)

    # -- binding -------------------------------------------------------------------------

    def bind(self, design: Study) -> DesignBoundBambiRegression:
        """Return the same model bound to one design, as a generative sampled model.

        :class:`BambiRegression` cannot satisfy
        :class:`~behavio.contracts.posterior.GenerativePosteriorBehaviourModel`, because
        ``parameter_names`` is a study-independent property and a mixed-effects model's
        parameter vector is not: the coordinates of ``1|subject``, the columns of
        ``C(condition)`` and the basis of ``bs(x, df=5)`` are all facts about the data.
        Binding supplies the missing fact. Since ``run_parameter_recovery`` already takes the
        design as its second argument, ``model.bind(design)`` states something true about
        recovery rather than working around something false about the contract.
        """

        return DesignBoundBambiRegression(self, design)

    # -- the Bambi boundary --------------------------------------------------------------

    def _model(self, bambi: Any, frame: Any) -> Any:
        """Construct the Bambi model. No pytensor graph is built until ``build()``."""

        keywords: dict[str, Any] = {
            "family": self.family,
            "priors": self._bambi_priors(bambi),
            "auto_scale": self.auto_scale,
            "noncentered": self.noncentered,
            "dropna": False,
        }
        if self.link is not None:
            keywords["link"] = self.link
        if self.family == "categorical":
            keywords["categorical"] = [self.outcome]
        with _quiet_bambi():
            model = bambi.Model(self.formula, frame, **keywords)
        if self.family == "categorical":
            levels = tuple(str(value) for value in model.response_component.term.levels)
            declared = tuple(str(value) for value in self.categories or ())
            if levels != declared:
                raise ModelDataError(
                    f"Bambi built the response coordinate as {list(levels)} and the model "
                    f"declares {list(declared)}; every declared level must occur in the study "
                    "and the declaration must be in Bambi's sorted order"
                )
        return model

    def _bambi_priors(self, bambi: Any) -> dict[str, Any] | None:
        if not self.priors:
            return None
        return {name: _bambi_prior(bambi, spec) for name, spec in self.priors.items()}

    def _rebuilt(self, bambi: Any, posterior: PosteriorResult) -> tuple[Any, Any]:
        """Rebuild the fitted Bambi model from the training rows retained on the posterior."""

        self._require_posterior(posterior)
        frame = _training_frame(posterior, self.formula_columns)
        model = self._model(bambi, frame)
        with _quiet_bambi():
            model.build()
        return model, frame

    def _response_probability(
        self, study: Study, posterior: PosteriorResult
    ) -> NDArray[np.float64]:
        bambi = require_bambi()
        model, _ = self._rebuilt(bambi, posterior)
        frame = self._frame(study, outcome=False)
        self._require_known_groups(frame, posterior)
        parent = str(model.family.likelihood.parent)
        with _quiet_bambi():
            predicted = model.predict(
                _to_arviz(posterior),
                kind="response_params",
                data=frame,
                inplace=False,
                include_group_specific=True,
                sample_new_groups=self.sample_new_groups,
            )
        values = np.asarray(_group_variable(predicted, "posterior", parent), dtype=np.float64)
        averaged = values.reshape(-1, *values.shape[2:]).mean(axis=0)
        if self.family == "categorical":
            averaged = averaged / np.sum(averaged, axis=1, keepdims=True)
        return protected_array(averaged, dtype=np.float64)

    def _require_known_groups(self, frame: Any, posterior: PosteriorResult) -> None:
        """Refuse a prediction whose group levels the fit never estimated, naming them."""

        if self.sample_new_groups:
            return
        training = _training_frame(posterior, self.formula_columns)
        for factor in self.grouping_factors:
            seen = {str(value) for value in training[factor]}
            asked = {str(value) for value in frame[factor]}
            unseen = sorted(asked - seen)
            if unseen:
                raise ModelDataError(
                    f"the group-specific term over {factor!r} has no estimated effect for "
                    f"{unseen[:8]}{'...' if len(unseen) > 8 else ''}: these levels are absent "
                    "from the fitted rows. Set sample_new_groups=True to draw them from the "
                    "hyperprior instead, which is a prior-predictive claim about a new group "
                    "rather than a held-out prediction about a known one"
                )

    def _frame(self, study: Study, *, outcome: bool) -> Any:
        """Return the pandas frame Bambi reads, with only the columns the formula names."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        pandas = _require_pandas()
        columns: dict[str, Any] = {}
        for name in self.formula_columns[1:]:
            if name not in study.columns:
                raise ModelDataError(f"study is missing formula column {name!r}")
            columns[name] = np.asarray(study[name])
        if outcome:
            if self.outcome not in study.columns:
                raise ModelDataError(f"study is missing outcome column {self.outcome!r}")
            columns[self.outcome] = self._response_column(pandas, np.asarray(study[self.outcome]))
        elif self.outcome in study.columns:
            columns[self.outcome] = self._response_column(pandas, np.asarray(study[self.outcome]))
        else:
            columns[self.outcome] = self._placeholder_response(pandas, len(study))
        return pandas.DataFrame(columns)

    def _response_column(self, pandas: Any, values: NDArray[Any]) -> Any:
        if self.family != "categorical":
            return np.asarray(values, dtype=np.int64)
        return pandas.Categorical([str(value) for value in values])

    def _placeholder_response(self, pandas: Any, length: int) -> Any:
        """A stand-in outcome so ``formulae`` can build a design for a study without one.

        A recovery design carries no observations. Bambi needs a response column to parse the
        formula at all, and nothing that reads this column is ever consulted: simulation
        draws the response from the family, and every path that *scores* an outcome refuses a
        study that does not have one.
        """

        if self.family != "categorical":
            return np.zeros(length, dtype=np.int64)
        levels = [str(value) for value in self.categories or ()]
        return pandas.Categorical([levels[index % len(levels)] for index in range(length)])

    def _attributes(self, model: Any, study: Study) -> dict[str, Any]:
        return {
            "backend": "bambi.pymc-nuts",
            "backend_config": {
                "draws": self.draws,
                "tune": self.tune,
                "chains": self.chains,
                "cores": self.cores,
                "target_accept": self.target_accept,
                "seed": self.seed,
                "auto_scale": self.auto_scale,
                "noncentered": self.noncentered,
            },
            "prediction_mode": "filtered",
            "scored_columns": list(self.scored_columns),
            "formula": self.formula,
            "family": self.family,
            "trial_coordinate": "study source-row position",
            "blocking_variables": list(BLOCKING_VARIABLES),
            "grouping_factors": list(self.grouping_factors),
            "training_columns": list(self.formula_columns),
            "n_training_rows": len(study),
            # The one thing a Bambi series or an auto_scale run can change without changing
            # the signature, recorded exactly as Bambi built it.
            "prior_specification": _prior_specification(model),
        }

    def _sampling_seeds(self) -> list[int]:
        """Derive one 32-bit chain seed per chain from the declared entropy.

        ``seed`` is entropy for :class:`numpy.random.SeedSequence`, so any non-negative
        integer is admissible and a seed emitted by
        :mod:`behavio.recovery.parameters` can be handed straight here -- the same contract
        :class:`~behavio.pymc_backend.PyMCHierarchicalGLMBackend` already keeps.
        """

        words = np.random.SeedSequence(self.seed).generate_state(self.chains)
        return [int(value) for value in words]

    def _require_posterior(self, posterior: PosteriorResult) -> None:
        if not isinstance(posterior, PosteriorResult):
            raise TypeError("posterior must be a PosteriorResult")
        if posterior.model_signature != self.signature:
            raise ValueError("posterior belongs to a different model specification")

    def _require_mode(self, mode: PredictionMode) -> None:
        if PredictionMode(mode) not in self.supported_prediction_modes:
            raise UnsupportedPredictionMode(
                "BambiRegression supports only filtered prediction: a regression row's linear "
                "predictor is a function of that row's own design values, so there is no "
                "smoothed counterpart to report"
            )


@dataclass(frozen=True, slots=True)
class DesignBoundBambiRegression(Describable):
    """A :class:`BambiRegression` bound to one design, and therefore generative.

    Binding resolves the one thing
    :class:`~behavio.contracts.posterior.GenerativePosteriorBehaviourModel` needs and a
    formula alone cannot supply: *which scalar parameters exist*. Bambi answers that only
    once it has data -- ``1|subject`` has as many coordinates as the design has subjects --
    so the bound object derives :attr:`parameter_names` from a one-draw prior predictive over
    the design, which is exhaustive by construction and costs no sampling.

    :attr:`posterior_parameter_labels` is then the identity map, because the names are read
    from the same flattening
    :func:`~behavio.contracts.posterior.posterior_point_summary` uses. That is the whole
    reason the labels exist -- a first-party model has to declare a correspondence between
    two vocabularies, and here there is genuinely only one.

    The bound design is the design ``simulate`` is called with; passing a different one is
    refused, because the parameter vector would then name coordinates the new design does not
    have.
    """

    model: BambiRegression
    design: Study
    _names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _template: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, BambiRegression):
            raise TypeError("model must be a BambiRegression")
        if not isinstance(self.design, Study):
            raise TypeError("design must be a Study")
        bambi = require_bambi()
        built = self.model._model(bambi, self.model._frame(self.design, outcome=False))
        with _quiet_bambi():
            built.build()
            prior = built.prior_predictive(draws=1, random_seed=0)
        parent = str(built.family.likelihood.parent)
        template = _prior_template(prior, exclude=(parent,))
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_names", _flattened_names(template))

    # -- the sampled contract, delegated unchanged ---------------------------------------

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def signature(self) -> str:
        return self.model.signature

    @property
    def scored_columns(self) -> tuple[str, ...]:
        return self.model.scored_columns

    @property
    def required_task_columns(self) -> tuple[str, ...]:
        return self.model.required_task_columns

    @property
    def supported_prediction_modes(self) -> tuple[PredictionMode, ...]:
        return self.model.supported_prediction_modes

    @property
    def declared_priors(self) -> tuple[str, ...]:
        return self.model.declared_priors

    def additional_findings(self, study: Study) -> tuple[ModelFinding, ...]:
        return self.model.additional_findings(study)

    def sample(self, study: Study) -> PosteriorResult:
        return self.model.sample(study)

    def point_summary(
        self,
        posterior: PosteriorResult,
        *,
        converged: bool,
        centre: PosteriorCentre = PosteriorCentre.MEAN,
    ) -> FitResult:
        return self.model.point_summary(posterior, converged=converged, centre=centre)

    def predict(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> ModelPrediction:
        return self.model.predict(study, posterior, mode=mode)

    def pointwise_log_prob(
        self,
        study: Study,
        posterior: PosteriorResult,
        *,
        mode: PredictionMode = PredictionMode.FILTERED,
    ) -> NDArray[np.float64]:
        return self.model.pointwise_log_prob(study, posterior, mode=mode)

    def outcome_codes(self, study: Study) -> NDArray[np.int64]:
        return self.model.outcome_codes(study)

    # -- what binding adds ---------------------------------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Every scalar coordinate of this design's posterior, in Bambi's own order.

        Includes the group-level effects themselves, not only their hyperparameters, because
        ``simulate`` writes exact values into the graph and a partially specified linear
        predictor is not a generative model. A recovery truth must therefore name a value for
        each subject; that is more work to write down than a two-parameter GLM, and it is
        what recovering a mixed model actually requires.
        """

        return self._names

    @property
    def posterior_parameter_labels(self) -> Mapping[str, str]:
        """The identity map: simulator names *are* the projected posterior coordinates."""

        return MappingProxyType({name: name for name in self._names})

    def simulate(
        self,
        design: Study,
        parameters: Mapping[str, float],
        *,
        seed: int | np.random.Generator,
    ) -> Study:
        """Draw the outcome from Bambi's own family at the supplied parameter values.

        The values are written into a one-draw posterior carrying the model's own dimensions
        and coordinates, and ``bambi.Model.predict(kind="response")`` draws the observation.
        Nothing about the family's random variate is implemented here, so recovery tests the
        inference rather than two implementations of one likelihood.
        """

        if not isinstance(design, Study):
            raise TypeError("design must be a Study")
        if design is not self.design and not _same_study(design, self.design):
            raise ValueError(
                "a bound model simulates only the design it was bound to; its parameter names "
                "are coordinates of that design's posterior"
            )
        values = self._validated_parameters(parameters)
        bambi = require_bambi()
        frame = self.model._frame(design, outcome=False)
        built = self.model._model(bambi, frame)
        generator = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        draw_seed = int(generator.integers(0, 2**31 - 1))
        with _quiet_bambi():
            built.build()
            drawn = built.predict(
                _filled_template(self._template, values),
                kind="response",
                data=frame,
                inplace=False,
                random_seed=draw_seed,
                sample_new_groups=False,
            )
        sampled = np.asarray(
            _group_variable(drawn, "posterior_predictive", self.model.outcome)
        ).reshape(-1)
        columns: dict[str, Any] = {name: design[name] for name in design.columns}
        columns[self.model.outcome] = self._decoded(sampled)
        return Study(columns)

    def _decoded(self, sampled: NDArray[Any]) -> NDArray[Any]:
        if self.model.family != "categorical":
            return np.asarray(sampled, dtype=np.int64)
        levels = np.asarray([str(value) for value in self.model.categories or ()])
        return levels[np.asarray(sampled, dtype=np.int64)]

    def _validated_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        expected, observed = set(self._names), set(parameters)
        if observed != expected:
            raise ValueError(
                "parameters must match the bound design's coordinates exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            values = {name: float(parameters[name]) for name in self._names}
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite numeric values") from None
        if not np.all(np.isfinite(list(values.values()))):
            raise ValueError("parameters must contain finite numeric values")
        return values


# ------------------------------------------------------------------------------------------
# Formula reading. Everything here is a fact about the formula string alone, so it is
# available before a study exists and cannot disagree with what Bambi later builds.
# ------------------------------------------------------------------------------------------


def _model_description(formula: str) -> Any:
    formulae = _require_formulae()
    try:
        return formulae.model_description(formula)
    except Exception as error:  # formulae raises many shapes for one kind of mistake
        raise ValueError(f"{formula!r} is not a valid Bambi formula: {error}") from error


def _response_name(formula: str) -> str:
    description = _model_description(formula)
    response = getattr(description, "response", None)
    if response is None:
        raise ValueError(f"{formula!r} declares no response column")
    name = str(response.term.name)
    if not name.isidentifier():
        raise ValueError(
            f"the response of {formula!r} is the expression {name!r}; this wrapper needs a "
            "plain column name, because Behavio scores a Study column and a transformed "
            "response is not one"
        )
    return name


def _formula_variables(formula: str) -> set[str]:
    return {str(name) for name in _model_description(formula).var_names}


def _grouping_factors(formula: str) -> tuple[str, ...]:
    factors: list[str] = []
    for term in _model_description(formula).terms:
        factor = getattr(term, "factor", None)
        if factor is None:
            continue
        name = str(getattr(factor, "name", ""))
        if name and name not in factors:
            factors.append(name)
    return tuple(factors)


# ------------------------------------------------------------------------------------------
# Posterior assembly.
# ------------------------------------------------------------------------------------------


def _sampled_parameter_names(inference: Any) -> tuple[str, ...]:
    """Every posterior variable Bambi retained, which is every parameter it sampled.

    Bambi omits the non-centred offsets from its returned posterior by default and keeps the
    reconstructed group effects, so this is the constrained, natural coordinate set the
    convergence audit should be run on -- not a subset chosen here.

    Sorted, and that is load-bearing rather than tidiness. This order is the order
    :func:`~behavio.contracts.posterior.posterior_point_summary` flattens in, and
    :attr:`DesignBoundBambiRegression.parameter_names` is read from a *prior* predictive over
    the same model, which PyMC lists in a different order. Sorting both makes the simulator's
    vocabulary and the projection's vocabulary identical rather than merely equal as sets --
    which is what lets the declared label map be the identity, and what lets a conformance
    harness compare the two tuples at all.
    """

    names = tuple(sorted(str(name) for name in _group_dataset(inference, "posterior").data_vars))
    if not names:
        raise PosteriorError("Bambi returned an empty posterior group")
    return names


def _rename_observation_dim(result: PosteriorResult) -> PosteriorResult:
    """Rename Bambi's ``__obs__`` dimension to :data:`TRIAL_DIM` throughout."""

    groups = []
    for group in result.groups:
        variables = tuple(
            PosteriorVariable(
                name=variable.name,
                values=variable.values,
                dims=tuple(TRIAL_DIM if dim == "__obs__" else dim for dim in variable.dims),
                coords={
                    (TRIAL_DIM if dim == "__obs__" else dim): values
                    for dim, values in variable.coords.items()
                },
            )
            for variable in group.variables
        )
        groups.append(PosteriorGroup(group.name, variables, attrs=group.attrs))
    return _replace_groups(result, tuple(groups), result.attrs)


def _with_behavio_evidence(
    result: PosteriorResult,
    *,
    study: Study,
    frame: Any,
    attrs: Mapping[str, Any],
) -> PosteriorResult:
    """Attach the grouping columns and the training rows Bambi does not retain."""

    trial = np.arange(len(study), dtype=np.int64)
    coords = {TRIAL_DIM: trial}
    retained = [
        PosteriorVariable("trial_subject", study["subject"], (TRIAL_DIM,), coords),
        PosteriorVariable("trial_session", study["session"], (TRIAL_DIM,), coords),
        PosteriorVariable("trial_in_session", study["trial"], (TRIAL_DIM,), coords),
        PosteriorVariable("trial_session_order", study["session_order"], (TRIAL_DIM,), coords),
    ]
    for column in frame.columns:
        name = f"training.{column}"
        retained.append(PosteriorVariable(name, np.asarray(frame[column]), (TRIAL_DIM,), coords))
    groups = list(result.groups)
    existing = next(
        (index for index, item in enumerate(groups) if item.name == "constant_data"), -1
    )
    if existing < 0:
        groups.append(PosteriorGroup("constant_data", tuple(retained)))
    else:
        collisions = set(groups[existing].variable_names) & {item.name for item in retained}
        if collisions:
            raise PosteriorError(
                f"Bambi constant data collides with retained trial identity: {sorted(collisions)}"
            )
        groups[existing] = PosteriorGroup(
            "constant_data",
            (*groups[existing].variables, *retained),
            attrs=groups[existing].attrs,
        )
    return _replace_groups(result, tuple(groups), {**dict(result.attrs), **dict(attrs)})


def _training_frame(posterior: PosteriorResult, columns: Sequence[str]) -> Any:
    """Rebuild the pandas frame the sampler saw, from ``constant_data``."""

    pandas = _require_pandas()
    if "constant_data" not in posterior.group_names:
        raise PosteriorError(
            "this posterior retains no training rows, so the fitted Bambi model cannot be "
            "rebuilt; it was not produced by BambiRegression.sample"
        )
    group = posterior["constant_data"]
    frame: dict[str, Any] = {}
    for name in columns:
        stored = f"training.{name}"
        if stored not in group.variable_names:
            raise PosteriorError(f"this posterior does not retain the training column {name!r}")
        frame[name] = np.asarray(group[stored].values)
    return pandas.DataFrame(frame)


def _replace_groups(
    result: PosteriorResult,
    groups: tuple[PosteriorGroup, ...],
    attrs: Mapping[str, Any],
) -> PosteriorResult:
    return PosteriorResult(
        model_name=result.model_name,
        model_signature=result.model_signature,
        inference_library=result.inference_library,
        inference_library_version=result.inference_library_version,
        parameter_names=result.parameter_names,
        groups=groups,
        parameter_space_fingerprint=result.parameter_space_fingerprint,
        attrs=attrs,
    )


# ------------------------------------------------------------------------------------------
# Simulation template.
# ------------------------------------------------------------------------------------------


def _prior_template(prior: Any, *, exclude: Sequence[str]) -> Any:
    """Return a one-draw prior dataset carrying every parameter's dims and coordinates."""

    dataset = _group_dataset(prior, "prior")
    # Sorted for the same reason ``_sampled_parameter_names`` is: the simulator's names and
    # the projected posterior's names must be one vocabulary, and PyMC lists a prior
    # predictive's variables in a different order from a posterior's.
    keep = sorted(str(name) for name in dataset.data_vars if str(name) not in set(exclude))
    if not keep:
        raise PosteriorError("Bambi's prior predictive declared no parameters to simulate from")
    return dataset[keep].isel(draw=[0])


def _flattened_names(template: Any) -> tuple[str, ...]:
    """Flatten a template dataset with the ``name[dim=label]`` convention the projection uses."""

    names: list[str] = []
    for raw in sorted(str(name) for name in template.data_vars):
        variable = template[raw]
        intrinsic = [str(dim) for dim in variable.dims if dim not in SAMPLE_DIMS]
        if not intrinsic:
            names.append(str(raw))
            continue
        labels = [
            [_scalar(value) for value in np.asarray(variable.coords[dim].values)]
            for dim in intrinsic
        ]
        for index in np.ndindex(*(len(values) for values in labels)):
            coordinate = ",".join(
                f"{dim}={labels[axis][position]!r}"
                for axis, (dim, position) in enumerate(zip(intrinsic, index, strict=True))
            )
            names.append(f"{raw}[{coordinate}]")
    if len(set(names)) != len(names):
        raise PosteriorError("Bambi produced two parameters with the same flattened name")
    return tuple(names)


def _filled_template(template: Any, values: Mapping[str, float]) -> Any:
    """Write scalar parameter values into a copy of the one-draw template, as a DataTree."""

    xarray = importlib.import_module("xarray")
    dataset = template.copy(deep=True)
    position = 0
    names = _flattened_names(template)
    for raw in sorted(str(name) for name in dataset.data_vars):
        variable = dataset[raw]
        intrinsic = [dim for dim in variable.dims if dim not in SAMPLE_DIMS]
        count = int(np.prod([len(variable.coords[dim]) for dim in intrinsic], dtype=np.int64))
        block = np.asarray(
            [values[name] for name in names[position : position + count]], dtype=np.float64
        )
        position += count
        dataset[raw] = (variable.dims, block.reshape(variable.shape))
    return xarray.DataTree.from_dict({"posterior": dataset})


# ------------------------------------------------------------------------------------------
# Small shared helpers.
# ------------------------------------------------------------------------------------------


def _bambi_prior(bambi: Any, spec: PriorSpec) -> Any:
    distribution, *pairs = _PRIOR_CORRESPONDENCE[spec.family]
    return bambi.Prior(distribution, **{theirs: spec.arguments[ours] for ours, theirs in pairs})


def _prior_specification(model: Any) -> str:
    """The priors Bambi actually built, as text, for the posterior's own record."""

    try:
        return str(model)
    except Exception:  # pragma: no cover - a summary that cannot render is not a fit failure
        return "unavailable"


def _to_arviz(posterior: PosteriorResult) -> Any:
    return posterior.to_arviz()


def _group_dataset(data: Any, name: str) -> Any:
    children = getattr(data, "children", None)
    if children is not None:
        if name not in children:
            raise PosteriorError(f"Bambi returned no {name} group")
        group = data[name]
        return group.to_dataset() if hasattr(group, "to_dataset") else group
    if not hasattr(data, name):
        raise PosteriorError(f"Bambi returned no {name} group")
    return getattr(data, name)


def _group_variable(data: Any, group: str, name: str) -> Any:
    dataset = _group_dataset(data, group)
    if name not in dataset:
        raise PosteriorError(f"Bambi's {group} group has no variable {name!r}")
    return dataset[name].values


def _same_study(left: Study, right: Study) -> bool:
    if len(left) != len(right) or set(left.columns) != set(right.columns):
        return False
    return all(
        np.array_equal(np.asarray(left[name]), np.asarray(right[name])) for name in left.columns
    )


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _require_pandas() -> Any:
    require_bambi()
    return importlib.import_module("pandas")


def _require_formulae() -> Any:
    require_bambi()
    return importlib.import_module("formulae")


def _backend_version() -> str:
    """Bambi's version and the PyMC that compiled the graph, as one provenance string."""

    try:
        pymc = importlib.metadata.version("pymc")
    except Exception:  # pragma: no cover - a missing distribution is not a fit failure
        pymc = "unknown"
    return f"{bambi_version()}+pymc{pymc}"


def _quiet_bambi() -> Any:
    """Silence Bambi's and PyMC's per-run logging and their sampling advisories.

    PyMC logs the initialisation strategy, the chain layout and its own R-hat/ESS advice on
    every call, and Bambi announces which level of a binary response it is modelling. None of
    it is a finding about the fit; what *is* a finding -- convergence, divergences, ESS -- is
    computed by :func:`~behavio.posterior.audit_posterior` from the returned draws and gates
    every fold.
    """

    return _QuietBambi()


class _QuietBambi:
    __slots__ = ("_catcher", "_levels")

    def __enter__(self) -> None:
        self._levels = {}
        for name in ("bambi", "pymc", "pytensor"):
            logger = logging.getLogger(name)
            self._levels[name] = logger.level
            logger.setLevel(logging.ERROR)
        self._catcher = warnings.catch_warnings()
        self._catcher.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        # PyTensor's Numba linker cannot cache a graph that closes over pointers, and says so
        # on every compile. It is a statement about a cache, not about the model.
        warnings.filterwarnings("ignore", message="Cannot cache compiled function")

    def __exit__(self, *exception: Any) -> None:
        self._catcher.__exit__(*exception)
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)


__all__ = [
    "BLOCKING_VARIABLES",
    "SUPPORTED_FAMILIES",
    "TRIAL_DIM",
    "BambiRegression",
    "DesignBoundBambiRegression",
    "prior_correspondence",
]
