"""What ``behavio`` promises at its top level, and the argument for every name in it.

``behavio.__all__`` is not a re-export of the package; it is a curated golden path, and the
sets below *are* that path. The final test asserts set equality, not containment: a name
added to the top level without an argument written here fails, and so does a name silently
dropped. Everything the package implements that is not named below is public and supported
at ``behavio.<area>.<name>``.

The rule the boundary is drawn with: **pin what a user writes down on the shortest correct
path from a table of trials to a validated, compared and audited result.** A type read back
off a call, an optional refinement of a pinned argument, a display, or a mechanism the
library uses to talk to code it did not write lives in its area instead.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import behavio

ROOT = Path(__file__).parents[1]

GOLDEN_PATH = {
    "Study",
    "ChoiceSpec",
    "RewardSpec",
    "ResponseTimeSpec",
    "TaskSpec",
    "fit_model",
    "compare_models",
    "cohort_forward_session_splits",
    "nested_select_model",
    "run_parameter_recovery",
    "run_model_recovery",
    "StudyProtocol",
    "compile_execution_plan",
    "run_protocol",
    "build_evidence_bundle",
    "EstimatorRegistry",
    # The registry is only an extension point if a caller can obtain the one the package
    # itself runs through and add to it. Pinning the constructor without the builtin would
    # pin an empty allowlist nothing resolves against.
    "builtin_estimator_registry",
    "export_fit",
}

MODEL_CATALOGUE = {
    "BiasOnly",
    "Psychometric",
    "PsychometricFunction",
    "Perseveration",
    "WinStayLoseShift",
    "BernoulliHistoryGLM",
    "BernoulliGLMHMM",
    "SessionDynamicBernoulliGLMHMM",
    "HierarchicalSessionDynamicBernoulliGLMHMM",
    "LabHierarchicalSessionDynamicBernoulliGLMHMM",
    "BinaryQLearning",
    "BinaryRLAgent",
    "MultinomialLogit",
    "WienerDriftDiffusion",
    "EqualVarianceSDT",
    "UnequalVarianceSDT",
    "MetaSDT",
}

#: The three combinators and the one call that turns a formula into a model. This is the
#: headline capability of the model layer: eleven hand-written variant classes across two axes
#: plus three unrelated hard-coded mixtures became three functions over an ordinary estimator,
#: so what a user is promised is the *set of verbs*, not a grid of nouns. ``SmoothModel``,
#: ``HierarchicalModel``, ``MixtureModel`` and ``MixtureRowModel`` are pinned because they are
#: what the verbs return and therefore what a caller annotates -- ``mix`` returns the first of
#: the two mixtures for a model composed through a linear predictor and the second for one
#: composed through a row objective, so a caller who annotates one has to be able to name both;
#: ``HierarchicalFitResult`` because a hierarchical fit's group deviations are read off it by
#: name and no other type carries them. The three components are pinned because ``mix`` is
#: useless without one and there is no way to write your own without first seeing these.
#:
#: Deliberately unpinned, and therefore reached at ``behavio.compose.<name>``:
#: ``HierarchicalSimulation``, ``MixtureSimulation`` and ``CoefficientTrajectory`` (records you
#: read back off a call, free to gain fields, in the same class as ``PsychometricSummary``), and
#: the whole of ``behavio.contracts.compose`` and ``behavio.contracts.mixture`` --
#: ``PenalisedLinearEstimator``, ``MixtureComponent`` and their neighbours are the surface a
#: *model author* implements, and they are reached through ``behavio.contracts`` alongside the
#: other extension protocols rather than from the top level.
COMBINATOR_SURFACE = {
    "smooth",
    "hierarchical",
    "mix",
    "model_from_formula",
    "SmoothModel",
    "HierarchicalModel",
    "HierarchicalFitResult",
    "MixtureModel",
    "MixtureRowModel",
    "UniformChoiceGuess",
    "UniformCategoryGuess",
    "UniformResponseGuess",
}

#: The closed-form detection-theory summaries. These are pinned for the same reason as the
#: model classes above and by the same test: they are the catalogue a psychophysicist reaches
#: for by name, and unlike a fitted model they are pure functions of a count table whose
#: published equations cannot move. ``forced_choice_proportion_correct`` is left out because
#: it only inverts ``forced_choice_d_prime``, and ``erf_two_gamma_probability`` because it
#: exists to reproduce one released implementation rather than to be a general entry point;
#: both remain public at ``behavio.models``.
CLOSED_FORM_SUMMARIES = {
    "detection_rates",
    "equal_variance_summary",
    "forced_choice_d_prime",
    "roc_points",
    "z_roc_summary",
}

# Documented and supported, deliberately not pinned by any set here, and therefore reached
# through their own area rather than from ``behavio``.
#
# The whole of ``behavio.plot``: the twelve ``plot_*`` displays, ``figure_style``,
# ``configure_figure_style``, ``save_svg`` and ``MatplotlibUnavailableError``. Plotting is the
# newest surface in the package and the one most likely to move: every function reads a report
# object, so each field a report gains is a figure that should show it, and three follow-ups
# already want to change these signatures. Pinning the shape of a display now would freeze it
# against the reports it exists to present. The palette constants and ``FIGURE_RC_PARAMS`` are
# not re-exported at the top level at all -- a house style is a detail of how a figure is
# drawn, not a promise about what a caller may draw -- and the displays themselves are now
# reached the same way: ``from behavio.plot import plot_pareto_k``.
#
# The detection-theory and psychometric records: ``PsychometricFitResult``,
# ``PsychometricParameters``, ``PsychometricSummary``, ``SignalDetectionSummary``,
# ``CorrectedRates`` and ``ZRocSummary``. (``SignalDetectionFitResult``,
# ``UnequalVarianceFitResult`` and ``MetaSDTFitResult`` were deleted once a fit could carry
# ``derived`` quantities directly; the two surviving records earn their place by holding
# per-fit multistart evidence rather than typed readers over ``derived``.)
# These follow the principle already applied to
# ``DeclarationCheck`` and its neighbours: a type the user reads back off a call stays free to
# gain fields. ``DetectionCounts``, ``RatingCounts``, ``RateCorrection`` and
# ``PsychometricLink`` are arguments the user does write down, but each is reachable only
# through a pinned call that already names it in its own signature, so pinning them separately
# would add nothing the model catalogue does not already promise.

# The task ontology -- ``TaskFamily``, ``TaskProtocol``, ``CanonicalTrial``,
# ``CanonicalVariable``, ``ChoiceDeclaration``, the twelve vocabulary enums,
# ``CONTROLLED_VOCABULARIES``, the three JSON-Schema emitters, and
# ``study_from_canonical_trials`` -- is deliberately absent from every set below, and its
# absence is the promise rather than an oversight.
#
# The rule this file is drawn with pins what a user writes down *on the shortest correct
# path* from a table of trials to a validated result. The ontology is not on that path in
# either direction. Someone fitting a GLM to their own CSV writes
# ``TaskSpec(choice=ChoiceSpec(options=(0, 1)))`` and never names a family; that call is
# pinned in ``GOLDEN_PATH`` and is unchanged by everything the ontology adds. Someone who
# *does* need a family needs it because they are reconciling several separately curated
# sources, which is a deliberate, one-off act of declaration rather than a step in an
# analysis, and they reach it at ``behavio.task.<name>`` and
# ``behavio.adapters.study_from_canonical_trials``.
#
# Pinning it would also assert that the vocabulary is finished. It is not: nine of the
# eighteen curated protocols the enums were derived from cannot yet name what their
# alternatives mean, and a wagering task has no term for its opt-out. A closed set that is
# still growing does not belong in a namespace whose whole argument is that it is small and
# stable.
#
# Two names *were* added to already-pinned types and are covered by the existing sets:
# ``ResponseTimeSpec.origin`` and ``RewardSpec.units``. Both are optional members of
# constructors ``GOLDEN_PATH`` already promises, so the surface did not widen.

INTEROPERABILITY_AND_EVIDENCE = {
    "ParameterSpace",
    "ScipyMultistart",
    "PyBADSMultistart",
    "PosteriorResult",
    "PyMCBinaryQLearning",
    "PyMCHierarchicalGLMBackend",
    "audit_posterior",
    "posterior_predictive_check",
    "psis_loo",
    "run_simulation_based_calibration",
    "run_sensitivity_analysis",
    "assess_test_retest_reliability",
    "generate_bounded_report",
    "write_evidence_bundle",
}

#: The posterior extension surface a user writes code against: the protocol a third-party
#: estimator implements, the predicate that decides whether an estimator satisfies it, the
#: projections that read one, the policy that governs how its folds are scored, the blocked
#: model-comparison entry point, and the report types those calls hand back. The
#: auditor-registry inversion (``FitAuditor``, ``fit_auditor``, ``register_fit_auditor``) is
#: deliberately absent: it is the mechanism by which ``behavio.diagnostics`` registers itself
#: with the leaf contracts package, not a promise that users should build on, and pinning it
#: now would freeze an internal wiring decision.
POSTERIOR_EXTENSION_SURFACE = {
    "PosteriorBehaviourEstimator",
    "is_posterior_estimator",
    "posterior_point_summary",
    "posterior_model_capabilities",
    "PosteriorFoldPolicy",
    "compare_posterior_models",
    "PosteriorModelComparison",
    "SBCUniformity",
    "PosteriorEvidence",
}

#: How trials get into Behavio and how a protocol gets checked before it runs: the shortest
#: path from a file to a ``Study``, the contract a third-party data source implements, the
#: observation typing a protocol declares, and the two pre-run checks a protocol author calls.
#:
#: Three names here are pinned because a pinned neighbour is unusable without them, not
#: because they are interesting on their own. ``StudyAdapter`` annotates two of its five
#: protocol members with ``SourceType`` and ``SessionOrderPolicy``, so an implementer must be
#: able to import both; ``read_tables`` takes ``Iterable[TableSource]`` and has no path-only
#: overload, so its argument type is part of the call. A promise the user cannot fulfil from
#: the top level is not a promise.
#:
#: Where the line is drawn, and why these and not their neighbours: pin what a user writes
#: down, not what runs underneath it. Deliberately unpinned, and reached through their area:
#: the ``Any*`` structural aliases and ``posterior_draw_matrix`` /
#: ``posterior_summary_message`` / ``posterior_log_predictive_density`` /
#: ``posterior_parameter_columns`` (projections the library uses to talk to an estimator it
#: did not write -- internal mechanism, reached through ``behavio.contracts``);
#: ``DeclarationCheck``, ``DeclarationFinding``, ``CandidateVerification``,
#: ``ObservationContractRule`` and ``ObservationContractViolation`` (the shape of a report you
#: read back, not one you construct, and report shapes should stay free to gain fields); the
#: conformance harness (``check_study_adapter``, ``assert_study_adapter_conforms``,
#: ``AdapterConformance``, ``ConformanceCheck``, ``CheckStatus``) -- a test-time aid, secondary
#: to the ``StudyAdapter`` contract it checks; ``AdapterCapabilities`` and
#: ``adapter_capabilities`` (registry-side validation, not needed to implement an adapter);
#: the ``WALD_INTERVAL`` / ``POSTERIOR_QUANTILE_INTERVAL`` interval-kind constants; and every
#: optional refinement of ``TableSource`` (``TableFormat``, ``ColumnType``,
#: ``SessionOrderDerivation``, ``SessionOrderRule``, ``TableReadError``,
#: ``DEFAULT_MISSING_VALUES``, the ``session_order_from_*`` helpers), since
#: ``TableSource(path=...)`` alone is a working call and the rest are opt-in. Everything
#: unpinned here is reached at ``behavio.adapters`` or ``behavio.contracts``.
#:
#: Over-pinning is cheap now and expensive to undo. A name earns a place here by being
#: unavoidable on the shortest correct path, not by being useful.
DATA_SOURCE_AND_VERIFICATION_SURFACE = {
    "read_table",
    "read_tables",
    "TableSource",
    "StudyAdapter",
    "SourceType",
    "SessionOrderPolicy",
    "ObservationDataType",
    "validate_observation_contract",
    "verify_candidate_declarations",
}

#: One fold loop, one paired contrast, one bootstrap, one simultaneous family. These names
#: are pinned because the collapse of the two execution stacks is only real if the shared
#: pieces are the public ones: a caller who wants the protocol runner's semantics from the
#: interactive path, or vice versa, must be able to name the option that differs
#: (``FoldFailurePolicy``) and the record that makes a ranking readable
#: (``ComparisonFamily``) without reaching into a private module. ``PairedComparison`` is
#: pinned because both reports carry it and ``decisive`` is the member a winner rule reads.
#:
#: ``ComparisonFamily`` and ``ComparisonMultiplicity`` are now *defined* in
#: ``behavio._internal.scoring`` beside the step-up they describe, so that
#: ``behavio.protocol.schema`` can freeze the adjustment and ``behavio.posterior.comparison`` can
#: size an ELPD family without either importing the estimator stack. That is exactly why
#: the names are pinned here: three callers share two types, and the public address of both
#: has to stay ``behavio``.
SHARED_EXECUTION_SURFACE = {
    "evaluate_splits",
    "FoldEvaluation",
    "FoldFailure",
    "FoldFailurePolicy",
    "FoldStage",
    "SplitEvaluation",
    "PairedComparison",
    "ComparisonFamily",
    "ComparisonMultiplicity",
    "bootstrap_interval",
    # ``compare_models(metrics=...)`` cannot be written without naming the rules, and the
    # declaration decides both what the table carries and which rule ranks it. The refusals
    # it can raise -- ``UnscoreableByBrier``, ``UndeclaredMetric`` -- stay unpinned at
    # ``behavio.compare``: their bases, ``ValueError`` and ``LookupError``, are what a caller
    # catches, and ``DEFAULT_COMPARISON_METRICS`` names a default that applies without being
    # named.
    "ScoreMetric",
}

#: The bridge from continuously observed behaviour to trial-level ``Study`` columns. This is a
#: headline capability with a small closed API: declare when the trials happened
#: (``TrialTiming``), declare the window to summarise (``TrialWindow``), reduce a covariate or an
#: ethogram over it, and join the result onto a study. Those five names are the whole call path,
#: and every one of them is something the user writes down, so the same rule that pinned
#: ``read_table`` / ``TableSource`` applies here.
#:
#: ``TrialCoverageStatus`` was considered and deliberately left unpinned, which is where this
#: differs from ``SourceType`` and ``SessionOrderPolicy``. Those two are pinned because
#: ``StudyAdapter`` *annotates* protocol members with them: an implementer cannot write the
#: contract down without importing them. Nothing here forces that. The enum appears in no pinned
#: signature; it is a ``StrEnum``, so ``reduction.status[i] == "ok"`` is a correct comparison
#: rather than a workaround, and ``attach_trial_columns`` coerces it with ``str()`` on the way
#: into the study, so the column a user actually reads back holds plain strings. A name earns a
#: pin by being unavoidable, and this one is avoidable.
#:
#: Also unpinned, and reached at ``behavio.observed``: the reducers (``MeanValue``, ``MedianValue``,
#: ``MinimumValue``, ``MaximumValue``, ``FractionOfTimeInState``, ``EventCount``,
#: ``FirstOccurrenceLatency``) and the two reducer protocols, because the set is explicitly
#: designed to be open and will grow -- pinning today's members would imply the list is the
#: promise, when the protocol is; ``TrialReduction``, a report you read back rather than
#: construct, which should stay free to gain fields; ``TrializationError``, whose base
#: ``ValueError`` is what a caller catches; and ``trial_timing_from_events``, one optional way to
#: build a ``TrialTiming`` that the pinned constructor does not depend on.
OBSERVED_BEHAVIOUR_SURFACE = {
    "TrialTiming",
    "TrialWindow",
    "reduce_covariate_to_trials",
    "reduce_annotations_to_trials",
    "attach_trial_columns",
}

#: Two contracts promoted out of ``behavio.contracts`` that a user writes code against.
#: ``NaturalParameterisation`` is an extension surface in the same sense as
#: ``PosteriorBehaviourEstimator``: a third party implements it to declare that a result is
#: reported in a coordinate other than the one it was estimated in, and an extension point nobody
#: can rely on is not an extension point. ``DerivedQuantity`` is read off every fit and is the
#: mechanism by which derived scientific quantities became first-class rather than an ad-hoc
#: dictionary, so its presence -- not its field list -- is the promise.
#:
#: The rest of that widening is deliberately unpinned. ``MultistartFit`` and ``LatentStateFit``
#: were promoted mainly so ``behavio.diagnostics`` could stop duck-typing
#: against private names; they describe evidence the library reads out of a fit it did not write,
#: which is internal mechanism by the same argument that keeps ``FitAuditor`` and
#: ``posterior_draw_matrix`` out. ``natural_quantities`` and ``natural_covariance`` are helpers
#: for the model author implementing ``NaturalParameterisation``, reachable through
#: ``behavio.contracts`` alongside ``model_task_columns`` and ``validate_required_task_columns``,
#: which are not re-exported at the top level at all.
MODEL_EXTENSION_SURFACE = {
    "NaturalParameterisation",
    "DerivedQuantity",
}

#: What a caller needs to *use* opt-in parallelism and to *handle* it going wrong. The
#: ``workers`` and ``backend`` arguments themselves are pinned by the functions that take
#: them -- ``run_model_recovery``, ``compare_models``, ``nested_select_model`` and
#: ``run_simulation_based_calibration`` are all pinned above -- so what remains is the
#: vocabulary those arguments are written in.
#:
#: ``WorkerBackend`` is pinned even though ``backend="thread"`` is a correct call, for the
#: same reason ``SourceType`` is: it is a ``StrEnum``, so the string works, but a caller
#: selecting a backend from configuration wants to validate it against the closed set rather
#: than discover a typo at dispatch. ``UnpicklableTaskError`` and ``ParallelWorkerError`` are
#: pinned because they are meant to be caught: the first is the routine outcome of passing a
#: lambda splitter and has a documented fix, and the second is the only thing that stands
#: between a caller and a bare ``BrokenProcessPool``. An exception a user cannot name is an
#: exception they cannot handle.
#:
#: ``map_ordered`` and ``resolve_workers`` are deliberately unpinned and stay in
#: ``behavio._internal.parallel``. They are the scheduler the library runs its own loops
#: through, not a general-purpose parallel map offered to users; pinning them would promise
#: that Behavio's determinism guarantee extends to arbitrary user callables, which is
#: precisely the thing it cannot promise, because the guarantee rests on the caller's task
#: being pure and position-seeded.
PARALLELISM_SURFACE = {
    "WorkerBackend",
    "UnpicklableTaskError",
    "ParallelWorkerError",
}

#: Four names that no set above promises and that the shortest correct first analysis still
#: cannot be written without. This is the *second* clause of the rule, and it is deliberately
#: small: each entry names something a user types before they have any result to read.
#:
#: ``forward_session_splits`` is the plain single-cohort forecast split. Its multi-cohort
#: sibling ``cohort_forward_session_splits`` is already in ``GOLDEN_PATH``, and the two are
#: chosen between on one line of the first analysis; promising only the harder one would be
#: an accident of which was written first.
#:
#: ``audit_fit`` is the frequentist twin of ``audit_posterior``, which
#: ``INTEROPERABILITY_AND_EVIDENCE`` already pins. "Keep optimization failures and boundary
#: estimates visible" is a stated scientific requirement of this package, and the call that
#: does it for a maximum-likelihood fit cannot be one level less reachable than the call that
#: does it for a sampled one.
#:
#: ``StudyValidationError`` and ``TaskValidationError`` are the two exceptions raised by the
#: first two constructors a user writes, before any model exists. The same argument that pins
#: ``ParallelWorkerError`` and ``UnpicklableTaskError`` -- an exception a caller cannot name
#: is an exception they cannot handle -- reaches further here, not less far.
#:
#: Deliberately *not* here, and reached through their area instead: the whole of
#: ``behavio.plot`` (a display is not on the path to a result); ``DesignSpec`` and the design
#: terms (``behavio.design``); ``ClockSpec`` and the landmark clocks (``behavio.time``);
#: ``model_capabilities`` and the fit records (``behavio.models``); the remaining splitters
#: (``behavio.evaluate``); every ``*Policy`` and every report type (their area); and the NWB,
#: DANDI and IBL readers (``behavio.adapters``), since ``read_table`` is the shortest path
#: from a file to a ``Study`` and the rest are opt-in.
FIRST_ANALYSIS_SURFACE = {
    "forward_session_splits",
    "audit_fit",
    "StudyValidationError",
    "TaskValidationError",
}


PROMISED = (
    GOLDEN_PATH
    | MODEL_CATALOGUE
    | COMBINATOR_SURFACE
    | CLOSED_FORM_SUMMARIES
    | INTEROPERABILITY_AND_EVIDENCE
    | POSTERIOR_EXTENSION_SURFACE
    | DATA_SOURCE_AND_VERIFICATION_SURFACE
    | OBSERVED_BEHAVIOUR_SURFACE
    | MODEL_EXTENSION_SURFACE
    | SHARED_EXECUTION_SURFACE
    | PARALLELISM_SURFACE
    | FIRST_ANALYSIS_SURFACE
)


def test_completed_release_surface_is_public_and_explicit() -> None:
    assert set(behavio.__all__) >= PROMISED
    assert all(hasattr(behavio, name) for name in PROMISED)


def test_the_top_level_namespace_is_exactly_the_promised_surface() -> None:
    """``__all__`` is the golden path, not a re-export of the package.

    Containment is not enough. A name that reaches the top level without an argument written
    in one of the sets above is exactly the drift that produced a 530-name namespace, and a
    name quietly dropped from a promised set is a break nobody declared. Both fail here.
    """

    assert set(behavio.__all__) == PROMISED | {"__version__"}


def test_nothing_beyond_the_golden_path_is_reachable_from_the_top_level() -> None:
    """A culled name is gone from ``behavio``, not merely absent from ``__all__``.

    ``__all__`` governs ``import *`` and tooling; it does not stop ``behavio.SomeCulledName``
    from resolving if the root package still imports it. This release removed the aliases
    rather than hiding them, so the module's public attributes and its ``__all__`` agree.
    """

    public = {
        name
        for name in vars(behavio)
        if not name.startswith("_") and not isinstance(vars(behavio)[name], ModuleType)
    }

    assert public == PROMISED


def test_release_orientation_documents_are_in_the_strict_site_navigation() -> None:
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    promised_pages = (
        "getting-started/index.md",
        "getting-started/installation.md",
        "getting-started/first-analysis.md",
        "model-choice-guide.md",
        "model-cards.md",
        "tutorials/recipe-contract.md",
        "tutorials/cell2025-figure1gi-reproduction.md",
        "tutorials/cell2025-figure1hj-reproduction.md",
        "tutorials/chen2021-bandit.md",
        "migration-guides.md",
        "extensions.md",
        "reference/validation-and-comparison.md",
        "reference/figure-standard.md",
        "parallelism.md",
        "roadmap.md",
    )

    for page in promised_pages:
        assert page in navigation
