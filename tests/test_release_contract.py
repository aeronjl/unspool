"""Public-surface regression for the completed 0.21--0.24 package boundary."""

from __future__ import annotations

from pathlib import Path

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
#: ``HierarchicalModel`` and ``MixtureModel`` are pinned because they are what the verbs return
#: and therefore what a caller annotates; ``HierarchicalFitResult`` because a hierarchical fit's
#: group deviations are read off it by name and no other type carries them. The three
#: components are pinned because ``mix`` is useless without one and there is no way to write
#: your own without first seeing these.
#:
#: Deliberately unpinned, though all remain importable from ``behavio``:
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
#: both remain exported.
CLOSED_FORM_SUMMARIES = {
    "detection_rates",
    "equal_variance_summary",
    "forced_choice_d_prime",
    "roc_points",
    "z_roc_summary",
}

# Exported from the top level, documented, and deliberately not pinned by any set here.
#
# The whole of ``behavio.plot``: the twelve ``plot_*`` displays, ``figure_style``,
# ``configure_figure_style``, ``save_svg`` and ``MatplotlibUnavailableError``. Plotting is the
# newest surface in the package and the one most likely to move: every function reads a report
# object, so each field a report gains is a figure that should show it, and three follow-ups
# already want to change these signatures. Pinning the shape of a display now would freeze it
# against the reports it exists to present. The palette constants and ``FIGURE_RC_PARAMS`` are
# not re-exported at the top level at all -- a house style is a detail of how a figure is
# drawn, not a promise about what a caller may draw -- and stay in ``behavio.plot``.
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

INTEROPERABILITY_AND_EVIDENCE = {
    "ParameterSpace",
    "ScipyMultistart",
    "PyBADSMultistart",
    "PosteriorResult",
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
#: down, not what runs underneath it. Deliberately unpinned, though all remain importable:
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
#: ``TableSource(path=...)`` alone is a working call and the rest are opt-in.
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
#: ``behavio._internal.multiplicity`` beside the step-up they describe, so that
#: ``behavio.protocol`` can freeze the adjustment and ``behavio.posterior_comparison`` can
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
#: Also unpinned, though all remain exported: the reducers (``MeanValue``, ``MedianValue``,
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


def test_completed_release_surface_is_public_and_explicit() -> None:
    promised = (
        GOLDEN_PATH
        | MODEL_CATALOGUE
        | CLOSED_FORM_SUMMARIES
        | INTEROPERABILITY_AND_EVIDENCE
        | POSTERIOR_EXTENSION_SURFACE
        | DATA_SOURCE_AND_VERIFICATION_SURFACE
        | OBSERVED_BEHAVIOUR_SURFACE
        | MODEL_EXTENSION_SURFACE
        | SHARED_EXECUTION_SURFACE
        | PARALLELISM_SURFACE
    )

    assert promised <= set(behavio.__all__)
    assert all(hasattr(behavio, name) for name in promised)


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
