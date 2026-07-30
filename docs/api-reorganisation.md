# API reorganisation

This release is the single planned API break. Forty-odd modules in one flat namespace became
a small set of named areas, seven overloaded words were given one meaning each, and
`behavio.__all__` narrowed from a re-export of everything to a curated golden path.

There are **no back-compatibility aliases and no deprecation shims**. A name that moved is
gone from its old address. Every rename below is mechanical: search, replace, re-run.

## The new tree

| Area | What lives there |
| --- | --- |
| `behavio` | The golden path: the names a first analysis cannot be written without. |
| `behavio.trials` | `Study`, the trial-level data contract, and its required columns. |
| `behavio.observed` | Continuously observed behaviour in seconds — pose, ethograms, covariates, interval policies, device clocks — and the trialization bridge onto a `Study`. |
| `behavio.time` | Learning-time clocks and the fold-fitted landmark clocks built on them. |
| `behavio.task` | The behavioural task contract: what a trial means, and its response times. |
| `behavio.design` | Fixed design-matrix terms and the formula notation that desugars onto them. |
| `behavio.inference` | Parameter spaces, priors, and the optimizer backends. |
| `behavio.models` | The model catalogue and the kernels underneath it. |
| `behavio.compose` | The `smooth` / `hierarchical` / `mix` combinators. |
| `behavio.posterior` | Labelled posterior draws and every check that reads them. |
| `behavio.evaluate` | Validation splits and the fold-evaluation loop that runs over them. |
| `behavio.compare` | Prospective model comparison and parameter-trajectory comparison. |
| `behavio.recovery.parameters` | Parameter recovery and model recovery. |
| `behavio.protocol.schema` | Protocol schema, compiler, runner, and exact-design recovery. |
| `behavio.report` | Bounded reports, evidence bundles, and fit artifacts. |
| `behavio.adapters` | Data-source adapters (tables, IBL ONE, NWB, DANDI). |
| `behavio.contracts` | The extension protocols a third party implements. |
| `behavio.plot` | Figure displays. |

## `behavio.observed` — observed behaviour

The five continuous-signal modules and the trialization bridge were closed under imports
already; they are now a package that says so.

| Old | New |
| --- | --- |
| `behavio.pose` | `behavio.observed.pose` |
| `behavio.ethograms` | `behavio.observed.ethograms` |
| `behavio.predictors` | `behavio.observed.covariates` |
| `behavio.interval_policy` | `behavio.observed.interval_policy` |
| `behavio.trialization` | `behavio.observed.trialization` |
| `behavio.sync` | `behavio.observed.device_clocks` |
| `behavio._arrays` | `behavio.observed._arrays` (private) |

`sync.py` opened by disclaiming its sibling: *"This is not `behavio.time.clocks`."* A module that
has to introduce itself by denying another module is misnamed. It synchronises **physical
device clocks measured in seconds**, so it is now `device_clocks` and every name it exports
carries the prefix:

| Old | New |
| --- | --- |
| `ClockPulseMatches` | `DeviceClockPulses` |
| `ClockSynchronization` | `DeviceClockSync` |
| `ClockSynchronizationSpec` | `DeviceClockSyncSpec` |
| `fit_clock_synchronization` | `fit_device_clock_sync` |

Unqualified "clock" now means one thing in Behavio: the learning-time coordinate a trial
sits on, in `behavio.time.clocks`. That is the package's primary sense — it is the axis every
longitudinal claim is made against — so it keeps the bare word, and the physical clocks are
the ones that carry a qualifier.

## `behavio.posterior` — posterior draws and their checks

Eight modules sharing a prefix became a package. `behavio.posterior.result` is
dependency-free, which is what lets `behavio.contracts` name `PosteriorResult` without a
cycle; everything else in the package consumes a result and returns evidence about it.

| Old | New |
| --- | --- |
| `behavio.posterior` | `behavio.posterior.result` |
| `behavio.posterior_diagnostics` | `behavio.posterior.diagnostics` |
| `behavio.posterior_predictive` | `behavio.posterior.predictive` |
| `behavio.posterior_loo` | `behavio.posterior.loo` |
| `behavio.posterior_comparison` | `behavio.posterior.comparison` |
| `behavio.sensitivity` | `behavio.posterior.sensitivity` |
| `behavio.reliability` | `behavio.posterior.reliability` |
| `behavio.sbc` | `behavio.posterior.simulation_based_calibration` |
| `behavio.plot.sbc` | `behavio.plot.simulation_based_calibration` |

`sbc.py` was the only filename in the package that hid behind an unexpanded acronym, while
the function it exists for is spelled `run_simulation_based_calibration` and the
documentation page is spelled `simulation-based-calibration.md`. The class names keep the
`SBC` prefix: an acronym in a type name reads fine once the module has spelled it out.

No symbol in this area was renamed. The module paths were doing all the ambiguity.

## `behavio.trials` — the trial table

| Old | New |
| --- | --- |
| `behavio.study` | `behavio.trials` |

`Study` itself did not move and did not change its name. The defect the audit named was the
*module*: a reader looking for "the study" found a table, because in Behavio the study — the
pre-registered scientific object — is `StudyProtocol`, in `behavio.protocol.schema`. The
module now says what it holds. `Study` keeps its name because it is the data of a
longitudinal study and because it is the single most-used name in the package; the pair
`Study` (the trials) and `StudyProtocol` (the plan) is legible once the module paths stop
disagreeing with it.

One `Study`-prefixed name did move, because it was not a study at all:

| Old | New | Why |
| --- | --- | --- |
| `MaterializedStudy` | `MaterializedProtocol` | it holds a protocol, a study and a manifest, and is what `materialize_protocol()` returns |

`ClockedStudy`, `StudyTransform` and `StudyAdapter` keep their names: each really is *of a
`Study`*, and each already carries the qualifier the rule asks for.

## `behavio.task`, `behavio.design`, `behavio.time`, `behavio.inference`

| Old | New |
| --- | --- |
| `behavio.task` | `behavio.task.spec` |
| `behavio.response_times` | `behavio.task.response_times` |
| `behavio.design` | `behavio.design.matrix` |
| `behavio.formula` | `behavio.design.formula` |
| `behavio.clocks` | `behavio.time.clocks` |
| `behavio.transforms` (landmark clocks) | `behavio.time.landmarks` |
| `behavio.transforms` (`StudyTransform`, `fit_transform_split`) | `behavio.time.transforms` |
| `behavio.parameters` | `behavio.inference.parameters` |
| `behavio.inference` | `behavio.inference.optimize` |
| `behavio.state_alignment` | `behavio.models.state_alignment` |

Two of these are more than a move.

`transforms.py` was split. It held the *landmark clocks* — threshold detection, bootstrap
uncertainty, the fitted clock — under a name that said only "transforms", plus the generic
fold-fitting driver that runs any `StudyTransform` over a split. Those are two different
things and now live in two modules.

`behavio.task.spec` is now typed against `behavio.contracts.estimator` rather than
`behavio.models.base`. It imported the identical objects either way, but the old spelling
made the task layer depend on the model catalogue, which put `TaskSpec` *above* the models
that declare which task columns they need. The task contract is now below them.

`state_alignment` moved into `behavio.models` because its only in-package consumer is the
GLM-HMM estimator, which uses it to canonicalise state order. Grouping it with the recovery
modules — its conceptual neighbours — would have made `behavio.models` and
`behavio.recovery` import each other.

## `behavio.evaluate`, `behavio.compare`, `behavio.recovery`

| Old | New |
| --- | --- |
| `behavio.validation` | `behavio.evaluate.splits` |
| `behavio.evaluation` | `behavio.evaluate.folds` |
| `behavio.comparison` | `behavio.compare.models` |
| `behavio.trajectory_shapes` | `behavio.compare.parameter_trajectories` |
| `behavio.recovery` | `behavio.recovery.parameters` |
| `behavio.model_recovery` | `behavio.recovery.models` |

### What "validation" means now

`validation.py` contained only splitters, which is why the audit called this the worst of
the seven: the module named for the concept did not implement the concept. The rule
adopted here is:

> **"validation" means checking an input against a declared contract.** The mechanics of
> holding data out are called *splits*, *folds* and *evaluation*, never validation.

So the splitter vocabulary lost the word:

| Old | New |
| --- | --- |
| `ValidationSplit` | `Split` |
| `PopulationValidationSplit` | `PopulationSplit` |
| `CohortValidationSplit` | `CohortSplit` |
| `ValidationFold` (`behavio.contracts.fold`) | `EvaluationFold` |

and the contract-checking vocabulary kept it unchanged: `StudyValidationError`,
`TaskValidationError`, `ClockValidationError`, `DesignValidationError`,
`ProtocolValidationError` and `TaskValidation` all mean "this input does not satisfy a
declared contract", which is what the word means in Python everywhere else.

`ValidationSpec` and `ValidationGeometry` in `behavio.protocol.schema` are the deliberate
exception, and the reason is not taste. They are not splitter mechanics: they are a
protocol's frozen *declaration* that the study will be validated in a stated geometry —
the scientific sense of the word. They are also serialised field names in
the published, content-addressed `behavio.study-protocol` schema. Renaming them would
require a schema version bump plus a second reader for every protocol ever recorded, which
is more legacy machinery than the clarity would buy.

### "recovery" and "trajectory"

`recovery.py` was parameter recovery and its name said only "recovery". The path now says
which recovery: `behavio.recovery.parameters` and `behavio.recovery.models`, with
`behavio.protocol.exact_recovery` for the third one that runs through a compiled plan.

`trajectory_shapes` compared the shape a *fitted parameter* traces across sessions, while
`PoseTrajectory` is an animal moving. The qualifier went on both sides, so the module is
`parameter_trajectories` and:

| Old | New |
| --- | --- |
| `TrajectoryPanel` | `ParameterTrajectoryPanel` |
| `GroupTrajectorySummary` | `GroupParameterTrajectorySummary` |

`CoefficientTrajectory`, `ValueTrajectory` and `RLTrajectory` were already qualified and
did not move.

## `behavio.protocol` and `behavio.report`

| Old | New |
| --- | --- |
| `behavio.protocol` | `behavio.protocol.schema` |
| `behavio.compiler` | `behavio.protocol.compiler` |
| `behavio.runner` | `behavio.protocol.runner` |
| `behavio.protocol_recovery` | `behavio.protocol.exact_recovery` |
| `behavio.reporting` | `behavio.report.bounded` |
| `behavio.evidence` | `behavio.report.evidence_bundles` |
| `behavio.interchange` | `behavio.report.fit_artifacts` |

`protocol_recovery` exported `run_exact_recovery`, `ExactRecoveryReport` and
`ExactRecoveryRun`; the module now has the name its contents already had.
`interchange.py` suggested interoperability between tools and has always been
fitted-artefact serialisation. `evidence.py` suggested statistical evidence and has always
been ZIP bundles.

### `SCHEMA_VERSION`

| Old | New |
| --- | --- |
| `behavio.SCHEMA_VERSION` | `behavio.protocol.PROTOCOL_SCHEMA_VERSION` |
| `SUPERSEDED_SCHEMA_VERSIONS` | `SUPERSEDED_PROTOCOL_SCHEMA_VERSIONS` |
| `ACCEPTED_SCHEMA_VERSIONS` | `ACCEPTED_PROTOCOL_SCHEMA_VERSIONS` |

It was exported bare from the root while its three siblings —
`BUNDLE_SCHEMA_VERSION`, `FIT_ARTIFACT_SCHEMA` and `PARAMETER_SPACE_SCHEMA` — were all
prefixed. The **value** of the constant was unchanged by the rename: the
`behavio.study-protocol/N` string is a published wire identifier, and renaming the Python
name it is bound to must not restamp recorded protocols.

## `covariates` became `predictors`

Behavio had four names for one idea — a study column used as a predictor — and one of them
collided with a genuinely different thing.

| Old | New |
| --- | --- |
| `BernoulliHistoryGLM(covariates=...)` | `BernoulliHistoryGLM(predictors=...)` |
| `WienerDriftDiffusion(covariates=...)` | `WienerDriftDiffusion(predictors=...)` |
| `BernoulliGLMHMM(covariates=...)` | `BernoulliGLMHMM(predictors=...)` |
| `model.covariates` | `model.predictors` |

This unifies with `TaskSpec.predictors` and `ObservationRole.PREDICTOR`, which already used
the word, and it leaves exactly one meaning of "covariate" in the package:
`BehaviorCovariate`, a named scalar observed over time. The qualifier did not have to be
added to anything, because a concept was removed instead.

Two consequences a reader should expect. A model's `signature` now reads
`...[outcome=choice;predictors=stimulus;...]`, so any frozen signature string, recorded
protocol payload or content address that quoted the old spelling changes. And a
`CandidateSpec` declaring `Setting("covariates", ...)` must be rewritten to
`Setting("predictors", ...)`, or candidate verification will report a contradiction —
which is the check working.

## The top-level namespace shrank from ~530 names to 103

`behavio.__all__` used to re-export the entire package. It is now a curated golden path, and
the rule that decides membership is written down in `tests/test_release_contract.py` beside
each set:

> **Pin what a user writes down on the shortest correct path from a table of trials to a
> validated, compared and audited result.**

A type you read back off a call, an optional refinement of a pinned argument, a display, or a
mechanism the library uses to talk to code it did not write does not qualify. Those are
public, documented and supported — at `behavio.<area>.<name>`.

Two tests hold the line. One asserts `set(behavio.__all__)` equals the union of the declared
sets exactly, so a name cannot arrive at the top level without an argument written for it, and
cannot silently leave. The other asserts that the module's public attributes equal that same
set, because this release **removed** the culled imports rather than hiding them behind
`__all__`: `behavio.PosteriorAuditPolicy` raises `AttributeError`, it does not quietly work.

### Where the culled names went

| If you imported | Import it from |
| --- | --- |
| `DesignSpec`, `NumericTerm`, `CategoricalTerm`, `HistoryTerm`, `HistoryKernelTerm`, `InteractionTerm`, `StandardizeTerm`, `DesignMatrix`, `FeatureBlock` | `behavio.design` |
| `ClockSpec`, `ClockKind`, `ClockScope`, `ClockedStudy`, `ThresholdLandmarkClock`, `BootstrapThresholdLandmarkClock`, `StudyTransform`, `fit_transform_split(s)`, `with_elapsed_time_clock`, `with_cumulative_trial_clock`, `session_order_clock` | `behavio.time` |
| `PoseTrajectory`, `BehaviorCovariate`, `BehaviorAnnotations`, `IntervalPolicy`, the reducers, the DeepLabCut/SLEAP/MoSeq/BORIS readers, `DeviceClockSync` | `behavio.observed` |
| `FitResult`, `ModelCapabilities`, `model_capabilities`, `DetectionCounts`, `RatingCounts`, `PsychometricLink`, `SoftmaxPolicy`, `ChoiceKernel`, `align_latent_states` | `behavio.models` |
| `CoefficientTrajectory`, `HierarchicalSimulation`, `MixtureSimulation`, `UnseenGroupPrediction` | `behavio.compose` |
| `ParameterSpec`, `ParameterRole`, `ParameterTransform`, `PriorSpec`, `PriorFamily`, `OptimizationProblem`, `ObjectiveTarget`, `PriorMeasure` | `behavio.inference` |
| `PosteriorGroup`, `PosteriorVariable`, `PosteriorAuditPolicy`, `PosteriorPredictivePolicy`, `SBCSimulation`, `SensitivityScenario`, `ReliabilityPolicy`, `SubjectEstimates`, `PSISLOOResult` | `behavio.posterior` |
| `leave_one_subject_out_splits`, `leave_one_lab_out_splits`, `leave_one_session_out_splits`, `within_session_rolling_splits`, `historical_cohort_forecast_splits`, `Split`, `PopulationSplit`, `CohortSplit` | `behavio.evaluate` |
| `ProspectiveComparisonReport`, `NestedProspectiveSelectionReport`, `paired_comparisons`, `compare_trajectory_shapes`, `ParameterTrajectoryPanel` | `behavio.compare` |
| `ModelRecoveryScenario`, `ParameterRecoveryReport`, `run_model_recovery_grid`, `WALD_INTERVAL`, `POSTERIOR_QUANTILE_INTERVAL` | `behavio.recovery` |
| `CandidateSpec`, `Setting`, `ValidationSpec`, `ValidationGeometry`, `materialize_protocol`, `protocol_from_json`, `run_nested_protocol`, `run_exact_recovery`, `ProtocolRun` | `behavio.protocol` |
| `EvidenceBundle`, `read_evidence_bundle`, `replay_evidence_bundle`, `compare_evidence_bundles`, `FitArtifact`, `fit_artifact_from_json`, `BoundedReport` | `behavio.report` |
| `read_nwb`, `write_nwb`, `study_from_dandi`, `study_from_ibl_one`, `NWBSessionSource`, `DANDINWBSource`, `IBLONETrialSource`, `check_study_adapter` | `behavio.adapters` |
| every `plot_*`, `figure_style`, `configure_figure_style`, `save_svg`, `MatplotlibUnavailableError` | `behavio.plot` |
| `FitAudit`, `FitAuditPolicy`, `FitAuditStatus`, `FitIssue`, `AuditSeverity`, `RestartAudit`, `ConvergenceStatus` | `behavio.diagnostics` |
| `BehaviourEstimator`, `GenerativeBehaviourModel`, `FitAuditor`, `PosteriorCentre`, `posterior_draw_matrix`, `AdapterCapabilities` | `behavio.contracts` |
| `REQUIRED_COLUMNS` | `behavio.trials` |

The full golden path is:

```text
BernoulliGLMHMM                     BernoulliHistoryGLM                 BiasOnly
BinaryQLearning                     BinaryRLAgent                       ChoiceSpec
ComparisonFamily                    ComparisonMultiplicity              DerivedQuantity
EqualVarianceSDT                    EstimatorRegistry                   FoldEvaluation
FoldFailure                         FoldFailurePolicy                   FoldStage
HierarchicalFitResult               HierarchicalModel                   MetaSDT
MixtureModel                        MultinomialLogit                    NaturalParameterisation
ObservationDataType                 PairedComparison                    ParallelWorkerError
ParameterSpace                      Perseveration                       PosteriorBehaviourEstimator
PosteriorEvidence                   PosteriorFoldPolicy                 PosteriorModelComparison
PosteriorResult                     Psychometric                        PsychometricFunction
PyBADSMultistart                    PyMCHierarchicalGLMBackend          ResponseTimeSpec
RewardSpec                          SBCUniformity                       ScipyMultistart
SessionOrderPolicy                  SmoothModel                         SourceType
SplitEvaluation                     Study                               StudyAdapter
StudyProtocol                       StudyValidationError                TableSource
TaskSpec                            TaskValidationError                 TrialTiming
TrialWindow                         UnequalVarianceSDT                  UniformCategoryGuess
UniformChoiceGuess                  UniformResponseGuess                UnpicklableTaskError
WienerDriftDiffusion                WinStayLoseShift                    WorkerBackend
assess_test_retest_reliability      attach_trial_columns                audit_fit
audit_posterior                     bootstrap_interval                  build_evidence_bundle
builtin_estimator_registry          cohort_forward_session_splits       compare_models
compare_posterior_models            compile_execution_plan              detection_rates
equal_variance_summary              evaluate_splits                     export_fit
fit_model                           forced_choice_d_prime               forward_session_splits
generate_bounded_report             hierarchical                        is_posterior_estimator
mix                                 model_from_formula                  nested_select_model
posterior_model_capabilities        posterior_point_summary             posterior_predictive_check
psis_loo                            read_table                          read_tables
reduce_annotations_to_trials        reduce_covariate_to_trials          roc_points
run_model_recovery                  run_parameter_recovery              run_protocol
run_sensitivity_analysis            run_simulation_based_calibration    smooth
validate_observation_contract       verify_candidate_declarations       write_evidence_bundle
z_roc_summary
```

### Layering is now a test

`tests/test_contracts.py` already built the real import graph by AST and topologically sorted
it. It now also declares the package's layers, asserts that every module belongs to exactly
one of them and imports only at or below its own, and asserts that no two *areas* import each
other. A package may span layers — `behavio.posterior.result` sits below `behavio.contracts`
because the contracts name it, while its siblings sit above — but a module may not, and an
area may not depend on an area that depends on it.
