"""Validation-first tools for modelling behaviour across learning.

This namespace is the golden path and nothing else. A name is here if it is promised by
``tests/test_release_contract.py`` -- the curated release surface, with the argument for
each set written beside it -- or if the shortest correct first analysis cannot be written
without it. Everything else the package implements is public, documented and supported at
its own address:

``behavio.trials``, ``behavio.task``, ``behavio.design``, ``behavio.time``,
``behavio.observed``, ``behavio.models``, ``behavio.compose``, ``behavio.inference``,
``behavio.posterior``, ``behavio.evaluate``, ``behavio.compare``, ``behavio.recovery``,
``behavio.protocol``, ``behavio.report``, ``behavio.adapters``, ``behavio.contracts``,
``behavio.plot``, ``behavio.registry``, ``behavio.diagnostics``.

The rule that decides the boundary: pin what a user *writes down* on the path from a table
of trials to a validated, compared and audited result. A type read back off a call, an
optional refinement of a pinned argument, a display, or a mechanism the library uses to
talk to code it did not write all live in their area instead. Over-pinning is cheap now and
expensive to undo.
"""

from behavio._internal.parallel import (
    ParallelWorkerError,
    UnpicklableTaskError,
    WorkerBackend,
)
from behavio.adapters import (
    TableSource,
    read_table,
    read_tables,
)
from behavio.compare.models import (
    ComparisonFamily,
    ComparisonMultiplicity,
    PairedComparison,
    ScoreMetric,
    bootstrap_interval,
    compare_models,
    nested_select_model,
)
from behavio.compose import (
    HierarchicalFitResult,
    HierarchicalModel,
    MixtureModel,
    MixtureRowModel,
    SmoothModel,
    UniformCategoryGuess,
    UniformChoiceGuess,
    UniformResponseGuess,
    hierarchical,
    mix,
    model_from_formula,
    smooth,
)
from behavio.contracts import (
    DerivedQuantity,
    NaturalParameterisation,
    PosteriorBehaviourEstimator,
    SessionOrderPolicy,
    SourceType,
    StudyAdapter,
    is_posterior_estimator,
    posterior_model_capabilities,
    posterior_point_summary,
)
from behavio.diagnostics import audit_fit
from behavio.evaluate.folds import (
    FoldEvaluation,
    FoldFailure,
    FoldFailurePolicy,
    FoldStage,
    PosteriorFoldPolicy,
    SplitEvaluation,
    evaluate_splits,
)
from behavio.evaluate.splits import (
    cohort_forward_session_splits,
    forward_session_splits,
)
from behavio.inference.optimize import (
    PyBADSMultistart,
    ScipyMultistart,
)
from behavio.inference.parameters import ParameterSpace
from behavio.models import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    BiasOnly,
    BinaryQLearning,
    BinaryRLAgent,
    EqualVarianceSDT,
    HierarchicalSessionDynamicBernoulliGLMHMM,
    MetaSDT,
    MultinomialLogit,
    Perseveration,
    Psychometric,
    PsychometricFunction,
    SessionDynamicBernoulliGLMHMM,
    UnequalVarianceSDT,
    WienerDriftDiffusion,
    WinStayLoseShift,
    detection_rates,
    equal_variance_summary,
    forced_choice_d_prime,
    roc_points,
    z_roc_summary,
)
from behavio.observed.trialization import (
    TrialTiming,
    TrialWindow,
    attach_trial_columns,
    reduce_annotations_to_trials,
    reduce_covariate_to_trials,
)
from behavio.posterior.comparison import (
    PosteriorModelComparison,
    compare_posterior_models,
)
from behavio.posterior.diagnostics import audit_posterior
from behavio.posterior.loo import psis_loo
from behavio.posterior.predictive import posterior_predictive_check
from behavio.posterior.reliability import assess_test_retest_reliability
from behavio.posterior.result import PosteriorResult
from behavio.posterior.sensitivity import run_sensitivity_analysis
from behavio.posterior.simulation_based_calibration import (
    SBCUniformity,
    run_simulation_based_calibration,
)
from behavio.protocol.compiler import (
    compile_execution_plan,
    validate_observation_contract,
)
from behavio.protocol.runner import (
    run_protocol,
    verify_candidate_declarations,
)
from behavio.protocol.schema import (
    ObservationDataType,
    StudyProtocol,
)
from behavio.pymc_backend import PyMCBinaryQLearning, PyMCHierarchicalGLMBackend
from behavio.recovery.models import run_model_recovery
from behavio.recovery.parameters import run_parameter_recovery
from behavio.registry import (
    EstimatorRegistry,
    builtin_estimator_registry,
)
from behavio.report.bounded import generate_bounded_report
from behavio.report.evidence_bundles import (
    PosteriorEvidence,
    build_evidence_bundle,
    write_evidence_bundle,
)
from behavio.report.fit_artifacts import export_fit
from behavio.task.response_times import ResponseTimeSpec
from behavio.task.spec import (
    ChoiceSpec,
    RewardSpec,
    TaskSpec,
    TaskValidationError,
    fit_model,
)
from behavio.trials import (
    Study,
    StudyValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "BernoulliGLMHMM",
    "BernoulliHistoryGLM",
    "BiasOnly",
    "BinaryQLearning",
    "BinaryRLAgent",
    "ChoiceSpec",
    "ComparisonFamily",
    "ComparisonMultiplicity",
    "DerivedQuantity",
    "EqualVarianceSDT",
    "EstimatorRegistry",
    "FoldEvaluation",
    "FoldFailure",
    "FoldFailurePolicy",
    "FoldStage",
    "HierarchicalFitResult",
    "HierarchicalModel",
    "HierarchicalSessionDynamicBernoulliGLMHMM",
    "MetaSDT",
    "MixtureModel",
    "MixtureRowModel",
    "MultinomialLogit",
    "NaturalParameterisation",
    "ObservationDataType",
    "PairedComparison",
    "ParallelWorkerError",
    "ParameterSpace",
    "Perseveration",
    "PosteriorBehaviourEstimator",
    "PosteriorEvidence",
    "PosteriorFoldPolicy",
    "PosteriorModelComparison",
    "PosteriorResult",
    "Psychometric",
    "PsychometricFunction",
    "PyBADSMultistart",
    "PyMCBinaryQLearning",
    "PyMCHierarchicalGLMBackend",
    "ResponseTimeSpec",
    "RewardSpec",
    "SBCUniformity",
    "ScipyMultistart",
    "ScoreMetric",
    "SessionDynamicBernoulliGLMHMM",
    "SessionOrderPolicy",
    "SmoothModel",
    "SourceType",
    "SplitEvaluation",
    "Study",
    "StudyAdapter",
    "StudyProtocol",
    "StudyValidationError",
    "TableSource",
    "TaskSpec",
    "TaskValidationError",
    "TrialTiming",
    "TrialWindow",
    "UnequalVarianceSDT",
    "UniformCategoryGuess",
    "UniformChoiceGuess",
    "UniformResponseGuess",
    "UnpicklableTaskError",
    "WienerDriftDiffusion",
    "WinStayLoseShift",
    "WorkerBackend",
    "__version__",
    "assess_test_retest_reliability",
    "attach_trial_columns",
    "audit_fit",
    "audit_posterior",
    "bootstrap_interval",
    "build_evidence_bundle",
    "builtin_estimator_registry",
    "cohort_forward_session_splits",
    "compare_models",
    "compare_posterior_models",
    "compile_execution_plan",
    "detection_rates",
    "equal_variance_summary",
    "evaluate_splits",
    "export_fit",
    "fit_model",
    "forced_choice_d_prime",
    "forward_session_splits",
    "generate_bounded_report",
    "hierarchical",
    "is_posterior_estimator",
    "mix",
    "model_from_formula",
    "nested_select_model",
    "posterior_model_capabilities",
    "posterior_point_summary",
    "posterior_predictive_check",
    "psis_loo",
    "read_table",
    "read_tables",
    "reduce_annotations_to_trials",
    "reduce_covariate_to_trials",
    "roc_points",
    "run_model_recovery",
    "run_parameter_recovery",
    "run_protocol",
    "run_sensitivity_analysis",
    "run_simulation_based_calibration",
    "smooth",
    "validate_observation_contract",
    "verify_candidate_declarations",
    "write_evidence_bundle",
    "z_roc_summary",
]
