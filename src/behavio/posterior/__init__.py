"""Labelled posterior draws, and every check that reads them.

:mod:`behavio.posterior.result` is the currency: a backend-neutral, dimension-labelled
container of draws that any sampler can produce and that
:class:`~behavio.contracts.posterior.PosteriorBehaviourEstimator` promises to return. It
is deliberately dependency-free, which is what lets ``behavio.contracts`` name it.

Every other module here consumes one and returns evidence about it -- convergence
diagnostics, predictive checks, PSIS-LOO, blocked model comparison, simulation-based
calibration, prior/likelihood sensitivity, and test--retest reliability. None of them
knows how the draws were produced.

``result`` is imported first below so that the package initialises cleanly no matter
which of ``behavio.contracts`` or ``behavio.posterior`` a caller reaches for first.
"""

from behavio.posterior.result import (
    POSTERIOR_GROUPS,
    SAMPLE_DIMS,
    ArviZUnavailableError,
    PosteriorError,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    posterior_result_from_arviz,
)

from behavio.posterior.comparison import (  # isort: skip
    ModelComparisonIssue,
    ModelComparisonStatus,
    PairedELPDDifference,
    PosteriorModelComparison,
    ScoredModel,
    compare_posterior_models,
)
from behavio.posterior.diagnostics import (  # isort: skip
    PosteriorAudit,
    PosteriorAuditIssue,
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    PosteriorDiagnostic,
    audit_posterior,
)
from behavio.posterior.loo import PSISLOOIssue, PSISLOOResult, psis_loo  # isort: skip
from behavio.posterior.predictive import (  # isort: skip
    CategoryRateDiscrepancy,
    MeanDiscrepancy,
    PosteriorPredictiveAudit,
    PosteriorPredictiveCheck,
    PosteriorPredictiveIssue,
    PosteriorPredictivePolicy,
    PredictiveDiscrepancy,
    PredictiveFamily,
    PredictiveMultiplicity,
    PredictiveTail,
    SwitchRateDiscrepancy,
    VarianceDiscrepancy,
    posterior_predictive_check,
)
from behavio.posterior.reliability import (  # isort: skip
    ReliabilityError,
    ReliabilityEstimate,
    ReliabilityIssue,
    ReliabilityPolicy,
    ReliabilityStatistic,
    SubjectEstimates,
    SubjectPooling,
    TestRetestReliabilityReport,
    assess_test_retest_reliability,
    posterior_subject_estimates,
)
from behavio.posterior.sensitivity import (  # isort: skip
    SensitivityAnalysis,
    SensitivityContrast,
    SensitivityError,
    SensitivityFailure,
    SensitivityMetric,
    SensitivityOutcome,
    SensitivityReport,
    SensitivityRun,
    SensitivityScenario,
    SensitivitySummary,
    posterior_sensitivity_outcome,
    run_sensitivity_analysis,
)
from behavio.posterior.simulation_based_calibration import (  # isort: skip
    PosteriorParameterQuantity,
    SBCError,
    SBCFailure,
    SBCInference,
    SBCRank,
    SBCReport,
    SBCSimulation,
    SBCSimulator,
    SBCSummary,
    SBCTestQuantity,
    SBCUniformity,
    run_simulation_based_calibration,
)

__all__ = [
    "POSTERIOR_GROUPS",
    "SAMPLE_DIMS",
    "ArviZUnavailableError",
    "CategoryRateDiscrepancy",
    "MeanDiscrepancy",
    "ModelComparisonIssue",
    "ModelComparisonStatus",
    "PSISLOOIssue",
    "PSISLOOResult",
    "PairedELPDDifference",
    "PosteriorAudit",
    "PosteriorAuditIssue",
    "PosteriorAuditPolicy",
    "PosteriorAuditStatus",
    "PosteriorDiagnostic",
    "PosteriorError",
    "PosteriorGroup",
    "PosteriorModelComparison",
    "PosteriorParameterQuantity",
    "PosteriorPredictiveAudit",
    "PosteriorPredictiveCheck",
    "PosteriorPredictiveIssue",
    "PosteriorPredictivePolicy",
    "PosteriorResult",
    "PosteriorVariable",
    "PredictiveDiscrepancy",
    "PredictiveFamily",
    "PredictiveMultiplicity",
    "PredictiveTail",
    "ReliabilityError",
    "ReliabilityEstimate",
    "ReliabilityIssue",
    "ReliabilityPolicy",
    "ReliabilityStatistic",
    "SBCError",
    "SBCFailure",
    "SBCInference",
    "SBCRank",
    "SBCReport",
    "SBCSimulation",
    "SBCSimulator",
    "SBCSummary",
    "SBCTestQuantity",
    "SBCUniformity",
    "ScoredModel",
    "SensitivityAnalysis",
    "SensitivityContrast",
    "SensitivityError",
    "SensitivityFailure",
    "SensitivityMetric",
    "SensitivityOutcome",
    "SensitivityReport",
    "SensitivityRun",
    "SensitivityScenario",
    "SensitivitySummary",
    "SubjectEstimates",
    "SubjectPooling",
    "SwitchRateDiscrepancy",
    "TestRetestReliabilityReport",
    "VarianceDiscrepancy",
    "assess_test_retest_reliability",
    "audit_posterior",
    "compare_posterior_models",
    "posterior_predictive_check",
    "posterior_result_from_arviz",
    "posterior_sensitivity_outcome",
    "posterior_subject_estimates",
    "psis_loo",
    "run_sensitivity_analysis",
    "run_simulation_based_calibration",
]
