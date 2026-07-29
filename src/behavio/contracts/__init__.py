"""The single address for Behavio's extension surface.

``docs/extensions.md`` documents seven extension surfaces. Before this package they were
scattered across six modules, so "implement the Behavio contracts" had no import to point
at. Every protocol a downstream package implements, and every dataclass those protocols
declare structurally, now lives here:

===============================================  =====================================
Surface                                          Contract
===============================================  =====================================
A fitted predictive model                        :class:`BehaviourEstimator`
A model that can also simulate                   :class:`GenerativeBehaviourModel`
A sampler-backed model                           :class:`PosteriorBehaviourEstimator`
A parameter description                          :class:`ParameterSpaceProvider`
An optimizer                                     :class:`OptimizationBackend`
A behavioural summary                            :class:`PredictiveDiscrepancy`
A fold-fitted temporal transform                 :class:`StudyTransform`
A training/test partition                        :class:`ValidationFold`
A reporting coordinate distinct from the fit     :class:`NaturalParameterisation`
A retained multistart optimizer                  :class:`MultistartFit`
A retained latent-state fit                      :class:`LatentStateFit`
Predictive context a model reads                 :class:`TaskColumnEstimator`
===============================================  =====================================

The last four are *optional widenings*: a model that implements none of them is described
exactly as it was before they existed. :class:`NaturalParameterisation` names the
coordinate a result is reported in, which is not in general the coordinate it is estimated
in; :class:`MultistartFit` and :class:`LatentStateFit` name the evidence that
:class:`RestartAudit` and :class:`LatentStateAudit` are derived from, which used to be
duck-typed against a private protocol in ``behavio.diagnostics``; and
:class:`TaskColumnEstimator` names the predictive context a model consumes without
scoring, which every model already declared by convention.

Layering rule
-------------
``behavio.contracts`` is a runtime leaf with respect to the modules that implement these
contracts. It imports only ``behavio._internal``, ``behavio.study``, ``behavio.clocks``
and ``behavio.posterior`` -- none of which re-export anything from here -- so
``behavio.models.base``, ``behavio.inference``, ``behavio.parameters``,
``behavio.transforms``, ``behavio.validation``, ``behavio.posterior_predictive`` and
``behavio.diagnostics`` can all import this package as thin re-export shims without a
cycle. Two payload types that stay in their implementation modules
(:class:`behavio.parameters.ParameterSpace` and
:class:`behavio.inference.OptimizationProblem`/``OptimizationRun``) are referenced under
``TYPE_CHECKING`` only.

The old ``behavio.diagnostics`` <-> ``behavio.models.base`` cycle is broken by inverting
the dependency: see :mod:`behavio.contracts.estimator` for the :class:`FitAuditor`
registry that :meth:`FitResult.audit` dispatches through.
"""

from __future__ import annotations

from behavio.contracts.adapter import (
    AdapterCapabilities,
    SessionOrderPolicy,
    SourceType,
    StudyAdapter,
    adapter_capabilities,
)
from behavio.contracts.audit import (
    AuditSeverity,
    FitAudit,
    FitAuditPolicy,
    FitAuditStatus,
    FitDiagnostics,
    FitIssue,
    LatentStateAudit,
    LatentStateFit,
    MultistartFit,
    RestartAudit,
)
from behavio.contracts.backend import ObjectiveTarget, OptimizationBackend, PriorMeasure
from behavio.contracts.discrepancy import PredictiveDiscrepancy, PredictiveTail
from behavio.contracts.estimator import (
    BehaviourEstimator,
    BehaviourModel,
    CategoricalBehaviourEstimator,
    CategoricalPrediction,
    DerivedQuantity,
    FitAuditor,
    FitResult,
    GenerativeBehaviourModel,
    ModelCapabilities,
    ModelDataError,
    ModelPrediction,
    Prediction,
    PredictionMode,
    TaskColumnEstimator,
    UnsupportedPredictionMode,
    fit_auditor,
    model_capabilities,
    model_task_columns,
    register_fit_auditor,
    validate_required_task_columns,
)
from behavio.contracts.fold import ValidationFold
from behavio.contracts.natural import (
    NaturalParameterisation,
    natural_covariance,
    natural_quantities,
)
from behavio.contracts.parameters import ParameterSpaceProvider
from behavio.contracts.posterior import (
    AnyBehaviourEstimator,
    AnyGenerativeBehaviourModel,
    GenerativePosteriorBehaviourModel,
    PosteriorBehaviourEstimator,
    PosteriorCentre,
    any_model_capabilities,
    is_posterior_estimator,
    posterior_draw_matrix,
    posterior_log_predictive_density,
    posterior_model_capabilities,
    posterior_parameter_columns,
    posterior_point_summary,
    posterior_summary_message,
)
from behavio.contracts.transform import (
    FittedStudyTransform,
    StudyTransform,
    TransformProvenance,
)

__all__ = [
    "AdapterCapabilities",
    "AnyBehaviourEstimator",
    "AnyGenerativeBehaviourModel",
    "AuditSeverity",
    "BehaviourEstimator",
    "BehaviourModel",
    "CategoricalBehaviourEstimator",
    "CategoricalPrediction",
    "DerivedQuantity",
    "FitAudit",
    "FitAuditPolicy",
    "FitAuditStatus",
    "FitAuditor",
    "FitDiagnostics",
    "FitIssue",
    "FitResult",
    "FittedStudyTransform",
    "GenerativeBehaviourModel",
    "GenerativePosteriorBehaviourModel",
    "LatentStateAudit",
    "LatentStateFit",
    "ModelCapabilities",
    "ModelDataError",
    "ModelPrediction",
    "MultistartFit",
    "NaturalParameterisation",
    "ObjectiveTarget",
    "OptimizationBackend",
    "ParameterSpaceProvider",
    "PosteriorBehaviourEstimator",
    "PosteriorCentre",
    "Prediction",
    "PredictionMode",
    "PredictiveDiscrepancy",
    "PredictiveTail",
    "PriorMeasure",
    "RestartAudit",
    "SessionOrderPolicy",
    "SourceType",
    "StudyAdapter",
    "StudyTransform",
    "TaskColumnEstimator",
    "TransformProvenance",
    "UnsupportedPredictionMode",
    "ValidationFold",
    "adapter_capabilities",
    "any_model_capabilities",
    "fit_auditor",
    "is_posterior_estimator",
    "model_capabilities",
    "model_task_columns",
    "natural_covariance",
    "natural_quantities",
    "posterior_draw_matrix",
    "posterior_log_predictive_density",
    "posterior_model_capabilities",
    "posterior_parameter_columns",
    "posterior_point_summary",
    "posterior_summary_message",
    "register_fit_auditor",
    "validate_required_task_columns",
]
