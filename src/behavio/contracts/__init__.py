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
A composable penalised linear model              :class:`PenalisedLinearEstimator`
A composable bounded-coordinate model            :class:`BoundedCoordinateEstimator`
A simpler process a model can be mixed with      :class:`MixtureComponent`
A fold-fitted temporal transform                 :class:`StudyTransform`
A training/test partition                        :class:`EvaluationFold`
A reporting coordinate distinct from the fit     :class:`NaturalParameterisation`
A retained multistart optimizer                  :class:`MultistartFit`
A retained latent-state fit                      :class:`LatentStateFit`
===============================================  =====================================

The last three are *optional widenings*: a model that implements none of them is described
exactly as it was before they existed. :class:`NaturalParameterisation` names the
coordinate a result is reported in, which is not in general the coordinate it is estimated
in; :class:`MultistartFit` and :class:`LatentStateFit` name the evidence that
:class:`RestartAudit` and :class:`LatentStateAudit` are derived from, which used to be
duck-typed against a private protocol in ``behavio.diagnostics``.

A fourth widening, ``TaskColumnEstimator``, has been removed rather than kept: the
predictive context a model reads but does not score is now ``required_task_columns`` on
:class:`BehaviourEstimator` itself. It was a side protocol only so that models which had
not yet written the declaration down were not evicted from the base contract, and every
first-party model declares it. A model that reads nothing but the column it scores
declares ``()``.

Layering rule
-------------
``behavio.contracts`` is a runtime leaf with respect to the modules that implement these
contracts. It imports only ``behavio._internal``, ``behavio.trials`` and
``behavio.posterior.result`` -- none of which import anything at all from inside the
package, which is what makes them safe to name from here -- so
``behavio.models.base``, ``behavio.inference.optimize``, ``behavio.inference.parameters``,
``behavio.time.transforms``, ``behavio.evaluate.splits``, ``behavio.posterior.predictive`` and
``behavio.diagnostics`` can all import this package as thin re-export shims without a
cycle. Two payload types that stay in their implementation modules
(:class:`behavio.inference.parameters.ParameterSpace` and
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
    ConvergenceStatus,
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
from behavio.contracts.bounded import (
    BoundedCoordinateEstimator,
    RowCoefficientDesign,
    RowObjective,
    require_bounded_coordinate,
    require_composable,
)
from behavio.contracts.compose import (
    GroupBlocks,
    LinearPredictorLikelihood,
    PenalisedDesign,
    PenalisedLinearEstimator,
    VaryingEffects,
    expand_group_design,
    expand_group_penalty,
    group_blocks,
    information_matrix,
    joint_parameter_names,
    linear_predictor,
    parameter_gradient,
    require_penalised_linear,
    ridge_group_draw,
    ridge_group_penalty,
    validate_predictor_shape,
)
from behavio.contracts.discrepancy import PredictiveDiscrepancy, PredictiveTail
from behavio.contracts.estimator import (
    LOG_DENSITY_FLOOR,
    BehaviourEstimator,
    BehaviourModel,
    CategoricalBehaviourEstimator,
    CategoricalPrediction,
    DensityBehaviourEstimator,
    DensityPrediction,
    DerivedQuantity,
    FitAuditor,
    FitResult,
    GenerativeBehaviourModel,
    ModelCapabilities,
    ModelDataError,
    ModelPrediction,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    fit_auditor,
    model_capabilities,
    model_score_metrics,
    model_task_columns,
    register_fit_auditor,
    validate_required_task_columns,
)
from behavio.contracts.fold import EvaluationFold
from behavio.contracts.mixture import (
    MixtureComponent,
    mixture_logit,
    mixture_weight,
    require_independent_rows,
    require_mixable,
    require_mixture_component,
    validate_weight_bounds,
)
from behavio.contracts.natural import (
    NaturalParameterisation,
    natural_covariance,
    natural_quantities,
)
from behavio.contracts.parameters import ParameterSpaceProvider
from behavio.contracts.posterior import (
    AnyBehaviourEstimator,
    AnyGenerativeBehaviourModel,
    DesignGenerativeBehaviourModel,
    GenerativePosteriorBehaviourModel,
    PosteriorBehaviourEstimator,
    PosteriorCentre,
    any_model_capabilities,
    bind_to_design,
    is_design_generative,
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
    "LOG_DENSITY_FLOOR",
    "AdapterCapabilities",
    "AnyBehaviourEstimator",
    "AnyGenerativeBehaviourModel",
    "AuditSeverity",
    "BehaviourEstimator",
    "BehaviourModel",
    "BoundedCoordinateEstimator",
    "CategoricalBehaviourEstimator",
    "CategoricalPrediction",
    "ConvergenceStatus",
    "DensityBehaviourEstimator",
    "DensityPrediction",
    "DerivedQuantity",
    "DesignGenerativeBehaviourModel",
    "EvaluationFold",
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
    "GroupBlocks",
    "LatentStateAudit",
    "LatentStateFit",
    "LinearPredictorLikelihood",
    "MixtureComponent",
    "ModelCapabilities",
    "ModelDataError",
    "ModelPrediction",
    "MultistartFit",
    "NaturalParameterisation",
    "ObjectiveTarget",
    "OptimizationBackend",
    "ParameterSpaceProvider",
    "PenalisedDesign",
    "PenalisedLinearEstimator",
    "PosteriorBehaviourEstimator",
    "PosteriorCentre",
    "Prediction",
    "PredictionMode",
    "PredictiveDiscrepancy",
    "PredictiveTail",
    "PriorMeasure",
    "RestartAudit",
    "RowCoefficientDesign",
    "RowObjective",
    "SessionOrderPolicy",
    "SourceType",
    "StudyAdapter",
    "StudyTransform",
    "TransformProvenance",
    "UnsupportedPredictionMode",
    "VaryingEffects",
    "adapter_capabilities",
    "any_model_capabilities",
    "bind_to_design",
    "expand_group_design",
    "expand_group_penalty",
    "fit_auditor",
    "group_blocks",
    "information_matrix",
    "is_design_generative",
    "is_posterior_estimator",
    "joint_parameter_names",
    "linear_predictor",
    "mixture_logit",
    "mixture_weight",
    "model_capabilities",
    "model_score_metrics",
    "model_task_columns",
    "natural_covariance",
    "natural_quantities",
    "parameter_gradient",
    "posterior_draw_matrix",
    "posterior_log_predictive_density",
    "posterior_model_capabilities",
    "posterior_parameter_columns",
    "posterior_point_summary",
    "posterior_summary_message",
    "register_fit_auditor",
    "require_bounded_coordinate",
    "require_composable",
    "require_independent_rows",
    "require_mixable",
    "require_mixture_component",
    "require_penalised_linear",
    "ridge_group_draw",
    "ridge_group_penalty",
    "validate_predictor_shape",
    "validate_required_task_columns",
    "validate_weight_bounds",
]
