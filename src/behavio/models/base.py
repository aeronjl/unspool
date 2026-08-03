"""Shared contracts for behavioural models and their outputs.

These names are declared in :mod:`behavio.contracts`, the single address for Behavio's
extension surface. This module re-exports them at the address a *model author* is already
reading: someone writing an estimator beside the first-party catalogue should not have to
know that the protocol they implement is defined one package over.
"""

from __future__ import annotations

from behavio._internal.arrays import protected_array
from behavio.contracts.audit import FitDiagnostics
from behavio.contracts.estimator import (
    LOG_DENSITY_FLOOR,
    BehaviourEstimator,
    BehaviourModel,
    CategoricalBehaviourEstimator,
    CategoricalPrediction,
    CensoredDensityPrediction,
    DensityBehaviourEstimator,
    DensityPrediction,
    FitResult,
    GenerativeBehaviourModel,
    ModelCapabilities,
    ModelDataError,
    ModelPrediction,
    Prediction,
    PredictionMode,
    UnsupportedPredictionMode,
    model_capabilities,
)

# ``_protected_array`` was this module's private array-immutability helper and became the
# de-facto package-wide utility. It now lives in ``behavio._internal.arrays``; this alias
# keeps any remaining importer working.
_protected_array = protected_array

__all__ = [
    "LOG_DENSITY_FLOOR",
    "BehaviourEstimator",
    "BehaviourModel",
    "CategoricalBehaviourEstimator",
    "CategoricalPrediction",
    "CensoredDensityPrediction",
    "DensityBehaviourEstimator",
    "DensityPrediction",
    "FitDiagnostics",
    "FitResult",
    "GenerativeBehaviourModel",
    "ModelCapabilities",
    "ModelDataError",
    "ModelPrediction",
    "Prediction",
    "PredictionMode",
    "UnsupportedPredictionMode",
    "model_capabilities",
]
