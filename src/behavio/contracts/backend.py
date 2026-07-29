"""The deterministic optimization-backend contract.

``ObjectiveTarget`` and ``PriorMeasure`` are pure declarations and live here.
``OptimizationProblem``, ``OptimizationAttempt`` and ``OptimizationRun`` stay in
:mod:`behavio.inference`: they carry substantial validation logic and
``OptimizationProblem`` needs a concrete :class:`behavio.parameters.ParameterSpace` at
runtime, which ``behavio.contracts`` must not import (``behavio.parameters`` re-exports
:class:`~behavio.contracts.parameters.ParameterSpaceProvider` from here, so a runtime
import would be circular). They are referenced under ``TYPE_CHECKING`` only, which keeps
this package a runtime leaf while the protocol stays fully typed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from behavio.inference import OptimizationProblem, OptimizationRun


class ObjectiveTarget(StrEnum):
    """The statistical quantity minimized by an optimization problem."""

    MAXIMUM_LIKELIHOOD = "maximum-likelihood"
    MAXIMUM_A_POSTERIORI = "maximum-a-posteriori"


class PriorMeasure(StrEnum):
    """Coordinate measure used when a MAP objective includes transform Jacobians."""

    NATURAL = "natural"
    OPTIMIZER = "optimizer"


@runtime_checkable
class OptimizationBackend(Protocol):
    """Structural backend contract for an identical optimization problem."""

    @property
    def backend_name(self) -> str: ...

    def run(self, problem: OptimizationProblem) -> OptimizationRun: ...
