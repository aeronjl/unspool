"""The posterior-predictive discrepancy contract.

``PredictiveTail`` moves here with :class:`PredictiveDiscrepancy` because the protocol
declares it structurally; keeping them together avoids a back-reference into
:mod:`behavio.posterior_predictive`, which re-exports both.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray


class PredictiveTail(StrEnum):
    """Reference tail used to summarize a replicated discrepancy."""

    LOWER = "lower"
    UPPER = "upper"
    TWO_SIDED = "two-sided"


@runtime_checkable
class PredictiveDiscrepancy(Protocol):
    """A named, provenance-bearing scalar summary of one observation vector."""

    @property
    def name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    @property
    def tail(self) -> PredictiveTail: ...

    def evaluate(self, values: NDArray[Any]) -> float: ...
