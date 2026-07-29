"""The fold-fitted study-transform contract.

:class:`TransformProvenance` moves here with the two protocols because
:class:`FittedStudyTransform` declares it structurally. ``behavio.clocks`` and
``behavio.study`` are imported normally: neither re-exports anything from
``behavio.contracts``, so there is no cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from behavio.clocks import ClockedStudy, ClockSpec
from behavio.study import Study


@dataclass(frozen=True, slots=True)
class TransformProvenance:
    """Training-only state retained by a fitted temporal transform."""

    transform_signature: str
    n_fit_trials: int
    fit_subjects: tuple[Any, ...]
    learned_values: Mapping[Any, float | None]

    def __post_init__(self) -> None:
        if self.n_fit_trials < 1:
            raise ValueError("n_fit_trials must be positive")
        object.__setattr__(self, "fit_subjects", tuple(self.fit_subjects))
        object.__setattr__(self, "learned_values", MappingProxyType(dict(self.learned_values)))


@runtime_checkable
class FittedStudyTransform(Protocol):
    """A transform whose learned state is fixed and inspectable."""

    @property
    def signature(self) -> str: ...

    @property
    def output_clock(self) -> ClockSpec: ...

    @property
    def provenance(self) -> TransformProvenance: ...

    def transform(self, study: Study) -> ClockedStudy: ...


@runtime_checkable
class StudyTransform(Protocol):
    """A transform that learns state from one training Study."""

    @property
    def signature(self) -> str: ...

    def fit(self, study: Study) -> FittedStudyTransform: ...
