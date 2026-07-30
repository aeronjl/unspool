"""Fold-fitted study transforms: learn on training rows, freeze, apply to both sides.

``StudyTransform``, ``FittedStudyTransform`` and ``TransformProvenance`` are declared in
:mod:`behavio.contracts.transform` -- an extension author implements them there -- and are
re-exported here beside the driver that runs one over a fold.

The driver is the whole point: a transform that is fitted on the full study and then applied
to a held-out fold has leaked, and no amount of care at the call site makes that visible.
:func:`fit_transform_split` makes the boundary a property of the API instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from behavio.contracts.fold import EvaluationFold
from behavio.contracts.transform import (
    FittedStudyTransform,
    StudyTransform,
    TransformProvenance,
)
from behavio.time.clocks import ClockedStudy
from behavio.trials import Study

__all__ = [
    "FittedStudyTransform",
    "FoldTransformResult",
    "StudyTransform",
    "TransformProvenance",
    "fit_transform_split",
    "fit_transform_splits",
]


@dataclass(frozen=True, slots=True)
class FoldTransformResult:
    """Training and test studies transformed by training-only fitted state."""

    split: EvaluationFold
    fitted_transform: FittedStudyTransform
    training: ClockedStudy
    testing: ClockedStudy


def fit_transform_split(
    transform: StudyTransform,
    study: Study,
    split: EvaluationFold,
    *,
    require_prospective: bool = True,
) -> FoldTransformResult:
    """Fit on training rows and apply the frozen transform to both sides of one split."""

    if require_prospective and not split.prospective:
        raise ValueError(
            f"split scheme {split.scheme!r} is not prospective; "
            "set require_prospective=False only for an intentional interpolation analysis"
        )
    training = study.take(split.train_indices)
    testing = study.take(split.test_indices)
    fitted = transform.fit(training)
    return FoldTransformResult(
        split=split,
        fitted_transform=fitted,
        training=fitted.transform(training),
        testing=fitted.transform(testing),
    )


def fit_transform_splits(
    transform: StudyTransform,
    study: Study,
    splits: Iterable[EvaluationFold],
    *,
    require_prospective: bool = True,
) -> tuple[FoldTransformResult, ...]:
    """Apply ``fit_transform_split`` independently at every supplied origin."""

    return tuple(
        fit_transform_split(
            transform,
            study,
            split,
            require_prospective=require_prospective,
        )
        for split in splits
    )
