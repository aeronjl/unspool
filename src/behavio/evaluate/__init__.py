"""Prospective validation: how a study is split, and how a candidate is scored over folds.

:mod:`behavio.evaluate.splits` builds the folds. Every scheme states its leakage semantics
in its own name and carries a ``prospective`` flag, so a forecast that was scored on rows
from the future says so rather than being inferred from the scheme's name.

:mod:`behavio.evaluate.folds` runs a candidate over those folds and reports one
:class:`~behavio.evaluate.folds.SplitEvaluation`. It owns the failure policy -- what a fold
that did not fit is allowed to do to a comparison -- and the convergence gate that a sampled
candidate passes through.

Note the vocabulary. In Behavio *validation* means "checking an input against a declared
contract", and it raises a ``*ValidationError``. What this package does is called
*evaluation over splits*, never validation.
"""

from behavio.evaluate.folds import (
    FoldEvaluation,
    FoldFailure,
    FoldFailurePolicy,
    FoldStage,
    PosteriorFoldEvidence,
    PosteriorFoldPolicy,
    SplitEvaluation,
    evaluate_splits,
)
from behavio.evaluate.splits import (
    CohortSplit,
    EvaluationFold,
    HistoricalCohortForecastSplit,
    PopulationForecastSplit,
    PopulationSplit,
    Split,
    cohort_forward_session_splits,
    forward_session_splits,
    historical_cohort_forecast_splits,
    leave_one_lab_out_session_forecast_splits,
    leave_one_lab_out_splits,
    leave_one_session_out_splits,
    leave_one_subject_out_splits,
    within_session_rolling_splits,
)

__all__ = [
    "CohortSplit",
    "EvaluationFold",
    "FoldEvaluation",
    "FoldFailure",
    "FoldFailurePolicy",
    "FoldStage",
    "HistoricalCohortForecastSplit",
    "PopulationForecastSplit",
    "PopulationSplit",
    "PosteriorFoldEvidence",
    "PosteriorFoldPolicy",
    "Split",
    "SplitEvaluation",
    "cohort_forward_session_splits",
    "evaluate_splits",
    "forward_session_splits",
    "historical_cohort_forecast_splits",
    "leave_one_lab_out_session_forecast_splits",
    "leave_one_lab_out_splits",
    "leave_one_session_out_splits",
    "leave_one_subject_out_splits",
    "within_session_rolling_splits",
]
