"""The bridge that makes ``covariates`` a special case of :class:`DesignSpec`.

Behavio grew two parallel ways of saying what a linear predictor is made of. A model took
``covariates: tuple[str, ...]``, meaning "one raw numeric column per name, plus an
intercept, plus whatever history I add myself"; :mod:`behavio.design` took a composable
term algebra that could say considerably more but that only one model accepted. Anything
the term algebra could express and the tuple could not -- an interaction, a fixed-level
contrast, a weighted history kernel -- had to be materialized into the ``Study`` by hand
and then referred to by an opaque column name.

This module removes the second mechanism by deriving it from the first.
:func:`covariate_design` writes down exactly what a ``covariates`` tuple has always meant,
so a model can hold one :class:`DesignSpec` internally no matter which way it was
configured, and :func:`resolve_design` is the single place that chooses between them.

The correspondence is exact rather than approximate, and deliberately so: a
``NumericTerm`` with the default centre and scale reproduces its source column bit for
bit, and a ``HistoryTerm`` with ``coding="effect"`` reproduces the -1/+1 lagged outcome
with the same session resets and the same zero fill. A design-built model and a
``covariates``-built model therefore agree on their matrices *and* on their parameter
names, which is what lets fitted artefacts, benchmark results, and signatures survive the
change.
"""

from __future__ import annotations

from behavio.contracts.estimator import ModelDataError
from behavio.design import DesignSpec, DesignTerm, DesignValidationError, HistoryTerm, NumericTerm

DEFAULT_HISTORY_RESET: tuple[str, ...] = ("subject", "session")
"""Behavio's trial-order contract: history never crosses a subject or session boundary."""


def covariate_terms(covariates: tuple[str, ...]) -> tuple[DesignTerm, ...]:
    """Return one identity :class:`NumericTerm` per covariate name, in order."""

    return tuple(NumericTerm(column=name) for name in covariates)


def covariate_design(covariates: tuple[str, ...], *, intercept: bool = True) -> DesignSpec:
    """Return the design a ``covariates`` tuple has always denoted."""

    return DesignSpec(terms=covariate_terms(covariates), intercept=intercept)


def outcome_history_term(
    outcome: str,
    lags: int,
    *,
    name: str = "choice",
    reset_by: tuple[str, ...] = DEFAULT_HISTORY_RESET,
) -> HistoryTerm:
    """Return the effect-coded lagged-outcome term a model's ``choice_lags`` denotes.

    ``name`` defaults to ``"choice"`` rather than to the outcome column because the
    resulting coefficients have always been called ``choice_lag_1`` and so on, whatever
    the outcome column was named.
    """

    return HistoryTerm(
        column=outcome,
        lags=tuple(range(1, lags + 1)),
        reset_by=reset_by,
        coding="effect",
        fill_value=0.0,
        name=name,
    )


def resolve_design(
    design: DesignSpec | None,
    covariates: tuple[str, ...],
    *,
    intercept: bool = True,
) -> DesignSpec:
    """Return the exogenous design, from an explicit spec or from ``covariates``."""

    if design is None:
        return covariate_design(covariates, intercept=intercept)
    return design


def extend(design: DesignSpec, *terms: DesignTerm) -> DesignSpec:
    """Return ``design`` with further terms appended, preserving its intercept."""

    if not terms:
        return design
    return DesignSpec(terms=(*design.terms, *terms), intercept=design.intercept)


def validate_design_choice(design: DesignSpec | None, covariates: tuple[str, ...]) -> None:
    """Reject the one ambiguous construction: a design *and* a covariate tuple."""

    if design is None:
        return
    if not isinstance(design, DesignSpec):
        raise TypeError("design must be a DesignSpec")
    if covariates:
        raise ValueError(
            "pass either covariates or design, not both; covariates=(...) is shorthand for "
            "DesignSpec(terms=(NumericTerm(column=...), ...))"
        )


def build_matrix(design: DesignSpec, study: object) -> object:
    """Build a design matrix, reporting a design failure as a model data failure.

    A malformed study is a :class:`ModelDataError` from every other entry point in the
    package, and a user who mistyped a covariate should not have to know that the column
    was reached through a design term to catch the error.
    """

    try:
        return design.build(study)  # type: ignore[arg-type]
    except DesignValidationError as error:
        raise ModelDataError(str(error)) from error
