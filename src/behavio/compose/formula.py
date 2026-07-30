"""Turning a parsed formula, group terms and all, into a model.

``behavio.formula`` has always parsed ``(1 | subject)`` into a :class:`GroupTerm` holding
the grouping column and the within-group design, and has always refused to *use* one:
a :class:`~behavio.design.DesignSpec` is a single fixed matrix and has no varying-effect
representation, so there was nothing to hand the declaration to. There is now.
:func:`model_from_formula` splits a formula into its fixed part, which becomes the model's
design, and its group terms, which become :func:`behavio.compose.hierarchical`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from behavio.compose.hierarchy import hierarchical
from behavio.formula import Formula, FormulaError
from behavio.study import Study

__all__ = ["model_from_formula"]


def model_from_formula(
    formula: str | Formula,
    model: Any,
    *,
    training_study: Study | None = None,
    scale: float = 0.5,
    parameter_scales: Mapping[str, float] | None = None,
    estimate_scale: bool = False,
) -> Any:
    """Configure ``model`` from ``formula``, wrapping it if the formula declares groups.

    The fixed terms become the model's ``design``; a response ``choice ~ ...`` sets the
    scored column; a group term ``(1 + stimulus | subject)`` becomes
    ``hierarchical(model, over="subject", parameters=("intercept", "stimulus"))``.

    ``training_study`` is required exactly when the formula contains a term that estimates
    its own coordinate -- ``scale(x)``, or ``C(x)`` without a declared level set -- and must
    then be the *training* rows of the fold the model will be fitted in, for the same reason
    :meth:`behavio.formula.Formula.fit` requires it.
    """

    parsed = formula if isinstance(formula, Formula) else Formula.parse(formula)
    design = (
        parsed.fixed_design() if training_study is None else parsed.fixed_design(training_study)
    )
    configured = _reconfigure(model, parsed, design)
    if not parsed.groups:
        return configured
    if len(parsed.groups) > 1:
        raise FormulaError(
            "a model can carry one grouping level; this formula declares "
            f"{len(parsed.groups)}. Nest the levels explicitly, or drop one",
            source=parsed.source,
            position=parsed.groups[1].position,
        )
    group = parsed.groups[0]
    within = group.to_design(training_study)
    return hierarchical(
        configured,
        over=group.grouping,
        parameters=within.feature_names,
        scale=scale,
        parameter_scales=parameter_scales,
        estimate_scale=estimate_scale,
    )


def _reconfigure(model: Any, parsed: Formula, design: Any) -> Any:
    """Return ``model`` rebuilt around a formula-built design.

    ``covariates`` and ``choice_lags`` are cleared because the formula already spells both:
    a lag it declares is an ordinary design term, and leaving the model-level shorthand in
    place would silently append a second copy of the same column.
    """

    available = {field.name for field in fields(model)}
    if "design" not in available:
        raise TypeError(
            f"{type(model).__name__} has no design field, so a formula cannot configure it"
        )
    changes: dict[str, Any] = {"design": design}
    if "covariates" in available:
        changes["covariates"] = ()
    if "choice_lags" in available:
        changes["choice_lags"] = 0
    if parsed.response is not None and "outcome" in available:
        changes["outcome"] = parsed.response.choice
    return replace(model, **changes)
