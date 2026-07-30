"""Fixed design-matrix terms, and the formula notation that desugars onto them.

:mod:`behavio.design.matrix` is the term algebra -- numeric, categorical, history, kernel,
standardising and interaction terms that expand a :class:`~behavio.trials.Study` into named
columns with explicit history boundaries. :mod:`behavio.design.formula` is a concise way to
write one down; every accepted formula desugars onto terms the matrix layer already has, and
the parser invents no second algebra.
"""

from behavio.design.formula import (
    CategoricalFormulaTerm,
    ColumnFormulaTerm,
    Formula,
    FormulaError,
    FormulaResponse,
    FormulaSyntaxError,
    FormulaTerm,
    GroupTerm,
    InteractionFormulaTerm,
    KernelFormulaTerm,
    LagFormulaTerm,
    StandardizeFormulaTerm,
    describe_design,
    describe_term,
)
from behavio.design.matrix import (
    CategoricalTerm,
    DesignMatrix,
    DesignSpec,
    DesignTerm,
    DesignValidationError,
    FeatureBlock,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
    NumericTerm,
    StandardizeTerm,
)

__all__ = [
    "CategoricalFormulaTerm",
    "CategoricalTerm",
    "ColumnFormulaTerm",
    "DesignMatrix",
    "DesignSpec",
    "DesignTerm",
    "DesignValidationError",
    "FeatureBlock",
    "Formula",
    "FormulaError",
    "FormulaResponse",
    "FormulaSyntaxError",
    "FormulaTerm",
    "GroupTerm",
    "HistoryKernelTerm",
    "HistoryTerm",
    "InteractionFormulaTerm",
    "InteractionTerm",
    "KernelFormulaTerm",
    "LagFormulaTerm",
    "NumericTerm",
    "StandardizeFormulaTerm",
    "StandardizeTerm",
    "describe_design",
    "describe_term",
]
