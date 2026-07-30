"""A small formula language that desugars onto the fixed terms in :mod:`behavio.design.matrix`.

The design layer already has a complete term algebra -- :class:`~behavio.design.NumericTerm`,
:class:`~behavio.design.CategoricalTerm`, :class:`~behavio.design.HistoryTerm`,
:class:`~behavio.design.HistoryKernelTerm`, :class:`~behavio.design.StandardizeTerm` and
:class:`~behavio.design.InteractionTerm` -- but no concise way to write one down. This module
is that notation and nothing more. Every accepted formula desugars onto terms that already
exist; the parser invents no second algebra and no term that :mod:`behavio.design.matrix` cannot
build.

The grammar, in precedence order from loosest to tightest binding::

    formula     := [ response "~" ] sum
    response    := column | column "|" column        # response_time | choice
    sum         := ["+" | "-"] product (("+" | "-") product)*
    product     := interaction ("*" interaction)*
    interaction := atom (":" atom)*
    atom        := "1" | "0" | column | call | "(" sum ")" | "(" sum "|" column ")"
    call        := "numeric" "(" column [, kwargs] ")"
                 | "C"       "(" column [, levels] [, kwargs] ")"
                 | "scale"   "(" column [, kwargs] ")"
                 | "lag"     "(" column [, lags] [, kwargs] ")"
                 | "kernel"  "(" column [, weights] [, kwargs] ")"

``a * b`` expands to ``a + b + a:b``; ``a:b`` is the interaction alone. ``1`` and ``0``
declare and suppress the intercept, and ``- 1`` is a synonym for ``0``.

``lag()`` and ``kernel()`` default to ``coding="effect"``, so ``lag(choice, 1)`` builds the
same -1/+1 ``choice_lag_1`` column that ``BernoulliHistoryGLM(choice_lags=1)`` has always
built. Write ``coding="identity"`` for the literal lagged value of a non-binary column.
:data:`_DEFAULT_CODING` says why the formula's default is not ``HistoryTerm``'s.

Two atoms estimate their coordinate from rows -- ``scale(x)`` and ``C(x)`` without an explicit
level set. They can only be resolved through :meth:`Formula.fit`, which names the training
study in the call. :meth:`Formula.to_design` refuses them by construction rather than
silently reading the whole study, which is the same information boundary
:func:`behavio.time.transforms.fit_transform_split` enforces for clocks.

Group terms such as ``(1|subject)`` parse into :class:`GroupTerm` and are honoured by
:func:`behavio.compose.model_from_formula`, which hands the grouping column and the
within-group design to :func:`behavio.compose.hierarchical`. They are still not part of a
:class:`~behavio.design.DesignSpec`, which is one fixed matrix with no varying-effect
representation, so :meth:`Formula.to_design` and :meth:`Formula.fit` route a formula that
declares one to the combinator rather than silently dropping it.
:meth:`Formula.fixed_design` is the accessor that deliberately keeps only the fixed part.
"""

from __future__ import annotations

import difflib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np

from behavio.design.matrix import (
    CategoricalTerm,
    DesignSpec,
    DesignTerm,
    DesignValidationError,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
    NumericTerm,
    StandardizeTerm,
)
from behavio.trials import Study

__all__ = [
    "CategoricalFormulaTerm",
    "ColumnFormulaTerm",
    "Formula",
    "FormulaError",
    "FormulaResponse",
    "FormulaSyntaxError",
    "FormulaTerm",
    "GroupTerm",
    "InteractionFormulaTerm",
    "KernelFormulaTerm",
    "LagFormulaTerm",
    "StandardizeFormulaTerm",
    "describe_design",
    "describe_term",
]

FUNCTION_NAMES: Final = ("numeric", "C", "scale", "lag", "kernel")

_BARE_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTIFIER: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_OPERATORS: Final = frozenset("~+-*:()|,=[]")
_DEFAULT_RESET_BY: Final = ("subject", "session")

_DEFAULT_CODING: Final = "effect"
"""The coding ``lag()`` and ``kernel()`` assume, which is *not* ``HistoryTerm``'s own.

:class:`~behavio.design.HistoryTerm` defaults to ``"identity"``, because a design term is
written by someone who has already decided what the column means. A formula is the surface
users migrate to from the ``choice_lags=`` shorthand, and that shorthand has always meant
effect coding: ``BernoulliHistoryGLM(choice_lags=1)`` builds a -1/+1 column called
``choice_lag_1``. If ``lag(choice, 1)`` built a 0/1 column under the same name, the two
spellings would produce different matrices and differently scaled coefficients while
agreeing on every label, and nothing would say so.

Effect coding cannot fail quietly -- :meth:`~behavio.design.HistoryTerm.build` rejects a
column that is not zero/one -- so the cost of the choice is a loud error on a continuous
column, answered by :func:`_identity_coding_hint`, rather than a silent divergence.
"""


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class FormulaError(ValueError):
    """Raised when a formula cannot be parsed, checked, or honoured.

    The rendered message points at the offending position in the source string, because a
    formula is a user-facing surface and ``invalid syntax`` is not a usable report.
    """

    def __init__(
        self,
        detail: str,
        *,
        source: str | None = None,
        position: int | None = None,
    ) -> None:
        self.detail = detail
        self.source = source
        self.position = position
        super().__init__(_annotate(detail, source, position))


class FormulaSyntaxError(FormulaError):
    """Raised when the text of a formula is not well formed."""


def _annotate(detail: str, source: str | None, position: int | None) -> str:
    if source is None or position is None:
        return detail
    line = source.replace("\n", " ").replace("\t", " ")
    caret = f"{' ' * max(position, 0)}^"
    return f"{detail} at position {position}\n  {line}\n  {caret}"


# --------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str
    position: int
    value: Any = None
    quoted: bool = False


def _tokenize(source: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    length = len(source)
    while index < length:
        character = source[index]
        if character.isspace():
            index += 1
            continue
        if character == "`":
            end = source.find("`", index + 1)
            if end == -1:
                raise FormulaSyntaxError(
                    "unterminated backquoted column name", source=source, position=index
                )
            name = source[index + 1 : end]
            if not name:
                raise FormulaSyntaxError(
                    "a backquoted column name cannot be empty", source=source, position=index
                )
            tokens.append(_Token("name", name, index, name, quoted=True))
            index = end + 1
            continue
        if character in "'\"":
            text, end = _read_string(source, index)
            tokens.append(_Token("string", text, index, text))
            index = end
            continue
        number = _NUMBER.match(source, index)
        if character.isdigit() and number is not None:
            text = number.group()
            value: Any = float(text) if ("." in text or "e" in text or "E" in text) else int(text)
            tokens.append(_Token("number", text, index, value))
            index = number.end()
            continue
        match = _IDENTIFIER.match(source, index)
        if match is not None:
            text = match.group()
            tokens.append(_Token("name", text, index, text))
            index = match.end()
            continue
        if character in _OPERATORS:
            tokens.append(_Token("op", character, index))
            index += 1
            continue
        raise FormulaSyntaxError(
            f"unexpected character {character!r}", source=source, position=index
        )
    tokens.append(_Token("end", "", length))
    return tuple(tokens)


def _read_string(source: str, index: int) -> tuple[str, int]:
    quote = source[index]
    characters: list[str] = []
    position = index + 1
    while position < len(source):
        character = source[position]
        if character == "\\" and position + 1 < len(source):
            characters.append(source[position + 1])
            position += 2
            continue
        if character == quote:
            return "".join(characters), position + 1
        characters.append(character)
        position += 1
    raise FormulaSyntaxError("unterminated string literal", source=source, position=index)


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------


def _render_column(column: str) -> str:
    return column if _BARE_NAME.match(column) else f"`{column}`"


def _render_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_literal(item) for item in value) + "]"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    raise FormulaError(f"a formula cannot spell the literal {value!r}")


def _render_call(function: str, column: str, *arguments: str) -> str:
    body = ", ".join((_render_column(column), *(argument for argument in arguments if argument)))
    return f"{function}({body})"


def _keyword(name: str, value: Any, default: Any) -> str:
    if value == default and type(value) is type(default):
        return ""
    return f"{name}={_render_literal(value)}"


# --------------------------------------------------------------------------------------
# Parsed right-hand-side terms
# --------------------------------------------------------------------------------------


class FormulaTerm(ABC):
    """One parsed right-hand-side term that desugars onto exactly one fixed design term.

    Every node carries ``position``, the offset in the source string where it began, so
    that a failure to check or resolve it can point back at what the user wrote.
    """

    __slots__ = ()

    position: int

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Study columns this term reads, in declaration order."""

    @property
    @abstractmethod
    def data_dependent(self) -> bool:
        """Whether resolving this term estimates anything from study rows."""

    @abstractmethod
    def render(self) -> str:
        """Return the canonical formula spelling of this term."""

    @abstractmethod
    def resolve(self, study: Study | None) -> DesignTerm:
        """Return the fixed design term, estimating from ``study`` when data-dependent."""

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True, slots=True)
class ColumnFormulaTerm(FormulaTerm):
    """A bare column, or ``numeric(column, ...)``, desugaring to a ``NumericTerm``."""

    column: str
    name: str | None = None
    center: float = 0.0
    scale: float = 1.0
    position: int = field(default=0, compare=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    @property
    def data_dependent(self) -> bool:
        return False

    def render(self) -> str:
        if self.name is None and self.center == 0.0 and self.scale == 1.0:
            return _render_column(self.column)
        return _render_call(
            "numeric",
            self.column,
            _keyword("center", self.center, 0.0),
            _keyword("scale", self.scale, 1.0),
            _keyword("name", self.name, None),
        )

    def resolve(self, study: Study | None) -> DesignTerm:
        return NumericTerm(column=self.column, name=self.name, center=self.center, scale=self.scale)


@dataclass(frozen=True, slots=True)
class CategoricalFormulaTerm(FormulaTerm):
    """``C(column, ...)``, desugaring to a ``CategoricalTerm``.

    ``levels=None`` means the level set is read off a training study. That is data
    dependent and therefore only reachable through :meth:`Formula.fit`.
    """

    column: str
    levels: tuple[Any, ...] | None = None
    reference: Any | None = None
    name: str | None = None
    drop_reference: bool = True
    position: int = field(default=0, compare=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    @property
    def data_dependent(self) -> bool:
        return self.levels is None

    def render(self) -> str:
        levels = "" if self.levels is None else f"levels={_render_literal(list(self.levels))}"
        return _render_call(
            "C",
            self.column,
            levels,
            _keyword("reference", self.reference, None),
            _keyword("name", self.name, None),
            _keyword("drop_reference", self.drop_reference, True),
        )

    def resolve(self, study: Study | None) -> DesignTerm:
        levels = self.levels
        if levels is None:
            if study is None:
                raise FormulaError(
                    f"{self.render()} has no declared level set", position=self.position
                )
            levels = _observed_levels(study, self.column, self.position)
        return CategoricalTerm(
            column=self.column,
            levels=levels,
            reference=self.reference,
            name=self.name,
            drop_reference=self.drop_reference,
        )


@dataclass(frozen=True, slots=True)
class StandardizeFormulaTerm(FormulaTerm):
    """``scale(column, ...)``, desugaring to a training-fitted ``StandardizeTerm``."""

    column: str
    ddof: int = 0
    name: str | None = None
    position: int = field(default=0, compare=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    @property
    def data_dependent(self) -> bool:
        return True

    def render(self) -> str:
        return _render_call(
            "scale",
            self.column,
            _keyword("ddof", self.ddof, 0),
            _keyword("name", self.name, None),
        )

    def resolve(self, study: Study | None) -> DesignTerm:
        if study is None:
            raise FormulaError(
                f"{self.render()} estimates a centre and scale from rows", position=self.position
            )
        return StandardizeTerm(column=self.column, name=self.name, ddof=self.ddof).fit(study)


def _identity_coding_hint(term: LagFormulaTerm | KernelFormulaTerm) -> str | None:
    """Return this call spelled with ``coding='identity'``, if its coding was defaulted.

    The term renders its own escape hatch, so the suggestion is always a formula the user
    can paste back: the same column, the same lags or weights, the same reset boundary,
    with the one argument they need added.
    """

    if not term.coding_defaulted:
        return None
    return replace(term, coding="identity").render()


@dataclass(frozen=True, slots=True)
class LagFormulaTerm(FormulaTerm):
    """``lag(column, ...)``, desugaring to a ``HistoryTerm``.

    ``coding`` defaults to ``"effect"`` rather than to ``HistoryTerm``'s own
    ``"identity"``. See :data:`_DEFAULT_CODING`.
    """

    column: str
    lags: tuple[int, ...] = (1,)
    reset_by: tuple[str, ...] = _DEFAULT_RESET_BY
    coding: str = _DEFAULT_CODING
    fill_value: float = 0.0
    name: str | None = None
    coding_defaulted: bool = field(default=False, compare=False)
    position: int = field(default=0, compare=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.column, *self.reset_by)))

    @property
    def data_dependent(self) -> bool:
        return False

    def render(self) -> str:
        lags = ", ".join(str(lag) for lag in self.lags)
        return _render_call(
            "lag",
            self.column,
            lags,
            _keyword("reset_by", list(self.reset_by), list(_DEFAULT_RESET_BY)),
            _keyword("coding", self.coding, _DEFAULT_CODING),
            _keyword("fill_value", self.fill_value, 0.0),
            _keyword("name", self.name, None),
        )

    def resolve(self, study: Study | None) -> DesignTerm:
        return HistoryTerm(
            column=self.column,
            lags=self.lags,
            reset_by=self.reset_by,
            coding=self.coding,
            fill_value=self.fill_value,
            name=self.name,
            coding_hint=_identity_coding_hint(self),
        )


@dataclass(frozen=True, slots=True)
class KernelFormulaTerm(FormulaTerm):
    """``kernel(column, [weights], ...)``, desugaring to a ``HistoryKernelTerm``.

    ``coding`` defaults to ``"effect"``, exactly as for ``lag()``: a weighted history
    kernel is a weighted sum of lags, and the two spellings must not disagree about what a
    lag of a binary column is. See :data:`_DEFAULT_CODING`.
    """

    column: str
    weights: tuple[float, ...] = ()
    lags: tuple[int, ...] | None = None
    reset_by: tuple[str, ...] = _DEFAULT_RESET_BY
    coding: str = _DEFAULT_CODING
    fill_value: float = 0.0
    name: str | None = None
    coding_defaulted: bool = field(default=False, compare=False)
    position: int = field(default=0, compare=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.column, *self.reset_by)))

    @property
    def data_dependent(self) -> bool:
        return False

    def render(self) -> str:
        default_lags = tuple(range(1, len(self.weights) + 1))
        lags = "" if self.lags in (None, default_lags) else _keyword("lags", list(self.lags), None)
        return _render_call(
            "kernel",
            self.column,
            _render_literal(list(self.weights)),
            lags,
            _keyword("reset_by", list(self.reset_by), list(_DEFAULT_RESET_BY)),
            _keyword("coding", self.coding, _DEFAULT_CODING),
            _keyword("fill_value", self.fill_value, 0.0),
            _keyword("name", self.name, None),
        )

    def resolve(self, study: Study | None) -> DesignTerm:
        return HistoryKernelTerm(
            column=self.column,
            weights=self.weights,
            lags=self.lags,
            reset_by=self.reset_by,
            coding=self.coding,
            fill_value=self.fill_value,
            name=self.name,
            coding_hint=_identity_coding_hint(self),
        )


@dataclass(frozen=True, slots=True)
class InteractionFormulaTerm(FormulaTerm):
    """``a:b``, desugaring to left-associated ``InteractionTerm`` products."""

    factors: tuple[FormulaTerm, ...]
    position: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        if len(self.factors) < 2:
            raise FormulaError("an interaction needs at least two factors")

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(column for factor in self.factors for column in factor.required_columns)
        )

    @property
    def data_dependent(self) -> bool:
        return any(factor.data_dependent for factor in self.factors)

    def render(self) -> str:
        return ":".join(factor.render() for factor in self.factors)

    def resolve(self, study: Study | None) -> DesignTerm:
        resolved = self.factors[0].resolve(study)
        for factor in self.factors[1:]:
            resolved = InteractionTerm(left=resolved, right=factor.resolve(study))
        return resolved


# --------------------------------------------------------------------------------------
# Group terms and responses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupTerm:
    """A parsed ``(expression | grouping)`` varying-effect declaration.

    ``DesignSpec`` describes a single fixed matrix and has no representation for effects
    that vary by group, so this declaration is not a design term; it is an argument to
    :func:`behavio.compose.hierarchical`, which needs exactly two things and finds both
    here: the grouping column, and the within-group design that :meth:`to_design` returns.
    :func:`behavio.compose.model_from_formula` performs that hand-off.
    """

    grouping: str
    terms: tuple[FormulaTerm, ...] = ()
    intercept: bool = True
    position: int = field(default=0, compare=False)

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = [self.grouping]
        columns.extend(column for term in self.terms for column in term.required_columns)
        return tuple(dict.fromkeys(columns))

    @property
    def data_dependent(self) -> bool:
        return any(term.data_dependent for term in self.terms)

    def render(self) -> str:
        parts = ["1" if self.intercept else "0"]
        parts.extend(term.render() for term in self.terms)
        return f"({' + '.join(parts)} | {_render_column(self.grouping)})"

    def to_design(self, study: Study | None = None) -> DesignSpec:
        """Return the within-group design this term declares."""

        return DesignSpec(
            terms=tuple(term.resolve(study) for term in self.terms), intercept=self.intercept
        )

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True, slots=True)
class FormulaResponse:
    """The observed columns a formula scores, in ``TaskSpec`` order.

    ``TaskSpec.outcome_columns`` is ``(choice, response_time)``, and this record keeps the
    same two roles under the same two names. The surface spelling puts the response time
    first -- ``rt | choice`` -- because that reads as "a response time, resolved by a
    choice", which is what a joint drift-diffusion likelihood scores.
    """

    choice: str
    response_time: str | None = None
    position: int = field(default=0, compare=False)

    @property
    def outcome_columns(self) -> tuple[str, ...]:
        if self.response_time is None:
            return (self.choice,)
        return (self.choice, self.response_time)

    def render(self) -> str:
        choice = _render_column(self.choice)
        if self.response_time is None:
            return choice
        return f"{_render_column(self.response_time)} | {choice}"

    def __str__(self) -> str:
        return self.render()


# --------------------------------------------------------------------------------------
# The formula
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Formula:
    """A parsed formula: an optional response, an intercept flag, terms, and group terms."""

    terms: tuple[FormulaTerm, ...] = ()
    intercept: bool = True
    response: FormulaResponse | None = None
    groups: tuple[GroupTerm, ...] = ()
    source: str = field(default="", compare=False, repr=False)

    @classmethod
    def parse(cls, text: str) -> Formula:
        """Parse ``text`` into a formula, or raise :class:`FormulaSyntaxError`."""

        if not isinstance(text, str):
            raise TypeError("a formula must be a string")
        return _Parser(text).parse()

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Every study column the formula names, in declaration order."""

        return tuple(dict.fromkeys(column for column, _ in self._references()))

    @property
    def data_dependent_terms(self) -> tuple[FormulaTerm, ...]:
        """Terms that estimate their coordinate from rows and so need a training study."""

        terms = [term for term in self.terms if term.data_dependent]
        terms.extend(term for group in self.groups for term in group.terms if term.data_dependent)
        return tuple(terms)

    def render(self) -> str:
        """Return the canonical spelling, with ``*`` expanded and the intercept explicit."""

        parts = ["1" if self.intercept else "0"]
        parts.extend(term.render() for term in self.terms)
        parts.extend(group.render() for group in self.groups)
        body = " + ".join(parts)
        if self.response is None:
            return body
        return f"{self.response.render()} ~ {body}"

    def __str__(self) -> str:
        return self.render()

    def validate(self, study: Study) -> None:
        """Check every referenced column against ``study``, naming the first typo found."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        known = set(study.columns)
        for column, position in self._references():
            if column in known:
                continue
            close = difflib.get_close_matches(column, sorted(known), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise FormulaError(
                f"study has no column {column!r}{hint}", source=self.source, position=position
            )

    def to_design(self) -> DesignSpec:
        """Build the fixed design a fully determined formula declares.

        Raises :class:`FormulaError` if any term would have to estimate its coordinate from
        rows. Those terms are reachable only through :meth:`fit`, whose signature names the
        training study.
        """

        dependent = self.data_dependent_terms
        if dependent:
            term = dependent[0]
            raise FormulaError(
                f"{term.render()} estimates its coordinate from study rows, so it cannot be "
                "built without naming a training fold; call Formula.fit(training_study) "
                "instead, or declare the estimate in the formula",
                source=self.source,
                position=term.position,
            )
        return self._build(None)

    def fit(self, study: Study) -> DesignSpec:
        """Freeze the formula against one **training** study and return a fixed design.

        ``study`` must be the training rows of the fold the design will be used in, exactly
        as for :meth:`behavio.design.StandardizeTerm.fit`. Everything data dependent --
        standardisation centres and scales, inferred category level sets -- is estimated
        here and frozen, so the returned :class:`~behavio.design.DesignSpec` builds an
        identical coordinate on held-out rows.
        """

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        self.validate(study)
        return self._build(study)

    def fixed_design(self, study: Study | None = None) -> DesignSpec:
        """Return the fixed part of this formula, deliberately without its group terms.

        This is the accessor :func:`behavio.compose.model_from_formula` uses: it has
        already read :attr:`groups` and is about to honour them, so dropping them here is a
        split rather than a loss. Nothing else should call it -- a caller that wanted the
        whole formula and got this would be fitting a model the formula does not describe,
        which is why :meth:`to_design` refuses instead of quietly returning it.
        """

        if study is None:
            dependent = self.data_dependent_terms
            if dependent:
                term = dependent[0]
                raise FormulaError(
                    f"{term.render()} estimates its coordinate from study rows, so it "
                    "cannot be built without naming a training fold",
                    source=self.source,
                    position=term.position,
                )
            return self._build(None, groups="ignore")
        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        self.validate(study)
        return self._build(study, groups="ignore")

    def _build(self, study: Study | None, *, groups: str = "reject") -> DesignSpec:
        if self.groups and groups == "reject":
            group = self.groups[0]
            raise FormulaError(
                f"the group term {group.render()} declares an effect that varies by "
                f"{group.grouping!r}, and a DesignSpec is one fixed matrix with nowhere to "
                "put it. Call behavio.compose.model_from_formula(formula, model) to build "
                "the hierarchical model this formula describes, or Formula.fixed_design() "
                "for the fixed part alone",
                source=self.source,
                position=group.position,
            )
        resolved: list[DesignTerm] = []
        for term in self.terms:
            resolved.append(self._resolve(term, study))
        try:
            spec = DesignSpec(terms=tuple(resolved), intercept=self.intercept)
            # Touched here so a coordinate the formula cannot label fails at declaration
            # rather than at the first build.
            _ = spec.feature_names
        except (DesignValidationError, TypeError) as error:
            raise FormulaError(str(error), source=self.source, position=0) from error
        return spec

    def _resolve(self, term: FormulaTerm, study: Study | None) -> DesignTerm:
        try:
            return term.resolve(study)
        except (ValueError, TypeError, KeyError) as error:
            detail = error.detail if isinstance(error, FormulaError) else str(error)
            position = term.position
            if isinstance(error, FormulaError) and error.position is not None:
                position = error.position
            raise FormulaError(detail, source=self.source, position=position) from error

    def _references(self) -> Iterator[tuple[str, int]]:
        if self.response is not None:
            for column in self.response.outcome_columns:
                yield column, self.response.position
        for term in self.terms:
            for column in term.required_columns:
                yield column, term.position
        for group in self.groups:
            yield group.grouping, group.position
            for term in group.terms:
                for column in term.required_columns:
                    yield column, term.position


# --------------------------------------------------------------------------------------
# Describing a hand-built design
# --------------------------------------------------------------------------------------


def describe_term(term: DesignTerm) -> str | None:
    """Return the formula spelling of one fixed design term, or ``None`` if it has none."""

    if isinstance(term, NumericTerm):
        return ColumnFormulaTerm(
            column=term.column, name=term.name, center=term.center, scale=term.scale
        ).render()
    if isinstance(term, CategoricalTerm):
        return CategoricalFormulaTerm(
            column=term.column,
            levels=term.levels,
            reference=term.reference,
            name=term.name,
            drop_reference=term.drop_reference,
        ).render()
    if isinstance(term, HistoryKernelTerm):
        return KernelFormulaTerm(
            column=term.column,
            weights=term.weights,
            lags=term.lags,
            reset_by=term.reset_by,
            coding=term.coding,
            fill_value=term.fill_value,
            name=term.name,
        ).render()
    if isinstance(term, HistoryTerm):
        return LagFormulaTerm(
            column=term.column,
            lags=term.lags,
            reset_by=term.reset_by,
            coding=term.coding,
            fill_value=term.fill_value,
            name=term.name,
        ).render()
    if isinstance(term, InteractionTerm):
        left = describe_term(term.left)
        right = describe_term(term.right)
        if left is None or right is None:
            return None
        return f"{left}:{right}"
    return None


def describe_design(spec: DesignSpec) -> str:
    """Render a hand-built :class:`~behavio.design.DesignSpec` as a canonical formula.

    The result re-parses: ``DesignSpec.from_formula(describe_design(spec))`` reproduces
    ``spec``. This is what makes a design signature a fingerprint you can read back rather
    than only compare, which a formatted repr is not.
    """

    if not isinstance(spec, DesignSpec):
        raise TypeError("spec must be a DesignSpec")
    parts = ["1" if spec.intercept else "0"]
    for term in spec.terms:
        rendered = describe_term(term)
        if rendered is None:
            raise FormulaError(
                f"{type(term).__name__} has no formula spelling; its design signature is "
                f"{term.signature}"
            )
        parts.append(rendered)
    return " + ".join(parts)


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Part:
    """One parsed fragment: zero or more terms, zero or more groups, an intercept flag."""

    terms: tuple[FormulaTerm, ...] = ()
    groups: tuple[GroupTerm, ...] = ()
    intercept: bool | None = None


class _Parser:
    """A recursive-descent parser over the grammar documented at module level."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._tokens = _tokenize(source)
        self._index = 0

    # -- token helpers ---------------------------------------------------------------

    def _peek(self, offset: int = 0) -> _Token:
        index = min(self._index + offset, len(self._tokens) - 1)
        return self._tokens[index]

    def _next(self) -> _Token:
        token = self._peek()
        if token.kind != "end":
            self._index += 1
        return token

    def _at_op(self, text: str) -> bool:
        token = self._peek()
        return token.kind == "op" and token.text == text

    def _error(self, detail: str, position: int) -> FormulaSyntaxError:
        return FormulaSyntaxError(detail, source=self._source, position=position)

    def _expect_op(self, text: str, detail: str) -> _Token:
        token = self._peek()
        if not self._at_op(text):
            raise self._error(f"{detail}; found {_describe_token(token)}", token.position)
        return self._next()

    # -- entry point -----------------------------------------------------------------

    def parse(self) -> Formula:
        response = self._parse_response()
        start = self._peek().position
        intercept, terms, groups = self._parse_sum()
        token = self._peek()
        if token.kind == "op" and token.text == "|":
            raise self._error(
                "a group term must be parenthesised, as in '(1 | subject)'", token.position
            )
        if token.kind != "end":
            raise self._error(
                f"unexpected {_describe_token(token)} after the formula", token.position
            )
        resolved = True if intercept is None else intercept
        if not terms and not groups and not resolved:
            raise self._error("a design must keep an intercept or at least one term", max(start, 0))
        return Formula(
            terms=terms,
            intercept=resolved,
            response=response,
            groups=groups,
            source=self._source,
        )

    def _parse_response(self) -> FormulaResponse | None:
        tildes = [
            index
            for index, token in enumerate(self._tokens)
            if token.kind == "op" and token.text == "~"
        ]
        if not tildes:
            return None
        if len(tildes) > 1:
            raise self._error("a formula has exactly one '~'", self._tokens[tildes[1]].position)
        head = self._tokens[: tildes[0]]
        self._index = tildes[0] + 1
        if not head:
            return None
        if head[0].kind != "name":
            raise self._error(
                f"a response must be a column name; found {_describe_token(head[0])}",
                head[0].position,
            )
        if len(head) == 1:
            return FormulaResponse(choice=head[0].value, position=head[0].position)
        expected = (
            len(head) == 3
            and head[1].kind == "op"
            and head[1].text == "|"
            and head[2].kind == "name"
        )
        if not expected:
            raise self._error(
                "a two-part response is spelled 'response_time | choice'", head[1].position
            )
        return FormulaResponse(
            choice=head[2].value, response_time=head[0].value, position=head[0].position
        )

    # -- grammar ---------------------------------------------------------------------

    def _parse_sum(self) -> tuple[bool | None, tuple[FormulaTerm, ...], tuple[GroupTerm, ...]]:
        intercept: bool | None = None
        terms: list[FormulaTerm] = []
        groups: list[GroupTerm] = []
        sign = "+"
        token = self._peek()
        if token.kind == "op" and token.text in "+-":
            self._next()
            sign = token.text
        while True:
            position = self._peek().position
            part = self._parse_product()
            if sign == "+":
                if part.intercept is not None:
                    intercept = part.intercept
                terms = _merge(terms, part.terms)
                groups = _merge_groups(groups, part.groups)
            else:
                if part.intercept is False:
                    raise self._error(
                        "'- 0' is not meaningful; write '0' to drop the intercept", position
                    )
                if part.intercept is True:
                    intercept = False
                if part.groups:
                    raise self._error("a group term cannot be subtracted", position)
                terms = _remove(terms, part.terms)
            token = self._peek()
            if token.kind == "op" and token.text in "+-":
                self._next()
                sign = token.text
                continue
            return intercept, tuple(terms), tuple(groups)

    def _parse_product(self) -> _Part:
        part = self._parse_interaction()
        while self._at_op("*"):
            token = self._next()
            right = self._parse_interaction()
            part = _cross(part, right, self._source, token.position)
        return part

    def _parse_interaction(self) -> _Part:
        part = self._parse_atom()
        while self._at_op(":"):
            token = self._next()
            right = self._parse_atom()
            part = _interact(part, right, self._source, token.position)
        return part

    def _parse_atom(self) -> _Part:
        token = self._peek()
        if token.kind == "op" and token.text == "(":
            return self._parse_parenthesised()
        if token.kind == "number":
            self._next()
            if token.value == 1:
                return _Part(intercept=True)
            if token.value == 0:
                return _Part(intercept=False)
            raise self._error(
                f"only '1' and '0' may stand alone as terms; found {token.text}", token.position
            )
        if token.kind == "name":
            following = self._peek(1)
            if not token.quoted and following.kind == "op" and following.text == "(":
                return self._parse_call()
            self._next()
            return _Part(terms=(ColumnFormulaTerm(column=token.value, position=token.position),))
        raise self._error(f"expected a term; found {_describe_token(token)}", token.position)

    def _parse_parenthesised(self) -> _Part:
        open_token = self._next()
        intercept, terms, groups = self._parse_sum()
        if self._at_op("|"):
            self._next()
            if groups:
                raise self._error("group terms cannot be nested", open_token.position)
            name = self._peek()
            if name.kind != "name":
                raise self._error(
                    f"a group term is grouped by a column name; found {_describe_token(name)}",
                    name.position,
                )
            self._next()
            self._expect_op(")", "a group term must be closed with ')'")
            return _Part(
                groups=(
                    GroupTerm(
                        grouping=name.value,
                        terms=terms,
                        intercept=True if intercept is None else intercept,
                        position=open_token.position,
                    ),
                )
            )
        self._expect_op(")", "expected ')' to close the group")
        return _Part(terms=terms, groups=groups, intercept=intercept)

    # -- calls -----------------------------------------------------------------------

    def _parse_call(self) -> _Part:
        name = self._next()
        self._next()
        if name.text not in FUNCTION_NAMES:
            supported = ", ".join(f"{function}()" for function in FUNCTION_NAMES)
            raise self._error(
                f"unknown formula function {name.text!r}; Behavio supports {supported}",
                name.position,
            )
        column_token = self._peek()
        if column_token.kind not in ("name", "string"):
            raise self._error(
                "the first argument of a formula function is a column name; "
                f"found {_describe_token(column_token)}",
                column_token.position,
            )
        self._next()
        positional: list[tuple[Any, int]] = []
        keywords: dict[str, tuple[Any, int]] = {}
        while self._at_op(","):
            self._next()
            if self._at_op(")"):
                break
            token = self._peek()
            following = self._peek(1)
            if (
                token.kind == "name"
                and not token.quoted
                and following.kind == "op"
                and following.text == "="
            ):
                self._next()
                self._next()
                if token.text in keywords:
                    raise self._error(f"duplicate argument {token.text!r}", token.position)
                keywords[token.text] = (self._parse_literal(), token.position)
                continue
            positional.append((self._parse_literal(), token.position))
        self._expect_op(")", f"expected ')' to close {name.text}()")
        term = _build_atom(
            name.text,
            column_token.value,
            positional,
            keywords,
            name.position,
            self._source,
        )
        return _Part(terms=(term,))

    def _parse_literal(self) -> Any:
        token = self._next()
        if token.kind == "number":
            return token.value
        if token.kind == "string":
            return token.value
        if token.kind == "op" and token.text == "-":
            following = self._next()
            if following.kind != "number":
                raise self._error(
                    f"expected a number after '-'; found {_describe_token(following)}",
                    following.position,
                )
            return -following.value
        if token.kind == "op" and token.text == "[":
            return self._parse_sequence()
        if token.kind == "name" and not token.quoted:
            if token.text == "True":
                return True
            if token.text == "False":
                return False
            if token.text == "None":
                return None
        raise self._error(
            f"expected a literal value; found {_describe_token(token)}", token.position
        )

    def _parse_sequence(self) -> list[Any]:
        items: list[Any] = []
        if self._at_op("]"):
            self._next()
            return items
        items.append(self._parse_literal())
        while self._at_op(","):
            self._next()
            if self._at_op("]"):
                break
            items.append(self._parse_literal())
        self._expect_op("]", "expected ']' to close the list")
        return items


def _describe_token(token: _Token) -> str:
    if token.kind == "end":
        return "the end of the formula"
    if token.kind == "string":
        return f"the string {token.text!r}"
    return f"{token.text!r}"


# --------------------------------------------------------------------------------------
# Term algebra used by the parser
# --------------------------------------------------------------------------------------


def _merge(terms: list[FormulaTerm], additions: Sequence[FormulaTerm]) -> list[FormulaTerm]:
    merged = list(terms)
    seen = {term.render() for term in merged}
    for term in additions:
        rendered = term.render()
        if rendered in seen:
            continue
        seen.add(rendered)
        merged.append(term)
    return merged


def _merge_groups(groups: list[GroupTerm], additions: Sequence[GroupTerm]) -> list[GroupTerm]:
    merged = list(groups)
    seen = {group.render() for group in merged}
    for group in additions:
        rendered = group.render()
        if rendered in seen:
            continue
        seen.add(rendered)
        merged.append(group)
    return merged


def _remove(terms: list[FormulaTerm], removals: Sequence[FormulaTerm]) -> list[FormulaTerm]:
    dropped = {term.render() for term in removals}
    return [term for term in terms if term.render() not in dropped]


def _factors(term: FormulaTerm) -> tuple[FormulaTerm, ...]:
    if isinstance(term, InteractionFormulaTerm):
        return term.factors
    return (term,)


def _product(left: FormulaTerm, right: FormulaTerm) -> FormulaTerm:
    return InteractionFormulaTerm(
        factors=(*_factors(left), *_factors(right)), position=left.position
    )


def _require_plain(part: _Part, operator: str, source: str, position: int) -> None:
    if part.intercept is not None:
        raise FormulaSyntaxError(
            f"'1' and '0' cannot take part in a '{operator}' expansion",
            source=source,
            position=position,
        )
    if part.groups:
        raise FormulaSyntaxError(
            f"a group term cannot take part in a '{operator}' expansion",
            source=source,
            position=position,
        )


def _interact(left: _Part, right: _Part, source: str, position: int) -> _Part:
    _require_plain(left, ":", source, position)
    _require_plain(right, ":", source, position)
    products = [_product(one, other) for one in left.terms for other in right.terms]
    return _Part(terms=tuple(_merge([], products)))


def _cross(left: _Part, right: _Part, source: str, position: int) -> _Part:
    _require_plain(left, "*", source, position)
    _require_plain(right, "*", source, position)
    terms = _merge(list(left.terms), right.terms)
    products = [_product(one, other) for one in left.terms for other in right.terms]
    return _Part(terms=tuple(_merge(terms, products)))


# --------------------------------------------------------------------------------------
# Call atoms
# --------------------------------------------------------------------------------------


def _build_atom(
    function: str,
    column: str,
    positional: list[tuple[Any, int]],
    keywords: dict[str, tuple[Any, int]],
    position: int,
    source: str,
) -> FormulaTerm:
    reader = _Arguments(function, positional, keywords, position, source)
    if function == "numeric":
        reader.reject_positional()
        term = ColumnFormulaTerm(
            column=column,
            name=reader.optional_string("name"),
            center=reader.number("center", 0.0),
            scale=reader.number("scale", 1.0),
            position=position,
        )
        reader.reject_unknown(("center", "scale", "name"))
        return term
    if function == "C":
        levels = reader.sequence_argument("levels", 0)
        term = CategoricalFormulaTerm(
            column=column,
            levels=None if levels is None else tuple(levels),
            reference=reader.literal("reference"),
            name=reader.optional_string("name"),
            drop_reference=reader.boolean("drop_reference", True),
            position=position,
        )
        reader.reject_unknown(("levels", "reference", "name", "drop_reference"))
        return term
    if function == "scale":
        reader.reject_positional()
        term = StandardizeFormulaTerm(
            column=column,
            ddof=reader.integer("ddof", 0),
            name=reader.optional_string("name"),
            position=position,
        )
        reader.reject_unknown(("ddof", "name"))
        return term
    if function == "lag":
        lags = reader.lags()
        term = LagFormulaTerm(
            column=column,
            lags=lags,
            reset_by=reader.columns("reset_by", _DEFAULT_RESET_BY),
            coding=reader.choice("coding", _DEFAULT_CODING, ("identity", "effect")),
            fill_value=reader.number("fill_value", 0.0),
            name=reader.optional_string("name"),
            coding_defaulted="coding" not in keywords,
            position=position,
        )
        reader.reject_unknown(("lags", "reset_by", "coding", "fill_value", "name"))
        return term
    weights = reader.sequence_argument("weights", 0)
    if weights is None:
        raise FormulaSyntaxError(
            "kernel() needs its fixed weights, as in kernel(choice, [0.6, 0.3, 0.1])",
            source=source,
            position=position,
        )
    lag_values = reader.sequence_argument("lags", 1)
    term = KernelFormulaTerm(
        column=column,
        weights=tuple(_as_number(value, "kernel weight", position, source) for value in weights),
        lags=(
            None
            if lag_values is None
            else tuple(_as_lag(value, position, source) for value in lag_values)
        ),
        reset_by=reader.columns("reset_by", _DEFAULT_RESET_BY),
        coding=reader.choice("coding", _DEFAULT_CODING, ("identity", "effect")),
        fill_value=reader.number("fill_value", 0.0),
        name=reader.optional_string("name"),
        coding_defaulted="coding" not in keywords,
        position=position,
    )
    reader.reject_unknown(("weights", "lags", "reset_by", "coding", "fill_value", "name"))
    return term


class _Arguments:
    """Typed reads of one call's positional and keyword arguments, with positions."""

    def __init__(
        self,
        function: str,
        positional: list[tuple[Any, int]],
        keywords: dict[str, tuple[Any, int]],
        position: int,
        source: str,
    ) -> None:
        self._function = function
        self._positional = positional
        self._keywords = keywords
        self._position = position
        self._source = source
        self._claimed: set[int] = set()

    def _fail(self, detail: str, position: int | None = None) -> FormulaSyntaxError:
        return FormulaSyntaxError(
            f"{self._function}(): {detail}",
            source=self._source,
            position=self._position if position is None else position,
        )

    def reject_positional(self) -> None:
        if self._positional:
            raise self._fail(
                "takes no positional argument after the column", self._positional[0][1]
            )

    def reject_unknown(self, allowed: Sequence[str]) -> None:
        unknown = [name for name in self._keywords if name not in allowed]
        if unknown:
            supported = ", ".join(allowed)
            raise self._fail(
                f"unknown argument {unknown[0]!r}; supported arguments are {supported}",
                self._keywords[unknown[0]][1],
            )
        extra = [index for index in range(len(self._positional)) if index not in self._claimed]
        if extra:
            raise self._fail("too many positional arguments", self._positional[extra[0]][1])

    def _entry(self, name: str, slot: int | None) -> tuple[Any, int] | None:
        keyword = self._keywords.get(name)
        by_slot = None
        if slot is not None and slot < len(self._positional):
            by_slot = self._positional[slot]
            self._claimed.add(slot)
        if keyword is not None and by_slot is not None:
            raise self._fail(f"{name!r} was given both positionally and by name", keyword[1])
        return keyword if keyword is not None else by_slot

    def literal(self, name: str) -> Any:
        entry = self._keywords.get(name)
        return None if entry is None else entry[0]

    def number(self, name: str, default: float) -> float:
        entry = self._keywords.get(name)
        if entry is None:
            return default
        return _as_number(entry[0], name, entry[1], self._source)

    def integer(self, name: str, default: int) -> int:
        entry = self._keywords.get(name)
        if entry is None:
            return default
        value = entry[0]
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._fail(f"{name!r} must be an integer", entry[1])
        return value

    def boolean(self, name: str, default: bool) -> bool:
        entry = self._keywords.get(name)
        if entry is None:
            return default
        if not isinstance(entry[0], bool):
            raise self._fail(f"{name!r} must be True or False", entry[1])
        return entry[0]

    def optional_string(self, name: str) -> str | None:
        entry = self._keywords.get(name)
        if entry is None:
            return None
        if not isinstance(entry[0], str) or not entry[0]:
            raise self._fail(f"{name!r} must be a non-empty string", entry[1])
        return entry[0]

    def choice(self, name: str, default: str, allowed: Sequence[str]) -> str:
        entry = self._keywords.get(name)
        if entry is None:
            return default
        if entry[0] not in allowed:
            options = ", ".join(repr(option) for option in allowed)
            raise self._fail(f"{name!r} must be one of {options}", entry[1])
        return str(entry[0])

    def columns(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        entry = self._keywords.get(name)
        if entry is None:
            return default
        values = entry[0] if isinstance(entry[0], list) else [entry[0]]
        for value in values:
            if not isinstance(value, str) or not value:
                raise self._fail(f"{name!r} must name study columns", entry[1])
        return tuple(values)

    def sequence_argument(self, name: str, slot: int | None) -> list[Any] | None:
        entry = self._entry(name, slot)
        if entry is None:
            return None
        if not isinstance(entry[0], list):
            raise self._fail(f"{name!r} must be a list, as in {name}=[1, 2]", entry[1])
        return entry[0]

    def lags(self) -> tuple[int, ...]:
        keyword = self._keywords.get("lags")
        if keyword is not None and self._positional:
            raise self._fail("lags were given both positionally and as 'lags'", keyword[1])
        if keyword is not None:
            values = keyword[0] if isinstance(keyword[0], list) else [keyword[0]]
            return tuple(_as_lag(value, keyword[1], self._source) for value in values)
        if not self._positional:
            return (1,)
        first = self._positional[0]
        if isinstance(first[0], list):
            self._claimed.add(0)
            if len(self._positional) > 1:
                raise self._fail("give lags either as a list or one by one", self._positional[1][1])
            return tuple(_as_lag(value, first[1], self._source) for value in first[0])
        for index in range(len(self._positional)):
            self._claimed.add(index)
        return tuple(_as_lag(value, place, self._source) for value, place in self._positional)


def _as_number(value: Any, label: str, position: int, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormulaSyntaxError(f"{label} must be a number", source=source, position=position)
    return float(value)


def _as_lag(value: Any, position: int, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FormulaSyntaxError(
            "a lag must be a positive integer", source=source, position=position
        )
    return value


def _observed_levels(study: Study, column: str, position: int) -> tuple[Any, ...]:
    if column not in study.columns:
        raise FormulaError(f"study is missing categorical column {column!r}", position=position)
    seen: dict[Any, Any] = {}
    for raw in study[column]:
        value = raw.item() if isinstance(raw, np.generic) else raw
        seen.setdefault(value, value)
    values = list(seen.values())
    if len(values) < 2:
        raise FormulaError(
            f"C({column}) needs at least two categories in the training study; "
            f"it has {len(values)}",
            position=position,
        )
    try:
        return tuple(sorted(values))
    except TypeError:
        raise FormulaError(
            f"C({column}) cannot order its observed categories, so the reference level "
            f"would depend on row order; declare them, as in C({column}, ['a', 'b'])",
            position=position,
        ) from None
