"""Fixed, composable design-matrix terms with explicit history boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from behavio._internal.arrays import protected_array
from behavio.study import Study


class DesignValidationError(ValueError):
    """Raised when a fixed design specification cannot represent a study."""


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """One or more aligned, named columns produced by a design term."""

    names: tuple[str, ...]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        values = protected_array(self.values, dtype=np.float64)
        if not names or len(set(names)) != len(names):
            raise DesignValidationError("feature names must be non-empty and unique")
        if any(not isinstance(name, str) or not name for name in names):
            raise DesignValidationError("feature names must be non-empty strings")
        if values.ndim != 2 or values.shape[1] != len(names):
            raise DesignValidationError("feature values must have one column per name")
        if not np.all(np.isfinite(values)):
            raise DesignValidationError("feature values must be finite")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)


class DesignTerm(Protocol):
    """A fixed transformation from a Study into named numeric features."""

    @property
    def signature(self) -> str: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def required_columns(self) -> tuple[str, ...]: ...

    def build(self, study: Study) -> FeatureBlock: ...


@dataclass(frozen=True, slots=True)
class NumericTerm:
    """One numeric source column under a fixed affine transformation."""

    column: str
    name: str | None = None
    center: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        _name(self.column, "numeric column")
        if self.name is not None:
            _name(self.name, "numeric feature name")
        if not np.isfinite(self.center):
            raise DesignValidationError("numeric center must be finite")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise DesignValidationError("numeric scale must be finite and positive")

    @property
    def feature_name(self) -> str:
        return self.column if self.name is None else self.name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (self.feature_name,)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    @property
    def signature(self) -> str:
        return (
            f"numeric(column={self.column!r},name={self.feature_name!r},"
            f"center={self.center!r},scale={self.scale!r})"
        )

    def build(self, study: Study) -> FeatureBlock:
        values = _numeric_column(study, self.column)
        transformed = ((values - self.center) / self.scale)[:, None]
        return FeatureBlock(names=(self.feature_name,), values=transformed)


@dataclass(frozen=True, slots=True)
class StandardizeTerm:
    """Training-only estimator that freezes into a fixed :class:`NumericTerm`."""

    column: str
    name: str | None = None
    ddof: int = 0

    def __post_init__(self) -> None:
        _name(self.column, "standardized column")
        if self.name is not None:
            _name(self.name, "standardized feature name")
        if isinstance(self.ddof, bool) or not isinstance(self.ddof, int) or self.ddof < 0:
            raise DesignValidationError("standardization ddof must be a non-negative integer")

    def fit(self, study: Study) -> NumericTerm:
        """Estimate centre and scale from one training study and return a fixed term."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        values = _numeric_column(study, self.column)
        if self.ddof >= len(values):
            raise DesignValidationError("standardization ddof must be smaller than training rows")
        center = float(np.mean(values))
        scale = float(np.std(values, ddof=self.ddof))
        if not np.isfinite(scale) or scale <= 0:
            raise DesignValidationError("cannot standardize a constant training column")
        return NumericTerm(column=self.column, name=self.name, center=center, scale=scale)


@dataclass(frozen=True, slots=True)
class CategoricalTerm:
    """Fixed-level treatment coding that cannot learn categories from test data."""

    column: str
    levels: tuple[Any, ...]
    reference: Any | None = None
    name: str | None = None
    drop_reference: bool = True

    def __post_init__(self) -> None:
        _name(self.column, "categorical column")
        if self.name is not None:
            _name(self.name, "categorical feature name")
        levels = _levels(self.levels)
        reference = levels[0] if self.reference is None else _scalar(self.reference)
        if _label_key(reference) not in {_label_key(level) for level in levels}:
            raise DesignValidationError("categorical reference must be one of the fixed levels")
        if not isinstance(self.drop_reference, bool):
            raise DesignValidationError("drop_reference must be boolean")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "reference", reference)

    @property
    def prefix(self) -> str:
        return self.column if self.name is None else self.name

    @property
    def encoded_levels(self) -> tuple[Any, ...]:
        if not self.drop_reference:
            return self.levels
        reference_key = _label_key(self.reference)
        return tuple(level for level in self.levels if _label_key(level) != reference_key)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}[{level!r}]" for level in self.encoded_levels)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    @property
    def signature(self) -> str:
        return (
            f"categorical(column={self.column!r},levels={self.levels!r},"
            f"reference={self.reference!r},name={self.prefix!r},"
            f"drop_reference={self.drop_reference!r})"
        )

    def build(self, study: Study) -> FeatureBlock:
        if self.column not in study.columns:
            raise DesignValidationError(f"study is missing categorical column {self.column!r}")
        level_index = {_label_key(level): level for level in self.levels}
        observed = tuple(_scalar(value) for value in study[self.column])
        for row, value in enumerate(observed):
            if _is_missing(value) or _label_key(value) not in level_index:
                raise DesignValidationError(
                    f"categorical value at row {row} is outside the fixed levels: {value!r}"
                )
        encoded = self.encoded_levels
        values = np.column_stack(
            [
                np.asarray(
                    [_label_key(value) == _label_key(level) for value in observed],
                    dtype=np.float64,
                )
                for level in encoded
            ]
        )
        return FeatureBlock(names=self.feature_names, values=values)


@dataclass(frozen=True, slots=True)
class HistoryTerm:
    """Filtered lagged values with explicit reset columns and fixed coding."""

    column: str
    lags: tuple[int, ...] = (1,)
    reset_by: tuple[str, ...] = ("subject", "session")
    coding: str = "identity"
    fill_value: float = 0.0
    name: str | None = None
    coding_hint: str | None = field(default=None, compare=False, repr=False)
    """How to ask for identity coding, when ``coding`` is a default the user did not write.

    Only :mod:`behavio.formula` sets it, because only there is ``coding="effect"`` chosen
    on the user's behalf: ``lag()`` and ``kernel()`` default to the -1/+1 coding that the
    ``choice_lags=`` shorthand has always built. A default that fails has to explain
    itself, and here the explanation is a formula, so the formula layer supplies its own
    spelling of the escape hatch rather than this module guessing at one.

    It is excluded from equality, from ``repr`` and from :attr:`signature`: it is
    provenance for one error message, not part of what the term builds. A term carrying a
    hint and the same term without one are the same term.
    """

    def __post_init__(self) -> None:
        _name(self.column, "history column")
        lags = tuple(self.lags)
        if not lags or len(set(lags)) != len(lags):
            raise DesignValidationError("history lags must be non-empty and unique")
        if any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 for lag in lags):
            raise DesignValidationError("history lags must be positive integers")
        if tuple(sorted(lags)) != lags:
            raise DesignValidationError("history lags must be in increasing order")
        reset_by = tuple(self.reset_by)
        if not reset_by or len(set(reset_by)) != len(reset_by):
            raise DesignValidationError("history reset columns must be non-empty and unique")
        for column in reset_by:
            _name(column, "history reset column")
        if self.coding not in {"identity", "effect"}:
            raise DesignValidationError("history coding must be 'identity' or 'effect'")
        if not np.isfinite(self.fill_value):
            raise DesignValidationError("history fill value must be finite")
        if self.name is not None:
            _name(self.name, "history feature name")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "reset_by", reset_by)

    @property
    def prefix(self) -> str:
        return self.column if self.name is None else self.name

    @property
    def signature(self) -> str:
        return (
            f"history(column={self.column!r},lags={self.lags!r},"
            f"reset_by={self.reset_by!r},coding={self.coding!r},"
            f"fill_value={self.fill_value!r},name={self.prefix!r})"
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}_lag_{lag}" for lag in self.lags)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.column, *self.reset_by)))

    def _effect_coding_failure(self) -> str:
        """Explain a failed effect coding, and say who chose it.

        A user who wrote ``coding="effect"`` has already been told what effect coding is
        and only needs to know that this column is not binary. A user who wrote
        ``lag(reward_magnitude, 1)`` chose nothing, so the message has to name the default
        before it can be acted on, and then name the spelling that opts out of it.
        """

        requirement = "effect-coded history requires zero/one values"
        if self.coding_hint is None:
            return requirement
        return (
            f"{requirement}, and {self.column!r} has others. A formula's lag() and kernel() "
            "default to coding='effect', the -1/+1 coding that the choice_lags= shorthand "
            f"has always built for a binary history column. Write {self.coding_hint} to lag "
            f"{self.column!r} literally instead."
        )

    def build(self, study: Study) -> FeatureBlock:
        missing = [
            column for column in (self.column, *self.reset_by) if column not in study.columns
        ]
        if missing:
            raise DesignValidationError(f"study is missing history columns: {missing}")
        source = _numeric_column(study, self.column)
        if self.coding == "effect":
            if np.any((source != 0.0) & (source != 1.0)):
                raise DesignValidationError(self._effect_coding_failure())
            source = 2.0 * source - 1.0

        values = np.full((len(study), len(self.lags)), self.fill_value, dtype=np.float64)
        groups: dict[tuple[Any, ...], list[float]] = {}
        for raw_index in study.chronological_indices():
            index = int(raw_index)
            key = tuple(_label_key(_scalar(study[column][index])) for column in self.reset_by)
            previous = groups.setdefault(key, [])
            for output, lag in enumerate(self.lags):
                if len(previous) >= lag:
                    values[index, output] = previous[-lag]
            previous.append(float(source[index]))
        return FeatureBlock(names=self.feature_names, values=values)


@dataclass(frozen=True, slots=True)
class HistoryKernelTerm:
    """One fixed weighted history feature over explicitly declared lags."""

    column: str
    weights: tuple[float, ...]
    lags: tuple[int, ...] | None = None
    reset_by: tuple[str, ...] = ("subject", "session")
    coding: str = "identity"
    fill_value: float = 0.0
    name: str | None = None
    coding_hint: str | None = field(default=None, compare=False, repr=False)
    """Passed straight through to the :class:`HistoryTerm` this term lags with."""

    def __post_init__(self) -> None:
        weights = tuple(float(weight) for weight in self.weights)
        if not weights or not np.all(np.isfinite(weights)):
            raise DesignValidationError("history-kernel weights must be non-empty and finite")
        lags = tuple(range(1, len(weights) + 1)) if self.lags is None else tuple(self.lags)
        if len(lags) != len(weights):
            raise DesignValidationError("history-kernel lags and weights must have equal length")
        history = HistoryTerm(
            column=self.column,
            lags=lags,
            reset_by=self.reset_by,
            coding=self.coding,
            fill_value=self.fill_value,
            name=self.name,
        )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "lags", history.lags)
        object.__setattr__(self, "reset_by", history.reset_by)

    @property
    def feature_name(self) -> str:
        return f"{self.column}_kernel" if self.name is None else self.name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (self.feature_name,)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.column, *self.reset_by)))

    @property
    def signature(self) -> str:
        return (
            f"history_kernel(column={self.column!r},lags={self.lags!r},"
            f"weights={self.weights!r},reset_by={self.reset_by!r},"
            f"coding={self.coding!r},fill_value={self.fill_value!r},"
            f"name={self.feature_name!r})"
        )

    def build(self, study: Study) -> FeatureBlock:
        history = HistoryTerm(
            column=self.column,
            lags=self.lags,
            reset_by=self.reset_by,
            coding=self.coding,
            fill_value=self.fill_value,
            name=self.feature_name,
            coding_hint=self.coding_hint,
        ).build(study)
        values = history.values @ np.asarray(self.weights, dtype=np.float64)
        return FeatureBlock(names=(self.feature_name,), values=values[:, None])


@dataclass(frozen=True, slots=True)
class InteractionTerm:
    """All pairwise products between two fixed design terms."""

    left: DesignTerm
    right: DesignTerm

    @property
    def signature(self) -> str:
        return f"interaction(left={self.left.signature},right={self.right.signature})"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            f"{left_name}:{right_name}"
            for left_name in self.left.feature_names
            for right_name in self.right.feature_names
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.left.required_columns, *self.right.required_columns)))

    def build(self, study: Study) -> FeatureBlock:
        left = self.left.build(study)
        right = self.right.build(study)
        # Broadcast multiplication rather than np.einsum("ij,ik->ijk", ...): einsum reduces
        # through a sum and so returns +0.0 where IEEE multiplication returns -0.0, which
        # makes an interaction column built here differ from the same column multiplied out
        # by hand. The values are equal as numbers, but Study columns are hashed into
        # FitArtifact provenance, so the sign of a zero decides whether two identical
        # studies share a fingerprint. The layout is the one einsum produced: feature j of
        # the left block varies slowest, feature k of the right block fastest.
        #
        # Follow-up, deliberately not done here: canonicalising signed zeros at the Study
        # boundary would fix the class rather than this instance, but it would move the
        # hash of every existing study that already contains a -0.0. That is a breaking
        # provenance change and needs an audit of the committed artefacts first.
        values = (left.values[:, :, None] * right.values[:, None, :]).reshape(len(study), -1)
        return FeatureBlock(names=self.feature_names, values=values)


@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """Immutable labelled numeric matrix produced by one fixed specification."""

    names: tuple[str, ...]
    values: NDArray[np.float64]
    specification: str

    def __post_init__(self) -> None:
        block = FeatureBlock(self.names, self.values)
        _name(self.specification, "design specification")
        object.__setattr__(self, "names", block.names)
        object.__setattr__(self, "values", block.values)


@dataclass(frozen=True, slots=True)
class DesignSpec:
    """A fixed, reusable collection of numeric, categorical, history, and interaction terms."""

    terms: tuple[DesignTerm, ...] = ()
    intercept: bool = True

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not isinstance(self.intercept, bool):
            raise DesignValidationError("intercept must be boolean")
        if not terms and not self.intercept:
            raise DesignValidationError("a design must contain an intercept or at least one term")
        for term in terms:
            if (
                not hasattr(term, "build")
                or not hasattr(term, "signature")
                or not hasattr(term, "feature_names")
                or not hasattr(term, "required_columns")
            ):
                raise TypeError(
                    "every design term must provide build(), signature, feature_names, "
                    "and required_columns"
                )
        object.__setattr__(self, "terms", terms)

    @classmethod
    def from_formula(cls, formula: str, *, training_study: Study | None = None) -> DesignSpec:
        """Build a fixed design from a formula string.

        ``formula`` uses the notation documented in :mod:`behavio.formula`, for example
        ``"choice ~ stimulus * phase + lag(choice, 1)"``. Everything it can say desugars
        onto the terms in this module; the response, if present, is checked but does not
        enter the matrix.

        ``training_study`` is required, and required to be *training* rows, whenever the
        formula contains a term that estimates its coordinate from data -- ``scale(x)``, or
        ``C(x)`` without a declared level set. Without it those formulas raise rather than
        quietly reading the whole study, so leakage stays harder to write than to avoid.
        """

        from behavio.formula import Formula

        parsed = Formula.parse(formula)
        if training_study is None:
            return parsed.to_design()
        return parsed.fit(training_study)

    def describe(self) -> str:
        """Return this design as a canonical formula string that parses back to it.

        Raises :class:`~behavio.formula.FormulaError` if a third-party term has no formula
        spelling. Use :attr:`signature` when an unparseable fingerprint is enough.
        """

        from behavio.formula import describe_design

        return describe_design(self)

    def __str__(self) -> str:
        from behavio.formula import describe_term

        parts = ["1" if self.intercept else "0"]
        parts.extend(describe_term(term) or term.signature for term in self.terms)
        return " + ".join(parts)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Stable output coordinate declared without inspecting study rows."""

        names = ("intercept",) if self.intercept else ()
        for term in self.terms:
            names += tuple(term.feature_names)
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise DesignValidationError(f"design feature names collide: {duplicates}")
        return names

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Source columns needed by every term, in first-declaration order."""

        return tuple(
            dict.fromkeys(column for term in self.terms for column in term.required_columns)
        )

    @property
    def signature(self) -> str:
        terms = ",".join(term.signature for term in self.terms)
        return f"design(intercept={self.intercept!r};terms={terms})"

    def build(self, study: Study) -> DesignMatrix:
        """Construct a labelled matrix without learning anything from its rows."""

        if not isinstance(study, Study):
            raise TypeError("study must be a Study")
        blocks = [term.build(study) for term in self.terms]
        names = self.feature_names
        matrices: list[NDArray[np.float64]] = []
        if self.intercept:
            matrices.append(np.ones((len(study), 1), dtype=np.float64))
        for term, block in zip(self.terms, blocks, strict=True):
            if block.names != tuple(term.feature_names):
                raise DesignValidationError(
                    "design term built names that differ from its declared feature_names"
                )
            matrices.append(block.values)
        values = np.column_stack(matrices)
        return DesignMatrix(names=names, values=values, specification=self.signature)


def _numeric_column(study: Study, column: str) -> NDArray[np.float64]:
    if column not in study.columns:
        raise DesignValidationError(f"study is missing numeric column {column!r}")
    try:
        values = np.asarray(study[column], dtype=np.float64)
    except (TypeError, ValueError):
        raise DesignValidationError(f"numeric column {column!r} must be numeric") from None
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise DesignValidationError(f"numeric column {column!r} must be finite")
    return values


def _levels(values: tuple[Any, ...]) -> tuple[Any, ...]:
    if not isinstance(values, tuple) or len(values) < 2:
        raise DesignValidationError("categorical levels must be a tuple with at least two values")
    levels = tuple(_scalar(value) for value in values)
    if any(_is_missing(value) for value in levels):
        raise DesignValidationError("categorical levels cannot contain missing values")
    try:
        keys = tuple(_label_key(value) for value in levels)
        unique = set(keys)
    except TypeError:
        raise DesignValidationError("categorical levels must be scalar and hashable") from None
    if len(unique) != len(levels):
        raise DesignValidationError("categorical levels must be unique")
    return levels


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise DesignValidationError(f"{label} must be a non-empty string")


def _label_key(value: Any) -> tuple[Any, Any]:
    if isinstance(value, bool):
        return bool, value
    if isinstance(value, (int, float)):
        return "number", float(value)
    return type(value), value


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False
