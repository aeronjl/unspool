"""What a model will do to a study, answered before the study is fitted.

Two silent failures motivated this module, and both are failures of *timing* rather than of
checking. A mistyped covariate survives construction, survives :attr:`signature`, survives
serialization, and finally raises from deep inside ``fit`` with a traceback through the
optimizer. Knots placed outside the observed clock range never raise at all: the basis
happily interpolates towards a knot no observation stands near, and the fit reports its
value with the same standard error it reports for a knot the data surrounds.

:class:`ModelDescription` answers the question a user actually has before fitting -- which
columns will be built, what the parameters are called and where they are bounded, what
priors are in force, which study columns are required -- and reports both failures as
:class:`ModelFinding` records against a specific study.

:class:`Describable` gives an estimator ``describe(study)`` and ``validate(study)``.
:meth:`~behavio.task.TaskSpec.validate` has existed for a long time and models have had no
counterpart, which is precisely why these two defects were discovered at fit time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

import numpy as np

from behavio.contracts.estimator import ModelDataError
from behavio.design import (
    DesignSpec,
    DesignValidationError,
    HistoryKernelTerm,
    HistoryTerm,
    InteractionTerm,
)
from behavio.models._kernels.basis import knot_support_findings
from behavio.study import Study

ERROR = "error"
"""A finding that will make ``fit`` raise."""

WARNING = "warning"
"""A finding that will not stop the fit but makes its result harder to defend."""


@dataclass(frozen=True, slots=True)
class ModelFinding:
    """One named problem with fitting a particular model to a particular study."""

    code: str
    severity: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in {ERROR, WARNING}:
            raise ValueError("finding severity must be 'error' or 'warning'")

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class ModelDescription:
    """Everything a user needs before calling ``fit``, plus what is already wrong."""

    model_name: str
    signature: str
    n_observations: int | None
    design_columns: tuple[str, ...]
    source_columns: tuple[str, ...]
    scored_columns: tuple[str, ...]
    required_task_columns: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    priors: tuple[str, ...] = ()
    findings: tuple[ModelFinding, ...] = ()
    design_column_notes: Mapping[str, str] = field(default_factory=dict)
    """What a design column is, where the name alone does not say.

    ``choice_lag_1`` is the name of an effect-coded lag and of an identity-coded one, and
    the two are different columns: -1/+1 against 0/1, with coefficients on different
    scales. The name cannot be changed -- it is baked into fitted artefacts and committed
    benchmark results -- so the coding is reported alongside it instead, and two fits that
    both report ``choice_lag_1`` can be told apart by reading their descriptions.
    """

    @property
    def errors(self) -> tuple[ModelFinding, ...]:
        """Findings that will make ``fit`` raise."""

        return tuple(finding for finding in self.findings if finding.severity == ERROR)

    @property
    def warnings(self) -> tuple[ModelFinding, ...]:
        """Findings that will not stop the fit but weaken what it can support."""

        return tuple(finding for finding in self.findings if finding.severity == WARNING)

    def __str__(self) -> str:
        lines = [f"{self.model_name}", f"  signature: {self.signature}"]
        if self.n_observations is not None:
            lines.append(f"  study rows: {self.n_observations}")
        lines.append(f"  design columns ({len(self.design_columns)}):")
        for name in self.design_columns:
            note = self.design_column_notes.get(name)
            lines.append(f"    {name}" if note is None else f"    {name}  {note}")
        lines.append(f"  parameters ({len(self.parameter_names)}):")
        for name in self.parameter_names:
            bound = self.parameter_bounds.get(name)
            suffix = "" if bound is None else f"  in [{bound[0]:g}, {bound[1]:g}]"
            lines.append(f"    {name}{suffix}")
        lines.append("  priors:")
        lines.extend(f"    {prior}" for prior in self.priors or ("none declared",))
        lines.append("  required study columns:")
        lines.append(f"    scored: {', '.join(self.scored_columns) or 'none'}")
        lines.append(f"    task:   {', '.join(self.required_task_columns) or 'none'}")
        lines.append(f"    design: {', '.join(self.source_columns) or 'none'}")
        lines.append("  findings:")
        lines.extend(f"    {finding}" for finding in self.findings)
        if not self.findings:
            lines.append("    none")
        return "\n".join(lines)


class Describable:
    """Mixin giving an estimator pre-fit introspection and validation.

    ``__slots__`` is empty so that frozen, slotted model dataclasses can inherit it
    without gaining an instance dictionary.
    """

    __slots__ = ()

    def describe(self, study: Study | None = None) -> ModelDescription:
        """Return what this model will build, and what is wrong with doing so here."""

        return describe_model(self, study)

    def validate(self, study: Study) -> ModelDescription:
        """Raise :class:`ModelDataError` if ``fit`` would fail; otherwise describe.

        Warnings are returned rather than raised. A knot outside the data's support is a
        modelling decision that only its author can judge, but it should never be made by
        accident, so it is reported and the caller decides.
        """

        description = describe_model(self, study)
        if description.errors:
            raise ModelDataError(
                f"{description.model_name} cannot fit this study: "
                + "; ".join(finding.message for finding in description.errors)
            )
        return description


def describe_model(model: Any, study: Study | None = None) -> ModelDescription:
    """Build a :class:`ModelDescription` from an estimator and an optional study."""

    if study is not None and not isinstance(study, Study):
        raise TypeError("study must be a Study")
    design: DesignSpec | None = getattr(model, "design_spec", None)
    design_columns = _design_columns(model, design)
    source_columns = () if design is None else design.required_columns
    scored = tuple(getattr(model, "scored_columns", ()))
    required = tuple(getattr(model, "required_task_columns", ()))
    parameter_names = tuple(getattr(model, "parameter_names", ()))
    findings: list[ModelFinding] = []
    if study is not None:
        findings.extend(_missing_column_findings(study, design, scored, required))
        findings.extend(_column_type_findings(study, source_columns))
        findings.extend(_knot_findings(model, study))
    return ModelDescription(
        model_name=str(getattr(model, "model_name", type(model).__name__)),
        signature=str(getattr(model, "signature", "")),
        n_observations=None if study is None else len(study),
        design_columns=design_columns,
        source_columns=source_columns,
        scored_columns=scored,
        required_task_columns=required,
        parameter_names=parameter_names,
        parameter_bounds=_parameter_bounds(model, parameter_names),
        priors=tuple(getattr(model, "declared_priors", ())),
        findings=tuple(findings),
        design_column_notes=_design_column_notes(design),
    )


def _design_column_notes(design: DesignSpec | None) -> dict[str, str]:
    """Annotate every history column with the coding that decides what its values are."""

    notes: dict[str, str] = {}
    if design is None:
        return notes
    for term in design.terms:
        try:
            notes.update(_history_notes(term))
        except DesignValidationError:  # pragma: no cover - reported as a design column
            continue
    return notes


def _history_notes(term: Any) -> dict[str, str]:
    """Map this term's own feature names to a coding note, for history terms only.

    An interaction is followed into, because ``stimulus:choice_lag_1`` is as ambiguous as
    ``choice_lag_1`` is, and the coding that resolves it belongs to a factor rather than
    to the product.
    """

    if isinstance(term, (HistoryTerm, HistoryKernelTerm)):
        return {name: f"[coding={term.coding}]" for name in term.feature_names}
    if not isinstance(term, InteractionTerm):
        return {}
    left_notes = _history_notes(term.left)
    right_notes = _history_notes(term.right)
    notes: dict[str, str] = {}
    for left_name in term.left.feature_names:
        for right_name in term.right.feature_names:
            found = dict.fromkeys(
                note
                for note in (left_notes.get(left_name), right_notes.get(right_name))
                if note is not None
            )
            if found:
                notes[f"{left_name}:{right_name}"] = " ".join(found)
    return notes


def _design_columns(model: Any, design: DesignSpec | None) -> tuple[str, ...]:
    if design is None:
        return tuple(getattr(model, "coefficient_names", ()))
    try:
        return design.feature_names
    except DesignValidationError as error:
        return (f"<invalid design: {error}>",)


def _missing_column_findings(
    study: Study,
    design: DesignSpec | None,
    scored: tuple[str, ...],
    required: tuple[str, ...],
) -> list[ModelFinding]:
    wanted: list[tuple[str, str]] = []
    if design is not None:
        wanted.extend((column, "design") for column in design.required_columns)
    wanted.extend((column, "scored") for column in scored)
    wanted.extend((column, "task") for column in required)
    seen: set[str] = set()
    findings: list[ModelFinding] = []
    available = sorted(study.columns)
    for column, role in wanted:
        if column in seen or column in study.columns:
            seen.add(column)
            continue
        seen.add(column)
        suggestion = get_close_matches(column, available, n=1, cutoff=0.6)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        findings.append(
            ModelFinding(
                code="missing_column",
                severity=ERROR,
                message=f"study has no {role} column {column!r}{hint}",
            )
        )
    return findings


def _column_type_findings(study: Study, columns: Sequence[str]) -> list[ModelFinding]:
    findings: list[ModelFinding] = []
    for column in columns:
        if column not in study.columns:
            continue
        try:
            values = np.asarray(study[column], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if values.ndim == 1 and not np.all(np.isfinite(values)):
            findings.append(
                ModelFinding(
                    code="nonfinite_column",
                    severity=ERROR,
                    message=f"design column {column!r} contains non-finite values",
                )
            )
    return findings


def _knot_findings(model: Any, study: Study) -> list[ModelFinding]:
    knots = getattr(model, "knots", None)
    clock = getattr(model, "time", None)
    if not knots or not isinstance(clock, str):
        return []
    if clock not in study.columns:
        return [
            ModelFinding(
                code="missing_column",
                severity=ERROR,
                message=f"study has no clock column {clock!r}",
            )
        ]
    try:
        times = np.asarray(study[clock], dtype=np.float64)
    except (TypeError, ValueError):
        return [
            ModelFinding(
                code="nonnumeric_clock",
                severity=ERROR,
                message=f"clock column {clock!r} is not numeric",
            )
        ]
    times = times[np.isfinite(times)]
    findings = [
        ModelFinding(code="unsupported_knot", severity=WARNING, message=message)
        for message in knot_support_findings(tuple(knots), times, clock)
    ]
    if len(times) and (float(np.min(times)) < knots[0] or float(np.max(times)) > knots[-1]):
        findings.append(
            ModelFinding(
                code="clock_outside_knots",
                severity=ERROR,
                message=(
                    f"observed {clock} values fall outside the fixed knot range "
                    f"[{knots[0]}, {knots[-1]}]"
                ),
            )
        )
    return findings


def _parameter_bounds(model: Any, names: tuple[str, ...]) -> Mapping[str, tuple[float, float]]:
    declared = getattr(model, "parameter_bounds", None)
    if not declared:
        return {}
    if isinstance(declared, Mapping):
        return {name: (float(low), float(high)) for name, (low, high) in declared.items()}
    return {
        name: (float(low), float(high)) for name, (low, high) in zip(names, declared, strict=False)
    }
