"""A runnable conformance harness for :class:`behavio.contracts.adapter.StudyAdapter`.

``docs/extensions.md`` lists nine things an estimator extension should test. Data-source
adapters had no equivalent, so the conventions they must honour -- preserve trial order,
preserve subject and session boundaries, never fabricate ``session_order`` -- were prose an
author could not execute. :func:`check_study_adapter` executes them.

The harness deliberately lives in ``behavio.adapters`` rather than ``behavio.contracts``:
``behavio.contracts`` is a runtime leaf that declares protocols, and a harness that reads
real sources is a testing tool, not part of the contract's type surface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from behavio.contracts.adapter import AdapterCapabilities, adapter_capabilities
from behavio.trials import Study


class AdapterConformanceError(AssertionError):
    """Raised by :func:`assert_study_adapter_conforms` when a required check fails."""


class CheckStatus(StrEnum):
    """Outcome of one conformance check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    """One named conformance check and what it observed."""

    name: str
    status: CheckStatus
    detail: str

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("a conformance check needs a non-empty name")
        object.__setattr__(self, "status", CheckStatus(self.status))


@dataclass(frozen=True, slots=True)
class AdapterConformance:
    """The complete result of running the harness against one adapter."""

    capabilities: AdapterCapabilities | None
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        """True when no check failed. Skipped checks do not fail a run."""

        return not self.failures

    @property
    def failures(self) -> tuple[ConformanceCheck, ...]:
        """Checks that failed, in execution order."""

        return tuple(check for check in self.checks if check.status is CheckStatus.FAILED)

    @property
    def skipped(self) -> tuple[ConformanceCheck, ...]:
        """Checks the caller did not supply enough evidence to run."""

        return tuple(check for check in self.checks if check.status is CheckStatus.SKIPPED)

    def summary(self) -> str:
        """A single readable line per check, suitable for assertion messages."""

        return "\n".join(f"{check.status.value}: {check.name}: {check.detail}" for check in self)

    def __iter__(self) -> Iterator[ConformanceCheck]:
        return iter(self.checks)


def check_study_adapter(
    adapter: Any,
    *,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
    chronology_withheld: Callable[[], Any] | None = None,
    repeatable: bool = True,
) -> AdapterConformance:
    """Run the study-adapter conformance checks and return what each one observed.

    Args:
        adapter: The adapter under test, already pointed at a small fixture source.
        expected_rows: The source rows in source order, as mappings of column name to
            expected value. Only the columns present in a mapping are compared, so a row
            may name just ``subject``, ``session`` and one behavioural field. Supplying
            these enables the row-order and boundary checks.
        chronology_withheld: A zero-argument callable returning the same adapter with its
            chronology evidence removed -- no ``session_order`` column, no declared order,
            no named derivation. Reading it must raise. This is the check that an adapter
            does not fabricate ``session_order``.
        repeatable: Whether reading twice is expected to give the same study. Set it to
            ``False`` only for sources that are genuinely mutable between reads.

    Returns:
        An :class:`AdapterConformance` record; ``passed`` is False if any check failed.
    """

    checks: list[ConformanceCheck] = []
    capabilities: AdapterCapabilities | None = None
    try:
        capabilities = adapter_capabilities(adapter)
    except (TypeError, ValueError) as error:
        checks.append(
            ConformanceCheck(
                "declares-adapter-identity",
                CheckStatus.FAILED,
                f"adapter_capabilities() rejected the adapter: {error}",
            )
        )
        return AdapterConformance(capabilities=None, checks=tuple(checks))
    checks.append(
        ConformanceCheck(
            "declares-adapter-identity",
            CheckStatus.PASSED,
            f"{capabilities.adapter_name} {capabilities.adapter_version} "
            f"({capabilities.source_type.value}, "
            f"session_order {capabilities.session_order_policy.value})",
        )
    )

    try:
        study = adapter.read()
    except Exception as error:  # any failure here is a conformance failure
        checks.append(
            ConformanceCheck(
                "reads-a-valid-study",
                CheckStatus.FAILED,
                f"read() raised {type(error).__name__}: {error}",
            )
        )
        return AdapterConformance(capabilities=capabilities, checks=tuple(checks))
    if not isinstance(study, Study):
        checks.append(
            ConformanceCheck(
                "reads-a-valid-study",
                CheckStatus.FAILED,
                f"read() returned {type(study).__name__}, not a Study",
            )
        )
        return AdapterConformance(capabilities=capabilities, checks=tuple(checks))
    checks.append(
        ConformanceCheck(
            "reads-a-valid-study",
            CheckStatus.PASSED,
            f"{len(study)} trials, {len(study.subjects)} subjects, {len(study.columns)} columns",
        )
    )

    checks.append(_check_row_order(study, expected_rows))
    checks.append(_check_boundaries(study, expected_rows))
    checks.append(_check_chronology_is_not_fabricated(capabilities, chronology_withheld))
    checks.append(_check_chronology_is_usable(study))
    checks.append(_check_repeatable(adapter, study, repeatable=repeatable))
    return AdapterConformance(capabilities=capabilities, checks=tuple(checks))


def assert_study_adapter_conforms(
    adapter: Any,
    *,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
    chronology_withheld: Callable[[], Any] | None = None,
    repeatable: bool = True,
    require_complete: bool = False,
) -> AdapterConformance:
    """Run :func:`check_study_adapter` and raise on any failure.

    Set ``require_complete=True`` to also reject a run in which the caller left a check
    without the evidence it needs; that is the strict mode an adapter's own test suite
    should use.
    """

    report = check_study_adapter(
        adapter,
        expected_rows=expected_rows,
        chronology_withheld=chronology_withheld,
        repeatable=repeatable,
    )
    incomplete = report.skipped if require_complete else ()
    if not report.passed or incomplete:
        raise AdapterConformanceError("study adapter conformance failed:\n" + report.summary())
    return report


def _check_row_order(
    study: Study, expected_rows: Sequence[Mapping[str, Any]] | None
) -> ConformanceCheck:
    name = "preserves-source-trial-order"
    if expected_rows is None:
        return ConformanceCheck(
            name, CheckStatus.SKIPPED, "no expected_rows were supplied to compare against"
        )
    rows = list(expected_rows)
    if len(rows) != len(study):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"expected {len(rows)} trials in source order, read {len(study)}",
        )
    for index, row in enumerate(rows):
        for column, expected in row.items():
            if column not in study.columns:
                return ConformanceCheck(name, CheckStatus.FAILED, f"study has no column {column!r}")
            observed = study[column][index]
            if not _equal(observed, expected):
                return ConformanceCheck(
                    name,
                    CheckStatus.FAILED,
                    f"row {index} column {column!r} is {observed!r}, expected {expected!r}; "
                    "the adapter reordered or altered source rows",
                )
    return ConformanceCheck(name, CheckStatus.PASSED, f"{len(rows)} rows match the source in order")


def _check_boundaries(
    study: Study, expected_rows: Sequence[Mapping[str, Any]] | None
) -> ConformanceCheck:
    name = "preserves-subject-and-session-boundaries"
    observed = _runs(study["subject"], study["session"])
    if expected_rows is None:
        return ConformanceCheck(
            name,
            CheckStatus.SKIPPED,
            f"no expected_rows were supplied; the study itself has {len(observed)} "
            "contiguous subject/session runs",
        )
    rows = list(expected_rows)
    if any("subject" not in row or "session" not in row for row in rows):
        return ConformanceCheck(
            name, CheckStatus.SKIPPED, "expected_rows do not all name subject and session"
        )
    expected = _runs(
        [row["subject"] for row in rows],
        [row["session"] for row in rows],
    )
    if len(expected) != len(observed) or any(
        not _equal(left[0], right[0]) or not _equal(left[1], right[1]) or left[2] != right[2]
        for left, right in zip(expected, observed, strict=True)
    ):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"subject/session runs are {observed}, expected {expected}; the adapter "
            "regrouped, sorted, or merged sessions",
        )
    return ConformanceCheck(
        name, CheckStatus.PASSED, f"{len(observed)} subject/session runs match the source"
    )


def _check_chronology_is_not_fabricated(
    capabilities: AdapterCapabilities,
    chronology_withheld: Callable[[], Any] | None,
) -> ConformanceCheck:
    name = "refuses-to-fabricate-session-order"
    if chronology_withheld is None:
        return ConformanceCheck(
            name,
            CheckStatus.SKIPPED,
            "no chronology_withheld factory was supplied; supply one that removes the "
            "session_order column, the declared order, and any named derivation",
        )
    try:
        withheld = chronology_withheld()
    except (ValueError, TypeError) as error:
        return ConformanceCheck(
            name,
            CheckStatus.PASSED,
            f"declaring an adapter without chronology already fails: {error}",
        )
    try:
        study = withheld.read()
    except (ValueError, TypeError, KeyError, RuntimeError) as error:
        return ConformanceCheck(
            name, CheckStatus.PASSED, f"reading without chronology fails: {error}"
        )
    orders = np.asarray(study["session_order"]).tolist()
    policy = capabilities.session_order_policy.value
    return ConformanceCheck(
        name,
        CheckStatus.FAILED,
        f"the adapter invented session_order {orders} with no record and no named "
        f"derivation (declared policy {policy}); session chronology is a claim about "
        "time that row, file, and filename order do not carry",
    )


def _check_chronology_is_usable(study: Study) -> ConformanceCheck:
    name = "chronology-orders-the-study"
    try:
        indices = study.chronological_indices()
        reordered = study.take(indices)
    except Exception as error:  # surfaced as a conformance failure
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"chronological_indices()/take() raised {type(error).__name__}: {error}",
        )
    if len(reordered) != len(study):
        return ConformanceCheck(
            name, CheckStatus.FAILED, "the chronological view dropped or duplicated trials"
        )
    return ConformanceCheck(
        name,
        CheckStatus.PASSED,
        f"session_order sorts {len(study)} trials without violating the study contract",
    )


def _check_repeatable(adapter: Any, study: Study, *, repeatable: bool) -> ConformanceCheck:
    name = "reading-twice-is-stable"
    if not repeatable:
        return ConformanceCheck(
            name, CheckStatus.SKIPPED, "the caller declared the source mutable between reads"
        )
    try:
        again = adapter.read()
    except Exception as error:  # surfaced as a conformance failure
        return ConformanceCheck(
            name, CheckStatus.FAILED, f"the second read raised {type(error).__name__}: {error}"
        )
    if again.columns != study.columns or len(again) != len(study):
        return ConformanceCheck(
            name,
            CheckStatus.FAILED,
            f"the second read produced {len(again)} trials with columns {again.columns}",
        )
    for column in study.columns:
        if not np.array_equal(np.asarray(again[column]), np.asarray(study[column])):
            return ConformanceCheck(
                name, CheckStatus.FAILED, f"column {column!r} changed between reads"
            )
    return ConformanceCheck(
        name, CheckStatus.PASSED, "two reads of an unchanged source agree exactly"
    )


def _runs(subjects: Sequence[Any], sessions: Sequence[Any]) -> list[tuple[Any, Any, int]]:
    runs: list[tuple[Any, Any, int]] = []
    for subject, session in zip(subjects, sessions, strict=True):
        key = (_scalar(subject), _scalar(session))
        if runs and _equal(runs[-1][0], key[0]) and _equal(runs[-1][1], key[1]):
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
            continue
        runs.append((key[0], key[1], 1))
    return runs


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _equal(left: Any, right: Any) -> bool:
    left = _scalar(left)
    right = _scalar(right)
    if isinstance(left, float) and isinstance(right, float):
        return bool(left == right) or (left != left and right != right)
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


__all__ = [
    "AdapterConformance",
    "AdapterConformanceError",
    "CheckStatus",
    "ConformanceCheck",
    "assert_study_adapter_conforms",
    "check_study_adapter",
]
