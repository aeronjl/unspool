"""Deterministic, claim-bounded reports generated from protocol evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from behavio.protocol import ProtocolState, ProtocolValidationError
from behavio.protocol_recovery import ExactRecoveryRun
from behavio.runner import NestedProtocolRun, ProtocolRun


class ReportGenerationError(ValueError):
    """Raised when required report evidence is absent or lifecycle-ineligible."""


class ReportItemKind(StrEnum):
    """Closed set of report artifacts required by a protocol."""

    TABLE = "table"
    FIGURE = "figure"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True, slots=True)
class ReportItem:
    """One named report artifact and its bounded textual interpretation."""

    name: str
    kind: ReportItemKind
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("report item name must be non-empty")
        object.__setattr__(self, "kind", ReportItemKind(self.kind))
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("report item summary must be non-empty")


@dataclass(frozen=True, slots=True)
class BoundedReport:
    """Markdown report whose claims are derived from declared evidence gates."""

    schema_version: str
    protocol_fingerprint: str
    evaluation_fingerprint: str
    recovery_fingerprint: str | None
    blocked_claims: tuple[str, ...]
    markdown: str

    @property
    def fingerprint(self) -> str:
        """Content address of the report and the evidence identities it cites."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the complete report as standards-compliant JSON data."""

        return {
            "schema_version": self.schema_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "recovery_fingerprint": self.recovery_fingerprint,
            "blocked_claims": list(self.blocked_claims),
            "markdown": self.markdown,
        }

    def canonical_json(self) -> str:
        """Serialize report content deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ReportedProtocol:
    """Reported lifecycle state paired with observed and recovery evidence."""

    protocol: Any
    evaluation: ProtocolRun | NestedProtocolRun
    recovery: ExactRecoveryRun | None
    report: BoundedReport
    items: tuple[ReportItem, ...]

    def __post_init__(self) -> None:
        if self.protocol.state != ProtocolState.REPORTED:
            raise ProtocolValidationError("a reported protocol requires reported lifecycle state")
        if self.protocol.fingerprint != self.report.protocol_fingerprint:
            raise ProtocolValidationError("bounded report and protocol fingerprints differ")


def generate_bounded_report(
    evidence: ProtocolRun | NestedProtocolRun | ExactRecoveryRun,
    *,
    items: tuple[ReportItem, ...],
) -> ReportedProtocol:
    """Generate a deterministic report only after all declared evidence stages.

    Required tables, figures, and diagnostics must be supplied by name. Recovery-gated
    protocols cannot report from observed evaluation alone. Failed gates are rendered as
    explicit blocked claims, and an unresolved ranking is never converted into a winner.
    """

    if isinstance(evidence, ExactRecoveryRun):
        recovery = evidence
        evaluation = evidence.evaluation
        protocol = evidence.protocol
        if protocol.state != ProtocolState.RECOVERED:
            raise ReportGenerationError("recovery evidence has not reached recovered state")
    elif isinstance(evidence, (ProtocolRun, NestedProtocolRun)):
        recovery = None
        evaluation = evidence
        protocol = evidence.protocol
        if protocol.recovery:
            raise ReportGenerationError(
                "the frozen protocol requires recovery before report generation"
            )
        if protocol.state != ProtocolState.EVALUATED:
            raise ReportGenerationError("observed evidence has not reached evaluated state")
    else:
        raise TypeError("evidence must be a ProtocolRun, NestedProtocolRun, or ExactRecoveryRun")

    items = tuple(items)
    _validate_items(protocol, items)
    blocked = tuple(recovery.report.blocked_claims) if recovery else ()
    markdown = _render_markdown(protocol, evaluation, recovery, items, blocked)
    report = BoundedReport(
        schema_version="behavio.bounded-report/1",
        protocol_fingerprint=protocol.fingerprint,
        evaluation_fingerprint=evaluation.report.fingerprint,
        recovery_fingerprint=recovery.report.fingerprint if recovery else None,
        blocked_claims=blocked,
        markdown=markdown,
    )
    reported = protocol.advance(
        ProtocolState.REPORTED,
        artifact_fingerprint=report.fingerprint,
    )
    return ReportedProtocol(reported, evaluation, recovery, report, items)


def _validate_items(protocol: Any, items: tuple[ReportItem, ...]) -> None:
    keys = [(item.kind, item.name) for item in items]
    if len(set(keys)) != len(keys):
        raise ReportGenerationError("report items must be unique within each kind")
    supplied = {kind: {item.name for item in items if item.kind == kind} for kind in ReportItemKind}
    required = {
        ReportItemKind.TABLE: set(protocol.reporting.required_tables),
        ReportItemKind.FIGURE: set(protocol.reporting.required_figures),
        ReportItemKind.DIAGNOSTIC: set(protocol.reporting.required_diagnostics),
    }
    missing = {
        kind.value: sorted(names - supplied[kind])
        for kind, names in required.items()
        if names - supplied[kind]
    }
    if missing:
        raise ReportGenerationError(f"required report evidence is missing: {missing}")


def _render_markdown(
    protocol: Any,
    evaluation: ProtocolRun | NestedProtocolRun,
    recovery: ExactRecoveryRun | None,
    items: tuple[ReportItem, ...],
    blocked: tuple[str, ...],
) -> str:
    manifest = evaluation.compiled.materialized.manifest
    comparison = protocol.comparison
    lines = [
        f"# {protocol.title}",
        "",
        "> This report is generated from a frozen Behavio study protocol. "
        "Its claims are bounded by the declared validation, numerical-audit, and recovery gates.",
        "",
        "## Scientific question",
        "",
        protocol.question,
        "",
        "## Frozen design",
        "",
        f"- Protocol: `{protocol.identifier}` (`{protocol.fingerprint}`)",
        f"- Schema: `{protocol.schema_version}`",
        f"- Source release: `{protocol.source.release}`",
        f"- Source checksum: `{protocol.source.checksum_algorithm}:{protocol.source.checksum}`",
        f"- Selected denominator: {manifest.selected_subjects} subjects, "
        f"{manifest.selected_sessions} sessions, {manifest.selected_observations} observations",
        f"- Validation geometry: `{protocol.validation.geometry.value}`",
        f"- Prediction information: `{protocol.validation.prediction_information.value}`",
        f"- Score: `{comparison.metric.value}` with `{comparison.weighting.value}` weighting "
        f"over `{comparison.aggregation_unit}`",
        "",
        "### Estimands",
        "",
    ]
    for estimand in protocol.estimands:
        lines.append(
            f"- **{estimand.name}:** {estimand.contrast}; population: {estimand.population}; "
            f"aggregation: `{estimand.weighting.value}` over `{estimand.aggregation_unit}`."
        )
    _render_unverified_declarations(lines, evaluation)
    if isinstance(evaluation, NestedProtocolRun):
        _render_nested_performance(lines, evaluation)
    else:
        _render_flat_performance(lines, evaluation, comparison)
    lines.extend(["", "## Exact-design recovery", ""])
    if recovery is None:
        lines.append("The frozen protocol did not require a recovery stage.")
    else:
        lines.extend(
            [
                "| Requirement | Kind | Metric | Observed | Gate | Status |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for gate in recovery.report.gates:
            observed = _number(gate.observed)
            comparison_symbol = "≥" if gate.direction == "at-least" else "≤"
            lines.append(
                f"| {_cell(gate.requirement)} | {gate.kind.value} | {_cell(gate.metric)} | "
                f"{observed} | {comparison_symbol} {gate.threshold:.6g} | {gate.status.value} |"
            )
        lines.append("")
        lines.append(
            f"Recovery retained {len(recovery.report.replicates)} successful replicates and "
            f"{len(recovery.report.failures)} failures. A passed gate permits consideration; "
            "it does not by itself establish the associated scientific claim."
        )
    lines.extend(["", "## Required evidence", ""])
    for kind in ReportItemKind:
        kind_items = tuple(item for item in items if item.kind == kind)
        if not kind_items:
            continue
        lines.extend([f"### {kind.value.title()}s", ""])
        for item in kind_items:
            lines.append(f"- **{item.name}:** {item.summary}")
        lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in protocol.reporting.limitations)
    lines.extend(["", "## Claim boundaries", ""])
    lines.append("The following claims are not established by this study:")
    lines.extend(f"- {claim}" for claim in protocol.reporting.prohibited_claims)
    if blocked:
        lines.extend(["", "Failed or insufficient recovery gates additionally block:"])
        lines.extend(f"- {claim}" for claim in blocked)
    lines.extend(["", "## Protocol history", ""])
    if protocol.amendments:
        for amendment in protocol.amendments:
            lines.append(
                f"- `{amendment.identifier}` changed {', '.join(amendment.changed_sections)}: "
                f"{amendment.reason} (parent `{amendment.parent_fingerprint}`)."
            )
    else:
        lines.append("No pre-evidence amendments were recorded.")
    lines.extend(
        [
            "",
            "## Reproduction identities",
            "",
            f"- Cohort manifest: `{manifest.fingerprint}`",
            f"- Execution plan: `{evaluation.compiled.plan.fingerprint}`",
            f"- Evaluation: `{evaluation.report.fingerprint}`",
        ]
    )
    if recovery:
        lines.append(f"- Recovery: `{recovery.report.fingerprint}`")
    return "\n".join(lines).rstrip() + "\n"


def _render_unverified_declarations(
    lines: list[str], evaluation: ProtocolRun | NestedProtocolRun
) -> None:
    """Disclose declarations that could not be checked against the object that ran.

    A contradicted declaration never reaches a report -- the runner refuses the plan -- so
    this section exists only for declarations that were retained without being proved. It
    is emitted only when such a finding exists, so a fully verified study keeps a
    byte-identical report and an unchanged report fingerprint.
    """

    findings = tuple(
        finding for verification in evaluation.verification for finding in verification.unverifiable
    )
    if not findings:
        return
    lines.extend(
        [
            "",
            "### Unverified declarations",
            "",
            "These frozen declarations were retained but could not be checked against the "
            "estimator that ran. They are recorded rather than treated as satisfied.",
            "",
            "| Candidate | Declaration | Declared | Supplied | Why unverified |",
            "|---|---|---|---|---|",
        ]
    )
    for finding in findings:
        lines.append(
            f"| {_cell(finding.candidate)} | {_cell(finding.subject)} | "
            f"{_cell(finding.declared)} | {_cell(finding.observed) or '—'} | "
            f"{_cell(finding.detail)} |"
        )


def _render_flat_performance(lines: list[str], evaluation: ProtocolRun, comparison: Any) -> None:
    report = evaluation.report
    score_label = comparison.metric.value
    lines.extend(
        [
            "",
            "## Candidate performance",
            "",
            f"| Candidate | Unit-balanced {score_label} | Interval | Pooled {score_label} | "
            "Audit | Failures |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for candidate in report.candidates:
        score = candidate.score
        interval = score.unit_balanced_interval if score else None
        interval_text = (
            f"[{interval.lower:.6f}, {interval.upper:.6f}]" if interval else "unavailable"
        )
        lines.append(
            f"| {_cell(candidate.name)} | "
            f"{_number(score.unit_balanced_score if score else None)} | "
            f"{interval_text} | {_number(score.pooled_score if score else None)} | "
            f"{candidate.audit_status.value} | {len(candidate.failures)} |"
        )
    lines.extend(["", "### Paired comparisons", ""])
    if report.paired_comparisons:
        lines.extend([f"| Contrast | Difference in {score_label} | Interval |", "|---|---:|---:|"])
        for paired in report.paired_comparisons:
            interval = paired.left_minus_right
            lines.append(
                f"| {_cell(paired.left_model)} minus {_cell(paired.right_model)} | "
                f"{interval.estimate:.6f} | [{interval.lower:.6f}, {interval.upper:.6f}] |"
            )
    else:
        lines.append("No paired comparison had two audit-eligible candidates with common units.")
    lines.extend(["", "### Ranking decision", ""])
    if report.ranking.winner is None:
        lines.append(f"**Unresolved.** {report.ranking.reason}")
    else:
        lines.append(
            f"**Resolved under `{comparison.winner_policy.value}`:** "
            f"`{report.ranking.winner}`. {report.ranking.reason}"
        )


def _render_nested_performance(lines: list[str], evaluation: NestedProtocolRun) -> None:
    report = evaluation.report
    score = report.score
    score_label = evaluation.protocol.comparison.metric.value
    interval = score.unit_balanced_interval if score else None
    interval_text = f"[{interval.lower:.6f}, {interval.upper:.6f}]" if interval else "unavailable"
    lines.extend(
        [
            "",
            "## Nested selection performance",
            "",
            "Every candidate comparison was fitted inside its outer training study. Only the "
            "selected candidate was refitted before the untouched outer rows were scored.",
            "",
            f"| Procedure | Unit-balanced {score_label} | Interval | Pooled {score_label} | "
            "Complete |",
            "|---|---:|---:|---:|---|",
            "| Nested selected procedure | "
            f"{_number(score.unit_balanced_score if score else None)} | "
            f"{interval_text} | {_number(score.pooled_score if score else None)} | "
            f"{report.eligible} |",
            "",
            "### Training-only selections",
            "",
            "| Candidate | Outer folds selected |",
            "|---|---:|",
        ]
    )
    for name, count in report.selection_counts.items():
        lines.append(f"| {_cell(name)} | {count} |")
    lines.extend(["", "No candidate ranking was computed from outer-test outcomes."])


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
