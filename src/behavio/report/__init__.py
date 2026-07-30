"""Artifacts a study hands to someone else: bounded reports, bundles, and fit records.

:mod:`behavio.report.bounded` generates prose whose claims cannot exceed the evidence the
protocol actually produced. :mod:`behavio.report.evidence_bundles` packs a run into a
deterministic, content-addressed ZIP that replays. :mod:`behavio.report.fit_artifacts`
serialises one fitted model as a portable, non-executable record.

That last module was called ``interchange``, which suggested interoperability between
tools; it has always been fit-artifact serialisation, and now says so. Its neighbour was
called ``evidence``, which suggested statistical evidence; it has always been ZIP bundles.
"""

from behavio.report.bounded import (
    BoundedReport,
    ReportedProtocol,
    ReportGenerationError,
    ReportItem,
    ReportItemKind,
    generate_bounded_report,
)
from behavio.report.evidence_bundles import (
    BUNDLE_SCHEMA_VERSION,
    BundleComparison,
    BundleFigure,
    BundleFile,
    BundleReplay,
    EvidenceBundle,
    EvidenceBundleError,
    PosteriorEvidence,
    build_evidence_bundle,
    capture_environment,
    compare_evidence_bundles,
    read_evidence_bundle,
    replay_evidence_bundle,
    write_evidence_bundle,
)
from behavio.report.fit_artifacts import (
    FIT_ARTIFACT_SCHEMA,
    FitArtifact,
    FitArtifactError,
    export_fit,
    fit_artifact_from_dict,
    fit_artifact_from_json,
)

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "FIT_ARTIFACT_SCHEMA",
    "BoundedReport",
    "BundleComparison",
    "BundleFigure",
    "BundleFile",
    "BundleReplay",
    "EvidenceBundle",
    "EvidenceBundleError",
    "FitArtifact",
    "FitArtifactError",
    "PosteriorEvidence",
    "ReportGenerationError",
    "ReportItem",
    "ReportItemKind",
    "ReportedProtocol",
    "build_evidence_bundle",
    "capture_environment",
    "compare_evidence_bundles",
    "export_fit",
    "fit_artifact_from_dict",
    "fit_artifact_from_json",
    "generate_bounded_report",
    "read_evidence_bundle",
    "replay_evidence_bundle",
    "write_evidence_bundle",
]
