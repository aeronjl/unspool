"""Portable, deterministic, content-addressed protocol evidence bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import re
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from behavio.protocol import ProtocolState, protocol_from_json
from behavio.reporting import ReportedProtocol

BUNDLE_SCHEMA_VERSION = "behavio.evidence-bundle/1"
#: Nested-report schema names this format has been published under. Reports written before
#: the ``unspool`` to ``behavio`` rename are byte-identical apart from this name, and are
#: content-addressed, so they are recognised rather than restamped.
_NESTED_EVALUATION_SCHEMAS = (
    "behavio.nested-evaluation-report/1",
    "unspool.nested-evaluation-report/1",
)
_FORBIDDEN_SUFFIXES = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".dill",
    ".pyc",
    ".npy",
    ".npz",
}
_FIGURE_MEDIA = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_REQUIRED_PATHS = {
    "protocol/protocol.json",
    "protocol/amendments.json",
    "environment/environment.json",
    "source/source.json",
    "cohort/manifest.json",
    "execution/plan.json",
    "execution/folds.json",
    "audits/protocol.json",
    "audits/fits.json",
    "comparison/evaluation.json",
    "predictions/pointwise.json",
    "report/report.json",
    "report/report.md",
    "report/items.json",
    "reproduction/replay.json",
}


class EvidenceBundleError(ValueError):
    """Raised when bundle content, integrity, or cross-artifact identity is invalid."""


@dataclass(frozen=True, slots=True)
class BundleFigure:
    """One named rendered figure tied to a required report item."""

    name: str
    filename: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("bundle figure name must be non-empty")
        if not isinstance(self.filename, str) or Path(self.filename).name != self.filename:
            raise ValueError("bundle figure filename must be one local filename")
        if Path(self.filename).suffix.lower() not in _FIGURE_MEDIA:
            raise ValueError("bundle figure must use a supported static image format")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("bundle figure content must be non-empty bytes")


@dataclass(frozen=True, slots=True)
class BundleFile:
    """One immutable portable artifact stored under a safe POSIX path."""

    path: str
    media_type: str
    content: bytes

    def __post_init__(self) -> None:
        _validate_path(self.path)
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("bundle file media type must be non-empty")
        if not isinstance(self.content, bytes):
            raise TypeError("bundle file content must be bytes")
        if PurePosixPath(self.path).suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise EvidenceBundleError(f"executable serialization is forbidden: {self.path!r}")

    @property
    def sha256(self) -> str:
        """Digest used by the enclosing bundle manifest."""

        return hashlib.sha256(self.content).hexdigest()

    @property
    def size(self) -> int:
        """Exact stored byte count."""

        return len(self.content)


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Sorted immutable files with a manifest-derived bundle identity."""

    files: tuple[BundleFile, ...]

    def __post_init__(self) -> None:
        files = tuple(sorted(self.files, key=lambda item: item.path))
        paths = [item.path for item in files]
        if not files or len(set(paths)) != len(paths):
            raise EvidenceBundleError("bundle paths must be non-empty and unique")
        if "bundle.json" in paths:
            raise EvidenceBundleError("bundle.json is reserved for the generated manifest")
        missing = sorted(_REQUIRED_PATHS - set(paths))
        if missing:
            raise EvidenceBundleError(f"bundle is missing required evidence: {missing}")
        object.__setattr__(self, "files", files)

    @property
    def bundle_id(self) -> str:
        """Content address over every logical path, media type, size, and file digest."""

        return _digest(self._identity_manifest())

    @property
    def manifest(self) -> dict[str, Any]:
        """Portable integrity manifest written as ``bundle.json``."""

        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "files": self._file_entries(),
        }

    def canonical_manifest_json(self) -> str:
        """Serialize the bundle manifest deterministically."""

        return _canonical_json(self.manifest)

    def file(self, path: str) -> BundleFile:
        """Return one verified logical artifact."""

        for item in self.files:
            if item.path == path:
                return item
        raise KeyError(f"unknown bundle path: {path!r}")

    def _file_entries(self) -> list[dict[str, Any]]:
        return [
            {
                "path": item.path,
                "media_type": item.media_type,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in self.files
        ]

    def _identity_manifest(self) -> dict[str, Any]:
        return {"schema_version": BUNDLE_SCHEMA_VERSION, "files": self._file_entries()}


@dataclass(frozen=True, slots=True)
class BundleReplay:
    """Cross-checked scientific identities recovered without executing serialized code."""

    bundle_id: str
    protocol_identifier: str
    protocol_fingerprint: str
    protocol_state: str
    cohort_observations: int
    n_folds: int
    ranking_status: str
    winner: str | None
    blocked_claims: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BundleComparison:
    """Deterministic logical and scientific differences between two bundles."""

    left_bundle_id: str
    right_bundle_id: str
    same_protocol: bool
    added_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    left_ranking: str
    right_ranking: str
    left_winner: str | None
    right_winner: str | None
    left_blocked_claims: tuple[str, ...]
    right_blocked_claims: tuple[str, ...]

    @property
    def identical(self) -> bool:
        """Whether every content-addressed artifact is identical."""

        return self.left_bundle_id == self.right_bundle_id


def capture_environment() -> dict[str, Any]:
    """Capture a path-free deterministic runtime description for reproduction."""

    packages: dict[str, str] = {}
    for package in ("behavio", "numpy", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "source-tree"
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": packages,
    }


def build_evidence_bundle(
    reported: ReportedProtocol,
    *,
    figures: tuple[BundleFigure, ...],
    environment: Mapping[str, Any] | None = None,
) -> EvidenceBundle:
    """Assemble the canonical final evidence bundle without source trial values."""

    if reported.protocol.state != ProtocolState.REPORTED:
        raise EvidenceBundleError("only a reported protocol can produce a final evidence bundle")
    required_figures = set(reported.protocol.reporting.required_figures)
    figure_names = [figure.name for figure in figures]
    if len(set(figure_names)) != len(figure_names):
        raise EvidenceBundleError("bundle figure names must be unique")
    missing_figures = sorted(required_figures - set(figure_names))
    if missing_figures:
        raise EvidenceBundleError(f"required rendered figures are missing: {missing_figures}")
    _reject_sensitive_source_metadata(reported)

    evaluation = reported.evaluation
    protocol = reported.protocol
    plan = evaluation.compiled.plan
    candidate_evidence = {
        **evaluation.report.to_dict(retain_predictions=False),
        "artifact_fingerprint": evaluation.report.fingerprint,
    }
    predictions = _evaluation_predictions(evaluation.report)
    fit_audits = _evaluation_audits(evaluation.report)
    replay = {
        "protocol_fingerprint": protocol.fingerprint,
        "source": {
            "adapter": protocol.source.adapter,
            "release": protocol.source.release,
            "locator": protocol.source.locator,
            "checksum_algorithm": protocol.source.checksum_algorithm,
            "checksum": protocol.source.checksum,
        },
        "cohort_manifest_fingerprint": evaluation.compiled.materialized.manifest.fingerprint,
        "execution_plan_fingerprint": plan.fingerprint,
        "evaluation_fingerprint": evaluation.report.fingerprint,
        "recovery_fingerprint": (
            reported.recovery.report.fingerprint if reported.recovery else None
        ),
        "report_fingerprint": reported.report.fingerprint,
    }
    files = [
        _json_file("protocol/protocol.json", protocol.to_dict()),
        _json_file(
            "protocol/amendments.json",
            [asdict(amendment) for amendment in protocol.amendments],
        ),
        _json_file(
            "environment/environment.json",
            dict(environment) if environment is not None else capture_environment(),
        ),
        _json_file("source/source.json", asdict(protocol.source)),
        _json_file("cohort/manifest.json", evaluation.compiled.materialized.manifest.to_dict()),
        _json_file("execution/plan.json", plan.to_dict()),
        _json_file("execution/folds.json", [asdict(fold) for fold in plan.folds]),
        _json_file("audits/protocol.json", asdict(plan.audit)),
        _json_file("audits/fits.json", fit_audits),
        _json_file("comparison/evaluation.json", candidate_evidence),
        _json_file("predictions/pointwise.json", predictions),
        _json_file("report/report.json", reported.report.to_dict()),
        BundleFile("report/report.md", "text/markdown", reported.report.markdown.encode("utf-8")),
        _json_file("report/items.json", [asdict(item) for item in reported.items]),
        _json_file("reproduction/replay.json", replay),
    ]
    if reported.recovery:
        files.append(_json_file("recovery/recovery.json", reported.recovery.report.to_dict()))
    for figure in figures:
        slug = _slug(figure.name)
        suffix = Path(figure.filename).suffix.lower()
        files.append(BundleFile(f"figures/{slug}{suffix}", _FIGURE_MEDIA[suffix], figure.content))
    return EvidenceBundle(tuple(files))


def write_evidence_bundle(bundle: EvidenceBundle, path: str | Path) -> Path:
    """Write a byte-deterministic ZIP archive without overwriting an existing path."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"evidence bundle already exists: {destination}")
    if destination.suffix.lower() != ".zip":
        raise EvidenceBundleError("evidence bundle path must end in .zip")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        _write_zip_entry(
            archive,
            "bundle.json",
            (bundle.canonical_manifest_json() + "\n").encode("utf-8"),
        )
        for item in bundle.files:
            _write_zip_entry(archive, item.path, item.content)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(buffer.getvalue())
    return destination


def read_evidence_bundle(path: str | Path) -> EvidenceBundle:
    """Read and verify every path, size, digest, and bundle identity without extraction."""

    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise EvidenceBundleError("bundle archive contains duplicate paths")
            if "bundle.json" not in names:
                raise EvidenceBundleError("bundle archive has no bundle.json manifest")
            for name in names:
                _validate_path(name)
            manifest = _load_json(archive.read("bundle.json"), "bundle.json")
            files: list[BundleFile] = []
            entries = manifest.get("files") if isinstance(manifest, dict) else None
            if not isinstance(entries, list):
                raise EvidenceBundleError("bundle manifest files must be an array")
            expected_names = {
                "bundle.json",
                *(entry.get("path") for entry in entries if isinstance(entry, dict)),
            }
            if set(names) != expected_names:
                raise EvidenceBundleError("archive paths differ from the bundle manifest")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise EvidenceBundleError("bundle file entry must be an object")
                path_value = entry.get("path")
                media_type = entry.get("media_type")
                if not isinstance(path_value, str) or not isinstance(media_type, str):
                    raise EvidenceBundleError("bundle file entry has invalid path or media type")
                content = archive.read(path_value)
                item = BundleFile(path_value, media_type, content)
                if entry != {
                    "path": item.path,
                    "media_type": item.media_type,
                    "size": item.size,
                    "sha256": item.sha256,
                }:
                    raise EvidenceBundleError(f"bundle file integrity mismatch: {path_value!r}")
                files.append(item)
    except (zipfile.BadZipFile, KeyError, OSError) as error:
        raise EvidenceBundleError(f"cannot read evidence bundle: {error}") from error
    bundle = EvidenceBundle(tuple(files))
    if manifest != bundle.manifest:
        raise EvidenceBundleError("bundle manifest identity does not match verified content")
    return bundle


def replay_evidence_bundle(bundle_or_path: EvidenceBundle | str | Path) -> BundleReplay:
    """Cross-check protocol, manifest, plan, evaluation, recovery, and report identities."""

    bundle = _as_bundle(bundle_or_path)
    protocol = protocol_from_json(bundle.file("protocol/protocol.json").content.decode("utf-8"))
    cohort = _bundle_json(bundle, "cohort/manifest.json")
    plan = _bundle_json(bundle, "execution/plan.json")
    evaluation = _bundle_json(bundle, "comparison/evaluation.json")
    report = _bundle_json(bundle, "report/report.json")
    replay = _bundle_json(bundle, "reproduction/replay.json")
    recovery = (
        _bundle_json(bundle, "recovery/recovery.json")
        if any(item.path == "recovery/recovery.json" for item in bundle.files)
        else None
    )
    expected = {
        "protocol": (
            protocol.fingerprint,
            cohort.get("protocol_fingerprint"),
            plan.get("protocol_fingerprint"),
            evaluation.get("protocol_fingerprint"),
            report.get("protocol_fingerprint"),
            replay.get("protocol_fingerprint"),
            recovery.get("protocol_fingerprint") if recovery else protocol.fingerprint,
        ),
        "plan": (
            _digest(plan),
            evaluation.get("execution_plan_fingerprint"),
            replay.get("execution_plan_fingerprint"),
            recovery.get("execution_plan_fingerprint") if recovery else _digest(plan),
        ),
        "evaluation": (
            evaluation.get("artifact_fingerprint"),
            report.get("evaluation_fingerprint"),
            replay.get("evaluation_fingerprint"),
        ),
    }
    for identity, values in expected.items():
        if len(set(values)) != 1:
            raise EvidenceBundleError(f"bundle {identity} identities are inconsistent: {values!r}")
    if replay.get("cohort_manifest_fingerprint") != _digest(cohort):
        raise EvidenceBundleError("bundle cohort manifest fingerprint is inconsistent")
    if recovery and replay.get("recovery_fingerprint") != _digest(recovery):
        raise EvidenceBundleError("bundle recovery fingerprint is inconsistent")
    if report.get("recovery_fingerprint") != replay.get("recovery_fingerprint"):
        raise EvidenceBundleError("bundle report recovery identity is inconsistent")
    if replay.get("report_fingerprint") != _digest(report):
        raise EvidenceBundleError("bundle report fingerprint is inconsistent")
    if protocol.lifecycle[-1].artifact_fingerprint != _digest(report):
        raise EvidenceBundleError("reported lifecycle event does not identify the bundled report")
    ranking = evaluation.get("ranking")
    if isinstance(ranking, dict):
        ranking_status = str(ranking["status"])
        winner = ranking.get("winner")
    elif evaluation.get("schema_version") in _NESTED_EVALUATION_SCHEMAS:
        ranking_status = "nested-training-only-selection"
        winner = None
    else:
        raise EvidenceBundleError("bundle evaluation decision evidence is missing")
    return BundleReplay(
        bundle_id=bundle.bundle_id,
        protocol_identifier=protocol.identifier,
        protocol_fingerprint=protocol.fingerprint,
        protocol_state=protocol.state.value,
        cohort_observations=int(cohort["selected_observations"]),
        n_folds=len(plan["folds"]),
        ranking_status=ranking_status,
        winner=winner,
        blocked_claims=tuple(report.get("blocked_claims", ())),
    )


def compare_evidence_bundles(
    left: EvidenceBundle | str | Path,
    right: EvidenceBundle | str | Path,
) -> BundleComparison:
    """Compare content paths and key scientific decisions after full replay verification."""

    left_bundle = _as_bundle(left)
    right_bundle = _as_bundle(right)
    left_replay = replay_evidence_bundle(left_bundle)
    right_replay = replay_evidence_bundle(right_bundle)
    left_files = {item.path: item.sha256 for item in left_bundle.files}
    right_files = {item.path: item.sha256 for item in right_bundle.files}
    common = set(left_files) & set(right_files)
    return BundleComparison(
        left_bundle_id=left_bundle.bundle_id,
        right_bundle_id=right_bundle.bundle_id,
        same_protocol=left_replay.protocol_fingerprint == right_replay.protocol_fingerprint,
        added_paths=tuple(sorted(set(right_files) - set(left_files))),
        removed_paths=tuple(sorted(set(left_files) - set(right_files))),
        changed_paths=tuple(
            sorted(path for path in common if left_files[path] != right_files[path])
        ),
        left_ranking=left_replay.ranking_status,
        right_ranking=right_replay.ranking_status,
        left_winner=left_replay.winner,
        right_winner=right_replay.winner,
        left_blocked_claims=left_replay.blocked_claims,
        right_blocked_claims=right_replay.blocked_claims,
    )


def _evaluation_predictions(report: Any) -> dict[str, Any]:
    if hasattr(report, "candidates"):
        return {
            candidate.name: {
                fold.fold: [asdict(point) for point in fold.predictions] for fold in candidate.folds
            }
            for candidate in report.candidates
        }
    return {
        fold.outer_fold: {
            "selected_candidate": fold.selected_candidate,
            "inner": {
                candidate.name: {
                    inner_fold.fold: [asdict(point) for point in inner_fold.predictions]
                    for inner_fold in candidate.folds
                }
                for candidate in fold.inner_candidates
            },
            "outer": (
                {
                    outer_fold.fold: [asdict(point) for point in outer_fold.predictions]
                    for outer_fold in fold.outer_result.folds
                }
                if fold.outer_result
                else None
            ),
        }
        for fold in report.folds
    }


def _evaluation_audits(report: Any) -> dict[str, Any]:
    if hasattr(report, "candidates"):
        return {
            candidate.name: {
                "failures": [asdict(failure) for failure in candidate.failures],
                "folds": {fold.fold: fold.audit.to_dict() for fold in candidate.folds},
            }
            for candidate in report.candidates
        }
    return {
        fold.outer_fold: {
            "selected_candidate": fold.selected_candidate,
            "selection_failure": fold.selection_failure,
            "inner": {
                candidate.name: {
                    "failures": [asdict(failure) for failure in candidate.failures],
                    "folds": {
                        inner_fold.fold: inner_fold.audit.to_dict()
                        for inner_fold in candidate.folds
                    },
                }
                for candidate in fold.inner_candidates
            },
            "outer": (
                {
                    "failures": [asdict(failure) for failure in fold.outer_result.failures],
                    "folds": {
                        outer_fold.fold: outer_fold.audit.to_dict()
                        for outer_fold in fold.outer_result.folds
                    },
                }
                if fold.outer_result
                else None
            ),
        }
        for fold in report.folds
    }


def _json_file(path: str, value: Any) -> BundleFile:
    return BundleFile(
        path, "application/json", (_canonical_json(_json_safe(value)) + "\n").encode()
    )


def _write_zip_entry(archive: zipfile.ZipFile, path: str, content: bytes) -> None:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def _validate_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceBundleError("bundle paths must be non-empty POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise EvidenceBundleError(f"unsafe bundle path: {value!r}")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise EvidenceBundleError(f"figure name cannot form a safe path: {value!r}")
    return slug


def _reject_sensitive_source_metadata(reported: ReportedProtocol) -> None:
    sensitive = ("token", "password", "secret", "credential", "api-key", "api_key")
    names = [setting.name.lower() for setting in reported.protocol.source.metadata]
    found = sorted(name for name in names if any(marker in name for marker in sensitive))
    if found:
        raise EvidenceBundleError(
            f"source metadata may contain credentials and cannot be bundled: {found}"
        )


def _load_json(content: bytes, label: str) -> Any:
    def reject(constant: str) -> None:
        raise EvidenceBundleError(f"non-finite JSON constant in {label}: {constant}")

    try:
        return json.loads(content, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceBundleError(f"invalid JSON in {label}: {error}") from error


def _bundle_json(bundle: EvidenceBundle, path: str) -> Any:
    return _load_json(bundle.file(path).content, path)


def _as_bundle(value: EvidenceBundle | str | Path) -> EvidenceBundle:
    return value if isinstance(value, EvidenceBundle) else read_evidence_bundle(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(_json_safe(value)).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
