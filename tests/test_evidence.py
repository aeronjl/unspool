"""Tests for deterministic, corruption-detecting evidence bundles."""

import json
import zipfile

import pytest
from test_reporting import recovered, report_items
from test_runner import candidate_models, compiled_nested

from unspool.evidence import (
    BundleFigure,
    BundleFile,
    EvidenceBundle,
    EvidenceBundleError,
    build_evidence_bundle,
    compare_evidence_bundles,
    read_evidence_bundle,
    replay_evidence_bundle,
    write_evidence_bundle,
)
from unspool.reporting import generate_bounded_report
from unspool.runner import run_nested_protocol


def reported():
    return generate_bounded_report(recovered(), items=report_items())


def figure():
    return BundleFigure(
        name="paired-score-contrast",
        filename="paired-score-contrast.png",
        content=b"\x89PNG\r\n\x1a\nsynthetic-test-figure",
    )


def environment(*, marker="a"):
    return {
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "platform": {"system": "test", "machine": "test"},
        "packages": {"unspool": "0.20-test"},
        "marker": marker,
    }


def bundle(*, marker="a"):
    return build_evidence_bundle(
        reported(),
        figures=(figure(),),
        environment=environment(marker=marker),
    )


def test_bundle_contains_canonical_evidence_without_source_trials() -> None:
    evidence = bundle()

    paths = {item.path for item in evidence.files}
    assert "protocol/protocol.json" in paths
    assert "cohort/manifest.json" in paths
    assert "execution/folds.json" in paths
    assert "predictions/pointwise.json" in paths
    assert "recovery/recovery.json" in paths
    assert "figures/paired-score-contrast.png" in paths
    assert "report/report.md" in paths
    assert len(evidence.bundle_id) == 64
    assert evidence.manifest["bundle_id"] == evidence.bundle_id
    cohort = evidence.file("cohort/manifest.json").content.decode()
    assert "excluded-rat" not in cohort
    assert '"choice"' not in cohort
    assert all(not item.path.endswith((".pkl", ".pickle", ".npy")) for item in evidence.files)


def test_zip_bytes_are_deterministic_and_replay_cross_checks_identities(tmp_path) -> None:
    evidence = bundle()
    first = write_evidence_bundle(evidence, tmp_path / "first.zip")
    second = write_evidence_bundle(bundle(), tmp_path / "second.zip")

    assert first.read_bytes() == second.read_bytes()
    restored = read_evidence_bundle(first)
    assert restored.bundle_id == evidence.bundle_id
    replay = replay_evidence_bundle(restored)
    assert replay.bundle_id == evidence.bundle_id
    assert replay.protocol_identifier == "learning-forecast-v1"
    assert replay.protocol_state == "reported"
    assert replay.cohort_observations == 12
    assert replay.n_folds == 1
    assert replay.ranking_status in {"resolved", "unresolved"}


def test_bundle_comparison_reports_logical_and_scientific_changes() -> None:
    left = bundle(marker="a")
    right = bundle(marker="b")
    comparison = compare_evidence_bundles(left, right)

    assert not comparison.identical
    assert comparison.same_protocol
    assert comparison.added_paths == ()
    assert comparison.removed_paths == ()
    assert comparison.changed_paths == ("environment/environment.json",)
    assert comparison.left_ranking == comparison.right_ranking
    assert comparison.left_winner == comparison.right_winner


def test_changed_zip_content_is_rejected_by_manifest(tmp_path) -> None:
    original = write_evidence_bundle(bundle(), tmp_path / "original.zip")
    corrupted = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(corrupted, "w") as target:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename == "report/report.md":
                content += b"tampered"
            target.writestr(info, content)

    with pytest.raises(EvidenceBundleError, match="integrity mismatch"):
        read_evidence_bundle(corrupted)


def test_cross_artifact_tampering_is_rejected_even_with_new_valid_manifest() -> None:
    original = bundle()
    replay = json.loads(original.file("reproduction/replay.json").content)
    replay["protocol_fingerprint"] = "0" * 64
    changed = tuple(
        BundleFile(
            item.path,
            item.media_type,
            (
                (json.dumps(replay, sort_keys=True, separators=(",", ":")) + "\n").encode()
                if item.path == "reproduction/replay.json"
                else item.content
            ),
        )
        for item in original.files
    )
    valid_but_inconsistent = EvidenceBundle(changed)

    with pytest.raises(EvidenceBundleError, match="protocol identities are inconsistent"):
        replay_evidence_bundle(valid_but_inconsistent)


def test_unsafe_or_executable_bundle_paths_are_forbidden() -> None:
    with pytest.raises(EvidenceBundleError, match="unsafe bundle path"):
        BundleFile("../escape.json", "application/json", b"{}")
    with pytest.raises(EvidenceBundleError, match="executable serialization"):
        BundleFile("fits/result.pkl", "application/octet-stream", b"pickle")


def test_required_rendered_figure_cannot_be_omitted() -> None:
    with pytest.raises(EvidenceBundleError, match="rendered figures are missing"):
        build_evidence_bundle(reported(), figures=(), environment=environment())


def test_writer_never_overwrites_an_existing_bundle(tmp_path) -> None:
    path = write_evidence_bundle(bundle(), tmp_path / "evidence.zip")
    with pytest.raises(FileExistsError):
        write_evidence_bundle(bundle(), path)


def test_nested_selection_bundle_retains_inner_and_outer_evidence() -> None:
    nested = run_nested_protocol(compiled_nested(), candidate_models())
    reported_nested = generate_bounded_report(nested, items=report_items())
    evidence = build_evidence_bundle(
        reported_nested,
        figures=(figure(),),
        environment=environment(),
    )

    replay = replay_evidence_bundle(evidence)
    predictions = json.loads(evidence.file("predictions/pointwise.json").content)
    assert replay.ranking_status == "nested-training-only-selection"
    assert replay.winner is None
    assert "fold-0000" in predictions
    assert set(predictions["fold-0000"]) == {"inner", "outer", "selected_candidate"}
