"""Tests for deterministic, corruption-detecting evidence bundles."""

import hashlib
import json
import zipfile

import pytest
from test_reporting import recovered, report_items
from test_runner import candidate_models, compiled_nested

from behavio.evidence import (
    BUNDLE_SCHEMA_VERSION,
    POSTERIOR_CONVERGENCE_PATH,
    POSTERIOR_LOO_PATH,
    SUPERSEDED_BUNDLE_SCHEMA_VERSIONS,
    BundleFigure,
    BundleFile,
    EvidenceBundle,
    EvidenceBundleError,
    PosteriorEvidence,
    build_evidence_bundle,
    capture_environment,
    capture_source_control,
    compare_evidence_bundles,
    read_evidence_bundle,
    replay_evidence_bundle,
    write_evidence_bundle,
)
from behavio.posterior_diagnostics import PosteriorAudit, PosteriorAuditPolicy
from behavio.posterior_loo import PSISLOOResult
from behavio.reporting import generate_bounded_report
from behavio.runner import run_nested_protocol

#: Identity of the bundle the fixed fixtures produced under schema version 1, recorded
#: before the posterior slots existed. A purely frequentist study must keep producing this
#: exact archive, so the constants are pinned rather than recomputed.
#:
#: These moved once, when ``run_protocol`` began verifying each supplied estimator against
#: its frozen ``CandidateSpec``. The shared fixture protocol had declared
#: ``HierarchicalBernoulliHistoryGLM`` and ``HierarchicalSmoothBernoulliHistoryGLM`` while
#: the runner fixtures supplied two ``BernoulliHistoryGLM`` instances -- exactly the
#: contradiction the new check refuses -- so the declaration was corrected to name what
#: actually runs. Only the protocol fingerprint changed; every fitted number is unchanged.
#: Previously: bundle ``0dd4026a143cc4a08ec6c318574cb088bd92364f5ba8af5deb1b225fe437af82``,
#: zip ``7bd4c89a98cc5782bf581b04ba2f7a99c921a80243721ef502c9335217ad5d22``.
SCHEMA_1_BUNDLE_ID = "dac213700d892e96f7394c9015cbd5b92fd46f86bd8def03bf78caa234474437"
SCHEMA_1_ZIP_SHA256 = "d248c6da5e21ff2079975487f68236d2c84fa3aa01b972ea728af977c7fd77b9"


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
        "packages": {"behavio": "0.20-test"},
        "marker": marker,
    }


def bundle(*, marker="a", posterior=None):
    return build_evidence_bundle(
        reported(),
        figures=(figure(),),
        environment=environment(marker=marker),
        posterior=posterior,
    )


def convergence_audit(*, model_name="hierarchical-glm"):
    return PosteriorAudit(
        model_name=model_name,
        model_signature="hierarchical-glm/1",
        inference_library="pymc",
        inference_library_version="6.1.0",
        n_chains=4,
        n_draws=1000,
        divergences=0,
        max_treedepth_hits=0,
        policy=PosteriorAuditPolicy(),
        diagnostics=(),
        issues=(),
    )


def loo_result(*, model_name="hierarchical-glm", block=None):
    dim = "subject" if block == "subject" else "observation"
    return PSISLOOResult(
        model_name=model_name,
        model_signature="hierarchical-glm/1",
        inference_library="arviz",
        inference_library_version="1.2.0",
        log_likelihood_name="choice",
        dims=(dim,),
        coords={dim: ["a", "b", "c"]},
        elpd_loo=-42.5,
        se=3.25,
        p_loo=4.0,
        n_samples=4000,
        n_data_points=3,
        good_k=0.7,
        pointwise_elpd=[-14.0, -14.25, -14.25],
        pareto_k=[0.1, 0.2, 0.15],
        block=block,
    )


def posterior_evidence():
    return PosteriorEvidence(
        convergence={"hierarchical-glm": convergence_audit()},
        loo={
            "hierarchical-glm/observation": loo_result(),
            "hierarchical-glm/subject": loo_result(block="subject"),
        },
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


def test_frequentist_bundle_is_byte_identical_across_the_posterior_schema_bump(tmp_path) -> None:
    frequentist = bundle()
    empty_posterior = bundle(posterior=PosteriorEvidence())

    assert PosteriorEvidence().is_empty
    assert not posterior_evidence().is_empty
    assert frequentist.schema_version == SUPERSEDED_BUNDLE_SCHEMA_VERSIONS[-1]
    assert frequentist.posterior_paths == ()
    assert frequentist.bundle_id == SCHEMA_1_BUNDLE_ID
    assert empty_posterior.bundle_id == SCHEMA_1_BUNDLE_ID
    written = write_evidence_bundle(frequentist, tmp_path / "frequentist.zip")
    assert hashlib.sha256(written.read_bytes()).hexdigest() == SCHEMA_1_ZIP_SHA256
    restored = read_evidence_bundle(written)
    assert restored.manifest["schema_version"] == SUPERSEDED_BUNDLE_SCHEMA_VERSIONS[-1]
    assert replay_evidence_bundle(restored).posterior_paths == ()


def test_posterior_evidence_is_archived_and_survives_a_round_trip(tmp_path) -> None:
    evidence = bundle(posterior=posterior_evidence())

    assert evidence.schema_version == BUNDLE_SCHEMA_VERSION
    assert evidence.posterior_paths == (POSTERIOR_CONVERGENCE_PATH, POSTERIOR_LOO_PATH)
    assert evidence.bundle_id != SCHEMA_1_BUNDLE_ID
    restored = read_evidence_bundle(write_evidence_bundle(evidence, tmp_path / "posterior.zip"))
    assert restored.bundle_id == evidence.bundle_id
    convergence = json.loads(restored.file(POSTERIOR_CONVERGENCE_PATH).content)
    assert convergence["hierarchical-glm"]["status"] == "pass"
    replay = replay_evidence_bundle(restored)
    assert replay.schema_version == BUNDLE_SCHEMA_VERSION
    assert replay.loo_estimands == ("leave-one-observation-out", "leave-one-subject-out")


def test_archived_loo_keeps_differently_blocked_estimands_apart() -> None:
    evidence = bundle(posterior=posterior_evidence())

    archived = json.loads(evidence.file(POSTERIOR_LOO_PATH).content)
    assert set(archived) == {"leave-one-observation-out", "leave-one-subject-out"}
    assert archived["leave-one-observation-out"]["block"] is None
    assert archived["leave-one-subject-out"]["block"] == "subject"
    observation = archived["leave-one-observation-out"]["results"]
    assert set(observation) == {"hierarchical-glm/observation"}
    assert observation["hierarchical-glm/observation"]["estimand"] == "leave-one-observation-out"


def test_indistinguishable_loo_results_cannot_be_archived_together() -> None:
    with pytest.raises(EvidenceBundleError, match="cannot be told apart"):
        PosteriorEvidence(loo={"first": loo_result(), "second": loo_result()})


def test_bundle_comparison_reports_a_posterior_half_the_other_bundle_lacks() -> None:
    comparison = compare_evidence_bundles(bundle(), bundle(posterior=posterior_evidence()))

    assert not comparison.identical
    assert comparison.same_protocol
    assert not comparison.same_posterior_evidence
    assert comparison.added_posterior_paths == (POSTERIOR_CONVERGENCE_PATH, POSTERIOR_LOO_PATH)
    assert comparison.removed_posterior_paths == ()
    assert comparison.left_loo_estimands == ()
    assert comparison.right_loo_estimands == (
        "leave-one-observation-out",
        "leave-one-subject-out",
    )
    assert set(comparison.added_paths) == {POSTERIOR_CONVERGENCE_PATH, POSTERIOR_LOO_PATH}


def test_unknown_bundle_schema_version_is_refused(tmp_path) -> None:
    original = write_evidence_bundle(bundle(), tmp_path / "original.zip")
    forged = tmp_path / "forged.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(forged, "w") as target:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename == "bundle.json":
                manifest = json.loads(content)
                manifest["schema_version"] = "behavio.evidence-bundle/99"
                content = json.dumps(manifest).encode()
            target.writestr(info, content)

    with pytest.raises(EvidenceBundleError, match="unsupported bundle schema version"):
        read_evidence_bundle(forged)


def test_environment_records_absent_optional_dependencies_and_the_working_tree() -> None:
    captured = capture_environment(root=".")

    assert set(captured) == {"python", "platform", "packages", "source_control"}
    assert captured["packages"]["behavio"]
    for optional in ("arviz", "arviz-stats", "pymc", "pytensor", "pybads"):
        assert optional in captured["packages"]
    assert captured == capture_environment(root=".")
    source_control = captured["source_control"]
    assert source_control["system"] == "git"
    if source_control["available"]:
        assert len(source_control["commit"]) == 40
        assert isinstance(source_control["dirty"], bool)
    else:
        assert source_control["reason"]


def test_source_control_reports_a_directory_that_is_not_a_repository(tmp_path) -> None:
    outside = capture_source_control(tmp_path)

    assert outside == {
        "system": "git",
        "available": False,
        "reason": "not-a-git-repository",
    }
