"""A sampled candidate runs the whole protocol spine, and its convergence gates it.

The frozen declaration, the recovery gates, the bounded report and the content-addressed
evidence bundle are this package's central claim, and until now none of them were available
to a Bayesian model: :func:`~behavio.protocol.run_protocol` was typed and documented for
optimizer-fitted candidates, and a :class:`~behavio.protocol.CandidateSpec` had no way to say
that a candidate is sampled at all.

What closes it is one member -- :attr:`~behavio.protocol.CandidateSpec.inference` -- and the
observation that everything downstream already knows what to do. ``evaluate_splits``
dispatches on the estimator contract, samples, audits the posterior and projects it with a
``converged`` flag that *is* the convergence verdict; ``audit_fit`` turns a false flag into
``FitAuditStatus.FAIL``; and ``CandidateRun.eligible`` already reads that. So the tests here
check the two things that are genuinely new -- that the declaration is frozen and verified,
and that a failed convergence audit removes a sampled candidate exactly as a failed optimizer
audit removes an optimized one -- and then follow one sampled run all the way to a bundle.

The reference sampled estimator is the analytic Laplace GLM from
``tests/test_posterior_estimator_wiring.py``, reused rather than reinvented: its
``diverging=True`` variant produces a posterior the default audit rejects, which is what a
convergence gate needs to be shown refusing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from test_bounded_report import report_items
from test_compiler import source_study
from test_evidence_bundles import environment, figure
from test_posterior_estimator_wiring import LaplaceSampledGLM
from test_protocol import example_protocol
from test_runner import candidate_models, declared_glm

from behavio.contracts.posterior import any_model_capabilities
from behavio.diagnostics import FitAuditStatus
from behavio.evaluate.folds import PosteriorFoldPolicy
from behavio.evaluate.splits import cohort_forward_session_splits
from behavio.posterior.diagnostics import PosteriorAuditPolicy, PosteriorAuditStatus
from behavio.protocol.compiler import compile_execution_plan, materialize_protocol
from behavio.protocol.runner import (
    DeclarationCheck,
    ProtocolRunError,
    RankingStatus,
    run_protocol,
    verify_candidate_declarations,
)
from behavio.protocol.schema import (
    PROTOCOL_SCHEMA_VERSION,
    CandidateInference,
    CandidateSpec,
    ProtocolValidationError,
    Setting,
    protocol_from_dict,
)
from behavio.registry import EstimatorRegistry, builtin_estimator_registry
from behavio.report.bounded import generate_bounded_report
from behavio.report.evidence_bundles import (
    BUNDLE_SCHEMA_VERSION,
    POSTERIOR_CONVERGENCE_PATH,
    PosteriorEvidence,
    build_evidence_bundle,
    read_evidence_bundle,
    replay_evidence_bundle,
    write_evidence_bundle,
)

SAMPLED_IMPLEMENTATION = "mylab.models.LaplaceSampledGLM"


def declared_sampled(name: str = "sampled", **hyperparameters) -> CandidateSpec:
    """Declare the Laplace GLM candidate, and declare that it is sampled."""

    return CandidateSpec(
        name=name,
        implementation=SAMPLED_IMPLEMENTATION,
        hyperparameters=tuple(
            Setting(key, value) for key, value in sorted(hyperparameters.items())
        ),
        scored_columns=("choice",),
        inference=CandidateInference.SAMPLED,
    )


def sampled_registry() -> EstimatorRegistry:
    registry = builtin_estimator_registry()
    registry.add(
        SAMPLED_IMPLEMENTATION,
        LaplaceSampledGLM,
        provider="example-extension",
        version="1.0.0",
        produces=LaplaceSampledGLM,
    )
    return registry


def mixed_protocol(*, sampled: CandidateSpec | None = None):
    """A frozen protocol declaring one optimized and one sampled candidate."""

    protocol = example_protocol(
        with_recovery=False,
        candidates=(
            declared_glm("static", predictors=("stimulus",), choice_lags=0, l2=0.1),
            sampled if sampled is not None else declared_sampled(prior_precision=1.0),
        ),
    )
    return replace(
        protocol,
        cohort=replace(
            protocol.cohort,
            expected_subjects=2,
            expected_sessions=6,
            expected_observations=12,
        ),
        panel=replace(protocol.panel, minimum_sessions=3),
        comparison=replace(
            protocol.comparison, bootstrap_repetitions=50, reference_candidate="static"
        ),
    ).freeze()


def mixed_models(**overrides):
    return {
        "static": candidate_models()["static"],
        "sampled": LaplaceSampledGLM(prior_precision=1.0, **overrides),
    }


def compiled_mixed(*, sampled: CandidateSpec | None = None, models=None):
    materialized = materialize_protocol(mixed_protocol(sampled=sampled), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)
    declared = mixed_models() if models is None else models
    return compile_execution_plan(
        materialized,
        splits,
        capabilities={name: any_model_capabilities(model) for name, model in declared.items()},
    )


# --------------------------------------------------------------------------------------
# The declaration.
# --------------------------------------------------------------------------------------


def test_a_frozen_protocol_declares_which_candidates_are_sampled() -> None:
    protocol = mixed_protocol()

    optimized, sampled = protocol.candidates
    assert optimized.inference is CandidateInference.OPTIMIZED
    assert sampled.inference is CandidateInference.SAMPLED
    assert protocol.schema_version == PROTOCOL_SCHEMA_VERSION
    # The declaration is part of the content address, exactly as the multiplicity is: a
    # protocol whose candidate is sampled and one whose candidate is optimized are two
    # protocols, because different evidence decides whether each is eligible.
    draft = example_protocol(with_recovery=False, candidates=(optimized, sampled))
    relabelled = replace(
        draft,
        candidates=(optimized, replace(sampled, inference=CandidateInference.OPTIMIZED)),
    )
    assert relabelled.fingerprint != draft.fingerprint


def test_a_schema_that_predates_the_member_cannot_smuggle_a_sampled_candidate() -> None:
    with pytest.raises(ProtocolValidationError, match="predates the candidate inference"):
        replace(
            example_protocol(candidates=(declared_glm("static", l2=0.1), declared_sampled())),
            schema_version="behavio.study-protocol/2",
        )


def test_a_version_two_payload_still_loads_and_keeps_its_own_fingerprint() -> None:
    """Version 2 could express exactly one inference, so nothing was left unsaid by it.

    A protocol frozen before the member existed keeps its recorded content address, and
    round-trips byte for byte: ``inference`` is supplied on the way in and omitted on the
    way out, so the declaration nobody amended keeps the identity its freeze event quotes.
    """

    frozen = example_protocol().freeze()
    recorded = json.loads(frozen.canonical_json())
    # Exactly what a protocol frozen under version 2 looks like on disk.
    for candidate in recorded["candidates"]:
        del candidate["inference"]
    del recorded["comparison"]["metrics"]
    recorded["schema_version"] = "behavio.study-protocol/2"
    recorded["lifecycle"][0]["artifact_fingerprint"] = _recorded_fingerprint(recorded)

    restored = protocol_from_dict(recorded)

    assert restored.schema_version == "behavio.study-protocol/2"
    assert all(
        candidate.inference is CandidateInference.OPTIMIZED for candidate in restored.candidates
    )
    assert restored.fingerprint == recorded["lifecycle"][0]["artifact_fingerprint"]
    assert json.loads(restored.canonical_json()) == recorded


def _recorded_fingerprint(recorded: dict) -> str:
    scientific = {
        key: value for key, value in recorded.items() if key not in ("state", "lifecycle")
    }
    payload = json.dumps(
        scientific, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_the_declared_inference_is_verified_against_the_object_that_runs() -> None:
    protocol = mixed_protocol()

    verification = verify_candidate_declarations(
        protocol, mixed_models(), registry=sampled_registry()
    )

    inference = {
        item.candidate: next(finding for finding in item.findings if finding.subject == "inference")
        for item in verification
    }
    assert inference["static"].status is DeclarationCheck.VERIFIED
    assert inference["static"].declared == "optimized"
    assert inference["sampled"].status is DeclarationCheck.VERIFIED
    assert inference["sampled"].declared == "sampled"
    assert all(item.verified for item in verification)


def test_supplying_an_optimized_model_for_a_sampled_candidate_refuses_the_run() -> None:
    swapped = {"static": candidate_models()["static"], "sampled": candidate_models()["smooth"]}
    compiled = compiled_mixed(models=mixed_models())

    with pytest.raises(ProtocolRunError, match=r"sampled\.inference"):
        run_protocol(compiled, swapped)


def test_the_pre_fit_audit_refuses_a_runtime_whose_inference_differs() -> None:
    """The compiler says so before any fit, from the capability matrix alone."""

    materialized = materialize_protocol(mixed_protocol(), source_study())
    splits = cohort_forward_session_splits(materialized.study, min_train_sessions=2)

    compiled = compile_execution_plan(
        materialized,
        splits,
        capabilities={
            "static": any_model_capabilities(candidate_models()["static"]),
            "sampled": any_model_capabilities(candidate_models()["smooth"]),
        },
    )

    assert not compiled.plan.audit.passed
    mismatch = [issue for issue in compiled.plan.audit.issues if issue.code == "inference-mismatch"]
    assert [issue.candidate for issue in mismatch] == ["sampled"]
    assert "declares 'sampled'" in mismatch[0].message


def test_a_registry_resolves_a_sampled_implementation_like_any_other() -> None:
    registry = sampled_registry()

    model = registry.create(SAMPLED_IMPLEMENTATION, {"prior_precision": 2.0})

    assert isinstance(model, LaplaceSampledGLM)
    assert any_model_capabilities(model).is_sampled
    assert registry.verify(SAMPLED_IMPLEMENTATION, model) is True


# --------------------------------------------------------------------------------------
# The run, end to end.
# --------------------------------------------------------------------------------------


def test_a_sampled_candidate_runs_the_frozen_protocol_and_reaches_an_evidence_bundle(
    tmp_path,
) -> None:
    run = run_protocol(compiled_mixed(), mixed_models())

    sampled = next(candidate for candidate in run.report.candidates if candidate.name == "sampled")
    optimized = next(candidate for candidate in run.report.candidates if candidate.name == "static")
    assert sampled.eligible
    assert sampled.audit_status is FitAuditStatus.PASS
    assert run.report.ranking.status is not RankingStatus.NO_ELIGIBLE_CANDIDATE
    # Every fold carries the convergence audit that decided its projected fit, and the
    # optimized candidate carries none, so the two are never confused in the archive.
    assert [identifier for identifier, _ in sampled.posterior_folds] == [
        fold.identifier for fold in sampled.folds
    ]
    assert optimized.posterior_folds == ()
    payload = run.report.to_dict()
    json.dumps(payload, allow_nan=False)
    folds = payload["candidates"]["sampled"]["folds"]
    assert all(fold["posterior"]["convergence_audit"]["status"] == "pass" for fold in folds)
    assert all(fold["posterior"]["converged"] for fold in folds)
    assert "posterior" not in payload["candidates"]["static"]["folds"][0]

    reported = generate_bounded_report(run, items=report_items())
    bundle = build_evidence_bundle(
        reported,
        figures=(figure(),),
        environment=environment(),
        posterior=PosteriorEvidence(
            convergence={
                f"sampled/{identifier}": evidence.audit
                for identifier, evidence in sampled.posterior_folds
            }
        ),
    )

    assert bundle.schema_version == BUNDLE_SCHEMA_VERSION
    assert bundle.posterior_paths == (POSTERIOR_CONVERGENCE_PATH,)
    archived = json.loads(bundle.file(POSTERIOR_CONVERGENCE_PATH).content)
    assert set(archived) == {f"sampled/{identifier}" for identifier, _ in sampled.posterior_folds}
    assert all(entry["status"] == "pass" for entry in archived.values())
    restored = read_evidence_bundle(write_evidence_bundle(bundle, tmp_path / "sampled.zip"))
    assert replay_evidence_bundle(restored).posterior_paths == (POSTERIOR_CONVERGENCE_PATH,)


def test_a_failed_convergence_audit_makes_a_sampled_candidate_ineligible() -> None:
    """The sampled counterpart of ``FitAuditStatus.FAIL`` removing an optimized candidate.

    Nothing here is special-cased for a sampler: the divergent posterior is projected with
    ``converged=False``, ``audit_fit`` raises ``optimizer_nonconvergence`` on the projection,
    the fold's audit fails, and the candidate is dropped from ranking by the same property
    that drops a non-convergent optimizer. The score is still computed and still reported --
    silently omitting its rows would break the equal-unit pairing that makes the comparison
    readable at all -- and marked unusable instead.
    """

    diverging = mixed_models(diverging=True)

    run = run_protocol(compiled_mixed(models=diverging), diverging)

    sampled = next(candidate for candidate in run.report.candidates if candidate.name == "sampled")
    assert not sampled.eligible
    assert sampled.audit_status is FitAuditStatus.FAIL
    assert sampled.score is not None
    assert not sampled.failures
    assert all(
        evidence.status is PosteriorAuditStatus.FAIL for _, evidence in sampled.posterior_folds
    )
    assert all(not fold.fit.diagnostics.converged for fold in sampled.folds)
    assert any(
        "optimizer_nonconvergence" in {issue.code for issue in fold.fit_audit.issues}
        for fold in sampled.folds
    )
    # It is out of the ranking, and out of the paired comparisons, but still in the report.
    assert run.report.ranking.eligible_candidates == ("static",)
    assert run.report.paired_comparisons == ()
    payload = run.report.to_dict()
    assert payload["candidates"]["sampled"]["eligible"] is False
    assert payload["candidates"]["sampled"]["audit_status"] == "fail"
    codes = payload["candidates"]["sampled"]["folds"][0]["posterior"]["convergence_audit"]
    assert "posterior.divergences" in {issue["code"] for issue in codes["issues"]}


def test_the_declared_posterior_policy_decides_the_gate_and_is_recorded() -> None:
    """A stricter gate is a different verdict, and the archive says which one was applied."""

    strict = PosteriorFoldPolicy(audit_policy=PosteriorAuditPolicy(min_ess_bulk=1e9))
    compiled = compiled_mixed()

    default_run = run_protocol(compiled_mixed(), mixed_models())
    strict_run = run_protocol(compiled, mixed_models(), posterior_policy=strict)

    default_sampled = next(
        candidate for candidate in default_run.report.candidates if candidate.name == "sampled"
    )
    strict_sampled = next(
        candidate for candidate in strict_run.report.candidates if candidate.name == "sampled"
    )
    # Low ESS is a precision finding rather than a validity one, so the stricter policy warns
    # and does not disqualify; what must differ is the recorded policy, not the verdict.
    assert default_sampled.eligible and strict_sampled.eligible
    assert strict_sampled.audit_status is FitAuditStatus.PASS
    archived = strict_run.report.to_dict()["candidates"]["sampled"]["folds"][0]["posterior"]
    assert archived["convergence_audit"]["policy"]["min_ess_bulk"] == 1e9
    assert archived["convergence_audit"]["status"] == "warning"
    assert (
        default_run.report.to_dict()["candidates"]["sampled"]["folds"][0]["posterior"][
            "convergence_audit"
        ]["policy"]["min_ess_bulk"]
        == PosteriorAuditPolicy().min_ess_bulk
    )
