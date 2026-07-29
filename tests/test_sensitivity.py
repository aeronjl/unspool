import json

import numpy as np
import pytest

from behavio import (
    PosteriorAuditPolicy,
    PosteriorGroup,
    PosteriorResult,
    PosteriorVariable,
    SensitivityMetric,
    SensitivityOutcome,
    SensitivityScenario,
    posterior_sensitivity_outcome,
    run_sensitivity_analysis,
)
from behavio.contracts import AuditSeverity
from behavio.sensitivity import SensitivityError


def unaudited_policy() -> PosteriorAuditPolicy:
    """Declare, check by check, that unusable draws are being accepted anyway."""

    return PosteriorAuditPolicy(
        divergence_severity=AuditSeverity.WARNING,
        rhat_severity=AuditSeverity.WARNING,
        nonfinite_severity=AuditSeverity.WARNING,
    )


def sampled_result(*, unconverged: bool = False) -> PosteriorResult:
    generator = np.random.default_rng(404)
    chains, draws = 4, 500
    values = generator.normal(size=(chains, draws, 2)) + np.asarray([0.5, 1.5])
    diverging = np.zeros((chains, draws), dtype=bool)
    if unconverged:
        values[0, :, 0] += 9.0
        diverging[0, :23] = True
    coords = {
        "chain": np.arange(chains),
        "draw": np.arange(draws),
        "subject": np.asarray(["a", "b"]),
    }
    posterior = PosteriorGroup(
        "posterior",
        (PosteriorVariable("bias", values, ("chain", "draw", "subject"), coords),),
    )
    sample_stats = PosteriorGroup(
        "sample_stats",
        (
            PosteriorVariable(
                "diverging",
                diverging,
                ("chain", "draw"),
                {"chain": coords["chain"], "draw": coords["draw"]},
            ),
        ),
    )
    return PosteriorResult(
        model_name="hierarchical-bias",
        model_signature="hierarchical-bias[v1]",
        inference_library="test",
        inference_library_version="1",
        parameter_names=("bias",),
        groups=(posterior, sample_stats),
        parameter_space_fingerprint="sha256:test",
    )


def scenarios() -> tuple[SensitivityScenario, ...]:
    return (
        SensitivityScenario(
            "reference",
            "normal-prior[scale=1]",
            {},
            is_reference=True,
        ),
        SensitivityScenario(
            "wider prior",
            "normal-prior[scale=2]",
            {"prior_scale": 2.0},
        ),
        SensitivityScenario(
            "short history",
            "normal-prior[scale=1;history=2]",
            {"history_lags": 2},
        ),
    )


def scalar_analysis(scenario: SensitivityScenario, seed: int) -> SensitivityOutcome:
    estimates = {"reference": 0.40, "wider prior": 0.46, "short history": 0.35}
    return SensitivityOutcome(
        artifact_signature=f"fit[{scenario.signature}]",
        metrics=(
            SensitivityMetric(
                "future log loss",
                "future-log-loss[v1;unit=subject]",
                estimates[scenario.name],
                interval=(estimates[scenario.name] - 0.02, estimates[scenario.name] + 0.02),
                unit="nats/trial",
            ),
        ),
        diagnostic_codes=("boundary",) if scenario.name == "short history" else (),
        provenance={"seed": seed},
    )


def test_sensitivity_runner_retains_reference_contrasts_and_spread() -> None:
    report = run_sensitivity_analysis(
        scenarios(),
        scalar_analysis,
        seed=71,
        analysis_signature="future-score-sensitivity[v1]",
    )

    assert report.is_interpretable
    assert report.n_successful == 3
    assert report.n_failed == 0
    assert report.success_rate == 1.0
    assert [item.scenario_name for item in report.contrasts] == [
        "wider prior",
        "short history",
    ]
    assert [item.difference for item in report.contrasts] == pytest.approx([0.06, -0.05])
    summary = report.summary()[0]
    assert summary.reference_estimate == pytest.approx(0.40)
    assert summary.minimum == pytest.approx(0.35)
    assert summary.maximum == pytest.approx(0.46)
    assert summary.range == pytest.approx(0.11)
    assert summary.maximum_absolute_difference == pytest.approx(0.06)
    assert summary.largest_difference_scenario == "wider prior"
    json.dumps(report.to_dict(), allow_nan=False)


def test_sensitivity_seeds_are_reproducible_and_scenario_stable() -> None:
    first = run_sensitivity_analysis(
        scenarios(),
        scalar_analysis,
        seed=18,
        analysis_signature="stable-seeds[v1]",
    )
    reordered = run_sensitivity_analysis(
        tuple(reversed(scenarios())),
        scalar_analysis,
        seed=18,
        analysis_signature="stable-seeds[v1]",
    )

    first_seeds = {run.scenario.signature: run.seed for run in first.runs}
    reordered_seeds = {run.scenario.signature: run.seed for run in reordered.runs}
    assert first_seeds == reordered_seeds
    assert len(set(first_seeds.values())) == 3


def test_sensitivity_retains_analysis_and_metric_contract_failures() -> None:
    declared = (
        *scenarios(),
        SensitivityScenario("broken", "broken[v1]", {"sampler": "broken"}),
    )

    def analysis(scenario: SensitivityScenario, seed: int) -> SensitivityOutcome:
        if scenario.name == "broken":
            raise RuntimeError()
        outcome = scalar_analysis(scenario, seed)
        if scenario.name == "short history":
            return SensitivityOutcome(
                artifact_signature=outcome.artifact_signature,
                metrics=(SensitivityMetric("accuracy", "accuracy[v1]", 0.8),),
            )
        return outcome

    report = run_sensitivity_analysis(
        declared,
        analysis,
        seed=4,
        analysis_signature="failure-retention[v1]",
    )

    assert report.n_successful == 2
    assert report.n_failed == 2
    failures = {run.scenario.name: run.failure for run in report.runs if run.failure}
    assert failures["short history"].stage == "evaluation"
    assert "changed relative" in failures["short history"].message
    assert failures["broken"].stage == "analysis"
    assert failures["broken"].message == "<exception had no message>"
    assert len(report.contrasts) == 1


def test_failed_reference_is_retained_without_fabricating_contrasts() -> None:
    def analysis(scenario: SensitivityScenario, seed: int) -> SensitivityOutcome:
        if scenario.is_reference:
            raise RuntimeError("reference did not converge")
        return scalar_analysis(scenario, seed)

    report = run_sensitivity_analysis(
        scenarios(),
        analysis,
        seed=9,
        analysis_signature="failed-reference[v1]",
    )

    assert not report.is_interpretable
    assert report.reference.failure is not None
    assert report.reference.failure.stage == "analysis"
    assert report.contrasts == ()
    assert report.summary() == ()


def test_posterior_sensitivity_outcome_preserves_labelled_parameters() -> None:
    pytest.importorskip("arviz")
    outcome = posterior_sensitivity_outcome(
        sampled_result(),
        interval_probability=0.8,
        diagnostic_codes=("prior-sensitivity-checked",),
    )

    assert [metric.target for metric in outcome.metrics] == [
        "bias[subject='a']",
        "bias[subject='b']",
    ]
    assert [metric.estimate for metric in outcome.metrics] == pytest.approx([0.5, 1.5], abs=0.05)
    assert all(metric.interval is not None for metric in outcome.metrics)
    assert outcome.diagnostic_codes == ("prior-sensitivity-checked",)
    assert outcome.provenance["parameter_space_fingerprint"] == "sha256:test"


def test_unconverged_posterior_refuses_to_become_a_sensitivity_metric() -> None:
    pytest.importorskip("arviz")
    with pytest.raises(SensitivityError, match="failed its convergence audit"):
        posterior_sensitivity_outcome(sampled_result(unconverged=True))


def test_a_refused_posterior_is_retained_as_a_scenario_failure() -> None:
    pytest.importorskip("arviz")

    def analysis(scenario: SensitivityScenario, seed: int) -> SensitivityOutcome:
        return posterior_sensitivity_outcome(
            sampled_result(unconverged=not scenario.is_reference),
            artifact_signature=f"fit[{scenario.signature}]",
        )

    report = run_sensitivity_analysis(
        scenarios(),
        analysis,
        seed=12,
        analysis_signature="posterior-convergence-gate[v1]",
    )

    assert report.is_interpretable
    assert report.n_successful == 1
    assert report.n_failed == 2
    failed = next(run for run in report.runs if run.failure is not None)
    assert failed.failure is not None
    assert failed.failure.stage == "analysis"
    assert failed.failure.error_type == "SensitivityError"
    assert "posterior.rhat" in failed.failure.message
    assert report.contrasts == ()
    json.dumps(report.to_dict(), allow_nan=False)


def test_the_convergence_gate_is_injectable_and_its_concession_is_recorded() -> None:
    pytest.importorskip("arviz")
    outcome = posterior_sensitivity_outcome(
        sampled_result(unconverged=True),
        audit_policy=unaudited_policy(),
    )

    assert "posterior.rhat" in outcome.diagnostic_codes
    assert "posterior.divergences" in outcome.diagnostic_codes
    assert len(set(outcome.diagnostic_codes)) == len(outcome.diagnostic_codes)

    with pytest.raises(TypeError, match="audit_policy"):
        posterior_sensitivity_outcome(sampled_result(), audit_policy="skip")  # type: ignore[arg-type]


def test_sensitivity_contracts_reject_ambiguous_declarations() -> None:
    with pytest.raises(ValueError, match="at least one change"):
        SensitivityScenario("alternative", "alternative[v1]", {})
    with pytest.raises(ValueError, match="exactly one reference"):
        run_sensitivity_analysis(
            scenarios()[1:],
            scalar_analysis,
            seed=1,
            analysis_signature="invalid[v1]",
        )
    scenario = scenarios()[1]
    with pytest.raises(TypeError):
        scenario.changes["prior_scale"] = 3.0  # type: ignore[index]
