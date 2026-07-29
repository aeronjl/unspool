import json
from pathlib import Path

import pytest

from benchmarks.ddm_predictive_uncertainty.benchmark import run

RESULT_PATH = Path("benchmarks/ddm_predictive_uncertainty/result.json")


def test_ddm_predictive_uncertainty_benchmark_validates_cheap_configuration_errors() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        run(repetitions=0)
    with pytest.raises(ValueError, match="seed"):
        run(repetitions=1, seed=-1)


def test_pinned_ddm_predictive_result_retains_stability_and_monte_carlo_precision() -> None:
    """Debiased simulation made every panel's EM map stable; the refusal path still holds."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["repetitions"] == 20
    assert result["prediction"]["n_unseen_subjects"] == 80
    assert result["diagnostics"]["all_outer_fits_converged"]
    assert result["diagnostics"]["supplemented_interval_successes"] == 20
    assert result["diagnostics"]["supplemented_interval_failures"] == 0
    assert result["diagnostics"]["maximum_scale_em_spectral_radius"] < 1.0
    assert result["prediction"]["mean_joint_log_probability_improvement"] > 0
    assert result["prediction"]["random_effect_win_rate"] > 0.5
    assert 1 <= result["prediction"]["minimum_effective_draws"] <= 4096
    assert result["prediction"]["maximum_log_probability_mcse"] == pytest.approx(0.194634, abs=1e-6)

    for summary in result["intervals"].values():
        assert summary["supplemented_interval_denominator"] == 20
        assert summary["supplemented_coverage"] >= summary["local_coverage"]

    failures = 0
    for panel in result["runs"]:
        failed = panel["supplemented_interval_error"] is not None
        failures += failed
        assert (panel["supplemented_intervals"] is None) is failed
        assert (panel["scale_em_spectral_radius"] is None) is failed
        assert len(panel["predictive_effective_draws"]) == 4
        assert len(panel["predictive_log_probability_mcse"]) == 4
    assert failures == 0
