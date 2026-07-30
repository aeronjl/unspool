"""Structural tests for the optional plotting layer.

Figures are checked by their artists, data, limits, and labels rather than by pixels: an
image comparison would fail on a font-metric change that alters no plotted value, which is
exactly the noise the figure standard already treats as out of scope.

The suite never needs a display. Plot functions build ``matplotlib.figure.Figure`` objects
directly rather than through ``pyplot``, so nothing is registered globally and nothing leaks;
the Agg backend is selected anyway for the few places ``pyplot`` is touched.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="plotting requires behavio[plots]")
matplotlib.use("Agg")

from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from behavio.compare.models import ComparisonFamily, ComparisonMultiplicity  # noqa: E402
from behavio.contracts.audit import AuditSeverity, FitAudit, FitDiagnostics, FitIssue  # noqa: E402
from behavio.plot import (  # noqa: E402
    FIGURE_RC_PARAMS,
    MatplotlibUnavailableError,
    configure_figure_style,
    figure_style,
    plot_calibration,
    plot_convergence,
    plot_elpd_differences,
    plot_ess,
    plot_parameter_recovery,
    plot_parameter_recovery_grid,
    plot_pareto_k,
    plot_predictive_check,
    plot_predictive_checks,
    plot_rhat,
    plot_sbc_ecdf_difference,
    plot_sbc_rank_histogram,
    save_svg,
)
from behavio.posterior.comparison import (  # noqa: E402
    ModelComparisonIssue,
    ModelComparisonStatus,
    PairedELPDDifference,
    PosteriorModelComparison,
    ScoredModel,
)
from behavio.posterior.diagnostics import (  # noqa: E402
    PosteriorAudit,
    PosteriorAuditIssue,
    PosteriorAuditPolicy,
    PosteriorAuditStatus,
    PosteriorDiagnostic,
)
from behavio.posterior.loo import PSISLOOIssue, PSISLOOResult  # noqa: E402
from behavio.posterior.predictive import (  # noqa: E402
    PosteriorPredictiveAudit,
    PosteriorPredictiveCheck,
    PosteriorPredictiveIssue,
    PosteriorPredictivePolicy,
    PredictiveFamily,
    PredictiveMultiplicity,
)
from behavio.posterior.simulation_based_calibration import SBCSummary, SBCUniformity  # noqa: E402
from behavio.protocol.runner import CalibrationSummary  # noqa: E402
from behavio.recovery.parameters import (  # noqa: E402
    POSTERIOR_QUANTILE_INTERVAL,
    WALD_INTERVAL,
    ParameterRecoveryReport,
)

PLOT_PACKAGE = Path(__file__).parents[1] / "src" / "behavio" / "plot"


def figure_text(figure) -> str:
    return "\n".join(artist.get_text() for artist in figure.findobj(Text))


# --------------------------------------------------------------------------------------
# report fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def sbc_summary() -> SBCSummary:
    return SBCSummary(
        target="theta",
        n_replicates=40,
        mean_normalized_rank=0.51,
        interval_coverage=0.9,
        histogram_counts=(3, 5, 4, 4, 4, 4, 4, 4, 4, 4),
        repeats_requested=50,
        n_unconverged=6,
        n_other_failures=4,
        mean_relative_ess=0.8,
        min_relative_ess=0.5,
    )


@pytest.fixture
def sbc_uniformity() -> SBCUniformity:
    points = np.linspace(0.0, 1.0, 11)
    difference = np.full(points.size, 0.02)
    difference[5] = 0.2
    lower = np.full(points.size, -0.1)
    upper = np.full(points.size, 0.1)
    outside = int(np.count_nonzero((difference < lower) | (difference > upper)))
    return SBCUniformity(
        target="theta",
        n_replicates=40,
        null="discrete-uniform",
        n_posterior_draws=999,
        confidence_level=0.95,
        pointwise_level=0.01,
        n_band_simulations=2000,
        band_seed=0,
        evaluation_points=tuple(points),
        null_cdf=tuple(points),
        ecdf_difference=tuple(difference),
        lower_difference_band=tuple(lower),
        upper_difference_band=tuple(upper),
        n_points_outside_band=outside,
        bins=10,
        chi_square=8.5,
        chi_square_dof=9,
        chi_square_p_value=0.48,
        min_expected_bin_count=4.0,
    )


def make_loo(*, issues: tuple[PSISLOOIssue, ...] = ()) -> PSISLOOResult:
    labels = np.array([f"s{index}" for index in range(6)])
    pareto = np.array([0.2, 0.3, 0.9, 0.1, 0.4, 1.2])
    return PSISLOOResult(
        model_name="glm",
        model_signature="glm@1",
        inference_library="arviz",
        inference_library_version="1.2.0",
        log_likelihood_name="y",
        dims=("subject",),
        coords={"subject": labels},
        elpd_loo=-120.5,
        se=8.25,
        p_loo=4.5,
        n_samples=4000,
        n_data_points=6,
        good_k=0.7,
        pointwise_elpd=np.linspace(-25.0, -15.0, 6),
        pareto_k=pareto,
        issues=issues,
        block="subject",
    )


def make_scored(name: str, elpd: float, status: PosteriorAuditStatus) -> ScoredModel:
    return ScoredModel(
        name=name,
        model_signature=f"{name}@1",
        log_likelihood_name="y",
        elpd_loo=elpd,
        se=6.0,
        p_loo=3.0,
        n_data_points=12,
        max_pareto_k=0.4,
        good_k=0.7,
        status=status,
    )


def make_comparison(
    *,
    status: ModelComparisonStatus = ModelComparisonStatus.UNRESOLVED,
    best_model: str | None = None,
    decisive: bool = False,
    ineligible: bool = False,
) -> PosteriorModelComparison:
    difference = PairedELPDDifference(
        left_model="rich",
        right_model="plain",
        elpd_difference=12.0,
        se=3.0,
        lower=6.0 if decisive else -4.0,
        upper=18.0 if decisive else 28.0,
        interval_scale=2.0,
        n_data_points=12,
        two_sided_probability=0.0001 if decisive else 0.4,
        adjusted_probability=0.0003 if decisive else 0.6,
        decisive=decisive,
    )
    second = PairedELPDDifference(
        left_model="rich",
        right_model="null",
        elpd_difference=4.0,
        se=5.0,
        lower=-6.0,
        upper=14.0,
        interval_scale=2.0,
        n_data_points=12,
        two_sided_probability=0.42,
        adjusted_probability=0.63,
    )
    third = PairedELPDDifference(
        left_model="plain",
        right_model="null",
        elpd_difference=-8.0,
        se=4.0,
        lower=-16.0,
        upper=0.0,
        interval_scale=2.0,
        n_data_points=12,
        two_sided_probability=0.0455,
        adjusted_probability=0.1365,
    )
    third_status = PosteriorAuditStatus.FAIL if ineligible else PosteriorAuditStatus.PASS
    return PosteriorModelComparison(
        block="subject",
        estimand="leave-one-subject-out",
        dims=("subject",),
        coords={"subject": np.arange(12)},
        n_data_points=12,
        interval_scale=2.0,
        models=(
            make_scored("rich", -100.0, PosteriorAuditStatus.PASS),
            make_scored("plain", -112.0, PosteriorAuditStatus.PASS),
            make_scored("null", -104.0, third_status),
        ),
        differences=(difference, second, third),
        status=status,
        best_model=best_model,
        reason="the paired interval does not exclude zero",
        family=ComparisonFamily(
            n_candidates=2 if ineligible else 3,
            n_comparisons=1 if ineligible else 3,
            interval_level=0.9545,
            multiplicity=ComparisonMultiplicity.BENJAMINI_HOCHBERG,
            family_error_rate=0.0455,
            n_separated=1 if decisive else 0,
            expected_separated=0.1365,
            excess_probability=0.13 if decisive else 1.0,
            adjusted_threshold=0.0001 if decisive else 0.0,
            n_decisive=1 if decisive else 0,
        ),
        issues=(
            (ModelComparisonIssue(code="comparison.ineligible-model", message="null failed"),)
            if ineligible
            else ()
        ),
    )


def make_check(
    *, group: tuple[tuple[str, object], ...] = (("lab", "A"),), observed: float = 0.5
) -> PosteriorPredictiveCheck:
    rng = np.random.default_rng(11)
    replicated = rng.normal(0.5, 0.1, size=(4, 250))
    return PosteriorPredictiveCheck(
        discrepancy_name="mean",
        discrepancy_signature="mean@two-sided",
        tail="two-sided",
        group=group,
        n_observations=180,
        observed=observed,
        replicated=replicated,
        interval=(0.31, 0.69),
        lower_probability=0.4,
        upper_probability=0.6,
        tail_probability=0.8,
    )


def make_predictive_audit(*, failed: bool = False) -> PosteriorPredictiveAudit:
    checks = (
        make_check(group=(("lab", "A"),), observed=0.5),
        make_check(group=(("lab", "B"),), observed=0.95),
    )
    issues = (
        (
            PosteriorPredictiveIssue(
                code="predictive.unconverged-posterior",
                message="the posterior failed its convergence audit",
                severity=AuditSeverity.ERROR,
            ),
        )
        if failed
        else ()
    )
    return PosteriorPredictiveAudit(
        model_name="glm",
        model_signature="glm@1",
        variable_name="y",
        policy=PosteriorPredictivePolicy(),
        checks=checks,
        issues=issues,
        family=PredictiveFamily(
            n_checks=2,
            n_groups=2,
            n_discrepancies=1,
            tail_probability_warning=0.05,
            multiplicity=PredictiveMultiplicity.BENJAMINI_HOCHBERG,
            family_discovery_rate=0.05,
            n_extreme=1,
            expected_extreme=0.1,
            excess_probability=0.0975,
            adjusted_threshold=0.025,
            n_flagged=1,
        ),
    )


def make_audit(*, code: str, severity: AuditSeverity) -> FitAudit:
    return FitAudit(
        model_name="glm",
        model_signature="glm@1",
        n_observations=200,
        numerical=FitDiagnostics(
            converged=severity is AuditSeverity.WARNING,
            optimizer="scipy",
            status=0,
            message="ok",
            n_iterations=20,
            objective=-1.0,
            gradient_norm=1e-6,
            hessian_condition=10.0,
            boundary_estimate=False,
        ),
        issues=(FitIssue(code=code, severity=severity, message="an issue"),),
    )


def clean_audit() -> FitAudit:
    return FitAudit(
        model_name="glm",
        model_signature="glm@1",
        n_observations=200,
        numerical=FitDiagnostics(
            converged=True,
            optimizer="scipy",
            status=0,
            message="ok",
            n_iterations=20,
            objective=-1.0,
            gradient_norm=1e-6,
            hessian_condition=10.0,
            boundary_estimate=False,
        ),
        issues=(),
    )


def make_recovery(*, kind: str = WALD_INTERVAL, failures: int = 1) -> ParameterRecoveryReport:
    rng = np.random.default_rng(3)
    n_runs = 8
    truth = np.stack([np.linspace(-1.0, 1.0, n_runs), np.linspace(0.5, 2.0, n_runs)], axis=1)
    estimates = truth + rng.normal(0.0, 0.05, size=truth.shape)
    errors = np.full(truth.shape, 0.1)
    audits = [clean_audit() for _ in range(n_runs)]
    for index in range(failures):
        audits[index] = make_audit(code="fit.failed", severity=AuditSeverity.ERROR)
    common = {
        "model_name": "glm",
        "model_signature": "glm@1",
        "parameter_names": ("bias", "slope"),
        "true_values": truth,
        "estimates": estimates,
        "standard_errors": errors,
        "converged": np.ones(n_runs, dtype=np.bool_),
        "messages": tuple("ok" for _ in range(n_runs)),
        "audits": tuple(audits),
        "seeds": np.arange(n_runs, dtype=np.uint64),
        "n_trials": 200,
        "n_subjects": 4,
        "repeats": 1,
        "root_seed": 7,
    }
    if kind == POSTERIOR_QUANTILE_INTERVAL:
        return ParameterRecoveryReport(
            **common,
            interval_kind=POSTERIOR_QUANTILE_INTERVAL,
            interval_lower=estimates - 0.2,
            interval_upper=estimates + 0.2,
            posterior_audits=tuple(None for _ in range(n_runs)),
        )
    return ParameterRecoveryReport(**common)


def make_posterior_audit(*, failing: bool = False) -> PosteriorAudit:
    policy = PosteriorAuditPolicy()
    diagnostics = (
        PosteriorDiagnostic(
            name="beta",
            dims=("predictor",),
            coords={"predictor": np.array(["bias", "stimulus"])},
            rhat=np.array([1.002, 1.05 if failing else 1.004]),
            ess_bulk=np.array([1200.0, 150.0 if failing else 900.0]),
            ess_tail=np.array([1100.0, 800.0]),
        ),
        PosteriorDiagnostic(
            name="sigma",
            dims=(),
            coords={},
            rhat=1.001,
            ess_bulk=1500.0,
            ess_tail=1400.0,
        ),
    )
    issues = (
        (
            PosteriorAuditIssue(
                code="posterior.rhat",
                message="R-hat above policy",
                targets=("beta[predictor='stimulus']",),
                severity=AuditSeverity.ERROR,
            ),
        )
        if failing
        else ()
    )
    return PosteriorAudit(
        model_name="glm",
        model_signature="glm@1",
        inference_library="pymc",
        inference_library_version="6.1.0",
        n_chains=4,
        n_draws=1000,
        divergences=12 if failing else 0,
        max_treedepth_hits=0,
        policy=policy,
        diagnostics=diagnostics,
        issues=issues,
    )


# --------------------------------------------------------------------------------------
# optional dependency
# --------------------------------------------------------------------------------------


def test_no_plot_module_imports_matplotlib_at_module_scope() -> None:
    offenders = []
    for path in sorted(PLOT_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] == "matplotlib" for alias in node.names
            ):
                offenders.append(path.name)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("matplotlib"):
                offenders.append(path.name)
    assert offenders == []


def test_missing_matplotlib_raises_an_import_error_naming_the_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    summary = CalibrationSummary(False, 0, None, None, None, None, "no successful predictions")

    with pytest.raises(MatplotlibUnavailableError) as excinfo:
        plot_calibration(summary)

    assert issubclass(MatplotlibUnavailableError, ImportError)
    assert "behavio[plots]" in str(excinfo.value)


def test_an_old_matplotlib_is_refused_with_its_own_version(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(__version__="3.5.1"))
    summary = CalibrationSummary(False, 0, None, None, None, None, "no successful predictions")

    with pytest.raises(MatplotlibUnavailableError) as excinfo:
        plot_calibration(summary)

    assert "3.5.1" in str(excinfo.value)
    assert "3.9" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# figure standard
# --------------------------------------------------------------------------------------


def test_figure_style_restores_the_previous_rcparams() -> None:
    before = matplotlib.rcParams["font.size"]

    with figure_style(**{"font.size": 33}):
        assert matplotlib.rcParams["font.size"] == 33

    assert matplotlib.rcParams["font.size"] == before


def test_configure_figure_style_applies_the_frozen_hash_salt() -> None:
    with matplotlib.rc_context():
        configure_figure_style()
        assert matplotlib.rcParams["svg.hashsalt"] == "unspool-documentation-v1"
        assert matplotlib.rcParams["svg.fonttype"] == "none"
        assert matplotlib.rcParams["font.sans-serif"][0] == "DejaVu Sans"
    assert FIGURE_RC_PARAMS["svg.hashsalt"] == "unspool-documentation-v1"


def test_save_svg_writes_searchable_text_without_trailing_whitespace(tmp_path, sbc_summary) -> None:
    figure = plot_sbc_rank_histogram(sbc_summary)
    destination = tmp_path / "sbc.svg"

    save_svg(figure, destination)

    body = destination.read_text(encoding="utf-8")
    assert "<text" in body
    assert all(line == line.rstrip() for line in body.splitlines())


def test_plot_functions_do_not_register_figures_with_pyplot(sbc_summary) -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close("all")

    plot_sbc_rank_histogram(sbc_summary)

    assert pyplot.get_fignums() == []


# --------------------------------------------------------------------------------------
# simulation-based calibration
# --------------------------------------------------------------------------------------


def test_rank_histogram_draws_one_bar_per_bin_against_the_uniform_expectation(
    sbc_summary,
) -> None:
    figure = plot_sbc_rank_histogram(sbc_summary)
    axes = figure.axes[0]

    heights = [patch.get_height() for patch in axes.patches]
    assert heights == list(sbc_summary.histogram_counts)
    assert axes.get_xlim() == (0.0, 1.0)
    reference = [line.get_ydata()[0] for line in axes.lines]
    assert pytest.approx(sbc_summary.expected_bin_count) in reference


def test_rank_histogram_surfaces_the_replicate_accounting(sbc_summary) -> None:
    text = figure_text(plot_sbc_rank_histogram(sbc_summary))

    assert "40/50 replicates retained" in text
    assert "6 unconverged" in text
    assert "4 other failures" in text


def test_ecdf_difference_draws_the_band_as_a_band(sbc_uniformity) -> None:
    figure = plot_sbc_ecdf_difference(sbc_uniformity)
    axes = figure.axes[0]

    bands = [item for item in axes.collections if isinstance(item, PolyCollection)]
    assert len(bands) == 1
    assert "simultaneous band" in figure_text(figure)


def test_ecdf_difference_plots_the_reported_curve_and_marks_exceedances(
    sbc_uniformity,
) -> None:
    figure = plot_sbc_ecdf_difference(sbc_uniformity)
    axes = figure.axes[0]

    curves = {tuple(np.round(line.get_ydata(), 6)) for line in axes.lines}
    assert tuple(np.round(sbc_uniformity.ecdf_difference, 6)) in curves
    marked = [line for line in axes.lines if line.get_marker() == "x"]
    assert len(marked) == 1
    assert len(marked[0].get_xdata()) == sbc_uniformity.n_points_outside_band
    assert "outside band (1 points)" in figure_text(figure)


def test_ecdf_difference_reports_the_null_and_its_test_statistics(sbc_uniformity) -> None:
    text = figure_text(plot_sbc_ecdf_difference(sbc_uniformity))

    assert "discrete-uniform" in text
    assert "chi-square 8.50 on 9 df" in text
    assert "seed 0" in text


# --------------------------------------------------------------------------------------
# PSIS-LOO
# --------------------------------------------------------------------------------------


def test_pareto_k_marks_the_good_k_threshold_and_the_points_above_it() -> None:
    result = make_loo()
    figure = plot_pareto_k(result)
    axes = figure.axes[0]

    thresholds = [line.get_ydata()[0] for line in axes.lines if line.get_linestyle() == "--"]
    assert pytest.approx(result.good_k) in thresholds
    above = [line for line in axes.lines if line.get_marker() == "^"]
    assert len(above[0].get_xdata()) == 2
    assert "k > good_k (2)" in figure_text(figure)


def test_pareto_k_names_the_block_estimand_and_its_coordinates() -> None:
    figure = plot_pareto_k(make_loo())
    axes = figure.axes[0]

    assert "leave-one-subject-out" in axes.get_title()
    assert "subject" in axes.get_xlabel()
    assert [label.get_text() for label in axes.get_xticklabels()][:2] == ["s0", "s1"]


def test_a_failed_psis_loo_result_is_watermarked() -> None:
    result = make_loo(
        issues=(
            PSISLOOIssue(
                code="psis.unconverged-posterior",
                message="the posterior failed its convergence audit",
                severity=AuditSeverity.ERROR,
            ),
        )
    )

    text = figure_text(plot_pareto_k(result))

    assert result.status is PosteriorAuditStatus.FAIL
    assert "FAILED AUDIT" in text
    assert "psis.unconverged-posterior" in text


# --------------------------------------------------------------------------------------
# model comparison
# --------------------------------------------------------------------------------------


def test_elpd_differences_draw_the_reported_centres_and_bounds() -> None:
    comparison = make_comparison()
    figure = plot_elpd_differences(comparison)
    axes = figure.axes[0]

    containers = axes.containers
    assert len(containers) == len(comparison.differences)
    centres = sorted(float(container[0].get_xdata()[0]) for container in containers)
    assert centres == sorted(item.elpd_difference for item in comparison.differences)


def test_an_unresolved_comparison_never_names_a_winner() -> None:
    comparison = make_comparison()

    text = figure_text(plot_elpd_differences(comparison))

    assert comparison.best_model is None
    assert "UNRESOLVED" in text
    assert "no model is selected" in text
    assert "best model" not in text


def test_a_resolved_comparison_names_its_best_model() -> None:
    comparison = make_comparison(
        status=ModelComparisonStatus.RESOLVED, best_model="rich", decisive=True
    )

    text = figure_text(plot_elpd_differences(comparison))

    assert "RESOLVED" in text
    assert "best model rich" in text


def test_elpd_differences_keep_the_reported_row_order() -> None:
    comparison = make_comparison()
    figure = plot_elpd_differences(comparison)

    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert labels == [f"{item.left_model} - {item.right_model}" for item in comparison.differences]


def test_elpd_differences_name_ineligible_models() -> None:
    text = figure_text(plot_elpd_differences(make_comparison(ineligible=True)))

    assert "ineligible: null" in text
    assert "comparison.ineligible-model" in text


# --------------------------------------------------------------------------------------
# posterior predictive checks
# --------------------------------------------------------------------------------------


def test_a_predictive_check_places_the_observation_in_its_reference_distribution() -> None:
    check = make_check()
    figure = plot_predictive_check(check, interval_probability=0.9)
    axes = figure.axes[0]

    observed = [line for line in axes.lines if line.get_xdata()[0] == pytest.approx(check.observed)]
    assert observed
    assert axes.patches
    text = figure_text(figure)
    assert "tail probability 0.8000" in text
    assert "90% predictive interval" in text
    assert "lab='A'" in text


def test_a_predictive_grid_draws_one_panel_per_check_with_the_family_accounting() -> None:
    audit = make_predictive_audit()

    figure = plot_predictive_checks(audit)

    assert len(figure.axes) == len(audit.checks)
    text = figure_text(figure)
    assert "1 of 2 checks below 0.05" in text
    assert "benjamini-hochberg" in text
    assert "threshold 0.02500 flags 1" in text


def test_a_failed_predictive_audit_is_watermarked() -> None:
    audit = make_predictive_audit(failed=True)

    text = figure_text(plot_predictive_checks(audit))

    assert audit.status is PosteriorAuditStatus.FAIL
    assert "FAILED AUDIT" in text
    assert "predictive.unconverged-posterior" in text


def test_a_predictive_grid_rejects_an_unknown_discrepancy() -> None:
    with pytest.raises(ValueError, match="no retained check"):
        plot_predictive_checks(make_predictive_audit(), discrepancy_name="variance")


# --------------------------------------------------------------------------------------
# parameter recovery
# --------------------------------------------------------------------------------------


def test_wald_recovery_labels_its_interval_and_draws_the_identity_line() -> None:
    report = make_recovery(kind=WALD_INTERVAL)
    figure = plot_parameter_recovery(report, "bias")
    axes = figure.axes[0]

    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("Wald" in label for label in labels)
    assert "identity" in labels
    identity = next(line for line in axes.lines if line.get_label() == "identity")
    assert list(identity.get_xdata()) == list(identity.get_ydata())


def test_posterior_quantile_recovery_uses_the_retained_bounds_and_a_distinct_label() -> None:
    report = make_recovery(kind=POSTERIOR_QUANTILE_INTERVAL)
    figure = plot_parameter_recovery(report, "bias")
    axes = figure.axes[0]

    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("posterior quantile" in label for label in labels)
    assert not any("Wald" in label for label in labels)
    assert "interval kind posterior-quantile" in figure_text(figure)


def test_the_two_interval_kinds_are_never_labelled_the_same_way() -> None:
    wald = figure_text(plot_parameter_recovery(make_recovery(kind=WALD_INTERVAL), "bias"))
    sampled = figure_text(
        plot_parameter_recovery(make_recovery(kind=POSTERIOR_QUANTILE_INTERVAL), "bias")
    )

    assert "interval kind wald" in wald
    assert "interval kind posterior-quantile" in sampled


def test_failed_recovery_runs_stay_visible_and_carry_no_interval() -> None:
    report = make_recovery(failures=2)
    figure = plot_parameter_recovery(report, "slope")
    axes = figure.axes[0]

    failed = [line for line in axes.lines if line.get_marker() == "x"]
    assert len(failed[0].get_xdata()) == 2
    assert "fit audit FAIL (2)" in figure_text(figure)


def test_recovery_grid_draws_one_panel_per_parameter() -> None:
    report = make_recovery()

    figure = plot_parameter_recovery_grid(report)

    assert len(figure.axes) == len(report.parameter_names)
    assert "Parameter recovery: glm" in figure_text(figure)


def test_recovery_rejects_an_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="not a parameter"):
        plot_parameter_recovery(make_recovery(), "drift")


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------


def test_available_calibration_places_its_aggregate_point_against_the_diagonal() -> None:
    summary = CalibrationSummary(True, 400, 0.62, 0.55, 0.21, 0.07)
    figure = plot_calibration(summary, label="glm")
    axes = figure.axes[0]

    points = [line for line in axes.lines if line.get_marker() == "o"]
    assert points[0].get_xdata()[0] == pytest.approx(0.62)
    assert points[0].get_ydata()[0] == pytest.approx(0.55)
    text = figure_text(figure)
    assert "Brier score 0.2100" in text
    assert "expected calibration error 0.0700" in text
    assert "per-bin reliability" in text


def test_unavailable_calibration_states_the_declared_reason() -> None:
    summary = CalibrationSummary(
        False, 120, None, None, None, None, "declared calibration outcome is not binary 0/1"
    )
    figure = plot_calibration(summary)
    axes = figure.axes[0]

    assert not [line for line in axes.lines if line.get_marker() == "o"]
    assert "not binary 0/1" in figure_text(figure)


# --------------------------------------------------------------------------------------
# convergence
# --------------------------------------------------------------------------------------


def test_rhat_uses_the_audit_policy_threshold_and_its_own_target_labels() -> None:
    audit = make_posterior_audit()
    figure = plot_rhat(audit)
    axes = figure.axes[0]

    thresholds = [line.get_ydata()[0] for line in axes.lines if line.get_linestyle() == "--"]
    assert pytest.approx(audit.policy.max_rhat) in thresholds
    labels = [label.get_text() for label in axes.get_xticklabels()]
    assert labels == ["beta[predictor='bias']", "beta[predictor='stimulus']", "sigma"]


def test_a_failing_posterior_audit_marks_its_exceedances_and_divergences() -> None:
    audit = make_posterior_audit(failing=True)

    text = figure_text(plot_rhat(audit))

    assert "above policy (1)" in text
    assert "divergences 12" in text
    assert "FAILED AUDIT" in text


def test_ess_draws_bulk_and_tail_against_the_policy_minima() -> None:
    audit = make_posterior_audit(failing=True)
    figure = plot_ess(audit)
    axes = figure.axes[0]

    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert "ESS bulk" in labels
    assert "ESS tail" in labels
    assert any("min_ess_bulk" in label for label in labels)
    assert any("below policy (1)" in label for label in labels)


def test_plot_convergence_stacks_rhat_above_ess() -> None:
    figure = plot_convergence(make_posterior_audit())

    assert len(figure.axes) == 2
    assert "R-hat" in figure.axes[0].get_ylabel()
    assert "effective sample size" in figure.axes[1].get_ylabel()


def test_convergence_plots_reject_a_non_audit() -> None:
    with pytest.raises(TypeError, match="PosteriorAudit"):
        plot_rhat(object())
