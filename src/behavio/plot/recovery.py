"""Parameter-recovery displays: true against estimated, with the declared interval.

:class:`~behavio.recovery.ParameterRecoveryReport` records which interval its coverage came
from -- a Wald interval from the standard errors, or the equal-tailed 95% quantiles of the
posterior draws. The two are not the same quantity, so they are drawn differently and always
labelled with :attr:`~behavio.recovery.ParameterRecoveryReport.interval_kind`. A report
carries exactly one kind, so the two can never be pooled inside one figure; the label stops
them being pooled by eye across figures.

Runs whose fit audit failed are drawn, but as crosses without an interval, because they are
excluded from every summary the report computes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from behavio.contracts.audit import FitAuditStatus
from behavio.plot._axes import annotate_note, new_figure, resolve_axes
from behavio.plot.style import ALERT, BLUE, INK, MUTED, TEAL, figure_style
from behavio.recovery import _NORMAL_95 as WALD_MULTIPLIER
from behavio.recovery import (
    POSTERIOR_QUANTILE_INTERVAL,
    ParameterRecoveryReport,
    ParameterRecoverySummary,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_parameter_recovery", "plot_parameter_recovery_grid"]

_INTERVAL_STYLE = {
    POSTERIOR_QUANTILE_INTERVAL: (TEAL, "95% posterior quantile interval", 0),
}
_WALD_STYLE = (BLUE, f"estimate +/- {WALD_MULTIPLIER:.2f} SE (Wald)", 3)


def plot_parameter_recovery(
    report: ParameterRecoveryReport,
    parameter: str,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (4.6, 4.4),
) -> Figure:
    """Draw true against estimated values for one parameter with the identity line.

    The interval is the one the report declares. A Wald report has capped error bars from
    ``estimate +/- 1.96 * standard_error``, matching the coverage its own
    :meth:`~behavio.recovery.ParameterRecoveryReport.summary` computes; a posterior-quantile
    report has uncapped bars drawn straight from the retained bounds.
    """

    if not isinstance(report, ParameterRecoveryReport):
        raise TypeError("plot_parameter_recovery requires a ParameterRecoveryReport")
    if parameter not in report.parameter_names:
        raise ValueError(f"{parameter!r} is not a parameter of this recovery report")
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        _draw_parameter(axes, report, parameter)
        summary = _summary_for(report, parameter)
        annotate_note(
            axes,
            _summary_text(report, summary),
            alert=report.audit_failure_rate > 0.0,
        )
    return figure


def plot_parameter_recovery_grid(
    report: ParameterRecoveryReport,
    *,
    parameters: tuple[str, ...] | None = None,
    max_columns: int = 3,
    panel_size: tuple[float, float] = (3.2, 3.0),
) -> Figure:
    """Draw one recovery panel per parameter with a shared interval-kind caption."""

    if not isinstance(report, ParameterRecoveryReport):
        raise TypeError("plot_parameter_recovery_grid requires a ParameterRecoveryReport")
    names = report.parameter_names if parameters is None else tuple(parameters)
    unknown = tuple(name for name in names if name not in report.parameter_names)
    if unknown:
        raise ValueError(f"unknown recovery parameters: {', '.join(unknown)}")
    if not names:
        raise ValueError("a recovery grid needs at least one parameter")
    columns = max(1, min(max_columns, len(names)))
    rows = math.ceil(len(names) / columns)
    with figure_style():
        figure = new_figure(figsize=(panel_size[0] * columns, panel_size[1] * rows + 1.1))
        for index, name in enumerate(names):
            axes = figure.add_subplot(rows, columns, index + 1)
            _draw_parameter(axes, report, name, legend=index == 0)
        figure.suptitle(
            f"Parameter recovery: {report.model_name} "
            f"({report.n_subjects} subjects x {report.n_trials} trials, "
            f"{report.n_runs} runs)",
            fontsize=10,
        )
        figure.text(
            0.01,
            0.01,
            _report_text(report),
            fontsize=7.5,
            color=ALERT if report.audit_failure_rate > 0.0 else MUTED,
            ha="left",
            va="bottom",
        )
        figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    return figure


def _draw_parameter(
    axes: Axes,
    report: ParameterRecoveryReport,
    parameter: str,
    *,
    legend: bool = True,
) -> None:
    column = report.parameter_names.index(parameter)
    truth = np.asarray(report.true_values, dtype=np.float64)[:, column]
    estimate = np.asarray(report.estimates, dtype=np.float64)[:, column]
    failed = np.asarray(
        [audit.status is FitAuditStatus.FAIL for audit in report.audits], dtype=np.bool_
    )
    finite = np.isfinite(truth) & np.isfinite(estimate)
    eligible = finite & ~failed
    lower, upper, colour, label, capsize = _intervals(report, column)
    drawable = eligible & np.isfinite(lower) & np.isfinite(upper)
    if np.any(drawable):
        axes.errorbar(
            truth[drawable],
            estimate[drawable],
            yerr=[
                estimate[drawable] - lower[drawable],
                upper[drawable] - estimate[drawable],
            ],
            fmt="o",
            markersize=4,
            color=colour,
            ecolor=colour,
            elinewidth=1.0,
            capsize=capsize,
            linestyle="none",
            alpha=0.85,
            label=label,
        )
    bare = eligible & ~drawable
    if np.any(bare):
        axes.plot(
            truth[bare],
            estimate[bare],
            linestyle="none",
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor=MUTED,
            label=f"no usable interval ({int(np.count_nonzero(bare))})",
        )
    shown_failures = failed & finite
    if np.any(shown_failures):
        axes.plot(
            truth[shown_failures],
            estimate[shown_failures],
            linestyle="none",
            marker="x",
            markersize=6,
            color=ALERT,
            label=f"fit audit FAIL ({int(np.count_nonzero(failed))})",
        )
    _identity_line(axes, truth[finite], estimate[finite])
    axes.set_xlabel(f"true {parameter}")
    axes.set_ylabel(f"estimated {parameter}")
    axes.set_title(parameter, fontsize=9)
    if legend:
        axes.legend(loc="best", fontsize=7, frameon=False)


def _identity_line(axes: Axes, truth: np.ndarray, estimate: np.ndarray) -> None:
    values = np.concatenate([truth, estimate])
    if values.size == 0:
        low, high = 0.0, 1.0
    else:
        low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        low, high = low - 0.5, high + 0.5
    axes.plot([low, high], [low, high], color=INK, linestyle="--", linewidth=1.0, label="identity")


def _intervals(
    report: ParameterRecoveryReport, column: int
) -> tuple[np.ndarray, np.ndarray, str, str, int]:
    estimate = np.asarray(report.estimates, dtype=np.float64)[:, column]
    if report.interval_kind == POSTERIOR_QUANTILE_INTERVAL:
        assert report.interval_lower is not None and report.interval_upper is not None
        colour, label, capsize = _INTERVAL_STYLE[POSTERIOR_QUANTILE_INTERVAL]
        lower = np.asarray(report.interval_lower, dtype=np.float64)[:, column]
        upper = np.asarray(report.interval_upper, dtype=np.float64)[:, column]
        return lower, upper, colour, label, capsize
    colour, label, capsize = _WALD_STYLE
    errors = np.asarray(report.standard_errors, dtype=np.float64)[:, column]
    errors = np.where(errors >= 0.0, errors, np.nan)
    half_width = WALD_MULTIPLIER * errors
    return estimate - half_width, estimate + half_width, colour, label, capsize


def _summary_for(
    report: ParameterRecoveryReport, parameter: str
) -> ParameterRecoverySummary | None:
    for summary in report.summary():
        if summary.parameter == parameter:
            return summary
    return None  # pragma: no cover - summary() covers every declared parameter


def _summary_text(report: ParameterRecoveryReport, summary: ParameterRecoverySummary | None) -> str:
    if summary is None:  # pragma: no cover - unreachable for a valid report
        return _report_text(report)
    return (
        f"bias {summary.bias:.4g}, RMSE {summary.rmse:.4g}, r {summary.correlation:.3f}\n"
        f"coverage {summary.coverage_95:.3f} ({summary.interval_kind}) over "
        f"{summary.n_with_uncertainty}/{summary.n_successful} eligible runs\n"
        + _report_text(report)
    )


def _report_text(report: ParameterRecoveryReport) -> str:
    return (
        f"{report.n_runs} runs, convergence {report.convergence_rate:.3f}, "
        f"audit pass {report.audit_pass_rate:.3f} / warning {report.audit_warning_rate:.3f} / "
        f"fail {report.audit_failure_rate:.3f}; interval kind {report.interval_kind}"
    )
