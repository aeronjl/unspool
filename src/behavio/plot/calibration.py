"""Reliability display for the runner's retained calibration estimand and bins.

For categorical predictions the primary curve is confidence calibration. Conditional
top-label and classwise curves remain available on the summary rather than being pooled into
that display. Unavailability is a result, not an absence: the declared reason is shown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from behavio.plot._axes import annotate_note, resolve_axes
from behavio.plot.style import ALERT, INDIGO, INK, MUTED, figure_style
from behavio.protocol.runner import CalibrationSummary

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_calibration"]


def plot_calibration(
    summary: CalibrationSummary,
    *,
    label: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (4.4, 4.2),
) -> Figure:
    """Draw retained populated reliability bins and the aggregate calibration point.

    The aggregate marker is the mean predicted probability against its observed rate; the
    smaller markers are the populated equal-width bins used by the reported ECE.
    """

    if not isinstance(summary, CalibrationSummary):
        raise TypeError("plot_calibration requires a CalibrationSummary")
    title = "Probability calibration" if label is None else f"Probability calibration: {label}"
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        axes.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color=INK,
            linestyle="--",
            linewidth=1.0,
            label="perfect calibration",
        )
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.set_xlabel("mean predicted probability or confidence")
        axes.set_ylabel("observed outcome rate or correctness")
        axes.set_title(title)
        if not summary.available:
            axes.text(
                0.5,
                0.5,
                f"calibration unavailable\n{summary.reason}",
                transform=axes.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=ALERT,
                wrap=True,
            )
            annotate_note(
                axes,
                f"{summary.n_observations} predictions considered; no calibration retained",
                alert=True,
            )
            axes.legend(loc="lower right", fontsize=7.5, frameon=False)
            return figure
        assert summary.mean_probability is not None and summary.observed_rate is not None
        if summary.bins:
            axes.plot(
                [item.mean_probability for item in summary.bins],
                [item.observed_rate for item in summary.bins],
                color=MUTED,
                linewidth=1.0,
                marker="s",
                markersize=4,
                label=f"populated reliability bins ({len(summary.bins)})",
            )
        axes.plot(
            [summary.mean_probability, summary.mean_probability],
            [summary.mean_probability, summary.observed_rate],
            color=MUTED,
            linewidth=1.0,
            linestyle=":",
            label="deviation from the diagonal",
        )
        axes.plot(
            [summary.mean_probability],
            [summary.observed_rate],
            linestyle="none",
            marker="o",
            markersize=8,
            color=INDIGO,
            label=f"aggregate calibration (n={summary.n_observations})",
        )
        axes.legend(loc="lower right", fontsize=7.5, frameon=False)
        annotate_note(
            axes,
            f"Brier score {summary.brier_score:.4f}, expected calibration error "
            f"{summary.expected_calibration_error:.4f}\n"
            f"{summary.estimand.value} estimand; {len(summary.bins)} populated bins; "
            f"{len(summary.top_label)} top-label and {len(summary.classwise)} classwise curves",
        )
    return figure
