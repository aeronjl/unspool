"""Probability-calibration display for the runner's retained calibration summary.

.. warning::

   :class:`~behavio.protocol.runner.CalibrationSummary` retains *aggregate* calibration only. The
   ten-bin reliability decomposition behind
   :attr:`~behavio.protocol.runner.CalibrationSummary.expected_calibration_error` is computed inside
   :mod:`behavio.protocol.runner` and then discarded, so a full reliability curve cannot be drawn
   without recomputing it. This module presents the aggregate point rather than
   re-deriving the bins, because the plotting layer presents; it does not compute. Retaining
   per-bin counts and rates on ``CalibrationSummary`` is the change that would upgrade this
   display to a reliability curve.

Unavailability is a result, not an absence: when the runner declares calibration
unavailable, the figure states the declared reason instead of drawing an empty axes.
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
    """Draw the retained aggregate calibration against perfect calibration.

    The single marker is the summary's mean predicted probability against its observed
    outcome rate; the diagonal is perfect calibration. Brier score, expected calibration
    error, and the observation count are written under the axes.
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
        axes.set_xlabel("mean predicted probability")
        axes.set_ylabel("observed outcome rate")
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
            "aggregate only: CalibrationSummary does not retain its per-bin reliability",
        )
    return figure
