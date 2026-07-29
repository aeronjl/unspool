"""Importance-sampling reliability display for pointwise and blocked PSIS-LOO.

Pareto-:math:`k` is the diagnostic that says whether an ELPD number may be read at all. It
is per observation, or per block when :func:`~behavio.posterior_loo.psis_loo` was given a
grouping variable, and its threshold is sample-size dependent, so the report carries its own
``good_k`` rather than the folklore 0.7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from behavio.plot._axes import annotate_note, annotate_status, resolve_axes
from behavio.plot.style import ALERT, BLUE, INK, MUTED, figure_style
from behavio.posterior_loo import PSISLOOResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_pareto_k"]

_MAX_TICK_LABELS = 25


def plot_pareto_k(
    result: PSISLOOResult,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.4, 3.8),
) -> Figure:
    """Draw Pareto-:math:`k` per observation or per block with the ``good_k`` threshold.

    Points above ``good_k`` are drawn with a different marker as well as a different colour,
    so the exceedance survives a greyscale print. Non-finite values cannot be placed on the
    axis; they are counted in the note rather than dropped silently.

    The result's :attr:`~behavio.posterior_loo.PSISLOOResult.status` and issue codes are
    written onto the axes: an ELPD from a non-converged posterior must not be able to look
    like an ELPD from a converged one.
    """

    if not isinstance(result, PSISLOOResult):
        raise TypeError("plot_pareto_k requires a PSISLOOResult")
    values = np.asarray(result.pareto_k, dtype=np.float64).reshape(-1)
    positions = np.arange(values.size, dtype=np.float64)
    finite = np.isfinite(values)
    above = finite & (values > result.good_k)
    below = finite & ~above
    unit = result.block or (result.dims[0] if result.dims else "index")
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        axes.plot(
            positions[below],
            values[below],
            linestyle="none",
            marker="o",
            markersize=4,
            color=BLUE,
            label=f"k <= good_k ({int(np.count_nonzero(below))})",
        )
        if np.any(above):
            axes.plot(
                positions[above],
                values[above],
                linestyle="none",
                marker="^",
                markersize=6,
                color=ALERT,
                label=f"k > good_k ({int(np.count_nonzero(above))})",
            )
        axes.axhline(
            result.good_k,
            color=INK,
            linestyle="--",
            linewidth=1.2,
            label=f"good_k = {result.good_k:.2f}",
        )
        axes.set_xlabel(f"{unit} (pointwise unit)")
        axes.set_ylabel("Pareto k")
        axes.set_title(f"PSIS-LOO reliability: {result.model_name} ({result.estimand})")
        _apply_tick_labels(axes, result, values.size)
        axes.legend(loc="best", fontsize=7.5, frameon=False)
        annotate_status(axes, result.status, result.issue_codes, label="PSIS-LOO")
        non_finite = int(values.size - np.count_nonzero(finite))
        annotate_note(
            axes,
            f"elpd_loo {result.elpd_loo:.2f} +/- {result.se:.2f} (se), p_loo {result.p_loo:.2f}, "
            f"{result.n_data_points} pointwise units from {result.n_samples} draws\n"
            f"{result.inference_library} {result.inference_library_version}"
            + (f"; {non_finite} non-finite k not drawn" if non_finite else ""),
            alert=non_finite > 0,
        )
    return figure


def _apply_tick_labels(axes: Any, result: PSISLOOResult, size: int) -> None:
    if len(result.dims) != 1 or size > _MAX_TICK_LABELS:
        return
    labels = np.asarray(result.coords[result.dims[0]]).reshape(-1)
    if labels.size != size:  # pragma: no cover - guarded by PSISLOOResult validation
        return
    axes.set_xticks(np.arange(size, dtype=np.float64))
    axes.set_xticklabels([str(label) for label in labels], rotation=45, ha="right", fontsize=7.5)
    axes.tick_params(axis="x", colors=MUTED)
