"""Simulation-based calibration displays: the rank histogram and the ECDF-difference band.

Simulation-based calibration is a visual diagnostic. A mean normalized rank of one half is
compatible with a well-calibrated posterior, an over-dispersed one, and an under-dispersed
one, so the tuple of summary numbers cannot do the work the picture does.

Both functions present what
:class:`~behavio.posterior.simulation_based_calibration.SBCSummary` and
:class:`~behavio.posterior.simulation_based_calibration.SBCUniformity` already carry.
Neither recomputes a rank, a band, or a test statistic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from behavio.plot._axes import annotate_note, resolve_axes
from behavio.plot.style import ALERT, BLUE, INDIGO, INK, LIGHT, MUTED, figure_style
from behavio.posterior.simulation_based_calibration import SBCSummary, SBCUniformity

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_sbc_ecdf_difference", "plot_sbc_rank_histogram"]


def plot_sbc_rank_histogram(
    summary: SBCSummary,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.0, 3.6),
) -> Figure:
    """Draw one target's rank histogram against its exact discrete-uniform expectation.

    The reference line is
    :attr:`~behavio.posterior.simulation_based_calibration.SBCSummary.expected_bin_count`,
    the count each bin would hold under the exact discrete uniform null. No binomial
    envelope is drawn around it, because the calibrated reference for deviation is the
    *simultaneous* band on :func:`plot_sbc_ecdf_difference` and a second, pointwise envelope here
    would invite the eye to read bin-by-bin exceedances as significant.

    The replicate accounting travels with the figure: a histogram built from forty of one
    hundred intended replicates says so under the axes.
    """

    if not isinstance(summary, SBCSummary):
        raise TypeError("plot_sbc_rank_histogram requires an SBCSummary")
    counts = np.asarray(summary.histogram_counts, dtype=np.float64)
    bins = counts.size
    edges = np.linspace(0.0, 1.0, bins + 1)
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        axes.bar(
            edges[:-1],
            counts,
            width=1.0 / bins,
            align="edge",
            color=BLUE,
            edgecolor=INDIGO,
            linewidth=0.6,
            label=f"observed ranks (n={summary.n_replicates})",
        )
        axes.axhline(
            summary.expected_bin_count,
            color=INK,
            linestyle="--",
            linewidth=1.2,
            label=f"exact discrete-uniform expectation ({summary.expected_bin_count:.1f})",
        )
        axes.set_xlim(0.0, 1.0)
        axes.set_xlabel("normalized posterior rank of the true value")
        axes.set_ylabel("replicates per bin")
        axes.set_title(f"SBC rank histogram: {summary.target}")
        axes.legend(loc="upper right", fontsize=7.5, frameon=False)
        retained = summary.retained_fraction
        annotate_note(
            axes,
            f"mean normalized rank {summary.mean_normalized_rank:.3f} - "
            f"interval coverage {summary.interval_coverage:.3f}\n"
            f"{summary.n_replicates}/{summary.repeats_requested} replicates retained "
            f"({summary.n_unconverged} unconverged, {summary.n_other_failures} other failures)",
            alert=retained < 1.0,
        )
    return figure


def plot_sbc_ecdf_difference(
    uniformity: SBCUniformity,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.0, 3.6),
) -> Figure:
    """Draw the rank ECDF minus its null CDF inside the simultaneous confidence band.

    The band is drawn as a filled region, not as error bars, because it is a *simultaneous*
    envelope: under the null the whole difference curve stays inside it with probability
    :attr:`~behavio.posterior.simulation_based_calibration.SBCUniformity.confidence_level`.
    Error bars would suggest a pointwise reading, which would be exceeded far more often
    than the nominal level.

    Evaluation points where the curve leaves the band are marked individually and counted,
    matching
    :attr:`~behavio.posterior.simulation_based_calibration.SBCUniformity.n_points_outside_band`.
    """

    if not isinstance(uniformity, SBCUniformity):
        raise TypeError("plot_sbc_ecdf_difference requires an SBCUniformity")
    points = np.asarray(uniformity.evaluation_points, dtype=np.float64)
    difference = np.asarray(uniformity.ecdf_difference, dtype=np.float64)
    lower = np.asarray(uniformity.lower_difference_band, dtype=np.float64)
    upper = np.asarray(uniformity.upper_difference_band, dtype=np.float64)
    outside = (difference < lower) | (difference > upper)
    level = uniformity.confidence_level
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        axes.fill_between(
            points,
            lower,
            upper,
            color=LIGHT,
            linewidth=0.0,
            zorder=1,
            label=f"{level:.0%} simultaneous band ({uniformity.null} null)",
        )
        axes.plot(points, lower, color=MUTED, linewidth=0.7, zorder=2)
        axes.plot(points, upper, color=MUTED, linewidth=0.7, zorder=2)
        axes.axhline(0.0, color=MUTED, linestyle=":", linewidth=0.9, zorder=2)
        axes.plot(
            points,
            difference,
            color=INDIGO,
            linewidth=1.6,
            zorder=3,
            label=f"ECDF difference (n={uniformity.n_replicates})",
        )
        if np.any(outside):
            axes.plot(
                points[outside],
                difference[outside],
                linestyle="none",
                marker="x",
                markersize=5,
                color=ALERT,
                zorder=4,
                label=f"outside band ({uniformity.n_points_outside_band} points)",
            )
        axes.set_xlim(0.0, 1.0)
        axes.set_xlabel("normalized posterior rank")
        axes.set_ylabel("empirical CDF - null CDF")
        axes.set_title(f"SBC ECDF difference: {uniformity.target}")
        axes.legend(loc="best", fontsize=7.5, frameon=False)
        draws = uniformity.n_posterior_draws
        annotate_note(
            axes,
            f"chi-square {uniformity.chi_square:.2f} on {uniformity.chi_square_dof} df, "
            f"p = {uniformity.chi_square_p_value:.3f} over {uniformity.bins} bins\n"
            f"null {uniformity.null}"
            + (f" over {draws + 1} rank cells" if draws is not None else "")
            + f"; band from {uniformity.n_band_simulations} simulations, seed "
            f"{uniformity.band_seed}",
            alert=uniformity.n_points_outside_band > 0,
        )
    return figure
