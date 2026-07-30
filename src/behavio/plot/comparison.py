"""Paired ELPD-difference display for blocked posterior model comparison.

The report refuses to name a winner when the paired interval does not exclude zero, and the
figure must refuse with it. Differences whose interval covers zero are drawn with hollow
markers and a dashed interval; the comparison's status, reason, and ``best_model`` (or its
absence) are written onto the axes. Rows keep the order the report produced -- the plot never
re-sorts models, because re-sorting would manufacture the ranking the report declined to
make.

The markers report whether an interval excludes zero, which is a fact about one contrast.
Whether the *verdict* accepted it is a fact about the whole family, so the note states the
declared multiplicity adjustment and how many contrasts survived it beside how many
separated without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from behavio._internal.scoring import ComparisonFamily
from behavio.plot._axes import annotate_note, resolve_axes, status_colour
from behavio.plot.style import ALERT, INDIGO, INK, MUTED, TEAL, figure_style
from behavio.posterior.comparison import ModelComparisonStatus, PosteriorModelComparison

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_elpd_differences"]


def plot_elpd_differences(
    comparison: PosteriorModelComparison,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw every paired ELPD difference with its interval and the zero reference.

    ``xerr`` is built from the report's own ``lower`` and ``upper``, which come from the
    paired pointwise standard error at
    :attr:`~behavio.posterior.comparison.PosteriorModelComparison.interval_scale`. The plot
    does not recompute a standard error and does not combine the per-model ``se`` values,
    which are positively correlated and would give a wrong interval.
    """

    if not isinstance(comparison, PosteriorModelComparison):
        raise TypeError("plot_elpd_differences requires a PosteriorModelComparison")
    differences = comparison.differences
    if not differences:
        raise ValueError("this comparison retains no paired differences to draw")
    height = max(2.6, 0.42 * len(differences) + 2.0)
    resolved_size = figsize if figsize is not None else (6.6, height)
    centres = np.asarray([item.elpd_difference for item in differences], dtype=np.float64)
    lower = centres - np.asarray([item.lower for item in differences], dtype=np.float64)
    upper = np.asarray([item.upper for item in differences], dtype=np.float64) - centres
    excludes = np.asarray([item.excludes_zero for item in differences], dtype=np.bool_)
    ordinates = np.arange(len(differences), dtype=np.float64)[::-1]
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=resolved_size)
        axes.axvline(0.0, color=INK, linestyle="--", linewidth=1.1, label="no difference")
        for index in range(len(differences)):
            decisive = bool(excludes[index])
            axes.errorbar(
                centres[index],
                ordinates[index],
                xerr=[[lower[index]], [upper[index]]],
                fmt="o" if decisive else "s",
                markersize=5,
                markerfacecolor=INDIGO if decisive else "none",
                markeredgecolor=INDIGO if decisive else MUTED,
                ecolor=INDIGO if decisive else MUTED,
                elinewidth=1.6 if decisive else 1.0,
                capsize=3 if decisive else 0,
                linestyle="none",
            )
        axes.set_yticks(ordinates)
        axes.set_yticklabels(
            [f"{item.left_model} - {item.right_model}" for item in differences], fontsize=8
        )
        axes.set_ylim(-0.7, len(differences) - 0.3)
        scale = differences[0].interval_scale
        axes.set_xlabel(f"paired ELPD difference (+/- {scale:g} se)")
        block = comparison.block or "observation"
        axes.set_title(f"Blocked ELPD comparison ({comparison.estimand}, per {block})")
        _annotate_verdict(axes, comparison)
        _annotate_legend(axes, int(np.count_nonzero(excludes)), len(differences))
        eligible = set(comparison.eligible_models)
        ineligible = tuple(model.name for model in comparison.models if model.name not in eligible)
        annotate_note(
            axes,
            f"{comparison.n_data_points} pointwise units; {_family_note(comparison.family)}"
            + (f"; ineligible: {', '.join(ineligible)}" if ineligible else "")
            + (f"; issues: {', '.join(comparison.issue_codes)}" if comparison.issue_codes else ""),
            alert=bool(ineligible),
        )
    return figure


def _family_note(family: ComparisonFamily) -> str:
    """State how many contrasts survived the family adjustment, and which one.

    The markers say which intervals exclude zero; only this says how many of them the
    verdict actually accepted. Without it a reader sees three filled markers beside an
    ``UNRESOLVED`` verdict and has no way to tell that the correction is what separates
    the two.
    """

    if not family.corrected:
        return (
            f"{family.n_separated} of {family.n_comparisons} contrasts separate, uncorrected "
            f"at {family.family_error_rate:.3g}"
        )
    return (
        f"{family.n_decisive} of {family.n_comparisons} contrasts decisive after "
        f"{family.multiplicity.value} at {family.family_error_rate:.3g} "
        f"({family.n_separated} separated before it)"
    )


def _annotate_verdict(axes: Axes, comparison: PosteriorModelComparison) -> None:
    status = comparison.status
    if status is ModelComparisonStatus.RESOLVED and comparison.best_model is not None:
        text = f"status: RESOLVED - best model {comparison.best_model}"
        colour = TEAL
    elif status is ModelComparisonStatus.NO_ELIGIBLE_MODEL:
        text = "status: NO ELIGIBLE MODEL - no model may be ranked"
        colour = ALERT
    else:
        text = "status: UNRESOLVED - no model is selected"
        colour = status_colour("warning")
    axes.text(
        0.0,
        1.01,
        f"{text}\n{comparison.reason}",
        transform=axes.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=colour,
    )


def _annotate_legend(axes: Axes, decisive: int, total: int) -> None:
    axes.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=5,
        color=INDIGO,
        label=f"interval excludes zero ({decisive})",
    )
    axes.plot(
        [],
        [],
        linestyle="none",
        marker="s",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor=MUTED,
        label=f"interval covers zero ({total - decisive})",
    )
    axes.legend(loc="best", fontsize=7.5, frameon=False)
