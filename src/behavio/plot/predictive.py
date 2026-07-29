"""Posterior-predictive displays: an observed statistic inside its reference distribution.

A tail probability is a summary of a picture. Whether the observed value sits just outside a
long tail or far outside a tight one is the difference between a model that is nearly right
and one that is wrong in kind, and only the reference distribution shows it.

The family-level accounting from :class:`~behavio.posterior_predictive.PredictiveFamily` is
drawn onto the grid, because a single extreme check among forty is not the same finding as a
single extreme check among two.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from behavio.plot._axes import (
    annotate_note,
    annotate_status,
    format_group,
    new_figure,
    resolve_axes,
)
from behavio.plot.style import ALERT, BLUE, INDIGO, INK, LIGHT, MUTED, figure_style
from behavio.posterior_predictive import PosteriorPredictiveAudit, PosteriorPredictiveCheck

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_predictive_check", "plot_predictive_checks"]

_DEFAULT_BINS = 30


def plot_predictive_check(
    check: PosteriorPredictiveCheck,
    *,
    ax: Axes | None = None,
    bins: int = _DEFAULT_BINS,
    interval_probability: float | None = None,
    flagged: bool = False,
    figsize: tuple[float, float] = (5.2, 3.4),
) -> Figure:
    """Draw one check: replicated discrepancies, the retained interval, and the observation.

    ``interval_probability`` only labels the shaded interval; the interval itself is the one
    the check retained. Pass
    :attr:`~behavio.posterior_predictive.PosteriorPredictivePolicy.interval_probability` when
    the policy is to hand.
    """

    if not isinstance(check, PosteriorPredictiveCheck):
        raise TypeError("plot_predictive_check requires a PosteriorPredictiveCheck")
    replicated = np.asarray(check.replicated, dtype=np.float64).reshape(-1)
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        _draw_check(axes, check, replicated, bins=bins, interval_probability=interval_probability)
        axes.set_title(f"{check.discrepancy_name}: {format_group(check.group)}", fontsize=9)
        axes.set_xlabel(check.discrepancy_name)
        axes.set_ylabel("replicated draws")
        axes.legend(loc="best", fontsize=7, frameon=False)
        annotate_note(
            axes,
            f"tail probability {check.tail_probability:.4f} ({check.tail.value}); "
            f"lower {check.lower_probability:.4f}, upper {check.upper_probability:.4f}; "
            f"{check.n_observations} observations",
            alert=flagged,
        )
    return figure


def plot_predictive_checks(
    audit: PosteriorPredictiveAudit,
    *,
    discrepancy_name: str | None = None,
    bins: int = _DEFAULT_BINS,
    max_columns: int = 3,
    panel_size: tuple[float, float] = (3.4, 2.6),
) -> Figure:
    """Draw one panel per retained check, grouped into a grid, with the family accounting.

    ``discrepancy_name`` restricts the grid to a single discrepancy so a per-group display
    stays readable. The audit status and its issue codes are written above the grid.
    """

    if not isinstance(audit, PosteriorPredictiveAudit):
        raise TypeError("plot_predictive_checks requires a PosteriorPredictiveAudit")
    checks = tuple(
        check
        for check in audit.checks
        if discrepancy_name is None or check.discrepancy_name == discrepancy_name
    )
    if not checks:
        raise ValueError(f"no retained check is named {discrepancy_name!r}")
    columns = max(1, min(max_columns, len(checks)))
    rows = math.ceil(len(checks) / columns)
    family = audit.family
    with figure_style():
        figure = new_figure(figsize=(panel_size[0] * columns, panel_size[1] * rows + 1.0))
        for index, check in enumerate(checks):
            axes = figure.add_subplot(rows, columns, index + 1)
            replicated = np.asarray(check.replicated, dtype=np.float64).reshape(-1)
            _draw_check(
                axes,
                check,
                replicated,
                bins=bins,
                interval_probability=audit.policy.interval_probability,
            )
            flagged = check.tail_probability <= family.adjusted_threshold
            extreme = check.tail_probability <= family.tail_probability_warning
            axes.set_title(
                f"{check.discrepancy_name}\n{format_group(check.group)}",
                fontsize=8,
                color=ALERT if extreme else INK,
            )
            axes.set_xlabel(f"p_tail = {check.tail_probability:.4f}", fontsize=7.5)
            axes.tick_params(labelsize=7)
            if flagged:
                for spine in axes.spines.values():
                    spine.set_edgecolor(ALERT)
                    spine.set_linewidth(1.3)
        figure.suptitle(
            f"Posterior predictive checks: {audit.model_name} ({audit.variable_name})",
            fontsize=10,
        )
        head = figure.axes[0]
        annotate_status(head, audit.status, audit.issue_codes, label="predictive audit")
        figure.text(
            0.01,
            0.01,
            f"{family.n_extreme} of {family.n_checks} checks below "
            f"{family.tail_probability_warning:g} (expected {family.expected_extreme:.2f}); "
            f"{family.multiplicity.value} threshold {family.adjusted_threshold:.5f} flags "
            f"{family.n_flagged}; excess probability {family.excess_probability:.4f}",
            fontsize=7.5,
            color=ALERT if family.n_flagged else MUTED,
            ha="left",
            va="bottom",
        )
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.94))
    return figure


def _draw_check(
    axes: Axes,
    check: PosteriorPredictiveCheck,
    replicated: np.ndarray,
    *,
    bins: int,
    interval_probability: float | None,
) -> None:
    axes.hist(
        replicated,
        bins=bins,
        color=LIGHT,
        edgecolor=MUTED,
        linewidth=0.5,
        label=f"replicated (n={replicated.size})",
    )
    interval_label = (
        f"{interval_probability:.0%} predictive interval"
        if interval_probability is not None
        else "retained predictive interval"
    )
    axes.axvspan(
        check.interval[0],
        check.interval[1],
        color=BLUE,
        alpha=0.16,
        linewidth=0.0,
        label=interval_label,
    )
    inside = check.interval[0] <= check.observed <= check.interval[1]
    axes.axvline(
        check.observed,
        color=INDIGO if inside else ALERT,
        linestyle="-" if inside else "--",
        linewidth=1.8,
        label=f"observed = {check.observed:.4g}",
    )
