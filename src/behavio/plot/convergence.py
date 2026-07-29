"""Convergence displays: R-hat and effective sample size against the audit's own policy.

The thresholds are read from :class:`~behavio.posterior_diagnostics.PosteriorAuditPolicy`
rather than hard-coded, so a figure drawn from an audit run at a stricter policy shows the
stricter line. Targets flagged by the audit are labelled with the same
``variable[dim=value]`` text the issues use, so a reader can move between the figure and
:attr:`~behavio.posterior_diagnostics.PosteriorAuditIssue.targets` without translating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from behavio.plot._axes import annotate_note, annotate_status, new_figure, resolve_axes
from behavio.plot.style import ALERT, BLUE, INK, MUTED, figure_style
from behavio.posterior_diagnostics import PosteriorAudit, PosteriorDiagnostic, _target_label

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_convergence", "plot_ess", "plot_rhat"]

_MAX_TICK_LABELS = 30


def plot_rhat(
    audit: PosteriorAudit,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.4, 3.6),
) -> Figure:
    """Draw R-hat per labelled target with the policy's ``max_rhat`` threshold."""

    if not isinstance(audit, PosteriorAudit):
        raise TypeError("plot_rhat requires a PosteriorAudit")
    labels, values = _flatten(audit, "rhat")
    threshold = audit.policy.max_rhat
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        _draw_targets(
            axes,
            labels,
            values,
            threshold=threshold,
            exceeds=values > threshold,
            threshold_label=f"policy max_rhat = {threshold:g}",
        )
        axes.set_ylabel("R-hat")
        axes.set_title(f"Convergence: {audit.model_name}")
        annotate_status(axes, audit.status, audit.issue_codes, label="posterior audit")
        annotate_note(axes, _sampling_text(audit), alert=bool(audit.divergences))
    return figure


def plot_ess(
    audit: PosteriorAudit,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.4, 3.6),
) -> Figure:
    """Draw bulk and tail effective sample size per target with the policy minima."""

    if not isinstance(audit, PosteriorAudit):
        raise TypeError("plot_ess requires a PosteriorAudit")
    labels, bulk = _flatten(audit, "ess_bulk")
    _, tail = _flatten(audit, "ess_tail")
    positions = np.arange(labels.size, dtype=np.float64)
    with figure_style():
        figure, axes = resolve_axes(ax, figsize=figsize)
        axes.plot(
            positions,
            bulk,
            linestyle="none",
            marker="o",
            markersize=4,
            color=BLUE,
            label="ESS bulk",
        )
        axes.plot(
            positions,
            tail,
            linestyle="none",
            marker="v",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor=INK,
            label="ESS tail",
        )
        axes.axhline(
            audit.policy.min_ess_bulk,
            color=INK,
            linestyle="--",
            linewidth=1.1,
            label=f"policy min_ess_bulk = {audit.policy.min_ess_bulk:g}",
        )
        if audit.policy.min_ess_tail != audit.policy.min_ess_bulk:
            axes.axhline(
                audit.policy.min_ess_tail,
                color=MUTED,
                linestyle=":",
                linewidth=1.1,
                label=f"policy min_ess_tail = {audit.policy.min_ess_tail:g}",
            )
        below = (bulk < audit.policy.min_ess_bulk) | (tail < audit.policy.min_ess_tail)
        if np.any(below):
            axes.plot(
                positions[below],
                np.minimum(bulk, tail)[below],
                linestyle="none",
                marker="x",
                markersize=7,
                color=ALERT,
                label=f"below policy ({int(np.count_nonzero(below))})",
            )
        _apply_tick_labels(axes, labels, positions)
        axes.set_ylabel("effective sample size")
        axes.set_title(f"Effective sample size: {audit.model_name}")
        axes.legend(loc="best", fontsize=7.5, frameon=False)
        annotate_status(axes, audit.status, audit.issue_codes, label="posterior audit")
        annotate_note(axes, _sampling_text(audit), alert=bool(audit.divergences))
    return figure


def plot_convergence(
    audit: PosteriorAudit,
    *,
    figsize: tuple[float, float] = (6.6, 6.8),
) -> Figure:
    """Draw R-hat above effective sample size in one two-panel convergence figure."""

    if not isinstance(audit, PosteriorAudit):
        raise TypeError("plot_convergence requires a PosteriorAudit")
    with figure_style():
        figure = new_figure(figsize=figsize)
        upper = figure.add_subplot(2, 1, 1)
        lower = figure.add_subplot(2, 1, 2)
        plot_rhat(audit, ax=upper)
        plot_ess(audit, ax=lower)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def _draw_targets(
    axes: Axes,
    labels: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
    exceeds: np.ndarray,
    threshold_label: str,
) -> None:
    positions = np.arange(labels.size, dtype=np.float64)
    axes.plot(
        positions[~exceeds],
        values[~exceeds],
        linestyle="none",
        marker="o",
        markersize=4,
        color=BLUE,
        label=f"within policy ({int(np.count_nonzero(~exceeds))})",
    )
    if np.any(exceeds):
        axes.plot(
            positions[exceeds],
            values[exceeds],
            linestyle="none",
            marker="^",
            markersize=6,
            color=ALERT,
            label=f"above policy ({int(np.count_nonzero(exceeds))})",
        )
    axes.axhline(threshold, color=INK, linestyle="--", linewidth=1.1, label=threshold_label)
    _apply_tick_labels(axes, labels, positions)
    axes.legend(loc="best", fontsize=7.5, frameon=False)


def _apply_tick_labels(axes: Axes, labels: np.ndarray, positions: np.ndarray) -> None:
    axes.set_xlabel("posterior target")
    if labels.size > _MAX_TICK_LABELS:
        axes.set_xlabel(f"posterior target ({labels.size} targets, labels omitted)")
        return
    axes.set_xticks(positions)
    axes.set_xticklabels([str(label) for label in labels], rotation=60, ha="right", fontsize=7)


def _flatten(audit: PosteriorAudit, field: str) -> tuple[np.ndarray, np.ndarray]:
    labels: list[str] = []
    values: list[float] = []
    for diagnostic in audit.diagnostics:
        array = np.asarray(getattr(diagnostic, field), dtype=np.float64)
        for index in np.ndindex(array.shape):
            labels.append(_label(diagnostic, index))
            values.append(float(array[index]))
    return np.asarray(labels, dtype=object), np.asarray(values, dtype=np.float64)


def _label(diagnostic: PosteriorDiagnostic, index: tuple[int, ...]) -> str:
    return _target_label(diagnostic, index)


def _sampling_text(audit: PosteriorAudit) -> str:
    divergences = "unavailable" if audit.divergences is None else str(audit.divergences)
    treedepth = "unavailable" if audit.max_treedepth_hits is None else str(audit.max_treedepth_hits)
    return (
        f"{audit.n_chains} chains x {audit.n_draws} draws = {audit.n_samples} samples; "
        f"divergences {divergences}, max-treedepth hits {treedepth}\n"
        f"{audit.inference_library} {audit.inference_library_version}"
    )
