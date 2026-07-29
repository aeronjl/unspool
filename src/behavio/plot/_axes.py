"""Shared figure construction and audit-status annotation for Behavio displays.

Two rules from the package's stance are enforced here rather than repeated in every module:

* figures are built from :class:`matplotlib.figure.Figure` directly, never through
  ``pyplot``, so drawing has no global side effect and nothing needs closing; and
* a report that carries a status carries it onto the figure. A display drawn from a failed
  audit is marked as such in the axes itself, so it cannot be cropped, re-captioned, or
  pasted into a slide and read as clean evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from behavio.contracts.audit import FitAuditStatus
from behavio.plot._optional import require_figure_type
from behavio.plot.style import ALERT, AMBER, MUTED, TEAL
from behavio.posterior_diagnostics import PosteriorAuditStatus

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

STATUS_COLOURS: dict[str, str] = {
    "pass": TEAL,
    "warning": AMBER,
    "fail": ALERT,
}
"""Status colour, always paired with the status word itself so colour is never the signal."""


def status_colour(status: PosteriorAuditStatus | FitAuditStatus | str) -> str:
    """Return the colour reserved for an audit status."""

    return STATUS_COLOURS.get(str(getattr(status, "value", status)), MUTED)


def new_figure(*, figsize: tuple[float, float]) -> Figure:
    """Return a bare figure that is not registered with ``pyplot``."""

    return require_figure_type()(figsize=figsize)


def resolve_axes(
    ax: Axes | None,
    *,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    """Return the figure and axes to draw on, creating both when ``ax`` is ``None``."""

    if ax is not None:
        figure = ax.get_figure()
        if figure is None:  # pragma: no cover - detached axes are not constructible normally
            raise ValueError("the supplied axes is not attached to a figure")
        return figure, ax
    figure = new_figure(figsize=figsize)
    return figure, figure.add_subplot()


def annotate_status(
    ax: Axes,
    status: PosteriorAuditStatus | FitAuditStatus | str,
    issue_codes: Sequence[str] = (),
    *,
    label: str = "audit",
) -> None:
    """Write a status banner above the axes, and watermark the axes when it failed.

    ``FAIL`` is deliberately loud. The package retains failed evidence rather than deleting
    it, which only works if the failure travels with the display.
    """

    value = str(getattr(status, "value", status))
    text = f"{label}: {value.upper()}"
    if issue_codes:
        text = f"{text} ({', '.join(issue_codes)})"
    failed = value == "fail"
    ax.text(
        0.0,
        1.01,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=status_colour(value),
        fontweight="bold" if failed else "normal",
    )
    if failed:
        ax.text(
            0.5,
            0.5,
            "FAILED AUDIT",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=22,
            color=ALERT,
            alpha=0.16,
            rotation=18,
            zorder=6,
            fontweight="bold",
        )


def annotate_note(ax: Axes, text: str, *, alert: bool = False) -> None:
    """Write an accounting or caveat note under the axes."""

    ax.text(
        0.0,
        -0.24,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=ALERT if alert else MUTED,
        wrap=False,
    )


def format_group(group: Sequence[tuple[str, Any]]) -> str:
    """Render a labelled grouping tuple as ``dim=value`` text."""

    if not group:
        return "all observations"
    return ", ".join(f"{name}={value!r}" for name, value in group)
