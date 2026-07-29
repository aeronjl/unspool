"""Behavio's figure standard: the palette, the rcParams, and deterministic SVG export.

This module owns the visual contract described by the scientific figure standard. It used to
live outside the package in ``scripts/figure_style.py``; the documentation generator is now a
caller of this module rather than the owner of the style, so a figure drawn by
:mod:`behavio.plot` and a figure drawn by the documentation build obey the same contract.

Nothing here mutates global state on import. :func:`configure_figure_style` mutates
``matplotlib.rcParams`` only when a caller asks it to; :func:`figure_style` applies the same
settings for the duration of a block and restores them afterwards, which is what every
plotting function in this package uses.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from behavio.plot._optional import require_matplotlib, require_pyplot

if TYPE_CHECKING:
    from matplotlib.figure import Figure

INDIGO: Final = "#26345e"
BLUE: Final = "#4f6d9a"
AMBER: Final = "#c57928"
TEAL: Final = "#2d7f78"
INK: Final = "#202534"
MUTED: Final = "#778092"
LIGHT: Final = "#e8ebf2"
ALERT: Final = "#9c2b32"
"""Reserved for failed audits and threshold exceedances; never for an ordinary series."""

FONT_FAMILY: Final = "DejaVu Sans"
MINIMUM_TEXT_SIZE: Final = 7.0

FIGURE_RC_PARAMS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "axes.edgecolor": "#c5cad5",
        "axes.labelcolor": INK,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "font.size": 9,
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        # Frozen at the pre-rename value on purpose: the salt seeds every generated
        # SVG element id, so changing it would rewrite all committed figures without
        # changing a single plotted value.
        "svg.hashsalt": "unspool-documentation-v1",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
"""The complete rcParams patch that defines a Behavio figure."""


def configure_figure_style() -> None:
    """Apply Behavio's deterministic, sans-serif scientific figure style globally.

    Intended for scripts that own their whole process, such as the documentation figure
    generator. Library code should prefer :func:`figure_style`, which does not leak.
    """

    matplotlib = require_matplotlib()
    matplotlib.rcParams.update(dict(FIGURE_RC_PARAMS))


@contextmanager
def figure_style(**overrides: Any) -> Iterator[None]:
    """Apply the figure standard for the duration of a block and then restore rcParams."""

    matplotlib = require_matplotlib()
    with matplotlib.rc_context({**FIGURE_RC_PARAMS, **overrides}):
        yield


def save_svg(figure: Figure, path: Path, *, tight: bool = True) -> None:
    """Save a deterministic, searchable SVG and normalize trailing whitespace.

    Trailing whitespace is stripped for the same reason ``examples/first_analysis.py``
    strips it: matplotlib emits platform-dependent trailing spaces that would otherwise
    make byte comparison of a committed asset fail for a non-scientific reason.

    The style is applied around the write itself, not only around the drawing: ``svg.fonttype``
    and ``svg.hashsalt`` are read during ``savefig``, so a figure saved outside the context
    would emit converted text paths and unsalted element identifiers even though every
    plotted value was correct.
    """

    pyplot = require_pyplot()
    with figure_style():
        if tight:
            figure.tight_layout()
        figure.savefig(
            path,
            format="svg",
            bbox_inches="tight",
            metadata={
                "Creator": "Behavio documentation figure generator",
                "Date": None,
                "Format": "image/svg+xml",
            },
        )
    pyplot.close(figure)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
