"""Shared visual contract for generated documentation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

INDIGO = "#26345e"
BLUE = "#4f6d9a"
AMBER = "#c57928"
TEAL = "#2d7f78"
INK = "#202534"
MUTED = "#778092"
LIGHT = "#e8ebf2"

FONT_FAMILY = "DejaVu Sans"
MINIMUM_TEXT_SIZE = 7.0


def configure_figure_style() -> None:
    """Apply Behavio's deterministic, sans-serif scientific figure style."""

    mpl.rcParams.update(
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


def save_svg(figure: plt.Figure, path: Path, *, tight: bool = True) -> None:
    """Save a deterministic, searchable SVG and normalize trailing whitespace."""

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
    plt.close(figure)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
