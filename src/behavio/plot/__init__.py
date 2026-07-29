"""Deterministic matplotlib displays for Behavio's diagnostic reports.

Every central diagnostic in this package is inherently visual. A simulation-based
calibration report, a Pareto-:math:`k` vector, a posterior-predictive reference
distribution, and a parameter-recovery scatter all lose most of their meaning when reduced
to a tuple of numbers: the shape of the deviation is the finding.

Contract
--------
Every function here takes a report object that some other part of the package produced and
returns a :class:`matplotlib.figure.Figure`. They do not call ``show``, do not write files,
do not mutate ``matplotlib.rcParams`` at import, and do not register anything with
``pyplot``. The house style is applied per call through
:func:`~behavio.plot.style.figure_style`, which restores the previous rcParams afterwards.

Plots present; they do not compute. A number that is not already on the report is not
invented here. Where a report carries a status, an issue list, or a replicate accounting,
the figure carries it too -- a display drawn from a failed audit must not be able to look
like a display drawn from a clean one.

Optional dependency
-------------------
matplotlib is an extra, not a core dependency. ``import behavio`` and ``import behavio.plot``
both work without it; only calling a plotting function raises
:class:`~behavio.plot._optional.MatplotlibUnavailableError`, whose message names the
``behavio[plots]`` extra.
"""

from behavio.plot._optional import (
    MINIMUM_MATPLOTLIB,
    PLOTS_EXTRA,
    MatplotlibUnavailableError,
)
from behavio.plot.calibration import plot_calibration
from behavio.plot.comparison import plot_elpd_differences
from behavio.plot.convergence import plot_convergence, plot_ess, plot_rhat
from behavio.plot.loo import plot_pareto_k
from behavio.plot.predictive import plot_predictive_check, plot_predictive_checks
from behavio.plot.recovery import plot_parameter_recovery, plot_parameter_recovery_grid
from behavio.plot.sbc import plot_sbc_ecdf_difference, plot_sbc_rank_histogram
from behavio.plot.style import (
    ALERT,
    AMBER,
    BLUE,
    FIGURE_RC_PARAMS,
    FONT_FAMILY,
    INDIGO,
    INK,
    LIGHT,
    MINIMUM_TEXT_SIZE,
    MUTED,
    TEAL,
    configure_figure_style,
    figure_style,
    save_svg,
)

__all__ = [
    "ALERT",
    "AMBER",
    "BLUE",
    "FIGURE_RC_PARAMS",
    "FONT_FAMILY",
    "INDIGO",
    "INK",
    "LIGHT",
    "MINIMUM_MATPLOTLIB",
    "MINIMUM_TEXT_SIZE",
    "MUTED",
    "PLOTS_EXTRA",
    "TEAL",
    "MatplotlibUnavailableError",
    "configure_figure_style",
    "figure_style",
    "plot_calibration",
    "plot_convergence",
    "plot_elpd_differences",
    "plot_ess",
    "plot_parameter_recovery",
    "plot_parameter_recovery_grid",
    "plot_pareto_k",
    "plot_predictive_check",
    "plot_predictive_checks",
    "plot_rhat",
    "plot_sbc_ecdf_difference",
    "plot_sbc_rank_histogram",
    "save_svg",
]
