"""Lazy, version-checked access to the optional matplotlib dependency.

Plotting is an extra rather than a core dependency: ``import behavio`` and every numerical
API must keep working on a machine without matplotlib. This module is therefore the single
place matplotlib is named, and nothing here imports it at module scope. Every plotting entry
point calls :func:`require_matplotlib` (or one of the wrappers), so the failure a user meets
is an actionable :class:`MatplotlibUnavailableError` naming the extra, never a bare
``ModuleNotFoundError`` from somewhere inside a drawing routine.

The version is sniffed at call time for the same reason ArviZ is in
:mod:`behavio.posterior`: an installed-but-too-old dependency is a different failure from a
missing one and deserves a different message.
"""

from __future__ import annotations

import importlib
from itertools import takewhile
from typing import Any, Final

PLOTS_EXTRA: Final = "plots"
"""Name of the optional dependency group that installs the plotting stack."""

MINIMUM_MATPLOTLIB: Final = (3, 9)
"""Oldest matplotlib whose figure-level API this package draws against."""

_INSTALL_HINT: Final = (
    f"Behavio plotting requires matplotlib: install it with `pip install 'behavio[{PLOTS_EXTRA}]'`."
)


class MatplotlibUnavailableError(ImportError):
    """Raised when a plotting entry point runs without a usable matplotlib.

    Mirrors :class:`~behavio.posterior.ArviZUnavailableError`: the package refuses the call
    with the install command rather than importing an optional dependency eagerly.
    """


def require_matplotlib() -> Any:
    """Return the imported ``matplotlib`` module, or explain how to install it."""

    try:
        matplotlib = importlib.import_module("matplotlib")
    except ImportError as error:
        raise MatplotlibUnavailableError(_INSTALL_HINT) from error
    version = _version(matplotlib)
    if version is not None and version < MINIMUM_MATPLOTLIB:
        minimum = ".".join(str(part) for part in MINIMUM_MATPLOTLIB)
        raise MatplotlibUnavailableError(
            f"Behavio plotting requires matplotlib >= {minimum}, found "
            f"{getattr(matplotlib, '__version__', 'an unknown version')}: upgrade with "
            f"`pip install --upgrade 'behavio[{PLOTS_EXTRA}]'`."
        )
    return matplotlib


def require_module(name: str) -> Any:
    """Return a matplotlib submodule after the availability and version gate."""

    if name != "matplotlib" and not name.startswith("matplotlib."):
        raise ValueError("require_module only imports matplotlib submodules")
    require_matplotlib()
    try:
        return importlib.import_module(name)
    except ImportError as error:  # pragma: no cover - a broken install, not a missing one
        raise MatplotlibUnavailableError(_INSTALL_HINT) from error


def require_figure_type() -> Any:
    """Return ``matplotlib.figure.Figure``.

    Figures are built directly rather than through ``pyplot`` so drawing never touches the
    global pyplot figure registry: nothing is retained after the caller drops the return
    value, and a test suite cannot leak figures it forgot to close.
    """

    return require_module("matplotlib.figure").Figure


def require_pyplot() -> Any:
    """Return ``matplotlib.pyplot`` for the few entry points that need global state."""

    return require_module("matplotlib.pyplot")


def _version(matplotlib: Any) -> tuple[int, ...] | None:
    raw = str(getattr(matplotlib, "__version__", ""))
    parts: list[int] = []
    for piece in raw.split(".")[:2]:
        digits = "".join(takewhile(str.isdigit, piece))
        if not digits:
            return None
        parts.append(int(digits))
    return tuple(parts) if len(parts) == 2 else None
