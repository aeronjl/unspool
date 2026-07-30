"""Lazy, version-checked access to the optional PyDDM dependency.

Nothing in :mod:`behavio.foreign` imports a third-party model package at module scope, for
the same reason :mod:`behavio.plot` does not import matplotlib: ``import behavio`` and every
numerical API must keep working on a machine that has none of the wrapped packages. Each
wrapper names its dependency in exactly one place -- here -- so the failure a user meets is
an actionable :class:`ForeignPackageUnavailableError` naming the extra rather than a
``ModuleNotFoundError`` raised from somewhere inside a solve.

The version is checked, not merely sniffed, and the check is a *series* rather than a
minimum. A wrapper's :attr:`signature` is a scientific fingerprint, and a solver whose
numerics changed is not the same model even under the same parameters, so the supported
series is written into the signature and enforced on import. Patch releases within the
series are accepted and recorded in the fit's diagnostics, where provenance belongs.
"""

from __future__ import annotations

import importlib
from itertools import takewhile
from typing import Any, Final

PYDDM_EXTRA: Final = "pyddm"
"""Name of the optional dependency group that installs PyDDM."""

PYDDM_SERIES: Final = "0.9"
"""The PyDDM minor series this wrapper is written and fingerprinted against.

Pinned as ``pyddm>=0.9,<0.10`` in ``pyproject.toml``. It appears in
:attr:`behavio.foreign.PyDDMDriftDiffusion.signature` because a first-passage density is
the model: two fits produced by solvers with different numerics should not share a
fingerprint, and a fingerprint that read the *installed* version would change under a patch
release that changed nothing scientific, and could not be computed at all without the extra.
"""

_INSTALL_HINT: Final = (
    f"This wrapper requires PyDDM: install it with `pip install 'behavio[{PYDDM_EXTRA}]'`."
)


class ForeignPackageUnavailableError(ImportError):
    """Raised when a wrapper runs without the third-party package it wraps.

    Mirrors :class:`behavio.plot.MatplotlibUnavailableError`: the call is refused with the
    install command rather than an optional dependency being imported eagerly.
    """


def require_pyddm() -> Any:
    """Return the imported ``pyddm`` module, or explain how to install a usable one."""

    try:
        pyddm = importlib.import_module("pyddm")
    except ImportError as error:
        raise ForeignPackageUnavailableError(_INSTALL_HINT) from error
    series = _series(pyddm)
    if series is not None and series != PYDDM_SERIES:
        raise ForeignPackageUnavailableError(
            f"Behavio's PyDDM wrapper is written against PyDDM {PYDDM_SERIES}.x and found "
            f"{getattr(pyddm, '__version__', 'an unknown version')}. A different series may "
            "solve the same model to different numbers, so it is refused rather than used "
            f"silently: `pip install 'behavio[{PYDDM_EXTRA}]'` installs a supported one."
        )
    return pyddm


def pyddm_version() -> str:
    """Return the exact installed PyDDM version, for a fit's provenance record."""

    return str(getattr(require_pyddm(), "__version__", "unknown"))


def _series(module: Any) -> str | None:
    raw = str(getattr(module, "__version__", ""))
    parts: list[str] = []
    for piece in raw.split(".")[:2]:
        digits = "".join(takewhile(str.isdigit, piece))
        if not digits:
            return None
        parts.append(digits)
    return ".".join(parts) if len(parts) == 2 else None


__all__ = [
    "PYDDM_EXTRA",
    "PYDDM_SERIES",
    "ForeignPackageUnavailableError",
    "pyddm_version",
    "require_pyddm",
]
