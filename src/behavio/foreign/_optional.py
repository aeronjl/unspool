"""Lazy, version-checked access to the optional foreign-model dependencies.

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

One wrapper supports *two* series rather than one. Bambi 0.17 is the last release that runs
on Python 3.11 and it pairs with PyMC 5; Bambi 0.18 and later require Python 3.12 and PyMC 6.
Behavio supports 3.11, so refusing the older series would make the extra a silent no-op on
the interpreter floor the package still claims. Both are therefore accepted, and unlike
PyDDM's the accepted set is *not* written into the model's signature -- see
:attr:`behavio.foreign.bambi.BambiRegression.signature` for the argument, which turns on a
sampler being an approximation to a declared target rather than being the model.

One accessor here does more than import and check, and it is stated rather than hidden:
:func:`require_dynamax` switches jax's 64-bit mode on, process-wide, because there is no
scoped form of that switch left in jax 0.11 and because a 32-bit forward-backward pass and a
32-bit Hessian are not the computation the wrapper claims to perform. Nothing else in Behavio
uses jax, so the only code affected is the caller's own.
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

BAMBI_EXTRA: Final = "bambi"
"""Name of the optional dependency group that installs Bambi."""

BAMBI_SERIES: Final = ("0.17", "0.18", "0.19")
"""The Bambi minor series this wrapper is written and tested against.

Pinned as ``bambi>=0.17.2,<0.18`` below Python 3.12 and ``bambi>=0.19,<0.20`` at or above
it, because Bambi 0.18 raised its own floor to 3.12 and moved to PyMC 6. ``0.18`` is
accepted here but is not installed by either pin: it is the same API as ``0.19`` and
refusing an environment that already has it would be pedantry rather than a safeguard.

Unlike :data:`PYDDM_SERIES` this does *not* appear in the wrapper's signature. See
:attr:`behavio.foreign.bambi.BambiRegression.signature`.
"""

_BAMBI_INSTALL_HINT: Final = (
    f"This wrapper requires Bambi: install it with `pip install 'behavio[{BAMBI_EXTRA}]'`. "
    "On Python 3.11 that installs Bambi 0.17.x with PyMC 5; on 3.12 and newer it installs "
    "Bambi 0.19.x with PyMC 6."
)


DYNAMAX_EXTRA: Final = "dynamax"
"""Name of the optional dependency group that installs dynamax."""

DYNAMAX_SERIES: Final = "1.0"
"""The dynamax minor series this wrapper is written and tested against.

Pinned as ``dynamax>=1.0,<1.1`` in ``pyproject.toml``. Like :data:`BAMBI_SERIES` and unlike
:data:`PYDDM_SERIES` it does *not* appear in the wrapper's signature. PyDDM's series is in
its fingerprint because a first-passage density is computed by a truncated series whose term
selection differs between releases, so the same parameters give different numbers. Forward
filtering and backward smoothing are exact arithmetic on an exactly specified likelihood;
what a dynamax release can change is the *default initialisation* of EM, and
:class:`~behavio.foreign.dynamax.DynamaxSwitchingAutoregression` puts the initialisation
method, the restart count, the iteration count and the seed into its own signature instead.
The exact installed version is provenance and is recorded on every fit.
"""

_DYNAMAX_INSTALL_HINT: Final = (
    f"This wrapper requires dynamax: install it with `pip install 'behavio[{DYNAMAX_EXTRA}]'`. "
    "It brings jax, which conflicts with any environment pinning jax<0.4.32 (pyhgf does); see "
    "`docs/foreign-models.md`."
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


def require_bambi() -> Any:
    """Return the imported ``bambi`` module, or explain how to install a usable one."""

    try:
        bambi = importlib.import_module("bambi")
    except ImportError as error:
        raise ForeignPackageUnavailableError(_BAMBI_INSTALL_HINT) from error
    series = _series(bambi)
    if series is not None and series not in BAMBI_SERIES:
        raise ForeignPackageUnavailableError(
            f"Behavio's Bambi wrapper is written against Bambi {', '.join(BAMBI_SERIES)} and "
            f"found {getattr(bambi, '__version__', 'an unknown version')}. A different series "
            "may build a different model from the same formula -- Bambi's default priors are "
            "part of its release, not of the formula -- so it is refused rather than used "
            f"silently: `pip install 'behavio[{BAMBI_EXTRA}]'` installs a supported one."
        )
    return bambi


def bambi_version() -> str:
    """Return the exact installed Bambi version, for a posterior's provenance record."""

    return str(getattr(require_bambi(), "__version__", "unknown"))


def require_dynamax() -> Any:
    """Return the imported ``dynamax`` module, or explain how to install a usable one.

    This is also the single place jax's 64-bit mode is switched on, and it is switched on
    **process-wide**, which is a real side effect and is stated rather than hidden. jax
    defaults to 32-bit floats; a forward-backward pass over a thousand trials accumulates
    log probabilities in that precision, and the observed-information Hessian this wrapper
    differentiates out of the marginal likelihood is not meaningfully computable in it. jax
    0.11 removed the ``enable_x64`` context manager, so there is no scoped form of the
    switch left. Nothing else in Behavio uses jax, so the only code affected is the caller's
    own.
    """

    try:
        dynamax = importlib.import_module("dynamax")
    except ImportError as error:
        raise ForeignPackageUnavailableError(_DYNAMAX_INSTALL_HINT) from error
    series = _series(dynamax)
    if series is not None and series != DYNAMAX_SERIES:
        raise ForeignPackageUnavailableError(
            f"Behavio's dynamax wrapper is written against dynamax {DYNAMAX_SERIES}.x and "
            f"found {getattr(dynamax, '__version__', 'an unknown version')}. A different "
            "series may build a different model from the same declaration -- the default EM "
            "initialisation is part of the release, not of the specification -- so it is "
            f"refused rather than used silently: `pip install 'behavio[{DYNAMAX_EXTRA}]'` "
            "installs a supported one."
        )
    jax = importlib.import_module("jax")
    jax.config.update("jax_enable_x64", True)
    return dynamax


def dynamax_version() -> str:
    """Return the exact installed dynamax version, for a fit's provenance record."""

    return str(getattr(require_dynamax(), "__version__", "unknown"))


def jax_version() -> str:
    """Return the exact installed jax version, which is half of a dynamax fit's numerics."""

    require_dynamax()
    return str(getattr(importlib.import_module("jax"), "__version__", "unknown"))


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
    "BAMBI_EXTRA",
    "BAMBI_SERIES",
    "DYNAMAX_EXTRA",
    "DYNAMAX_SERIES",
    "PYDDM_EXTRA",
    "PYDDM_SERIES",
    "ForeignPackageUnavailableError",
    "bambi_version",
    "dynamax_version",
    "jax_version",
    "pyddm_version",
    "require_bambi",
    "require_dynamax",
    "require_pyddm",
]
