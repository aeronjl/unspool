"""Shared array coercion, time-vector validation and identity rules.

``behavio.observed.pose``, ``behavio.observed.ethograms``, ``behavio.observed.covariates``,
``behavio.observed.device_clocks`` and ``behavio.observed.trialization`` all coerce caller-supplied
sequences into read-only float arrays and enforce the same strictly increasing,
finite time contract. These helpers are private and exist only so those modules
agree on it.

``_identity`` is the same contract for ``subject`` and ``session``. It exists so
that observed behaviour and ``behavio.trials.Study`` agree on what an identifier
is: any hashable value, normalised out of its NumPy scalar wrapper. Before it,
the observed types annotated both fields ``str`` while ``Study`` accepted any
hashable, so an integer subject id that round-tripped through NWB into a
``Study`` could only be matched against a pose through a lossy ``str()``.

Future work: ``behavio._internal.arrays.protected_array`` is the package-wide
array-immutability helper. It coerces to a caller-supplied dtype and enforces no time
contract, so it is not a drop-in replacement for either helper here. Consolidating the two
modules is a deliberate follow-up, not part of the contracts re-homing.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _identity(value: Any, *, name: str) -> Hashable:
    """Normalise one ``subject``/``session`` identifier to the Study contract.

    Mirrors ``behavio.trials``: a NumPy scalar becomes its Python value, a
    missing value is refused, and anything unhashable is refused because it
    could never be used as a join key.
    """

    normalized = value.item() if isinstance(value, np.generic) else value
    if normalized is None:
        raise ValueError(f"{name} must not be missing")
    if isinstance(normalized, (float, complex)) and np.isnan(normalized):
        raise ValueError(f"{name} must not be missing")
    try:
        hash(normalized)
    except TypeError:
        raise ValueError(f"{name} must be hashable; received {value!r}") from None
    return normalized


def _readonly_float(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


def _validate_time(time_s: NDArray[np.float64]) -> None:
    if len(time_s) == 0:
        raise ValueError("time_s must contain at least one sample")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s must contain only finite values")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing")
