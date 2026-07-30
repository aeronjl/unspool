"""Array immutability helpers used package-wide.

``protected_array`` was previously ``behavio.models.base._protected_array``: a private
helper that had become the de-facto array-immutability utility for more than a dozen
unrelated modules. It lives here so that importing it no longer drags in the model
contracts, and so that ``behavio.contracts`` can depend on it without a cycle.

Future work: ``behavio.observed._arrays`` holds ``_readonly_float`` and ``_validate_time``, whose
docstring deliberately scopes them to the four observed-behaviour modules
(``pose``, ``ethograms``, ``predictors``, ``sync``). Those helpers coerce to a single
dtype and enforce a time contract, so they are not a drop-in for ``protected_array``.
Consolidating the two modules is a separate change and is intentionally not done here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def protected_array(
    values: Sequence[Any] | NDArray[Any], *, dtype: np.dtype[Any] | type[Any]
) -> NDArray[Any]:
    """Return an immutable copy of ``values`` coerced to ``dtype``.

    Both the owning buffer and the returned view are marked read-only, so neither the
    caller's input nor a later ``base`` unwrapping can mutate the stored data.
    """

    owner = np.array(values, dtype=dtype, copy=True)
    owner.setflags(write=False)
    view = owner.view()
    view.setflags(write=False)
    return view
