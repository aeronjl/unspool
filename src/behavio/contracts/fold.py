"""The validation-fold contract.

``ValidationFold`` used to be a bare union alias at ``behavio.validation:501`` while doing
structural-contract work across ``transforms``, ``evaluation``, ``comparison`` and
``model_recovery``: those modules only ever read the five members declared below, and a
union alias silently excluded any downstream split type. It is now a runtime-checkable
protocol, so an extension package can supply its own fold without editing the union, and
``isinstance(split, ValidationFold)`` is meaningful. Every first-party split in
:mod:`behavio.validation` satisfies it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ValidationFold(Protocol):
    """One training/test partition of a study's source row positions.

    ``prediction_context_indices`` are observed training rows replayed to initialize a
    filtered prediction. They are never part of ``test_indices`` and are never scored.
    ``prospective`` states whether the fold's deployment order protects the forecast
    horizon; consumers must not infer it from the scheme name.
    """

    @property
    def train_indices(self) -> NDArray[np.intp]: ...

    @property
    def test_indices(self) -> NDArray[np.intp]: ...

    @property
    def prediction_context_indices(self) -> NDArray[np.intp]: ...

    @property
    def scheme(self) -> Any: ...

    @property
    def prospective(self) -> bool: ...
