"""The validation-fold contract.

``EvaluationFold`` used to be a bare union alias at ``behavio.evaluate.splits:501`` while doing
structural-contract work across ``transforms``, ``evaluation``, ``comparison`` and
``model_recovery``: those modules only ever read the members declared below, and a
union alias silently excluded any downstream split type. It is now a runtime-checkable
protocol, so an extension package can supply its own fold without editing the union, and
``isinstance(split, EvaluationFold)`` is meaningful. Every first-party split in
:mod:`behavio.evaluate.splits` satisfies it.

``identifier`` joined the contract once it became load-bearing:
:attr:`behavio.evaluate.folds.FoldEvaluation.identifier` names retained failures and keys the
evidence bundle's prediction and audit maps. ``evaluate_splits`` read it through a
``getattr`` fallback that numbered folds by position when a split did not supply one,
which is the hidden name-based contract this package has been removing -- the same shape
as the four private attribute names ``diagnostics`` used to duck-type before they became
public protocols. A fold now names itself or is not a fold.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EvaluationFold(Protocol):
    """One training/test partition of a study's source row positions.

    ``prediction_context_indices`` are observed training rows replayed to initialize a
    filtered prediction. They are never part of ``test_indices`` and are never scored.
    ``prospective`` states whether the fold's deployment order protects the forecast
    horizon; consumers must not infer it from the scheme name.

    ``identifier`` is the fold's name wherever a result is reported: a retained failure,
    a serialized fold record, and the prediction and audit maps of an evidence bundle all
    key on it. It must be stable across runs of the same splitter on the same study,
    unique within one split set, and meaningful to a reader -- a leave-one-subject-out
    fold naming the held-out subject is worth more than ``fold-0003``. The first-party
    splitters derive it from the fold's own scientific coordinates rather than from its
    position in the returned sequence, so inserting or reordering folds does not rename
    the others.
    """

    @property
    def identifier(self) -> str: ...

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
