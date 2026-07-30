"""The parameter-semantics contract.

:class:`behavio.inference.parameters.ParameterSpace` is referenced under ``TYPE_CHECKING`` only:
``behavio.inference.parameters`` re-exports :class:`ParameterSpaceProvider` from here, so importing
it at runtime would be circular. The protocol never touches a ``ParameterSpace`` value,
only its type, so nothing is lost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from behavio.inference.parameters import ParameterSpace


@runtime_checkable
class ParameterSpaceProvider(Protocol):
    """Object exposing stable parameter semantics to inference and interchange code."""

    @property
    def parameter_space(self) -> ParameterSpace: ...
