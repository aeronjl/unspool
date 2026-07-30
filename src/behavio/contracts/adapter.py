"""The data-source adapter contract: identity, source type, and one ``read`` method.

Behavio shipped three adapters (local NWB, streamed DANDI, IBL ONE) before it had an
adapter *contract*. Each was an ad-hoc pair of a frozen ``XSource`` dataclass and a free
``study_from_x()`` function, and the conventions an adapter must honour -- above all that
it must never invent ``session_order`` -- existed only in prose.

:class:`StudyAdapter` makes those conventions structural. It is deliberately shaped around
what the existing adapters already do: the frozen source dataclass *is* the adapter. It
declares the source it points at, and :meth:`StudyAdapter.read` turns that declaration into
a validated :class:`behavio.trials.Study`. The free functions remain the primary API; the
protocol adds the stable identity, the declared source type, and the chronology policy that
a registry, an evidence bundle, or a conformance harness can read without special-casing
each adapter.

The conformance harness that exercises this contract lives in
:mod:`behavio.adapters.conformance` so that ``behavio.contracts`` stays a runtime leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from behavio.trials import Study


class SourceType(StrEnum):
    """Where an adapter's trials physically come from.

    This is the axis that decides whether reading is reproducible offline, whether a test
    needs a network marker, and whether a checksum can pin the bytes.
    """

    LOCAL_FILE = "local-file"
    LOCAL_DIRECTORY = "local-directory"
    REMOTE_ARCHIVE = "remote-archive"
    IN_MEMORY = "in-memory"


class SessionOrderPolicy(StrEnum):
    """How an adapter obtains ``session_order``, the one claim about time it may not guess.

    ``RECORDED`` means the chronology was either supplied by the caller or read from an
    explicit record in the source. ``DERIVED`` means the caller named a rule -- a date
    column, an explicit ordering, source row or file order -- and the adapter applied it.
    A derivation is a recorded choice, not an inference: an adapter that has neither a
    record nor a named rule must fail rather than number sessions by arrival.
    """

    RECORDED = "recorded"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Validated description of what one data-source adapter advertises."""

    adapter_name: str
    adapter_version: str
    source_type: SourceType
    session_order_policy: SessionOrderPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_name, str) or not self.adapter_name:
            raise ValueError("adapter_name must be a non-empty string")
        if not isinstance(self.adapter_version, str) or not self.adapter_version:
            raise ValueError("adapter_version must be a non-empty string")
        object.__setattr__(self, "source_type", SourceType(self.source_type))
        object.__setattr__(
            self, "session_order_policy", SessionOrderPolicy(self.session_order_policy)
        )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready record for manifests and evidence bundles."""

        return {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "source_type": self.source_type.value,
            "session_order_policy": self.session_order_policy.value,
        }


@runtime_checkable
class StudyAdapter(Protocol):
    """A declared data source that can be read once into a canonical ``Study``.

    An implementation is normally an immutable dataclass describing *which* trials to read
    -- a path, a pinned asset, a dataset UUID -- together with the identity and chronology
    the source cannot supply by itself. Reading must not mutate the declaration, so the
    same adapter read twice yields the same study whenever the source is unchanged.
    """

    @property
    def adapter_name(self) -> str:
        """Stable dotted name, unique per adapter implementation."""

    @property
    def adapter_version(self) -> str:
        """Version of the adapter's *contract behaviour*, not of the source dependency."""

    @property
    def source_type(self) -> SourceType:
        """Where the trials are read from."""

    @property
    def session_order_policy(self) -> SessionOrderPolicy:
        """Whether chronology was recorded by the source/caller or derived by a named rule."""

    def read(self) -> Study:
        """Return the declared trials as a validated study in source row order."""


def adapter_capabilities(adapter: Any) -> AdapterCapabilities:
    """Validate and return the capabilities advertised by a study adapter.

    The runtime-checkable protocol establishes attribute presence. This function
    additionally validates the semantic metadata that registries, provenance records, and
    :func:`behavio.adapters.conformance.check_study_adapter` rely on.
    """

    if not isinstance(adapter, StudyAdapter):
        raise TypeError("adapter must satisfy the StudyAdapter contract")
    return AdapterCapabilities(
        adapter_name=adapter.adapter_name,
        adapter_version=adapter.adapter_version,
        source_type=adapter.source_type,
        session_order_policy=adapter.session_order_policy,
    )
