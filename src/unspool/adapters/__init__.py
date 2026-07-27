"""Optional data-source adapters for the canonical :class:`unspool.Study`."""

from unspool.adapters.dandi import (
    DANDIAdapterError,
    DANDINWBSource,
    ResolvedDANDIAsset,
    resolve_dandi_nwb_asset,
    study_from_dandi,
)
from unspool.adapters.nwb import (
    NWBAdapterError,
    NWBSessionSource,
    add_study_trials,
    read_nwb,
    read_nwb_sessions,
    study_from_nwbfile,
    write_nwb,
)

__all__ = [
    "DANDIAdapterError",
    "DANDINWBSource",
    "NWBAdapterError",
    "NWBSessionSource",
    "ResolvedDANDIAsset",
    "add_study_trials",
    "read_nwb",
    "read_nwb_sessions",
    "resolve_dandi_nwb_asset",
    "study_from_dandi",
    "study_from_nwbfile",
    "write_nwb",
]
