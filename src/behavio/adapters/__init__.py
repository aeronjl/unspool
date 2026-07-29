"""Optional data-source adapters for the canonical :class:`behavio.Study`."""

from behavio.adapters.dandi import (
    DANDIAdapterError,
    DANDINWBSource,
    ResolvedDANDIAsset,
    resolve_dandi_nwb_asset,
    study_from_dandi,
)
from behavio.adapters.ibl_one import (
    DEFAULT_IBL_ALYX_URL,
    IBLONEAdapterError,
    IBLONETrialSource,
    read_ibl_one_sessions,
    study_from_ibl_one,
)
from behavio.adapters.nwb import (
    NWBAdapterError,
    NWBSessionSource,
    add_study_trials,
    read_nwb,
    read_nwb_sessions,
    study_from_nwbfile,
    write_nwb,
)

__all__ = [
    "DEFAULT_IBL_ALYX_URL",
    "DANDIAdapterError",
    "DANDINWBSource",
    "IBLONEAdapterError",
    "IBLONETrialSource",
    "NWBAdapterError",
    "NWBSessionSource",
    "ResolvedDANDIAsset",
    "add_study_trials",
    "read_ibl_one_sessions",
    "read_nwb",
    "read_nwb_sessions",
    "resolve_dandi_nwb_asset",
    "study_from_dandi",
    "study_from_ibl_one",
    "study_from_nwbfile",
    "write_nwb",
]
