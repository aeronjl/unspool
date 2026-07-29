"""Data-source adapters for the canonical :class:`behavio.Study`.

The tabular reader is deliberately not optional: CSV and TSV are how most behavioural data
first reaches Python, so :func:`behavio.adapters.table.read_table` uses only the standard
library and NumPy. Parquet, NWB, DANDI, and IBL ONE keep their third-party dependencies
behind extras, as required by the repository's architecture rules.

Every source dataclass here satisfies :class:`behavio.contracts.adapter.StudyAdapter`, and
:func:`behavio.adapters.conformance.check_study_adapter` is the runnable harness a
downstream adapter author can point at their own implementation.
"""

from behavio.adapters.conformance import (
    AdapterConformance,
    AdapterConformanceError,
    CheckStatus,
    ConformanceCheck,
    assert_study_adapter_conforms,
    check_study_adapter,
)
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
from behavio.adapters.table import (
    DEFAULT_MISSING_VALUES,
    ColumnType,
    SessionOrderDerivation,
    SessionOrderRule,
    TableFormat,
    TableReadError,
    TableSource,
    read_table,
    read_tables,
    session_order_from_appearance,
    session_order_from_column,
    session_order_from_explicit,
)

__all__ = [
    "DEFAULT_IBL_ALYX_URL",
    "DEFAULT_MISSING_VALUES",
    "AdapterConformance",
    "AdapterConformanceError",
    "CheckStatus",
    "ColumnType",
    "ConformanceCheck",
    "DANDIAdapterError",
    "DANDINWBSource",
    "IBLONEAdapterError",
    "IBLONETrialSource",
    "NWBAdapterError",
    "NWBSessionSource",
    "ResolvedDANDIAsset",
    "SessionOrderDerivation",
    "SessionOrderRule",
    "TableFormat",
    "TableReadError",
    "TableSource",
    "add_study_trials",
    "assert_study_adapter_conforms",
    "check_study_adapter",
    "read_ibl_one_sessions",
    "read_nwb",
    "read_nwb_sessions",
    "read_table",
    "read_tables",
    "resolve_dandi_nwb_asset",
    "session_order_from_appearance",
    "session_order_from_column",
    "session_order_from_explicit",
    "study_from_dandi",
    "study_from_ibl_one",
    "study_from_nwbfile",
    "write_nwb",
]
