"""Adapters and the tooling that keeps an adapter honest.

The tabular reader is deliberately not optional: CSV and TSV are how most behavioural data
first reaches Python, so :func:`behavio.adapters.table.read_table` uses only the standard
library and NumPy. Parquet, NWB, DANDI, and IBL ONE keep their third-party dependencies
behind extras, as required by the repository's architecture rules.

Every source dataclass here satisfies :class:`behavio.contracts.adapter.StudyAdapter`, and
:func:`behavio.adapters.conformance.check_study_adapter` is the runnable harness a
downstream adapter author can point at their own implementation.

Three modules serve the *other* kind of adapter -- a wrapper around a foreign model
implementation -- and none of them names a third-party package:

- :mod:`behavio.adapters.sequences` derives session boundaries and restores source row order
  once, so every wrapper that has to hand a foreign package a list of per-sequence arrays
  does it the same way;
- :mod:`behavio.adapters.prediction` adds :class:`~behavio.adapters.prediction.DensityPrediction`,
  the predictive object a response-time, confidence, or race model produces and the
  estimator contract had no shape for;
- :mod:`behavio.adapters.estimator_conformance` executes the estimator half of the
  compatibility list in ``docs/extensions.md``, including the behavioural test that
  distinguishes a filtered prediction from a smoothed one.

Concrete wrappers around third-party model packages live in :mod:`behavio.foreign`, which
sits above ``behavio.models`` because a wrapped model is a model.
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
from behavio.adapters.estimator_conformance import (
    EstimatorConformance,
    EstimatorConformanceError,
    assert_behaviour_estimator_conforms,
    check_behaviour_estimator,
    perturb_future_rows,
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
from behavio.adapters.prediction import (
    LOG_DENSITY_FLOOR,
    DensityBehaviourEstimator,
    DensityPrediction,
)
from behavio.adapters.sequences import (
    SequenceGrouping,
    SequenceLayout,
    SequenceLayoutError,
    TrialSequence,
    sequence_layout,
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
    "LOG_DENSITY_FLOOR",
    "AdapterConformance",
    "AdapterConformanceError",
    "CheckStatus",
    "ColumnType",
    "ConformanceCheck",
    "DANDIAdapterError",
    "DANDINWBSource",
    "DensityBehaviourEstimator",
    "DensityPrediction",
    "EstimatorConformance",
    "EstimatorConformanceError",
    "IBLONEAdapterError",
    "IBLONETrialSource",
    "NWBAdapterError",
    "NWBSessionSource",
    "ResolvedDANDIAsset",
    "SequenceGrouping",
    "SequenceLayout",
    "SequenceLayoutError",
    "SessionOrderDerivation",
    "SessionOrderRule",
    "TableFormat",
    "TableReadError",
    "TableSource",
    "TrialSequence",
    "add_study_trials",
    "assert_behaviour_estimator_conforms",
    "assert_study_adapter_conforms",
    "check_behaviour_estimator",
    "check_study_adapter",
    "perturb_future_rows",
    "read_ibl_one_sessions",
    "read_nwb",
    "read_nwb_sessions",
    "read_table",
    "read_tables",
    "resolve_dandi_nwb_asset",
    "sequence_layout",
    "session_order_from_appearance",
    "session_order_from_column",
    "session_order_from_explicit",
    "study_from_dandi",
    "study_from_ibl_one",
    "study_from_nwbfile",
    "write_nwb",
]
