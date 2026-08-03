"""Adapters and the tooling that keeps an adapter honest.

The tabular reader is deliberately not optional: CSV and TSV are how most behavioural data
first reaches Python, so :func:`behavio.adapters.table.read_table` uses only the standard
library and NumPy. Parquet, NWB, DANDI, and IBL ONE keep their third-party dependencies
behind extras, as required by the repository's architecture rules.

Every source dataclass here satisfies :class:`behavio.contracts.adapter.StudyAdapter`, and
:func:`behavio.adapters.conformance.check_study_adapter` is the runnable harness a
downstream adapter author can point at their own implementation.

One module serves the *other* kind of adapter -- a wrapper around a foreign model
implementation -- and it names no third-party package:
:mod:`behavio.adapters.estimator_conformance` executes the estimator half of the
compatibility list in ``docs/extensions.md``, including the behavioural test that
distinguishes a filtered prediction from a smoothed one. It has one entry point per
estimator contract -- :func:`~behavio.adapters.check_behaviour_estimator` for a model that
is fitted and :func:`~behavio.adapters.check_posterior_behaviour_estimator` for one that is
sampled -- running the same behavioural checks either way, so a sampled model reaches them
without its author writing an adapter first.

Two things a wrapper author needs used to live here and no longer do, because neither was
adapter-specific. :class:`behavio.trials.SequenceLayout` derives session boundaries and
restores source row order; it is a fact about a :class:`~behavio.trials.Study`, so it lives
beside one. :class:`behavio.contracts.DensityPrediction` is the predictive object a
response-time, confidence, or race model produces; it is a prediction, so it lives beside
:class:`~behavio.contracts.Prediction` and :class:`~behavio.contracts.CategoricalPrediction`
in the estimator contract. ``CensoredDensityPrediction`` widens the density shape with exact
right-tail survival at declared observation limits, and ``ModelPrediction`` names all four.

Concrete wrappers around third-party model packages live in :mod:`behavio.foreign`, which
sits above ``behavio.models`` because a wrapped model is a model.
"""

from behavio.adapters.canonical import (
    CanonicalTrialError,
    CanonicalTrialSource,
    study_from_canonical_trials,
)
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
    assert_posterior_behaviour_estimator_conforms,
    check_behaviour_estimator,
    check_posterior_behaviour_estimator,
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
    "CanonicalTrialError",
    "CanonicalTrialSource",
    "CheckStatus",
    "ColumnType",
    "ConformanceCheck",
    "DANDIAdapterError",
    "DANDINWBSource",
    "EstimatorConformance",
    "EstimatorConformanceError",
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
    "assert_behaviour_estimator_conforms",
    "assert_posterior_behaviour_estimator_conforms",
    "assert_study_adapter_conforms",
    "check_behaviour_estimator",
    "check_posterior_behaviour_estimator",
    "check_study_adapter",
    "perturb_future_rows",
    "read_ibl_one_sessions",
    "read_nwb",
    "read_nwb_sessions",
    "read_table",
    "read_tables",
    "resolve_dandi_nwb_asset",
    "session_order_from_appearance",
    "session_order_from_column",
    "session_order_from_explicit",
    "study_from_canonical_trials",
    "study_from_dandi",
    "study_from_ibl_one",
    "study_from_nwbfile",
    "write_nwb",
]
