"""Stream a pinned public NWB trial table through the canonical Study contract."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.provenance import render
from unspool import DANDINWBSource, resolve_dandi_nwb_asset, study_from_dandi

DANDISET_ID = "000004"
VERSION = "0.220126.1852"
ASSET_PATH = "sub-P11HMH/sub-P11HMH_ses-20061101_ecephys+image.nwb"
ASSET_ID = "0f57f0b0-f021-42bb-8eaa-56cd482e2a29"
ASSET_SIZE = 72_628_704
ASSET_SHA256 = "ed70cbaa4c9ac65a4365bb805e4fb2f8689de7e0f867f9a8247285980acf7781"


def source() -> DANDINWBSource:
    """Return the exact published asset and explicit longitudinal mapping."""

    return DANDINWBSource(
        dandiset_id=DANDISET_ID,
        version=VERSION,
        asset_path=ASSET_PATH,
        session_order=0,
        session="2006-11-01",
        columns=(
            "start_time",
            "stop_time",
            "stim_phase",
            "stimCategory",
            "category_name",
            "new_old_labels_recog",
            "response_value",
            "response_time",
        ),
        column_map={
            "stimCategory": "stimulus_category",
            "response_time": "response_timestamp",
        },
    )


def run() -> dict[str, Any]:
    """Resolve, stream, adapt, and audit one public behavioral NWB session."""

    declared = source()
    asset = resolve_dandi_nwb_asset(declared)
    study = study_from_dandi(declared)
    trials = np.asarray(study["trial"], dtype=np.int64)
    starts = np.asarray(study["start_time"], dtype=np.float64)
    stops = np.asarray(study["stop_time"], dtype=np.float64)
    phase_counts = dict(sorted(Counter(study["stim_phase"].tolist()).items()))
    category_counts = {
        str(key): value
        for key, value in sorted(Counter(study["stimulus_category"].tolist()).items())
    }
    provenance_columns = (
        "source_dandiset_id",
        "source_dandi_version",
        "source_dandi_asset_path",
        "source_dandi_asset_id",
        "source_dandi_asset_size",
        "source_dandi_sha256",
        "source_nwb_identifier",
    )
    contract = {
        "published_version_is_pinned": declared.version != "draft",
        "asset_identity_matches": asset.asset_id == ASSET_ID,
        "asset_size_matches": asset.size == ASSET_SIZE,
        "asset_sha256_matches": asset.sha256 == ASSET_SHA256,
        "one_explicit_subject": study.subjects == ("P11HMH",),
        "one_explicit_session": set(study["session"].tolist()) == {"2006-11-01"},
        "session_order_is_explicit": set(study["session_order"].tolist()) == {0},
        "source_trial_ids_preserved": np.array_equal(trials, np.arange(len(study))),
        "source_row_order_preserved": bool(np.all(np.diff(starts) >= 0)),
        "trial_intervals_are_valid": bool(np.all(np.isfinite(starts) & np.isfinite(stops)))
        and bool(np.all(stops >= starts)),
        "provenance_is_trial_addressable": all(
            name in study.columns and len(set(study[name].tolist())) == 1
            for name in provenance_columns
        ),
        "trial_provenance_matches_asset": set(study["source_dandi_asset_id"].tolist())
        == {asset.asset_id}
        and set(study["source_dandi_asset_size"].tolist()) == {asset.size}
        and set(study["source_dandi_sha256"].tolist()) == {asset.sha256},
        "source_response_semantics_are_unmodified": "response_value" in study.columns
        and "choice" not in study.columns,
    }
    return {
        "benchmark": "NWB/DANDI longitudinal interoperability",
        "source": {
            "dandiset_id": asset.dandiset_id,
            "version": asset.version,
            "asset_path": asset.path,
            "asset_id": asset.asset_id,
            "size": asset.size,
            "sha256": asset.sha256,
            "content_url": asset.content_url,
            "license": "CC-BY-4.0",
            "dandiset_doi": "10.48324/dandi.000004/0.220126.1852",
        },
        "study": {
            "n_trials": len(study),
            "n_subjects": len(study.subjects),
            "columns": list(study.columns),
            "phase_counts": phase_counts,
            "stimulus_category_counts": category_counts,
            "response_value_minimum": float(np.min(study["response_value"])),
            "response_value_maximum": float(np.max(study["response_value"])),
        },
        "contract": contract,
        "contract_passed": all(contract.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("result.json"),
    )
    arguments = parser.parse_args()
    result = run()
    rendered = render(result, allow_nan=True, sort_keys=False)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
