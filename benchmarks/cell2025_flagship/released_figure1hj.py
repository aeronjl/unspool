"""Replay the released Cell 2025 Figure 1H/1J trajectory computations.

Run this optional module in the exact numerical environment recorded by the released
analysis. The pinned stack is deliberately kept outside Behavio's runtime dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cell2025.fetch_data import sha256

ANALYSIS_COMMIT = "2faa4680d5e9c0d6a9df516e3dede8c641e39a72"
ANALYSIS_DOI = "10.6084/m9.figshare.28877942.v1"
ANCHORS = {"DAP110": "right", "DAP009": "balanced", "DAP028": "left"}
TRAJECTORY_SHA256 = "e5cd064e575fe792a1e2b265ef05f7fb23daa68131fea0e230a2ad2a5b6664ee"
CLUSTER_SHA256 = "6cd23f17c6634532bab1af5fd11ed212c8d9431a81a80c8ad28261fb63ab49dc"
SEMANTIC_ORDER = ("left", "balanced", "right")
DEFAULT_RELEASED_DIRECTORY = Path(__file__).with_name("data") / "released"


def _checked_digest(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} checksum mismatch: observed {observed}, expected {expected}")
    return observed


def reproduce(trajectory_path: Path, cluster_path: Path) -> dict[str, Any]:
    """Return the released Figure 1H/1J centroids and an exact-membership audit."""

    import pandas as pd
    import scipy
    import sklearn
    import tslearn
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset

    trajectory_digest = _checked_digest(
        trajectory_path, TRAJECTORY_SHA256, "released trajectory table"
    )
    cluster_digest = _checked_digest(cluster_path, CLUSTER_SHA256, "released cluster table")

    trajectories = pd.read_csv(trajectory_path)
    released_clusters = pd.read_csv(cluster_path)
    required = {
        "mouseNum",
        "sessionNum_interpolated",
        "psych_right_slopes",
        "psych_left_slopes",
        "prop_below",
    }
    if not required <= set(trajectories.columns):
        raise ValueError(f"trajectory table lacks columns: {sorted(required - set(trajectories))}")
    subject_order = trajectories["mouseNum"].drop_duplicates().astype(str).tolist()
    if len(subject_order) != 30:
        raise ValueError(f"expected 30 released trajectories, observed {len(subject_order)}")

    rows_by_subject: dict[str, np.ndarray] = {}
    colour_by_subject: dict[str, float] = {}
    right_left_paths: list[np.ndarray] = []
    for subject in subject_order:
        rows = trajectories.loc[trajectories["mouseNum"] == subject].sort_values(
            "sessionNum_interpolated"
        )
        if len(rows) != 100:
            raise ValueError(
                f"expected 100 interpolation points for {subject}, observed {len(rows)}"
            )
        values = rows[
            ["sessionNum_interpolated", "psych_right_slopes", "psych_left_slopes"]
        ].to_numpy(dtype=np.float64)
        rows_by_subject[subject] = values
        colour_by_subject[subject] = float(rows["prop_below"].iloc[0])
        right_left_paths.append(values[:, [1, 2]])

    model_1j = TimeSeriesKMeans(
        n_clusters=3,
        metric="softdtw",
        max_iter=20,
        random_state=1,
    )
    model_1j.fit(to_time_series_dataset(right_left_paths))
    raw_by_subject = {
        subject: int(label) for subject, label in zip(subject_order, model_1j.labels_, strict=True)
    }
    semantic_by_raw = {raw_by_subject[subject]: semantic for subject, semantic in ANCHORS.items()}
    if len(semantic_by_raw) != 3:
        raise RuntimeError("released semantic anchors no longer identify three clusters")
    semantic_by_subject = {
        subject: semantic_by_raw[raw_label] for subject, raw_label in raw_by_subject.items()
    }
    released_by_subject = dict(
        zip(
            released_clusters["mouseNum"].astype(str),
            released_clusters["cluster"].astype(str),
            strict=True,
        )
    )
    if semantic_by_subject != released_by_subject:
        mismatches = {
            subject: {
                "reproduced": semantic_by_subject.get(subject),
                "released": released_by_subject.get(subject),
            }
            for subject in sorted(set(semantic_by_subject) | set(released_by_subject))
            if semantic_by_subject.get(subject) != released_by_subject.get(subject)
        }
        raise RuntimeError(f"Figure 1J membership mismatch: {mismatches}")

    memberships = {
        semantic: [subject for subject in subject_order if semantic_by_subject[subject] == semantic]
        for semantic in SEMANTIC_ORDER
    }
    average_prop_below = {
        semantic: float(np.mean([colour_by_subject[subject] for subject in members]))
        for semantic, members in memberships.items()
    }
    centers_1j = {
        semantic_by_raw[raw]: np.asarray(center, dtype=np.float64).tolist()
        for raw, center in enumerate(model_1j.cluster_centers_)
    }

    centers_1h: dict[str, Any] = {}
    for semantic in SEMANTIC_ORDER:
        paths = []
        for subject in memberships[semantic]:
            values = rows_by_subject[subject]
            paths.append(np.column_stack((values[:, 0], values[:, 1] - values[:, 2])))
        model_1h = TimeSeriesKMeans(
            n_clusters=1,
            metric="softdtw",
            max_iter=20,
            random_state=0,
        )
        model_1h.fit(to_time_series_dataset(paths))
        centers_1h[semantic] = np.asarray(model_1h.cluster_centers_[0]).tolist()

    return {
        "schema_version": 1,
        "analysis": "exact replay of released Cell 2025 Figure 1H and 1J trajectory displays",
        "interpretation": "retrospective visualization of continuous diversity; not classes",
        "released_analysis": {
            "doi": ANALYSIS_DOI,
            "commit": ANALYSIS_COMMIT,
            "notebook": "scripts/behaviour.ipynb",
            "figure_1j_cell": 8,
            "figure_1h_cell": 25,
        },
        "sources": {
            "trajectory_member": "data/psych_metric_trajectory_fit_df.csv",
            "trajectory_sha256": trajectory_digest,
            "cluster_member": "data/left_right_balanced_cluster_df.csv",
            "cluster_sha256": cluster_digest,
        },
        "environment": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "tslearn": tslearn.__version__,
        },
        "contract": {
            "interpolation_points": 100,
            "figure_1j": (
                "TimeSeriesKMeans(n_clusters=3, metric=softdtw, max_iter=20, "
                "random_state=1) on [right_slope, left_slope]"
            ),
            "figure_1h": (
                "within-cluster TimeSeriesKMeans(n_clusters=1, metric=softdtw, max_iter=20, "
                "random_state=0) on [session, right_slope-left_slope]"
            ),
            "semantic_anchors": ANCHORS,
        },
        "subject_order": subject_order,
        "semantic_label_by_subject": semantic_by_subject,
        "memberships": memberships,
        "average_prop_below": average_prop_below,
        "released_membership_validation": {
            "exact_semantic_membership_match": True,
            "n_subjects": len(released_by_subject),
        },
        "slope_difference_centers": centers_1h,
        "right_left_centers": centers_1j,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=DEFAULT_RELEASED_DIRECTORY / "psych_metric_trajectory_fit_df.csv",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_RELEASED_DIRECTORY / "left_right_balanced_cluster_df.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("figure1hj_trajectories.json"),
    )
    arguments = parser.parse_args()
    result = reproduce(arguments.trajectories.resolve(), arguments.clusters.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "memberships": result["memberships"]}))


if __name__ == "__main__":
    main()
