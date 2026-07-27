"""Reproduce the released Cell 2025 GP plus soft-DTW trajectory clustering.

Run this module in the isolated versions recorded by the paper's ``environment.yml``;
these optional packages are deliberately not Unspool runtime dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cell2025.benchmark import calculate_session_metrics, load_study
from benchmarks.cell2025.fetch_data import DEFAULT_DESTINATION, MEMBER_SHA256, sha256

ANALYSIS_DOI = "10.6084/m9.figshare.28877942.v1"
ANCHORS = {"DAP110": "right", "DAP009": "balanced", "DAP028": "left"}
RELEASED_CLUSTER_SHA256 = "6cd23f17c6634532bab1af5fd11ed212c8d9431a81a80c8ad28261fb63ab49dc"


def reproduce(path: Path, *, released_clusters: Path | None = None) -> dict[str, Any]:
    """Return released trajectory fits and semantic cluster membership."""

    import pandas as pd
    import scipy
    import sklearn
    import tslearn
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset

    digest = sha256(path)
    if digest != MEMBER_SHA256:
        raise ValueError(f"input checksum mismatch: observed {digest}, expected {MEMBER_SHA256}")
    study = load_study(path)
    metrics_by_subject: dict[str, list[Any]] = defaultdict(list)
    for row in calculate_session_metrics(study):
        metrics_by_subject[row.subject].append(row)
    eligible = {
        subject for subject, rows in metrics_by_subject.items() if rows[0].session_order < 3
    }
    # The released session-level groupby sorts mouse identifiers. K-means initialization
    # is order-sensitive, so preserve that otherwise implicit dependency explicitly.
    subject_order = tuple(sorted(str(subject) for subject in eligible))

    trajectories: list[np.ndarray] = []
    fitted: dict[str, dict[str, Any]] = {}
    for subject in subject_order:
        rows = sorted(metrics_by_subject[subject], key=lambda row: row.session_order)
        x = np.asarray([row.session_order for row in rows], dtype=np.float64).reshape(-1, 1)
        right = np.asarray([row.right_slope for row in rows], dtype=np.float64)
        left = np.asarray([row.left_slope for row in rows], dtype=np.float64)
        kernel = ConstantKernel() * RBF(length_scale_bounds=(3, 1e5)) + WhiteKernel()
        regressor = GaussianProcessRegressor(
            kernel=kernel,
            random_state=0,
            n_restarts_optimizer=10,
        )
        x_new = np.linspace(float(x.min()), float(x.max()), 100).reshape(-1, 1)
        right_new = regressor.fit(x, right).predict(x_new)
        right_kernel = str(regressor.kernel_)
        left_new = regressor.fit(x, left).predict(x_new)
        left_kernel = str(regressor.kernel_)
        xy = np.column_stack((right_new, left_new))
        trajectories.append(xy)
        fitted[subject] = {
            "paper_session_min": int(x.min()),
            "paper_session_max": int(x.max()),
            "right_slope": right_new.tolist(),
            "left_slope": left_new.tolist(),
            "right_kernel": right_kernel,
            "left_kernel": left_kernel,
        }

    dataset = to_time_series_dataset(trajectories)
    model = TimeSeriesKMeans(
        n_clusters=3,
        metric="softdtw",
        max_iter=20,
        random_state=1,
    )
    model.fit(dataset)
    raw_by_subject = {
        subject: int(label) for subject, label in zip(subject_order, model.labels_, strict=True)
    }
    anchor_labels = {semantic: raw_by_subject[subject] for subject, semantic in ANCHORS.items()}
    if len(set(anchor_labels.values())) != 3:
        raise RuntimeError("released semantic anchors no longer identify three clusters")
    semantic_by_raw = {raw: semantic for semantic, raw in anchor_labels.items()}
    semantic_by_subject = {
        subject: semantic_by_raw[label] for subject, label in raw_by_subject.items()
    }
    memberships = {
        semantic: [subject for subject in subject_order if semantic_by_subject[subject] == semantic]
        for semantic in ("left", "balanced", "right")
    }
    released_validation = None
    if released_clusters is not None:
        released_digest = sha256(released_clusters)
        if released_digest != RELEASED_CLUSTER_SHA256:
            raise ValueError(
                "released cluster checksum mismatch: "
                f"observed {released_digest}, expected {RELEASED_CLUSTER_SHA256}"
            )
        with released_clusters.open(newline="", encoding="utf-8") as handle:
            released_by_subject = {
                row["mouseNum"]: row["cluster"] for row in csv.DictReader(handle)
            }
        if semantic_by_subject != released_by_subject:
            mismatches = {
                subject: {
                    "reproduced": semantic_by_subject.get(subject),
                    "released": released_by_subject.get(subject),
                }
                for subject in sorted(set(semantic_by_subject) | set(released_by_subject))
                if semantic_by_subject.get(subject) != released_by_subject.get(subject)
            }
            raise RuntimeError(f"trajectory cluster reproduction mismatch: {mismatches}")
        released_validation = {
            "source_member": "data/left_right_balanced_cluster_df.csv",
            "source_sha256": released_digest,
            "exact_semantic_membership_match": True,
            "n_subjects": len(released_by_subject),
        }
    return {
        "analysis": "released Gaussian-process plus soft-DTW trajectory visualization",
        "interpretation": "retrospective visualization of continuous diversity; not classes",
        "source_member_sha256": digest,
        "analysis_doi": ANALYSIS_DOI,
        "environment": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "tslearn": tslearn.__version__,
        },
        "contract": {
            "gp": "ConstantKernel * RBF(length_scale_bounds=(3, 1e5)) + WhiteKernel",
            "gp_random_state": 0,
            "gp_optimizer_restarts": 10,
            "interpolation_points": 100,
            "clustering": "TimeSeriesKMeans(n_clusters=3, metric=softdtw, max_iter=20)",
            "cluster_random_state": 1,
            "semantic_anchors": ANCHORS,
        },
        "subject_order": list(subject_order),
        "raw_label_by_subject": raw_by_subject,
        "semantic_label_by_subject": semantic_by_subject,
        "memberships": memberships,
        "released_membership_validation": released_validation,
        "cluster_centers": np.asarray(model.cluster_centers_).tolist(),
        "fitted_trajectories": fitted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("trajectory_clusters.json"),
    )
    parser.add_argument("--released-clusters", type=Path)
    arguments = parser.parse_args()
    result = reproduce(
        arguments.data.resolve(),
        released_clusters=(
            arguments.released_clusters.resolve() if arguments.released_clusters else None
        ),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "memberships": result["memberships"]}))


if __name__ == "__main__":
    main()
