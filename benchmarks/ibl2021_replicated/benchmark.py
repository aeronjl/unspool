"""Run the outcome-blind, replicated-lab IBL endpoint-window benchmark."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from behavio import (
    Study,
    TrajectoryPanel,
    compare_trajectory_shapes,
    leave_one_lab_out_splits,
    read_ibl_one_sessions,
)
from benchmarks.ibl2021.refresh_manifest import PUBLIC_PASSWORD
from benchmarks.ibl2021_replicated.manifest import (
    EXPECTED_MANIFEST_SHA256,
    load_manifest,
    sources_from_manifest,
)
from benchmarks.provenance import render

DEFAULT_CACHE = Path(__file__).with_name("data")


@dataclass(frozen=True, slots=True)
class SessionAccuracy:
    """Easy-trial accuracy for one retained source session."""

    subject: str
    lab: str
    phase: str
    window_position: int
    source_session_order: int
    n_choice_trials: int
    n_easy_trials: int
    easy_accuracy: float


def load_study(cache_directory: Path = DEFAULT_CACHE) -> Study:
    """Load the release cache once, then adapt all exact dataset UUIDs through ONE."""

    try:
        from one.api import ONE
    except ImportError as error:
        raise RuntimeError("the replicated IBL benchmark requires `behavio[ibl]`") from error
    manifest = load_manifest()
    cache_directory.mkdir(parents=True, exist_ok=True)
    one = ONE(
        base_url=str(manifest["public_alyx_url"]),
        password=PUBLIC_PASSWORD,
        silent=True,
        cache_dir=cache_directory,
    )
    one.load_cache(tag=str(manifest["release_tag"]))
    return read_ibl_one_sessions(sources_from_manifest(manifest), client=one)


def calculate_session_accuracy(study: Study) -> tuple[SessionAccuracy, ...]:
    """Calculate rewarded fraction on valid easy-choice trials per source session."""

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in study.chronological_indices():
        grouped[(str(study["subject"][index]), str(study["session"][index]))].append(int(index))
    rows: list[SessionAccuracy] = []
    for (subject, _session), indices in grouped.items():
        positions = np.asarray(indices, dtype=np.intp)
        source_choice = np.asarray(study["source_choice"][positions], dtype=np.float64)
        feedback = np.asarray(study["source_feedback"][positions], dtype=np.float64)
        left = np.asarray(study["contrastLeft"][positions], dtype=np.float64)
        right = np.asarray(study["contrastRight"][positions], dtype=np.float64)
        signed_contrast = np.nan_to_num(right, nan=0.0) - np.nan_to_num(left, nan=0.0)
        choice_trials = np.isfinite(source_choice) & (source_choice != 0)
        easy_trials = (
            choice_trials & np.isfinite(signed_contrast) & (np.abs(signed_contrast) >= 0.5)
        )
        if not np.any(easy_trials):
            raise ValueError(f"retained session for {subject} contains no valid easy trials")
        first = positions[0]
        rows.append(
            SessionAccuracy(
                subject=subject,
                lab=str(study["lab"][first]),
                phase=str(study["phase"][first]),
                window_position=int(study["window_position"][first]),
                source_session_order=int(study["session_order"][first]),
                n_choice_trials=int(np.count_nonzero(choice_trials)),
                n_easy_trials=int(np.count_nonzero(easy_trials)),
                easy_accuracy=float(np.mean(feedback[easy_trials] == 1)),
            )
        )
    return tuple(rows)


def run(
    cache_directory: Path = DEFAULT_CACHE,
    *,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 20_261,
) -> dict[str, Any]:
    """Adapt the pinned cohort and retain bounded cross-lab descriptive evidence."""

    manifest = load_manifest()
    study = load_study(cache_directory)
    sessions = calculate_session_accuracy(study)
    subjects = tuple(dict.fromkeys(row.subject for row in sessions))
    subject_labs: dict[str, str] = {}
    trajectories: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    subject_changes: dict[str, float] = {}
    for subject in subjects:
        rows = sorted(
            (row for row in sessions if row.subject == subject),
            key=lambda row: row.window_position,
        )
        if len(rows) != 6 or [row.window_position for row in rows] != list(range(6)):
            raise ValueError(f"subject {subject} does not have all six endpoint-window positions")
        labs = {row.lab for row in rows}
        if len(labs) != 1:
            raise ValueError(f"subject {subject} belongs to multiple labs")
        subject_labs[subject] = labs.pop()
        trajectories[subject] = np.asarray([row.easy_accuracy for row in rows])
        subject_changes[subject] = float(np.mean(trajectories[subject][3:])) - float(
            np.mean(trajectories[subject][:3])
        )

    panel = TrajectoryPanel(
        grid=np.arange(6, dtype=np.float64),
        values=np.stack([trajectories[subject] for subject in subjects]),
        subjects=subjects,
        groups=tuple(subject_labs[subject] for subject in subjects),
        clock_name="endpoint_window_position",
        parameter_name="session_easy_trial_accuracy",
    )
    comparison = compare_trajectory_shapes(
        panel,
        minimum_subjects_per_group=int(manifest["minimum_subjects_per_lab"]),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    lab_summaries: dict[str, dict[str, int | float]] = {}
    for lab in panel.group_order:
        lab_subjects = [subject for subject in subjects if subject_labs[subject] == lab]
        changes = [subject_changes[subject] for subject in lab_subjects]
        lab_summaries[str(lab)] = {
            "subjects": len(lab_subjects),
            "mean_early_accuracy": float(
                np.mean([np.mean(trajectories[subject][:3]) for subject in lab_subjects])
            ),
            "mean_late_training_accuracy": float(
                np.mean([np.mean(trajectories[subject][3:]) for subject in lab_subjects])
            ),
            "mean_accuracy_change": float(np.mean(changes)),
            "subjects_with_positive_change": int(np.count_nonzero(np.asarray(changes) > 0)),
        }
    early_accuracies = [float(np.mean(trajectories[subject][:3])) for subject in subjects]
    late_accuracies = [float(np.mean(trajectories[subject][3:])) for subject in subjects]
    overall = {
        "mean_early_accuracy": float(np.mean(early_accuracies)),
        "mean_late_training_accuracy": float(np.mean(late_accuracies)),
        "mean_accuracy_change": float(np.mean(list(subject_changes.values()))),
        "subjects_with_positive_change": int(
            np.count_nonzero(np.asarray(list(subject_changes.values())) > 0)
        ),
    }

    lab_splits = leave_one_lab_out_splits(study)
    tested = np.concatenate([split.test_indices for split in lab_splits])
    provenance_columns = (
        "source_alyx_url",
        "source_ibl_release_tag",
        "source_ibl_session_id",
        "source_ibl_dataset_id",
        "source_ibl_dataset_path",
        "source_ibl_dataset_size",
        "source_ibl_dataset_md5",
    )
    contract = {
        "manifest_is_pinned": manifest["sessions_sha256"] == EXPECTED_MANIFEST_SHA256,
        "all_eligible_subjects_retained": len(subjects) == int(manifest["n_subjects"]),
        "all_labs_replicated": comparison.replication_audit.inferentially_ready,
        "all_endpoint_windows_complete": len(sessions) == int(manifest["n_sessions"]),
        "all_labs_improve_descriptively": all(
            summary["mean_accuracy_change"] > 0 for summary in lab_summaries.values()
        ),
        "all_subjects_improve_descriptively": overall["subjects_with_positive_change"]
        == len(subjects),
        "complete_lab_holdout": len(lab_splits) == int(manifest["n_labs"])
        and np.array_equal(np.sort(tested), np.arange(len(study)))
        and all(
            len(split.test_subjects) == int(manifest["subjects_per_lab"][split.held_out_group])
            for split in lab_splits
        ),
        "provenance_is_trial_addressable": all(
            name in study.columns and len(study[name]) == len(study) for name in provenance_columns
        ),
        "source_choice_semantics_preserved": "source_choice" in study.columns
        and "choice" not in study.columns
        and set(np.unique(study["source_choice"])).issubset({-1, 0, 1}),
    }
    result = {
        "benchmark": "IBL 2021 replicated-lab endpoint-window trajectories",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_release": str(manifest["release_tag"]),
        "source_doi": "10.7554/eLife.63711",
        "selection": {
            "outcome_blind": True,
            "subjects": len(subjects),
            "labs": len(panel.group_order),
            "sessions": len(sessions),
            "subjects_per_lab": dict(manifest["subjects_per_lab"]),
            "minimum_subjects_per_lab": int(manifest["minimum_subjects_per_lab"]),
            "source_bytes": int(manifest["total_file_size"]),
        },
        "study": {
            "trials": len(study),
            "choice_trials": sum(row.n_choice_trials for row in sessions),
            "easy_trials": sum(row.n_easy_trials for row in sessions),
            "source_datasets": len(set(study["source_ibl_dataset_id"].tolist())),
            "source_choice_counts": {
                str(key): value
                for key, value in sorted(Counter(study["source_choice"].tolist()).items())
            },
        },
        "clock": {
            "name": panel.clock_name,
            "grid": panel.grid.tolist(),
            "interpretation": str(manifest["clock_boundary"]),
            "source_session_order_retained": True,
        },
        "overall": overall,
        "lab_summaries": lab_summaries,
        "trajectory_comparison": comparison.to_dict(),
        "population_validation": {
            "leave_one_lab_out_folds": len(lab_splits),
            "held_out_subjects": {
                str(split.held_out_group): len(split.test_subjects) for split in lab_splits
            },
            "all_trials_tested_once": bool(np.array_equal(np.sort(tested), np.arange(len(study)))),
        },
        "contract": contract,
        "contract_passed": all(contract.values()),
    }
    if not result["contract_passed"]:
        failed = sorted(name for name, passed in contract.items() if not passed)
        raise AssertionError(f"replicated IBL benchmark contract failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_261)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    result = run(
        args.cache,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered = render(result, allow_nan=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
