"""Run the bounded IBL 2021 public learning benchmark."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from behavio import Study
from behavio.compare import ParameterTrajectoryPanel, audit_trajectory_replication
from behavio.evaluate import leave_one_lab_out_splits, leave_one_subject_out_splits
from benchmarks.ibl2021.fetch_data import (
    DEFAULT_DESTINATION,
    EXPECTED_MANIFEST_SHA256,
    load_manifest,
    verify_file,
)
from benchmarks.provenance import render

EXPECTED: dict[str, int | float] = {
    "n_trials": 28_400,
    "n_choice_trials": 28_097,
    "n_easy_trials": 13_964,
    "n_labs": 9,
    "n_subjects": 9,
    "n_sessions": 54,
    "early_easy_accuracy": 0.423_900_270_241_994_34,
    "late_training_easy_accuracy": 0.853_331_571_282_641,
    "accuracy_change": 0.429_431_301_040_646_7,
    "n_improved_subjects": 9,
    "n_leave_subject_out_folds": 9,
    "n_leave_lab_out_folds": 9,
    "n_matching_subject_lab_partitions": 9,
    "n_lab_holdouts_with_multiple_subjects": 0,
}


@dataclass(frozen=True)
class SessionMetrics:
    subject: str
    session: str
    phase: str
    n_trials: int
    n_choice_trials: int
    n_easy_trials: int
    easy_accuracy: float


def build_study(
    session_trials: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Study:
    """Map IBL trial arrays and pinned session metadata to the canonical ``Study``."""

    columns: dict[str, list[Any]] = {
        "subject": [],
        "session": [],
        "trial": [],
        "session_order": [],
        "lab": [],
        "phase": [],
        "task_protocol": [],
        "choice": [],
        "reward": [],
        "signed_stimulus": [],
        "reaction_time": [],
    }
    for metadata, trials in session_trials:
        choice = np.asarray(trials["choice"])
        feedback = np.asarray(trials["feedbackType"])
        contrast_right = np.asarray(trials["contrastRight"], dtype=float)
        contrast_left = np.asarray(trials["contrastLeft"], dtype=float)
        response_times = np.asarray(trials["response_times"], dtype=float)
        stimulus_times = np.asarray(trials["stimOn_times"], dtype=float)
        lengths = {
            len(choice),
            len(feedback),
            len(contrast_right),
            len(contrast_left),
            len(response_times),
            len(stimulus_times),
        }
        if len(lengths) != 1:
            raise ValueError(f"trial columns differ in length for session {metadata['session']}")
        n_trials = lengths.pop()
        signed_stimulus = np.nan_to_num(contrast_right, nan=0.0) - np.nan_to_num(
            contrast_left, nan=0.0
        )
        reaction_time = response_times - stimulus_times

        columns["subject"].extend([str(metadata["subject"])] * n_trials)
        columns["session"].extend([str(metadata["session"])] * n_trials)
        columns["trial"].extend(range(n_trials))
        columns["session_order"].extend([int(metadata["session_order"])] * n_trials)
        columns["lab"].extend([str(metadata["lab"])] * n_trials)
        columns["phase"].extend([str(metadata["phase"])] * n_trials)
        columns["task_protocol"].extend([str(metadata["task_protocol"])] * n_trials)
        columns["choice"].extend(choice.tolist())
        columns["reward"].extend((feedback == 1).astype(np.int8).tolist())
        columns["signed_stimulus"].extend(signed_stimulus.tolist())
        columns["reaction_time"].extend(reaction_time.tolist())
    return Study.from_columns(columns)


def load_study(data_directory: Path) -> Study:
    """Verify and load the 54 pinned Parquet tables without adding a core dependency."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "loading the IBL benchmark requires PyArrow; run with `uv run --with pyarrow`"
        ) from error

    manifest = load_manifest()
    required_columns = (
        "contrastRight",
        "contrastLeft",
        "feedbackType",
        "choice",
        "response_times",
        "stimOn_times",
    )
    loaded: list[tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]] = []
    for row in manifest["sessions"]:
        path = data_directory / f"{row['session']}.pqt"
        if not verify_file(path, row):
            raise ValueError(
                f"missing or mismatched trial table for {row['session']}; run fetch_data first"
            )
        table = parquet.read_table(path, columns=list(required_columns))
        arrays = {
            name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
            for name in required_columns
        }
        loaded.append((row, arrays))
    return build_study(loaded)


def calculate_session_metrics(study: Study) -> list[SessionMetrics]:
    """Calculate easy-trial accuracy for each selected session."""

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in study.chronological_indices():
        grouped[(str(study["subject"][index]), str(study["session"][index]))].append(int(index))

    rows: list[SessionMetrics] = []
    for (subject, session), indices in grouped.items():
        positions = np.asarray(indices, dtype=np.intp)
        choices = np.asarray(study["choice"][positions], dtype=float)
        stimulus = np.asarray(study["signed_stimulus"][positions], dtype=float)
        rewards = np.asarray(study["reward"][positions], dtype=float)
        choice_trials = np.isfinite(choices) & (choices != 0)
        easy_trials = choice_trials & np.isfinite(stimulus) & (np.abs(stimulus) >= 0.5)
        if not np.any(easy_trials):
            raise ValueError(f"session {session} contains no valid easy trials")
        first = positions[0]
        rows.append(
            SessionMetrics(
                subject=subject,
                session=session,
                phase=str(study["phase"][first]),
                n_trials=len(positions),
                n_choice_trials=int(np.count_nonzero(choice_trials)),
                n_easy_trials=int(np.count_nonzero(easy_trials)),
                easy_accuracy=float(np.mean(rewards[easy_trials])),
            )
        )
    return rows


def population_validation_summary(study: Study) -> dict[str, int]:
    """Summarize subject- and lab-held-out coverage without fitting a model."""

    subject_splits = leave_one_subject_out_splits(study)
    lab_splits = leave_one_lab_out_splits(study)
    expected_positions = set(range(len(study)))
    subject_test_positions = [int(row) for split in subject_splits for row in split.test_indices]
    lab_test_positions = [int(row) for split in lab_splits for row in split.test_indices]
    if set(subject_test_positions) != expected_positions or len(subject_test_positions) != len(
        study
    ):
        raise ValueError("leave-subject-out folds must test every trial exactly once")
    if set(lab_test_positions) != expected_positions or len(lab_test_positions) != len(study):
        raise ValueError("leave-lab-out folds must test every trial exactly once")

    subject_partitions = {frozenset(split.test_subjects) for split in subject_splits}
    lab_partitions = {frozenset(split.test_subjects) for split in lab_splits}
    return {
        "n_leave_subject_out_folds": len(subject_splits),
        "n_leave_lab_out_folds": len(lab_splits),
        "n_matching_subject_lab_partitions": len(subject_partitions & lab_partitions),
        "n_lab_holdouts_with_multiple_subjects": sum(
            len(split.test_subjects) > 1 for split in lab_splits
        ),
    }


def run(data_directory: Path, *, check: bool = True) -> dict[str, Any]:
    """Run the benchmark and optionally enforce its numerical regression contract."""

    study = load_study(data_directory)
    session_metrics = calculate_session_metrics(study)
    phase_accuracy: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in session_metrics:
        phase_accuracy[(row.subject, row.phase)].append(row.easy_accuracy)

    subjects = sorted({row.subject for row in session_metrics})
    subject_changes: dict[str, float] = {}
    early_accuracy: list[float] = []
    late_accuracy: list[float] = []
    for subject in subjects:
        early = phase_accuracy[(subject, "early")]
        late = phase_accuracy[(subject, "late_training")]
        if len(early) != 3 or len(late) != 3:
            raise ValueError(f"subject {subject} does not have three sessions in each phase")
        early_mean = float(np.mean(early))
        late_mean = float(np.mean(late))
        early_accuracy.append(early_mean)
        late_accuracy.append(late_mean)
        subject_changes[subject] = late_mean - early_mean

    values: dict[str, int | float] = {
        "n_trials": len(study),
        "n_choice_trials": sum(row.n_choice_trials for row in session_metrics),
        "n_easy_trials": sum(row.n_easy_trials for row in session_metrics),
        "n_labs": len(set(study["lab"])),
        "n_subjects": len(subjects),
        "n_sessions": len(session_metrics),
        "early_easy_accuracy": float(np.mean(early_accuracy)),
        "late_training_easy_accuracy": float(np.mean(late_accuracy)),
        "accuracy_change": float(np.mean(late_accuracy) - np.mean(early_accuracy)),
        "n_improved_subjects": sum(change > 0 for change in subject_changes.values()),
        **population_validation_summary(study),
    }
    subject_labs: list[str] = []
    for subject in subjects:
        labs = {
            str(lab)
            for row_subject, lab in zip(study["subject"], study["lab"], strict=True)
            if str(row_subject) == subject
        }
        if len(labs) != 1:
            raise ValueError(f"subject {subject} must belong to exactly one lab")
        subject_labs.append(labs.pop())
    trajectory_panel = ParameterTrajectoryPanel(
        grid=np.asarray([0.0, 1.0]),
        values=np.column_stack((early_accuracy, late_accuracy)),
        subjects=tuple(subjects),
        groups=tuple(subject_labs),
        clock_name="transition_anchored_phase",
        parameter_name="easy_trial_accuracy",
    )
    trajectory_audit = audit_trajectory_replication(trajectory_panel)
    passed = contract_matches(values)
    if check and not passed:
        differences = {
            key: {"observed": values.get(key), "expected": expected}
            for key, expected in EXPECTED.items()
            if key not in values or not _matches(values[key], expected)
        }
        raise AssertionError(f"IBL 2021 benchmark contract failed: {differences}")
    return {
        "benchmark": "IBL 2021 public learning: early versus late training",
        "source_release": "2021_Q1_IBL_et_al_Behaviour",
        "source_doi": "10.7554/eLife.63711",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        **values,
        "subject_accuracy_change": subject_changes,
        "trajectory_shape_replication_audit": trajectory_audit.to_dict(),
        "contract_passed": passed,
    }


def contract_matches(values: Mapping[str, int | float]) -> bool:
    return bool(EXPECTED) and all(
        key in values and _matches(values[key], expected) for key, expected in EXPECTED.items()
    )


def _matches(observed: int | float, expected: int | float) -> bool:
    if isinstance(expected, int):
        return observed == expected
    return math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--no-check", action="store_true", help="report without enforcing contract")
    parser.add_argument("--output", type=Path, help="also write the JSON result to this path")
    args = parser.parse_args()
    result = run(args.data.resolve(), check=not args.no_check)
    rendered = render(result, allow_nan=True)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
