"""Run the prospective Chen et al. (2021) restless-bandit literature recipe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.chen2021_bandit.fetch_data import (
    ARCHIVE_MD5,
    ARCHIVE_SHA256,
    DATASET_DOI,
    DEFAULT_DESTINATION,
    ZENODO_RECORD,
)
from unspool import (
    BiasOnly,
    BinaryQLearning,
    ChoiceSpec,
    ModelRecoveryScenario,
    Perseveration,
    RewardSpec,
    Study,
    TaskSpec,
    WinStayLoseShift,
    cohort_forward_session_splits,
    compare_models,
    run_model_recovery,
)

PAPER_DOI = "10.7554/eLife.69748"
TRIALS_PER_SESSION = 100
N_SESSIONS = 8
N_SUBJECTS = 32
TRAIN_SESSIONS = 7
COMPARISON_BOOTSTRAP_RESAMPLES = 5_000
COMPARISON_SEED = 2_402
RECOVERY_REPEATS = 5
RECOVERY_SEED = 2_403
EXPECTED_HEADERS = (
    "",
    "left",
    "right",
    "choice",
    "reward",
    "state",
    "RT",
    "retrieval",
    "initiation",
)
EXPECTED_POINT_LOSSES = {
    "bias": 0.7078661247261041,
    "perseveration": 0.6621524765727385,
    "win-stay-lose-shift": 0.6118685044581251,
    "q-learning": 0.603341807523871,
}


def load_study(source: Path, *, trials_per_session: int = TRIALS_PER_SESSION) -> Study:
    """Map the source CSV tree into a validated canonical longitudinal ``Study``."""

    if isinstance(trials_per_session, bool) or not isinstance(trials_per_session, int):
        raise TypeError("trials_per_session must be an integer")
    if trials_per_session < 1:
        raise ValueError("trials_per_session must be positive")
    source = source.resolve()
    expected_files = {
        source / f"session{session}" / f"{mouse}.csv"
        for session in range(1, N_SESSIONS + 1)
        for mouse in range(1, N_SUBJECTS + 1)
    }
    actual_files = set(source.glob("session*/*.csv"))
    if actual_files != expected_files:
        missing = sorted(str(path.relative_to(source)) for path in expected_files - actual_files)
        extra = sorted(str(path.relative_to(source)) for path in actual_files - expected_files)
        raise ValueError(f"source file contract failed; missing={missing}, extra={extra}")

    columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "subject",
            "session",
            "trial",
            "source_trial",
            "session_order",
            "choice",
            "reward",
            "reward_probability_0",
            "reward_probability_1",
            "sex",
            "source_state",
        )
    }
    for session in range(1, N_SESSIONS + 1):
        for mouse in range(1, N_SUBJECTS + 1):
            path = source / f"session{session}" / f"{mouse}.csv"
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != EXPECTED_HEADERS:
                    raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
                for trial, row in enumerate(reader):
                    if trial >= trials_per_session:
                        break
                    left = _probability(row["left"], path, trial, "left")
                    right = _probability(row["right"], path, trial, "right")
                    choice = _binary_integer(row["choice"], path, trial, allowed=(1, 2)) - 1
                    reward = _binary_integer(row["reward"], path, trial, allowed=(0, 1))
                    source_state = _integer(row["state"], path, trial, "state")
                    source_trial = _integer(row[""], path, trial, "source trial")
                    columns["subject"].append(f"mouse-{mouse:02d}")
                    columns["session"].append(f"session-{session}")
                    columns["trial"].append(trial)
                    columns["source_trial"].append(source_trial)
                    columns["session_order"].append(session - 1)
                    columns["choice"].append(choice)
                    columns["reward"].append(reward)
                    columns["reward_probability_0"].append(left)
                    columns["reward_probability_1"].append(right)
                    columns["sex"].append("male" if mouse <= 16 else "female")
                    columns["source_state"].append(source_state)
    return Study.from_columns(columns)


def run(source: Path, *, check: bool = True) -> dict[str, Any]:
    """Execute the fixed 7→8 forecast and its design-matched recovery experiment."""

    study = load_study(source)
    task = TaskSpec(
        choice=ChoiceSpec(options=(0, 1)),
        reward=RewardSpec(minimum=0, maximum=1),
    )
    validation = task.validate(study)
    splits = cohort_forward_session_splits(study, min_train_sessions=TRAIN_SESSIONS)
    if len(splits) != 1:
        raise AssertionError(f"expected one 7→8 cohort fold, observed {len(splits)}")

    models = _models()
    comparison = compare_models(
        models,
        study,
        splits,
        aggregation_column="subject",
        bootstrap_resamples=COMPARISON_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=COMPARISON_SEED,
    )
    q_model = models["q-learning"]
    q_fit = comparison.result_for("q-learning").evaluations[0].fit
    q_parameters = q_model.parameter_components(q_fit)
    test = study.take(splits[0].test_indices)
    q_trajectory = q_model.value_trajectory(test, q_fit)
    example = _example_session(test, q_trajectory, subject="mouse-01")

    recovery_candidates = {
        "win-stay-lose-shift": models["win-stay-lose-shift"],
        "q-learning": q_model,
    }
    recovery = run_model_recovery(
        study,
        (
            ModelRecoveryScenario(
                name="observable-reward-history",
                truth_label="win-stay-lose-shift",
                generator=models["win-stay-lose-shift"],
                parameters={"intercept": 0.0, "win_stay": 1.2, "lose_shift": 1.0},
            ),
            ModelRecoveryScenario(
                name="incremental-value-learning",
                truth_label="q-learning",
                generator=q_model,
                parameters=q_model.parameters_from_components(
                    learning_rate=0.10,
                    inverse_temperature=3.0,
                    choice_bias=0.0,
                    perseveration=0.4,
                ),
            ),
        ),
        recovery_candidates,
        repeats=RECOVERY_REPEATS,
        seed=RECOVERY_SEED,
        splitter=lambda value: cohort_forward_session_splits(
            value, min_train_sessions=TRAIN_SESSIONS
        ),
        splitter_name="cohort-forward-session-7-to-8",
        aggregation_column="subject",
        tie_tolerance=0.002,
    )

    comparison_payload = comparison.to_dict()
    confusion = recovery.confusion_matrix()
    payload: dict[str, Any] = {
        "benchmark": "Chen et al. (2021) restless-bandit prospective recipe",
        "paper_doi": PAPER_DOI,
        "data": {
            "dataset_doi": DATASET_DOI,
            "zenodo_record": ZENODO_RECORD,
            "archive_md5": ARCHIVE_MD5,
            "archive_sha256": ARCHIVE_SHA256,
            "license": "CC0-1.0",
        },
        "design": {
            "selection_status": "literature-shaped analysis; not a reproduction of the sex effect",
            "n_trials": len(study),
            "n_subjects": len(study.subjects),
            "n_sessions": N_SESSIONS,
            "trials_per_session_cap": TRIALS_PER_SESSION,
            "cap_rule": "first source rows, declared before inspecting outcomes",
            "train_session_orders": list(range(TRAIN_SESSIONS)),
            "test_session_orders": [TRAIN_SESSIONS],
            "aggregation_unit": "subject",
            "source_state_role": "retained provenance only; never treated as ground truth",
            "task_validation": {
                "n_trials": validation.n_trials,
                "n_observed_choices": validation.n_observed_choices,
                "n_omissions": validation.n_omissions,
                "choice_counts": [[value, count] for value, count in validation.choice_counts],
                "has_rewards": validation.has_rewards,
                "has_response_times": validation.has_response_times,
            },
        },
        "comparison": comparison_payload,
        "q_learning_training_fit": {
            "learning_rate": q_parameters.learning_rate,
            "inverse_temperature": q_parameters.inverse_temperature,
            "choice_bias": q_parameters.choice_bias,
            "perseveration": q_parameters.perseveration,
            "audit": q_fit.audit().to_dict(),
        },
        "example_heldout_session": example,
        "recovery": {
            "candidate_labels": list(recovery.candidate_labels),
            "scenario_names": list(recovery.scenario_names),
            "truth_labels": list(recovery.truth_labels),
            "selected_labels": list(recovery.selected_labels),
            "mean_log_probabilities": recovery.mean_log_probabilities.tolist(),
            "audit_statuses": [[status.value for status in row] for row in recovery.audit_statuses],
            "audit_issue_codes": [
                [list(codes) for codes in row] for row in recovery.audit_issue_codes
            ],
            "confusion": {
                "truth_labels": list(confusion.truth_labels),
                "selected_labels": list(confusion.selected_labels),
                "counts": confusion.counts.tolist(),
                "rates": confusion.rates.tolist(),
            },
            "repeats": recovery.repeats,
            "seed": recovery.root_seed,
            "tie_tolerance": recovery.tie_tolerance,
            "aggregation_unit": recovery.aggregation_column,
            "interpretation_boundary": (
                "discrimination evidence for these two parameter regimes and this fixed design; "
                "not evidence that either generated the observed animals"
            ),
        },
    }
    payload["contract_passed"] = contract_matches(payload)
    if check and not payload["contract_passed"]:
        raise AssertionError("Chen 2021 prospective recipe contract failed")
    return payload


def contract_matches(payload: Mapping[str, Any]) -> bool:
    """Check the stable denominators, point estimates, audits, and recovery result."""

    try:
        design = payload["design"]
        comparison = payload["comparison"]
        models = comparison["models"]
        recovery = payload["recovery"]
        counts_match = (
            design["n_trials"] == 25_279
            and design["n_subjects"] == N_SUBJECTS
            and design["task_validation"]["n_observed_choices"] == 25_279
            and design["task_validation"]["n_omissions"] == 0
        )
        losses_match = all(
            math.isclose(
                models[name]["unit_balanced_log_loss"],
                expected,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            for name, expected in EXPECTED_POINT_LOSSES.items()
        )
        audits_pass = all(models[name]["audit_status"] == "pass" for name in models)
        recovery_matches = recovery["selected_labels"] == [
            "win-stay-lose-shift"
        ] * RECOVERY_REPEATS + ["q-learning"] * RECOVERY_REPEATS and recovery["confusion"][
            "counts"
        ] == [[5, 0, 0], [0, 5, 0]]
        return bool(counts_match and losses_match and audits_pass and recovery_matches)
    except (KeyError, TypeError):
        return False


def write_result(payload: Mapping[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _models() -> dict[str, Any]:
    return {
        "bias": BiasOnly(l2=0.01),
        "perseveration": Perseveration(l2=0.01),
        "win-stay-lose-shift": WinStayLoseShift(l2=0.01),
        "q-learning": BinaryQLearning(n_restarts=2, random_seed=2_401),
    }


def _example_session(test: Study, trajectory: Any, *, subject: str) -> dict[str, Any]:
    positions = np.flatnonzero(test["subject"] == subject)
    return {
        "subject": subject,
        "session": str(test["session"][positions[0]]),
        "trial": test["trial"][positions].astype(int).tolist(),
        "choice": test["choice"][positions].astype(int).tolist(),
        "reward": test["reward"][positions].astype(int).tolist(),
        "reward_probability_0": test["reward_probability_0"][positions].astype(float).tolist(),
        "reward_probability_1": test["reward_probability_1"][positions].astype(float).tolist(),
        "q_value_0": trajectory.pre_choice[positions, 0].tolist(),
        "q_value_1": trajectory.pre_choice[positions, 1].tolist(),
    }


def _integer(value: str, path: Path, row: int, label: str) -> int:
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{path}: row {row + 2} has non-numeric {label}") from None
    if not np.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{path}: row {row + 2} has non-integer {label}")
    return int(parsed)


def _binary_integer(
    value: str,
    path: Path,
    row: int,
    *,
    allowed: tuple[int, int],
) -> int:
    parsed = _integer(value, path, row, "binary value")
    if parsed not in allowed:
        raise ValueError(f"{path}: row {row + 2} must be one of {allowed}")
    return parsed


def _probability(value: str, path: Path, row: int, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{path}: row {row + 2} has non-numeric {label}") from None
    if not np.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{path}: row {row + 2} has invalid {label} probability")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("result.json"),
    )
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    result = run(args.data, check=not args.no_check)
    write_result(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
