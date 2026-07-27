"""Compare static and drifting hierarchical forecasts on the replicated IBL cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.ibl2021_replicated.benchmark import DEFAULT_CACHE, load_study
from benchmarks.ibl2021_replicated.manifest import EXPECTED_MANIFEST_SHA256
from unspool import (
    HierarchicalBernoulliHistoryGLM,
    HierarchicalSmoothBernoulliHistoryGLM,
    ProspectiveComparisonReport,
    Study,
    cohort_forward_session_splits,
    compare_models,
    leave_one_lab_out_session_forecast_splits,
)

KNOTS = (0.0, 2.0, 5.0)
TRIALS_PER_SESSION = 100
TRAIN_SESSION_COUNT = 5
METHODS = ("static_partial_pooling", "hierarchical_smooth_drift")
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_727


def build_panel(study: Study, *, trials_per_session: int = TRIALS_PER_SESSION) -> Study:
    """Apply the fixed source mapping and outcome-blind per-session trial cap."""

    if (
        isinstance(trials_per_session, bool)
        or not isinstance(trials_per_session, int)
        or trials_per_session < 1
    ):
        raise ValueError("trials_per_session must be a positive integer")
    required = {
        "source_choice",
        "contrastLeft",
        "contrastRight",
        "lab",
        "phase",
        "window_position",
        "source_ibl_dataset_id",
    }
    missing = required - set(study.columns)
    if missing:
        raise ValueError(f"replicated IBL study is missing required columns: {sorted(missing)}")

    counts: Counter[tuple[Any, Any]] = Counter()
    capped: list[int] = []
    for raw_index in study.chronological_indices():
        index = int(raw_index)
        key = (_scalar(study["subject"][index]), _scalar(study["session"][index]))
        if counts[key] < trials_per_session:
            capped.append(index)
        counts[key] += 1
    capped_positions = np.asarray(capped, dtype=np.intp)
    source_choice = np.asarray(study["source_choice"][capped_positions], dtype=np.float64)
    valid_choice = np.isfinite(source_choice) & (source_choice != 0)
    positions = capped_positions[valid_choice]
    source_choice = source_choice[valid_choice]
    left = np.asarray(study["contrastLeft"][positions], dtype=np.float64)
    right = np.asarray(study["contrastRight"][positions], dtype=np.float64)
    stimulus = np.nan_to_num(right, nan=0.0) - np.nan_to_num(left, nan=0.0)
    if not np.all(np.isfinite(stimulus)):
        raise ValueError("signed contrast must be finite after the declared source mapping")

    panel = Study(
        {
            "subject": study["subject"][positions],
            "session": study["session"][positions],
            "trial": study["trial"][positions],
            "session_order": np.asarray(study["window_position"][positions], dtype=np.int64),
            "source_session_order": study["session_order"][positions],
            "lab": study["lab"][positions],
            "phase": study["phase"][positions],
            "choice": (source_choice < 0).astype(np.int8),
            "stimulus": stimulus,
            "source_choice": source_choice.astype(np.int8),
            "source_ibl_dataset_id": study["source_ibl_dataset_id"][positions],
        }
    )
    for subject in panel.subjects:
        subject_rows = panel["subject"] == subject
        if set(panel["session_order"][subject_rows]) != set(range(6)):
            raise ValueError(f"subject {subject!r} lacks a retained endpoint-window session")
    return panel


def analyze_panel(
    panel: Study,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate within-animal and held-out-lab forecasts on the same final session."""

    within_splits = cohort_forward_session_splits(
        panel,
        min_train_sessions=TRAIN_SESSION_COUNT,
        horizon=1,
    )
    if len(within_splits) != 1:
        raise ValueError("the six-position panel must produce exactly one cohort forecast")
    lab_splits = leave_one_lab_out_session_forecast_splits(
        panel,
        train_session_count=TRAIN_SESSION_COUNT,
        horizon=1,
    )
    within = compare_models(
        _models(),
        panel,
        within_splits,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    lab_transfer = compare_models(
        _models(),
        panel,
        lab_splits,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    lab_balanced = _lab_balanced_summary(
        lab_transfer,
        panel,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 2,
    )
    final_rows = np.flatnonzero(panel["session_order"] == 5)
    contracts = {
        "balanced_six_position_panel": all(
            set(panel["session_order"][panel["subject"] == subject]) == set(range(6))
            for subject in panel.subjects
        ),
        "within_subject_final_session_scored_once": all(
            result.n_scored_observations == len(final_rows) for result in within.model_results
        ),
        "held_out_lab_final_session_scored_once": all(
            result.n_scored_observations == len(final_rows) for result in lab_transfer.model_results
        ),
        "every_lab_held_out_once": len(lab_splits) == len(set(panel["lab"])),
        "all_training_precedes_testing": all(
            max(within_splits[0].train_session_orders[subject])
            < min(within_splits[0].test_session_orders[subject])
            for subject in within_splits[0].subjects
        )
        and all(
            max(split.train_session_orders) < min(split.test_session_orders) for split in lab_splits
        ),
        "lab_forecasts_use_unseen_subjects": all(
            set(split.train_subjects).isdisjoint(split.test_subjects) for split in lab_splits
        ),
        "all_fits_numerically_usable": all(
            result.audit_status.value != "fail"
            for report in (within, lab_transfer)
            for result in report.model_results
        ),
    }
    if not all(contracts.values()):
        failed = sorted(name for name, passed in contracts.items() if not passed)
        raise AssertionError(f"prospective IBL benchmark contract failed: {failed}")
    return {
        "panel": _panel_summary(panel),
        "within_subject_future_session": within.to_dict(),
        "held_out_lab_future_session": {
            **lab_transfer.to_dict(),
            "lab_balanced_subject_log_loss": lab_balanced,
        },
        "contract": contracts,
        "contract_passed": True,
    }


def run(
    cache_directory: Path = DEFAULT_CACHE,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Load the exact public cohort and execute the fixed prospective comparison."""

    panel = build_panel(load_study(cache_directory))
    return {
        "benchmark": "replicated IBL prospective hierarchical model comparison",
        "source": {
            "release": "2021_Q1_IBL_et_al_Behaviour",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "doi": "10.7554/eLife.63711",
        },
        "analysis_contract": {
            "models": list(METHODS),
            "panel": "up to the first 100 source rows in each outcome-blind endpoint window",
            "choice_eligibility": "drop no-go choice=0 only after the source-row cap",
            "outcome": "rightward binary choice; IBL source choice -1 maps to 1",
            "covariates": ["signed contrast", "one-session-reset choice lag"],
            "clock": "outcome-blind ordinal endpoint window_position; not elapsed time",
            "forecast": "position 5 from positions 0 through 4",
            "prediction_mode": (
                "filtered one-step-ahead scoring within the test session; not open-loop"
            ),
            "population_forecast": (
                "position 5 in an entirely held-out lab from positions 0 through 4 "
                "in all other labs"
            ),
            "learned_preprocessing": "none",
            "hyperparameter_source": "fixed synthetic trajectory-recovery benchmark",
            "knots": list(KNOTS),
            "hyperparameters": {
                "l2": 0.02,
                "smoothness": 3.0,
                "subject_scale": 0.4,
                "subject_smoothness": 3.0,
            },
            "primary_metric": "mean subject-level prospective log loss",
            "lab_transfer_secondary_metric": (
                "mean within-lab subject loss, then equal mean across held-out labs"
            ),
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
        },
        **analyze_panel(
            panel,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def _models() -> Mapping[str, Any]:
    common = {"covariates": ("stimulus",), "choice_lags": 1, "l2": 0.02}
    return {
        "static_partial_pooling": HierarchicalBernoulliHistoryGLM(
            **common,
            subject_scale=0.4,
        ),
        "hierarchical_smooth_drift": HierarchicalSmoothBernoulliHistoryGLM(
            **common,
            knots=KNOTS,
            smoothness=3.0,
            subject_scale=0.4,
            subject_smoothness=3.0,
        ),
    }


def _panel_summary(panel: Study) -> dict[str, Any]:
    subjects = tuple(_scalar(subject) for subject in panel.subjects)
    subject_labs = {
        subject: _scalar(panel["lab"][np.flatnonzero(panel["subject"] == subject)[0]])
        for subject in subjects
    }
    return {
        "trials": len(panel),
        "subjects": len(subjects),
        "labs": len(set(subject_labs.values())),
        "sessions": 6 * len(subjects),
        "trials_by_position": {
            str(position): int(np.count_nonzero(panel["session_order"] == position))
            for position in range(6)
        },
        "subjects_per_lab": dict(sorted(Counter(subject_labs.values()).items())),
        "source_choice_counts": {
            str(value): count
            for value, count in sorted(Counter(panel["source_choice"].tolist()).items())
        },
    }


def _lab_balanced_summary(
    report: ProspectiveComparisonReport,
    panel: Study,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    subject_labs: dict[Any, Any] = {}
    for subject in panel.subjects:
        rows = np.flatnonzero(panel["subject"] == subject)
        labs = {_scalar(value) for value in panel["lab"][rows]}
        if len(labs) != 1:
            raise ValueError(f"subject {subject!r} maps to multiple labs")
        subject_labs[_scalar(subject)] = labs.pop()
    labs = tuple(dict.fromkeys(subject_labs.values()))
    losses: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    payloads: dict[str, Any] = {}
    for name in METHODS:
        result = report.result_for(name)
        by_subject = dict(zip(result.aggregation_units, result.unit_log_losses, strict=True))
        lab_losses = np.asarray(
            [
                np.mean(
                    [
                        by_subject[subject]
                        for subject, subject_lab in subject_labs.items()
                        if subject_lab == lab
                    ]
                )
                for lab in labs
            ],
            dtype=np.float64,
        )
        losses[name] = lab_losses
        payloads[name] = {
            "mean_log_loss": float(np.mean(lab_losses)),
            "log_loss_by_lab": {
                str(lab): float(loss) for lab, loss in zip(labs, lab_losses, strict=True)
            },
        }
    difference = losses[METHODS[0]] - losses[METHODS[1]]
    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(0, len(labs), size=(bootstrap_resamples, len(labs)))
    bootstrap = np.mean(difference[draws], axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "unit": "lab after equal weighting of subjects within lab",
        "labs": [str(lab) for lab in labs],
        "methods": payloads,
        "static_minus_drifting": {
            "direction": "positive favors hierarchical_smooth_drift",
            "estimate": float(np.mean(difference)),
            "interval_95": [float(lower), float(upper)],
            "bootstrap_probability_positive": float(np.mean(bootstrap > 0)),
        },
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "scope": "empirical held-out labs; no population-of-laboratories claim",
        },
    }


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    result = run(
        args.cache,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
