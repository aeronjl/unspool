"""Select hierarchical model structure inside untouched replicated-IBL outer folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from behavio import BernoulliHistoryGLM, Study, cohort_forward_session_splits, nested_select_model
from behavio.compare import NestedProspectiveSelectionReport
from behavio.compose import hierarchical, smooth
from behavio.evaluate import leave_one_lab_out_session_forecast_splits
from benchmarks.ibl2021_prospective.benchmark import (
    TRAIN_SESSION_COUNT,
    build_panel,
)
from benchmarks.ibl2021_replicated.benchmark import DEFAULT_CACHE, load_study
from benchmarks.ibl2021_replicated.manifest import EXPECTED_MANIFEST_SHA256
from benchmarks.provenance import render

CANDIDATES = (
    "static",
    "drift_smoothness_1",
    "drift_smoothness_3",
    "drift_smoothness_9",
)
KNOTS = (0.0, 2.0, 5.0)
BOOTSTRAP_RESAMPLES = 5_000
INNER_BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20_260_731
REFERENCE_PATH = Path(__file__).parents[1] / "ibl2021_prospective" / "result.json"
EXPECTED_REFERENCE_SHA256 = "3d2fc25771369d60955752aef4fcd731b4936985c399f5d08a6cc2f9008daf7c"


def fit_nested_procedures(
    panel: Study,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    inner_bootstrap_resamples: int = INNER_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[NestedProspectiveSelectionReport, NestedProspectiveSelectionReport]:
    """Fit nested selection for represented-animal and unseen-lab outer targets."""

    candidates = _candidates()
    within_outer = cohort_forward_session_splits(
        panel,
        min_train_sessions=TRAIN_SESSION_COUNT,
        horizon=1,
    )
    if len(within_outer) != 1:
        raise ValueError("the six-position panel must produce one within-animal outer fold")

    def within_inner(training: Study):
        return cohort_forward_session_splits(training, min_train_sessions=3, horizon=1)

    within = nested_select_model(
        candidates,
        panel,
        within_outer,
        within_inner,
        bootstrap_resamples=bootstrap_resamples,
        inner_bootstrap_resamples=inner_bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )

    lab_outer = leave_one_lab_out_session_forecast_splits(
        panel,
        train_session_count=TRAIN_SESSION_COUNT,
        horizon=1,
    )

    def lab_inner(training: Study):
        return leave_one_lab_out_session_forecast_splits(
            training,
            train_session_count=4,
            horizon=1,
        )

    transfer = nested_select_model(
        candidates,
        panel,
        lab_outer,
        lab_inner,
        bootstrap_resamples=bootstrap_resamples,
        inner_bootstrap_resamples=inner_bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 10_000,
    )
    return within, transfer


def run(
    cache_directory: Path = DEFAULT_CACHE,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    inner_bootstrap_resamples: int = INNER_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run both nested procedures and compare their untouched scores with fixed models."""

    panel = build_panel(load_study(cache_directory))
    within, transfer = fit_nested_procedures(
        panel,
        bootstrap_resamples=bootstrap_resamples,
        inner_bootstrap_resamples=inner_bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    reference = _load_reference()
    within_reference = _reference_comparisons(
        within,
        reference["within_subject_future_session"],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 20_000,
    )
    transfer_reference = _reference_comparisons(
        transfer,
        reference["held_out_lab_future_session"],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 30_000,
    )
    transfer_lab_balanced = _lab_balanced_summary(
        transfer,
        panel,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 40_000,
    )
    contracts = {
        "reference_result_is_pinned": _sha256(REFERENCE_PATH) == EXPECTED_REFERENCE_SHA256,
        "within_outer_position_is_untouched": _within_outer_is_untouched(within),
        "lab_outer_groups_are_untouched": _lab_outer_groups_are_untouched(transfer),
        "inner_selection_uses_only_earlier_positions": _inner_positions_are_prospective(
            within, transfer
        ),
        "all_selected_outer_fits_are_usable": all(
            report.audit_status.value != "fail" for report in (within, transfer)
        ),
        "all_outer_subjects_scored_once": within.n_scored_observations
        == transfer.n_scored_observations
        == int(reference["panel"]["trials_by_position"]["5"]),
    }
    if not all(contracts.values()):
        failed = sorted(name for name, passed in contracts.items() if not passed)
        raise AssertionError(f"nested replicated IBL contract failed: {failed}")
    return {
        "benchmark": "nested replicated IBL hierarchical model and smoothness selection",
        "source": {
            "release": "2021_Q1_IBL_et_al_Behaviour",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "fixed_comparison_sha256": EXPECTED_REFERENCE_SHA256,
            "doi": "10.7554/eLife.63711",
        },
        "selection_contract": {
            "candidate_order": list(CANDIDATES),
            "tie_break": "declared candidate order",
            "candidate_grid": {
                "static": {"subject_scale": 0.4},
                "drift": {
                    "knots": list(KNOTS),
                    "subject_scale": 0.4,
                    "smoothness_and_subject_smoothness": [1.0, 3.0, 9.0],
                },
                "common": {
                    "l2": 0.02,
                    "predictors": ["signed contrast", "one-session-reset choice lag"],
                },
            },
            "within_inner": "positions 3 and 4 forecast from earlier prefixes",
            "within_outer": "position 5 for all represented animals",
            "lab_inner": (
                "position 4 in each inner held-out lab from positions 0 through 3 "
                "in the remaining outer-training labs"
            ),
            "lab_outer": (
                "position 5 in the untouched outer lab from positions 0 through 4 in all other labs"
            ),
            "outer_test_outcomes_available_during_selection": False,
            "prediction_mode": "filtered one-step-ahead within test sessions",
            "outer_bootstrap_resamples": bootstrap_resamples,
            "inner_bootstrap_resamples": inner_bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
        },
        "panel": reference["panel"],
        "within_subject_future_session": {
            **_compact_report(within),
            "comparison_with_fixed_candidates": within_reference,
        },
        "held_out_lab_future_session": {
            **_compact_report(transfer),
            "comparison_with_fixed_candidates": transfer_reference,
            "lab_balanced_selected_procedure": transfer_lab_balanced,
        },
        "contract": contracts,
        "contract_passed": True,
    }


def _candidates() -> Mapping[str, Any]:
    common = {"predictors": ("stimulus",), "choice_lags": 1, "l2": 0.02}
    candidates: dict[str, Any] = {
        "static": hierarchical(BernoulliHistoryGLM(**common), over="subject", scale=0.4)
    }
    for smoothness in (1.0, 3.0, 9.0):
        candidates[f"drift_smoothness_{int(smoothness)}"] = hierarchical(
            smooth(
                BernoulliHistoryGLM(**common),
                over="session_order",
                knots=KNOTS,
                smoothness=smoothness,
            ),
            over="subject",
            scale=0.4,
        )
    return candidates


def _compact_report(report: NestedProspectiveSelectionReport) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold in report.folds:
        outer = fold.outer_split
        inner = fold.inner_report
        fold_payload: dict[str, Any] = {
            "selected_candidate": fold.selected_model,
            "selected_family": _family(fold.selected_model),
            "selected_outer_fit_audit": fold.outer_evaluation.fit.audit().to_dict(),
            "outer_mean_trial_log_loss": fold.outer_evaluation.mean_log_loss,
            "outer_test_trials": len(outer.test_indices),
            "outer_test_subjects": _outer_test_subject_count(outer),
            "inner_fold_count": len(inner.splits),
            "inner_winner": inner.winner,
            "inner_candidates": {
                name: {
                    "subject_balanced_log_loss": inner.result_for(name).unit_balanced_log_loss,
                    "audit_status": inner.result_for(name).audit_status.value,
                }
                for name in CANDIDATES
            },
            "inner_fold_targets": _inner_fold_targets(inner),
        }
        if hasattr(outer, "held_out_group"):
            fold_payload["outer_held_out_lab"] = str(outer.held_out_group)
        else:
            fold_payload["outer_test_position"] = 5
        folds.append(fold_payload)
    return {
        "candidate_order": list(report.candidate_names),
        "selection_counts": dict(report.selection_counts),
        "family_selection_counts": dict(
            Counter(_family(fold.selected_model) for fold in report.folds)
        ),
        "outer_folds": len(report.folds),
        "outer_scored_trials": report.n_scored_observations,
        "outer_audit_status": report.audit_status.value,
        "subject_balanced_log_loss": report.unit_balanced_log_loss,
        "subject_balanced_brier_score": report.unit_balanced_brier_score,
        "pooled_trial_log_loss": report.pooled_log_loss,
        "subject_bootstrap_log_loss_95_interval": [
            report.unit_balanced_log_loss_interval.lower,
            report.unit_balanced_log_loss_interval.upper,
        ],
        "subject_scores": {
            str(subject): {
                "log_loss": float(log_loss),
                "brier_score": float(brier),
            }
            for subject, log_loss, brier in zip(
                report.aggregation_units,
                report.unit_log_losses,
                report.unit_brier_scores,
                strict=True,
            )
        },
        "folds": folds,
    }


def _reference_comparisons(
    report: NestedProspectiveSelectionReport,
    reference: dict[str, Any],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    reference_names = {
        "fixed_static": "static_partial_pooling",
        "fixed_drift_smoothness_3": "hierarchical_smooth_drift",
    }
    selected = dict(zip(report.aggregation_units, report.unit_log_losses, strict=True))
    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(
        0,
        len(report.aggregation_units),
        size=(bootstrap_resamples, len(report.aggregation_units)),
    )
    comparisons: dict[str, Any] = {}
    for label, source_name in reference_names.items():
        source_scores = {
            row["unit"]: float(row["log_loss"])
            for row in reference["models"][source_name]["unit_scores"]
        }
        if set(source_scores) != set(selected):
            raise ValueError("fixed and nested outer score subjects do not match")
        difference = np.asarray(
            [selected[subject] - source_scores[subject] for subject in report.aggregation_units],
            dtype=np.float64,
        )
        bootstrap = np.mean(difference[draws], axis=1)
        lower, upper = np.quantile(bootstrap, (0.025, 0.975))
        comparisons[label] = {
            "direction": "selected procedure minus fixed candidate; negative favors selection",
            "estimate": float(np.mean(difference)),
            "interval_95": [float(lower), float(upper)],
            "bootstrap_probability_below_zero": float(np.mean(bootstrap < 0)),
        }
    return comparisons


def _lab_balanced_summary(
    report: NestedProspectiveSelectionReport,
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
    by_subject = dict(zip(report.aggregation_units, report.unit_log_losses, strict=True))
    labs = tuple(dict.fromkeys(subject_labs.values()))
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
    generator = np.random.default_rng(bootstrap_seed)
    draws = generator.integers(0, len(labs), size=(bootstrap_resamples, len(labs)))
    bootstrap = np.mean(lab_losses[draws], axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean_log_loss": float(np.mean(lab_losses)),
        "interval_95": [float(lower), float(upper)],
        "log_loss_by_lab": {
            str(lab): float(loss) for lab, loss in zip(labs, lab_losses, strict=True)
        },
        "bootstrap": {
            "unit": "empirical held-out lab after equal subject weighting",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "population_of_labs_inference": False,
        },
    }


def _within_outer_is_untouched(report: NestedProspectiveSelectionReport) -> bool:
    if len(report.folds) != 1:
        return False
    outer = report.folds[0].outer_split
    return all(
        max(outer.train_session_orders[subject]) < min(outer.test_session_orders[subject]) == 5
        for subject in outer.subjects
    )


def _lab_outer_groups_are_untouched(report: NestedProspectiveSelectionReport) -> bool:
    return len(report.folds) == 9 and all(
        set(fold.outer_split.train_subjects).isdisjoint(fold.outer_split.test_subjects)
        and set(fold.outer_split.train_groups).isdisjoint(fold.outer_split.test_groups)
        and fold.outer_split.test_session_orders == (5,)
        for fold in report.folds
    )


def _inner_positions_are_prospective(
    within: NestedProspectiveSelectionReport,
    transfer: NestedProspectiveSelectionReport,
) -> bool:
    within_targets = _inner_fold_targets(within.folds[0].inner_report)
    if within_targets != [3, 4]:
        return False
    return all(
        all(split.train_session_orders == (0, 1, 2, 3) for split in fold.inner_report.splits)
        and all(split.test_session_orders == (4,) for split in fold.inner_report.splits)
        and fold.outer_split.held_out_group
        not in {split.held_out_group for split in fold.inner_report.splits}
        for fold in transfer.folds
    )


def _inner_fold_targets(report: Any) -> list[Any]:
    targets: list[Any] = []
    for split in report.splits:
        if hasattr(split, "held_out_group"):
            targets.append(str(split.held_out_group))
        else:
            orders = {
                order for subject in split.subjects for order in split.test_session_orders[subject]
            }
            targets.append(next(iter(orders)) if len(orders) == 1 else sorted(orders))
    return targets


def _outer_test_subject_count(split: Any) -> int:
    if hasattr(split, "test_subjects"):
        return len(split.test_subjects)
    return len(split.subjects)


def _family(candidate: str) -> str:
    return "static" if candidate == "static" else "drift"


def _load_reference() -> dict[str, Any]:
    if _sha256(REFERENCE_PATH) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("fixed prospective comparison checksum mismatch")
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--inner-bootstrap-resamples",
        type=int,
        default=INNER_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    result = run(
        args.cache,
        bootstrap_resamples=args.bootstrap_resamples,
        inner_bootstrap_resamples=args.inner_bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered = render(result)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
