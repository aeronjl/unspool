"""Run repeated strong- and weak-signal recovery under one matched design."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.recovery_grid.benchmark import N_SESSIONS, build_design, experiment
from unspool import ModelRecoveryReport, ModelRecoveryScenario, run_model_recovery

DEFAULT_REPEATS = 10
ROOT_SEED = 7701


def recovery_scenarios() -> tuple[ModelRecoveryScenario, ...]:
    """Return stronger references and boundary-near regimes for all four families."""

    strong_scenarios, candidates = experiment()
    static = candidates["static"]
    smooth = candidates["smooth"]
    hidden_state = candidates["hmm"]
    q_learning = candidates["q-learning"]
    strong = tuple(
        ModelRecoveryScenario(
            name=f"strong-{scenario.name}",
            truth_label=scenario.truth_label,
            generator=scenario.generator,
            parameters=scenario.parameters,
        )
        for scenario in strong_scenarios
    )
    weak = (
        ModelRecoveryScenario(
            name="weak-low-signal-stationary",
            truth_label="static",
            generator=static,
            parameters={"intercept": -0.05, "stimulus": 0.35, "choice_lag_1": 0.1},
        ),
        ModelRecoveryScenario(
            name="weak-subtle-drift",
            truth_label="smooth",
            generator=smooth,
            parameters=smooth.parameters_from_paths(
                {
                    "intercept": np.linspace(-0.2, 0.2, N_SESSIONS),
                    "stimulus": np.linspace(0.75, 1.05, N_SESSIONS),
                    "choice_lag_1": np.linspace(0.3, 0.1, N_SESSIONS),
                }
            ),
        ),
        ModelRecoveryScenario(
            name="weak-overlapping-states",
            truth_label="hmm",
            generator=hidden_state,
            parameters=hidden_state.parameters_from_components(
                initial_probabilities=[0.5, 0.5],
                transition_matrix=[[0.96, 0.04], [0.04, 0.96]],
                emissions={
                    "intercept": [-0.35, 0.35],
                    "stimulus": [0.85, 1.15],
                    "choice_lag_1": [0.3, 0.1],
                },
            ),
        ),
        ModelRecoveryScenario(
            name="weak-slow-learning",
            truth_label="q-learning",
            generator=q_learning,
            parameters=q_learning.parameters_from_components(
                learning_rate=0.08,
                inverse_temperature=2.0,
                choice_bias=0.0,
                perseveration=0.1,
            ),
        ),
    )
    return strong + weak


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for a binomial recovery rate."""

    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z * np.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)) / denominator
    )
    return float(centre - half_width), float(centre + half_width)


def _signal_summary(report: ModelRecoveryReport, level: str) -> dict[str, Any]:
    indices = [
        index
        for index, scenario_name in enumerate(report.scenario_names)
        if scenario_name.startswith(f"{level}-")
    ]
    correct = sum(report.selected_labels[index] == report.truth_labels[index] for index in indices)
    resolved = sum(report.selected_labels[index] is not None for index in indices)
    audit_statuses = [status.value for index in indices for status in report.audit_statuses[index]]
    return {
        "n_runs": len(indices),
        "correct": correct,
        "accuracy": correct / len(indices),
        "accuracy_wilson_95": list(_wilson_interval(correct, len(indices))),
        "resolution_rate": resolved / len(indices),
        "audit_status_counts": dict(sorted(Counter(audit_statuses).items())),
    }


def _scenario_score_summary(report: ModelRecoveryReport) -> list[dict[str, Any]]:
    matrix = report.scenario_confusion_matrix()
    rows: list[dict[str, Any]] = []
    for scenario_name, truth_label in zip(matrix.scenario_names, matrix.truth_labels, strict=True):
        indices = [
            index
            for index, observed_name in enumerate(report.scenario_names)
            if observed_name == scenario_name
        ]
        scores = report.mean_log_probabilities[indices]
        rows.append(
            {
                "scenario": scenario_name,
                "truth_label": truth_label,
                "mean_log_probability": dict(
                    zip(report.candidate_labels, np.mean(scores, axis=0).tolist(), strict=True)
                ),
                "standard_deviation": dict(
                    zip(report.candidate_labels, np.std(scores, axis=0).tolist(), strict=True)
                ),
            }
        )
    return rows


def run(*, repeats: int = DEFAULT_REPEATS, check: bool = True) -> dict[str, Any]:
    """Run repeated recovery and return a compact reproducible result."""

    scenarios = recovery_scenarios()
    _, candidates = experiment()
    report = run_model_recovery(
        build_design(trials_per_session=60),
        scenarios,
        candidates,
        repeats=repeats,
        seed=ROOT_SEED,
        min_train_sessions=3,
        tie_tolerance=0.002,
    )
    scenario_matrix = report.scenario_confusion_matrix()
    family_matrix = report.confusion_matrix()
    signal_levels = {level: _signal_summary(report, level) for level in ("strong", "weak")}
    contract_passed = bool(
        signal_levels["strong"]["accuracy"] > signal_levels["weak"]["accuracy"]
        and signal_levels["weak"]["accuracy"] < 1.0
        and np.all(scenario_matrix.counts.sum(axis=1) == repeats)
    )
    if check and not contract_passed:
        raise AssertionError(f"weak-signal recovery contract failed: {signal_levels}")

    first_run_for_scenario = {
        name: report.scenario_names.index(name) for name in scenario_matrix.scenario_names
    }
    return {
        "benchmark": "repeated recovery near model-family limiting cases",
        "root_seed": report.root_seed,
        "repeats": repeats,
        "n_trials": report.n_trials,
        "n_subjects": report.n_subjects,
        "n_folds_per_run": report.n_folds.tolist(),
        "candidate_labels": list(report.candidate_labels),
        "candidate_signatures": list(report.candidate_signatures),
        "contract_passed": contract_passed,
        "signal_levels": signal_levels,
        "scenario_matrix": {
            "scenario_names": list(scenario_matrix.scenario_names),
            "truth_labels": list(scenario_matrix.truth_labels),
            "selected_labels": list(scenario_matrix.selected_labels),
            "counts": scenario_matrix.counts.tolist(),
            "rates": scenario_matrix.rates.tolist(),
        },
        "family_confusion": {
            "truth_labels": list(family_matrix.truth_labels),
            "selected_labels": list(family_matrix.selected_labels),
            "counts": family_matrix.counts.tolist(),
            "rates": family_matrix.rates.tolist(),
        },
        "scenario_scores": _scenario_score_summary(report),
        "scenarios": [
            {
                "name": scenario_name,
                "generator_signature": report.generator_signatures[index],
                "parameters": dict(report.generator_parameters[index]),
            }
            for scenario_name, index in first_run_for_scenario.items()
        ],
        "run_seeds": report.seeds.tolist(),
        "audit_warning_rate": report.audit_warning_rate,
        "audit_failure_rate": report.audit_failure_rate,
        "audit_issue_codes": sorted(
            {
                code
                for run_codes in report.audit_issue_codes
                for candidate_codes in run_codes
                for code in candidate_codes
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--no-check", action="store_true", help="report without enforcing contract")
    parser.add_argument("--output", type=Path, help="also write the JSON result to this path")
    args = parser.parse_args()
    result = run(repeats=args.repeats, check=not args.no_check)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
