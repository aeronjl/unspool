"""Fit prospective DDM and GLM-HMM examples to one pinned public IBL subject."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.ibl2021.refresh_manifest import PUBLIC_PASSWORD
from benchmarks.ibl2021_replicated.benchmark import DEFAULT_CACHE
from benchmarks.ibl2021_replicated.manifest import (
    EXPECTED_MANIFEST_SHA256,
    load_manifest,
)
from benchmarks.provenance import render
from unspool import (
    BernoulliGLMHMM,
    BernoulliHistoryGLM,
    IBLONETrialSource,
    ResponseTimeSpec,
    ResponseTimeUnit,
    Study,
    UniformResponseTimeContaminant,
    WienerDriftDiffusion,
    read_ibl_one_sessions,
)

TRIALS_PER_SESSION = 150
RT_MIN_SECONDS = 0.05
RT_MAX_SECONDS = 3.0
DDM_TRAIN_POSITIONS = (3, 4)
OUTER_TEST_POSITION = 5
HMM_SELECTION_POSITION = 4
HMM_STATE_CANDIDATES = (2, 3, 4)
HMM_L2 = 0.02
SEED = 20_260_727


def selected_manifest_rows() -> tuple[str, tuple[dict[str, Any], ...]]:
    """Select the lexicographically first eligible manifest subject without reading outcomes."""

    manifest = load_manifest()
    subject = min(str(row["subject"]) for row in manifest["sessions"])
    rows = tuple(
        sorted(
            (row for row in manifest["sessions"] if str(row["subject"]) == subject),
            key=lambda row: int(row["window_position"]),
        )
    )
    if len(rows) != 6 or [int(row["window_position"]) for row in rows] != list(range(6)):
        raise ValueError(f"selected subject {subject!r} does not have all six endpoint windows")
    return subject, rows


def load_source_study(cache_directory: Path = DEFAULT_CACHE) -> Study:
    """Load the selected subject's exact trial tables, including movement timing columns."""

    try:
        from one.api import ONE
    except ImportError as error:
        raise RuntimeError("the public IBL worked study requires `unspool[ibl]`") from error

    manifest = load_manifest()
    _subject, rows = selected_manifest_rows()
    sources = tuple(
        IBLONETrialSource(
            session_id=str(row["session"]),
            dataset_id=str(row["dataset_id"]),
            dataset_path=str(row["dataset_path"]),
            file_size=int(row["file_size"]),
            md5=str(row["md5"]),
            release_tag=str(manifest["release_tag"]),
            session_order=int(row["session_order"]),
            subject=str(row["subject"]),
            session=str(row["session"]),
            lab=str(row["lab"]),
            columns=(
                "contrastLeft",
                "contrastRight",
                "choice",
                "goCue_times",
                "firstMovement_times",
            ),
            column_map={"choice": "source_choice"},
            source_columns={
                "phase": str(row["phase"]),
                "window_position": int(row["window_position"]),
                "task_protocol": str(row["task_protocol"]),
            },
            alyx_url=str(manifest["public_alyx_url"]),
        )
        for row in rows
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    one = ONE(
        base_url=str(manifest["public_alyx_url"]),
        password=PUBLIC_PASSWORD,
        silent=True,
        cache_dir=cache_directory,
    )
    one.load_cache(tag=str(manifest["release_tag"]))
    return read_ibl_one_sessions(sources, client=one)


def build_panels(
    source: Study,
    *,
    trials_per_session: int = TRIALS_PER_SESSION,
) -> tuple[Study, Study, dict[str, Any]]:
    """Build choice-only and choice/RT panels after an outcome-blind source-row cap."""

    required = {
        "source_choice",
        "contrastLeft",
        "contrastRight",
        "goCue_times",
        "firstMovement_times",
        "window_position",
        "phase",
        "lab",
        "source_ibl_dataset_id",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"source IBL study is missing required columns: {sorted(missing)}")
    if (
        isinstance(trials_per_session, bool)
        or not isinstance(trials_per_session, int)
        or trials_per_session < 1
    ):
        raise ValueError("trials_per_session must be a positive integer")

    counts: Counter[tuple[Any, Any]] = Counter()
    capped: list[int] = []
    for raw_index in source.chronological_indices():
        index = int(raw_index)
        key = (_scalar(source["subject"][index]), _scalar(source["session"][index]))
        if counts[key] < trials_per_session:
            capped.append(index)
        counts[key] += 1
    capped_positions = np.asarray(capped, dtype=np.intp)

    source_choice = np.asarray(source["source_choice"][capped_positions], dtype=np.float64)
    left = np.asarray(source["contrastLeft"][capped_positions], dtype=np.float64)
    right = np.asarray(source["contrastRight"][capped_positions], dtype=np.float64)
    stimulus = np.nan_to_num(right, nan=0.0) - np.nan_to_num(left, nan=0.0)
    valid_choice = np.isfinite(source_choice) & (source_choice != 0) & np.isfinite(stimulus)
    choice_positions = capped_positions[valid_choice]
    choice_panel = _panel_from_positions(
        source,
        choice_positions,
        stimulus=stimulus[valid_choice],
        source_choice=source_choice[valid_choice],
    )

    go_cue = np.asarray(source["goCue_times"][choice_positions], dtype=np.float64)
    movement = np.asarray(source["firstMovement_times"][choice_positions], dtype=np.float64)
    response_time = movement - go_cue
    valid_rt = (
        np.isfinite(response_time)
        & (response_time >= RT_MIN_SECONDS)
        & (response_time <= RT_MAX_SECONDS)
    )
    rt_positions = np.flatnonzero(valid_rt)
    rt_panel = choice_panel.take(rt_positions)
    rt_columns = {name: rt_panel[name] for name in rt_panel.columns}
    rt_columns["response_time"] = response_time[valid_rt]
    rt_panel = Study(rt_columns)

    source_rows_by_position = _counts_by_position(
        source,
        capped_positions,
        column="window_position",
    )
    choice_rows_by_position = _counts_by_position(choice_panel, np.arange(len(choice_panel)))
    rt_rows_by_position = _counts_by_position(rt_panel, np.arange(len(rt_panel)))
    eligibility = {
        "source_rows_after_cap": source_rows_by_position,
        "choice_rows": choice_rows_by_position,
        "choice_rt_rows": rt_rows_by_position,
        "choice_rule": "drop source no-go choice=0 after the outcome-blind row cap",
        "response_time_definition": "firstMovement_times - goCue_times",
        "response_time_unit": "seconds",
        "response_time_validity_seconds": [RT_MIN_SECONDS, RT_MAX_SECONDS],
        "timing_rule": "retain finite movement-onset latencies inside the fixed validity window",
    }
    return choice_panel, rt_panel, eligibility


def analyze_panels(choice_panel: Study, rt_panel: Study) -> dict[str, Any]:
    """Fit the predeclared DDM and nested prospective GLM-HMM analyses."""

    ddm = _analyze_ddm(rt_panel)
    hmm = _analyze_glm_hmm(choice_panel)
    contracts = {
        "outer_test_position_is_sixth_window": OUTER_TEST_POSITION == 5,
        "ddm_training_precedes_test": max(DDM_TRAIN_POSITIONS) < OUTER_TEST_POSITION,
        "hmm_selection_precedes_outer_test": HMM_SELECTION_POSITION < OUTER_TEST_POSITION,
        "ddm_scores_choice_and_response_time": ddm["scored_columns"] == ["choice", "response_time"],
        "glm_hmm_scores_choice_only": hmm["scored_columns"] == ["choice"],
        "all_outer_fits_audit_eligible": all(
            payload["fit_audit"]["status"] != "fail"
            for payload in (
                ddm["naive"],
                ddm["robust"],
                hmm["static_glm"],
                hmm["selected_glm_hmm"],
            )
        ),
    }
    if not all(contracts.values()):
        failed = sorted(name for name, passed in contracts.items() if not passed)
        raise AssertionError(f"IBL decision-model benchmark contract failed: {failed}")
    return {
        "ddm": ddm,
        "glm_hmm": hmm,
        "contract": contracts,
        "contract_passed": True,
    }


def run(cache_directory: Path = DEFAULT_CACHE) -> dict[str, Any]:
    """Load the pinned public data and run both prospective worked studies."""

    subject, rows = selected_manifest_rows()
    choice_panel, rt_panel, eligibility = build_panels(load_source_study(cache_directory))
    return {
        "benchmark": "IBL 2021 prospective DDM and GLM-HMM worked studies",
        "source": {
            "doi": "10.7554/eLife.63711",
            "release": "2021_Q1_IBL_et_al_Behaviour",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "subject_selection": "lexicographically first eligible subject in the frozen manifest",
            "subject": subject,
            "lab": str(rows[0]["lab"]),
            "sessions": len(rows),
            "dataset_ids": [str(row["dataset_id"]) for row in rows],
        },
        "analysis_contract": {
            "trials_per_session": TRIALS_PER_SESSION,
            "clock": "outcome-blind endpoint window_position; not uniform elapsed time",
            "outer_test": "position 5 is untouched until final evaluation",
            "ddm_training": "positions 3 and 4; prespecified late-training stability window",
            "glm_hmm_selection": (
                "choose 2, 3, or 4 states on position 4 after fitting positions 0-3"
            ),
            "glm_hmm_refit": (
                "positions 0-4, followed by one-step-ahead filtered scoring on position 5"
            ),
            "random_seed": SEED,
        },
        "eligibility": eligibility,
        **analyze_panels(choice_panel, rt_panel),
    }


def _analyze_ddm(panel: Study) -> dict[str, Any]:
    train = panel.take(np.flatnonzero(np.isin(panel["session_order"], DDM_TRAIN_POSITIONS)))
    test = panel.take(np.flatnonzero(panel["session_order"] == OUTER_TEST_POSITION))
    naive = _ddm(robust=False)
    robust = _ddm(robust=True)
    naive_fit = naive.fit(train)
    robust_fit = robust.fit(train)
    naive_scores = naive.pointwise_log_prob(test, naive_fit)
    robust_scores = robust.pointwise_log_prob(test, robust_fit)
    prediction = robust.predict(test, robust_fit).probability
    responsibilities = robust.contaminant_responsibility(test, robust_fit)
    predictive = robust.simulate(test, robust_fit.parameters, seed=SEED)
    components = robust.parameter_components(robust_fit)

    return {
        "scored_columns": list(robust.scored_columns),
        "train_positions": list(DDM_TRAIN_POSITIONS),
        "test_position": OUTER_TEST_POSITION,
        "n_train": len(train),
        "n_test": len(test),
        "naive": {
            "mean_test_joint_log_density": float(np.mean(naive_scores)),
            "fit_audit": naive_fit.audit().to_dict(),
        },
        "robust": {
            "mean_test_joint_log_density": float(np.mean(robust_scores)),
            "improvement_over_naive": float(np.mean(robust_scores - naive_scores)),
            "parameters": {
                name: float(value)
                for name, value in zip(
                    robust_fit.parameter_names,
                    robust_fit.estimates,
                    strict=True,
                )
            },
            "expected_test_contaminants": float(np.sum(responsibilities)),
            "fit_audit": robust_fit.audit().to_dict(),
        },
        "heldout": {
            "trial": np.asarray(test["trial"], dtype=int).tolist(),
            "stimulus": np.asarray(test["stimulus"], dtype=float).tolist(),
            "choice": np.asarray(test["choice"], dtype=int).tolist(),
            "choice_probability": prediction.tolist(),
            "response_time_seconds": np.asarray(test["response_time"], dtype=float).tolist(),
            "predictive_response_time_seconds": np.asarray(
                predictive["response_time"], dtype=float
            ).tolist(),
            "contaminant_responsibility": responsibilities.tolist(),
            "conditional_accuracy": _conditional_accuracy(test, prediction),
        },
        "interpretation": {
            "diffusion_scale": 1.0,
            "response_time_unit": "seconds",
            "nondecision_time_seconds": components.nondecision_time,
            "contaminant_support_seconds": [RT_MIN_SECONDS, RT_MAX_SECONDS],
        },
    }


def _analyze_glm_hmm(panel: Study) -> dict[str, Any]:
    inner_train = panel.take(np.flatnonzero(panel["session_order"] < HMM_SELECTION_POSITION))
    selection = panel.take(np.flatnonzero(panel["session_order"] == HMM_SELECTION_POSITION))
    outer_train = panel.take(np.flatnonzero(panel["session_order"] < OUTER_TEST_POSITION))
    outer_test = panel.take(np.flatnonzero(panel["session_order"] == OUTER_TEST_POSITION))

    selection_rows: list[dict[str, Any]] = []
    for n_states in HMM_STATE_CANDIDATES:
        model = _hmm(n_states)
        fit = model.fit(inner_train)
        selection_rows.append(
            {
                "n_states": n_states,
                "mean_selection_log_loss": float(
                    -np.mean(model.pointwise_log_prob(selection, fit))
                ),
                "fit_audit": fit.audit().to_dict(),
            }
        )
    eligible = [row for row in selection_rows if row["fit_audit"]["status"] != "fail"]
    if not eligible:
        raise AssertionError("all GLM-HMM state-count candidates failed the training-only audit")
    selected_states = int(min(eligible, key=lambda row: row["mean_selection_log_loss"])["n_states"])

    static = BernoulliHistoryGLM(covariates=("stimulus",), choice_lags=1, l2=HMM_L2)
    hmm = _hmm(selected_states)
    static_fit = static.fit(outer_train)
    hmm_fit = hmm.fit(outer_train)
    static_scores = static.pointwise_log_prob(outer_test, static_fit)
    hmm_scores = hmm.pointwise_log_prob(outer_test, hmm_fit)
    probabilities = hmm.state_probabilities(outer_test, hmm_fit)
    components = hmm.parameter_components(hmm_fit)
    map_state = np.argmax(probabilities.filtered, axis=1)

    return {
        "literature_precedent": {
            "citation": "Ashwood et al. (2022), Nature Neuroscience 25:201-212",
            "doi": "10.1038/s41593-021-01007-z",
            "relationship": "structural analogue, not a reproduction",
        },
        "scored_columns": list(hmm.scored_columns),
        "selection": {
            "inner_train_positions": list(range(HMM_SELECTION_POSITION)),
            "selection_position": HMM_SELECTION_POSITION,
            "candidates": selection_rows,
            "selected_states": selected_states,
        },
        "outer_test_position": OUTER_TEST_POSITION,
        "n_outer_train": len(outer_train),
        "n_outer_test": len(outer_test),
        "static_glm": {
            "mean_test_log_loss": float(-np.mean(static_scores)),
            "fit_audit": static_fit.audit().to_dict(),
        },
        "selected_glm_hmm": {
            "mean_test_log_loss": float(-np.mean(hmm_scores)),
            "improvement_over_static": float(np.mean(hmm_scores - static_scores)),
            "coefficient_names": list(components.coefficient_names),
            "emission_coefficients": components.emission_coefficients.tolist(),
            "initial_probabilities": components.initial_probabilities.tolist(),
            "transition_matrix": components.transition_matrix.tolist(),
            "training_state_occupancy": hmm_fit.state_occupancy.tolist(),
            "heldout_filtered_state_occupancy": np.mean(probabilities.filtered, axis=0).tolist(),
            "heldout_map_switches": int(np.count_nonzero(np.diff(map_state))),
            "fit_audit": hmm_fit.audit().to_dict(),
        },
        "heldout": {
            "trial": np.asarray(outer_test["trial"], dtype=int).tolist(),
            "choice": np.asarray(outer_test["choice"], dtype=int).tolist(),
            "stimulus": np.asarray(outer_test["stimulus"], dtype=float).tolist(),
            "predictive_state_probability": probabilities.predictive.tolist(),
            "filtered_state_probability": probabilities.filtered.tolist(),
        },
        "interpretation_boundary": (
            "state labels are ordered by fitted sensory weight; filtered state probabilities "
            "are model-dependent and condition on observed choices within the held-out session"
        ),
    }


def _ddm(*, robust: bool) -> WienerDriftDiffusion:
    contaminant = None
    if robust:
        contaminant = UniformResponseTimeContaminant(
            time_bounds=(RT_MIN_SECONDS, RT_MAX_SECONDS),
            probability_bounds=(0.0, 0.3),
        )
    return WienerDriftDiffusion(
        covariates=("stimulus",),
        response_time=ResponseTimeSpec(unit=ResponseTimeUnit.SECONDS),
        n_restarts=4,
        max_iterations=500,
        nondecision_time_bounds=(0.0, RT_MIN_SECONDS - 0.001),
        contaminant=contaminant,
        simulation_time_step=0.002,
        simulation_max_time=60.0,
    )


def _hmm(n_states: int) -> BernoulliGLMHMM:
    return BernoulliGLMHMM(
        covariates=("stimulus",),
        choice_lags=1,
        l2=HMM_L2,
        n_states=n_states,
        n_restarts=5,
        random_seed=SEED + n_states,
        label_by="stimulus",
        max_iterations=500,
    )


def _panel_from_positions(
    source: Study,
    positions: np.ndarray[Any, np.dtype[np.intp]],
    *,
    stimulus: np.ndarray[Any, np.dtype[np.float64]],
    source_choice: np.ndarray[Any, np.dtype[np.float64]],
) -> Study:
    return Study(
        {
            "subject": source["subject"][positions],
            "session": source["session"][positions],
            "trial": source["trial"][positions],
            "session_order": np.asarray(source["window_position"][positions], dtype=np.int64),
            "source_session_order": source["session_order"][positions],
            "lab": source["lab"][positions],
            "phase": source["phase"][positions],
            "choice": (source_choice < 0).astype(np.int8),
            "stimulus": stimulus,
            "source_choice": source_choice.astype(np.int8),
            "source_ibl_dataset_id": source["source_ibl_dataset_id"][positions],
        }
    )


def _conditional_accuracy(study: Study, probability: np.ndarray[Any, Any]) -> list[dict[str, Any]]:
    stimulus = np.asarray(study["stimulus"], dtype=np.float64)
    choice = np.asarray(study["choice"], dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for strength in sorted(set(np.abs(stimulus[stimulus != 0]).tolist())):
        selected = np.isclose(np.abs(stimulus), strength)
        correct = np.where(stimulus[selected] > 0, choice[selected], 1 - choice[selected])
        predicted = np.where(
            stimulus[selected] > 0,
            probability[selected],
            1.0 - probability[selected],
        )
        rows.append(
            {
                "absolute_contrast": float(strength),
                "n_trials": int(np.count_nonzero(selected)),
                "observed_accuracy": float(np.mean(correct)),
                "predicted_accuracy": float(np.mean(predicted)),
            }
        )
    return rows


def _counts_by_position(
    study: Study,
    positions: np.ndarray[Any, Any],
    *,
    column: str = "session_order",
) -> dict[str, int]:
    return {
        str(position): int(np.count_nonzero(np.asarray(study[column])[positions] == position))
        for position in range(6)
    }


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    result = run(args.cache)
    rendered = render(result, allow_nan=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
