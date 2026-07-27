"""Convert the released Cell 2025 Q-value pickle into a safe JSON summary.

The official pickle requires the paper's pinned JAX version and must only be loaded from
the checksum-pinned release. The resulting JSON contains no executable serialization.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cell2025.fetch_data import sha256

ANALYSIS_DOI = "10.6084/m9.figshare.28877942.v1"
SOURCE_MEMBER = "data/first_5_session_action_value_model.pickle"
SOURCE_SHA256 = "ba69393ca8ceb8932c77958ba66f27d1c14089684adbb0fd32a38f0e27daee5e"
METHODS = (
    "only_innate",
    "only_sess",
    "only_reward",
    "innate_and_reward",
    "sess_and_reward",
)
PARAMETERS = (
    "beta",
    "alpha_plus",
    "alpha_minus",
    "q_left_innate",
    "q_right_innate",
    "q_left_day_1",
    "q_left_day_2",
    "q_left_day_3",
    "q_left_day_4",
    "q_left_day_5",
    "q_right_day_1",
    "q_right_day_2",
    "q_right_day_3",
    "q_right_day_4",
    "q_right_day_5",
)


def summarize(path: Path) -> dict[str, Any]:
    """Validate and summarize the official optimization artifact."""

    digest = sha256(path)
    if digest != SOURCE_SHA256:
        raise ValueError(f"input checksum mismatch: observed {digest}, expected {SOURCE_SHA256}")
    with path.open("rb") as handle:
        released = pickle.load(handle)
    if not isinstance(released, dict) or len(released) != 30:
        raise ValueError("released Q-value artifact must contain 30 animals")

    animals: dict[str, Any] = {}
    bics_by_method: dict[str, list[float]] = {method: [] for method in METHODS}
    best_counts = {method: 0 for method in METHODS}
    for raw_subject, methods in released.items():
        subject = str(raw_subject)
        if tuple(methods) != METHODS:
            raise ValueError(f"subject {subject!r} has an unexpected method contract")
        method_payloads: dict[str, Any] = {}
        for method in METHODS:
            result = methods[method]
            parameters = np.asarray(result["fit_params"], dtype=np.float64)
            losses = np.asarray(result["loss_traj"], dtype=np.float64)
            q_left = np.asarray(result["ql_traj"], dtype=np.float64)
            q_right = np.asarray(result["qr_traj"], dtype=np.float64)
            if parameters.shape != (15,) or losses.shape != (5_000,):
                raise ValueError(f"subject {subject!r}, method {method!r} changed shape")
            if q_left.shape != q_right.shape or q_left.ndim != 1:
                raise ValueError(f"subject {subject!r}, method {method!r} has invalid Q paths")
            final_loss = float(losses[-1])
            n_parameters = int(np.count_nonzero(parameters))
            n_datapoints = len(q_left)
            bic = float(2 * final_loss + n_parameters * np.log(n_datapoints))
            bics_by_method[method].append(bic)
            method_payloads[method] = {
                "final_negative_log_likelihood": final_loss,
                "bic": bic,
                "n_bic_datapoints": n_datapoints,
                "n_nonzero_parameters": n_parameters,
                "parameters": {
                    name: float(value) for name, value in zip(PARAMETERS, parameters, strict=True)
                },
            }
        best_method = min(METHODS, key=lambda method: method_payloads[method]["bic"])
        best_counts[best_method] += 1
        animals[subject] = {"best_bic_method": best_method, "methods": method_payloads}

    aggregate = {}
    for method in METHODS:
        values = np.asarray(bics_by_method[method], dtype=np.float64)
        aggregate[method] = {
            "mean_bic": float(np.mean(values)),
            "standard_error_bic": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "best_bic_animal_count": best_counts[method],
        }
    return {
        "analysis": "released first-five-day single-state Q-value nested comparison",
        "interpretation": (
            "retrospective released fit; summarized, not independently re-optimized"
        ),
        "analysis_doi": ANALYSIS_DOI,
        "source_member": SOURCE_MEMBER,
        "source_sha256": digest,
        "contract": {
            "days": [1, 2, 3, 4, 5],
            "optimizer": "optax Adam",
            "iterations": 5_000,
            "learning_rate": 0.001,
            "initial_parameters": [5.0, 0.01, 0.001, *([0.5] * 12)],
            "projection": "non-negative parameters",
            "methods": list(METHODS),
            "bic": "2 * final negative log likelihood + nonzero parameter count * log(len(Q))",
            "note": "len(Q) is choice count plus one in the released calculation",
        },
        "aggregate": aggregate,
        "animals": animals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("released_q_value_summary.json"),
    )
    arguments = parser.parse_args()
    result = summarize(arguments.pickle.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    arguments.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
