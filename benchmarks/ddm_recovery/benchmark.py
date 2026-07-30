"""Benchmark fixed-parameter Wiener recovery at two trial counts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from behavio import Study, WienerDriftDiffusion, run_parameter_recovery
from benchmarks.provenance import render

DESIGNS = {"400_trials": 400, "1200_trials": 1_200}


def build_design(*, n_trials: int, seed: int) -> Study:
    """Build four sessions with a balanced continuous evidence covariate."""

    if n_trials % 4:
        raise ValueError("n_trials must be divisible by four")
    generator = np.random.default_rng(seed)
    trials_per_session = n_trials // 4
    return Study(
        {
            "subject": ["synthetic-subject"] * n_trials,
            "session": [
                f"session-{session}" for session in range(4) for _ in range(trials_per_session)
            ],
            "trial": list(range(trials_per_session)) * 4,
            "session_order": [session for session in range(4) for _ in range(trials_per_session)],
            "stimulus": generator.normal(size=n_trials),
        }
    )


def run(*, repeats: int = 20, seed: int = 91_337) -> dict[str, Any]:
    """Run matched natural-scale parameter recovery at two sample sizes."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    model = WienerDriftDiffusion(
        predictors=("stimulus",),
        n_restarts=3,
        max_iterations=300,
        simulation_time_step=0.0001,
    )
    truth = model.parameters_from_components(
        drift={"drift.intercept": 0.2, "drift.stimulus": 1.2},
        boundary=1.2,
        starting_bias=0.45,
        nondecision_time=0.25,
    )
    design_sequences = np.random.SeedSequence(seed).spawn(len(DESIGNS))
    reports: dict[str, Any] = {}
    for (name, n_trials), sequence in zip(DESIGNS.items(), design_sequences, strict=True):
        design_seed, recovery_seed = sequence.generate_state(2, dtype=np.uint64)
        report = run_parameter_recovery(
            model,
            build_design(n_trials=n_trials, seed=int(design_seed)),
            [truth],
            repeats=repeats,
            seed=int(recovery_seed),
        )
        reports[name] = report.to_dict()
    small_summary = {row["parameter"]: row for row in reports["400_trials"]["summary"]}
    large_summary = {row["parameter"]: row for row in reports["1200_trials"]["summary"]}
    return {
        "benchmark": "fixed-parameter Wiener drift-diffusion recovery",
        "seed": seed,
        "repeats_per_design": repeats,
        "model_signature": model.signature,
        "scored_columns": list(model.scored_columns),
        "truth": dict(truth),
        "simulation": {
            "method": "Euler-Maruyama with Brownian-bridge absorption and crossing interpolation",
            "time_step_seconds": model.simulation_time_step,
            "maximum_time_seconds": model.simulation_max_time,
            "likelihood": "Navarro-Fuss paired small/large-time Wiener series",
        },
        "designs": reports,
        "rmse_decreased_for_every_parameter": all(
            large_summary[parameter]["rmse"] <= small_summary[parameter]["rmse"]
            for parameter in model.parameter_names
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=91_337)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    arguments = parser.parse_args()
    result = run(repeats=arguments.repeats, seed=arguments.seed)
    rendered = render(result)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
