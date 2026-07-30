"""Fit and inspect session-varying Wiener parameter trajectories."""

import numpy as np

from behavio import Study, WienerDriftDiffusion, evaluate_splits, forward_session_splits
from behavio.compose import smooth

n_sessions = 5
design = Study.factorial(
    trials=80,
    subjects="synthetic-subject",
    sessions=n_sessions,
    columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
    seed=301,
)
model = smooth(
    WienerDriftDiffusion(
        covariates=("stimulus",),
        n_restarts=2,
        max_iterations=300,
        simulation_time_step=0.001,
    ),
    over="session_order",
    knots=tuple(float(session) for session in range(n_sessions)),
    parameters=("drift.stimulus", "boundary"),
    smoothness=10.0,
)
truth = model.parameters_from_paths(
    {
        "drift.intercept": 0.1,
        "drift.stimulus": [0.4, 0.7, 1.0, 1.3, 1.6],
        "boundary": [1.5, 1.4, 1.3, 1.2, 1.1],
        "starting_bias": 0.48,
        "nondecision_time": 0.25,
    }
)
study = model.simulate(design, truth, seed=302)
fit = model.fit(study)
trajectory = model.coefficient_trajectory(fit)
fold = evaluate_splits(
    model,
    study,
    forward_session_splits(study, min_train_sessions=4),
)[0]

print("Session-varying drift-diffusion trajectories")
print(f"clock: {trajectory.clock}")
print(f"fit audit: {fit.audit().status.value} {list(fit.audit().issue_codes)}")
print("stimulus drift:", np.round(trajectory.path("drift.stimulus"), 3).tolist())
print("boundary:      ", np.round(trajectory.path("boundary"), 3).tolist())
print(f"Prospective final-session joint log loss: {fold.mean_log_loss:.3f}")
