"""Fit and inspect session-varying Wiener parameter trajectories."""

import numpy as np

from behavio import SmoothWienerDriftDiffusion, Study, evaluate_splits, forward_session_splits

generator = np.random.default_rng(301)
n_sessions = 5
trials_per_session = 80
n_trials = n_sessions * trials_per_session
design = Study(
    {
        "subject": ["synthetic-subject"] * n_trials,
        "session": [
            f"session-{session}" for session in range(n_sessions) for _ in range(trials_per_session)
        ],
        "trial": list(range(trials_per_session)) * n_sessions,
        "session_order": [
            session for session in range(n_sessions) for _ in range(trials_per_session)
        ],
        "stimulus": generator.normal(size=n_trials),
    }
)
model = SmoothWienerDriftDiffusion(
    covariates=("stimulus",),
    knots=tuple(float(session) for session in range(n_sessions)),
    varying_parameters=("drift.stimulus", "boundary"),
    smoothness=10.0,
    n_restarts=2,
    max_iterations=300,
    simulation_time_step=0.001,
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
trajectory = model.parameter_trajectory(fit)
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
