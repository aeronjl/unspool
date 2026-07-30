"""Mix a joint choice/response-time Wiener model with a uniform response process."""

import numpy as np

from behavio import Study, UniformResponseGuess, WienerDriftDiffusion, mix

design = Study.factorial(
    trials=800,
    subjects="synthetic-subject",
    sessions="session-1",
    columns={"stimulus": lambda rng, n_rows: rng.normal(size=n_rows)},
    seed=204,
)
model = mix(
    WienerDriftDiffusion(
        covariates=("stimulus",),
        nondecision_time_bounds=(0.1, 0.6),
        n_restarts=3,
    ),
    UniformResponseGuess(time_bounds=(0.05, 3.0)),
    weight_bounds=(0.0, 0.25),
)
truth = model.from_natural(
    {
        "drift.intercept": 0.2,
        "drift.stimulus": 1.2,
        "boundary": 1.2,
        "starting_bias": 0.45,
        "nondecision_time": 0.25,
        "contaminant_rate": 0.05,
    }
)
simulation = model.simulate_with_component(design, truth, seed=205)
fit = model.fit(simulation.study)
responsibility = model.component_responsibility(simulation.study, fit)
recovered = model.to_natural(fit.estimates)

print("Contaminant-aware joint choice/response-time model")
print(f"fit audit: {fit.audit().status.value} {list(fit.audit().issue_codes)}")
print(f"generated contaminants: {simulation.n_from_component}")
print(f"posterior expected count: {float(np.sum(responsibility)):.2f}")
for name in model.natural_names:
    true_value = float(model.to_natural(model.parameter_vector(truth))[name])
    print(f"{name:25s} truth={true_value:7.3f} estimate={recovered[name]:7.3f}")
