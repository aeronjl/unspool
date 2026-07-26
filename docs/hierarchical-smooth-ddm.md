# Partially pooled Wiener trajectories

`HierarchicalSmoothWienerDriftDiffusion` estimates a smooth population trajectory together
with shrunken animal-specific deviations. It occupies the middle ground between treating
all animals as copies of one average animal and fitting every animal independently.

For selected parameter \(p\), animal \(s\), and knot \(k\),

\[
\theta_{spk}=\mu_{pk}+\delta_{spk}.
\]

The population path \(\mu_p\) has the same time-scaled first-difference penalty as
`SmoothWienerDriftDiffusion`. Subject deviations use

\[
\frac{1}{2\sigma^2}\sum_k\delta_{spk}^2
+\frac{\lambda_s}{2}\sum_{k=2}^{K}
\frac{(\delta_{spk}-\delta_{sp,k-1})^2}{u_k-u_{k-1}}.
\]

`subject_scale` is the fixed natural-scale deviation size \(\sigma\), and
`subject_smoothness` is \(\lambda_s\). This first implementation performs a joint
penalized maximum-a-posteriori fit. It is hierarchical partial pooling, but it is not a
full Bayesian posterior sampler.

## Declaring the hierarchy

Only parameters already listed in `varying_parameters` may receive subject paths. A narrow
hypothesis keeps the optimization and interpretation tractable:

```python
from unspool import HierarchicalSmoothWienerDriftDiffusion

model = HierarchicalSmoothWienerDriftDiffusion(
    covariates=("stimulus",),
    time="session_order",
    knots=(0.0, 2.0, 4.0),
    varying_parameters=("drift.stimulus", "boundary"),
    subject_parameters=("drift.stimulus", "boundary"),
    smoothness=8.0,
    subject_scale=0.2,
    subject_smoothness=8.0,
)
```

Population simulation parameters retain the smooth Wiener's stable natural-scale
coordinates. `simulate_with_effects()` either draws deviation paths from the configured
Gaussian precision or accepts explicit paths for recovery experiments. Realized random
effects are returned in `HierarchicalSmoothDriftDiffusionSimulation`; they are never added
to observed `Study` columns.

```python
simulation = model.simulate_with_effects(design, population_truth, seed=31)
fit = model.fit(simulation.study)

population = model.population_trajectory(fit)
mouse_path = model.subject_trajectory(fit, "mouse-03")
```

`HierarchicalSmoothDriftDiffusionFitResult` retains population estimates, every subject
deviation and local standard error, restart evidence, the common fit audit, and the
declared population policy. Arrays are read-only.

## Natural-scale constraints

Deviations are additive on the public natural scale. Effective drift, boundary, and bias
paths must therefore remain within their configured bounds. Explicit simulation paths are
rejected if they violate those bounds. Random simulation draws use bounded rejection, and
the joint optimizer uses a continuous quadratic constraint penalty while evaluating the
likelihood at the nearest admissible path. A fitted optimum outside tolerance is rejected
rather than silently clipped.

The local Hessian has an arrowhead structure: all animals couple to the population block,
but one animal's deviation block does not couple directly to another's. Unspool evaluates
the population, subject, and population–subject curvature blocks numerically and inverts
them with the Schur complement. This retains population–subject uncertainty coupling while
avoiding evaluations of known zero cross-animal blocks. It remains a local Gaussian
approximation conditional on the fixed penalties.

## Seen and unseen animals

For an animal present during fitting, prediction uses its fitted population-plus-deviation
trajectory. A completely unseen animal uses the population trajectory plug-in, recorded as
`unseen_subject_policy="population-trajectory-plugin"`. No uncertainty for a new random
effect is integrated into that prediction.

Use complete-subject holdouts to test the population policy and cohort-forward session
splits to test future sessions of represented animals. These are different generalization
questions and should not be pooled under one generic cross-validation score.

## Interpretation boundary

Hierarchical DDMs are valuable because they estimate group and individual parameters
simultaneously rather than imposing either complete pooling or fully independent fits.
The original HDDM recovery experiments found the greatest benefit when individual trial
counts were small. Unspool adopts that partial-pooling motivation, not HDDM's MCMC engine;
see [Wiecki, Sofer, and Frank (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3731670/).

Current limitations are explicit:

- one fixed shared `subject_scale` is used for all selected parameters;
- variance components are not estimated;
- subject deviations for stationary non-decision time are not supported;
- contaminant mixtures and within-decision time-varying dynamics are not supported;
- unseen-animal predictions are plug-ins rather than posterior predictive distributions;
- lab-level structure and aligned cross-lab trajectory comparisons remain future work.

## Recovery evidence

The [hierarchical Wiener benchmark](../benchmarks/hierarchical_smooth_ddm/README.md) makes
complete pooling, shared smooth, independent smooth, and hierarchical smooth fits compete
across three regimes. Across 20 repetitions per regime, the scientifically matched model
wins both subject-path RMSE and fifth-session joint log loss: complete pooling for
stationary identical animals, shared smooth for shared change, and hierarchical smooth for
individual change. All 480 fits converge.

Run the example and benchmark with:

```bash
uv run python examples/hierarchical_smooth_drift_diffusion.py
uv run python -m benchmarks.hierarchical_smooth_ddm.benchmark
```
