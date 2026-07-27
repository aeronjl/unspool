# Cross-lab trajectory shape

“The labs differ” can refer to several distinct objects. One lab may have a higher average
parameter throughout learning, change by a larger amount, follow a genuinely different
path, or merely appear different because its animals were observed on a different clock.
Unspool makes those possibilities separate before attaching an inferential claim to any of
them.

## The panel contract

`TrajectoryPanel` contains one row per independent subject and one column per position on
an explicitly named common clock. It does not interpolate or align curves. Clock and
landmark construction belongs upstream, where fold-safe transforms can retain how the
alignment was learned. This is intentionally stricter than silently applying dynamic time
warping: a shift in *when* an animal learns can itself be the scientific result.

```python
from unspool import TrajectoryPanel, compare_trajectory_shapes

panel = TrajectoryPanel(
    grid=aligned_sessions,
    values=subject_parameter_paths,
    subjects=subject_ids,
    groups=lab_ids,
    clock_name="sessions_from_training_landmark",
    parameter_name="stimulus_weight",
)
report = compare_trajectory_shapes(panel, bootstrap_seed=2026)
```

Every subject must belong to exactly one group in a panel. `audit_trajectory_replication`
reports the number of animals per group before any distance is calculated. By default,
comparison requires at least two independent animals in every lab. Two is a structural
minimum, not a claim that two animals provide adequate power.

## What “shape” means here

<figure class="doc-figure doc-figure--wide">
  <img src="../assets/trajectory-components.svg" alt="Reference, level-shift, amplitude-shift, and sinusoidal shape-change trajectories alongside a heat map of level, amplitude, and shape contrast metrics from the committed benchmark.">
  <figcaption><strong>Trajectory components.</strong> The left panel defines the simulation geometry; the right panel shows that level, amplitude, and shape metrics respond differently to the three reference contrasts in the committed benchmark.</figcaption>
</figure>

For a group-mean trajectory \(f(t)\), trapezoidal weights on the declared grid define:

- **level**: the weighted mean of \(f\);
- **centered trajectory**: \(f\) minus its level;
- **amplitude**: the weighted root-mean-square of the centered trajectory;
- **scale-free shape**: the centered trajectory divided by its amplitude.

Pairwise reports retain raw distance, signed level difference, centered distance, signed
amplitude difference, and scale-free shape distance. The first four retain the scientific
parameter's scale where appropriate. Shape distance is dimensionless and ranges from zero
to two. A flat group mean has zero amplitude, so its scale-free shape is undefined and is
reported as unresolved rather than manufactured by division through a tolerance.

This decomposition is deliberately modest. Functional-data methods often separate
amplitude and phase variation through curve registration; Unspool does not estimate phase
warps here because doing so would change the scientific clock. Registration can be added
later as an explicit, validated transform rather than an invisible property of a metric.

## What the bootstrap means

The comparison resamples subjects independently within each named lab and recomputes the
lab means and all contrasts. Its percentile intervals therefore quantify uncertainty in
those fixed lab means under subject sampling. They do **not** support generalization to a
population of laboratories: that requires replicated labs as the sampling unit or a
hierarchical lab model.

Input trajectories are also treated as fixed summaries. The subject bootstrap does not
propagate trial-level fitting, alignment-landmark, or population-parameter uncertainty.
These are descriptive percentile intervals, not null-hypothesis tests; distance estimates
are non-negative and need a design-specific recovery or equivalence target to be
interpreted. Whole-lab prediction remains a separate prospective question and should use
`leave_one_lab_out_splits` so every animal and session from the test lab is excluded from
model fitting.

This distinction follows the nested-data warning emphasized by Saravanan, Berman, and
Sober: repeated trials or sessions do not replace independent animals. It also matters for
the IBL learning data. The full IBL study included many mice per institution and found
variation in learning speed across mice and laboratories, but Unspool's compact engineering
panel deliberately selects one animal from each of nine labs. Its audit therefore fails
cross-lab trajectory readiness even though its provenance and leave-one-lab-out coverage
remain useful.

## Validation evidence

The matched [`trajectory_shapes` benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/trajectory_shapes)
simulates replicated labs whose generating differences are level-only, amplitude-only, or
shape-changing. Across 20 pinned repetitions, the decomposition recovers all three
components and rejects a separate singleton-lab design. This is design-specific recovery
evidence, not a universal power guarantee.

Primary methodological context:

- Saravanan, Berman, and Sober, “Application of the hierarchical bootstrap to multi-level
  data in neuroscience,” *Neurons, Behavior, Data analysis, and Theory* (2020),
  <https://doi.org/10.51628/001c.13927>.
- International Brain Laboratory, “Standardized and reproducible measurement of
  decision-making in mice,” *eLife* (2021), <https://doi.org/10.7554/eLife.63711>.
- Srivastava et al., “Registration of functional data using Fisher–Rao metric,”
  <https://arxiv.org/abs/1103.3817>.
