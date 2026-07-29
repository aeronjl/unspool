# Replicated IBL learning trajectories

!!! warning "Transition-conditioned descriptive cohort"

    Animals enter this cohort because they reached a later training-policy transition.
    Endpoint positions are ordinal rather than uniform elapsed time. The trajectories are
    therefore a retrieval and longitudinal-geometry result, not an unbiased estimate of
    learning in all trained animals.

## Scientific question

Can one outcome-blind rule recover comparable early and late training windows for multiple
animals within each laboratory in the International Brain Laboratory's public behavioural
release?

The source study established a standardized decision-making task across laboratories
([IBL, 2021](https://doi.org/10.7554/eLife.63711)). Behavio addresses exact trial-table
UUIDs through ONE and retains the first and final three pre-transition sessions for every
eligible animal.

## Cohort and data contract

The frozen manifest contains 78 animals across nine labs, with at least four animals per
lab, 468 source datasets, and 260,833 trials. Source `choice = -1`, `0`, and `+1` remain
unmodified in the retrieval study. Every trial retains release, session, dataset, Alyx,
and row-level provenance.

## What changes across the six windows?

<figure class="doc-figure doc-figure--wide" data-figure-kind="Literature-shaped">
  <img src="../../assets/ibl-learning-trajectories.svg" alt="Nine laboratory-level easy-trial accuracy trajectories across six outcome-blind endpoint positions, with a bold animal-weighted population trajectory rising from early to late training.">
  <figcaption><strong>Literature-shaped · longitudinal IBL retrieval.</strong> Thin lines preserve laboratory-level variation; the bold population line weights animals equally rather than laboratories. The transition-conditioned cohort is not an unbiased population learning curve.<span class="doc-figure__meta"><strong>Unit:</strong> animal within laboratory · <strong>n:</strong> 78 animals, 9 laboratories, 468 sessions · <strong>Estimand:</strong> animal-weighted easy-trial accuracy across ordinal endpoint positions · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

All 78 animals have higher mean easy-trial accuracy in the final three windows than in the
first three. Subject-weighted accuracy rises from `0.49066` to `0.91347`, a mean increase
of `0.42281`.

The figure preserves laboratory means because the labs differ in sample size and trajectory
shape. Its bold population line weights animals equally; it is not a population-of-labs
estimate. The gap between positions 2 and 3 differs by animal because the retained early
and late windows anchor opposite ends of pre-transition training.

## Why this is not yet a cognitive-model result

Accuracy establishes that retrieval, chronology, and the endpoint-window contrast behave
as intended. It does not distinguish stable individual bias, smooth drift, discrete task
states, or learning rules. That question requires future-session prediction and competing
models, addressed in the [prospective worked study](ibl2021-prospective-selection.md).

## Reproduce it

```bash
uv run --extra ibl python -m benchmarks.ibl2021_replicated.benchmark
uv run --group docs python -m scripts.plot_documentation_figures \
  --skip-cell
```

The [replicated IBL benchmark](https://github.com/aeronjl/behavio/tree/main/benchmarks/ibl2021_replicated)
pins all source identities, hashes, cohort rules, lab summaries, trajectory geometry, and
population-validation coverage.
