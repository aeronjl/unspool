# Does longitudinal drift predict future IBL behaviour?

!!! warning "Prediction is not mechanism identification"

    Lower future-session log loss supports a predictive trajectory structure under this
    design. It does not show that the fitted coefficients are a biological learning law.

## Scientific question

Do session-varying individual trajectories predict a later session better than static
partial pooling—and does any advantage transport to animals in an entirely unseen lab?

The modelling panel keeps the same 78 animals and six outcome-blind positions, retaining up
to the first 100 source rows per session before removing no-go responses. It contains
46,152 valid left/right choices.

All candidates receive signed contrast and a one-trial, session-reset choice-history term.
Scoring is filtered one-step-ahead within the final session: an observed choice may
initialize history for the next trial, but the model never sees a future choice before it
is scored.

## Two prospective boundaries

For represented animals, positions 0–4 train the model and position 5 is scored. For lab
transfer, all animals from one lab are removed from fitting; positions 0–4 in the other
eight labs predict position 5 in the held-out lab.

The fixed comparison finds:

| Target | Static | Smooth drift | Static minus drift, 95% interval |
| --- | ---: | ---: | ---: |
| Same animals, future session | 0.6400 | **0.5549** | +0.0851 (+0.0162, +0.1460) |
| Unseen lab and animals | 0.6285 | **0.6049** | +0.0236 (−0.0744, +0.1049) |

Drift clearly improves the represented-animal forecast. Its smaller cross-lab point
advantage remains unresolved.

## Select smoothness without opening the test set

The nested procedure compares static partial pooling with smoothness 1, 3, and 9 inside
each outer training study. Same-animal inner folds forecast positions 3 and 4 from earlier
prefixes. Lab-transfer inner folds forecast position 4 in an inner held-out lab. Position 5
and the outer held-out lab remain structurally absent from selection.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Literature-shaped">
  <img src="../../assets/ibl-prospective-selection.svg" alt="Training-only candidate scores select smoothness nine, followed by paired subject-level outer-test log-loss differences for represented animals and held-out laboratory transfer.">
  <figcaption><strong>Literature-shaped · nested selection without test-set reuse.</strong> Candidate choice happens inside the outer training study; paired points and intervals summarize the untouched future-session evaluation. The result does not establish transport beyond the fixed IBL laboratories.<span class="doc-figure__meta"><strong>Unit:</strong> subject · <strong>n:</strong> 78 animals across 9 laboratories · <strong>Estimand:</strong> paired untouched-session log-loss difference after training-only smoothness selection · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

Smoothness 9 is selected in the one represented-animal fold and all nine held-out-lab
folds. On untouched position 5 it lowers subject-balanced log loss by `0.00768` relative to
fixed smoothness 3 for represented animals and by `0.00777` under lab transfer; both paired
95% intervals lie below zero.

## The result that should remain uncomfortable

Nine is the upper edge of the declared grid. The analysis supports stronger regularization
than three under this design, but does not locate a continuous optimum. Expanding the grid
after inspecting these outer results would require a new untouched evaluation layer or a
new dataset. The selected procedure's advantage over static for unseen labs also remains
unresolved.

## Reproduce it

```bash
uv run --extra ibl python -m benchmarks.ibl2021_prospective.benchmark
uv run --extra ibl python -m benchmarks.ibl2021_nested_selection.benchmark
uv run --group docs python -m scripts.plot_documentation_figures --skip-cell
```

The fixed [prospective comparison](https://github.com/aeronjl/unspool/tree/main/benchmarks/ibl2021_prospective)
and [nested-selection benchmark](https://github.com/aeronjl/unspool/tree/main/benchmarks/ibl2021_nested_selection)
retain every fold, audit, subject score, candidate score, interval, and seed.
