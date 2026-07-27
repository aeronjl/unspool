# Replicated IBL training-only nested selection

This benchmark turns the fixed prospective comparison into a model-selection procedure.
It asks whether static partial pooling or a hierarchical drifting trajectory should be
chosen using only earlier training data, before the final forecast outcomes are exposed.
The outer score therefore estimates the performance of the complete selection procedure,
not a model chosen after inspecting position 5.

## Candidate and sampling contract

The checksum-pinned panel is identical to the
[fixed prospective benchmark](../ibl2021_prospective/README.md): 78 animals in nine labs,
six outcome-blind endpoint windows, and 46,152 valid left/right trials after retaining up
to the first 100 source rows per session. The candidate order and tie break are fixed:

1. static hierarchical Bernoulli history GLM;
2. hierarchical smooth drift with smoothness `1`;
3. hierarchical smooth drift with smoothness `3`;
4. hierarchical smooth drift with smoothness `9`.

Every candidate uses signed contrast, a one-trial session-reset choice lag, `l2 = 0.02`,
and population/subject partial-pooling scale `0.4`. Drifting candidates share knots
`(0, 2, 5)` and use the declared smoothness for both population and subject paths.

## Two nested boundaries

For represented-animal forecasting, the outer fold fits positions 0–4 and scores position
5. Inner selection sees only that outer-training study: it forecasts position 3 from 0–2
and position 4 from 0–3.

For lab transfer, each outer fold withholds all animals and position-5 trials from one lab.
Within the other eight labs, inner leave-one-lab-out folds fit positions 0–3 in seven labs
and forecast position 4 in the eighth. The outer held-out lab is structurally absent during
candidate selection.

The primary metric is subject-balanced log loss. Paired 5,000-draw subject bootstraps
compare the selected procedure with the two candidates from the fixed benchmark. The
secondary lab-balanced interval resamples the nine empirical held-out labs and is not
population-of-laboratories inference.

## Result

| Outer target | Selected candidate(s) | Selected procedure | Difference from fixed smoothness 3 (95% interval) |
| --- | --- | ---: | ---: |
| Same animals, position 5 | smoothness 9 in 1/1 fold | **0.5472** | −0.00768 (−0.01285, −0.00338) |
| Unseen lab and animals, position 5 | smoothness 9 in 9/9 folds | **0.5971** | −0.00777 (−0.01375, −0.00257) |

Negative differences favor training-only selection. Every inner candidate fit and selected
outer fit passes the numerical audit. The represented-animal procedure also improves on
fixed static partial pooling by `−0.09278` (`−0.14802`, `−0.02672`). For unseen-lab
transfer, its difference from fixed static partial pooling is `−0.03133` (`−0.10978`,
`+0.05665`), so the transport advantage over static remains unresolved.

The inner preference for smoothness 9 is consistent across all nine lab-transfer outer
folds, and its untouched outer scores improve on the pre-existing smoothness-3 candidate.
But 9 is the upper edge of this small declared grid. This benchmark supports stronger
regularization than 3 under this design; it does not locate an optimal continuous
smoothness or establish a general learning law. Expanding the grid after seeing these
outer results would require a new untouched evaluation layer or a new dataset.

## Reproduce

```bash
uv run --extra ibl python -m benchmarks.ibl2021_nested_selection.benchmark
```

The committed `result.json` pins the candidate grid, inner and outer fold targets, every
selection, fit audit, subject score, fixed-model comparison, and bootstrap seed.
