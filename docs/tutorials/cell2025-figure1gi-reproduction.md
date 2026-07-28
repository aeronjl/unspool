# Reproducing Cell 2025 Figure 1G and 1I

!!! abstract "Panel-level reproduction"

    This worked example independently recomputes the two animal-level correlations in
    Figure 1G and 1I of Liebana, Laffere et al. (*Cell*, 2025). It preserves the published
    estimands, cohort, windows, axes, statistics, and released mouse-colour mapping while
    making every intentional rendering change explicit.

## Result first

<figure class="doc-figure doc-figure--wide" data-figure-kind="Independent reproduction">
  <img src="../../assets/cell2025-strategy.svg" alt="Two animal-level scatter plots reconstructing Cell Figure 1G and 1I: early bias is negatively associated with late bias and positively associated with late right-minus-left psychometric slope across 30 mice.">
  <a class="doc-figure__full-resolution" href="../../assets/cell2025-strategy.svg" target="_blank" rel="noopener">Open full-resolution Figure 1G/I ↗</a>
  <figcaption><strong>Independent reproduction · Cell Figure 1G and 1I.</strong> The checksum-pinned trial table recovers the published early-to-late bias reversal and early-bias/later-asymmetry relationship. The regression band is a deterministic paired-animal bootstrap; colour preserves the released continuous trajectory-asymmetry variable rather than replacing it with three discrete classes.<span class="doc-figure__meta"><strong>Unit:</strong> animal · <strong>n:</strong> 30 animals, 192,238 retained trials · <strong>Estimands:</strong> Pearson early-bias/late-bias and early-bias/late-slope correlations · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

| Panel | Published quantity | Paper | Independent reproduction |
| --- | --- | ---: | ---: |
| 1G | Early bias versus late bias | <em>r</em> = -0.53, <em>p</em> < 0.01 | <em>r</em> = -0.52764, <em>p</em> = 0.00273 |
| 1I | Early bias versus late R-L slope | <em>r</em> = 0.69, <em>p</em> < 0.0001 | <em>r</em> = 0.69479, <em>p</em> = 2.04 × 10<sup>-5</sup> |

The paper reports rounded coefficients and thresholded p values in the display. Unspool
shows the independently recomputed values at greater precision; it does not compare the
rounded labels as though they were separate targets.

## What like-for-like means here

The target is the scientific display contract, not a pixel trace. The paper PDF and the
released analysis define the following mapping:

| Contract field | Figure 1G | Figure 1I |
| --- | --- | --- |
| x variable | Mean zero-contrast bias over paper days 4-8 | Same |
| y variable | Mean zero-contrast bias in the final-five-paper-day window | Mean right-minus-left psychometric slope in that window |
| Experimental unit | Mouse | Mouse |
| Denominator | 30 mice | 30 mice |
| Statistic | Pearson correlation | Pearson correlation |
| x limits | -0.4 to 0.4 | -0.5 to 0.5 |
| y limits | -0.4 to 0.4 | -1.02 to 1.02 |
| Reference geometry | Horizontal and vertical zero lines | Horizontal and vertical zero lines |
| Point colour | Released continuous `prop_below` trajectory-asymmetry mapping | Same |

The late window is worth stating carefully. Released code uses:

```python
maximum_day - 5 < paper_day <= maximum_day
```

That is a final-five-**paper-day** window. It is not an unconditional slice of the last
five observed rows. Exclusions and sparse qualifying days mean an animal can contribute
fewer than five daily summaries inside the window. The prospective forecast in the
companion chapter has a different contract and deliberately selects five observed future
sessions.

## Source trace

The audit follows primary, versioned sources:

| Layer | Frozen source |
| --- | --- |
| Paper | [Cell article, Figure 1G and 1I](https://doi.org/10.1016/j.cell.2025.05.025), journal page 3790, CC BY 4.0 |
| Trial data | [Figshare data v1](https://doi.org/10.6084/m9.figshare.28877912.v1), member `long_term_learning_dataset_preprocessed_behaviour_all.csv`, SHA-256 `94a6d5…b8048a` |
| Released analysis | [Figshare software v1](https://doi.org/10.6084/m9.figshare.28877942.v1), MIT license |
| Released code | [`behaviour.ipynb` at commit `2faa468`](https://github.com/SamuelLiebana/da_long_term_learning/blob/2faa4680d5e9c0d6a9df516e3dede8c641e39a72/scripts/behaviour.ipynb), headings “Figure 1G” and “Figure 1I” |
| Colour artifact | `psych_metric_trajectory_fit_df.csv`, SHA-256 `e5cd06…664ee` |
| Machine-readable audit | [`figure1gi_audit.json`](https://github.com/aeronjl/unspool/blob/main/benchmarks/cell2025/figure1gi_audit.json) |

The public trial table is processed in the released order: remove no-go trials; compute
within-mouse/day response-time z scores; retain rows below 2; retain first presentations;
exclude shaped and non-learning animals; and require eligible early observations. Source
session identity is retained even where two DAP021 sessions share paper day 1.

## Preserved and intentionally changed

| Preserved | Intentionally changed |
| --- | --- |
| Animal-level summaries and denominator | Two separately exported panels are composed side by side |
| Days 4-8 and final-five-paper-day definitions | DejaVu Sans replaces the released Arial setting |
| Pearson correlations | The released unseeded seaborn bootstrap becomes 2,000 seeded animal resamples |
| Published axis limits and tick locations | Compact titles, panel letters, and a standalone source note are added |
| Zero lines and linear regression geometry | White marker borders improve separation without changing values |
| Five-anchor continuous colour map and `prop_below` values | SVG is retained as searchable text rather than rasterized labels |

These changes improve determinism, accessibility, and standalone interpretation. They do
not alter the plotted observations or estimands. Because the correlations are recomputed
from public trials rather than replayed from a released result file, the evidence class is
**independent reproduction**, not “released result.”

## Reproduce it

```bash
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025_flagship.fetch_released_artifacts
uv run python -m benchmarks.cell2025.benchmark \
  benchmarks/cell2025/data/long_term_learning_dataset_preprocessed_behaviour_all.csv
uv run --group docs python -m scripts.plot_documentation_figures
uv run --group docs pytest tests/test_cell2025_benchmark.py tests/test_documentation.py
```

The numerical benchmark rejects changed correlations, denominators, or source checksums.
The documentation contract rejects an unregistered figure, serif typography, text paths,
missing evidence classification, or incomplete caption metadata.

## Claim boundary

Figure 1G shows that early and late bias have opposite animal-level ordering; Figure 1I
shows that early bias covaries with which psychometric slope is larger late in learning.
Neither correlation shows that early bias causes the later strategy. They also do not
identify dopamine, reward history, innate preference, or any fitted model as the mechanism.
Those require the paper's other experimental layers or a separately validated analysis.

[Continue to the prospective Cell study](cell2025-learning-trajectories.md)
