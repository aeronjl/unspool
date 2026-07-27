# Reproducing early and late learning strategy in Cell 2025

!!! info "Bounded behavioural reproduction"

    This workflow reproduces Figure 1 behavioural relationships. It does not reproduce the
    paper's trajectory clustering, neural analyses, or causal claims about dopamine.

## Scientific question

Does an animal's early choice bias predict the structure of its strategy after extended
training?

Liebana, Laffere et al. reported that early strategy predicted later psychometric
asymmetry in “Dopamine encodes deep network teaching signals for individual learning
trajectories” ([Cell, 2025](https://doi.org/10.1016/j.cell.2025.05.025)). Unspool
independently reimplements the published Figure 1 trial exclusions and summary metrics.

## Cohort and denominator

The checksum-pinned input is the public behavioural member of the paper's Figshare archive.
After excluding no-go and repeat trials, response-time outliers, shaped animals, and
non-learners, the analysis retains:

| Unit | Count |
| --- | ---: |
| Animals | 30 |
| Trials | 192,238 |
| Canonical source sessions | 950 |
| Published session summaries | 949 |

The difference between the two session counts is deliberate. Two dated source sessions for
one animal share the paper's `sessionNum == 1`; Unspool preserves their identities and only
the paper-specific summary groups them.

## Estimand and result

Early bias is averaged over days 4–8. Late right-minus-left psychometric slope is averaged
over each animal's final five sessions. The experimental unit is the animal.

<figure class="doc-figure">
  <img src="../../assets/cell2025-strategy.svg" alt="Scatter plot for 30 mice showing that early choice bias predicts later right-minus-left psychometric slope asymmetry, with a fitted regression line and animal-level observations.">
  <figcaption><strong>Published relationship, independently reproduced.</strong> Each point is one animal; the display follows the paper-specific exclusions and summary windows described above.</figcaption>
</figure>

The reproduced relationship is `r = 0.69479`, `p = 2.04 × 10⁻⁵`. Mean non-zero-stimulus
accuracy rises from `0.51734` in the first session to `0.75803` in the last. The benchmark
also recovers the complementary early-versus-late bias reversal (`r = −0.52764`).

These correlations describe stable individual differences under the paper's inclusion and
summary rules. They do not show that early bias causes the later strategy.

## Reproduce it

```bash
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025.benchmark \
  benchmarks/cell2025/data/long_term_learning_dataset_preprocessed_behaviour_all.csv
uv run --group docs python -m scripts.plot_documentation_figures \
  --cell-data benchmarks/cell2025/data/long_term_learning_dataset_preprocessed_behaviour_all.csv
```

The [benchmark implementation](https://github.com/aeronjl/unspool/tree/main/benchmarks/cell2025)
pins the source member, checksum, counts, correlation coefficients, and regression
tolerances.
