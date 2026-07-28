# From learning trajectories to a prospective forecast

!!! abstract "Flagship worked study"

    This chapter independently reproduces public behavioural results from Liebana,
    Laffere et al. (*Cell*, 2025), then asks a new question under a frozen prospective
    design: after observing an animal's first eight training days, how well can we forecast
    its choices in its final five sessions?

## Why this study

“Dopamine encodes deep network teaching signals for individual learning trajectories”
reported that mice reached similar trained performance through diverse, persistent
behavioural strategies ([Cell, 2025](https://doi.org/10.1016/j.cell.2025.05.025)). That is
exactly the setting in which longitudinal software has to distinguish three things:

1. reproducing a published descriptive result;
2. forecasting observations that did not participate in fitting; and
3. identifying which latent explanation generated a trajectory.

Those claims do not follow from one another. Unspool keeps them as separate evidence
layers and retains negative or unresolved comparisons.

The primary sources are the [versioned public data release](https://doi.org/10.6084/m9.figshare.28877912.v1)
and [MIT-licensed analysis release](https://doi.org/10.6084/m9.figshare.28877942.v1). The
downloaded trial-table member is accepted only at its recorded SHA-256.

## Layer A: reproduce what the public release supports

The checksum-pinned public behaviour table supports the paper's choice, psychometric,
response-time, trajectory-visualization, and released first-five-day Q-value summaries.
It does not contain the paper's complete video, pupil, wheel, lick, photometry, dopamine,
or network-model evidence.

After the released no-go, response-time, repeat-trial, shaping, and learner exclusions,
the Figure 1 reproduction retains:

| Unit | Count |
| --- | ---: |
| Animals | 30 |
| Trials | 192,238 |
| Canonical source sessions | 950 |
| Published paper-day summaries | 949 |

Two source sessions for DAP021 share paper day 1. Unspool preserves both source identities
and combines them only when applying the paper's paper-day summary rule.

### Source-to-display correspondence

This table fixes the relationship between each Unspool display and the published or
released target. Exact paper-panel identifiers will be added only after the complete
paper-layout audit; until then, correspondence is stated at the reported-result level
rather than guessed from visual similarity.

| Unspool display | Published or released target | Relationship | Preserved | Changed or unavailable |
| --- | --- | --- | --- | --- |
| `cell2025-strategy.svg` | Reported early-bias/later-strategy association | Independent reproduction | public trials, exclusions, animal unit, early and late summaries | plotting geometry; no neural variables |
| `cell2025-trajectories.svg` | Released GP trajectory and soft-DTW visualization | Exact reproduction | released paths, ordering, memberships, normalized progress | explanatory annotation added |
| `cell2025-qvalue-response-time.svg`, A | Released first-five-day Q-value comparison | Released result | released per-animal BIC summary | no independent reoptimization |
| `cell2025-qvalue-response-time.svg`, B | Reported response-time change through learning | Independent reproduction | public trials, response-time definition, animal pairing | compact first-versus-final-five display |
| `cell2025-forecast.svg` | No published target | Literature-shaped analysis | public behavioural cohort and chronology | new candidates, prospective boundary, estimand, uncertainty |
| `cell2025-recovery.svg` | No published target | Synthetic benchmark | exact public trial/session/contrast geometry | simulated outcomes and declared generators |

### Early and late strategy

<figure class="doc-figure" data-figure-kind="Independent reproduction">
  <img src="../../assets/cell2025-strategy.svg" alt="Animal-level scatter plot showing a positive association between early zero-contrast bias and the final-five-session right-minus-left psychometric slope, with 30 observations and a fitted line.">
  <figcaption><strong>Independent reproduction · published strategy association.</strong> Early bias over days 4–8 predicts final-five-session psychometric asymmetry: <em>r</em> = 0.69479, <em>p</em> = 2.04 × 10⁻⁵. This animal-level association does not establish that early bias causes later strategy.<span class="doc-figure__meta"><strong>Unit:</strong> animal · <strong>n:</strong> 30 animals, 192,238 retained trials · <strong>Estimand:</strong> Pearson early-bias/later-asymmetry association · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

Mean non-zero-stimulus accuracy rises from `0.51734` in the first session to `0.75803`
in the last. The complementary early-versus-late bias reversal is also recovered
(`r = −0.52764`, `p = 0.00273`). These are animal-level associations, not evidence that
early bias causes the later strategy.

### Continuous trajectories, not natural kinds

The released implementation smooths each animal's left and right psychometric slopes with
Gaussian processes and visualizes three soft-DTW clusters. Reproduction required the
released numerical environment and its implicit alphabetical animal ordering. The final
semantic membership matches the released CSV for all 30 animals.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Exact reproduction">
  <img src="../../assets/cell2025-trajectories.svg" alt="Thirty overlapping right-minus-left psychometric-slope trajectories across normalized training progress, colored by the released left, balanced, and right visualization labels.">
  <figcaption><strong>Exact reproduction · the labels summarize continuous diversity.</strong> Individual paths overlap and change through training. The colours reconstruct the released retrospective visualization; they are not prospective classes or evidence for three biological kinds.<span class="doc-figure__meta"><strong>Unit:</strong> animal trajectory · <strong>n:</strong> 30 released trajectories · <strong>Estimand:</strong> GP right-minus-left psychometric slope over normalized training progress · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

### Reward history and response time

The released Q-value artifact compares five first-five-day models. Innate-plus-reward has
the lowest mean BIC (`1349.14`) and is the animal-level winner for 13 animals; reward only
wins for 12. These are summaries of the checksum-pinned released fit, not an independent
reoptimization of its approximately 115-minute procedure.

An independent chronometric summary finds mean response time falling from `3.714 s` in the
first session to `0.949 s` over the final five (`paired p = 9.16 × 10⁻¹¹`). This is a
descriptive choice-completion measure; response time is not part of the prospective model.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Mixed evidence">
  <img src="../../assets/cell2025-qvalue-response-time.svg" alt="Two panels showing mean BIC for five released Q-value models and paired first-session versus final-five-session response times for 30 animals.">
  <figcaption><strong>Mixed evidence · two bounded descriptive layers.</strong> The left panel summarizes a released retrospective fit; the right independently shows faster responses across learning. Neither identifies a dopaminergic mechanism.<span class="doc-figure__meta"><strong>Unit:</strong> animal · <strong>n:</strong> 30 animals · <strong>Estimands:</strong> released mean BIC and paired first-versus-final-five response time · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

## Layer B: freeze a new forecast before fitting

The new estimand is animal-balanced held-out choice log loss in an animal's final five
sessions after observing only its first eight paper days. Completed trajectories from the
other animals act as a historical reference cohort.

The derived panel contains 30 animals, 73,042 trials, 391 source sessions, and 390 derived
paper-day sessions. Each animal has 13 aligned coordinates: days 1–8 followed by its final
five sessions. Its intervening sessions are absent, rather than accidentally entering a
learned transform.

```text
completed reference animals:  day 1 ───────────────────────── final 5  [fit]
forecast animals:             day 1 ── day 8   · · · · · ·   final 5
                                       context                 score
```

Six deterministic animal-level folds each use 25 completed reference animals and five
forecast animals. The five forecast animals contribute days 1–8 as fitting and prediction
context; only their final five sessions are scored. Pairwise intervals resample animals,
not trials. This is prospective only under the declared deployment order: the reference
cohort must have completed training before a new animal is forecast.

The [frozen design](https://github.com/aeronjl/unspool/blob/main/benchmarks/cell2025_flagship/DESIGN.md)
declares exclusions, clocks, features, candidates, regularization, fold assignment,
recovery, and interpretation before final fitting. A smoke simulation falsified the first
static early-bias candidate; the design history records the amendment that restricted the
feature to the forecast phase and added a generic late-phase control.

## Prospective result

<figure class="doc-figure doc-figure--wide" data-figure-kind="Literature-shaped analysis">
  <img src="../../assets/cell2025-forecast.svg" alt="Six model log-loss estimates with animal-bootstrap intervals and three pairwise early-bias contrasts. Early bias clearly improves on complete pooling, while comparisons with a late-phase control and hierarchical smooth trajectory cross zero.">
  <figcaption><strong>Literature-shaped analysis · forecastability without overclaiming model identity.</strong> Early bias has the lowest mean log loss, but only its improvement over the pooled psychometric model is resolved by the frozen animal bootstrap. This new analysis has no claimed paper-panel counterpart.<span class="doc-figure__meta"><strong>Unit:</strong> animal · <strong>n:</strong> 30 animals, 73,042 trials, 6 folds · <strong>Estimand:</strong> animal-balanced final-five-session choice log loss with animal-bootstrap intervals · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

| Candidate | Animal-balanced log loss | 95% interval |
| --- | ---: | ---: |
| Early-bias forecast | **0.58109** | 0.55265–0.61128 |
| Hierarchical smooth trajectory | 0.58402 | 0.55450–0.61406 |
| Late-phase psychometric | 0.59672 | 0.57045–0.62343 |
| Shared smooth trajectory | 0.59796 | 0.57175–0.62432 |
| Pooled psychometric | 0.62327 | 0.60995–0.63723 |
| Static partial pooling | 0.67343 | 0.63485–0.71855 |

Every fit passes its numerical audit. Relative to complete pooling, early bias improves
log loss by `0.04219` (95% paired interval `0.01818–0.06425`). The added value beyond a
generic late-phase change is `0.01563`, but its interval crosses zero
(`−0.00514–0.03636`). Early bias and the hierarchical smooth trajectory are likewise
unresolved (`early bias − hierarchical = −0.00293`, interval
`−0.01880–0.01283`).

The defensible conclusion is therefore narrow: behaviour available by day 8 forecasts
later choices better than a stationary pooled psychometric curve. This dataset does not
select uniquely between the predeclared early-bias summary and a general individual smooth
trajectory.

## Can this design distinguish its explanations?

Recovery reuses all 73,042 trial positions, contrasts, animal identities, session
coordinates, and six forecast folds. This matters: generic toy recovery would not audit
the actual claim.

<figure class="doc-figure doc-figure--wide" data-figure-kind="Synthetic benchmark">
  <img src="../../assets/cell2025-recovery.svg" alt="A structural model-recovery matrix with 11 of 12 correct selections and stacked bars showing early-bias feature selection in null, predictive, and reward-history simulations.">
  <figcaption><strong>Synthetic benchmark · exact-design falsification.</strong> Stable heterogeneity, shared drift, and individual drift recover in all three repeats. Complete pooling is mistaken for shared drift once. The outcome-derived feature is sensitive but not perfectly specific; the result is conditional on the tested generators.<span class="doc-figure__meta"><strong>Unit:</strong> simulation repeat · <strong>n:</strong> 12 structural and 36 feature-recovery simulations · <strong>Estimand:</strong> generating-family and feature-selection proportion · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

Structural recovery resolves all 12 simulations and selects the generating family in 11.
The single failure selects shared smooth drift under a complete-pooling generator. The
hierarchical path model converges in all three parameter-recovery repeats; mean population
path RMSE is `0.168`, mean individual-path RMSE is `0.313`, and coefficient-wise
individual-path correlations range from `0.881` to `0.984`.

The early-bias feature is recomputed from simulated days 4–8 rather than copied from the
observed data. It wins 12/12 simulations when early context truly determines late
asymmetry. Under a null with no animal signal, the simpler late-phase control wins 10/12,
leaving a visible 2/12 false-selection rate. It also wins 10/12 under a symmetric
reward-history generator with no stable animal trait. The two early-bias selections in
that world have tiny improvements (at most `0.000024` log loss), but remain visible rather
than being relabeled as successes.

Recovery shows what this design can discriminate under its specified generators. It does
not prove that any generator is the biological truth.

## Reproduce the study

```bash
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025_flagship.fetch_released_artifacts
uv run python -m benchmarks.cell2025_flagship.benchmark
uv run --group docs python -m scripts.plot_documentation_figures
```

The full study is intentionally compute-heavy. The exact released trajectory visualization
uses a small pinned compatibility environment documented in the
[benchmark README](https://github.com/aeronjl/unspool/tree/main/benchmarks/cell2025_flagship).
Raw data stay outside Git; committed JSON retains source checksums, denominators, folds,
seeds, fit audits, subject-balanced scores, recovery runs, and limitations.

## What remains outside the claim

- The forecast is internally validated within one 30-animal, one-protocol cohort. It has
  not been transported to a new lab, task, species, or acquisition batch.
- The historical-cohort design assumes reference and future animals are exchangeable.
- The released trajectory labels are retrospective visual aids, not predicted outcomes.
- Forecastability does not establish that early bias causes late strategy.
- Behaviour-only evidence cannot identify dopamine as necessary, sufficient, or the
  carrier of a teaching signal.
- The released Q-value summary has not yet been reimplemented as a general Unspool agent;
  the current `BinaryQLearning` has a different scientific contract.
- The public table cannot reproduce the paper's video, neural, photometry, causal, or
  deep-network analyses.
