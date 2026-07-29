# Reproducing Cell 2025 Figure 1H and 1J

!!! abstract "Panel-level released-fit replay"

    This worked example replays the released Gaussian-process trajectories and exact
    soft-DTW procedures behind Figure 1H and 1J of Liebana, Laffere et al. (*Cell*,
    2025). It preserves the two panels' distinct coordinate systems and makes the
    boundary between continuous animal diversity and three retrospective visual
    summaries explicit.

## Result first

<figure class="doc-figure doc-figure--wide" data-figure-kind="Released replay">
  <img src="../../assets/cell2025-trajectories.svg" alt="Two-panel replay of Cell Figure 1H and 1J showing 30 overlapping mouse learning trajectories, three soft-DTW centroids, and distinct session-by-asymmetry and right-by-left slope geometries.">
  <a class="doc-figure__full-resolution" href="../../assets/cell2025-trajectories.svg" target="_blank" rel="noopener">Open full-resolution Figure 1H/J ↗</a>
  <figcaption><strong>Released replay · Cell Figure 1H and 1J.</strong> Thin curves are the 30 checksum-pinned released Gaussian-process fits; thick curves replay the released soft-DTW centroids in each panel's original coordinates. Cyan and navy mark the naive and expert ends of training. The clusters summarize a continuum for visualization; they are not prospective labels or evidence for three biological kinds.<span class="doc-figure__meta"><strong>Unit:</strong> animal trajectory · <strong>n:</strong> 30 mice × 100 released interpolation points · <strong>Estimands:</strong> R-L slope over session and left-versus-right slope paths · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

The replay matches the released left, balanced, and right memberships for all 30 mice:
9, 10, and 11 animals respectively. That exact match is a software audit, not evidence
that three categories are the best scientific description. The paper itself describes
the trajectories as diverse and uses clustering to make the main trends easier to see.

## Two panels, two computations

The old compact documentation plot collapsed these panels into one normalized-progress
display. That retained the broad message but was not like-for-like. The repaired display
keeps the published geometries separate:

| Contract field | Figure 1H | Figure 1J |
| --- | --- | --- |
| x variable | Released interpolated session | Right psychometric slope |
| y variable | Right-minus-left psychometric slope | Left psychometric slope |
| Thin curves | One released GP path per mouse | The same released GP slopes in phase space |
| Thick curves | One new soft-DTW centroid fitted within each Figure 1J membership | Three jointly fitted soft-DTW centroids |
| Clustering input | `[session, right slope - left slope]` | `[right slope, left slope]` |
| Soft-DTW model | Three separate one-centroid fits, seed 0 | One three-centroid fit, seed 1 |
| Reference geometry | R-L = 0 | L = R and L = 1-R |
| Axis contract | session 0-25; ticks 1, 10, 20 | both slopes -0.35 to 1.02; equal aspect |

This distinction matters. The thick Figure 1H curves are **not** Figure 1J centroids
projected onto a different axis. The released notebook refits one soft-DTW centroid
inside each already assigned group using session and slope difference together. Behavio
does the same.

## What each encoding means

Each thin trajectory has one continuous `prop_below` colour, computed in the released
analysis from the animal's fitted left and right slopes. Green-leaning and purple-leaning
paths occupy opposite ends of that continuous mapping; orange lies near its middle. The
colour does not arise from the three-cluster fit.

In Figure 1J, each thick centroid becomes brighter from the naive to the expert end. The
released code calls this a change in hue, but it actually increases the HSV **value**
channel from 0 to 1 while holding hue and saturation fixed. The documentation therefore
calls it a learning-progress brightness gradient. In Figure 1H, the three thick centroids
use their groups' mean continuous colour without a gradient.

## Source trace

| Layer | Frozen source |
| --- | --- |
| Paper | [Cell article, Figure 1H and 1J](https://doi.org/10.1016/j.cell.2025.05.025), journal page 3790, CC BY 4.0 |
| Released analysis | [Figshare software v1](https://doi.org/10.6084/m9.figshare.28877942.v1), MIT license |
| Released code | [`behaviour.ipynb` at commit `2faa468`](https://github.com/SamuelLiebana/da_long_term_learning/blob/2faa4680d5e9c0d6a9df516e3dede8c641e39a72/scripts/behaviour.ipynb), cells 8 and 25 |
| GP trajectory table | `psych_metric_trajectory_fit_df.csv`, SHA-256 `e5cd06…664ee` |
| Released memberships | `left_right_balanced_cluster_df.csv`, SHA-256 `6cd23…49dc` |
| Machine-readable panel audit | [`figure1hj_audit.json`](https://github.com/aeronjl/behavio/blob/main/benchmarks/cell2025_flagship/figure1hj_audit.json) |
| Checked-in numerical replay | [`figure1hj_trajectories.json`](https://github.com/aeronjl/behavio/blob/main/benchmarks/cell2025_flagship/figure1hj_trajectories.json) |

The replay uses NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, scikit-learn 1.5.1,
and tslearn 0.6.3, matching the released environment. The numerical artifact is generated
twice and required to be byte-identical. Semantic labels are assigned with the released
anchor mice—DAP028 left, DAP009 balanced, and DAP110 right—then checked mouse by mouse
against the released membership table.

## Preserved and intentionally changed

| Preserved | Intentionally changed |
| --- | --- |
| All 30 released GP paths and all 100 interpolation points | Figure 1H and 1J are composed side by side in one SVG |
| Continuous five-anchor colour map and `prop_below` values | DejaVu Sans replaces the released Arial setting |
| Exact soft-DTW specifications, random seeds, and memberships | Line widths and markers are scaled to the documentation canvas |
| Published limits, ticks, reference lines, and equal phase-space aspect | Panel letters, concise titles, endpoint key, and centroid labels aid standalone reading |
| Cyan naive and navy expert endpoints | “Hue” is corrected to the implemented HSV brightness gradient |

The target is the scientific display contract rather than a pixel trace. These rendering
changes improve readability and provenance without changing a plotted trajectory,
centroid, membership, or axis meaning.

## Reproduce it

First fetch and verify the released artifacts:

```bash
uv run python -m benchmarks.cell2025_flagship.fetch_released_artifacts
```

Replay the old numerical stack in an isolated environment rather than constraining
Behavio's supported runtime:

```bash
uv venv --python 3.12 .venv-cell2025-release
uv pip install --python .venv-cell2025-release/bin/python \
  numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 \
  scikit-learn==1.5.1 tslearn==0.6.3
.venv-cell2025-release/bin/python \
  -m benchmarks.cell2025_flagship.released_figure1hj
uv run --group docs python -m scripts.plot_documentation_figures
uv run --group docs pytest tests/test_cell2025_figure1hj.py tests/test_documentation.py
```

The replay rejects source checksum changes, an altered 30 × 100 trajectory shape,
different animal identities, or any membership mismatch. Documentation tests reject a
missing panel, absent full-resolution route, serif typography, or an unregistered claim.

## Claim boundary

This is an exact replay of **released fitted trajectories and released clustering code**.
It is not an independent refit of the Gaussian processes from the public trial table.
Neither smooth paths nor retrospective soft-DTW centroids establish that learning is
generated by three discrete mechanisms. They also do not identify a dopaminergic cause;
that claim depends on experimental evidence elsewhere in the paper.

[Continue to the prospective Cell study](cell2025-learning-trajectories.md)
