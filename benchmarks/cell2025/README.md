# Cell 2025 longitudinal-behaviour benchmark

This benchmark reproduces one bounded result from Figure 1 of Liebana, Laffere et al.,
“Dopamine encodes deep network teaching signals for individual learning trajectories,”
*Cell* 188 (2025), 3789–3805.e33
([article DOI](https://doi.org/10.1016/j.cell.2025.05.025)). It tests the paper's central
behavioural observation that an animal's early strategy predicts its later strategy.

This is deliberately not a wholesale port of the paper analysis. The code is an independent,
small reimplementation of the documented trial exclusions and Figure 1 behavioural metrics.
It maps the retained trials into Unspool's canonical `Study`, preserves subject/session
boundaries, and then calculates session-level bias, psychometric slopes, and accuracy.
The source contains two dated sessions for DAP021 both labelled `sessionNum == 1`.
Unspool preserves these as distinct canonical sessions with chronological orders; only the
paper-specific summary deliberately groups them under the published session number.

## Reproduce

The [public data archive](https://doi.org/10.6084/m9.figshare.28877912.v1) is an 11.3 GB
ZIP. The fetcher uses HTTP byte ranges to download only its 101.6 MB behaviour CSV (27.3 MB
compressed) and verifies the inner file's SHA-256 digest.

```bash
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025.benchmark \
  benchmarks/cell2025/data/long_term_learning_dataset_preprocessed_behaviour_all.csv
```

The input is ignored by Git. The small, machine-readable output from the verified run is
committed as [`result.json`](result.json).

## Numerical contract

The benchmark follows the Figure 1 definitions in the paper and released analysis:

- exclude no-go and repeat trials, trials with within-session response-time z-score at or
  above 2, shaped animals, and non-learner animals;
- retain the 30 mice observed from the first two sessions onward;
- define early bias over days 4–8 and late quantities over each animal's final five sessions;
- calculate Pearson correlations for early versus late bias (Figure 1G) and early bias
  versus late right-minus-left psychometric slope (Figure 1I).

The checksum-pinned run must yield 192,238 trials, 950 canonical source sessions, and 949
paper-session summaries. Its principal result is
`r = 0.6947896564`, `p = 2.0429e-05` for early bias versus late slope asymmetry. It also
recovers the complementary bias reversal (`r = -0.5276390752`, `p = 0.0027312`) and the
mean accuracy increase from 0.51734 in the first session to 0.75803 in the last.

Floating-point values use relative tolerance `1e-9` and absolute tolerance `1e-12`; counts
must match exactly. These are regression tolerances, not uncertainty intervals. Statistical
uncertainty remains the one reported by the analysis, and the benchmark does not establish
causality, recover the trajectory clusters, or validate the paper's neural claims.

## Provenance and licensing

- Data: Figshare article `10.6084/m9.figshare.28877912.v1`, file `54186326`, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Analysis reference: Figshare article
  [`10.6084/m9.figshare.28877942.v1`](https://doi.org/10.6084/m9.figshare.28877942.v1),
  MIT licensed.
- Unspool code: independently reimplemented rather than copied from the released notebook.

The input member is
`data/long_term_learning_dataset_preprocessed_behaviour_all.csv`, with SHA-256
`94a6d541bfde731f769e02a68dbc652ab5b73dbc1ec13b8b7c8100d181b8048a`.
