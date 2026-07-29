# IBL 2021 replicated-lab endpoint-window benchmark

This benchmark turns a fixed International Brain Laboratory behavioural release into a
reusable, checksum-pinned longitudinal `Study`. It is the first public-data application of
Behavio's general ONE adapter and the first IBL panel here in which every lab contains
multiple animals.

## Selection contract

The committed manifest is built from release tag `2021_Q1_IBL_et_al_Behaviour`. It retains
every animal with a first `biasedChoiceWorld` transition and at least six earlier
`trainingChoiceWorld` sessions containing a `trials.table` dataset, then keeps the first
three and final three pre-transition training sessions. Every observed lab must contribute
at least four eligible animals.

This selection is trial-outcome-blind: it never reads choice, feedback, reward, or accuracy.
It is nevertheless transition-conditioned. The protocol transition is a training-policy
landmark and may itself depend on past performance, so improvement around it is a positive
control for chronology and retrieval—not an unbiased estimate of learning in a randomly
sampled population.

The resulting panel contains 78 animals from nine labs, 468 sessions, and 260,833 trials.
The manifest pins every session UUID, trial-table UUID, relative path, byte size, and MD5.
The reusable adapter asks ONE for those exact dataset UUIDs with hash checking and retains
the same provenance on every trial.

## Clock and comparison boundary

`window_position = 0, ..., 5` is an ordinal endpoint-window coordinate. The gap between
positions 2 and 3 varies by animal, so it is not elapsed training time and is not a full
learning trajectory. Original `session_order` remains available for analyses that model
that gap explicitly.

The trajectory report compares fixed-lab descriptive geometry and bootstraps animals
within each lab. Because the nine laboratories are not sampled from a defined population
of labs, its intervals do not license population-of-laboratories inference or causal lab
effects. Nine leave-one-lab-out folds instead provide the prospective population-validation
boundary for later models.

## Pinned result

All 78 animals improve descriptively from their mean first-three easy-trial accuracy to
their mean final-three training accuracy. The subject-weighted mean rises from `0.49066` to
`0.91347`, a change of `+0.42281`. That strong result should be read in light of the
transition-conditioned selection above.

Run the exact public-data benchmark with:

```bash
uv run --extra ibl python -m benchmarks.ibl2021_replicated.benchmark
```

The first run downloads about 21 MB of checksum-pinned Parquet tables into the ignored
benchmark cache. The committed `result.json` makes later library, release-cache, and source
schema changes visible in ordinary tests without requiring network access.
