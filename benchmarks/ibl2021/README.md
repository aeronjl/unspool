# IBL 2021 public learning benchmark

This benchmark exercises Unspool's longitudinal data contract on the International Brain
Laboratory's fixed `2021_Q1_IBL_et_al_Behaviour` release. It makes one deliberately
bounded claim: in a metadata-selected panel spanning all nine contributing labs, accuracy
on easy trials is higher in the final three training sessions before the first biased-task
transition than in the first three training sessions.

The release contains behaviour collected throughout learning, including stimuli,
decisions, and response times. The associated paper analysed learning across a much larger
cohort; this 54-session panel is an engineering benchmark, not a reproduction of every
animal or every analysis in that paper.

## Run it

The trial tables are about 3 MB in total. They are fetched directly from the IBL public S3
bucket and checked against the file sizes and MD5 hashes returned by OpenAlyx:

```bash
uv run python -m benchmarks.ibl2021.fetch_data
uv run --with pyarrow python -m benchmarks.ibl2021.benchmark
```

The downloader uses only the standard library. PyArrow is an on-demand benchmark
dependency and is intentionally not part of Unspool's NumPy/SciPy core.

Maintainers can regenerate the committed manifest from the fixed release tag with:

```bash
uv run --with one-api python -m benchmarks.ibl2021.refresh_manifest
```

A changed manifest digest requires review and an explicit update to the pinned regression
contract.

## Selection contract

Selection uses session metadata and trial-table availability, never trial-level choices,
rewards, or the accuracy contrast reported below:

1. retain `trainingChoiceWorld` and `biasedChoiceWorld` sessions in the fixed release that
   have a `trials.table` dataset;
2. require a first biased-task transition and at least six preceding training sessions;
3. within each lab, select the subject with the greatest number of pre-transition training
   sessions, breaking ties by total eligible task-session coverage and then subject ID;
4. retain the first three and final three pre-transition training sessions.

This produces disjoint, chronologically ordered early and late-training windows for one
subject in each of nine labs. The exact 54 session IDs, dataset IDs, URLs, sizes, hashes,
protocols, transition landmarks, and source session orders live in
[`manifest.json`](manifest.json). Its canonical session-record SHA-256 is
`63ac8b40b35ea21bc036cbdb4819f2dc04b448c803f804361dc9e40768aa32a0`.

## Numerical contract

The adapter maps all 28,400 source trials into `Study`, preserving subject, session,
within-session trial position, release chronology, lab, task protocol, and phase. Accuracy
is the rewarded fraction among valid-choice trials with absolute signed contrast at least
0.5. Session accuracies are averaged within each three-session phase and then equally
across subjects.

The verified result is:

- early easy-trial accuracy: `0.423900`;
- late-training easy-trial accuracy: `0.853332`;
- paired descriptive change: `+0.429431`;
- subjects with a positive change: `9 / 9`.

The machine-readable result is committed in [`result.json`](result.json).

## Interpretation boundary

The transition is itself performance-gated in the training pipeline, and both transition
status and the number of preceding training sessions enter the selection rule. The
accuracy increase is therefore a positive-control check for data retrieval, session
identity, chronological alignment, phase construction, and metric calculation. It is not
an unbiased estimate of population learning, a comparison among labs, or evidence for a
cognitive model. Those claims require broader inclusion and held-out subject/lab analyses.

## Sources and licensing

- IBL, “A standardized and reproducible method to measure decision-making in mice,”
  *eLife* (2021), <https://doi.org/10.7554/eLife.63711>.
- IBL behaviour release documentation:
  <https://docs.internationalbrainlab.org/notebooks_external/2021_data_release_behavior.html>.
- Public bucket and license record: <https://registry.opendata.aws/ibl-behaviour/>.
- ONE public-data quickstart:
  <https://docs.internationalbrainlab.org/notebooks_external/one_quickstart.html>.

The IBL data are distributed under CC BY 4.0 and are fetched on demand rather than
redistributed by Unspool. Benchmark code is covered by Unspool's MIT license.
