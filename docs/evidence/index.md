# Evidence index

Unspool keeps scientist-facing interpretation separate from machine-readable evidence.
Worked studies explain the question and limitations; benchmark directories retain exact
protocol code, source checksums, seeds, fold provenance, audits, and JSON results.

| Evidence family | Public or controlled input | Primary contract |
| --- | --- | --- |
| Cell 2025 reproduction | Public Figshare trial table | Published numerical reproduction |
| Cell 2025 flagship forecast | 30 animals; first eight days and final five sessions | Historical-cohort forecasting, exact-design recovery, and competing explanations |
| IBL 2021 cohort | Exact ONE trial-table UUIDs | Outcome-blind selection and chronology |
| IBL prospective comparison | 78 animals, nine labs | Same-animal and held-out-lab forecasting |
| IBL nested selection | Same public panel | Training-only model and hyperparameter selection |
| Four-family recovery | Controlled simulation | Design-specific model discrimination |
| Trajectory recovery | Controlled simulation | Population and individual path recovery |
| DDM recovery suite | Controlled simulation | Joint choice/RT recovery and robustness |
| NWB/DANDI interoperability | Versioned public Dandiset | Trial identity and provenance |

Browse the complete [benchmark directory on GitHub](https://github.com/aeronjl/unspool/tree/main/benchmarks).
Failed fits, warning audits, unresolved selections, and boundary estimates remain part of
the record rather than being filtered out of summary pages.

Final protocol studies can be packaged as deterministic, content-addressed archives. Read
the [evidence-bundle anatomy, integrity, replay, and comparison guide](../protocols/evidence-bundles.md).
