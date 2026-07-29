# NWB/DANDI interoperability

This benchmark streams the trial table from one immutable NWB asset in published Dandiset
[`000004`](https://doi.org/10.48324/dandi.000004/0.220126.1852). It tests the data boundary,
not a behavioral claim: source row order and trial IDs must survive, subject and session
must be explicit, cross-session chronology must be supplied rather than guessed, and every
trial must retain the exact DANDI version, asset path, asset ID, byte size, and SHA-256.

The chosen 72.6 MB asset contains 200 trials from one human recognition-memory session.
Behavio streams only the requested HDF5 datasets rather than downloading the full asset.
The resulting `Study` preserves 100 learning and 100 recognition trials, five balanced
stimulus categories, valid intervals, and the original `response_value` name. It does not
reinterpret that source field as a binary choice: its observed values range from 0 to 36.
The source's absolute `response_time` event is explicitly renamed `response_timestamp` so
it cannot be mistaken for the decision-duration column expected by Behavio's DDMs.

Run the pinned public check with the optional DANDI dependencies:

```bash
uv run --extra dandi python -m benchmarks.nwb_dandi_interoperability.benchmark
```

The machine-readable [`result.json`](result.json) records every contract decision. Local
NWB write/read round trips and PyNWB schema validation are covered separately by the test
suite because they do not require network access.
