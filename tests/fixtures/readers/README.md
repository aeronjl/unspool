# Upstream behaviour-tool reader fixtures

These files are retained only for format-parity tests and are not included in the
behavio package wheel. `manifest.json` records their exact upstream path,
commit, checksum and license.

They moved here from `fipha` (`tests/fixtures/interoperability/`) with the
readers they pin, unchanged byte for byte; their checksums are still asserted
against `manifest.json`.

- `sleap-small-robot.analysis.h5` comes from the SLEAP test suite and retains its
  BSD-3-Clause-Clear terms.
- `boris-test-export-events-tabular.csv` comes from the BORIS test suite and
  retains its GPL-3.0-only terms.
- `deeplabcut-3.0.0-{single,multi}.h5` contain project-owned synthetic values
  serialized with DeepLabCut v3.0.0's exact prediction-writer contract.
- `keypoint-moseq-0.6.8-results.h5` contains project-owned synthetic values
  serialized by the pinned upstream `save_hdf5` implementation.
- `sleap-io-0.9.2-standard.analysis.h5` is the current standard-preset export of
  SLEAP-IO's official centered-pair prediction fixture.
- `boris-9.13.0-aggregated.tsv` is BORIS's official aggregated-event test export
  and retains its GPL-3.0-only terms.

Do not replace a fixture with a newly downloaded or regenerated file under the same name.
Update the manifest, expected semantics and compatibility documentation together.
