# Behaviour-tool interoperability validation

!!! note "Moved from fipha"
    This matrix, and the checksum-pinned fixtures it describes, moved here from
    the photometry package `fipha` with the readers they validate. The fixture
    files and their expected values are unchanged; the `tests/fixtures/readers`
    manifest still asserts their checksums.

This report separates three different claims that are easy to conflate:

1. an adapter accepts an in-memory array shaped like a source tool;
2. a file reader accepts the source tool's documented on-disk schema; and
3. a checksum-pinned file serialized by a pinned upstream writer passes a
   semantic parity test; and
4. an official upstream artifact passes without recreating its payload.

The last two are real-file fixture results, but they carry different provenance.
A writer-contract artifact validates current serialization with declared values;
an official artifact additionally tests a payload retained by the source project.
Neither validates a biological analysis.

## Compatibility matrix

| Source | In-memory boundary | File reader | Upstream fixture | Current evidence |
|---|---|---|---|---|
| DeepLabCut | MultiIndex scorer/bodypart/x-y-likelihood; multi-animal identity may be selected explicitly | prediction CSV and pandas HDF5 | **Pass: v3.0.0 single/multi writer contract** | exact `save_data` HDF5 key/table contract; likelihood range and mandatory identity selection are checked |
| SLEAP | standard and MATLAB-compatible Analysis HDF5 axis orders | Analysis HDF5 with stored or explicitly declared dimensions | **Pass: official legacy plus current standard export** | legacy scores/axes and a real 1,100-frame SLP exported by sleap-io 0.9.2 both pass |
| Keypoint-MoSeq | frame-level integer syllable sequence to duration-preserving bouts | `results.h5/<recording>/syllable` | **Pass: 0.6.8 writer contract** | pinned `save_hdf5` output preserves syllables and exact bout boundaries |
| BORIS | aggregated POINT/STATE columns | aggregated CSV/TSV plus tabular CSV with START/STOP pairing | **Pass: official tabular and aggregated exports** | explicit subject/observation selection preserves both point events and state intervals |
| ndx-pose (in `fipha`) | `PoseEstimation` objects with 2D/3D series | NWB processing-module discovery and explicit estimator selection | **Pass: 0.3.0 extension writer contract** | schema-valid PyNWB round trip preserves conversion/offset, timestamps, confidence missingness, reference frames, skeleton and source metadata; unsupplied object links are reported |
| Behavio `Study` handoff | canonical trial-level column handoff | optional `Study` construction | **Pass: public cross-package benchmark** | explicit chronology, retained neural columns, fingerprint and prospective fold composition; owned by [`fipha.behavio`](https://aeronjl.github.io/fipha/behavio-interoperability/) |
| Clock synchronization | explicit one-to-one source/target pulse pairs | versioned JSON evidence artifact | **Synthetic only** | known affine offset/drift recovery, residual and drift refusal, bounded extrapolation, and pose/covariate/annotation composition are tested |

No adapter in this matrix now relies only on an in-test schema facsimile. The
DeepLabCut, Keypoint-MoSeq, and ndx-pose files are deliberately labelled
writer-contract artifacts because their values were declared for the test; SLEAP
and BORIS also have official upstream payloads. The ndx-pose fixture is generated
inside a temporary directory by the real 0.3.0 extension and validated by PyNWB,
rather than committed as another binary copy.

## Pinned artifacts

| Tool | Upstream artifact | Commit | SHA-256 |
|---|---|---|---|
| SLEAP | [`small_robot...analysis.h5`](https://github.com/talmolab/sleap/blob/8ab323e060d827adc03e629122723aa54e1ca950/tests/data/hdf5_format_v1/small_robot.000_small_robot_3_frame.analysis.h5) | `8ab323e060d8` | `21446732fe6a…6f82de` |
| SLEAP | [`centered_pair_predictions.slp`](https://github.com/talmolab/sleap-io/blob/80bbcd9c3b005a7616c14b875b761071f10ecde6/tests/data/slp/centered_pair_predictions.slp), exported with 0.9.2 standard preset | `80bbcd9c3b00` | `ff7af9f57d68…4fd3` |
| DeepLabCut | v3.0.0 `save_data` single- and multi-animal writer-contract files | `cdc87245bfa7` | `5b9e9baba3ef…74e1`, `9d744d672adc…fb0c` |
| Keypoint-MoSeq | 0.6.8 `save_hdf5` writer-contract `results.h5` | `3f0307548686` | `d9adcb6b096e…0969` |
| BORIS | [`test_export_events_tabular.csv`](https://github.com/olivierfriard/BORIS/blob/1f6149f68e7c4df28d92fb50a8f8a38ed7a377d2/tests/files/test_export_events_tabular.csv) | `1f6149f68e7c` | `54554a4c9cc3…7131442` |
| BORIS | [`test_export_aggregated_events_test_full_1.tsv`](https://github.com/olivierfriard/BORIS/blob/e7aadfd25a5311b76bc6357921da81d28cfce9ec/tests/files/test_export_aggregated_events_test_full_1.tsv) | `e7aadfd25a53` | `1eeb914f1b7a…a6e5a` |

The complete source URLs, repository heads, checksums, generation qualifications,
notes and licenses live in `tests/fixtures/interoperability/manifest.json`.
BORIS fixtures retain GPL-3.0-only terms, are used only for tests, and are excluded
from the package wheel.

## What the real files changed

### SLEAP scores are not universally probabilities

The legacy official fixture contains finite `point_scores` above one. The shared
`PoseTrajectory` therefore treats SLEAP confidence-like scores as non-negative,
not as probabilities. DeepLabCut's source-specific adapter continues to enforce
likelihood in `[0, 1]`.

Legacy SLEAP Analysis HDF5 also omits dimension attributes. Its reader refuses to
guess and requires:

```python
pose = pose_from_sleap_analysis_h5(
    path,
    subject="robot-1",
    session="three-frames",
    node="front",
    dims=("track", "xy", "node", "frame"),
    fps=25.0,
)
```

The current 0.9.2 standard-preset export carries JSON dimension names on the
`tracks` and score datasets; the 1,100-frame fixture confirms they are read
automatically. Exporting the upstream SLP also exposed a sleap-io provenance JSON
failure for `Path` values; fixture generation string-normalized provenance only.

### BORIS has two useful export shapes

BORIS has two native file readers. A tabular export contains a metadata preamble,
then individual START and STOP rows; its reader locates the semantic header, pairs
rows by behavior, and rejects invalid boundaries. An aggregated CSV/TSV contains
one complete POINT or STATE event per row; its reader preserves that distinction
directly. Both require explicit selection when observations or subjects are
ambiguous.

```python
annotations = annotations_from_boris_tabular_file(
    path,
    subject="canonical-mouse-1",
    session="observation-1",
    source_subject="subject1",
)

aggregated = annotations_from_boris_aggregated_file(
    aggregated_path,
    subject="canonical-mouse-1",
    session="observation-1",
    source_subject="subject1",
)
```

Blank or `POINT` status rows are point events. START/STOP rows are retained as
positive-duration intervals.

## Remaining validation work

1. Add a second released version for each writer so format drift becomes an
   executable compatibility claim rather than a one-version snapshot.
2. Add a model-produced Keypoint-MoSeq result retaining syllable-reindexing and
   fitted-model provenance; the current fixture validates serialization only.
3. Exercise every behavioural source against acquisition timestamps with a real
   synchronisation record; synthetic affine recovery and file parity do not
   validate acquisition-specific clock alignment.
4. Add a public ndx-pose file produced independently by DeepLabCut or SLEAP; the
   current fixture validates the released extension writer and round-trip semantics.
