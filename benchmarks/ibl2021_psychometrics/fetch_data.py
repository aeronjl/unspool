"""Pin and fetch the exact public IBL trial tables this replication reads.

Accession
---------
Release tag ``2021_Q1_IBL_et_al_Behaviour`` on the public Alyx instance
``https://openalyx.internationalbrainlab.org`` (password ``international``), the behaviour
release accompanying International Brain Laboratory et al. (2021), *eLife* 10:e63711,
<https://doi.org/10.7554/eLife.63711>.

Licence
-------
Creative Commons Attribution 4.0 International (CC-BY-4.0), as published by the
International Brain Laboratory for this release. Attribution is carried on every trial by
``unspool.adapters.ibl_one``, which stamps the Alyx origin, release tag, session UUID,
dataset UUID, relative path, byte size and MD5 onto each row.

Cohort
------
Selection is fixed by ``PROTOCOL.md`` and reads only protocol names and dates: every
subject whose earliest session in the release runs ``_iblrig_tasks_trainingChoiceWorld``,
and every ``trainingChoiceWorld`` session belonging to those subjects that carries an
``alf/_ibl_trials.table.pqt`` dataset. It never reads a choice, a reward or an accuracy, so
it cannot select for the outcome being replicated.

The manifest written here pins each session UUID, dataset UUID, relative path, byte size
and released MD5. ``benchmark.py`` requests those exact dataset UUIDs through ONE with hash
checking enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

RELEASE_TAG = "2021_Q1_IBL_et_al_Behaviour"
PUBLIC_BUCKET_URL = "https://ibl-brain-wide-map-public.s3.amazonaws.com/data"
DOWNLOAD_WORKERS = 16
PUBLIC_ALYX_URL = "https://openalyx.internationalbrainlab.org"
PUBLIC_PASSWORD = "international"
LICENCE = "CC-BY-4.0"
SOURCE_DOI = "10.7554/eLife.63711"
TRIALS_TABLE_PATH = "alf/_ibl_trials.table.pqt"
TRAINING_PROTOCOL = "trainingChoiceWorld"

#: ``paper_behavior_functions.CUTOFF_DATE``: subjects must reach criterion on or before this.
CUTOFF_DATE = "2020-03-23"
#: ``paper_behavior_functions.EXCLUDED_SESSIONS``.
EXCLUDED_SESSIONS = ("a9fb578a-9d7d-42b4-8dbc-3b419ce9f424",)

#: ``paper_behavior_functions.institution_map`` recovers the paper's seven institutions.
LAB_TO_INSTITUTION = {
    "angelakilab": "NYU",
    "churchlandlab": "CSHL",
    "cortexlab": "UCL",
    "danlab": "Berkeley",
    "hoferlab": "SWC",
    "mainenlab": "CCU",
    "mrsicflogellab": "SWC",
    "wittenlab": "Princeton",
    "zadorlab": "CSHL",
}
INSTITUTIONS = ("UCL", "CCU", "CSHL", "NYU", "Princeton", "SWC", "Berkeley")

DEFAULT_CACHE = Path(__file__).with_name("data")
MANIFEST_PATH = Path(__file__).with_name("manifest.json")


def manifest_digest(rows: list[dict[str, Any]]) -> str:
    """Return a stable SHA-256 over the pinned session rows."""

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def open_one(cache_directory: Path = DEFAULT_CACHE) -> Any:
    """Return a ONE client bound to this benchmark's own cache and release tag."""

    try:
        from one.api import ONE
    except ImportError as error:
        raise RuntimeError(
            "the IBL 2021 psychometrics benchmark requires `unspool[ibl]`"
        ) from error
    cache_directory.mkdir(parents=True, exist_ok=True)
    one = ONE(
        base_url=PUBLIC_ALYX_URL,
        password=PUBLIC_PASSWORD,
        silent=True,
        cache_dir=cache_directory,
    )
    one.load_cache(tag=RELEASE_TAG)
    return one


def build_manifest(cache_directory: Path = DEFAULT_CACHE) -> dict[str, Any]:
    """Select the cohort from the release index and pin every dataset it needs."""

    import pandas as pd

    one = open_one(cache_directory)
    sessions = one._cache["sessions"].copy()
    datasets = one._cache["datasets"].copy()

    sessions["protocol"] = sessions["task_protocol"].str.extract(r"tasks_([A-Za-z]+)")[0]
    sessions["date"] = pd.to_datetime(sessions["date"])
    sessions = sessions.sort_values(["subject", "date", "number"])

    earliest = sessions.groupby("subject")["protocol"].first()
    cohort = set(earliest[earliest == TRAINING_PROTOCOL].index)

    trials = datasets[datasets["rel_path"] == TRIALS_TABLE_PATH].reset_index()
    trials = trials.set_index("eid")[["id", "file_size", "hash"]]

    selected = sessions[
        sessions["subject"].isin(cohort) & (sessions["protocol"] == TRAINING_PROTOCOL)
    ].join(trials, how="inner")
    selected = selected[~selected.index.isin(EXCLUDED_SESSIONS)]

    rows: list[dict[str, Any]] = []
    for subject, group in selected.groupby("subject", sort=True):
        ordered = group.sort_values(["date", "number"])
        for order, (session_id, row) in enumerate(ordered.iterrows()):
            lab = str(row["lab"])
            date = row["date"].strftime("%Y-%m-%d")
            rows.append(
                {
                    "session_path": f"{lab}/Subjects/{subject}/{date}/{int(row['number']):03d}",
                    "session": str(session_id),
                    "dataset_id": str(row["id"]),
                    "dataset_path": TRIALS_TABLE_PATH,
                    "file_size": int(row["file_size"]),
                    "md5": str(row["hash"]),
                    "subject": str(subject),
                    "lab": lab,
                    "institution": LAB_TO_INSTITUTION[lab],
                    "date": date,
                    "number": int(row["number"]),
                    "session_order": order,
                    "task_protocol": str(row["task_protocol"]),
                }
            )
    rows.sort(key=lambda row: (row["subject"], row["session_order"]))
    return {
        "release_tag": RELEASE_TAG,
        "public_alyx_url": PUBLIC_ALYX_URL,
        "source_doi": SOURCE_DOI,
        "licence": LICENCE,
        "cutoff_date": CUTOFF_DATE,
        "selection_policy": (
            "Every subject whose earliest release session runs trainingChoiceWorld, and "
            "every trainingChoiceWorld session of those subjects carrying a trials table. "
            "Reads only protocol names and dates, never choices, feedback or accuracy."
        ),
        "n_subjects": len({row["subject"] for row in rows}),
        "n_sessions": len(rows),
        "total_file_size": sum(row["file_size"] for row in rows),
        "sessions_sha256": manifest_digest(rows),
        "sessions": rows,
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load the committed manifest and verify its pinned digest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("sessions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest sessions must be a non-empty list")
    if manifest_digest(rows) != manifest.get("sessions_sha256"):
        raise ValueError("IBL 2021 psychometrics manifest checksum mismatch")
    return manifest


def write_manifest(path: Path = MANIFEST_PATH, cache_directory: Path = DEFAULT_CACHE) -> Path:
    """Rebuild the manifest from the release index, pinning the served content digests."""

    manifest = build_manifest(cache_directory)
    summary = fetch(cache_directory, manifest=manifest, pin=True)
    manifest["n_index_content_mismatches"] = summary["n_index_content_mismatches"]
    manifest["index_content_mismatches"] = summary["index_content_mismatches"]
    manifest["sessions_sha256"] = manifest_digest(manifest["sessions"])
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_one(row: dict[str, Any], cache_directory: Path, *, pin: bool) -> tuple[str, int]:
    """Fetch one trial table and return the SHA of the bytes actually served.

    The release publishes two things that can disagree: the ``datasets`` index, which
    records a byte size and MD5 per dataset UUID, and the objects the bucket serves. For a
    minority of datasets they do not match, and ONE's own ``check_hash=True`` does not
    notice. This benchmark therefore pins the **served content** as ``content_md5`` in
    addition to the index's ``md5``, and enforces the served content once pinned.
    """

    destination = cache_directory / row["session_path"] / row["dataset_path"]
    expected = row.get("content_md5")
    if destination.is_file():
        observed = _md5(destination)
        if expected is None or observed == expected:
            return observed, destination.stat().st_size
    url = f"{PUBLIC_BUCKET_URL}/{row['session_path']}/alf/_ibl_trials.table.{row['dataset_id']}.pqt"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    observed = hashlib.md5(payload).hexdigest()
    if not pin and expected is not None and observed != expected:
        raise RuntimeError(
            f"served content changed for {row['dataset_id']}: observed {observed}, "
            f"pinned {expected}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pqt.part")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return observed, len(payload)


def fetch(
    cache_directory: Path = DEFAULT_CACHE,
    *,
    manifest: dict[str, Any] | None = None,
    pin: bool = False,
) -> dict[str, Any]:
    """Fetch every pinned trial table into the ONE cache layout.

    Returns a summary including how many served objects disagree with the release index.
    With ``pin=True`` the served digests are recorded rather than enforced, which is how
    ``--rebuild-manifest`` freezes them.
    """

    declared = load_manifest() if manifest is None else manifest
    rows = declared["sessions"]
    open_one(cache_directory)
    digests: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for index, digest in enumerate(
            pool.map(lambda row: _fetch_one(row, cache_directory, pin=pin), rows), start=1
        ):
            digests.append(digest)
            if index % 500 == 0:
                print(f"{index}/{len(rows)} trial tables verified", flush=True)
    mismatched = [
        row["dataset_id"]
        for row, (observed, size) in zip(rows, digests, strict=True)
        if observed != row["md5"] or size != row["file_size"]
    ]
    if pin:
        for row, (observed, size) in zip(rows, digests, strict=True):
            row["content_md5"] = observed
            row["content_size"] = size
    return {
        "n_sessions": len(rows),
        "n_index_content_mismatches": len(mismatched),
        "index_content_mismatches": sorted(mismatched),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="reselect the cohort from the release index and rewrite manifest.json",
    )
    args = parser.parse_args()
    if args.rebuild_manifest:
        print(write_manifest(cache_directory=args.cache))
    summary = fetch(args.cache)
    print(
        f"{summary['n_sessions']} pinned trial tables available in {args.cache}; "
        f"{summary['n_index_content_mismatches']} served objects disagree with the "
        "release index"
    )


if __name__ == "__main__":
    main()
