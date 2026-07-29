"""Fetch the public IBL behaviour archive that Ashwood et al. (2022) modelled.

The archive is the International Brain Laboratory behavioural release deposited on
Figshare as ``10.6084/m9.figshare.11636748``. Ashwood's reference implementation
(``github.com/zashwood/glm-hmm``) downloads exactly this file, by numeric file id, in
``1_preprocess_data/ibl/1_download_data_begin_processing.py``.

Version 7 is the version the reference code resolves to, and it is pinned here by file id,
byte size, MD5 (as published by the Figshare API) and SHA-256 (computed locally on the
verified download). The archive is 218 MiB, so it is fetched whole rather than by HTTP
range: the benchmark reads roughly two thousand sessions scattered throughout the member
table, and a ranged read would be slower than a single transfer.

Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The depositor asks to be
cited as: International Brain Laboratory (2020). A standardized and reproducible method to
measure decision-making in mice: Data. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.11636748.v7
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

FIGSHARE_ARTICLE_DOI = "10.6084/m9.figshare.11636748.v7"
FIGSHARE_FILE_ID = 21623715
FIGSHARE_FILE_URL = f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}"
ARCHIVE_NAME = "ibl-behavior-data-Dec2019.zip"
ARCHIVE_SIZE = 228_602_597
ARCHIVE_MD5 = "fd219c14ff0f3caa88d5f8bed9a96443"
ARCHIVE_SHA256 = "18bfacccf615a767dd6e3935473b628fe4266e9b12c09200ee7f4eac2c54c4e6"
ARCHIVE_LICENCE = "CC BY 4.0"
DEFAULT_DESTINATION = Path(__file__).with_name("data") / ARCHIVE_NAME


def digest(path: Path, algorithm: str = "sha256") -> str:
    """Return a file's hex digest without loading it all into memory."""

    accumulator = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            accumulator.update(chunk)
    return accumulator.hexdigest()


def verify(path: Path) -> None:
    """Raise unless ``path`` is byte-for-byte the pinned Figshare archive."""

    size = path.stat().st_size
    if size != ARCHIVE_SIZE:
        raise RuntimeError(f"archive size mismatch: observed {size}, expected {ARCHIVE_SIZE}")
    observed_md5 = digest(path, "md5")
    if observed_md5 != ARCHIVE_MD5:
        raise RuntimeError(f"archive MD5 mismatch: observed {observed_md5}, expected {ARCHIVE_MD5}")
    observed_sha256 = digest(path)
    if observed_sha256 != ARCHIVE_SHA256:
        raise RuntimeError(
            f"archive SHA-256 mismatch: observed {observed_sha256}, expected {ARCHIVE_SHA256}"
        )


def fetch(destination: Path, *, force: bool = False) -> Path:
    """Fetch and verify the exact public archive the benchmark reads."""

    destination = destination.resolve()
    if destination.exists():
        try:
            verify(destination)
        except RuntimeError:
            if not force:
                raise FileExistsError(
                    f"{destination} exists but does not verify; pass --force to replace it"
                ) from None
        else:
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            with urllib.request.urlopen(FIGSHARE_FILE_URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        verify(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="replace a file that fails to verify")
    args = parser.parse_args()
    print(fetch(args.destination, force=args.force))


if __name__ == "__main__":
    main()
