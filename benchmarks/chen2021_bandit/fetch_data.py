"""Fetch and safely extract the public Chen et al. (2021) restless-bandit data."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DATASET_DOI = "10.5061/dryad.z612jm6c0"
ZENODO_RECORD = 4_753_370
ARCHIVE_NAME = "cleaned_up_restless_final_data.zip"
ARCHIVE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}/content"
ARCHIVE_MD5 = "05f3797252b90c566d123f8b73422df6"
ARCHIVE_SHA256 = "90f0f9fa843a16788d0dcd7b857f81db068e8d18b8dd4eabf20ccaee3b67db04"
ARCHIVE_SIZE = 1_185_140
SOURCE_DIRECTORY = "cleaned up restless final data"
DEFAULT_DESTINATION = Path(__file__).with_name("data") / SOURCE_DIRECTORY


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(destination: Path = DEFAULT_DESTINATION, *, force: bool = False) -> Path:
    """Download, verify, and safely extract the exact public archive."""

    destination = destination.resolve()
    if destination.exists():
        if _complete_source_tree(destination):
            return destination
        if not force:
            raise FileExistsError(
                f"{destination} exists but is not a complete 8-session by 32-mouse tree; "
                "pass --force to replace it"
            )
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path: Path | None = None
    extraction_root: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{ARCHIVE_NAME}.",
            suffix=".part",
            delete=False,
        ) as target:
            archive_path = Path(target.name)
            request = urllib.request.Request(
                ARCHIVE_URL,
                headers={"User-Agent": "unspool-public-recipe/0.1"},
            )
            with urllib.request.urlopen(request, timeout=90) as source:
                shutil.copyfileobj(source, target)

        observed_size = archive_path.stat().st_size
        observed_sha256 = sha256(archive_path)
        if observed_size != ARCHIVE_SIZE or observed_sha256 != ARCHIVE_SHA256:
            raise RuntimeError(
                "download contract failed: "
                f"size={observed_size} (expected {ARCHIVE_SIZE}), "
                f"sha256={observed_sha256} (expected {ARCHIVE_SHA256})"
            )

        extraction_root = Path(
            tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
        )
        extracted_source = _safe_extract(archive_path, extraction_root) / SOURCE_DIRECTORY
        if not _complete_source_tree(extracted_source):
            raise RuntimeError("archive does not contain the expected 8-session by 32-mouse tree")
        os.replace(extracted_source, destination)
        return destination
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        if extraction_root is not None:
            shutil.rmtree(extraction_root, ignore_errors=True)


def _safe_extract(archive_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe ZIP member: {member.filename!r}")
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return destination


def _complete_source_tree(path: Path) -> bool:
    return path.is_dir() and all(
        (path / f"session{session}" / f"{mouse}.csv").is_file()
        for session in range(1, 9)
        for mouse in range(1, 33)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="replace an incomplete destination")
    args = parser.parse_args()
    print(fetch(args.destination, force=args.force))


if __name__ == "__main__":
    main()
