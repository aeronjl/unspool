"""Fetch the checksum-pinned trial tables for the IBL 2021 learning benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from benchmarks.ibl2021.selection import manifest_digest

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
DEFAULT_DESTINATION = Path(__file__).with_name("data")
EXPECTED_MANIFEST_SHA256 = "63ac8b40b35ea21bc036cbdb4819f2dc04b448c803f804361dc9e40768aa32a0"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate the committed selection and dataset manifest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("sessions")
    if not isinstance(rows, list):
        raise ValueError("manifest sessions must be a list")
    observed = manifest_digest(rows)
    recorded = manifest.get("sessions_sha256")
    if observed != recorded or observed != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "manifest checksum mismatch: "
            f"observed {observed}, recorded {recorded}, expected {EXPECTED_MANIFEST_SHA256}"
        )
    if len(rows) != 54:
        raise ValueError(f"manifest must contain 54 sessions, observed {len(rows)}")
    return manifest


def md5(path: Path) -> str:
    """Return a file's MD5 digest, matching the public dataset metadata."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, row: dict[str, Any]) -> bool:
    return (
        path.is_file() and path.stat().st_size == int(row["file_size"]) and md5(path) == row["md5"]
    )


def fetch(destination: Path = DEFAULT_DESTINATION, *, force: bool = False) -> Path:
    """Fetch and verify every exact trial table in the committed manifest."""

    manifest = load_manifest()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(manifest["sessions"], start=1):
        target = destination / f"{row['session']}.pqt"
        if verify_file(target, row):
            continue
        if target.exists() and not force:
            raise FileExistsError(
                f"{target} exists but does not match the manifest; pass --force to replace it"
            )
        _download(row["data_url"], target)
        if not verify_file(target, row):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"download verification failed for session {row['session']}")
        print(f"[{position:02d}/54] {row['lab']}/{row['subject']} {row['phase']}")
    return destination


def _download(url: str, destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            request = urllib.request.Request(url, headers={"User-Agent": "unspool-benchmark/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="replace mismatched local files")
    args = parser.parse_args()
    print(fetch(args.destination, force=args.force))


if __name__ == "__main__":
    main()
