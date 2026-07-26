"""Fetch one file from the versioned Cell 2025 Figshare archive.

The full public ZIP is 11.3 GB. Figshare supports HTTP byte ranges, so this utility reads
the ZIP directory remotely and transfers only the 27.3 MB compressed behaviour member.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE_ARTICLE_DOI = "10.6084/m9.figshare.28877912.v1"
FIGSHARE_FILE_ID = 54186326
FIGSHARE_FILE_URL = f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}"
FIGSHARE_ZIP_SIZE = 11_278_756_453
ARCHIVE_MEMBER = "data/long_term_learning_dataset_preprocessed_behaviour_all.csv"
MEMBER_SIZE = 101_624_391
MEMBER_SHA256 = "94a6d541bfde731f769e02a68dbc652ab5b73dbc1ec13b8b7c8100d181b8048a"
DEFAULT_DESTINATION = Path(__file__).with_name("data") / Path(ARCHIVE_MEMBER).name


class RemoteRangeFile(io.RawIOBase):
    """A seekable, read-only view of a remote object with HTTP range support."""

    def __init__(self, url: str, size: int) -> None:
        self._url = url
        self._size = size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self._position >= self._size:
            return b""
        end = self._size - 1 if size < 0 else min(self._size - 1, self._position + size - 1)
        request = urllib.request.Request(
            self._url,
            headers={"Range": f"bytes={self._position}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            content_range = response.headers.get("Content-Range")
            if content_range is None:
                raise RuntimeError("the data host ignored the required HTTP Range header")
            data = response.read()
        self._position += len(data)
        return data


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(destination: Path, *, force: bool = False) -> Path:
    """Fetch and verify the exact public behaviour CSV used by the benchmark."""

    destination = destination.resolve()
    if destination.exists():
        if sha256(destination) == MEMBER_SHA256:
            return destination
        if not force:
            raise FileExistsError(
                f"{destination} exists but has the wrong checksum; pass --force to replace it"
            )

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
            remote = RemoteRangeFile(FIGSHARE_FILE_URL, FIGSHARE_ZIP_SIZE)
            with zipfile.ZipFile(remote) as archive:
                member = archive.getinfo(ARCHIVE_MEMBER)
                if member.file_size != MEMBER_SIZE:
                    raise RuntimeError(
                        f"archive member size changed: {member.file_size} != {MEMBER_SIZE}"
                    )
                with archive.open(member) as source:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)

        observed = sha256(temporary_path)
        if observed != MEMBER_SHA256:
            raise RuntimeError(
                f"download checksum mismatch: observed {observed}, expected {MEMBER_SHA256}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="replace a mismatched file")
    args = parser.parse_args()
    path = fetch(args.destination, force=args.force)
    print(path)


if __name__ == "__main__":
    main()
