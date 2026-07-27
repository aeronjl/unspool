"""Fetch only the small released Cell 2025 behavioural result artifacts."""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from benchmarks.cell2025.fetch_data import (
    FIGSHARE_FILE_URL,
    FIGSHARE_ZIP_SIZE,
    RemoteRangeFile,
    sha256,
)


@dataclass(frozen=True)
class ReleasedArtifact:
    member: str
    size: int
    sha256: str


ARTIFACTS = (
    ReleasedArtifact(
        "data/psych_metric_trajectory_fit_df.csv",
        567_103,
        "e5cd064e575fe792a1e2b265ef05f7fb23daa68131fea0e230a2ad2a5b6664ee",
    ),
    ReleasedArtifact(
        "data/first_5_session_action_value_model.pickle",
        4_326_708,
        "ba69393ca8ceb8932c77958ba66f27d1c14089684adbb0fd32a38f0e27daee5e",
    ),
    ReleasedArtifact(
        "data/left_right_balanced_cluster_df.csv",
        1_238,
        "6cd23f17c6634532bab1af5fd11ed212c8d9431a81a80c8ad28261fb63ab49dc",
    ),
)
DEFAULT_DIRECTORY = Path(__file__).with_name("data") / "released"


def fetch(directory: Path, *, force: bool = False) -> tuple[Path, ...]:
    """Range-fetch and checksum-verify the declared archive members."""

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    remote = RemoteRangeFile(FIGSHARE_FILE_URL, FIGSHARE_ZIP_SIZE)
    outputs: list[Path] = []
    with zipfile.ZipFile(remote) as archive:
        for artifact in ARTIFACTS:
            destination = directory / Path(artifact.member).name
            if destination.exists() and sha256(destination) == artifact.sha256:
                outputs.append(destination)
                continue
            if destination.exists() and not force:
                raise FileExistsError(
                    f"{destination} exists but has the wrong checksum; pass --force to replace it"
                )
            info = archive.getinfo(artifact.member)
            if info.file_size != artifact.size:
                raise RuntimeError(
                    f"archive member size changed: {info.file_size} != {artifact.size}"
                )
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=directory,
                    prefix=f".{destination.name}.",
                    suffix=".part",
                    delete=False,
                ) as target:
                    temporary_path = Path(target.name)
                    with archive.open(info) as source:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                observed = sha256(temporary_path)
                if observed != artifact.sha256:
                    raise RuntimeError(
                        f"download checksum mismatch: observed {observed}, "
                        f"expected {artifact.sha256}"
                    )
                os.replace(temporary_path, destination)
                temporary_path = None
                outputs.append(destination)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    for path in fetch(arguments.directory, force=arguments.force):
        print(path)


if __name__ == "__main__":
    main()
