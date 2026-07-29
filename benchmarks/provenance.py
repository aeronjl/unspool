"""Environment provenance recorded inside every committed benchmark result.

The stamp answers one question: which code and which library versions produced the
numbers stored beside it. It deliberately carries no wall-clock timestamp, so a
re-run on an unchanged tree at unchanged library versions produces a byte-identical
payload and a genuine numerical drift is never hidden behind a changed date.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
PROVENANCE_KEY = "provenance"
SCHEMA_VERSION = 1
BASE_LIBRARIES = ("numpy", "scipy", "unspool")
UNKNOWN_REVISION = "unknown"
MISSING_VERSION = "not installed"


def git_revision(root: Path = ROOT) -> str:
    """Return ``git describe --always --dirty`` for the working tree, or ``unknown``."""

    try:
        completed = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_REVISION
    return completed.stdout.strip() or UNKNOWN_REVISION


def library_versions(libraries: Iterable[str]) -> dict[str, str]:
    """Return installed distribution versions for the libraries a result depends on."""

    versions: dict[str, str] = {}
    for name in sorted(set(libraries)):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = MISSING_VERSION
    return versions


def environment(*, libraries: Iterable[str] = ()) -> dict[str, Any]:
    """Return the provenance block written beside a benchmark result."""

    return {
        "schema_version": SCHEMA_VERSION,
        "git_describe": git_revision(),
        "python": platform.python_version(),
        "libraries": library_versions((*BASE_LIBRARIES, *libraries)),
    }


def stamp(result: Mapping[str, Any], *, libraries: Iterable[str] = ()) -> dict[str, Any]:
    """Return ``result`` with a provenance block attached."""

    if PROVENANCE_KEY in result:
        raise ValueError(f"result already carries a {PROVENANCE_KEY!r} block")
    return {**result, PROVENANCE_KEY: environment(libraries=libraries)}


def render(
    result: Mapping[str, Any],
    *,
    libraries: Iterable[str] = (),
    allow_nan: bool = False,
    sort_keys: bool = True,
) -> str:
    """Serialize a stamped result exactly as it is committed."""

    stamped = stamp(result, libraries=libraries)
    return json.dumps(stamped, indent=2, sort_keys=sort_keys, allow_nan=allow_nan) + "\n"


def write_result(
    path: Path,
    result: Mapping[str, Any],
    *,
    libraries: Iterable[str] = (),
    allow_nan: bool = False,
    sort_keys: bool = True,
) -> str:
    """Write a stamped result to ``path`` and return the rendered text."""

    rendered = render(result, libraries=libraries, allow_nan=allow_nan, sort_keys=sort_keys)
    Path(path).write_text(rendered, encoding="utf-8")
    return rendered
