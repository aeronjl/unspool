"""Pinned DANDI asset resolution and streamed NWB trial import."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from unspool.adapters.nwb import study_from_nwbfile
from unspool.study import Study

DEFAULT_DANDI_API = "https://api.dandiarchive.org/api"


class DANDIAdapterError(ValueError):
    """Raised when a DANDI asset cannot satisfy a pinned NWB import request."""


@dataclass(frozen=True, slots=True)
class DANDINWBSource:
    """A version-pinned NWB asset and its explicit longitudinal identity."""

    dandiset_id: str
    version: str
    asset_path: str
    session_order: int
    subject: Any | None = None
    session: Any | None = None
    columns: tuple[str, ...] | None = None
    column_map: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dandiset_id, str)
            or len(self.dandiset_id) != 6
            or not self.dandiset_id.isdigit()
        ):
            raise ValueError("dandiset_id must contain exactly six digits")
        version_parts = self.version.split(".") if isinstance(self.version, str) else ()
        if (
            len(version_parts) != 3
            or not all(part.isdigit() for part in version_parts)
            or len(version_parts[1]) != 6
            or len(version_parts[2]) != 4
        ):
            raise ValueError("version must be a published DANDI version such as '0.220126.1852'")
        if (
            not isinstance(self.asset_path, str)
            or not self.asset_path
            or not self.asset_path.endswith(".nwb")
        ):
            raise ValueError("asset_path must identify an NWB file")
        if self.asset_path.startswith("/") or ".." in self.asset_path.split("/"):
            raise ValueError("asset_path must be a normalized relative DANDI path")
        if (
            isinstance(self.session_order, bool)
            or not isinstance(self.session_order, int)
            or self.session_order < 0
        ):
            raise ValueError("session_order must be a non-negative integer")
        if self.columns is not None:
            columns = tuple(self.columns)
            if not columns or any(not isinstance(name, str) or not name for name in columns):
                raise ValueError("columns must contain non-empty strings")
            if len(set(columns)) != len(columns):
                raise ValueError("columns must be unique")
            object.__setattr__(self, "columns", columns)
        if self.column_map is not None:
            mapping = dict(self.column_map)
            if any(
                not isinstance(source, str)
                or not source
                or not isinstance(target, str)
                or not target
                for source, target in mapping.items()
            ):
                raise ValueError("column_map must map non-empty strings to non-empty strings")
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("column_map targets must be unique")
            object.__setattr__(self, "column_map", MappingProxyType(mapping))


@dataclass(frozen=True, slots=True)
class ResolvedDANDIAsset:
    """Immutable provenance and content location for one resolved DANDI blob asset."""

    dandiset_id: str
    version: str
    path: str
    asset_id: str
    size: int
    sha256: str
    content_url: str

    def __post_init__(self) -> None:
        if not self.asset_id or self.size <= 0:
            raise ValueError("resolved DANDI asset identity and size must be present")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("resolved DANDI SHA-256 must contain 64 lowercase hexadecimal digits")
        if not self.content_url.startswith("https://"):
            raise ValueError("resolved DANDI content URL must use HTTPS")


def resolve_dandi_nwb_asset(
    source: DANDINWBSource,
    *,
    api_base: str = DEFAULT_DANDI_API,
    timeout: float = 30.0,
) -> ResolvedDANDIAsset:
    """Resolve an exact published DANDI path to immutable content metadata."""

    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be positive")
    api_base = api_base.rstrip("/")
    listing_url = (
        f"{api_base}/dandisets/{source.dandiset_id}/versions/{source.version}/assets/"
        f"?path={quote(source.asset_path, safe='')}"
    )
    listing = _read_json(listing_url, timeout=float(timeout))
    results = listing.get("results")
    if listing.get("count") != 1 or not isinstance(results, list) or len(results) != 1:
        raise DANDIAdapterError(
            f"expected exactly one DANDI asset at {source.asset_path!r}, "
            f"observed {listing.get('count')!r}"
        )
    row = results[0]
    if not isinstance(row, dict) or row.get("path") != source.asset_path:
        raise DANDIAdapterError("DANDI asset response did not preserve the requested path")
    if row.get("zarr") is not None or row.get("blob") is None:
        raise DANDIAdapterError("streamed trial import currently requires a blob-backed NWB asset")
    asset_id = row.get("asset_id")
    metadata = _read_json(f"{api_base}/assets/{asset_id}/", timeout=float(timeout))
    digest = metadata.get("digest")
    content_urls = metadata.get("contentUrl")
    if not isinstance(digest, dict) or not isinstance(content_urls, list):
        raise DANDIAdapterError("DANDI asset metadata omitted digest or content URLs")
    sha256 = digest.get("dandi:sha2-256")
    content_url = next(
        (
            value
            for value in content_urls
            if isinstance(value, str) and value.startswith("https://dandiarchive.s3.amazonaws.com/")
        ),
        None,
    )
    if not isinstance(asset_id, str) or not isinstance(sha256, str) or content_url is None:
        raise DANDIAdapterError("DANDI asset metadata omitted immutable blob provenance")
    try:
        size = int(row["size"])
    except (KeyError, TypeError, ValueError):
        raise DANDIAdapterError("DANDI asset response omitted a valid byte size") from None
    return ResolvedDANDIAsset(
        dandiset_id=source.dandiset_id,
        version=source.version,
        path=source.asset_path,
        asset_id=asset_id,
        size=size,
        sha256=sha256,
        content_url=content_url,
    )


def study_from_dandi(
    source: DANDINWBSource,
    *,
    api_base: str = DEFAULT_DANDI_API,
    timeout: float = 30.0,
) -> Study:
    """Stream one DANDI NWB asset and copy only its selected trial-table columns."""

    asset = resolve_dandi_nwb_asset(source, api_base=api_base, timeout=timeout)
    try:
        import h5py
        import remfile
        from pynwb import NWBHDF5IO
    except ImportError as error:
        raise RuntimeError(
            "DANDI streaming requires optional dependencies; install `unspool[dandi]`"
        ) from error
    remote = remfile.File(asset.content_url)
    with (
        closing(remote),
        h5py.File(remote, mode="r") as h5_file,
        NWBHDF5IO(file=h5_file, load_namespaces=True) as io,
    ):
        return study_from_nwbfile(
            io.read(),
            session_order=source.session_order,
            subject=source.subject,
            session=source.session,
            columns=source.columns,
            column_map=source.column_map,
            source_columns={
                "source_dandiset_id": asset.dandiset_id,
                "source_dandi_version": asset.version,
                "source_dandi_asset_path": asset.path,
                "source_dandi_asset_id": asset.asset_id,
                "source_dandi_asset_size": asset.size,
                "source_dandi_sha256": asset.sha256,
            },
        )


def _read_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "unspool/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except Exception as error:
        raise DANDIAdapterError(f"DANDI API request failed for {url!r}: {error}") from error
    if not isinstance(value, dict):
        raise DANDIAdapterError(f"DANDI API returned a non-object response for {url!r}")
    return value
