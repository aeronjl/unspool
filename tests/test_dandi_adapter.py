import pytest

from unspool import DANDIAdapterError, DANDINWBSource, ResolvedDANDIAsset
from unspool.adapters import dandi as dandi_adapter


def source() -> DANDINWBSource:
    return DANDINWBSource(
        dandiset_id="000004",
        version="0.220126.1852",
        asset_path="sub-P11HMH/sub-P11HMH_ses-20061101_ecephys+image.nwb",
        session_order=0,
        session="2006-11-01",
        columns=("start_time", "stop_time", "stim_phase", "response_value"),
        column_map={"response_value": "recorded_response"},
    )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"dandiset_id": "4"}, "six digits"),
        ({"version": "draft"}, "published"),
        ({"version": "latest"}, "published"),
        ({"asset_path": "file.txt"}, "NWB"),
        ({"session_order": -1}, "session_order"),
    ],
)
def test_dandi_source_requires_pinned_nwb_identity(changes, message) -> None:
    values = {
        "dandiset_id": "000004",
        "version": "0.220126.1852",
        "asset_path": "sub-subject/session.nwb",
        "session_order": 0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        DANDINWBSource(**values)


def test_dandi_asset_resolution_retains_path_hash_size_and_content_url(monkeypatch) -> None:
    listing = {
        "count": 1,
        "results": [
            {
                "asset_id": "asset-id",
                "blob": "blob-id",
                "zarr": None,
                "path": source().asset_path,
                "size": 72_628_704,
            }
        ],
    }
    metadata = {
        "digest": {"dandi:sha2-256": "e" * 64},
        "contentUrl": [
            "https://api.dandiarchive.org/api/assets/asset-id/download/",
            "https://dandiarchive.s3.amazonaws.com/blobs/blob-id",
        ],
    }
    monkeypatch.setattr(
        dandi_adapter,
        "_read_json",
        lambda url, timeout: listing if "versions" in url else metadata,
    )

    asset = dandi_adapter.resolve_dandi_nwb_asset(source())

    assert asset == ResolvedDANDIAsset(
        dandiset_id="000004",
        version="0.220126.1852",
        path=source().asset_path,
        asset_id="asset-id",
        size=72_628_704,
        sha256="e" * 64,
        content_url="https://dandiarchive.s3.amazonaws.com/blobs/blob-id",
    )


def test_dandi_resolution_rejects_ambiguous_and_zarr_assets(monkeypatch) -> None:
    monkeypatch.setattr(
        dandi_adapter,
        "_read_json",
        lambda url, timeout: {"count": 0, "results": []},
    )
    with pytest.raises(DANDIAdapterError, match="exactly one"):
        dandi_adapter.resolve_dandi_nwb_asset(source())

    monkeypatch.setattr(
        dandi_adapter,
        "_read_json",
        lambda url, timeout: {
            "count": 1,
            "results": [
                {
                    "asset_id": "asset-id",
                    "blob": None,
                    "zarr": "zarr-id",
                    "path": source().asset_path,
                    "size": 1,
                }
            ],
        },
    )
    with pytest.raises(DANDIAdapterError, match="blob-backed"):
        dandi_adapter.resolve_dandi_nwb_asset(source())
