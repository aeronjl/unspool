"""Shared parity contract between committed results and other people's published values.

Every ``benchmarks/*/published_claims.json`` records what a paper printed, what the
committed ``result.json`` recomputed, and the declared tolerance that separates the two
answers. The contract runs offline in milliseconds, so a silent drift away from a
published number cannot survive a default test run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
CONTRACT_PATHS = sorted(ROOT.glob("benchmarks/*/published_claims.json"))
SCHEMA_VERSION = 1
TOP_LEVEL_FIELDS = {"schema_version", "paper", "data", "result", "claims"}
PAPER_FIELDS = {"doi", "figure", "citation"}
DATA_FIELDS = {"accession", "member_sha256"}
CHECKABLE_FIELDS = {
    "id",
    "description",
    "result_path",
    "published_value",
    "tolerance",
    "tolerance_rationale",
    "observed_value",
    "status",
}
WAIVED_FIELDS = {
    "id",
    "description",
    "result_path",
    "published_value",
    "tolerance",
    "observed_value",
    "status",
    "waiver_rationale",
}
CHECKABLE_STATUSES = {"pass", "fail"}
STATUSES = CHECKABLE_STATUSES | {"waived"}
TOLERANCE_KINDS = {"absolute", "relative", "upper_bound"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contracts() -> list[tuple[str, Path, dict[str, Any]]]:
    return [(path.parent.name, path, _load(path)) for path in CONTRACT_PATHS]


def _claims() -> list[tuple[str, Path, dict[str, Any], dict[str, Any]]]:
    return [
        (f"{name}:{claim['id']}", path, contract, claim)
        for name, path, contract in _contracts()
        for claim in contract["claims"]
    ]


def _resolve(payload: Any, result_path: str) -> Any:
    current = payload
    for segment in result_path.split("."):
        current = current[int(segment)] if isinstance(current, list) else current[segment]
    return current


def _within_tolerance(observed: float, published: float, tolerance: dict[str, Any]) -> bool:
    kind = tolerance["kind"]
    value = float(tolerance["value"])
    if kind == "absolute":
        return abs(float(observed) - float(published)) <= value
    if kind == "relative":
        return abs(float(observed) - float(published)) <= value * abs(float(published))
    return float(observed) <= float(published) + value


CONTRACT_CASES = _contracts()
CLAIM_CASES = _claims()


def test_at_least_one_published_claims_contract_is_committed() -> None:
    assert CONTRACT_PATHS


@pytest.mark.parametrize(
    ("name", "path", "contract"),
    CONTRACT_CASES,
    ids=[case[0] for case in CONTRACT_CASES],
)
def test_contract_declares_its_paper_data_and_result(
    name: str, path: Path, contract: dict[str, Any]
) -> None:
    assert set(contract) == TOP_LEVEL_FIELDS, name
    assert contract["schema_version"] == SCHEMA_VERSION, name
    assert set(contract["paper"]) == PAPER_FIELDS, name
    assert set(contract["data"]) == DATA_FIELDS, name
    assert all(str(value).strip() for value in contract["paper"].values()), name
    assert all(str(value).strip() for value in contract["data"].values()), name
    assert len(contract["data"]["member_sha256"]) == 64, name
    assert (path.parent / contract["result"]).is_file(), name

    identifiers = [claim["id"] for claim in contract["claims"]]
    assert identifiers, name
    assert len(identifiers) == len(set(identifiers)), name


@pytest.mark.parametrize(
    ("case", "path", "contract", "claim"),
    CLAIM_CASES,
    ids=[case[0] for case in CLAIM_CASES],
)
def test_claim_is_well_formed(
    case: str, path: Path, contract: dict[str, Any], claim: dict[str, Any]
) -> None:
    assert claim["status"] in STATUSES, case
    if claim["status"] == "waived":
        assert set(claim) == WAIVED_FIELDS, case
        assert claim["result_path"] is None, case
        assert claim["observed_value"] is None, case
        assert claim["tolerance"] is None, case
        assert len(claim["waiver_rationale"].strip()) >= 40, case
        return

    assert set(claim) == CHECKABLE_FIELDS, case
    assert isinstance(claim["result_path"], str) and claim["result_path"].strip(), case
    assert isinstance(claim["published_value"], int | float), case
    assert isinstance(claim["observed_value"], int | float), case
    assert claim["tolerance"]["kind"] in TOLERANCE_KINDS, case
    assert isinstance(claim["tolerance"]["value"], int | float), case
    assert float(claim["tolerance"]["value"]) >= 0.0, case
    assert len(claim["tolerance_rationale"].strip()) >= 40, case
    assert len(claim["description"].strip()) >= 20, case


@pytest.mark.parametrize(
    ("case", "path", "contract", "claim"),
    CLAIM_CASES,
    ids=[case[0] for case in CLAIM_CASES],
)
def test_observed_value_comes_from_the_committed_result(
    case: str, path: Path, contract: dict[str, Any], claim: dict[str, Any]
) -> None:
    if claim["status"] == "waived":
        pytest.skip(f"{case} is waived and records no observed value")
    result = _load(path.parent / contract["result"])
    stored = _resolve(result, claim["result_path"])

    assert isinstance(stored, int | float), case
    assert float(stored) == float(claim["observed_value"]), case


@pytest.mark.parametrize(
    ("case", "path", "contract", "claim"),
    CLAIM_CASES,
    ids=[case[0] for case in CLAIM_CASES],
)
def test_status_reports_the_tolerance_comparison_honestly(
    case: str, path: Path, contract: dict[str, Any], claim: dict[str, Any]
) -> None:
    if claim["status"] == "waived":
        pytest.skip(f"{case} is waived and is compared to no published value")
    observed = float(claim["observed_value"])
    published = float(claim["published_value"])

    assert math.isfinite(observed), case
    assert (claim["status"] == "pass") is _within_tolerance(
        observed, published, claim["tolerance"]
    ), case


@pytest.mark.parametrize(
    ("case", "path", "contract", "claim"),
    CLAIM_CASES,
    ids=[case[0] for case in CLAIM_CASES],
)
def test_no_claim_is_left_pending(
    case: str, path: Path, contract: dict[str, Any], claim: dict[str, Any]
) -> None:
    assert claim["status"] != "pending", case
    assert claim["status"] in STATUSES, case
