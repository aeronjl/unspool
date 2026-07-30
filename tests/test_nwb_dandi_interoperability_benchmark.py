import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark

RESULT_PATH = Path("benchmarks/nwb_dandi_interoperability/result.json")


def test_pinned_nwb_dandi_result_preserves_identity_chronology_and_semantics() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["contract_passed"]
    assert all(result["contract"].values())
    assert result["source"]["dandiset_id"] == "000004"
    assert result["source"]["version"] == "0.220126.1852"
    assert result["source"]["asset_id"] == "0f57f0b0-f021-42bb-8eaa-56cd482e2a29"
    assert result["study"]["n_trials"] == 200
    assert result["study"]["n_subjects"] == 1
    assert result["study"]["phase_counts"] == {"learn": 100, "recog": 100}
    assert result["study"]["response_value_minimum"] == 0
    assert result["study"]["response_value_maximum"] == 36
