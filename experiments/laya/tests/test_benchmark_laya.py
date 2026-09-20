import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from benchmark_laya import build_cases, json_safe, summarize


def test_cases_are_synthetic_and_have_single_and_batch_paths():
    cases = build_cases()
    assert set(cases) == {"single", "batch"}
    assert cases["single"]["state"]["ticket_id"].startswith("synthetic-")
    assert len(cases["single"]["questions"]) == 1
    assert len(cases["batch"]["questions"]) == 4
    assert list(cases["batch"]["questions"]["department"]["criteria"]) == [
        "billing",
        "technical",
        "sales",
        "other",
    ]


def test_summarize_uses_interpolated_percentiles():
    result = summarize([10, 20, 30, 40])
    assert result["count"] == 4
    assert result["median_ms"] == 25
    assert result["p95_ms"] == 38.5


def test_json_safe_converts_non_finite_values_to_null():
    result = json_safe({"nan": float("nan"), "nested": (1, True)})
    assert result == {"nan": None, "nested": [1, True]}
    json.dumps(result)
