from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_fixture import load_rows  # noqa: E402


def test_fixture_is_sanitized_and_structured() -> None:
    rows = load_rows(ROOT / "data" / "questions.jsonl")
    assert [row["id"] for row in rows] == ["support-1", "route-1", "policy-1"]
    assert all(len(row["options"]) == 3 for row in rows)
    json.dumps(rows, ensure_ascii=False, allow_nan=False)


def test_fixture_has_no_private_input_markers() -> None:
    text = (ROOT / "data" / "questions.jsonl").read_text(encoding="utf-8").lower()
    for marker in ("oauth", "token", "password="):
        assert marker not in text
