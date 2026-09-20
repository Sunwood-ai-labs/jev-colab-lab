"""Validate the public, non-sensitive SemIf input fixture without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("id must be a non-empty string")
    if not isinstance(row["question"], str) or not row["question"]:
        raise ValueError("question must be a non-empty string")
    if not isinstance(row["state"], (str, dict, list)) or not row["state"]:
        raise ValueError("state must be a non-empty string, object, or array")
    options = row["options"]
    if not isinstance(options, list) or len(options) < 2 or len(options) > 16:
        raise ValueError("options must contain 2-16 entries")
    option_ids = []
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("each option must be an object")
        if not isinstance(option.get("id"), str) or not option["id"]:
            raise ValueError("each option needs a non-empty string id")
        if not isinstance(option.get("description"), str) or not option["description"]:
            raise ValueError("each option needs a non-empty string description")
        option_ids.append(option["id"])
    if len(option_ids) != len(set(option_ids)):
        raise ValueError("option ids must be unique")
    json.dumps(row, ensure_ascii=False, allow_nan=False)


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("fixture is empty")
    for row in rows:
        validate_row(row)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("row ids must be unique")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input)
    print(json.dumps({"status": "ok", "rows": len(rows), "ids": [row["id"] for row in rows]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
