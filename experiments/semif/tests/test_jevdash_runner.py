from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "jevdash_play.py"
ENTRY = ROOT / "scripts" / "jevdash_colab_entry.py"


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return result


def test_runner_contract_is_fixed_and_real_model_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assignments = _literal_assignments(RUNNER)
    assert assignments["GAME_COMMIT"] == "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
    assert assignments["MODEL_ID"] == "Qwen/Qwen3.5-4B"
    assert assignments["MODEL_REVISION"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert assignments["FPS"] == 60
    assert assignments["FRAMES_PER_DECISION"] == 8
    assert assignments["MAX_SIMULATION_FRAMES"] == 1800
    assert assignments["STATIC_TERMINAL_FRAMES"] == 120
    assert "MockJevAgent" not in source
    assert "JevLiveAgent" not in source
    assert '"fallback_used": False' in source


def test_action_order_and_colab_entry_are_explicit() -> None:
    assignments = _literal_assignments(RUNNER)
    assert assignments["ACTIONS"] == (
        "noop",
        "right",
        "right_run",
        "right_jump",
        "right_run_jump",
        "jump",
        "left",
    )
    entry = ENTRY.read_text(encoding="utf-8")
    for expected in (
        '"--game-root",\n    "/content/jevdash"',
        '"--seed",\n    "42"',
        '"--frames-per-decision",\n    "8"',
        '"--max-frames",\n    "1800"',
        '"--static-terminal-frames",\n    "120"',
    ):
        assert expected in entry


def test_game_commit_marker_is_sanitized() -> None:
    assert (ROOT / "data" / "jevdash-commit.txt").read_text(encoding="utf-8").strip() == (
        "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
    )
