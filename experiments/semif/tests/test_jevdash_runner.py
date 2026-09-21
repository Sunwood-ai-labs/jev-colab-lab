from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "jevdash_play.py"
ENTRY = ROOT / "scripts" / "jevdash_colab_entry.py"
V2_EVIDENCE = ROOT / "results" / "jevdash-clear-20260921-v2" / "episode.json"


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
    assert assignments["DEFAULT_CONTROLLER_PROMPT_VERSION"] == "jev-dash-rules-v2"
    assert assignments["CONTROLLER_PROMPT_VERSION"] == "jev-dash-rules-v3-physics"
    assert assignments["ACTION_DESCRIPTIONS_V2"]["right_run_jump"] == (
        "RIGHT_RUN_JUMP: hold right at running speed and start a jump; use for a pipe, pit gap, "
        "approaching enemy, or a stalled wall, and keep this input while airborne until landing."
    )
    assert "MockJevAgent" not in source
    assert "JevLiveAgent" not in source
    assert '"fallback_used": False' in source
    for expected in (
        "while grounded",
        "while airborne",
        "coyote time",
        "coarse scan columns",
        '"model_input_state"',
        '"raw_action"',
        '"executed_action"',
        '"override_applied": False',
        '"--controller-prompt-version"',
    ):
        assert expected in source


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
        '"--controller-prompt-version",\n    "jev-dash-rules-v2"',
    ):
        assert expected in entry


def test_game_commit_marker_is_sanitized() -> None:
    assert (ROOT / "data" / "jevdash-commit.txt").read_text(encoding="utf-8").strip() == (
        "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
    )


def test_saved_v2_gpu_log_matches_replayable_prompt_contract() -> None:
    spec = importlib.util.spec_from_file_location("jevdash_play_contract", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    episode = json.loads(V2_EVIDENCE.read_text(encoding="utf-8"))
    question, descriptions = module.PROMPT_CONFIGS[module.DEFAULT_CONTROLLER_PROMPT_VERSION]
    expected_options = [
        {"id": action, "description": descriptions[action]}
        for action in module.ACTIONS
    ]
    assert episode["source"]["controller_prompt_version"] == "jev-dash-rules-v2"
    assert len(episode["decisions"]) == 81
    for decision in episode["decisions"]:
        assert decision["question"] == question
        assert decision["options"] == expected_options
        assert decision["model_input_state"] == module.compact_state(decision["observation"])
        assert decision["raw_action"] == decision["executed_action"]
        assert decision["override_applied"] is False
