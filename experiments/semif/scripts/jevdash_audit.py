"""Audit SemIf JevDash inputs, tokenization, option order, and raw choices.

This is an inference-only companion to ``jevdash_play.py``.  It builds several
real telemetry fixtures from the fixed game, scores controlled prompt variants
with the pinned SemIf direct readout, and records the raw action probabilities.
It never alters game physics and never replaces a model choice with a helper
action.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

GAME_REPOSITORY = "https://github.com/Sunwood-ai-labs/jevdash"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
SEMIF_REPOSITORY = "https://github.com/TheoLeeCJ/semif"
SEMIF_COMMIT = "ca3ba65f142967030ecb453346e94d6f476a69df"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REPOSITORY = "https://huggingface.co/Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SESSION_NAME = "jev-semif-jevdash-clear"
CLI_VERSION = "google-colab-cli 0.6.0"
SEED = 42
MAX_TOKENS = 4096

ACTIONS = (
    "noop",
    "right",
    "right_run",
    "right_jump",
    "right_run_jump",
    "jump",
    "left",
)
REVERSE_ACTIONS = tuple(reversed(ACTIONS))
JUMP_FIRST_ACTIONS = (
    "right_run_jump",
    "right_jump",
    "right_run",
    "jump",
    "right",
    "noop",
    "left",
)
ROTATED_ACTIONS = (
    "right_run",
    "right_run_jump",
    "right_jump",
    "noop",
    "jump",
    "right",
    "left",
)
ACTION_ORDERS = {
    "canonical": ACTIONS,
    "reverse": REVERSE_ACTIONS,
    "jump_first": JUMP_FIRST_ACTIONS,
    "rotated": ROTATED_ACTIONS,
}

BASE_DESCRIPTIONS = {
    "noop": "Do nothing and preserve the current horizontal momentum.",
    "right": "Move right at walking speed.",
    "right_run": "Move right at running speed.",
    "right_jump": "Move right and jump.",
    "right_run_jump": "Move right at running speed and jump.",
    "jump": "Jump without selecting a horizontal direction.",
    "left": "Move left at walking speed.",
}
ENRICHED_DESCRIPTIONS = {
    "noop": "NOOP: apply no directional or jump input; preserve momentum.",
    "right": "RIGHT: hold right at walking speed without starting a jump.",
    "right_run": "RIGHT_RUN: hold right at running speed without jumping; use only on clear flat ground.",
    "right_jump": "RIGHT_JUMP: hold right and start a normal jump to clear a low obstacle or enemy.",
    "right_run_jump": "RIGHT_RUN_JUMP: hold right at running speed and start a jump; use for a pipe, pit gap, approaching enemy, or a stalled wall.",
    "jump": "JUMP: start a vertical jump without choosing a horizontal direction.",
    "left": "LEFT: hold left at walking speed to backtrack; do not use to advance toward the goal.",
}

BASE_QUESTION = (
    "Choose exactly one action macro for the next 8 simulation frames based only "
    "on the supplied JevDash Level 1 observation. Select the macro that best "
    "advances the player to the goal while avoiding hazards."
)
RULES_QUESTION = (
    "Choose exactly one action macro for the next 8 simulation frames. Use the "
    "telemetry literally and output only one listed option. Decision priority: "
    "(1) if grounded and stalled_frames is at least 3, choose RIGHT_RUN_JUMP; "
    "(2) if grounded and obstacle_distance_tiles is at most 2.5, gap_distance_tiles "
    "is at most 3.5, or jump_must_start_now is true, choose RIGHT_RUN_JUMP or "
    "RIGHT_JUMP; (3) if a hazard is not imminent and the path is clear, choose "
    "RIGHT_RUN. Do not choose RIGHT_RUN when the player is blocked or must jump. "
    "The option descriptions define the exact game input mapping."
)
COMPACT_QUESTION = (
    "Choose one macro for the next 8 frames. Read the structured player, hazard, "
    "terrain, and stall fields as control state. Use the priority rules exactly "
    "and output only the selected option."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def scrub_error(error: BaseException) -> str:
    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    message = re.sub(r"(?:[A-Za-z]:)?[/\\](?:[^\s'\"]+[/\\])*[^\s'\"]+", "<path>", message)
    message = re.sub(r"(?i)(token|secret|password|authorization)\s*[:=]\s*[^\s,;]+", r"\1=<redacted>", message)
    return message[:1000] or error.__class__.__name__


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dependency_versions() -> dict[str, str]:
    packages = (
        "semif-phase1",
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "tokenizers",
        "numpy",
        "sentencepiece",
        "protobuf",
        "pygame",
        "pydantic",
    )
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def gpu_metadata(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_count": int(torch.cuda.device_count()),
    }


def memory_snapshot(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def seed_everything(seed: int, torch: Any) -> dict[str, Any]:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "python_random": seed,
        "numpy": seed,
        "torch": seed,
        "torch_cuda": seed if torch.cuda.is_available() else None,
        "deterministic_cudnn": True,
    }


def import_game(game_root: Path, commit_file: Path) -> dict[str, Any]:
    actual_commit = commit_file.read_text(encoding="utf-8").strip()
    if actual_commit != GAME_COMMIT:
        raise RuntimeError("game commit marker does not match the required fixed commit")
    sys.path.insert(0, str(game_root / "src"))
    import pygame
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    return {"pygame": pygame, "Player": Player, "Level": Level, "TelemetryExtractor": TelemetryExtractor, "actual_commit": actual_commit}


def physics_tick(player: Any, level: Any, action: str) -> None:
    player.apply_action(action)
    player.update_physics(level.tiles)
    for enemy in level.enemies:
        enemy.update(level.tiles)
        if enemy.alive and player.rect.colliderect(enemy.rect):
            feet_y = player.y + player.height
            if feet_y <= enemy.y + enemy.height * 0.65:
                enemy.stomp()
                player.vy = -11.0
                player.score += 100
            else:
                player.is_dead = True
    for coin in level.coins:
        if not coin.collected and player.rect.colliderect(coin.rect):
            coin.collected = True
            player.coins += 1
            player.score += 50
    if level.goal and player.rect.colliderect(level.goal.rect):
        player.has_won = True


def observation_fixtures(game: dict[str, Any]) -> dict[str, dict[str, Any]]:
    level_cls = game["Level"]
    player_cls = game["Player"]
    extractor = game["TelemetryExtractor"]

    level = level_cls(1)
    player = player_cls(*level.start_pos)
    start = extractor.extract(player, level)

    level = level_cls(1)
    player = player_cls(*level.start_pos)
    for _ in range(56):
        physics_tick(player, level, "right_run")
    approach_pipe = extractor.extract(player, level)

    level = level_cls(1)
    player = player_cls(*level.start_pos)
    for _ in range(64):
        physics_tick(player, level, "right_run")
    blocked_pipe = extractor.extract(player, level)

    level = level_cls(1)
    player = player_cls(30 * 32, 540)
    player.grounded = True
    player.max_x = player.x
    gap = extractor.extract(player, level)

    level = level_cls(1)
    player = player_cls(600, 540)
    player.grounded = True
    player.max_x = player.x
    enemy = extractor.extract(player, level)

    level = level_cls(1)
    player = player_cls(*level.start_pos)
    physics_tick(player, level, "right_run_jump")
    airborne = extractor.extract(player, level)

    return {
        "start": {"observation": start.model_dump(mode="json"), "source": "fixed Level(1), initial player"},
        "approach_pipe": {"observation": approach_pipe.model_dump(mode="json"), "source": "56 exact right_run physics ticks before decision"},
        "blocked_pipe": {"observation": blocked_pipe.model_dump(mode="json"), "source": "64 exact right_run physics ticks before decision"},
        "first_gap": {"observation": gap.model_dump(mode="json"), "source": "fixed Level(1), player at column 30 grounded"},
        "enemy_ahead": {"observation": enemy.model_dump(mode="json"), "source": "fixed Level(1), player x=600 grounded"},
        "airborne": {"observation": airborne.model_dump(mode="json"), "source": "one exact right_run_jump physics tick"},
    }


def compact_state(state: dict[str, Any]) -> str:
    player = state["player"]
    hazard = state["hazard"]
    terrain = state["terrain"]
    episode = state["episode"]
    enemy = hazard["nearest_enemy"]
    enemy_text = "none"
    if enemy is not None:
        enemy_text = (
            f"kind={enemy['kind']} distance_pixels={enemy['distance_pixels']} "
            f"vertical_offset_pixels={enemy['vertical_offset_pixels']} "
            f"relative_velocity_x={enemy['relative_velocity_x']} "
            f"estimated_contact_frames={enemy['estimated_contact_frames']}"
        )
    return "\n".join(
        (
            "JevDash Level 1 control state:",
            f"player x={player['x']} y={player['y']} vx={player['vx']} vy={player['vy']} grounded={player['grounded']} jumping={player['jumping']} running={player['running']}",
            f"episode progress_pixels={episode['progress_pixels']} goal_distance_pixels={episode['goal_distance_pixels']} stalled_frames={episode['stalled_frames']} is_dead={episode['is_dead']} has_won={episode['has_won']}",
            f"terrain obstacle_ahead={terrain['obstacle_ahead']} obstacle_distance_tiles={terrain['obstacle_distance_tiles']} obstacle_height_tiles={terrain['obstacle_height_tiles']} gap_ahead={terrain['gap_ahead']} gap_distance_tiles={terrain['gap_distance_tiles']} gap_width_tiles={terrain['gap_width_tiles']} clear_forward_tiles={terrain['clear_forward_tiles']}",
            f"hazard enemy_ahead={hazard['enemy_ahead']} jump_must_start_now={hazard['jump_must_start_now']} in_danger_zone={hazard['in_danger_zone']} nearest_enemy=({enemy_text})",
            "local_grid:",
            *state["local_grid"],
        )
    )


def row_for(state: dict[str, Any], variant: str, actions: tuple[str, ...]) -> dict[str, Any]:
    if variant == "baseline_json":
        row_state: str | dict[str, Any] = state
        question = BASE_QUESTION
        descriptions = BASE_DESCRIPTIONS
    elif variant == "rules_json":
        row_state = state
        question = RULES_QUESTION
        descriptions = ENRICHED_DESCRIPTIONS
    elif variant == "rules_compact":
        row_state = compact_state(state)
        question = RULES_QUESTION
        descriptions = ENRICHED_DESCRIPTIONS
    elif variant == "semantic_compact":
        row_state = compact_state(state)
        question = COMPACT_QUESTION
        descriptions = ENRICHED_DESCRIPTIONS
    else:
        raise ValueError(f"unknown prompt variant: {variant}")
    return {
        "id": f"audit-{variant}",
        "state": row_state,
        "question": question,
        "options": [{"id": action, "description": descriptions[action]} for action in actions],
    }


def tokenization_metadata(tokenizer: Any, row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    from semif_phase1.core import direct_messages
    from semif_phase1.direct import encode_prompt

    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    return {
        "prompt_sha256": prompt_hash,
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
        "input_tokens": len(ids),
        "input_token_tail_ids": ids[-24:],
        "input_token_tail_text": tokenizer.decode(ids[-24:], skip_special_tokens=False),
        "input_token_tail_tokens": tokenizer.convert_ids_to_tokens(ids[-24:]),
        "option_slot_token_ids": slots,
        "option_slot_token_text": [tokenizer.decode([token]) for token in slots],
        "option_slot_single_token": True,
        "option_order": [option["id"] for option in row["options"]],
    }


def full_vocab_metadata(vocabulary: Any, tokenizer: Any, slot_ids: list[int], top_k: int = 12) -> dict[str, Any]:
    import torch

    log_normalizer = float(torch.logsumexp(vocabulary, dim=0).item())
    log_probabilities = torch.log_softmax(vocabulary, dim=0)
    option_log_probabilities = [float(log_probabilities[token].item()) for token in slot_ids]
    allowed_mass = float(sum(math.exp(value) for value in option_log_probabilities))
    top_values, top_indices = torch.topk(vocabulary, k=top_k)
    top_tokens = []
    for value, index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        top_tokens.append(
            {
                "token_id": int(index),
                "token_text": tokenizer.decode([int(index)], skip_special_tokens=False),
                "logit": float(value),
                "full_vocab_probability": float(math.exp(float(log_probabilities[index].item()))),
            }
        )
    argmax_id = int(torch.argmax(vocabulary).item())
    option_ranks = {
        str(token): int((vocabulary > vocabulary[token]).sum().item()) + 1
        for token in slot_ids
    }
    return {
        "full_vocab_log_normalizer": log_normalizer,
        "full_vocab_argmax_token_id": argmax_id,
        "full_vocab_argmax_token_text": tokenizer.decode([argmax_id], skip_special_tokens=False),
        "allowed_action_slot_probability_mass": allowed_mass,
        "action_slot_full_vocab_log_probabilities": option_log_probabilities,
        "action_slot_full_vocab_ranks": option_ranks,
        "full_vocab_top_tokens": top_tokens,
    }


def score_one(model: Any, tokenizer: Any, loader_metadata: dict[str, Any], state_name: str, state: dict[str, Any], variant: str, order_name: str, max_tokens: int) -> dict[str, Any]:
    import torch
    from semif_phase1.direct import score

    actions = ACTION_ORDERS[order_name]
    row = row_for(state, variant, actions)
    token_meta = tokenization_metadata(tokenizer, row, max_tokens)
    captured: dict[str, Any] = {}

    def capture_output(_module: Any, _inputs: Any, output: Any) -> None:
        logits = output.logits if hasattr(output, "logits") else output[0]
        captured["last_logits"] = logits[:, -1, :].detach().float().cpu()[0]

    handle = model.register_forward_hook(capture_output)
    started = time.perf_counter()
    try:
        result = score(model, tokenizer, row, loader_metadata, max_tokens)
    finally:
        handle.remove()
    torch.cuda.synchronize(0)
    wall_seconds = time.perf_counter() - started
    if "last_logits" not in captured:
        raise RuntimeError("could not capture the direct-readout full vocabulary")
    probabilities = [float(value) for value in result["probabilities"]]
    logits = [float(value) for value in result["option_logits"]]
    if len(probabilities) != len(actions) or len(logits) != len(actions):
        raise RuntimeError("unexpected number of option scores")
    if any(not math.isfinite(value) for value in probabilities + logits):
        raise RuntimeError("non-finite option score")
    selected_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "state_name": state_name,
        "prompt_variant": variant,
        "order_name": order_name,
        "selected_action": actions[selected_index],
        "selected_probability": probabilities[selected_index],
        "probabilities": probabilities,
        "probabilities_by_action": dict(zip(actions, probabilities, strict=True)),
        "option_logits": logits,
        "option_logits_by_action": dict(zip(actions, logits, strict=True)),
        "forward_seconds": float(result["forward_seconds"]),
        "total_seconds": float(result["total_seconds"]),
        "wall_seconds": float(wall_seconds),
        "readout": result["readout"],
        "probability_status": result["probability_status"],
        "tokenization": token_meta,
        "full_vocabulary": full_vocab_metadata(captured["last_logits"], tokenizer, token_meta["option_slot_token_ids"]),
        "fallback_used": False,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["state_name"], row["prompt_variant"])].append(row)
    order_comparison: list[dict[str, Any]] = []
    for (state_name, variant), items in grouped.items():
        by_order = {item["order_name"]: item for item in items}
        selected_by_order = {order: item["selected_action"] for order, item in by_order.items()}
        action_ranges: dict[str, float] = {}
        for action in ACTIONS:
            values = [item["probabilities_by_action"].get(action, 0.0) for item in items]
            action_ranges[action] = max(values) - min(values)
        order_comparison.append(
            {
                "state_name": state_name,
                "prompt_variant": variant,
                "selected_by_order": selected_by_order,
                "order_changes_selected_action": len(set(selected_by_order.values())) > 1,
                "max_probability_range_by_action": action_ranges,
            }
        )

    fixture_sensitivity: list[dict[str, Any]] = []
    for variant in sorted({row["prompt_variant"] for row in rows}):
        items = [row for row in rows if row["prompt_variant"] == variant and row["order_name"] == "canonical"]
        by_action = defaultdict(list)
        for item in items:
            for action, probability in item["probabilities_by_action"].items():
                by_action[action].append(probability)
        fixture_sensitivity.append(
            {
                "prompt_variant": variant,
                "selected_by_fixture": {item["state_name"]: item["selected_action"] for item in items},
                "max_probability_range_by_action": {action: max(values) - min(values) for action, values in by_action.items()},
            }
        )
    return {
        "rows": len(rows),
        "order_comparison": order_comparison,
        "fixture_sensitivity": fixture_sensitivity,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    payload: dict[str, Any] = {
        "schema": "jev-semif-jevdash-input-audit-v1",
        "status": "starting",
        "started_at_utc": utc_now(),
        "runtime": {
            "label": "Google Colab GPU runtime",
            "colab_cli": CLI_VERSION,
            "session_name": SESSION_NAME,
            "python_version": platform.python_version(),
            "platform": platform.platform(aliased=True),
            "dependencies": {},
            "gpu": {},
            "seed_sources": {},
        },
        "game": {
            "repository": GAME_REPOSITORY,
            "expected_commit": GAME_COMMIT,
            "actual_commit": None,
            "level": 1,
            "fixtures": {},
        },
        "source": {
            "semif_repository": SEMIF_REPOSITORY,
            "semif_commit": SEMIF_COMMIT,
            "model_id": MODEL_ID,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "training_source": "upstream Qwen/Qwen3.5-4B checkpoint; no local fine-tuning",
            "readout": "SemIf direct native next-token logits",
            "fallback_used": False,
            "mock_agent_used": False,
        },
        "prompt_contract": {
            "variants": ["baseline_json", "rules_json", "rules_compact", "semantic_compact"],
            "action_orders": {name: list(actions) for name, actions in ACTION_ORDERS.items()},
            "max_input_tokens": args.max_tokens,
            "no_generation": True,
        },
        "rows": [],
        "summary": {},
        "errors": [],
    }
    torch = None
    try:
        import torch as torch_module
        from semif_phase1.core import load_causal_model

        torch = torch_module
        payload["runtime"]["seed_sources"] = seed_everything(SEED, torch)
        payload["runtime"]["dependencies"] = dependency_versions()
        payload["runtime"].update({"torch_version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())})
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one visible CUDA GPU")
        gpu = gpu_metadata(torch)
        payload["runtime"]["gpu"] = gpu
        if "l4" not in gpu["name"].lower():
            raise RuntimeError(f"expected L4 GPU, got {gpu['name']!r}")

        load_started = time.perf_counter()
        model, tokenizer, loader_metadata = load_causal_model(MODEL_ID, MODEL_REVISION)
        torch.cuda.synchronize(0)
        payload["runtime"]["model_load_seconds"] = time.perf_counter() - load_started
        payload["runtime"]["load_memory"] = memory_snapshot(torch)
        torch.cuda.reset_peak_memory_stats(0)

        game = import_game(args.game_root, args.game_commit_file)
        payload["game"]["actual_commit"] = game["actual_commit"]
        fixture_map = observation_fixtures(game)
        payload["game"]["fixtures"] = fixture_map
        rows: list[dict[str, Any]] = []
        for state_name, fixture in fixture_map.items():
            state = fixture["observation"]
            for variant in payload["prompt_contract"]["variants"]:
                for order_name in ACTION_ORDERS:
                    rows.append(score_one(model, tokenizer, loader_metadata, state_name, state, variant, order_name, args.max_tokens))
        payload["rows"] = rows
        payload["summary"] = summarize(rows)
        payload["runtime"]["inference_peak_memory"] = memory_snapshot(torch)
        payload["runtime"]["total_inference_wall_seconds"] = sum(row["wall_seconds"] for row in rows)
        payload["status"] = "success"
        payload["finished_at_utc"] = utc_now()
        return payload, 0
    except Exception as error:  # noqa: BLE001 - sanitized audit failure is an artifact
        payload["status"] = "failed"
        payload["finished_at_utc"] = utc_now()
        payload["errors"].append({"type": error.__class__.__name__, "message": scrub_error(error)})
        if torch is not None and torch.cuda.is_available():
            try:
                payload["runtime"]["inference_peak_memory"] = memory_snapshot(torch)
            except Exception:  # noqa: BLE001 - retain original failure
                pass
        return payload, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, default=Path("/content/jevdash"))
    parser.add_argument("--game-commit-file", type=Path, default=Path("/content/jevdash-commit.txt"))
    parser.add_argument("--output", type=Path, default=Path("/content/semif-jevdash-audit.json"))
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    payload, exit_code = run(args)
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "rows": len(payload["rows"]), "errors": payload["errors"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
