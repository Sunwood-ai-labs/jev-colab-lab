"""Audit OpenJev input representations and candidate-order sensitivity.

This script loads the pinned model once on the Colab GPU, constructs real
JevDash telemetry states from the immutable game clone, and scores the legacy
and model-card-style NLI prompts.  It is an audit only: no game action is
executed and no fallback controller is used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


GAME_REPOSITORY = "https://github.com/Sunwood-ai-labs/jevdash.git"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
GAME_CLONE_DIR = Path("/content/jevdash-openjev-nli")
FPS = 60
SEED = 42


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        raw = list(sys.argv[1:])
        argv = []
        skip_next = False
        for argument in raw:
            if skip_next:
                skip_next = False
                continue
            if argument == "-f":
                skip_next = True
                continue
            argv.append(argument)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/content/openjev-nli-jevdash-input-audit.json")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args(argv)


def ensure_game_clone() -> str:
    if not GAME_CLONE_DIR.exists():
        subprocess.run(["git", "clone", "--quiet", GAME_REPOSITORY, str(GAME_CLONE_DIR)], check=True)
    subprocess.run(
        ["git", "-C", str(GAME_CLONE_DIR), "checkout", "--detach", "--quiet", GAME_COMMIT],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(GAME_CLONE_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != GAME_COMMIT:
        raise RuntimeError(f"game commit mismatch: expected {GAME_COMMIT}, got {actual}")
    return actual


def step_world(player: Any, level: Any, action: str) -> None:
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


def fresh_grounded_player(level: Any, x: float) -> Any:
    from jev_platformer.engine.entities import Player

    player = Player(x, level.get_ground_level() - 36)
    player.grounded = True
    player.coyote_frames = 6
    player.max_x = player.x
    return player


def collect_states() -> dict[str, dict[str, Any]]:
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    states: dict[str, dict[str, Any]] = {}

    level = Level(1)
    states["initial"] = TelemetryExtractor.extract(
        fresh_grounded_player(level, level.start_pos[0]), level
    ).model_dump(mode="json")

    level = Level(1)
    player = fresh_grounded_player(level, level.start_pos[0])
    for _ in range(240):
        step_world(player, level, "right_run")
        if player.stalled_frames >= 12 or player.is_dead:
            break
    states["stalled_at_pipe"] = TelemetryExtractor.extract(player, level).model_dump(mode="json")

    states["obstacle_takeoff"] = TelemetryExtractor.extract(
        fresh_grounded_player(level, 14 * 32), level
    ).model_dump(mode="json")
    states["gap_takeoff"] = TelemetryExtractor.extract(
        fresh_grounded_player(level, 30 * 32), level
    ).model_dump(mode="json")
    states["enemy_takeoff"] = TelemetryExtractor.extract(
        fresh_grounded_player(level, 19 * 32), level
    ).model_dump(mode="json")

    level = Level(1)
    player = fresh_grounded_player(level, 30 * 32)
    for _ in range(4):
        step_world(player, level, "right_run_jump")
    states["airborne_near_gap"] = TelemetryExtractor.extract(player, level).model_dump(mode="json")

    level = Level(1)
    player = fresh_grounded_player(level, 130 * 32)
    states["near_goal"] = TelemetryExtractor.extract(player, level).model_dump(mode="json")
    return states


def scores_by_action(decision: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        row["action"]: {label: float(value) for label, value in row["probabilities"].items()}
        for row in decision["probabilities"]
    }


def order_comparison(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        return {"status": "no_decisions"}
    reference = scores_by_action(decisions[0])
    max_delta = 0.0
    changed_choices = 0
    for decision in decisions[1:]:
        scores = scores_by_action(decision)
        max_delta = max(
            max_delta,
            max(
                abs(scores[action][label] - reference[action][label])
                for action in reference
                for label in ("contradiction", "entailment", "neutral")
            ),
        )
        if decision["action"] != decisions[0]["action"]:
            changed_choices += 1
    return {
        "status": "order_invariant" if max_delta == 0.0 else "order_sensitive",
        "reference_action": decisions[0]["action"],
        "changed_choices": changed_choices,
        "max_probability_delta": round(max_delta, 8),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    actual_commit = ensure_game_clone()
    sys.path.insert(0, str(GAME_CLONE_DIR / "src"))
    sys.path.insert(0, "/content")

    import pygame

    pygame.init()
    try:
        from openjev_nli_adapter import CARD_ACTION_OPTIONS, OpenJevNLIAdapter

        adapter = OpenJevNLIAdapter(max_length=args.max_length)
        states = collect_states()
        rng = random.Random(SEED)
        orders = [
            list(CARD_ACTION_OPTIONS),
            list(reversed(CARD_ACTION_OPTIONS)),
        ]
        for _ in range(3):
            order = list(CARD_ACTION_OPTIONS)
            rng.shuffle(order)
            orders.append(order)

        by_state: dict[str, Any] = {}
        for state_name, observation in states.items():
            profiles: list[dict[str, Any]] = []
            legacy = adapter.decide(observation, profile="legacy")
            profiles.append({"profile": "legacy", "candidate_order": legacy["candidate_order"], "decision": legacy})
            card_decisions = []
            for index, order in enumerate(orders):
                decision = adapter.decide(observation, profile="card", candidate_order=order)
                card_decisions.append(decision)
                profiles.append({
                    "profile": "card",
                    "order_index": index,
                    "candidate_order": decision["candidate_order"],
                    "decision": decision,
                })
            by_state[state_name] = {
                "observation": observation,
                "profiles": profiles,
                "card_order_comparison": order_comparison(card_decisions),
            }

        compact_decisions = [
            entry["profiles"][1]["decision"]
            for entry in by_state.values()
        ]
        payload = {
            "schema_version": "1.0",
            "status": "ok",
            "created_at_utc": utc_now(),
            "experiment": "openjev-nli-jevdash-input-audit",
            "source": {
                "game_repository": GAME_REPOSITORY,
                "game_commit_requested": GAME_COMMIT,
                "game_commit_actual": actual_commit,
                "model_id": "AlexWortega/openjev",
                "model_revision": "b32265f4700df7c02532933c9a4ff258a449d7ac",
                "model_subfolder": "qwen3.5-4b-nli",
            },
            "configuration": {
                "seed": SEED,
                "fps": FPS,
                "max_length": args.max_length,
                "profiles": ["legacy", "card"],
                "candidate_order_trials": len(orders),
                "execution": "audit only; no game action was executed",
            },
            "environment": adapter.environment(),
            "model_load": {
                "elapsed_ms": adapter.load_elapsed_ms,
                "gpu_memory_allocated_bytes": int(adapter.torch.cuda.memory_allocated()),
                "gpu_memory_reserved_bytes": int(adapter.torch.cuda.memory_reserved()),
            },
            "states": by_state,
            "summary": {
                "state_count": len(states),
                "legacy_actions": sorted({entry["profiles"][0]["decision"]["action"] for entry in by_state.values()}),
                "card_actions": sorted({decision["action"] for decision in compact_decisions}),
                "card_premise_byte_min": min(decision["premise_byte_count_utf8"] for decision in compact_decisions),
                "card_premise_byte_max": max(decision["premise_byte_count_utf8"] for decision in compact_decisions),
                "card_sequence_length_min": min(decision["sequence_length"] for decision in compact_decisions),
                "card_sequence_length_max": max(decision["sequence_length"] for decision in compact_decisions),
                "card_order_comparisons": {
                    name: entry["card_order_comparison"] for name, entry in by_state.items()
                },
            },
            "finished_at_utc": utc_now(),
        }
        write_json(output_path, payload)
        print(json.dumps({"status": "ok", "output": str(output_path), "summary": payload["summary"]}, ensure_ascii=False))
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
