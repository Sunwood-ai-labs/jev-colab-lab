"""Find bounded, state-conditioned teachers that are safe at the 8-frame cadence."""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
MAX_FRAMES = 1800
CADENCE = 8
START_OFFSETS = (0, -16, 16)


def step_game(player: Any, level: Any, action: str) -> None:
    player.apply_action(action)
    player.update_physics(level.tiles)
    for enemy in level.enemies:
        enemy.update(level.tiles)
        if enemy.alive and player.rect.colliderect(enemy.rect):
            if player.y + player.height <= enemy.y + enemy.height * 0.65:
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


def macro_action(obs: Any, policy: str) -> str:
    terrain, hazard, player = obs.terrain, obs.hazard, obs.player
    obstacle_distance = terrain.obstacle_distance_tiles or 99
    gap_distance = terrain.gap_distance_tiles or 99
    enemy_distance = hazard.nearest_enemy.distance_pixels if hazard.nearest_enemy else 999
    if policy == "constant_run_jump":
        return "right_run_jump"
    if policy == "air_or_hazard_8":
        if not player.grounded:
            return "right_run_jump"
        if obstacle_distance <= 8 or gap_distance <= 8 or enemy_distance <= 260 or hazard.jump_must_start_now:
            return "right_run_jump"
        return "right_run"
    if policy == "air_or_hazard_12":
        if not player.grounded:
            return "right_run_jump"
        if obstacle_distance <= 12 or gap_distance <= 12 or enemy_distance <= 360 or hazard.jump_must_start_now:
            return "right_run_jump"
        return "right_run"
    if policy == "hazard_8":
        if obstacle_distance <= 8 or gap_distance <= 8 or enemy_distance <= 260 or hazard.jump_must_start_now:
            return "right_run_jump"
        return "right_run"
    if policy == "hazard_12":
        if obstacle_distance <= 12 or gap_distance <= 12 or enemy_distance <= 360 or hazard.jump_must_start_now:
            return "right_run_jump"
        return "right_run"
    raise ValueError(policy)


def run_policy(level_type: Any, player_type: Any, extractor_type: Any, policy: str, offset: int) -> dict[str, Any]:
    level = level_type(1)
    player = player_type(level.start_pos[0] + offset, level.start_pos[1])
    counts: collections.Counter[str] = collections.Counter()
    decision_count = 0
    transitions = 0
    previous = None
    for frame in range(MAX_FRAMES):
        if frame % CADENCE == 0:
            obs = extractor_type.extract(player, level)
            action = macro_action(obs, policy)
            decision_count += 1
            if action != previous:
                transitions += 1
            previous = action
        counts[action] += 1
        step_game(player, level, action)
        if player.is_dead or player.has_won:
            break
    return {
        "policy": policy,
        "start_offset_pixels": offset,
        "frames": frame + 1,
        "decision_count": decision_count,
        "has_won": bool(player.has_won),
        "is_dead": bool(player.is_dead),
        "max_x": round(float(player.max_x), 2),
        "final_x": round(float(player.x), 2),
        "action_counts": dict(counts),
        "right_run_jump_rate": round(counts["right_run_jump"] / (frame + 1), 8),
        "decision_transitions": transitions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "schema": "jev-colab-lab/kev-macro-teachers-v1",
        "kind": "local deterministic 8-frame teacher controls; no model/GPU/Colab",
        "game_commit": GAME_COMMIT,
        "cadence_frames": CADENCE,
        "max_frames": MAX_FRAMES,
        "policies": ["constant_run_jump", "air_or_hazard_8", "air_or_hazard_12", "hazard_8", "hazard_12"],
        "cases": [],
        "status": "started",
        "error": None,
    }
    try:
        actual = subprocess.check_output(["git", "-C", str(args.game), "rev-parse", "HEAD"], text=True).strip()
        if actual != GAME_COMMIT:
            raise RuntimeError("game revision mismatch")
        if subprocess.check_output(["git", "-C", str(args.game), "status", "--porcelain"], text=True).strip():
            raise RuntimeError("game checkout is dirty")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        import pygame

        pygame.init()
        sys.path.insert(0, str(args.game / "src"))
        from jev_platformer.engine.entities import Player
        from jev_platformer.engine.world import Level
        from jev_platformer.telemetry.extractor import TelemetryExtractor

        for policy in result["policies"]:
            for offset in START_OFFSETS:
                result["cases"].append(run_policy(Level, Player, TelemetryExtractor, policy, offset))
        pygame.quit()
        result["status"] = "success"
    except Exception:
        result["status"] = "error"
        result["error"] = {"type": "macro_teacher_audit_error", "message": "paths and exception text omitted"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
