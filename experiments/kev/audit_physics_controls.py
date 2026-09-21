"""Deterministic JevDash physics controls for separating game risk from Kev.

This is a local CPU control audit.  It never loads a model and must not be
reported as a model or Colab result.  The game checkout is validated against
the exact commit used by the Kev episode before importing its modules.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
MAX_FRAMES = 1800
CADENCES = (1, 8)
START_OFFSETS = (0, -16, 16)
RAW_ACTIONS = ("right_run", "right_jump", "right_run_jump", "left", "noop")

ACTION_SEMANTICS = {
    "noop": "grounded vx *= 0.82; airborne horizontal inertia is preserved",
    "right": "grounded walk target 4.0; airborne horizontal vx is forced to run speed 6.6",
    "right_run": "right run target 6.6; no jump",
    "right_jump": "right control plus normal jump impulse -13.5 when grounded/coyote-valid",
    "right_run_jump": "right run control plus running jump impulse -15.5 when grounded/coyote-valid",
    "jump": "vertical normal jump impulse -13.5 when grounded/coyote-valid",
    "left": "left control, capped at walk speed -4.0",
}


def reflex_action(observation: Any, raw_action: str) -> tuple[str, list[str]]:
    """Faithful copy of JevDash AsyncJevAgent's published reflex conditions."""

    terrain, hazard, player = observation.terrain, observation.hazard, observation.player
    reasons: list[str] = []
    if terrain.gap_ahead and (terrain.gap_distance_tiles or 99) <= 3.8:
        reasons.append("gap_critical")
    if terrain.obstacle_ahead and (terrain.obstacle_distance_tiles or 99) <= 2.2:
        reasons.append("obstacle_critical")
    if hazard.enemy_ahead and hazard.nearest_enemy is not None and (
        hazard.nearest_enemy.distance_pixels <= 130 or hazard.jump_must_start_now
    ):
        reasons.append("enemy_critical")
    if observation.episode.stalled_frames >= 3:
        reasons.append("stalled")
    if reasons and player.grounded:
        return "right_run_jump", reasons
    if (
        not player.grounded
        and terrain.gap_ahead
        and terrain.gap_distance_tiles is not None
        and terrain.gap_distance_tiles <= 1.5
    ):
        return "right_run_jump", ["air_gap_critical"]
    return raw_action, []


def step_game(player: Any, level: Any, action: str) -> None:
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


def run_case(
    level_type: Any,
    player_type: Any,
    extractor_type: Any,
    raw_action: str,
    assisted: bool,
    cadence: int,
    start_offset: int,
) -> dict[str, Any]:
    random.seed(42)
    level = level_type(1)
    player = player_type(level.start_pos[0] + start_offset, level.start_pos[1])
    action = raw_action
    active_reasons: list[str] = []
    action_counts: collections.Counter[str] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    override_frames = 0
    last_observation = None

    for frame in range(MAX_FRAMES):
        reasons: list[str] = []
        if frame % cadence == 0:
            last_observation = extractor_type.extract(player, level)
            if assisted:
                action, reasons = reflex_action(last_observation, raw_action)
            else:
                action = raw_action
            active_reasons = reasons
        action_counts[action] += 1
        if action != raw_action:
            override_frames += 1
            reason_counts.update(active_reasons or ["override_without_reason"])
        step_game(player, level, action)
        if player.is_dead or player.has_won:
            break

    return {
        "raw_action": raw_action,
        "reflex_assist": assisted,
        "decision_cadence_frames": cadence,
        "start_offset_pixels": start_offset,
        "frames": frame + 1,
        "has_won": bool(player.has_won),
        "is_dead": bool(player.is_dead),
        "max_x": round(float(player.max_x), 2),
        "final_x": round(float(player.x), 2),
        "final_y": round(float(player.y), 2),
        "final_stalled_frames": int(player.stalled_frames),
        "override_frames": override_frames,
        "override_rate": round(override_frames / (frame + 1), 8),
        "executed_action_counts": dict(action_counts),
        "override_reason_counts": dict(reason_counts),
        "last_observation_available": last_observation is not None,
    }


def dependency_versions() -> dict[str, str | None]:
    names = ("pygame", "pydantic")
    return {name: importlib.metadata.version(name) if importlib.util.find_spec(name) else None for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": "jev-colab-lab/kev-physics-controls-v1",
        "kind": "local deterministic physics controls; no model inference; no Colab",
        "game_commit_expected": GAME_COMMIT,
        "seed": 42,
        "fps": 60,
        "max_frames": MAX_FRAMES,
        "action_semantics": ACTION_SEMANTICS,
        "cases": [],
        "hardware": {
            "name": "local CPU",
            "gpu": False,
            "platform": platform.platform(aliased=True, terse=True),
            "python": platform.python_version(),
            "dependencies": dependency_versions(),
        },
        "probabilities": {"applicable": False, "reason": "No model inference"},
        "peak_vram": {"applicable": False, "reason": "CPU-only physics; no GPU allocated"},
        "status": "running",
        "error": None,
        "timings": {},
    }
    phase = "validate_game_checkout"
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(args.game), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        if actual_commit != GAME_COMMIT:
            raise ValueError("game revision mismatch")
        dirty = subprocess.check_output(
            ["git", "-C", str(args.game), "status", "--porcelain"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        if dirty:
            raise ValueError("game checkout is not clean")

        phase = "load_game_modules"
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        import pygame

        pygame.init()
        sys.path.insert(0, str(args.game / "src"))
        from jev_platformer.engine.entities import Player
        from jev_platformer.engine.world import Level
        from jev_platformer.telemetry.extractor import TelemetryExtractor

        result["timings"]["validation_and_loading_seconds"] = time.perf_counter() - started
        phase = "simulate_controls"
        simulation_started = time.perf_counter()
        for start_offset in START_OFFSETS:
            for raw_action in RAW_ACTIONS:
                for assisted, cadence in ((False, 8), (True, 1), (True, 8)):
                    case_started = time.perf_counter()
                    row = run_case(Level, Player, TelemetryExtractor, raw_action, assisted, cadence, start_offset)
                    row["simulation_wall_seconds"] = round(time.perf_counter() - case_started, 6)
                    result["cases"].append(row)
        result["timings"]["simulation_seconds"] = time.perf_counter() - simulation_started
        result["status"] = "success"
        pygame.quit()
    except Exception:
        result["status"] = "error"
        result["error"] = {
            "phase": phase,
            "type": "control_audit_error",
            "message": "Control audit failed; exception text and paths are omitted.",
        }
    finally:
        result["timings"]["total_wall_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if result["status"] != "success":
        print(json.dumps(result["error"], ensure_ascii=False))
        return 1
    for row in result["cases"]:
        if row["start_offset_pixels"] == 0:
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
