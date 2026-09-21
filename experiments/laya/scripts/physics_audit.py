"""Run sanitized fixed-commit JevDash control contrasts without a model.

This is a local physics/control diagnostic only. It never counts as a Colab GPU
or real-model result; its purpose is to separate a reachable game from a model
input/action failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable


GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
FPS = 60
MAX_FRAMES = 1800
CONTROL_CASES = ("noop", "right_run", "right_jump", "right_run_jump", "left", "right_run_reflex")


def update_world(player: Any, level: Any, action: str) -> None:
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


def run_case(case: str, Level: Any, Player: Any, TelemetryExtractor: Any, select_action: Callable[..., Any]) -> dict[str, Any]:
    level = Level(1)
    player = Player(level.start_pos[0], level.start_pos[1])
    raw_action = "right_run" if case == "right_run_reflex" else case
    override_frames = 0
    terminal_frame = None
    executed_counts: dict[str, int] = {}

    for frame in range(MAX_FRAMES):
        observation = TelemetryExtractor.extract(player, level)
        if case == "right_run_reflex":
            executed_action, reasons = select_action(observation, raw_action, "reflex_assisted")
            if executed_action != raw_action:
                override_frames += 1
        else:
            executed_action, reasons = raw_action, []
        executed_counts[executed_action] = executed_counts.get(executed_action, 0) + 1
        update_world(player, level, executed_action)
        if player.has_won or player.is_dead:
            terminal_frame = frame + 1
            break

    return {
        "case": case,
        "raw_action": raw_action,
        "executed_action_counts": executed_counts,
        "override_frames": override_frames,
        "terminal_frame": terminal_frame,
        "simulation_frames": (terminal_frame or MAX_FRAMES),
        "progress_pixels": round(player.max_x, 1),
        "score": player.score,
        "coins": player.coins,
        "lives": player.lives,
        "is_dead": player.is_dead,
        "has_won": player.has_won,
        "local_physics_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, str(args.game_src))
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import pygame

    pygame.init()
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    from adapter.control import select_executed_action

    random.seed(42)
    results = [run_case(case, Level, Player, TelemetryExtractor, select_executed_action) for case in CONTROL_CASES]
    output = {
        "status": "ok",
        "game_commit": GAME_COMMIT,
        "level": 1,
        "seed": 42,
        "physics_fps": FPS,
        "max_simulation_frames": MAX_FRAMES,
        "cases": results,
        "note": "local fixed-game physics/control contrast; not a model or Colab GPU result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pygame.quit()
    print(json.dumps({"status": "ok", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
