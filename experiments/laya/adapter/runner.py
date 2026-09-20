"""Run fixed-commit JevDash with synchronous real Laya decisions and record MP4."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from jev_platformer.controller.actions import Action
from jev_platformer.engine.constants import (
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_HUD_BG,
    COLOR_HUD_BORDER,
    COLOR_HUD_CARD,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    FPS,
    GAME_BRAND,
    GAME_TITLE,
    GAME_VIEW_WIDTH,
    HUD_HEIGHT,
    HUD_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from jev_platformer.engine.entities import Player
from jev_platformer.engine.world import Level
from jev_platformer.telemetry.extractor import TelemetryExtractor
from jev_platformer.ui.renderer import GameRenderer
from jev_platformer.ui.video_recorder import VideoRecorder

try:
    from .laya_agent import GAME_ACTIONS, LayaActionAdapter
except ImportError:
    from laya_agent import GAME_ACTIONS, LayaActionAdapter


GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
GAME_REPOSITORY = "https://github.com/Sunwood-ai-labs/jevdash"
LEVEL = 1
SEED = 42
FRAMES_PER_DECISION = 8
MAX_SIMULATION_FRAMES = 1800
TERMINAL_HOLD_FRAMES = 120


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> Dict[str, bool]:
    random.seed(seed)
    seeded = {"python_random": True, "numpy": False, "torch": False, "torch_cuda": False}
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded["torch_cuda"] = True
    except Exception:
        pass
    return seeded


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def _error(exc: BaseException) -> Dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc).replace("\r", " ").replace("\n", " ")[:1200]}


class LayaHud:
    """Game HUD following the fixed game's layout while labeling Laya-only fields honestly."""

    def __init__(self, surface: pygame.Surface, gpu_name: str):
        self.surface = surface
        self.offset_x = GAME_VIEW_WIDTH
        self.gpu_name = gpu_name
        self.font_xs = pygame.font.SysFont("Consolas, Menlo, monospace", 10)
        self.font_sm = pygame.font.SysFont("Consolas, Menlo, monospace", 12)
        self.font_base = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 14, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 20, bold=True)

    def _text(self, text: str, pos: tuple[int, int], color=COLOR_TEXT_PRIMARY, font=None) -> None:
        self.surface.blit((font or self.font_sm).render(text, True, color), pos)

    def render(
        self,
        observation: Any,
        decision: Optional[Dict[str, Any]],
        sim_frame: int,
        terminal_reason: Optional[str] = None,
    ) -> None:
        x = self.offset_x + 20
        y = 14
        pygame.draw.rect(self.surface, COLOR_HUD_BG, (self.offset_x, 0, HUD_WIDTH, HUD_HEIGHT))
        pygame.draw.line(self.surface, COLOR_HUD_BORDER, (self.offset_x, 0), (self.offset_x, HUD_HEIGHT), 2)

        self._text("JevDash: System One", (x, y), font=self.font_large)
        y += 29
        badge = "AI AUTOPILOT / SYNC"
        pygame.draw.rect(self.surface, COLOR_HUD_CARD, (x, y, 152, 20), border_radius=4)
        pygame.draw.rect(self.surface, COLOR_ACCENT, (x, y, 152, 20), 1, border_radius=4)
        self._text(badge, (x + 7, y + 4), COLOR_ACCENT, self.font_xs)
        y += 26
        self._text("MODEL: Laya / ModernBERT-large", (x, y), COLOR_SUCCESS, self.font_xs)
        y += 15
        self._text(f"GPU: {self.gpu_name}", (x, y), COLOR_TEXT_MUTED, self.font_xs)
        y += 24

        card = pygame.Rect(x, y, HUD_WIDTH - 40, 88)
        pygame.draw.rect(self.surface, COLOR_HUD_CARD, card, border_radius=8)
        pygame.draw.rect(self.surface, COLOR_HUD_BORDER, card, 1, border_radius=8)
        action = (decision or {}).get("action", "waiting")
        latency = (decision or {}).get("inference_ms")
        self._text("MODEL DECISION (ARGMAX)", (x + 11, y + 8), COLOR_TEXT_MUTED, self.font_xs)
        self._text(action, (x + 11, y + 23), COLOR_ACCENT, self.font_large)
        self._text("INFERENCE", (x + 205, y + 8), COLOR_TEXT_MUTED, self.font_xs)
        self._text("--" if latency is None else f"{latency:.1f} ms", (x + 205, y + 23), COLOR_TEXT_PRIMARY, self.font_base)
        sim_time = sim_frame / FPS
        self._text(f"SIM TIME {sim_time:06.2f}s @ {FPS} FPS", (x + 11, y + 57), COLOR_TEXT_PRIMARY, self.font_xs)
        self._text("inference waits omitted", (x + 205, y + 57), COLOR_TEXT_MUTED, self.font_xs)
        y += 99

        self._text("Choice Probability Distribution", (x, y), font=self.font_base)
        y += 22
        chart = pygame.Rect(x, y, HUD_WIDTH - 40, 198)
        pygame.draw.rect(self.surface, COLOR_HUD_CARD, chart, border_radius=8)
        pygame.draw.rect(self.surface, COLOR_HUD_BORDER, chart, 1, border_radius=8)
        probabilities = (decision or {}).get("probabilities", {})
        bar_y = y + 10
        for action_name in GAME_ACTIONS:
            prob = max(0.0, min(1.0, float(probabilities.get(action_name, 0.0))))
            selected = action_name == (decision or {}).get("action")
            text_color = COLOR_ACCENT if selected else COLOR_TEXT_MUTED
            self._text(f"{action_name:<14}", (x + 12, bar_y), text_color, self.font_xs)
            bg = pygame.Rect(x + 130, bar_y + 1, 145, 9)
            pygame.draw.rect(self.surface, (30, 36, 54), bg, border_radius=3)
            if prob > 0:
                pygame.draw.rect(
                    self.surface,
                    COLOR_ACCENT if selected else (71, 85, 105),
                    (x + 130, bar_y + 1, int(145 * prob), 9),
                    border_radius=3,
                )
            self._text(f"{prob * 100:5.1f}%", (x + 288, bar_y), text_color, self.font_xs)
            bar_y += 24
        y += 212

        self._text("Local Radar Perception (7x11)", (x, y), font=self.font_base)
        y += 21
        radar = pygame.Rect(x, y, HUD_WIDTH - 40, 125)
        pygame.draw.rect(self.surface, COLOR_HUD_CARD, radar, border_radius=8)
        pygame.draw.rect(self.surface, COLOR_HUD_BORDER, radar, 1, border_radius=8)
        for index, line in enumerate(observation.local_grid):
            self._text(line, (x + 16, y + 8 + index * 15), (94, 234, 212), self.font_xs)
        y += 138

        self._text("DANGER / URGENCY: UNMEASURED", (x, y), COLOR_DANGER, self.font_xs)
        y += 17
        progress = observation.episode.progress_pixels
        self._text(f"PROGRESS {int(progress)}px   ACTIONS {len(GAME_ACTIONS)}", (x, y), COLOR_TEXT_PRIMARY, self.font_xs)
        y += 19
        status = terminal_reason or "RUNNING"
        status_color = COLOR_SUCCESS if terminal_reason == "clear" else COLOR_DANGER if terminal_reason == "death" else COLOR_TEXT_MUTED
        self._text(f"STATUS: {status.upper()}", (x, y), status_color, self.font_xs)
        self._text("SYNC EVERY 8 FRAMES  |  LEVEL 1  |  SEED 42", (x, HUD_HEIGHT - 27), COLOR_TEXT_MUTED, self.font_xs)


def _update_world(player: Player, level: Level, action: str) -> None:
    """Keep the fixed commit's physics/collision order identical to cli.run_play."""

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


def _decision_record(frame: int, observation: Any, decision: Any) -> Dict[str, Any]:
    return {
        "simulation_frame": frame,
        "simulation_time_seconds": frame / FPS,
        "observation": _safe(observation.model_dump(mode="json")),
        "candidate_actions": list(GAME_ACTIONS),
        "action": decision.action,
        "model_reported_choice": decision.model_choice,
        "probabilities": _safe(decision.probabilities),
        "confidence": decision.confidence,
        "inference_ms": decision.inference_ms,
        "input_tokens": decision.input_tokens,
        "prompt_stats": _safe(decision.prompt_stats),
        "raw_answer": _safe(decision.raw_answer),
        "danger_score": None,
        "jump_urgency": None,
    }


def run_capture(video_path: Path, json_path: Path) -> Dict[str, Any]:
    seed_evidence = seed_everything(SEED)
    adapter = LayaActionAdapter(device="cuda")
    pygame.init()
    pygame.display.set_caption(f"{GAME_TITLE} - Laya")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    level = Level(LEVEL)
    player = Player(level.start_pos[0], level.start_pos[1])
    game_renderer = GameRenderer(screen)
    gpu_name = adapter.metadata()["gpu"]["name"]
    hud = LayaHud(screen, gpu_name=gpu_name)
    recorder = VideoRecorder(str(video_path), SCREEN_WIDTH, SCREEN_HEIGHT, fps=FPS)
    decisions = []
    current_decision = None
    current_action = Action.NOOP.value
    simulation_frame = 0
    terminal_reason: Optional[str] = None
    episode_started = time.perf_counter()
    quit_requested = False
    video_frame_count = 0

    try:
        while simulation_frame < MAX_SIMULATION_FRAMES and terminal_reason is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    quit_requested = True
                    terminal_reason = "quit"

            if terminal_reason is not None:
                break

            observation = TelemetryExtractor.extract(player, level)
            if simulation_frame % FRAMES_PER_DECISION == 0:
                laya_decision = adapter.decide(observation)
                current_action = laya_decision.action
                current_decision = {
                    "action": laya_decision.action,
                    "probabilities": laya_decision.probabilities,
                    "confidence": laya_decision.confidence,
                    "inference_ms": laya_decision.inference_ms,
                }
                decisions.append(_decision_record(simulation_frame, observation, laya_decision))

            _update_world(player, level, current_action)
            if player.has_won:
                terminal_reason = "clear"
            elif player.is_dead:
                terminal_reason = "death"

            game_renderer.render(player, level)
            hud.render(observation, current_decision, simulation_frame, terminal_reason=terminal_reason)
            pygame.display.flip()
            recorder.record_frame(screen)
            video_frame_count += 1
            clock.tick(FPS)
            simulation_frame += 1

        if terminal_reason is None and simulation_frame >= MAX_SIMULATION_FRAMES:
            terminal_reason = "max_simulation_frames"

        # The terminal hold is video-only: no physics, model call, or decision is added.
        if terminal_reason in {"clear", "death", "max_simulation_frames"}:
            final_observation = TelemetryExtractor.extract(player, level)
            for _ in range(TERMINAL_HOLD_FRAMES):
                game_renderer.render(player, level)
                hud.render(final_observation, current_decision, simulation_frame, terminal_reason=terminal_reason)
                pygame.display.flip()
                recorder.record_frame(screen)
                video_frame_count += 1
    finally:
        recorder.close()
        pygame.quit()

    wall_clock_seconds = time.perf_counter() - episode_started
    video_duration_seconds = video_frame_count / FPS
    result = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "game": {
            "repository": GAME_REPOSITORY,
            "commit": GAME_COMMIT,
            "level": LEVEL,
            "seed": SEED,
            "physics_fps": FPS,
            "frames_per_decision": FRAMES_PER_DECISION,
            "max_simulation_frames": MAX_SIMULATION_FRAMES,
            "terminal_hold_frames": TERMINAL_HOLD_FRAMES,
            "screen": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
            "fixed_physics_and_collision_order": True,
            "control_mode": "synchronous real Laya argmax only",
            "mock_fallback": False,
            "model_driven_replay": False,
        },
        "model": adapter.metadata(),
        "reproducibility": {
            "python_seed_calls": seed_evidence,
            "candidate_actions": list(GAME_ACTIONS),
            "action_question": _safe(ACTION_QUESTION),
            "observation_schema": "JevObservation.model_dump() from fixed JevDash commit",
            "state_summary_or_control_rules_inserted": False,
            "danger_and_urgency_model_outputs": "unmeasured; Laya action-only question",
        },
        "episode": {
            "terminal_reason": terminal_reason,
            "simulation_frames": simulation_frame,
            "simulation_time_seconds": simulation_frame / FPS,
            "video_frames": video_frame_count,
            "video_duration_seconds": video_duration_seconds,
            "wall_clock_play_seconds_including_inference_waits": wall_clock_seconds,
            "inference_waits_are_excluded_from_simulation_time": True,
            "decision_count": len(decisions),
            "progress_pixels": round(player.max_x, 1),
            "score": player.score,
            "coins": player.coins,
            "lives": player.lives,
            "is_dead": player.is_dead,
            "has_won": player.has_won,
            "video_file": video_path.name,
        },
        "decisions": decisions,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not produce a non-empty video: {video_path}")
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_capture(args.video, args.json)
    except Exception as exc:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "error",
            "created_at_utc": utc_now(),
            "game_commit": GAME_COMMIT,
            "model_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
            "error": _error(exc),
        }
        args.json.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal_reason": result["episode"]["terminal_reason"],
                "simulation_frames": result["episode"]["simulation_frames"],
                "video_frames": result["episode"]["video_frames"],
                "decision_count": result["episode"]["decision_count"],
                "video": str(args.video),
                "json": str(args.json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
