"""Run fixed JevDash Level 1 with the pinned OpenJev NLI model on a Colab GPU.

The runner imports the game from an immutable external clone and mirrors the
physics/update order in ``src/jev_platformer/cli.py``.  It does not use the
game's mock/live API controller, and it has no heuristic fallback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


GAME_REPOSITORY = "https://github.com/Sunwood-ai-labs/jevdash.git"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
GAME_CLONE_DIR = Path("/content/jevdash-openjev-nli")
DEFAULT_VIDEO = "/content/openjev-nli-jevdash.mp4"
DEFAULT_JSON = "/content/openjev-nli-jevdash.json"
DEFAULT_FRAME_DIR = "/content/openjev-nli-frames"
FPS = 60
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FRAMES_PER_DECISION = 8
MAX_SIMULATION_FRAMES = 1800
TERMINAL_FRAMES = 120
SEED = 42


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def redact_error(message: str) -> str:
    redacted = str(message)
    redacted = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "<path>", redacted)
    redacted = re.sub(r"/home/[^\s'\"]+", "<path>", redacted)
    redacted = re.sub(r"/content/[^\s'\"]+", "<remote-path>", redacted)
    redacted = re.sub(r"(?i)(token|secret|password|api[_-]?key)=[^\s&]+", r"\1=<redacted>", redacted)
    return redacted[:1200]


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
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--json", dest="json_path", default=DEFAULT_JSON)
    parser.add_argument("--frame-dir", default=DEFAULT_FRAME_DIR)
    parser.add_argument("--frames-per-decision", type=int, default=FRAMES_PER_DECISION)
    parser.add_argument("--max-frames", type=int, default=MAX_SIMULATION_FRAMES)
    parser.add_argument("--terminal-frames", type=int, default=TERMINAL_FRAMES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--profile", choices=("legacy", "card"), default="legacy")
    parser.add_argument("--execution-mode", choices=("model-only", "assisted"), default="model-only")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    applied: dict[str, Any] = {"python_random": seed}
    try:
        import numpy as np

        np.random.seed(seed)
        applied["numpy"] = seed
    except Exception as error:
        applied["numpy"] = f"not_available: {type(error).__name__}"
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        applied["torch"] = seed
        applied["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception as error:
        applied["torch"] = f"not_available: {type(error).__name__}"
    return applied


def ensure_game_clone() -> str:
    if not GAME_CLONE_DIR.exists():
        subprocess.run(
            ["git", "clone", "--quiet", GAME_REPOSITORY, str(GAME_CLONE_DIR)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(GAME_CLONE_DIR), "checkout", "--detach", "--quiet", GAME_COMMIT],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(GAME_CLONE_DIR), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != GAME_COMMIT:
        raise RuntimeError(f"game commit mismatch: expected {GAME_COMMIT}, got {actual}")
    return actual


def initial_result(args: argparse.Namespace, seeds: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "running",
        "created_at_utc": utc_now(),
        "experiment": "openjev-nli-jevdash-level1",
        "source": {
            "game_repository": GAME_REPOSITORY,
            "game_commit_requested": GAME_COMMIT,
            "game_clone_path": str(GAME_CLONE_DIR),
            "model_id": "AlexWortega/openjev",
            "model_revision": "b32265f4700df7c02532933c9a4ff258a449d7ac",
            "model_subfolder": "qwen3.5-4b-nli",
            "model_license": "MIT",
            "base_model": "Qwen/Qwen3.5-4B",
        },
        "training_origin": {
            "checkpoint": "AlexWortega/openjev qwen3.5-4b-nli, a Qwen3.5-4B NLI sequence-classification checkpoint",
            "game_specific_finetuning": False,
            "decision_mapping": "zero-shot premise/hypothesis action scoring; no JevDash labels or mock controller",
        },
        "configuration": {
            "level": 1,
            "seed": args.seed,
            "seeds_applied": seeds,
            "fps": FPS,
            "frames_per_decision": args.frames_per_decision,
            "max_simulation_frames": args.max_frames,
            "terminal_static_frames": args.terminal_frames,
            "resolution": [SCREEN_WIDTH, SCREEN_HEIGHT],
            "video_time_basis": "60 FPS simulation frames; synchronous model wait wall-clock time is omitted from video time",
            "control_mode": "synchronous model decision every frames_per_decision frames",
            "input_profile": args.profile,
            "execution_mode": args.execution_mode,
            "action_selection": "argmax over NLI entailment probability across the fixed seven JevDash actions",
            "fallback": "none; model or runtime failure is recorded as error",
            "assistance": (
                "none; raw model action is applied exactly"
                if args.execution_mode == "model-only"
                else "every-physics-frame safety reflex; raw and executed actions are both recorded"
            ),
        },
        "action_options": [
            "noop",
            "right",
            "right_run",
            "right_jump",
            "right_run_jump",
            "jump",
            "left",
        ],
        "model_decisions": [],
        "trajectory": [],
    }


class OpenJevHUD:
    """A truthful JevDash-style HUD with model and simulation-time labels."""

    def __init__(self, pygame: Any, surface: Any, game_view_width: int, hud_width: int, hud_height: int, colors: dict[str, Any]):
        self.pygame = pygame
        self.surface = surface
        self.offset_x = game_view_width
        self.hud_width = hud_width
        self.hud_height = hud_height
        self.colors = colors
        self.font_xs = pygame.font.SysFont("Consolas, Menlo, monospace", 11)
        self.font_sm = pygame.font.SysFont("Consolas, Menlo, monospace", 13)
        self.font_base = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 22, bold=True)

    def _text(self, text: str, pos: tuple[int, int], color: Any, font: Any | None = None) -> None:
        self.surface.blit((font or self.font_xs).render(text, True, color), pos)

    def render(
        self,
        obs: Any,
        decision: dict[str, Any] | None,
        simulation_frame: int,
        status: str,
        gpu_name: str,
        raw_action: str,
        executed_action: str,
        override_reason: str | None,
    ) -> None:
        pygame = self.pygame
        c = self.colors
        x = self.offset_x + 20
        hud = pygame.Rect(self.offset_x, 0, self.hud_width, self.hud_height)
        pygame.draw.rect(self.surface, c["hud_bg"], hud)
        pygame.draw.line(self.surface, c["hud_border"], (self.offset_x, 0), (self.offset_x, self.hud_height), 2)

        y = 16
        self._text("JevDash: System One", (x, y), c["primary"], self.font_large)
        y += 30
        self._text("OPENJEV NLI 4B  |  MODEL-DRIVEN", (x, y), c["success"], self.font_xs)
        y += 17
        self._text("AlexWortega/openjev", (x, y), c["muted"], self.font_xs)
        y += 16
        self._text(f"GPU: {gpu_name[:31]}", (x, y), c["muted"], self.font_xs)
        y += 18

        card = pygame.Rect(x, y, self.hud_width - 40, 92)
        pygame.draw.rect(self.surface, c["card"], card, border_radius=8)
        pygame.draw.rect(self.surface, c["accent"], card, 1, border_radius=8)
        self._text("RAW MODEL ACTION", (x + 12, y + 8), c["muted"], self.font_xs)
        self._text(raw_action if decision else "waiting", (x + 12, y + 23), c["accent"], self.font_large)
        self._text(f"EXECUTED: {executed_action}", (x + 12, y + 50), c["primary"], self.font_xs)
        sim_time = simulation_frame / 60.0
        self._text(f"SIM TIME {sim_time:6.2f}s (INFERENCE WAITS OMITTED)", (x + 12, y + 68), c["primary"], self.font_xs)
        y += 106

        self._text(f"STATUS: {status.upper()}", (x, y), c["success"] if status == "clear" else c["danger"] if status == "dead" else c["primary"], self.font_base)
        y += 24
        if override_reason:
            self._text(f"ASSIST OVERRIDE: {override_reason}", (x, y), c["accent"], self.font_xs)
        else:
            self._text("ASSIST OVERRIDE: none", (x, y), c["muted"], self.font_xs)
        y += 20
        self._text(f"SIM FRAME {simulation_frame:04d} / {MAX_SIMULATION_FRAMES}", (x, y), c["muted"], self.font_xs)
        y += 20

        self._text("NLI ENTAILMENT SCORE (7 ACTIONS)", (x, y), c["primary"], self.font_base)
        y += 23
        chart = pygame.Rect(x, y, self.hud_width - 40, 204)
        pygame.draw.rect(self.surface, c["card"], chart, border_radius=8)
        pygame.draw.rect(self.surface, c["hud_border"], chart, 1, border_radius=8)
        if decision:
            bar_y = y + 10
            for item in decision["probabilities"]:
                chosen = item["action"] == decision["action"]
                text_color = c["accent"] if chosen else c["muted"]
                self._text(f"{item['action']:<14}", (x + 12, bar_y), text_color)
                bar = pygame.Rect(x + 130, bar_y + 2, 142, 9)
                pygame.draw.rect(self.surface, (30, 36, 54), bar, border_radius=3)
                width = int(142 * min(1.0, max(0.0, float(item["entailment_score"]))))
                if width:
                    pygame.draw.rect(self.surface, c["accent"] if chosen else (71, 85, 105), (x + 130, bar_y + 2, width, 9), border_radius=3)
                self._text(f"{item['entailment_score'] * 100:5.1f}%", (x + 282, bar_y), text_color)
                bar_y += 25
        y += 218

        self._text("LOCAL RADAR PERCEPTION (7x11)", (x, y), c["primary"], self.font_base)
        y += 23
        radar = pygame.Rect(x, y, self.hud_width - 40, 128)
        pygame.draw.rect(self.surface, c["card"], radar, border_radius=8)
        pygame.draw.rect(self.surface, c["hud_border"], radar, 1, border_radius=8)
        if obs is not None:
            grid_y = y + 9
            for line in obs.local_grid:
                self._text(line, (x + 16, grid_y), (94, 234, 212))
                grid_y += 16

        footer_y = self.hud_height - 45
        self._text("VIDEO: 60 FPS SIM TIME", (x, footer_y), c["primary"], self.font_xs)
        self._text("Model waits are excluded from playback time", (x, footer_y + 16), c["muted"], self.font_xs)


def update_world(player: Any, level: Any, action: str) -> None:
    """Keep the fixed commit's physics and collision order unchanged."""

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


def _count_actions(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row[key])
        counts[action] = counts.get(action, 0) + 1
    return counts


def _count_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("override_reason")
        if reason:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def assisted_action(observation: Any, raw_action: str) -> tuple[str, str | None]:
    """Apply only the fixed game's documented safety reflex, with a reason."""

    reasons: list[str] = []
    terrain = observation.terrain
    hazard = observation.hazard
    episode = observation.episode
    if terrain.gap_ahead and (terrain.gap_distance_tiles or 99) <= 3.8:
        reasons.append("gap_ahead")
    if terrain.obstacle_ahead and (terrain.obstacle_distance_tiles or 99) <= 2.2:
        reasons.append("obstacle_ahead")
    if (
        hazard.enemy_ahead
        and hazard.nearest_enemy is not None
        and (
            hazard.nearest_enemy.distance_pixels <= 130.0
            or hazard.jump_must_start_now
        )
    ):
        reasons.append("enemy_threat")
    if episode.stalled_frames >= 3:
        reasons.append("stalled")

    if reasons and observation.player.grounded:
        return "right_run_jump", "+".join(reasons)

    if (
        not observation.player.grounded
        and terrain.gap_ahead
        and terrain.gap_distance_tiles is not None
        and terrain.gap_distance_tiles <= 1.5
    ):
        return "right_run_jump", "airborne_gap"

    return raw_action, None


def trajectory_row(
    frame: int,
    decision_index: int,
    raw_action: str,
    executed_action: str,
    override_reason: str | None,
    player: Any,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "simulation_frame": frame,
        "simulation_time_s": round(frame / FPS, 6),
        "decision_index": decision_index,
        "raw_action": raw_action,
        "executed_action": executed_action,
        "action_applied": executed_action,
        "override_reason": override_reason,
        "overridden": override_reason is not None,
        "x": round(float(player.x), 4),
        "y": round(float(player.y), 4),
        "vx": round(float(player.vx), 4),
        "vy": round(float(player.vy), 4),
        "grounded": bool(player.grounded),
        "is_dead": bool(player.is_dead),
        "has_won": bool(player.has_won),
        "progress_pixels": round(float(player.max_x), 4),
        "status": status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = seed_everything(args.seed)
    result = initial_result(args, seeds)
    output_path = Path(args.json_path)
    recorder = None
    pygame = None
    surface_samples: list[tuple[str, Any]] = []
    wall_start = time.perf_counter()

    try:
        if args.frames_per_decision != FRAMES_PER_DECISION:
            raise ValueError(f"frames_per_decision must remain {FRAMES_PER_DECISION}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed on the Colab VM")

        actual_game_commit = ensure_game_clone()
        result["source"]["game_commit_actual"] = actual_game_commit
        sys.path.insert(0, str(GAME_CLONE_DIR / "src"))

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame as pygame_module

        pygame = pygame_module
        sys.path.insert(0, "/content")
        from openjev_nli_adapter import OpenJevNLIAdapter
        from jev_platformer.engine.constants import (
            COLOR_ACCENT,
            COLOR_DANGER,
            COLOR_HUD_BG,
            COLOR_HUD_BORDER,
            COLOR_HUD_CARD,
            COLOR_SUCCESS,
            COLOR_TEXT_MUTED,
            COLOR_TEXT_PRIMARY,
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

        adapter = OpenJevNLIAdapter(max_length=args.max_length)
        result["environment"] = adapter.environment()
        result["model_load"] = {
            "elapsed_ms": adapter.load_elapsed_ms,
            "gpu_memory_allocated_bytes": int(adapter.torch.cuda.memory_allocated()),
            "gpu_memory_reserved_bytes": int(adapter.torch.cuda.memory_reserved()),
        }

        pygame.init()
        pygame.display.set_caption(GAME_TITLE + " - OpenJev NLI 4B")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        level = Level(1)
        player = Player(level.start_pos[0], level.start_pos[1])
        game_renderer = GameRenderer(screen)
        colors = {
            "hud_bg": COLOR_HUD_BG,
            "hud_border": COLOR_HUD_BORDER,
            "card": COLOR_HUD_CARD,
            "primary": COLOR_TEXT_PRIMARY,
            "muted": COLOR_TEXT_MUTED,
            "accent": COLOR_ACCENT,
            "danger": COLOR_DANGER,
            "success": COLOR_SUCCESS,
        }
        hud = OpenJevHUD(pygame, screen, GAME_VIEW_WIDTH, HUD_WIDTH, HUD_HEIGHT, colors)
        recorder = VideoRecorder(args.video, SCREEN_WIDTH, SCREEN_HEIGHT, fps=FPS)
        if recorder.process is None:
            raise RuntimeError("VideoRecorder could not start ffmpeg")

        frame_dir = Path(args.frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        current_obs = None
        current_decision: dict[str, Any] | None = None
        current_action = "noop"
        model_decision: dict[str, Any] | None = None
        frame_count = 0
        decision_index = -1
        end_status = "timeout"
        last_regular_surface = None

        while frame_count < args.max_frames and not player.is_dead and not player.has_won:
            pygame.event.pump()
            frame_obs = TelemetryExtractor.extract(player, level)
            if frame_count % args.frames_per_decision == 0:
                current_obs = frame_obs
                decision_index += 1
                model_decision = adapter.decide(current_obs, profile=args.profile)
                current_action = model_decision["action"]
                model_decision["raw_action"] = current_action
                model_decision["decision_index"] = decision_index
                model_decision["simulation_frame"] = frame_count
                model_decision["simulation_time_s"] = round(frame_count / FPS, 6)
                model_decision["observation"] = current_obs.model_dump(mode="json")
                result["model_decisions"].append(model_decision)
                current_decision = model_decision

            executed_action = current_action
            override_reason = None
            if args.execution_mode == "assisted":
                executed_action, override_reason = assisted_action(frame_obs, current_action)
            update_world(player, level, executed_action)
            game_renderer.render(player, level)
            hud.render(
                frame_obs,
                current_decision or model_decision,
                frame_count,
                "running",
                adapter.gpu["name"],
                current_action,
                executed_action,
                override_reason,
            )
            pygame.display.flip()
            if frame_count == 0:
                surface_samples.append(("initial", screen.copy()))
            if frame_count == args.frames_per_decision:
                surface_samples.append(("early", screen.copy()))
            if frame_count % args.frames_per_decision == 0:
                last_regular_surface = screen.copy()
            recorder.record_frame(screen)
            result["trajectory"].append(
                trajectory_row(
                    frame_count,
                    decision_index,
                    current_action,
                    executed_action,
                    override_reason,
                    player,
                )
            )
            frame_count += 1

            if player.has_won:
                end_status = "clear"
            elif player.is_dead:
                end_status = "dead"

        if player.has_won:
            end_status = "clear"
        elif player.is_dead:
            end_status = "dead"

        terminal_obs = TelemetryExtractor.extract(player, level)
        terminal_raw_action = current_action
        terminal_executed_action = current_action
        for terminal_index in range(args.terminal_frames):
            pygame.event.pump()
            game_renderer.render(player, level)
            hud.render(
                terminal_obs,
                current_decision or model_decision,
                frame_count,
                end_status,
                adapter.gpu["name"],
                terminal_raw_action,
                terminal_executed_action,
                None,
            )
            pygame.display.flip()
            recorder.record_frame(screen)
            if terminal_index == 0:
                surface_samples.append(("terminal", screen.copy()))

        frame_dir.mkdir(parents=True, exist_ok=True)
        sample_names = ["initial", "early", "terminal"]
        sample_map = {name: surface for name, surface in surface_samples}
        if "early" not in sample_map and last_regular_surface is not None:
            sample_map["early"] = last_regular_surface
        if "initial" not in sample_map and last_regular_surface is not None:
            sample_map["initial"] = last_regular_surface
        if "terminal" not in sample_map:
            sample_map["terminal"] = screen.copy()
        for name in sample_names:
            sample = sample_map.get(name)
            if sample is not None:
                sample_path = frame_dir / f"{name}.png"
                pygame.image.save(sample, str(sample_path))
                result.setdefault("representative_frames", []).append(str(sample_path))

        result["status"] = "ok"
        result["termination"] = {
            "status": end_status,
            "simulation_frames": frame_count,
            "terminal_static_frames": args.terminal_frames,
            "video_frames": frame_count + args.terminal_frames,
            "simulation_duration_s": round(frame_count / FPS, 6),
            "video_duration_s": round((frame_count + args.terminal_frames) / FPS, 6),
            "progress_pixels": round(float(player.max_x), 4),
            "score": int(player.score),
            "coins": int(player.coins),
            "is_dead": bool(player.is_dead),
            "has_won": bool(player.has_won),
            "decision_count": len(result["model_decisions"]),
            "raw_action_counts": _count_actions(result["trajectory"], "raw_action"),
            "executed_action_counts": _count_actions(result["trajectory"], "executed_action"),
            "override_frames": sum(1 for row in result["trajectory"] if row["overridden"]),
            "override_rate": round(
                sum(1 for row in result["trajectory"] if row["overridden"]) / max(1, frame_count),
                8,
            ),
            "override_reason_counts": _count_reasons(result["trajectory"]),
        }
        result["video"] = {
            "path": str(Path(args.video)),
            "width": SCREEN_WIDTH,
            "height": SCREEN_HEIGHT,
            "fps": FPS,
            "frame_count_expected": frame_count + args.terminal_frames,
            "recorder_frame_count": int(recorder.frame_count),
            "time_basis": "simulation/video frames only; model inference waits are not encoded as extra frames",
        }
        result["finished_at_utc"] = utc_now()
        result["wall_clock_elapsed_s"] = round(time.perf_counter() - wall_start, 6)
        print(json.dumps({"status": "ok", "video": args.video, "json": args.json_path, "termination": result["termination"]}, ensure_ascii=False))
        return 0
    except Exception as error:
        result["status"] = "error"
        result["error"] = {"type": type(error).__name__, "message": redact_error(str(error))}
        result["finished_at_utc"] = utc_now()
        result["wall_clock_elapsed_s"] = round(time.perf_counter() - wall_start, 6)
        write_json(output_path, result)
        print(json.dumps({"status": "error", "error": result["error"]}, ensure_ascii=False))
        return 1
    finally:
        if recorder is not None:
            recorder.close()
        if pygame is not None:
            pygame.quit()
        if result.get("status") == "ok":
            write_json(output_path, result)


if __name__ == "__main__":
    raise SystemExit(main())
