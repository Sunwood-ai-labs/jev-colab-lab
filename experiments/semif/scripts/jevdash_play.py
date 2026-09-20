"""Play the fixed JevDash Level 1 with synchronous SemIf decisions.

The game implementation is loaded from an external, commit-pinned checkout.
This runner deliberately does not import JevDash's live or mock agents.  A
single pinned Qwen3.5-4B causal model is scored by SemIf's direct option-logit
readout every eight simulation frames, and the selected action is applied to
the unmodified game physics.
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
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The fixed game is rendered into a headless Colab VM.  These must be set
# before pygame is imported by any JevDash module.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

GAME_REPOSITORY = "https://github.com/Sunwood-ai-labs/jevdash"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
SEMIF_REPOSITORY = "https://github.com/TheoLeeCJ/semif"
SEMIF_COMMIT = "ca3ba65f142967030ecb453346e94d6f476a69df"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REPOSITORY = "https://huggingface.co/Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SESSION_NAME = "jev-semif-jevdash"
CLI_VERSION = "google-colab-cli 0.6.0"

LEVEL = 1
SEED = 42
FPS = 60
FRAMES_PER_DECISION = 8
MAX_SIMULATION_FRAMES = 1800
STATIC_TERMINAL_FRAMES = 120
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
GAME_VIEW_WIDTH = 880

ACTIONS = (
    "noop",
    "right",
    "right_run",
    "right_jump",
    "right_run_jump",
    "jump",
    "left",
)
ACTION_DESCRIPTIONS = {
    "noop": "Do nothing and preserve the current horizontal momentum.",
    "right": "Move right at walking speed.",
    "right_run": "Move right at running speed.",
    "right_jump": "Move right and jump.",
    "right_run_jump": "Move right at running speed and jump.",
    "jump": "Jump without selecting a horizontal direction.",
    "left": "Move left at walking speed.",
}
PROMPT_QUESTION = (
    "Choose exactly one action macro for the next 8 simulation frames based only "
    "on the supplied JevDash Level 1 observation. Select the macro that best "
    "advances the player to the goal while avoiding hazards."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def scrub_error(error: BaseException) -> str:
    """Keep failure evidence useful without persisting local paths or secrets."""

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
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        return scrub_error(RuntimeError(first_line)).replace("<path>", "")
    except Exception as error:  # noqa: BLE001 - provenance is best effort
        return f"unavailable: {scrub_error(error)}"


def seed_everything(seed: int, torch: Any) -> dict[str, Any]:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = bool(torch.cuda.is_available())
    if cuda_seeded:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "python_random": seed,
        "numpy": seed,
        "torch": seed,
        "torch_cuda": seed if cuda_seeded else None,
        "deterministic_cudnn": True,
        "benchmark_cudnn": False,
    }


def cuda_memory_snapshot(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def gpu_metadata(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_count": int(torch.cuda.device_count()),
        "device_index": 0,
    }


class SemIfActionAdapter:
    """Synchronous direct option-logit adapter with no fallback path."""

    def __init__(self, model_id: str, revision: str, max_tokens: int, torch: Any):
        from semif_phase1.core import load_causal_model

        self.torch = torch
        self.max_tokens = max_tokens
        self.model, self.tokenizer, self.loader_metadata = load_causal_model(model_id, revision)
        self.decision_count = 0
        self.total_inference_wall_seconds = 0.0

    def decide(self, observation: Any, simulation_frame: int) -> dict[str, Any]:
        from semif_phase1.direct import score

        state = observation.model_dump(mode="json")
        row = {
            "id": f"jevdash-frame-{simulation_frame:06d}",
            "state": state,
            "question": PROMPT_QUESTION,
            "options": [
                {"id": action, "description": ACTION_DESCRIPTIONS[action]}
                for action in ACTIONS
            ],
        }
        started = time.perf_counter()
        result = score(self.model, self.tokenizer, row, self.loader_metadata, self.max_tokens)
        self.torch.cuda.synchronize(0)
        wall_seconds = time.perf_counter() - started

        probabilities = [float(value) for value in result["probabilities"]]
        logits = [float(value) for value in result["option_logits"]]
        if len(probabilities) != len(ACTIONS) or len(logits) != len(ACTIONS):
            raise RuntimeError("SemIf returned an unexpected number of action scores")
        if any(not math.isfinite(value) for value in probabilities + logits):
            raise RuntimeError("SemIf returned a non-finite action score")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise RuntimeError("SemIf action probabilities are not normalized")

        selected_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        self.decision_count += 1
        self.total_inference_wall_seconds += wall_seconds
        return {
            "decision_index": self.decision_count,
            "simulation_frame": simulation_frame,
            "simulation_time_seconds": simulation_frame / FPS,
            "observation": state,
            "question": PROMPT_QUESTION,
            "options": row["options"],
            "action": ACTIONS[selected_index],
            "selected_probability": probabilities[selected_index],
            "probabilities": probabilities,
            "probabilities_by_action": dict(zip(ACTIONS, probabilities, strict=True)),
            "option_logits": logits,
            "option_logits_by_action": dict(zip(ACTIONS, logits, strict=True)),
            "input_tokens": int(result["input_tokens"]),
            "forward_seconds": float(result["forward_seconds"]),
            "total_seconds": float(result["total_seconds"]),
            "wall_seconds": float(wall_seconds),
            "prompt_sha256": result["prompt_sha256"],
            "prompt_version": result["prompt_version"],
            "readout": result["readout"],
            "probability_status": result["probability_status"],
            "auxiliary_judgments": {
                "danger": "not_measured",
                "urgency": "not_measured",
                "note": "Only the seven-way action choice controls the game.",
            },
            "fallback_used": False,
        }


class AuditOverlay:
    """Truthful SemIf audit panel drawn over the fixed game's dashboard."""

    def __init__(self, surface: Any, gpu_name: str):
        import pygame

        self.surface = surface
        self.gpu_name = gpu_name
        self.font_xs = pygame.font.SysFont("DejaVu Sans Mono", 11)
        self.font_sm = pygame.font.SysFont("DejaVu Sans Mono", 13)
        self.font_base = pygame.font.SysFont("DejaVu Sans", 15, bold=True)
        self.font_large = pygame.font.SysFont("DejaVu Sans", 21, bold=True)

    def _text(self, value: str, position: tuple[int, int], color: tuple[int, int, int], font: Any) -> None:
        self.surface.blit(font.render(value, True, color), position)

    def render(self, decision: dict[str, Any] | None, frame: int, decision_count: int) -> None:
        import pygame

        x0 = GAME_VIEW_WIDTH
        width = SCREEN_WIDTH - GAME_VIEW_WIDTH
        bg = (8, 15, 29)
        card = (15, 27, 47)
        border = (45, 65, 92)
        primary = (226, 232, 240)
        muted = (148, 163, 184)
        accent = (94, 234, 212)
        warning = (251, 191, 36)

        pygame.draw.rect(self.surface, bg, pygame.Rect(x0, 0, width, SCREEN_HEIGHT))
        pygame.draw.line(self.surface, border, (x0, 0), (x0, SCREEN_HEIGHT), 2)
        self._text("SemIf / Qwen3.5-4B", (x0 + 18, 16), primary, self.font_large)
        self._text("MODEL-DRIVEN  |  NO MOCK / NO FALLBACK", (x0 + 18, 48), accent, self.font_xs)

        header = pygame.Rect(x0 + 16, 76, width - 32, 92)
        pygame.draw.rect(self.surface, card, header, border_radius=7)
        pygame.draw.rect(self.surface, border, header, 1, border_radius=7)
        action = decision["action"] if decision else "waiting"
        model_ms = decision["wall_seconds"] * 1000.0 if decision else 0.0
        self._text(f"ACTION: {action}", (x0 + 28, 88), accent if decision else muted, self.font_base)
        self._text(f"DECISION {decision_count:03d}  |  FRAME {frame:04d}", (x0 + 28, 117), primary, self.font_xs)
        self._text(f"MODEL CALL: {model_ms:7.1f} ms", (x0 + 28, 138), muted, self.font_xs)
        self._text(f"GPU: {self.gpu_name[:33]}", (x0 + 28, 153), muted, self.font_xs)

        sim = pygame.Rect(x0 + 16, 180, width - 32, 91)
        pygame.draw.rect(self.surface, card, sim, border_radius=7)
        pygame.draw.rect(self.surface, border, sim, 1, border_radius=7)
        self._text("SIMULATION CONTRACT", (x0 + 28, 192), primary, self.font_base)
        self._text(f"PHYSICS FPS: {FPS}  |  DECISION STEP: {FRAMES_PER_DECISION}F", (x0 + 28, 217), accent, self.font_xs)
        self._text(f"SIM TIME: {frame / FPS:7.2f}s", (x0 + 28, 237), muted, self.font_xs)
        self._text("Inference waits omitted from simulation time", (x0 + 28, 253), muted, self.font_xs)

        chart = pygame.Rect(x0 + 16, 283, width - 32, 252)
        pygame.draw.rect(self.surface, card, chart, border_radius=7)
        pygame.draw.rect(self.surface, border, chart, 1, border_radius=7)
        self._text("SEVEN-WAY ACTION PROBABILITIES", (x0 + 28, 295), primary, self.font_base)
        probabilities = decision["probabilities"] if decision else [0.0] * len(ACTIONS)
        selected = decision["action"] if decision else None
        bar_y = 327
        for action_name, probability in zip(ACTIONS, probabilities, strict=True):
            text_color = accent if action_name == selected else muted
            self._text(f"{action_name:<14}", (x0 + 28, bar_y), text_color, self.font_xs)
            bar_bg = pygame.Rect(x0 + 154, bar_y + 2, 128, 10)
            pygame.draw.rect(self.surface, (29, 43, 62), bar_bg, border_radius=3)
            fill_width = int(128 * probability)
            if fill_width:
                pygame.draw.rect(
                    self.surface,
                    accent if action_name == selected else (71, 85, 105),
                    pygame.Rect(x0 + 154, bar_y + 2, fill_width, 10),
                    border_radius=3,
                )
            self._text(f"{probability * 100:5.1f}%", (x0 + 293, bar_y), text_color, self.font_xs)
            bar_y += 27

        warning_card = pygame.Rect(x0 + 16, 548, width - 32, 83)
        pygame.draw.rect(self.surface, (45, 33, 12), warning_card, border_radius=7)
        pygame.draw.rect(self.surface, (132, 93, 20), warning_card, 1, border_radius=7)
        self._text("DANGER / URGENCY: NOT MEASURED", (x0 + 28, 561), warning, self.font_base)
        self._text("SemIf emits only the seven-way action choice.", (x0 + 28, 588), muted, self.font_xs)
        self._text("No numeric safety judgment is shown or used.", (x0 + 28, 606), muted, self.font_xs)


def import_game(game_root: Path, commit_file: Path) -> dict[str, Any]:
    actual_commit = commit_file.read_text(encoding="utf-8").strip()
    if actual_commit != GAME_COMMIT:
        raise RuntimeError("game commit marker does not match the required fixed commit")
    source_root = game_root / "src"
    if not source_root.is_dir():
        raise RuntimeError("game source directory is missing")
    sys.path.insert(0, str(source_root))

    import pygame
    from jev_platformer.controller.actions import ALL_ACTIONS
    from jev_platformer.engine.constants import (
        FPS as GAME_FPS,
        GAME_VIEW_WIDTH as GAME_VIEW,
        SCREEN_HEIGHT as GAME_HEIGHT,
        SCREEN_WIDTH as GAME_WIDTH,
    )
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor
    from jev_platformer.ui.dashboard import DashboardRenderer
    from jev_platformer.ui.renderer import GameRenderer
    from jev_platformer.ui.video_recorder import VideoRecorder

    if (GAME_WIDTH, GAME_HEIGHT, GAME_VIEW, GAME_FPS) != (SCREEN_WIDTH, SCREEN_HEIGHT, GAME_VIEW_WIDTH, FPS):
        raise RuntimeError("fixed game display or FPS constants changed")
    if tuple(ALL_ACTIONS) != ACTIONS:
        raise RuntimeError("fixed game action ordering changed")
    return {
        "pygame": pygame,
        "Player": Player,
        "Level": Level,
        "TelemetryExtractor": TelemetryExtractor,
        "DashboardRenderer": DashboardRenderer,
        "GameRenderer": GameRenderer,
        "VideoRecorder": VideoRecorder,
        "actual_commit": actual_commit,
    }


def record_frame(recorder: Any, screen: Any) -> None:
    process = recorder.process
    if process is None or process.poll() is not None:
        raise RuntimeError("FFmpeg recorder is not running")
    recorder.record_frame(screen)
    if recorder.process is None:
        raise RuntimeError("FFmpeg recorder stopped while writing a frame")


def base_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "jev-semif-jevdash-episode-v1",
        "status": "starting",
        "started_at_utc": utc_now(),
        "runtime": {
            "label": "Google Colab GPU runtime",
            "colab_cli": CLI_VERSION,
            "session_name": SESSION_NAME,
            "expected_gpu": "L4",
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
            "level": args.level,
            "seed": args.seed,
            "physics_fps": args.fps,
            "frames_per_decision": args.frames_per_decision,
            "max_simulation_frames": args.max_frames,
            "static_terminal_frames": args.static_terminal_frames,
            "action_order": list(ACTIONS),
            "physics_source": "unmodified JevDash fixed checkout; runner mirrors cli.run_play order",
        },
        "source": {
            "implementation": "TheoLeeCJ/SemIf direct native next-token logit readout",
            "semif_repository": SEMIF_REPOSITORY,
            "semif_commit": SEMIF_COMMIT,
            "model_id": args.model,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": args.revision,
            "model_license": "Apache-2.0 (upstream model card)",
            "training_source": "upstream Qwen/Qwen3.5-4B checkpoint; no local fine-tuning",
            "fallback_used": False,
            "mock_agent_used": False,
            "auxiliary_judgments": {
                "danger": "not_measured",
                "urgency": "not_measured",
            },
        },
        "model_options": {
            "dtype": "bfloat16",
            "max_input_tokens": args.max_tokens,
            "decision_semantics": "conditional option score; uncalibrated as decision confidence",
            "readout": "full-vocabulary last-position logits restricted to seven declared action slots, then softmax",
        },
        "video": {
            "filename": args.output_video.name,
            "resolution": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
            "fps": FPS,
            "codec": "H.264 libx264",
            "pixel_format": "yuv420p",
            "time_basis": "simulation frames at 60 FPS; synchronous model waits are omitted from video time",
            "static_terminal_frames_requested": args.static_terminal_frames,
            "frames_written": 0,
            "static_terminal_frames_written": 0,
            "ffmpeg_version": ffmpeg_version(),
        },
        "decisions": [],
        "frame_trace": [],
        "summary": {},
        "errors": [],
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    payload = base_payload(args)
    recorder = None
    pygame = None
    pygame_initialized = False
    frame_count = 0
    outcome = "model_or_runner_error"
    terminal_frame: int | None = None
    action_counts: Counter[str] = Counter()
    torch = None
    adapter: SemIfActionAdapter | None = None
    player = None
    try:
        import torch as torch_module

        torch = torch_module
        payload["runtime"]["seed_sources"] = seed_everything(args.seed, torch)
        payload["runtime"]["dependencies"] = dependency_versions()
        payload["runtime"].update(
            {
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this is not a GPU result")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}")
        gpu = gpu_metadata(torch)
        payload["runtime"]["gpu"] = gpu
        if "l4" not in gpu["name"].lower():
            raise RuntimeError(f"expected an L4 GPU, got {gpu['name']!r}")

        load_started = time.perf_counter()
        adapter = SemIfActionAdapter(args.model, args.revision, args.max_tokens, torch)
        torch.cuda.synchronize(0)
        payload["runtime"]["model_load_seconds"] = time.perf_counter() - load_started
        payload["runtime"]["load_memory"] = cuda_memory_snapshot(torch)
        torch.cuda.reset_peak_memory_stats(0)

        game = import_game(args.game_root, args.game_commit_file)
        payload["game"]["actual_commit"] = game["actual_commit"]
        pygame = game["pygame"]
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("JevDash: System One / SemIf")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        level = game["Level"](args.level)
        player = game["Player"](level.start_pos[0], level.start_pos[1])
        game_renderer = game["GameRenderer"](screen)
        dashboard_renderer = game["DashboardRenderer"](screen, offset_x=GAME_VIEW_WIDTH)
        audit_overlay = AuditOverlay(screen, gpu["name"])
        recorder = game["VideoRecorder"](str(args.output_video), SCREEN_WIDTH, SCREEN_HEIGHT, fps=FPS)
        if recorder.process is None:
            raise RuntimeError("FFmpeg recorder failed to start")

        current_action = "noop"
        current_decision: dict[str, Any] | None = None
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    outcome = "quit"
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    outcome = "quit"
            if not running:
                break

            observation = game["TelemetryExtractor"].extract(player, level)
            if frame_count % args.frames_per_decision == 0:
                current_decision = adapter.decide(observation, frame_count)
                payload["decisions"].append(current_decision)
                current_action = current_decision["action"]

            # This is the fixed JevDash cli.run_play update order.
            player.apply_action(current_action)
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

            game_renderer.render(player, level)
            dashboard_renderer.render(
                obs=observation,
                decision=None,
                is_ai_mode=True,
                fps=FPS,
            )
            audit_overlay.render(current_decision, frame_count, adapter.decision_count)
            pygame.display.flip()
            record_frame(recorder, screen)

            action_counts[current_action] += 1
            payload["frame_trace"].append(
                {
                    "frame": frame_count,
                    "simulation_time_seconds": frame_count / FPS,
                    "action": current_action,
                    "decision_index": current_decision["decision_index"] if current_decision else None,
                    "x": round(float(player.x), 3),
                    "y": round(float(player.y), 3),
                    "vx": round(float(player.vx), 3),
                    "vy": round(float(player.vy), 3),
                    "progress_pixels": round(float(player.max_x), 3),
                    "score": int(player.score),
                    "coins": int(player.coins),
                    "is_dead": bool(player.is_dead),
                    "has_won": bool(player.has_won),
                }
            )
            frame_count += 1

            if player.has_won:
                outcome = "clear"
                terminal_frame = frame_count
                break
            if player.is_dead:
                outcome = "dead"
                terminal_frame = frame_count
                break
            if frame_count >= args.max_frames:
                outcome = "timeout"
                terminal_frame = frame_count
                break

        if outcome in {"clear", "dead"}:
            for _ in range(args.static_terminal_frames):
                record_frame(recorder, screen)
            payload["video"]["static_terminal_frames_written"] = args.static_terminal_frames
        elif outcome not in {"timeout", "quit"}:
            outcome = "quit"

        payload["status"] = "completed"
        payload["outcome"] = outcome
        payload["finished_at_utc"] = utc_now()
        payload["runtime"]["inference_peak_memory"] = cuda_memory_snapshot(torch)
        payload["summary"] = {
            "outcome": outcome,
            "terminal_frame": terminal_frame,
            "simulation_time_seconds": frame_count / FPS,
            "progress_pixels": round(float(player.max_x), 3),
            "score": int(player.score),
            "coins": int(player.coins),
            "decision_count": adapter.decision_count,
            "action_counts": dict(action_counts),
            "total_inference_wall_seconds": adapter.total_inference_wall_seconds,
            "peak_vram": payload["runtime"]["inference_peak_memory"],
        }
        return payload, 0
    except Exception as error:  # noqa: BLE001 - sanitized failure evidence is required
        payload["status"] = "failed"
        payload["outcome"] = "model_or_runner_error"
        payload["finished_at_utc"] = utc_now()
        payload["errors"].append({"type": error.__class__.__name__, "message": scrub_error(error)})
        if torch is not None and torch.cuda.is_available():
            try:
                payload["runtime"]["inference_peak_memory"] = cuda_memory_snapshot(torch)
            except Exception:  # noqa: BLE001 - preserve the original failure
                pass
        return payload, 1
    finally:
        if recorder is not None:
            payload["video"]["frames_written"] = int(recorder.frame_count)
            try:
                recorder.close()
            except Exception as error:  # noqa: BLE001 - preserve cleanup evidence
                payload["errors"].append({"type": "RecorderCloseError", "message": scrub_error(error)})
        if pygame_initialized and pygame is not None:
            pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, default=Path("/content/jevdash"))
    parser.add_argument("--game-commit-file", type=Path, default=Path("/content/jevdash-commit.txt"))
    parser.add_argument("--output-video", type=Path, default=Path("/content/semif-jevdash.mp4"))
    parser.add_argument("--output-json", type=Path, default=Path("/content/semif-jevdash-episode.json"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--frames-per-decision", type=int, default=FRAMES_PER_DECISION)
    parser.add_argument("--max-frames", type=int, default=MAX_SIMULATION_FRAMES)
    parser.add_argument("--static-terminal-frames", type=int, default=STATIC_TERMINAL_FRAMES)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_json}")
    if args.output_video.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_video}")
    fixed = {
        "level": (args.level, LEVEL),
        "seed": (args.seed, SEED),
        "fps": (args.fps, FPS),
        "frames_per_decision": (args.frames_per_decision, FRAMES_PER_DECISION),
        "max_frames": (args.max_frames, MAX_SIMULATION_FRAMES),
        "static_terminal_frames": (args.static_terminal_frames, STATIC_TERMINAL_FRAMES),
    }
    wrong = [name for name, (value, expected) in fixed.items() if value != expected]
    if wrong:
        parser.error(f"fixed JevDash experiment values changed: {', '.join(wrong)}")
    if args.model != MODEL_ID or args.revision != MODEL_REVISION:
        parser.error("this evidence run requires the pinned Qwen3.5-4B model revision")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    payload, exit_code = run(args)
    write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "outcome": payload.get("outcome"),
                "output_json": str(args.output_json),
                "output_video": str(args.output_video),
                "frames_written": payload["video"]["frames_written"],
                "errors": payload["errors"],
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
