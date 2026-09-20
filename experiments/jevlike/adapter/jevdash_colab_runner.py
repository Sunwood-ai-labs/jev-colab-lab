"""Run the pinned JevDash game with a real Jevlike TinyScorer on Colab T4.

The game source is fetched outside this repository at a fixed commit. The
adapter imports its engine, telemetry extractor, renderer, and VideoRecorder
without modifying the game checkout. The only controller is the Jevlike model;
there is no mock, rule fallback, live API, or hidden game-specific training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


GAME_URL = "https://github.com/Sunwood-ai-labs/jevdash.git"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
JEVLIKE_URL = "https://github.com/vinnylarouge/jevlike.git"
JEVLIKE_COMMIT = "94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452"
SEED = 42
LEVEL = 1
FPS = 60
FRAMES_PER_DECISION = 8
MAX_SIMULATION_FRAMES = 1800
TERMINAL_HOLD_FRAMES = 120
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MODEL_NAME = "Jevlike TinyScorer (synthetic badge checkpoint)"
ACTION_SPACE = (
    "noop",
    "right",
    "right_run",
    "right_jump",
    "right_run_jump",
    "jump",
    "left",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_command(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        rendered = " ".join(command)
        raise RuntimeError(f"command failed with exit {result.returncode}: {rendered}")


def fetch_fixed_repo(url: str, commit: str, root: Path) -> tuple[Path, float]:
    started = time.perf_counter()
    root = root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and not (root / ".git").is_dir():
        raise RuntimeError(f"refusing to reuse a non-git path: {root}")
    if not root.exists():
        run_command(["git", "init", "--quiet", str(root)])
        run_command(["git", "-C", str(root), "remote", "add", "origin", url])
    run_command([
        "git", "-C", str(root), "fetch", "--quiet", "--depth", "1", "origin", commit,
    ])
    run_command([
        "git", "-C", str(root), "checkout", "--quiet", "--detach", commit,
    ])
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual != commit:
        raise RuntimeError(f"checkout mismatch: expected {commit}, got {actual}")
    return root, time.perf_counter() - started


def install_colab_dependencies(game_root: Path, jevlike_root: Path) -> float:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required on the Colab VM")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required by the fixed game's VideoRecorder")
    started = time.perf_counter()
    run_command([
        uv, "pip", "install", "--system", "--quiet",
        "pygame>=2.6.0", "pydantic>=2.7.0",
    ])
    run_command([uv, "pip", "install", "--system", "--quiet", "-e", str(jevlike_root)])
    # Import the fixed game by source path; its live API dependencies are not
    # needed because this runner never imports or invokes the live controller.
    sys.path.insert(0, str(game_root / "src"))
    return time.perf_counter() - started


def set_seeds(torch: Any, numpy: Any) -> None:
    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_cuda(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def snapshot_trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_jevlike_checkpoint(
    torch: Any,
    output_dir: Path,
    jevlike_root: Path,
    device: Any,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from jevlike.data import JsonlDataset, write_synthetic
    from jevlike.model import load_checkpoint, make_system

    data_dir = output_dir / "training-data"
    checkpoint = output_dir / "tiny_scorer.pt"
    train_size, validation_size, test_size = 512, 128, 128
    epochs, batch_size = 4, 64
    learning_rate = 2e-3
    width, rank = 64, 64
    context_tokens, option_tokens = 192, 32

    write_synthetic(
        data_dir,
        {"train": train_size, "validation": validation_size, "test": test_size},
        SEED,
    )
    config = {
        "encoder": "tiny",
        "hf_model": "Qwen/Qwen2.5-0.5B",
        "width": width,
        "rank": rank,
        "context_tokens": context_tokens,
        "option_tokens": option_tokens,
    }
    model, collator = make_system(config, device)
    train_loader = DataLoader(
        JsonlDataset(data_dir / "train.jsonl"),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        JsonlDataset(data_dir / "validation.jsonl"),
        batch_size=batch_size,
        collate_fn=collator,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    epoch_logs: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync_cuda(torch, device)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for host_batch in train_loader:
            batch = {name: tensor.to(device) for name, tensor in host_batch.items()}
            loss = F.cross_entropy(model(batch), batch["labels"])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            total += float(loss.detach()) * batch["labels"].numel()
            count += batch["labels"].numel()

        model.eval()
        validation_total, validation_count = 0.0, 0
        with torch.no_grad():
            for host_batch in validation_loader:
                batch = {name: tensor.to(device) for name, tensor in host_batch.items()}
                validation_loss = F.cross_entropy(
                    model(batch), batch["labels"], reduction="sum",
                )
                validation_total += float(validation_loss)
                validation_count += batch["labels"].numel()
        validation_nll = validation_total / validation_count
        if validation_nll < best_loss:
            best_loss = validation_nll
            best_state = snapshot_trainable_state(model)
        epoch_logs.append({
            "epoch": epoch + 1,
            "train_nll": total / count,
            "validation_nll": validation_nll,
            "device": str(device),
        })
    sync_cuda(torch, device)
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("no best Jevlike checkpoint state was produced")
    torch.save({"config": config, "state_dict": best_state}, checkpoint)
    training_peak = None
    if device.type == "cuda":
        training_peak = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

    loaded_model, loaded_collator, loaded_config = load_checkpoint(checkpoint, device)
    loaded_model.eval()
    model_info = {
        "name": MODEL_NAME,
        "encoder": "tiny",
        "initialization": "TinyScorer initialized from torch seed 42; no pretrained weights.",
        "training_data": "jevlike.data.write_synthetic generated badge-selection JSONL; no game data.",
        "training_seed": SEED,
        "train_examples": train_size,
        "validation_examples": validation_size,
        "test_examples": test_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "width": width,
        "rank": rank,
        "context_tokens": context_tokens,
        "option_tokens": option_tokens,
        "training_seconds": training_seconds,
        "training_peak_vram": training_peak,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "upstream_repository": JEVLIKE_URL,
        "upstream_commit": JEVLIKE_COMMIT,
        "game_specific_training": False,
    }
    return loaded_model, loaded_collator, model_info, {"epochs": epoch_logs, "config": loaded_config}


def make_context(obs: Any) -> str:
    observation = obs.model_dump(mode="json")
    return (
        "Choose exactly one supplied JevDash action macro for the current observation.\n"
        "Do not invent an action; the seven supplied options are the complete action space.\n"
        + json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def decide(
    torch: Any,
    model: Any,
    collator: Any,
    choice_example_cls: Any,
    obs: Any,
    device: Any,
) -> dict[str, Any]:
    context = make_context(obs)
    started = time.perf_counter()
    sync_cuda(torch, device)
    batch = collator([choice_example_cls(context, ACTION_SPACE, 0)])
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    transfer_done = time.perf_counter()
    with torch.no_grad():
        logits = model(batch)
        probabilities = logits.softmax(-1)[0, :len(ACTION_SPACE)]
    sync_cuda(torch, device)
    finished = time.perf_counter()
    values = [float(value) for value in probabilities.detach().cpu()]
    probability_map = dict(zip(ACTION_SPACE, values))
    selected_index = max(range(len(values)), key=values.__getitem__)
    return {
        "action": ACTION_SPACE[selected_index],
        "probabilities": probability_map,
        "probability_sum": sum(values),
        "inference_wall_ms": (finished - started) * 1000.0,
        "forward_wall_ms": (finished - transfer_done) * 1000.0,
        "observation": obs.model_dump(mode="json"),
        "context": context,
        "options": list(ACTION_SPACE),
        "mock_used": False,
        "fallback_used": False,
    }


class JevlikeHUD:
    """A truthful HUD that never labels this scratch model as live Jev."""

    def __init__(self, surface: Any, offset_x: int, hud_width: int, hud_height: int):
        import pygame

        self.surface = surface
        self.offset_x = offset_x
        self.hud_width = hud_width
        self.hud_height = hud_height
        self.font_xs = pygame.font.SysFont("Consolas, Menlo, monospace", 11)
        self.font_sm = pygame.font.SysFont("Consolas, Menlo, monospace", 13)
        self.font_base = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 21, bold=True)
        self.bg = (15, 18, 28)
        self.card = (22, 27, 42)
        self.border = (38, 45, 66)
        self.primary = (248, 250, 252)
        self.muted = (148, 163, 184)
        self.accent = (56, 189, 248)
        self.warning = (245, 158, 11)
        self.danger = (239, 68, 68)
        self.success = (34, 197, 94)

    def text(self, value: str, position: tuple[int, int], color: tuple[int, int, int], font: Any = None) -> None:
        self.surface.blit((font or self.font_sm).render(value, True, color), position)

    def render(
        self,
        obs: Any,
        decision: dict[str, Any] | None,
        simulation_frame: int,
        status: str,
        gpu_name: str,
    ) -> None:
        import pygame

        x = self.offset_x + 18
        y = 16
        pygame.draw.rect(
            self.surface, self.bg,
            pygame.Rect(self.offset_x, 0, self.hud_width, self.hud_height),
        )
        pygame.draw.line(
            self.surface, self.border,
            (self.offset_x, 0), (self.offset_x, self.hud_height), 2,
        )
        self.text("JevDash / MODEL-DRIVEN", (x, y), self.primary, self.font_large)
        y += 30
        self.text("Jevlike TinyScorer", (x, y), self.accent, self.font_base)
        y += 21
        self.text("scratch byte encoder / synthetic badge training", (x, y), self.muted, self.font_xs)
        y += 17
        self.text(f"GPU: {gpu_name}", (x, y), self.muted, self.font_xs)
        y += 25

        status_color = self.success if status == "CLEARED" else self.danger if status == "DEAD" else self.warning if status != "RUNNING" else self.accent
        self.text(f"STATUS: {status}", (x, y), status_color, self.font_base)
        y += 23
        self.text(f"MODEL DECISION: {decision['action'] if decision else '--'}", (x, y), self.primary, self.font_sm)
        y += 22
        self.text(
            f"SIMULATION TIME (inference waits omitted): {simulation_frame / FPS:.2f}s",
            (x, y), self.primary, self.font_xs,
        )
        y += 16
        wait_ms = decision["inference_wall_ms"] if decision else None
        wait_text = f"last inference wall wait: {wait_ms:.2f} ms" if wait_ms is not None else "last inference wall wait: --"
        self.text(wait_text, (x, y), self.muted, self.font_xs)
        y += 16
        self.text("DANGER / URGENCY: NOT MEASURED", (x, y), self.warning, self.font_xs)
        y += 28

        card = pygame.Rect(x, y, self.hud_width - 36, 245)
        pygame.draw.rect(self.surface, self.card, card, border_radius=8)
        pygame.draw.rect(self.surface, self.border, card, 1, border_radius=8)
        self.text("MODEL ACTION PROBABILITIES", (x + 12, y + 10), self.primary, self.font_base)
        y += 39
        probabilities = decision["probabilities"] if decision else {}
        for action in ACTION_SPACE:
            prob = float(probabilities.get(action, 0.0))
            color = self.accent if decision and action == decision["action"] else self.muted
            self.text(f"{action:<14}", (x + 12, y), color, self.font_xs)
            pygame.draw.rect(self.surface, (30, 36, 54), pygame.Rect(x + 126, y + 2, 145, 10), border_radius=3)
            fill = int(145 * max(0.0, min(1.0, prob)))
            if fill:
                pygame.draw.rect(self.surface, color, pygame.Rect(x + 126, y + 2, fill, 10), border_radius=3)
            self.text(f"{prob * 100:5.1f}%", (x + 280, y), color, self.font_xs)
            y += 27
        y += 11
        self.text("OBSERVATION", (x, y), self.primary, self.font_base)
        y += 23
        self.text(f"progress: {obs.episode.progress_pixels:.1f}px", (x, y), self.muted, self.font_xs)
        self.text(f"goal: {obs.episode.goal_distance_pixels:.1f}px", (x + 145, y), self.muted, self.font_xs)
        y += 17
        self.text(f"grounded: {obs.player.grounded}", (x, y), self.muted, self.font_xs)
        self.text(f"gap: {obs.terrain.gap_ahead}", (x + 145, y), self.muted, self.font_xs)
        y += 17
        self.text("60 FPS simulation;", (x, y), self.warning, self.font_xs)
        self.text("inference waits excluded", (x, y + 14), self.warning, self.font_xs)


def run_episode(
    torch: Any,
    model: Any,
    collator: Any,
    output_path: Path,
    gpu_name: str,
    game_root: Path,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    import pygame

    from jev_platformer.controller.actions import Action
    from jev_platformer.engine.constants import (
        GAME_BRAND, GAME_TITLE, GAME_VIEW_WIDTH, HUD_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH,
    )
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor
    from jev_platformer.ui.renderer import GameRenderer
    from jev_platformer.ui.video_recorder import VideoRecorder
    from jevlike.data import ChoiceExample

    if tuple(Action(action).value for action in ACTION_SPACE) != ACTION_SPACE:
        raise RuntimeError("fixed game action space differs from the adapter action space")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    level = Level(LEVEL)
    player = Player(level.start_pos[0], level.start_pos[1])
    game_renderer = GameRenderer(screen)
    hud = JevlikeHUD(screen, GAME_VIEW_WIDTH, HUD_WIDTH, SCREEN_HEIGHT)
    recorder = VideoRecorder(str(output_path), SCREEN_WIDTH, SCREEN_HEIGHT, fps=FPS)
    if recorder.process is None:
        pygame.quit()
        raise RuntimeError("fixed game VideoRecorder could not start ffmpeg")

    current_action = Action.NOOP.value
    current_decision: dict[str, Any] | None = None
    frame_count = 0
    decisions: list[dict[str, Any]] = []
    frame_trace: list[dict[str, Any]] = []
    terminal_reason: str | None = None
    wall_started = time.perf_counter()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminal_reason = "quit"
                    running = False

            if not running:
                break

            obs = TelemetryExtractor.extract(player, level)
            if frame_count % FRAMES_PER_DECISION == 0:
                current_decision = decide(torch, model, collator, ChoiceExample, obs, torch.device("cuda"))
                current_decision.update({
                    "decision_index": len(decisions),
                    "simulation_frame": frame_count,
                    "simulation_time_seconds": frame_count / FPS,
                })
                decisions.append(current_decision)
                current_action = current_decision["action"]

            # This physics and collision order follows fixed commit cli.run_play.
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

            status = "CLEARED" if player.has_won else "DEAD" if player.is_dead else "RUNNING"
            game_renderer.render(player, level)
            hud.render(obs, current_decision, frame_count, status, gpu_name)
            pygame.display.flip()
            recorder.record_frame(screen)
            frame_trace.append({
                "frame": frame_count,
                "simulation_time_seconds": frame_count / FPS,
                "video_time_seconds": frame_count / FPS,
                "action": current_action,
                "x": round(player.x, 3),
                "y": round(player.y, 3),
                "vx": round(player.vx, 3),
                "vy": round(player.vy, 3),
                "progress_pixels": round(player.max_x, 3),
                "is_dead": player.is_dead,
                "has_won": player.has_won,
                "terminal_hold": False,
            })
            clock.tick(FPS)
            frame_count += 1

            if player.has_won:
                terminal_reason = "clear"
                running = False
            elif player.is_dead:
                terminal_reason = "death"
                running = False
            elif frame_count >= MAX_SIMULATION_FRAMES:
                terminal_reason = "timeout"
                running = False

        if terminal_reason is None:
            terminal_reason = "quit"

        terminal_frame = frame_count
        final_obs = TelemetryExtractor.extract(player, level)
        for _ in range(TERMINAL_HOLD_FRAMES):
            status = "CLEARED" if player.has_won else "DEAD" if player.is_dead else "TIMEOUT"
            game_renderer.render(player, level)
            hud.render(final_obs, current_decision, terminal_frame, status, gpu_name)
            pygame.display.flip()
            recorder.record_frame(screen)
            frame_trace.append({
                "frame": frame_count,
                "simulation_time_seconds": terminal_frame / FPS,
                "video_time_seconds": frame_count / FPS,
                "action": current_action,
                "x": round(player.x, 3),
                "y": round(player.y, 3),
                "vx": round(player.vx, 3),
                "vy": round(player.vy, 3),
                "progress_pixels": round(player.max_x, 3),
                "is_dead": player.is_dead,
                "has_won": player.has_won,
                "terminal_hold": True,
            })
            clock.tick(FPS)
            frame_count += 1
    finally:
        recorder.close()
        pygame.quit()

    wall_seconds = time.perf_counter() - wall_started
    if torch.cuda.is_available():
        inference_peak = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    else:
        inference_peak = None
    return {
        "status": "success",
        "generated_at_utc": utc_now(),
        "game": {
            "repository": GAME_URL,
            "commit": GAME_COMMIT,
            "level": LEVEL,
            "physics_fps": FPS,
            "frames_per_decision": FRAMES_PER_DECISION,
            "max_simulation_frames": MAX_SIMULATION_FRAMES,
            "terminal_hold_frames": TERMINAL_HOLD_FRAMES,
            "action_space": list(ACTION_SPACE),
            "engine_source": "fixed commit cli.run_play physics/collision/render order; game checkout was not modified",
        },
        "model": model_info,
        "runtime": {
            "backend": "local Jevlike checkpoint on Colab GPU",
            "model_name": MODEL_NAME,
            "gpu_name": gpu_name,
            "device": "cuda",
            "precision": "float32",
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", None),
            "torch_cuda": torch.version.cuda,
            "seed": SEED,
            "mock_used": False,
            "fallback_used": False,
            "live_api_used": False,
        },
        "trajectory": {
            "terminal_reason": terminal_reason,
            "terminal_simulation_frame": terminal_frame,
            "terminal_simulation_time_seconds": terminal_frame / FPS,
            "final_progress_pixels": round(player.max_x, 3),
            "final_score": player.score,
            "final_coins": player.coins,
            "final_is_dead": player.is_dead,
            "final_has_won": player.has_won,
            "decision_count": len(decisions),
            "decisions": decisions,
            "frames": frame_trace,
        },
        "video": {
            "path": output_path.name,
            "width": SCREEN_WIDTH,
            "height": SCREEN_HEIGHT,
            "fps": FPS,
            "recorded_frames": len(frame_trace),
            "duration_seconds": len(frame_trace) / FPS,
            "simulation_frames_excluding_terminal_hold": terminal_frame,
            "wall_clock_seconds": wall_seconds,
            "time_basis": "MP4 timestamps use every recorded frame at 60 FPS; physics simulation time is held at the terminal frame during the static terminal hold, and synchronous model waits are excluded.",
            "inference_peak_vram": inference_peak,
        },
        "artifacts": {
            "episode_json": "jevdash-jevlike-episode.json",
            "video_mp4": output_path.name,
        },
    }


def main() -> None:
    import numpy
    import torch

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a CPU or mock substitute")
    device = torch.device("cuda")
    set_seeds(torch, numpy)
    gpu_name = torch.cuda.get_device_name(device)
    output_dir = Path("/content/jevlike-jevdash-output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    game_root = Path("/content/jevdash-fixed").resolve()
    jevlike_root = Path("/content/jevlike-fixed").resolve()
    game_root, game_checkout_seconds = fetch_fixed_repo(GAME_URL, GAME_COMMIT, game_root)
    jevlike_root, jevlike_checkout_seconds = fetch_fixed_repo(JEVLIKE_URL, JEVLIKE_COMMIT, jevlike_root)
    install_seconds = install_colab_dependencies(game_root, jevlike_root)
    sys.path.insert(0, str(jevlike_root))
    importlib = __import__("importlib")
    importlib.invalidate_caches()
    from jevlike.model import make_system

    # The training helper imports the upstream package after installation and
    # then reloads the saved checkpoint before any game decision.
    model, collator, model_info, training_info = train_jevlike_checkpoint(
        torch, output_dir, jevlike_root, device,
    )
    video_path = output_dir / "jevdash-jevlike.mp4"
    episode = run_episode(
        torch, model, collator, video_path, gpu_name, game_root, model_info,
    )
    episode["provenance"] = {
        "game_checkout_seconds": game_checkout_seconds,
        "jevlike_checkout_seconds": jevlike_checkout_seconds,
        "uv_install_seconds": install_seconds,
        "training_logs": training_info["epochs"],
        "training_config": training_info["config"],
        "adapter": "experiments/jevlike/adapter/jevdash_colab_runner.py",
    }
    episode_path = output_dir / "jevdash-jevlike-episode.json"
    episode_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Do not print the trajectory or any environment variables. The JSON is
    # sanitized and contains only model/game measurements.
    summary = {
        "status": episode["status"],
        "gpu": episode["runtime"]["gpu_name"],
        "terminal_reason": episode["trajectory"]["terminal_reason"],
        "terminal_frame": episode["trajectory"]["terminal_simulation_frame"],
        "progress_pixels": episode["trajectory"]["final_progress_pixels"],
        "decision_count": episode["trajectory"]["decision_count"],
        "recorded_frames": episode["video"]["recorded_frames"],
        "video_duration_seconds": episode["video"]["duration_seconds"],
        "episode_json": str(episode_path),
        "video_mp4": str(video_path),
    }
    print("RESULT_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
