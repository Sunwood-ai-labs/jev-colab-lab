"""Train and run a Jevlike game-aware controller on the pinned JevDash level.

The original Jevlike capture used synthetic badge data and a long English
prefix which pushed every changing game field past the 192-byte TinyScorer
context window. This runner keeps the upstream TinyScorer architecture, but
trains a separately identified checkpoint on sanitized, game-teacher labels
and sends a compact state-first context. The primary episode is model-only;
the assisted episode uses the fixed game's per-physics-frame safety reflex and
records every raw/executed action and override reason separately.

Normal execution requires CUDA and is intended for a real Colab T4. The
``--audit-only`` path is a local deterministic fixture and never claims GPU
success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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
CONTEXT_TOKENS = 192
OPTION_TOKENS = 32
ACTION_SPACE = (
    "noop",
    "right",
    "right_run",
    "right_jump",
    "right_run_jump",
    "jump",
    "left",
)
MODEL_NAME = "Jevlike TinyScorer (JevDash teacher checkpoint)"


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
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}"
        )


def fetch_fixed_repo(url: str, commit: str, root: Path) -> tuple[Path, float]:
    started = time.perf_counter()
    root = root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and not (root / ".git").is_dir():
        raise RuntimeError(f"refusing to reuse a non-git path: {root}")
    if not root.exists():
        run_command(["git", "init", "--quiet", str(root)])
        run_command(["git", "-C", str(root), "remote", "add", "origin", url])
    run_command(["git", "-C", str(root), "fetch", "--quiet", "--depth", "1", "origin", commit])
    run_command(["git", "-C", str(root), "checkout", "--quiet", "--detach", commit])
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual != commit:
        raise RuntimeError(f"checkout mismatch: expected {commit}, got {actual}")
    return root, time.perf_counter() - started


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def install_colab_dependencies(jevlike_root: Path, game_root: Path) -> float:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required on the Colab VM")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required by the fixed game's VideoRecorder")
    started = time.perf_counter()
    run_command([uv, "pip", "install", "--system", "--quiet", "pygame>=2.6.0", "pydantic>=2.7.0"])
    run_command([uv, "pip", "install", "--system", "--quiet", "-e", str(jevlike_root)])
    sys.path.insert(0, str(game_root / "src"))
    return time.perf_counter() - started


def _q(value: float | int | None, scale: float, cap: int = 999) -> str:
    if value is None:
        number = cap
    else:
        number = int(round(float(value) / scale))
    number = max(-cap, min(cap, number))
    sign = "m" if number < 0 else "p"
    return f"{sign}{abs(number):03d}"


def _enemy_fields(obs: Any) -> tuple[int, float | None, float | None, int | None]:
    enemy = obs.hazard.nearest_enemy
    if enemy is None:
        return 0, None, None, None
    return (
        1,
        enemy.distance_pixels,
        enemy.relative_velocity_x,
        enemy.estimated_contact_frames,
    )


def legacy_context(obs: Any) -> str:
    """The previous prompt, retained only for the truncation audit."""

    observation = obs.model_dump(mode="json")
    return (
        "Choose exactly one supplied JevDash action macro for the current observation.\n"
        "Do not invent an action; the seven supplied options are the complete action space.\n"
        + json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def safety_reflex(obs: Any, raw_action: str) -> tuple[str, str | None]:
    """The fixed game's reflex, evaluated each physics frame for the assisted run."""

    p, h, t, e = obs.player, obs.hazard, obs.terrain, obs.episode
    gap_distance = t.gap_distance_tiles if t.gap_ahead else 99.0
    obstacle_distance = t.obstacle_distance_tiles if t.obstacle_ahead else 99.0
    enemy_critical = (
        h.enemy_ahead
        and h.nearest_enemy is not None
        and (h.nearest_enemy.distance_pixels <= 130.0 or h.jump_must_start_now)
    )
    reasons: list[str] = []
    if gap_distance <= 3.8:
        reasons.append("gap_critical")
    if obstacle_distance <= 2.2:
        reasons.append("obstacle_critical")
    if enemy_critical:
        reasons.append("enemy_critical")
    if e.stalled_frames >= 3:
        reasons.append("stall_critical")
    if reasons and p.grounded:
        return "right_run_jump", "+".join(reasons)
    if not p.grounded and t.gap_ahead and gap_distance <= 1.5:
        return "right_run_jump", "air_gap_momentum"
    return raw_action, None


def advance_world(player: Any, level: Any, action: str) -> None:
    """Exact physics/collision order from fixed game cli.run_play."""

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


def trace_state(frame: int, simulation_frame: int, raw: str, executed: str, reason: str | None, player: Any) -> dict[str, Any]:
    return {
        "frame": frame,
        "simulation_time_seconds": round(simulation_frame / FPS, 6),
        "video_time_seconds": round(frame / FPS, 6),
        "raw_action": raw,
        "executed_action": executed,
        "override_reason": reason,
        "x": round(player.x, 3),
        "y": round(player.y, 3),
        "vx": round(player.vx, 3),
        "vy": round(player.vy, 3),
        "progress_pixels": round(player.max_x, 3),
        "is_dead": player.is_dead,
        "has_won": player.has_won,
        "terminal_hold": False,
    }


def prepare_import_paths(jevlike_root: Path, game_root: Path) -> None:
    """Make the two pinned checkouts importable without changing their files."""

    for path in (str(game_root / "src"), str(jevlike_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import importlib

    importlib.invalidate_caches()


def precise_world_features(player: Any, level: Any) -> dict[str, Any]:
    """Return geometry features whose units and reference edge are explicit.

    JevDash telemetry reports integer scan-column distances. Those values are
    useful for a compact observation but are not the player's actual front-edge
    clearance. The X fields below are computed from the fixed tile geometry and
    are measured from the player's right edge in pixels. Floating platforms are
    intentionally not classified as ground-level pipes.
    """

    from jev_platformer.engine.constants import TILE_SIZE

    front_x = float(player.x + player.width)
    ground_row = level.height_tiles - 4
    current_col = int(front_x // TILE_SIZE)
    # Keep enough look-ahead to see an entire three-tile pit even when the
    # player is just before the telemetry scan horizon.
    max_col = min(level.width_tiles, current_col + 24)

    gap_distance_pixels: float | None = None
    gap_width_pixels = 0
    for col in range(max(0, current_col), max_col):
        is_base_ground = level.tile_types.get((col, ground_row)) == "ground"
        if is_base_ground:
            if gap_distance_pixels is not None:
                break
            continue
        if gap_distance_pixels is None:
            gap_distance_pixels = max(0.0, col * TILE_SIZE - front_x)
        gap_width_pixels += TILE_SIZE

    pipe_by_col: dict[int, list[int]] = {}
    for (col, row), tile_kind in level.tile_types.items():
        if tile_kind != "pipe" or col < current_col or col >= max_col:
            continue
        pipe_by_col.setdefault(col, []).append(row)
    obstacle_distance_pixels: float | None = None
    obstacle_height_tiles = 0
    if pipe_by_col:
        obstacle_col = min(pipe_by_col)
        obstacle_distance_pixels = max(0.0, obstacle_col * TILE_SIZE - front_x)
        obstacle_height_tiles = len(pipe_by_col[obstacle_col])

    return {
        "player_front_x_pixels": round(front_x, 3),
        "obstacle_front_distance_pixels": (
            round(obstacle_distance_pixels, 3)
            if obstacle_distance_pixels is not None else None
        ),
        "obstacle_height_tiles": obstacle_height_tiles,
        "gap_front_distance_pixels": (
            round(gap_distance_pixels, 3)
            if gap_distance_pixels is not None else None
        ),
        "gap_width_pixels": gap_width_pixels,
        "distance_reference": "player right edge to fixed tile left edge; pixels",
    }


def compact_context(obs: Any, physical: dict[str, Any] | None = None) -> str:
    """Serialize state first, retaining all decision fields within 192 bytes.

    ``T`` is the original telemetry scan in tile columns. ``X`` is the precise
    fixed-world front-edge measurement when a level is available. Keeping both
    makes the distinction auditable instead of silently treating a coarse scan
    as collision clearance.
    """

    p, h, t, e = obs.player, obs.hazard, obs.terrain, obs.episode
    enemy, enemy_distance, enemy_relative, contact = _enemy_fields(obs)
    physical = physical or {}
    radar = "".join(obs.local_grid)
    obstacle_px = physical.get("obstacle_front_distance_pixels")
    gap_px = physical.get("gap_front_distance_pixels")
    gap_width_px = physical.get("gap_width_pixels")
    pipe_height = physical.get("obstacle_height_tiles", t.obstacle_height_tiles)
    context = (
        "J1;"
        f"P{_q(p.x, 8)}{_q(p.y, 8)}{_q(p.vx, .5)}{_q(p.vy, .5)}"
        f"{int(p.grounded)}{int(p.jumping)}{min(p.airborne_frames, 99):02d}{int(p.running)};"
        f"H{enemy}{_q(enemy_distance, 8)}{_q(enemy_relative, .5)}{_q(contact, 1)}"
        f"{int(h.jump_must_start_now)}{int(h.in_danger_zone)}"
        f"{_q(h.nearest_enemy.vertical_offset_pixels if h.nearest_enemy else None, 8)};"
        f"T{int(t.obstacle_ahead)}{_q(t.obstacle_distance_tiles, 1)}{int(pipe_height):02d}"
        f"{int(t.gap_ahead)}{_q(t.gap_distance_tiles, 1)}{t.gap_width_tiles:02d}{t.clear_forward_tiles:02d};"
        f"X{_q(obstacle_px, 8)}{_q(gap_px, 8)}{_q(gap_width_px, 8)};"
        f"E{e.score:04d}{e.coins:02d}{_q(e.goal_distance_pixels, 16)}"
        f"{_q(e.progress_pixels, 16)}{min(e.stalled_frames, 99):02d}{int(e.is_dead)}{int(e.has_won)};"
        f"R{radar}"
    )
    if len(context.encode("utf-8")) > CONTEXT_TOKENS:
        raise ValueError(f"compact context exceeds TinyScorer window: {len(context.encode('utf-8'))}")
    return context


def teacher_action(obs: Any, physical: dict[str, Any] | None = None) -> str:
    """Produce a game-teacher label for the supervised dataset only."""

    p, h, t, e = obs.player, obs.hazard, obs.terrain, obs.episode
    physical = physical or {}
    enemy_distance = h.nearest_enemy.distance_pixels if h.nearest_enemy else 999.0
    enemy_vertical = abs(h.nearest_enemy.vertical_offset_pixels) if h.nearest_enemy else 999.0
    obstacle_distance = physical.get("obstacle_front_distance_pixels")
    gap_distance = physical.get("gap_front_distance_pixels")
    obstacle_hazard = obstacle_distance is not None and obstacle_distance <= 88.0
    # A run-jump needs a broad takeoff window; starting at the edge of a
    # three-tile pit is too late even though the coarse telemetry still sees it.
    gap_hazard = gap_distance is not None and gap_distance <= 400.0
    enemy_hazard = enemy_distance <= 155.0 and enemy_vertical <= 96.0
    hazard_now = (
        h.jump_must_start_now
        or obstacle_hazard
        or gap_hazard
        or enemy_hazard
        or e.stalled_frames >= 2
    )
    if not p.grounded:
        return "right_run_jump"
    return "right_run_jump" if hazard_now else "right_run"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def collect_teacher_split(
    game_root: Path,
    target_examples: int,
    split_seed: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect sanitized state/action rows from the fixed game teacher."""

    import pygame

    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    rng = random.Random(split_seed)
    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    episodes = 0
    pygame.init()
    try:
        while len(rows) < target_examples and episodes < 240:
            level = Level(LEVEL)
            start_offset = rng.choice((-16, 0, 16))
            player = Player(level.start_pos[0] + start_offset, level.start_pos[1])
            episodes += 1
            for frame in range(MAX_SIMULATION_FRAMES):
                if frame % FRAMES_PER_DECISION == 0:
                    obs = TelemetryExtractor.extract(player, level)
                    physical = precise_world_features(player, level)
                    label = teacher_action(obs, physical)
                    rows.append({
                        "context": compact_context(obs, physical),
                        "options": list(ACTION_SPACE),
                        "label": ACTION_SPACE.index(label),
                    })
                    action_counts[label] += 1
                    if len(rows) >= target_examples:
                        break
                    if rng.random() < 0.86:
                        action = label
                    else:
                        action = rng.choice(("right_run", "right_run_jump", "right_jump", "right"))
                else:
                    action = action
                advance_world(player, level, action)
                if player.has_won or player.is_dead:
                    break
    finally:
        pygame.quit()
    if len(rows) < target_examples:
        raise RuntimeError(f"teacher split {split_name} only produced {len(rows)} rows")
    return rows, {
        "name": split_name,
        "examples": len(rows),
        "episodes": episodes,
        "action_counts": dict(sorted(action_counts.items())),
        "seed": split_seed,
        "source": "fixed JevDash Level(1), deterministic physics, sanitized teacher labels",
    }


def snapshot_trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def train_game_teacher_checkpoint(
    torch: Any,
    output_dir: Path,
    jevlike_root: Path,
    game_root: Path,
    device: Any,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Train and reload a TinyScorer checkpoint on fixed-game state labels."""

    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from jevlike.data import JsonlDataset
    from jevlike.model import load_checkpoint, make_system

    data_dir = output_dir / "training-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sizes = {"train": 8192, "validation": 1024, "test": 1024}
    split_info: dict[str, Any] = {}
    data_hashes: dict[str, str] = {}
    for index, (split, size) in enumerate(sizes.items()):
        rows, info = collect_teacher_split(game_root, size, SEED + index * 1009, split)
        path = data_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        split_info[split] = info
        data_hashes[split] = sha256_file(path)

    checkpoint = output_dir / "tiny_scorer_game_teacher.pt"
    config = {
        "encoder": "tiny",
        "hf_model": "not used by tiny encoder",
        "width": 96,
        "rank": 64,
        "context_tokens": CONTEXT_TOKENS,
        "option_tokens": OPTION_TOKENS,
    }
    model, collator = make_system(config, device)
    batch_size = 256
    epochs = 12
    learning_rate = 2e-3
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
        raise RuntimeError("game-teacher training produced no checkpoint")
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
        "initialization": "TinyScorer initialized from torch seed 42; no pretrained model weights.",
        "training_data": "fixed JevDash Level(1) state/context rows with a documented game-teacher policy; no live API and no user data",
        "teacher_policy": {
            "normal": "right_run",
            "jump": "right_run_jump when precise pipe/gap front clearance or enemy/telemetry hazard enters the documented window",
            "airborne": "right_run_jump",
            "training_only": True,
        },
        "training_seed": SEED,
        "splits": sizes,
        "split_info": split_info,
        "data_sha256": data_hashes,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "width": config["width"],
        "rank": config["rank"],
        "context_tokens": CONTEXT_TOKENS,
        "option_tokens": OPTION_TOKENS,
        "training_seconds": training_seconds,
        "training_peak_vram": training_peak,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "upstream_repository": JEVLIKE_URL,
        "upstream_commit": JEVLIKE_COMMIT,
        "game_specific_training": True,
        "model_only_primary": True,
    }
    return loaded_model, loaded_collator, model_info, {
        "epochs": epoch_logs,
        "config": loaded_config,
        "data_sha256": data_hashes,
        "split_info": split_info,
    }


def decide(
    torch: Any,
    model: Any,
    collator: Any,
    choice_example_cls: Any,
    obs: Any,
    physical: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    context = compact_context(obs, physical)
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
        "physical_features": physical,
        "context": context,
        "context_utf8_bytes": len(context.encode("utf-8")),
        "options": list(ACTION_SPACE),
        "mock_used": False,
        "fallback_used": False,
    }


class ClearHUD:
    """Render model-only vs assisted control and the state/geometry semantics."""

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
        mode: str,
        raw_action: str,
        executed_action: str,
        override_reason: str | None,
        override_count: int,
    ) -> None:
        import pygame

        x = self.offset_x + 18
        y = 14
        pygame.draw.rect(self.surface, self.bg, pygame.Rect(self.offset_x, 0, self.hud_width, self.hud_height))
        pygame.draw.line(self.surface, self.border, (self.offset_x, 0), (self.offset_x, self.hud_height), 2)
        self.text("JevDash / CLEAR AUDIT", (x, y), self.primary, self.font_large)
        y += 30
        self.text("Jevlike TinyScorer", (x, y), self.accent, self.font_base)
        y += 20
        self.text(f"{mode}  |  GPU: {gpu_name}", (x, y), self.muted, self.font_xs)
        y += 17
        self.text("game-teacher checkpoint / no live API", (x, y), self.muted, self.font_xs)
        y += 23
        status_color = self.success if status == "CLEARED" else self.danger if status == "DEAD" else self.warning if status != "RUNNING" else self.accent
        self.text(f"STATUS: {status}", (x, y), status_color, self.font_base)
        y += 22
        self.text(f"RAW MODEL: {raw_action}", (x, y), self.primary, self.font_sm)
        y += 18
        action_color = self.warning if override_reason else self.primary
        self.text(f"EXECUTED: {executed_action}", (x, y), action_color, self.font_sm)
        y += 18
        reason = override_reason or "none"
        self.text(f"REFLEX OVERRIDE: {reason[:34]}", (x, y), action_color, self.font_xs)
        y += 16
        self.text(f"OVERRIDE PHYSICS FRAMES: {override_count}", (x, y), self.muted, self.font_xs)
        y += 17
        self.text(f"SIM FRAME: {simulation_frame:04d}  ({simulation_frame / FPS:5.2f}s)", (x, y), self.primary, self.font_xs)
        y += 22

        card = pygame.Rect(x, y, self.hud_width - 36, 229)
        pygame.draw.rect(self.surface, self.card, card, border_radius=8)
        pygame.draw.rect(self.surface, self.border, card, 1, border_radius=8)
        self.text("MODEL ACTION PROBABILITIES", (x + 12, y + 10), self.primary, self.font_base)
        y += 38
        probabilities = decision["probabilities"] if decision else {}
        selected = decision["action"] if decision else None
        for action in ACTION_SPACE:
            prob = float(probabilities.get(action, 0.0))
            color = self.accent if action == selected else self.muted
            self.text(f"{action:<14}", (x + 12, y), color, self.font_xs)
            pygame.draw.rect(self.surface, (30, 36, 54), pygame.Rect(x + 126, y + 2, 145, 10), border_radius=3)
            fill = int(145 * max(0.0, min(1.0, prob)))
            if fill:
                pygame.draw.rect(self.surface, color, pygame.Rect(x + 126, y + 2, fill, 10), border_radius=3)
            self.text(f"{prob * 100:5.1f}%", (x + 280, y), color, self.font_xs)
            y += 25
        y += 6
        self.text("STATE / GEOMETRY", (x, y), self.primary, self.font_base)
        y += 21
        self.text(f"x={obs.player.x:6.1f}  y={obs.player.y:5.1f}  vx={obs.player.vx:4.1f}", (x, y), self.muted, self.font_xs)
        y += 16
        self.text(f"grounded={obs.player.grounded}  airborne={obs.player.airborne_frames:02d}", (x, y), self.muted, self.font_xs)
        y += 16
        self.text(f"telemetry gap={obs.terrain.gap_distance_tiles} tiles", (x, y), self.warning, self.font_xs)
        y += 15
        self.text("X clearance = player-front to tile-left (px)", (x, y), self.warning, self.font_xs)
        y += 15
        self.text("60 FPS; inference waits pause physics", (x, y), self.warning, self.font_xs)


def run_episode(
    torch: Any,
    model: Any,
    collator: Any,
    output_path: Path,
    gpu_name: str,
    game_root: Path,
    model_info: dict[str, Any],
    mode: str,
    device: Any,
) -> dict[str, Any]:
    import pygame

    from jev_platformer.controller.actions import Action
    from jev_platformer.engine.constants import GAME_TITLE, GAME_VIEW_WIDTH, HUD_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor
    from jev_platformer.ui.renderer import GameRenderer
    from jev_platformer.ui.video_recorder import VideoRecorder
    from jevlike.data import ChoiceExample

    if tuple(Action(action).value for action in ACTION_SPACE) != ACTION_SPACE:
        raise RuntimeError("fixed game action space differs from adapter action space")
    if mode not in {"model-only", "assisted"}:
        raise ValueError(f"unknown episode mode: {mode}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    level = Level(LEVEL)
    player = Player(level.start_pos[0], level.start_pos[1])
    game_renderer = GameRenderer(screen)
    hud = ClearHUD(screen, GAME_VIEW_WIDTH, HUD_WIDTH, SCREEN_HEIGHT)
    recorder = VideoRecorder(str(output_path), SCREEN_WIDTH, SCREEN_HEIGHT, fps=FPS)
    if recorder.process is None:
        pygame.quit()
        raise RuntimeError("fixed game VideoRecorder could not start ffmpeg")

    current_raw_action = Action.NOOP.value
    current_decision: dict[str, Any] | None = None
    frame_count = 0
    decisions: list[dict[str, Any]] = []
    frame_trace: list[dict[str, Any]] = []
    raw_action_counts: Counter[str] = Counter()
    executed_action_counts: Counter[str] = Counter()
    override_reasons: Counter[str] = Counter()
    override_frames = 0
    terminal_reason: str | None = None
    wall_started = time.perf_counter()

    try:
        while frame_count < MAX_SIMULATION_FRAMES:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminal_reason = "quit"
            if terminal_reason == "quit":
                break

            obs = TelemetryExtractor.extract(player, level)
            physical = precise_world_features(player, level)
            if frame_count % FRAMES_PER_DECISION == 0:
                current_decision = decide(
                    torch, model, collator, ChoiceExample, obs, physical, device,
                )
                current_decision.update({
                    "decision_index": len(decisions),
                    "simulation_frame": frame_count,
                    "simulation_time_seconds": frame_count / FPS,
                })
                decisions.append(current_decision)
                current_raw_action = current_decision["action"]
            raw_action_counts[current_raw_action] += 1
            if mode == "assisted":
                executed_action, override_reason = safety_reflex(obs, current_raw_action)
            else:
                executed_action, override_reason = current_raw_action, None
            executed_action_counts[executed_action] += 1
            if override_reason:
                override_frames += 1
                override_reasons[override_reason] += 1

            advance_world(player, level, executed_action)
            post_obs = TelemetryExtractor.extract(player, level)
            status = "CLEARED" if player.has_won else "DEAD" if player.is_dead else "RUNNING"
            game_renderer.render(player, level)
            hud.render(
                post_obs, current_decision, frame_count, status, gpu_name,
                mode.upper(), current_raw_action, executed_action, override_reason,
                override_frames,
            )
            pygame.display.flip()
            recorder.record_frame(screen)
            trace = trace_state(
                frame_count, frame_count, current_raw_action, executed_action,
                override_reason, player,
            )
            frame_trace.append(trace)
            frame_count += 1

            if player.has_won:
                terminal_reason = "clear"
                break
            if player.is_dead:
                terminal_reason = "death"
                break

        if terminal_reason is None:
            terminal_reason = "timeout" if frame_count >= MAX_SIMULATION_FRAMES else "quit"
        terminal_frame = frame_count
        final_obs = TelemetryExtractor.extract(player, level)
        for _ in range(TERMINAL_HOLD_FRAMES):
            status = "CLEARED" if player.has_won else "DEAD" if player.is_dead else "TIMEOUT"
            pygame.event.pump()
            game_renderer.render(player, level)
            hud.render(
                final_obs, current_decision, terminal_frame, status, gpu_name,
                mode.upper(), current_raw_action, current_raw_action,
                None, override_frames,
            )
            pygame.display.flip()
            recorder.record_frame(screen)
            hold_trace = trace_state(
                frame_count, terminal_frame, current_raw_action, current_raw_action,
                None, player,
            )
            hold_trace["terminal_hold"] = True
            frame_trace.append(hold_trace)
            frame_count += 1
    finally:
        recorder.close()
        pygame.quit()

    sync_cuda(torch, device)
    wall_seconds = time.perf_counter() - wall_started
    inference_peak = None
    if device.type == "cuda":
        inference_peak = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    outcome = "cleared" if player.has_won else "dead" if player.is_dead else "timeout"
    return {
        "status": "success",
        "generated_at_utc": utc_now(),
        "mode": mode,
        "outcome": outcome,
        "game": {
            "repository": GAME_URL,
            "commit": GAME_COMMIT,
            "level": LEVEL,
            "seed": SEED,
            "physics_fps": FPS,
            "frames_per_decision": FRAMES_PER_DECISION,
            "max_simulation_frames": MAX_SIMULATION_FRAMES,
            "terminal_hold_frames": TERMINAL_HOLD_FRAMES,
            "action_space": list(ACTION_SPACE),
            "engine_source": "fixed commit cli.run_play physics/collision/render order; game checkout was not modified",
        },
        "model": model_info,
        "control": {
            "primary_model_only": mode == "model-only",
            "safety_reflex_enabled": mode == "assisted",
            "safety_reflex_evaluation": "every physics frame" if mode == "assisted" else "disabled",
            "raw_action_counts": dict(sorted(raw_action_counts.items())),
            "executed_action_counts": dict(sorted(executed_action_counts.items())),
            "override_frames": override_frames,
            "override_rate_over_simulation_frames": round(override_frames / max(1, terminal_frame), 6),
            "override_reasons": dict(sorted(override_reasons.items())),
        },
        "runtime": {
            "backend": "local Jevlike checkpoint on Colab GPU",
            "model_name": MODEL_NAME,
            "gpu_name": gpu_name,
            "device": str(device),
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
            "time_basis": "MP4 timestamps use every recorded frame at 60 FPS; physics simulation time is held at the terminal frame during the static terminal hold, and synchronous model waits occur before each physics frame.",
            "inference_peak_vram": inference_peak,
        },
        "artifacts": {
            "episode_json": output_path.with_suffix(".json").name,
            "video_mp4": output_path.name,
        },
    }


def run_control_fixture(game_root: Path, policy: str, reflex_cadence: str | None = None) -> dict[str, Any]:
    """Run a local no-model control fixture for the physics audit only."""

    import pygame

    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    pygame.init()
    level = Level(LEVEL)
    player = Player(level.start_pos[0], level.start_pos[1])
    current_raw = "noop"
    current_executed = "noop"
    override_count = 0
    override_reasons: Counter[str] = Counter()
    last_snapshot: dict[str, Any] = {}
    action_transitions: list[dict[str, Any]] = []
    try:
        for frame in range(MAX_SIMULATION_FRAMES):
            obs = TelemetryExtractor.extract(player, level)
            if frame % FRAMES_PER_DECISION == 0:
                if policy.startswith("constant:"):
                    current_raw = policy.split(":", 1)[1]
                elif policy == "teacher":
                    current_raw = teacher_action(obs, precise_world_features(player, level))
                else:
                    raise ValueError(f"unknown fixture policy: {policy}")
                action_transitions.append({
                    "frame": frame,
                    "raw_action": current_raw,
                    "precise": precise_world_features(player, level),
                    "x": round(player.x, 3),
                    "grounded": player.grounded,
                    "stalled_frames": player.stalled_frames,
                })
                if reflex_cadence == "every-decision":
                    current_executed, reason = safety_reflex(obs, current_raw)
                    if reason:
                        override_count += 1
                        override_reasons[reason] += 1
                else:
                    current_executed = current_raw
            if reflex_cadence == "every-frame":
                current_executed, reason = safety_reflex(obs, current_raw)
                if reason:
                    override_count += 1
                    override_reasons[reason] += 1
            advance_world(player, level, current_executed)
            last_snapshot = {
                "frame": frame + 1,
                "x": round(player.x, 3),
                "y": round(player.y, 3),
                "vx": round(player.vx, 3),
                "vy": round(player.vy, 3),
                "grounded": player.grounded,
                "raw_action": current_raw,
                "executed_action": current_executed,
                "is_dead": player.is_dead,
                "has_won": player.has_won,
                "enemy": (
                    TelemetryExtractor.extract(player, level).hazard.nearest_enemy.model_dump(mode="json")
                    if TelemetryExtractor.extract(player, level).hazard.nearest_enemy else None
                ),
                "physical": precise_world_features(player, level),
            }
            if player.has_won or player.is_dead:
                terminal_frame = frame + 1
                break
        else:
            terminal_frame = MAX_SIMULATION_FRAMES
    finally:
        pygame.quit()
    return {
        "policy": policy,
        "reflex_cadence": reflex_cadence,
        "terminal_frame": terminal_frame,
        "terminal_reason": "clear" if player.has_won else "death" if player.is_dead else "timeout",
        "final_progress_pixels": round(player.max_x, 3),
        "final_is_dead": player.is_dead,
        "final_has_won": player.has_won,
        "override_frames": override_count,
        "override_reasons": dict(sorted(override_reasons.items())),
        "terminal_snapshot": last_snapshot,
        "action_transitions": action_transitions[-16:],
        "gpu_claim": False,
        "model_inference": False,
    }


def audit_option_order(torch: Any, obs: Any, physical: dict[str, Any]) -> dict[str, Any]:
    """Verify that changing candidate order changes columns, not identities."""

    from jevlike.data import ChoiceExample
    from jevlike.model import make_system

    torch.manual_seed(SEED)
    config = {
        "encoder": "tiny",
        "hf_model": "not used by tiny encoder",
        "width": 32,
        "rank": 16,
        "context_tokens": CONTEXT_TOKENS,
        "option_tokens": OPTION_TOKENS,
    }
    model, collator = make_system(config, torch.device("cpu"))
    model.eval()
    context = compact_context(obs, physical)
    reversed_options = tuple(reversed(ACTION_SPACE))
    with torch.no_grad():
        first = model(collator([ChoiceExample(context, ACTION_SPACE, 0)]))[0, :len(ACTION_SPACE)]
        second = model(collator([ChoiceExample(context, reversed_options, 0)]))[0, :len(ACTION_SPACE)]
    first_by_action = dict(zip(ACTION_SPACE, [float(value) for value in first]))
    second_by_action = dict(zip(reversed_options, [float(value) for value in second]))
    differences = {
        action: abs(first_by_action[action] - second_by_action[action])
        for action in ACTION_SPACE
    }
    return {
        "options": list(ACTION_SPACE),
        "reversed_options": list(reversed_options),
        "identity_logit_max_abs_difference": max(differences.values()),
        "identity_logit_differences": differences,
        "interpretation": "option order is a column permutation; the upstream TinyScorer has no option-position embedding",
    }


def run_audit(game_root: Path, jevlike_root: Path, output_path: Path) -> dict[str, Any]:
    """Write local evidence for truncation, candidate order, and physics controls."""

    import pygame
    import torch

    prepare_import_paths(jevlike_root, game_root)
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    pygame.init()
    observations: list[tuple[Any, dict[str, Any]]] = []
    level = Level(LEVEL)
    player = Player(level.start_pos[0], level.start_pos[1])
    try:
        for frame in range(0, 256):
            if frame % FRAMES_PER_DECISION == 0:
                obs = TelemetryExtractor.extract(player, level)
                physical = precise_world_features(player, level)
                observations.append((obs, physical))
            advance_world(player, level, "right_run")
            if player.has_won or player.is_dead:
                break
    finally:
        pygame.quit()

    legacy_bytes = [len(legacy_context(obs).encode("utf-8")) for obs, _ in observations]
    compact_bytes = [len(compact_context(obs, physical).encode("utf-8")) for obs, physical in observations]
    legacy_prefixes = {
        hashlib.sha256(legacy_context(obs).encode("utf-8")[:CONTEXT_TOKENS]).hexdigest()
        for obs, _ in observations
    }
    compact_prefixes = {
        hashlib.sha256(compact_context(obs, physical).encode("utf-8")[:CONTEXT_TOKENS]).hexdigest()
        for obs, physical in observations
    }
    first_obs, first_physical = observations[0]
    controls = [
        run_control_fixture(game_root, "constant:right_run_jump"),
        run_control_fixture(game_root, "constant:right_jump"),
        run_control_fixture(game_root, "constant:right_run"),
        run_control_fixture(game_root, "teacher"),
        run_control_fixture(game_root, "constant:right_run", "every-frame"),
        run_control_fixture(game_root, "constant:right_run", "every-decision"),
    ]
    audit = {
        "status": "success",
        "generated_at_utc": utc_now(),
        "gpu_claim": False,
        "model_inference_claim": False,
        "source": {
            "game_repository": GAME_URL,
            "game_commit": GAME_COMMIT,
            "jevlike_repository": JEVLIKE_URL,
            "jevlike_commit": JEVLIKE_COMMIT,
            "level": LEVEL,
            "seed": SEED,
        },
        "context_window_audit": {
            "observations": len(observations),
            "legacy_utf8_byte_lengths": {"min": min(legacy_bytes), "max": max(legacy_bytes)},
            "compact_utf8_byte_lengths": {"min": min(compact_bytes), "max": max(compact_bytes)},
            "legacy_truncated_prefix_sha256_count": len(legacy_prefixes),
            "compact_truncated_prefix_sha256_count": len(compact_prefixes),
            "legacy_context_first_192_bytes_same": len(legacy_prefixes) == 1,
            "compact_context_within_window": max(compact_bytes) <= CONTEXT_TOKENS,
            "changing_state_signatures": [
                {
                    "frame": index * FRAMES_PER_DECISION,
                    "x": obs.player.x,
                    "y": obs.player.y,
                    "vx": obs.player.vx,
                    "gap_tiles": obs.terrain.gap_distance_tiles,
                    "obstacle_tiles": obs.terrain.obstacle_distance_tiles,
                    "exact_gap_front_px": physical["gap_front_distance_pixels"],
                    "exact_obstacle_front_px": physical["obstacle_front_distance_pixels"],
                    "stalled_frames": obs.episode.stalled_frames,
                }
                for index, (obs, physical) in enumerate(observations[:12])
            ],
            "compact_context_prefix_sha256": sorted(compact_prefixes),
        },
        "distance_semantics": {
            "telemetry_obstacle_and_gap": "coarse scan-column distance from player tile; not front-edge collision clearance",
            "physical_X_fields": "fixed level tile left edge minus player right edge, measured in pixels",
            "floating_platforms": "not classified as base ground or pipe obstacles by precise_world_features",
            "policy_fields": ["grounded", "airborne_frames", "vertical enemy offset", "stalled_frames", "precise X clearance"],
        },
        "candidate_order_audit": audit_option_order(torch, first_obs, first_physical),
        "physics_control_fixtures": controls,
        "teacher_policy": {
            "used_for": "training labels only",
            "not_used_for": "model-only episode decisions",
            "safety_reflex_cadence": "assisted episode only, every physics frame",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def run_colab_experiment() -> None:
    import numpy
    import torch

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a CPU or mock substitute")
    device = torch.device("cuda")
    set_seeds(torch, numpy)
    gpu_name = torch.cuda.get_device_name(device)
    output_dir = Path("/content/jevlike-jevdash-clear-output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    game_root = Path("/content/jevdash-fixed").resolve()
    jevlike_root = Path("/content/jevlike-fixed").resolve()
    game_root, game_checkout_seconds = fetch_fixed_repo(GAME_URL, GAME_COMMIT, game_root)
    jevlike_root, jevlike_checkout_seconds = fetch_fixed_repo(JEVLIKE_URL, JEVLIKE_COMMIT, jevlike_root)
    install_seconds = install_colab_dependencies(jevlike_root, game_root)
    prepare_import_paths(jevlike_root, game_root)
    model, collator, model_info, training_info = train_game_teacher_checkpoint(
        torch, output_dir, jevlike_root, game_root, device,
    )

    episodes: dict[str, dict[str, Any]] = {}
    for mode in ("model-only", "assisted"):
        video_path = output_dir / f"jevdash-jevlike-clear-{mode}.mp4"
        episode = run_episode(
            torch, model, collator, video_path, gpu_name, game_root,
            model_info, mode, device,
        )
        episode["provenance"] = {
            "game_checkout_seconds": game_checkout_seconds,
            "jevlike_checkout_seconds": jevlike_checkout_seconds,
            "uv_install_seconds": install_seconds,
            "training_logs": training_info["epochs"],
            "training_config": training_info["config"],
            "training_data_sha256": training_info["data_sha256"],
            "adapter": "experiments/jevlike/adapter/jevdash_clear_colab_runner.py",
            "execution_contract": "real CUDA inference; no mock, fallback, or live API; model-only and assisted are separate episodes",
        }
        episode_path = output_dir / f"jevdash-jevlike-clear-{mode}.json"
        episode_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        episodes[mode] = episode

    model_only = episodes["model-only"]
    assisted = episodes["assisted"]
    comparison = {
        "status": "success",
        "generated_at_utc": utc_now(),
        "game": model_only["game"],
        "model": model_info,
        "episodes": {
            mode: {
                "outcome": episode["outcome"],
                "final_has_won": episode["trajectory"]["final_has_won"],
                "final_is_dead": episode["trajectory"]["final_is_dead"],
                "terminal_reason": episode["trajectory"]["terminal_reason"],
                "terminal_simulation_frame": episode["trajectory"]["terminal_simulation_frame"],
                "final_progress_pixels": episode["trajectory"]["final_progress_pixels"],
                "decision_count": episode["trajectory"]["decision_count"],
                "override_frames": episode["control"]["override_frames"],
                "override_rate_over_simulation_frames": episode["control"]["override_rate_over_simulation_frames"],
                "episode_json": episode["artifacts"]["episode_json"],
                "video_mp4": episode["artifacts"]["video_mp4"],
            }
            for mode, episode in episodes.items()
        },
        "interpretation": (
            "model-only is the primary model success criterion; assisted success is reported separately and is not model-only success"
        ),
        "primary_model_only_success": bool(model_only["trajectory"]["final_has_won"]),
        "assisted_success": bool(assisted["trajectory"]["final_has_won"]),
    }
    comparison_path = output_dir / "jevdash-jevlike-clear-comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESULT_SUMMARY=" + json.dumps({
        "status": "success",
        "gpu": gpu_name,
        "model_only_outcome": model_only["outcome"],
        "model_only_has_won": model_only["trajectory"]["final_has_won"],
        "assisted_outcome": assisted["outcome"],
        "assisted_has_won": assisted["trajectory"]["final_has_won"],
        "model_only_progress_pixels": model_only["trajectory"]["final_progress_pixels"],
        "assisted_progress_pixels": assisted["trajectory"]["final_progress_pixels"],
        "comparison_json": str(comparison_path),
    }, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--game-root", type=Path)
    parser.add_argument("--jevlike-root", type=Path)
    parser.add_argument("--output", type=Path)
    # `colab exec --file` runs the file through a Jupyter kernel and appends
    # its connection-file argument. It is infrastructure metadata, not an
    # experiment option, so accept and ignore it explicitly.
    parser.add_argument("-f", dest="jupyter_connection_file")
    args = parser.parse_args()
    if args.audit_only:
        if not args.game_root or not args.jevlike_root or not args.output:
            parser.error("--audit-only requires --game-root, --jevlike-root, and --output")
        audit = run_audit(
            args.game_root.expanduser().resolve(),
            args.jevlike_root.expanduser().resolve(),
            args.output.expanduser().resolve(),
        )
        print("AUDIT_SUMMARY=" + json.dumps({
            "status": audit["status"],
            "observations": audit["context_window_audit"]["observations"],
            "legacy_prefix_count": audit["context_window_audit"]["legacy_truncated_prefix_sha256_count"],
            "compact_prefix_count": audit["context_window_audit"]["compact_truncated_prefix_sha256_count"],
            "compact_max_bytes": audit["context_window_audit"]["compact_utf8_byte_lengths"]["max"],
            "controls": audit["physics_control_fixtures"],
            "output": str(args.output.expanduser().resolve()),
        }, ensure_ascii=False, sort_keys=True))
        return
    run_colab_experiment()


if __name__ == "__main__":
    main()
