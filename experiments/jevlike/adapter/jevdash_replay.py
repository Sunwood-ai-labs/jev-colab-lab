"""Correct the presentation of a recorded Jevlike JevDash capture.

This script never calls Jevlike and never changes the recorded decisions. It
replays the fixed game's physics with the action selected in a source episode,
checks every recorded simulation state, and renders a corrected MP4. This is
presentation correction, not a second model run.
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
from pathlib import Path
from typing import Any


ADAPTER_DIR = Path(__file__).resolve().parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from jevdash_colab_runner import (  # noqa: E402
    ACTION_SPACE,
    FPS,
    FRAMES_PER_DECISION,
    GAME_COMMIT,
    JevlikeHUD,
    SEED,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def assert_close(actual: float, expected: float, field: str, frame: int) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=0.01):
        raise RuntimeError(
            f"state mismatch at frame {frame}: {field} actual={actual} expected={expected}"
        )


def compare_state(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    frame = int(expected["frame"])
    if int(actual["frame"]) != frame:
        raise RuntimeError(f"state mismatch at frame {frame}: frame number")
    if actual["action"] != expected["action"]:
        raise RuntimeError(
            f"state mismatch at frame {frame}: action actual={actual['action']} expected={expected['action']}"
        )
    for field in ("x", "y", "vx", "vy", "progress_pixels"):
        assert_close(actual[field], expected[field], field, frame)
    for field in ("is_dead", "has_won", "terminal_hold"):
        if bool(actual[field]) != bool(expected[field]):
            raise RuntimeError(
                f"state mismatch at frame {frame}: {field} actual={actual[field]} expected={expected[field]}"
            )


def advance_fixed_game(player: Any, level: Any, action: str) -> None:
    """Match cli.run_play's physics and collision order at the pinned commit."""

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


def trace_row(
    frame: int,
    simulation_frame: int,
    action: str,
    player: Any,
    terminal_hold: bool,
) -> dict[str, Any]:
    return {
        "frame": frame,
        "simulation_time_seconds": round(simulation_frame / FPS, 6),
        "video_time_seconds": round(frame / FPS, 6),
        "action": action,
        "x": round(player.x, 3),
        "y": round(player.y, 3),
        "vx": round(player.vx, 3),
        "vy": round(player.vy, 3),
        "progress_pixels": round(player.max_x, 3),
        "is_dead": player.is_dead,
        "has_won": player.has_won,
        "terminal_hold": terminal_hold,
    }


def validate_source(episode: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    game = episode["game"]
    if game["commit"] != GAME_COMMIT:
        raise RuntimeError(f"game commit mismatch: {game['commit']}")
    if tuple(game["action_space"]) != ACTION_SPACE:
        raise RuntimeError("source action space is not the fixed seven-action space")
    if int(game["physics_fps"]) != FPS:
        raise RuntimeError("source FPS is not 60")
    if int(game["frames_per_decision"]) != FRAMES_PER_DECISION:
        raise RuntimeError("source decision cadence is not eight frames")

    trajectory = episode["trajectory"]
    frames = trajectory["frames"]
    terminal_frame = int(trajectory["terminal_simulation_frame"])
    if terminal_frame <= 0 or terminal_frame > len(frames):
        raise RuntimeError("invalid terminal frame in source trajectory")
    hold_frames = sum(1 for row in frames if row["terminal_hold"])
    expected_hold = int(game["terminal_hold_frames"])
    if hold_frames != expected_hold or len(frames) != terminal_frame + expected_hold:
        raise RuntimeError("source trajectory does not contain the expected terminal hold")

    decisions = trajectory["decisions"]
    decision_frames = {int(item["simulation_frame"]): item for item in decisions}
    if not decision_frames or min(decision_frames) != 0:
        raise RuntimeError("source trajectory has no decision at simulation frame zero")
    for frame in decision_frames:
        if frame >= terminal_frame or frame % FRAMES_PER_DECISION:
            raise RuntimeError(f"invalid source decision frame: {frame}")
    return frames, terminal_frame, hold_frames


def replay(
    source_episode_path: Path,
    source_video_path: Path | None,
    game_root: Path,
    output_video_path: Path,
    output_episode_path: Path,
) -> dict[str, Any]:
    source_episode = json.loads(source_episode_path.read_text(encoding="utf-8"))
    source_frames, terminal_frame, hold_frames = validate_source(source_episode)
    source_trajectory_sha256 = sha256_json(source_episode["trajectory"])

    actual_commit = subprocess.check_output(
        ["git", "-C", str(game_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != GAME_COMMIT:
        raise RuntimeError(f"game checkout mismatch: {actual_commit}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required by the fixed game's VideoRecorder")

    sys.path.insert(0, str(game_root / "src"))
    import pygame

    from jev_platformer.controller.actions import Action
    from jev_platformer.engine.constants import (
        GAME_TITLE,
        GAME_VIEW_WIDTH,
        HUD_WIDTH,
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
    )
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor
    from jev_platformer.ui.renderer import GameRenderer
    from jev_platformer.ui.video_recorder import VideoRecorder

    random.seed(SEED)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_episode_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    level = Level(int(source_episode["game"]["level"]))
    player = Player(level.start_pos[0], level.start_pos[1])
    game_renderer = GameRenderer(screen)
    source_gpu = source_episode.get("runtime", {}).get("gpu_name", "Colab source GPU")
    hud = JevlikeHUD(screen, GAME_VIEW_WIDTH, HUD_WIDTH, SCREEN_HEIGHT)
    recorder = VideoRecorder(str(output_video_path), SCREEN_WIDTH, SCREEN_HEIGHT, fps=FPS)
    if recorder.process is None:
        pygame.quit()
        raise RuntimeError("fixed game VideoRecorder could not start ffmpeg")

    decisions = {
        int(item["simulation_frame"]): item
        for item in source_episode["trajectory"]["decisions"]
    }
    current_decision: dict[str, Any] | None = None
    current_action = Action.NOOP.value
    corrected_frames: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    try:
        for frame in range(terminal_frame):
            pygame.event.pump()
            obs = TelemetryExtractor.extract(player, level)
            if frame in decisions:
                current_decision = copy.deepcopy(decisions[frame])
                current_action = current_decision["action"]
            if current_action not in ACTION_SPACE:
                raise RuntimeError(f"source selected an action outside the fixed space: {current_action}")
            advance_fixed_game(player, level, current_action)
            status = "CLEARED" if player.has_won else "DEAD" if player.is_dead else "RUNNING"
            game_renderer.render(player, level)
            hud.render(obs, current_decision, frame, status, f"{source_gpu} source")
            pygame.display.flip()
            recorder.record_frame(screen)
            actual = trace_row(frame, frame, current_action, player, False)
            compare_state(actual, source_frames[frame])
            corrected_frames.append(actual)

        final_obs = TelemetryExtractor.extract(player, level)
        terminal_status = "CLEARED" if player.has_won else "DEAD" if player.is_dead else "TIMEOUT"
        for frame in range(terminal_frame, len(source_frames)):
            pygame.event.pump()
            game_renderer.render(player, level)
            hud.render(final_obs, current_decision, terminal_frame, terminal_status, f"{source_gpu} source")
            pygame.display.flip()
            recorder.record_frame(screen)
            actual = trace_row(frame, terminal_frame, current_action, player, True)
            compare_state(actual, source_frames[frame])
            corrected_frames.append(actual)
    finally:
        recorder.close()
        pygame.quit()

    if len(corrected_frames) != len(source_frames):
        raise RuntimeError("corrected frame count differs from source trajectory")
    corrected = copy.deepcopy(source_episode)
    corrected["generated_at_utc"] = utc_now()
    corrected["trajectory"]["frames"] = corrected_frames
    corrected["provenance"]["presentation_correction"] = {
        "mode": "recorded_trajectory_replay",
        "source_episode": source_episode_path.name,
        "additional_model_inference": False,
        "source_trajectory_sha256": source_trajectory_sha256,
        "state_match": {
            "matched": True,
            "simulation_frames_checked": terminal_frame,
            "terminal_hold_frames_checked": hold_frames,
            "video_frames_checked": len(corrected_frames),
        },
        "changes": [
            "terminal-hold HUD simulation time fixed at the terminal simulation frame",
            "terminal-hold frame trace separates simulation_time_seconds from video_time_seconds",
            "HUD footer wrapped to two short lines",
        ],
        "replay_wall_clock_seconds": time.perf_counter() - wall_started,
    }
    corrected["video"]["path"] = output_video_path.name
    corrected["video"]["presentation_mode"] = "recorded_trajectory_replay"
    corrected["video"]["time_basis"] = (
        "MP4 timestamps use every recorded frame at 60 FPS; physics simulation time is held at the terminal frame during the static terminal hold, and synchronous model waits are excluded."
    )
    if source_video_path and source_video_path.exists():
        corrected["video"]["source_colab_video_sha256"] = sha256_file(source_video_path)
    corrected["artifacts"]["episode_json"] = output_episode_path.name
    corrected["artifacts"]["video_mp4"] = output_video_path.name
    corrected["artifacts"]["source_colab_episode_json"] = source_episode_path.name
    corrected["runtime"]["presentation_replay_gpu_inference"] = False
    output_episode_path.write_text(
        json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "success",
        "video": str(output_video_path),
        "episode": str(output_episode_path),
        "source_trajectory_sha256": source_trajectory_sha256,
        "state_match": True,
        "simulation_frames_checked": terminal_frame,
        "terminal_hold_frames_checked": hold_frames,
        "video_frames": len(corrected_frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-episode", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.source_episode.resolve(),
        args.source_video.resolve() if args.source_video else None,
        args.game_root.resolve(),
        args.output_video.resolve(),
        args.output_episode.resolve(),
    )
    print("REPLAY_SUMMARY=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
