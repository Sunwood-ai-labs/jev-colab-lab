"""Run fixed-commit JevDash with the real Kev adapter on a Colab T4.

The game repository is cloned outside this repository and checked out at the
requested commit.  The loop is entered through the fixed commit's
``jev_platformer.cli.run_play``.  Only its agent, renderer overlay, and
recorder symbols are replaced: physics, collisions, camera, original HUD,
and frame cadence remain the fixed game's implementation.

The replacement agent is synchronous.  It sends only the canonical game
observation and the seven action choices to Kev; no mock/heuristic fallback,
danger score, urgency score, or sub-frame rescue is used.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


GAME_REPO = "https://github.com/Sunwood-ai-labs/jevdash.git"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
GAME_REF = "main at verification time"
GAME_DIR = Path("/content/jevdash-fixed")

SEED = 42
LEVEL = 1
FPS = 60
FRAMES_PER_DECISION = 8
MAX_SIMULATION_FRAMES = 1800
TERMINAL_HOLD_FRAMES = 120
UPSTREAM_TERMINAL_HOLD_FRAMES = {"died": 30, "cleared": 45, "timeout": 0}

OUTPUT_ROOT = Path(os.environ.get("KEV_JEVDASH_OUTPUT", "/content/kev-jevdash"))
VIDEO_PATH = OUTPUT_ROOT / "kev-jevdash-level1.mp4"
EPISODE_PATH = OUTPUT_ROOT / "kev-jevdash-level1.json"
FRAME_DIR = OUTPUT_ROOT / "frames"

ACTION_ORDER = ["noop", "right", "right_run", "right_jump", "right_run_jump", "jump", "left"]
ACTION_DESCRIPTIONS = {
    "noop": (
        "Apply no directional or jump input. On the ground this decelerates horizontal speed; "
        "in the air it preserves horizontal inertia."
    ),
    "right": (
        "Move right: walking speed while grounded, but running horizontal speed while airborne; "
        "do not start a jump."
    ),
    "right_run": "Move right at running speed across clear flat ground; do not start a jump.",
    "right_jump": (
        "Move right and start a normal jump with upward velocity -13.5; an airborne jump input "
        "cannot create a second jump."
    ),
    "right_run_jump": (
        "Move right at running speed and start the stronger running jump with upward velocity -15.5 "
        "over an immediate pipe, gap, or enemy; an airborne jump input cannot create a second jump."
    ),
    "jump": (
        "Start a vertical normal jump with upward velocity -13.5 without intentional horizontal direction; "
        "an airborne jump input cannot create a second jump."
    ),
    "left": "Move left away from the goal; use only when backtracking is explicitly necessary.",
}

QUESTION_VARIANTS = {
    "current": (
        "Choose exactly one action macro for the next fixed control interval. "
        "Use only the supplied game observation; do not output prose."
    ),
    "goal_oriented": (
        "Choose exactly one action macro for the next fixed control interval. "
        "The objective is to reach the goal flag to the right without dying. "
        "Prefer a right-moving action on clear ground; use a running jump for a gap, pipe, or enemy. "
        "Use left only when the observed state explicitly requires backtracking. Return no prose."
    ),
    "rules_v2": (
        "Choose exactly one action macro for the next 8 simulation frames to reach the goal flag to the right. "
        "On clear ground, keep advancing with right_run. If a pipe, gap, or enemy is ahead and the player is "
        "grounded, choose right_run_jump early enough to clear it. While airborne, keep moving right and do not "
        "expect jump to create a second jump; grounded and airborne_frames describe jump phase. "
        "Telemetry obstacle_distance_tiles and gap_distance_tiles are coarse column distances from the player's "
        "tile, not exact front-edge collision clearance. Use enemy distance and vertical position together. "
        "Treat stalled_frames >= 3 as a failure to make forward progress and recover by moving right. "
        "Use left only for explicit backtracking in the supplied state. Return no prose."
    ),
    "short": "Select one action for the next interval from the listed candidates; return no prose.",
}

REPRESENTATION = os.environ.get("KEV_REPRESENTATION", "compact_json")
QUESTION_VARIANT = os.environ.get("KEV_QUESTION_VARIANT", "rules_v2")
ORDER_NAME = os.environ.get("KEV_ORDER_VARIANT", "canonical")
ASSIST_MODE = os.environ.get("KEV_ASSIST", "none")

MODEL_NAME = "Kev-0.5B"
MODEL_TRAINING_SOURCE = (
    "Official jaredpalmer/kev-0.5b v0.1 research adapter; six public sources "
    "(Banking77, AG News, MNLI, BoolQ, SST-5, Yelp), not trained on JevDash."
)

CURRENT_SIM_FRAME = -1
LAST_GAME_STATE: dict[str, Any] = {}
LAST_DASHBOARD: Any = None
LAST_RECORDER: Any = None
MODEL_GPU_NAME = "unknown"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_command(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything() -> None:
    random.seed(SEED)
    try:
        import numpy as np

        np.random.seed(SEED)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass


ORDER_VARIANTS = {
    "canonical": tuple(ACTION_ORDER),
    "right_priority": ("right_run_jump", "right_jump", "right_run", "right", "noop", "jump", "left"),
    "reverse": tuple(reversed(ACTION_ORDER)),
}


def compact_state(full: dict[str, Any]) -> dict[str, Any]:
    nearest = full["hazard"].get("nearest_enemy")
    return {
        "mission": "Reach the goal flag to the right without dying.",
        "player": {
            key: full["player"][key]
            for key in ("x", "y", "vx", "vy", "grounded", "jumping", "airborne_frames", "running")
        },
        "progress": {
            key: full["episode"][key]
            for key in ("progress_pixels", "goal_distance_pixels", "stalled_frames", "score", "coins", "lives")
        },
        "hazard": {
            "enemy_ahead": full["hazard"]["enemy_ahead"],
            "nearest_enemy": nearest,
            "jump_must_start_now": full["hazard"]["jump_must_start_now"],
            "in_danger_zone": full["hazard"]["in_danger_zone"],
        },
        "terrain": full["terrain"],
        "local_radar": full["local_grid"],
    }


def state_for_representation(full: dict[str, Any]) -> dict[str, Any]:
    if REPRESENTATION == "full_json":
        return full
    if REPRESENTATION == "compact_json":
        return compact_state(full)
    if REPRESENTATION == "compact_policy_json":
        state = compact_state(full)
        state["control_hint"] = (
            "On clear ground advance right. If a gap, pipe, or enemy is ahead, use right_run_jump. "
            "Do not move left unless the state requires backtracking."
        )
        return state
    raise ValueError(f"unsupported KEV_REPRESENTATION: {REPRESENTATION}")


def reflex_override(obs: Any, raw_action: str) -> tuple[str, list[str]]:
    """The fixed game's published reflex, used only in explicitly assisted runs."""

    terrain, hazard, player = obs.terrain, obs.hazard, obs.player
    reasons: list[str] = []
    if terrain.gap_ahead and (terrain.gap_distance_tiles or 99) <= 3.8:
        reasons.append("gap_critical")
    if terrain.obstacle_ahead and (terrain.obstacle_distance_tiles or 99) <= 2.2:
        reasons.append("obstacle_critical")
    if hazard.enemy_ahead and hazard.nearest_enemy is not None and (
        hazard.nearest_enemy.distance_pixels <= 130 or hazard.jump_must_start_now
    ):
        reasons.append("enemy_critical")
    if obs.episode.stalled_frames >= 3:
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


def clone_game() -> dict[str, str]:
    if not (GAME_DIR / ".git").exists():
        GAME_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_command(["git", "clone", "--depth", "1", GAME_REPO, str(GAME_DIR)])
    observed = run_command(["git", "rev-parse", "HEAD"], cwd=GAME_DIR).stdout.strip()
    if observed != GAME_COMMIT:
        run_command(["git", "fetch", "--depth", "1", "origin", GAME_COMMIT], cwd=GAME_DIR)
        run_command(["git", "switch", "--detach", GAME_COMMIT], cwd=GAME_DIR)
        observed = run_command(["git", "rev-parse", "HEAD"], cwd=GAME_DIR).stdout.strip()
    if observed != GAME_COMMIT:
        raise RuntimeError(f"game commit mismatch: expected {GAME_COMMIT}, got {observed}")
    return {"repository": GAME_REPO, "ref": GAME_REF, "commit": observed}


class UnmeasuredMetric:
    """Make the fixed HUD say N/A without inventing a Kev danger value."""

    def __le__(self, _other: object) -> bool:
        return False

    def __format__(self, _spec: str) -> str:
        return "N/A"

    def __str__(self) -> str:
        return "N/A"


class KevDecisionAdapter:
    def __init__(self, upstream_dir: Path, adapter_dir: Path, base_dir: Path, model_sources: dict[str, Any]):
        import torch

        if REPRESENTATION not in {"full_json", "compact_json", "compact_policy_json"}:
            raise ValueError(f"unsupported KEV_REPRESENTATION: {REPRESENTATION}")
        if QUESTION_VARIANT not in QUESTION_VARIANTS:
            raise ValueError(f"unsupported KEV_QUESTION_VARIANT: {QUESTION_VARIANT}")
        if ORDER_NAME not in ORDER_VARIANTS:
            raise ValueError(f"unsupported KEV_ORDER_VARIANT: {ORDER_NAME}")
        if ASSIST_MODE not in {"none", "reflex"}:
            raise ValueError(f"unsupported KEV_ASSIST: {ASSIST_MODE}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this episode cannot be reported as a T4 run")
        self.device = "cuda"
        self.gpu_name = torch.cuda.get_device_name(0)
        if "T4" not in self.gpu_name.upper():
            raise RuntimeError(f"requested T4 but Colab reported {self.gpu_name!r}")
        self.model_sources = model_sources
        sys.path.insert(0, str(upstream_dir))
        from kev.api import SystemOneRequest, to_record
        from kev.model import DecisionModel, encode, load_tokenizer
        from peft import PeftModel

        self._SystemOneRequest = SystemOneRequest
        self._to_record = to_record
        self._encode = encode
        self._tokenizer = load_tokenizer(str(base_dir))
        self._model = DecisionModel(str(base_dir), self._tokenizer, self.device, lora=None)
        self._model.lm = PeftModel.from_pretrained(self._model.lm, str(adapter_dir)).to(self.device)
        head_meta = torch.load(adapter_dir / "head.pt", map_location="cpu", weights_only=False)
        self._model.head.load_state_dict(head_meta["head"])
        self._model.eval()
        self.order = ORDER_VARIANTS[ORDER_NAME]
        self.assist_mode = ASSIST_MODE
        self.decisions: list[dict[str, Any]] = []

    def decide(self, obs: Any, simulation_frame: int) -> Any:
        import torch

        started = time.perf_counter()
        full_observation = obs.model_dump(mode="json")
        state = state_for_representation(full_observation)
        request = {
            "state": state,
            "model": "kev-0.5b",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": QUESTION_VARIANTS[QUESTION_VARIANT],
                    "criteria": {key: ACTION_DESCRIPTIONS[key] for key in self.order},
                }
            },
        }
        validated = self._SystemOneRequest.model_validate(request)
        record, meta = self._to_record(validated)
        if meta[0].get("keys") != list(self.order):
            raise RuntimeError("Kev candidate mapping does not match the requested order")
        encoded = self._encode(self._tokenizer, record)
        with torch.inference_mode():
            probability_tensors = self._model.probs(encoded)
        probabilities = probability_tensors[0].tolist()
        probability_map = {key: float(value) for key, value in zip(self.order, probabilities)}
        raw_action = max(probability_map, key=probability_map.get)
        executed_action, override_reasons = (
            reflex_override(obs, raw_action) if self.assist_mode == "reflex" else (raw_action, [])
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rendered_state = record["state"]
        rendered_state_bytes = rendered_state.encode("utf-8")
        from jev_platformer.controller.mock_agent import DecisionResult

        decision = DecisionResult(
            action=executed_action,
            probabilities=probability_map,
            danger_score=UnmeasuredMetric(),  # type: ignore[arg-type]
            jump_recommended=None,  # type: ignore[arg-type]
            latency_ms=elapsed_ms,
            is_mock=False,
        )
        self.decisions.append(
            {
                "decision_index": len(self.decisions),
                "simulation_frame": simulation_frame,
                "simulation_time_s": simulation_frame / FPS,
                "inference_ms": elapsed_ms,
                "raw_action": raw_action,
                "executed_action": executed_action,
                "action": executed_action,
                "override": executed_action != raw_action,
                "override_reasons": override_reasons,
                "probabilities": probability_map,
                "observation": full_observation,
                "state_used": state,
                "representation": REPRESENTATION,
                "question_variant": QUESTION_VARIANT,
                "candidate_order": list(self.order),
                "mapping_matches_candidate_order": meta[0].get("keys") == list(self.order),
                "rendered_state": rendered_state,
                "rendered_state_utf8_bytes": len(rendered_state_bytes),
                "rendered_state_sha256": hashlib.sha256(rendered_state_bytes).hexdigest(),
                "encoded_token_count": len(encoded["ids"]),
                "state_token_count": encoded["seg"].count(0),
                "branch_token_count": len(encoded["ids"]) - encoded["seg"].count(0),
                "queried_outputs": ["action"],
                "unqueried_outputs": {"danger_score": None, "jump_urgency": None},
            }
        )
        return decision


class KevLiveAgent:
    """Adapter shape expected by the fixed game's live branch; no fallback."""

    def __init__(self, adapter: KevDecisionAdapter):
        self.adapter = adapter

    @property
    def is_live(self) -> bool:
        return True

    def decide(self, obs: Any) -> Any:
        return self.adapter.decide(obs, self._next_frame)

    _next_frame = 0


class KevSynchronousAgent:
    """Synchronous replacement for the fixed async wrapper, without its rescue reflex."""

    def __init__(self, live_agent: KevLiveAgent):
        self.live_agent = live_agent
        self.latest_decision = None
        self.next_frame = 0

    def request_decision(self, obs: Any) -> None:
        self.live_agent.adapter.decisions  # keep the adapter visible for diagnostics
        self.latest_decision = self.live_agent.adapter.decide(obs, self.next_frame)
        self.next_frame += FRAMES_PER_DECISION

    def get_action(self, _obs: Any) -> tuple[str, Any]:
        if self.latest_decision is None:
            raise RuntimeError("Kev decision was not available; refusing a fallback action")
        return self.latest_decision.action, self.latest_decision


class DisabledMockAgent:
    def __init__(self, *_args: Any, **_kwargs: Any):
        pass

    def decide(self, _obs: Any) -> Any:
        raise RuntimeError("mock agent path reached during Kev episode")


class KevGameRenderer:
    def __init__(self, original_cls: type[Any], surface: Any):
        self._original = original_cls(surface)

    def render(self, player: Any, level: Any) -> None:
        global CURRENT_SIM_FRAME, LAST_GAME_STATE

        CURRENT_SIM_FRAME += 1
        LAST_GAME_STATE = {
            "is_dead": bool(player.is_dead),
            "has_won": bool(player.has_won),
            "progress_pixels": float(player.max_x),
            "score": int(player.score),
            "coins": int(player.coins),
        }
        self._original.render(player, level)


class KevDashboardRenderer:
    def __init__(self, original_cls: type[Any], surface: Any, offset_x: int):
        self._original = original_cls(surface, offset_x=offset_x)
        self.surface = surface
        self.offset_x = offset_x
        self.font_xs = self._original.font_xs
        self.font_large = self._original.font_large
        self.last_obs = None

    def render(self, obs: Any, decision: Any, is_ai_mode: bool, fps: float) -> None:
        import pygame

        global LAST_DASHBOARD

        self.last_obs = obs
        LAST_DASHBOARD = self
        self._original.render(obs=obs, decision=decision, is_ai_mode=is_ai_mode, fps=fps)

        # Keep the fixed HUD layout, but cover labels that would falsely claim
        # Vercel Jev and expose the actual model/GPU and simulation clock.
        x = self.offset_x + 20
        pygame.draw.rect(self.surface, (22, 27, 42), (x + 138, 50, 235, 27))
        label = self.font_xs.render(f"KEV-0.5B / {MODEL_GPU_NAME}", True, (56, 189, 248))
        self.surface.blit(label, (x + 145, 56))

        pygame.draw.rect(self.surface, (22, 27, 42), (x, 88, 360, 68), border_radius=8)
        pygame.draw.rect(self.surface, (38, 45, 66), (x, 88, 360, 68), 1, border_radius=8)
        latency = self.font_xs.render("INFERENCE DELAY", True, (148, 163, 184))
        latency_value = self.font_large.render(
            f"{decision.latency_ms:.1f} ms" if decision else "--", True, (56, 189, 248)
        )
        self.surface.blit(latency, (x + 12, 98))
        self.surface.blit(latency_value, (x + 12, 116))
        metric = self.font_xs.render("DANGER: N/A", True, (148, 163, 184))
        self.surface.blit(metric, (x + 148, 104))
        metric2 = self.font_xs.render("URGENCY: N/A", True, (148, 163, 184))
        self.surface.blit(metric2, (x + 148, 124))
        progress = self.font_xs.render("PROGRESS", True, (148, 163, 184))
        progress_value = self.font_large.render(f"{int(obs.episode.progress_pixels)}px", True, (248, 250, 252))
        self.surface.blit(progress, (x + 265, 104))
        self.surface.blit(progress_value, (x + 265, 122))

        overlay = pygame.Surface((520, 66), pygame.SRCALPHA)
        overlay.fill((8, 15, 30, 225))
        self.surface.blit(overlay, (10, 10))
        model_line = self.font_xs.render(f"MODEL: {MODEL_NAME} | REAL ADAPTER | GPU: {MODEL_GPU_NAME}", True, (125, 211, 252))
        action_line = self.font_xs.render(
            f"MODEL DECISION: {decision.action if decision else '--'}", True, (248, 250, 252)
        )
        time_line = self.font_xs.render(
            f"SIMULATION TIME: {max(CURRENT_SIM_FRAME, 0) / FPS:06.2f}s (inference waits omitted)",
            True,
            (148, 163, 184),
        )
        self.surface.blit(model_line, (18, 16))
        self.surface.blit(action_line, (18, 34))
        self.surface.blit(time_line, (18, 50))


class KevVideoRecorder:
    def __init__(self, original_cls: type[Any], *args: Any, **kwargs: Any):
        global LAST_RECORDER

        self._original_cls = original_cls
        self._recorder = original_cls(*args, **kwargs)
        self.last_surface = None
        self.appended_terminal_frames = 0
        LAST_RECORDER = self

    @property
    def frame_count(self) -> int:
        return self._recorder.frame_count

    @property
    def process(self) -> Any:
        return self._recorder.process

    def record_frame(self, surface: Any) -> None:
        self.last_surface = surface.copy()
        self._recorder.record_frame(surface)

    def close(self) -> None:
        if self._recorder.process is not None and self.last_surface is not None:
            for _ in range(TERMINAL_HOLD_FRAMES):
                self._recorder.record_frame(self.last_surface)
            self.appended_terminal_frames = TERMINAL_HOLD_FRAMES
        self._recorder.close()


def ffprobe_video(path: Path) -> dict[str, Any]:
    probe = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames,duration,pix_fmt,codec_name",
            "-of",
            "json",
            str(path),
        ]
    )
    decoded = run_command(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=False)
    return {
        "probe": json.loads(probe.stdout),
        "full_decode_returncode": decoded.returncode,
        "full_decode_stderr": decoded.stderr[-4000:],
    }


def write_representative_frames(path: Path, duration_s: float) -> list[dict[str, Any]]:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    times = [("first", 0.0), ("middle", max(0.0, duration_s / 2.0)), ("last", max(0.0, duration_s - 1.0 / FPS))]
    outputs = []
    for name, timestamp in times:
        frame_path = FRAME_DIR / f"{name}.png"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        outputs.append({"name": name, "seconds": timestamp, "path": str(frame_path), "sha256": sha256_file(frame_path)})
    return outputs


def load_and_patch_game(adapter: KevDecisionAdapter) -> tuple[Any, dict[str, str]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, str(GAME_DIR / "src"))
    import jev_platformer.cli as cli

    from jev_platformer.ui.dashboard import DashboardRenderer as OriginalDashboardRenderer
    from jev_platformer.ui.renderer import GameRenderer as OriginalGameRenderer
    from jev_platformer.ui.video_recorder import VideoRecorder as OriginalVideoRecorder

    class PatchedDashboardRenderer(KevDashboardRenderer):
        def __init__(self, surface: Any, offset_x: int):
            super().__init__(OriginalDashboardRenderer, surface, offset_x)

    class PatchedGameRenderer(KevGameRenderer):
        def __init__(self, surface: Any):
            super().__init__(OriginalGameRenderer, surface)

    class PatchedRecorder(KevVideoRecorder):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(OriginalVideoRecorder, *args, **kwargs)

    class PatchedLiveAgent(KevLiveAgent):
        def __init__(self, *_args: Any, **_kwargs: Any):
            super().__init__(adapter)

    class PatchedAsyncAgent(KevSynchronousAgent):
        pass

    cli.GameRenderer = PatchedGameRenderer
    cli.DashboardRenderer = PatchedDashboardRenderer
    cli.VideoRecorder = PatchedRecorder
    cli.JevLiveAgent = PatchedLiveAgent
    cli.AsyncJevAgent = PatchedAsyncAgent
    cli.MockJevAgent = DisabledMockAgent
    return cli, {"game_renderer": "fixed commit + metadata overlay", "agent": "synchronous Kev adapter"}


def main() -> int:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": "jev-colab-lab/kev-jevdash-episode-v1",
        "status": "started",
        "started_at_utc": utc_now(),
        "seed": SEED,
        "level": LEVEL,
        "fps": FPS,
        "frames_per_decision": FRAMES_PER_DECISION,
        "max_simulation_frames": MAX_SIMULATION_FRAMES,
        "terminal_hold_frames_requested": TERMINAL_HOLD_FRAMES,
        "video_time_basis": "simulation_frames_at_60fps; synchronous Kev inference waits are omitted from video time",
        "model_driven": True,
        "model_only": ASSIST_MODE == "none",
        "assist_mode": ASSIST_MODE,
        "representation": REPRESENTATION,
        "question_variant": QUESTION_VARIANT,
        "candidate_order": list(ORDER_VARIANTS.get(ORDER_NAME, ())),
        "fallback_used": False,
        "errors": [],
    }
    try:
        seed_everything()
        game_source = clone_game()
        import t4_inference

        kev_source = t4_inference.clone_upstream()
        model_sources = t4_inference.download_models()
        adapter_dir = Path(os.environ.get("KEV_ADAPTER_PATH", str(t4_inference.ADAPTER_DIR)))
        adapter_variant = "teacher_finetuned" if os.environ.get("KEV_ADAPTER_PATH") else "official_checkpoint"
        adapter = KevDecisionAdapter(
            t4_inference.UPSTREAM_DIR,
            adapter_dir,
            t4_inference.BASE_DIR,
            model_sources,
        )
        global MODEL_GPU_NAME

        MODEL_GPU_NAME = adapter.gpu_name
        cli, patch_info = load_and_patch_game(adapter)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg is not available on PATH")

        cli.run_play(
            mode="live",
            frames_per_decision=FRAMES_PER_DECISION,
            display="all",
            record_path=str(VIDEO_PATH),
            max_frames=MAX_SIMULATION_FRAMES,
        )

        video_info = ffprobe_video(VIDEO_PATH)
        streams = video_info["probe"].get("streams", [{}])
        stream = streams[0] if streams else {}
        duration_s = float(stream.get("duration") or 0.0)
        representative_frames = write_representative_frames(VIDEO_PATH, duration_s)
        status = "cleared" if LAST_GAME_STATE.get("has_won") else ("died" if LAST_GAME_STATE.get("is_dead") else "timeout")
        record_count = len(adapter.decisions)
        video_frames = int(getattr(LAST_RECORDER, "frame_count", 0))
        simulation_frames = max(CURRENT_SIM_FRAME + 1, 0)
        upstream_terminal_frames = UPSTREAM_TERMINAL_HOLD_FRAMES.get(status, 0)
        expected_video_frames = simulation_frames + upstream_terminal_frames + TERMINAL_HOLD_FRAMES
        raw_action_counts = Counter(item["raw_action"] for item in adapter.decisions)
        executed_action_counts = Counter(item["executed_action"] for item in adapter.decisions)
        override_reason_counts = Counter(
            reason for item in adapter.decisions for reason in item.get("override_reasons", [])
        )
        override_count = sum(bool(item.get("override")) for item in adapter.decisions)
        result.update(
            {
                "status": status,
                "completed_at_utc": utc_now(),
                "wall_clock_s": time.perf_counter() - started,
                "game": game_source,
                "game_ref_note": "fixed commit used in isolated clone; the local JevDash checkout was not touched",
                "kev": {
                    "name": MODEL_NAME,
                    "training_source": MODEL_TRAINING_SOURCE,
                    "implementation": kev_source,
                    "model_sources": model_sources,
                    "adapter_variant": adapter_variant,
                    "gpu": adapter.gpu_name,
                    "device": "cuda",
                },
                "patches": patch_info,
                "episode": {
                    "outcome": status,
                    "simulation_frames": simulation_frames,
                    "simulation_time_s": simulation_frames / FPS,
                    "decision_count": record_count,
                    "progress_pixels": LAST_GAME_STATE.get("progress_pixels"),
                    "score": LAST_GAME_STATE.get("score"),
                    "coins": LAST_GAME_STATE.get("coins"),
                    "raw_action_counts": dict(raw_action_counts),
                    "executed_action_counts": dict(executed_action_counts),
                    "override_decision_count": override_count,
                    "override_rate": override_count / record_count if record_count else 0.0,
                    "override_reason_counts": dict(override_reason_counts),
                    "terminal_hold_frames_upstream_cli": upstream_terminal_frames,
                    "terminal_hold_frames_appended_by_adapter": getattr(LAST_RECORDER, "appended_terminal_frames", 0),
                    "expected_video_frames": expected_video_frames,
                },
                "inference": {
                    "total_wait_s": sum(item["inference_ms"] for item in adapter.decisions) / 1000.0,
                    "min_ms": min((item["inference_ms"] for item in adapter.decisions), default=None),
                    "mean_ms": (
                        sum(item["inference_ms"] for item in adapter.decisions) / record_count
                        if record_count
                        else None
                    ),
                    "max_ms": max((item["inference_ms"] for item in adapter.decisions), default=None),
                },
                "video": {
                    "path": str(VIDEO_PATH),
                    "sha256": sha256_file(VIDEO_PATH),
                    "frames_from_recorder": video_frames,
                    "expected_frames": expected_video_frames,
                    "fps": FPS,
                    "duration_s": duration_s,
                    "ffmpeg": ffmpeg_path,
                    "ffprobe_decode": video_info,
                    "representative_frames": representative_frames,
                },
                "decisions": adapter.decisions,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "completed_at_utc": utc_now(),
                "wall_clock_s": time.perf_counter() - started,
                "errors": [
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=30),
                    }
                ],
            }
        )
    EPISODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EPISODE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] in {"cleared", "died", "timeout"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
