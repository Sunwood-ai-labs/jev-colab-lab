"""Audit Laya state representations and candidate-order sensitivity on fixed JevDash fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
LAYA_SOURCE_COMMIT = "d113dca2512fb3eaca313534bc54c7162d87c1d4"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
GAME_ACTIONS = (
    "noop",
    "right",
    "right_run",
    "right_jump",
    "right_run_jump",
    "jump",
    "left",
)
MODEL_FILES = (
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
    "rl_agent_config.json",
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_fixtures(game_src: Path) -> dict[str, dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, str(game_src))
    import pygame

    pygame.init()
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    level = Level(1)

    def at(x: float, stalled: int = 0) -> dict[str, Any]:
        player = Player(x, 540)
        player.grounded = True
        player.stalled_frames = stalled
        player.max_x = x
        return TelemetryExtractor.extract(player, level).model_dump(mode="json")

    fixtures = {
        "start": at(96),
        "enemy_ahead": at(560),
        "gap_edge": at(2070),
        "pipe_stalled": at(2470, stalled=1414),
    }
    pygame.quit()
    return fixtures


def prompt_stats(agent: Any, state_payload: Any, question: dict[str, Any]) -> dict[str, Any]:
    from laya.common import build_sequence, serialize_state

    internal = agent._to_internal(question)
    sequence, markers = build_sequence(
        agent.tok,
        state_payload,
        internal,
        max_len=int(agent.cfg.get("max_len", 512)),
        head_max_len=int(agent.cfg.get("head_max_len", 192)),
    )
    serialized = serialize_state(state_payload)
    raw_tokens = len(agent.tok(serialized, add_special_tokens=False)["input_ids"])
    empty_sequence, _ = build_sequence(
        agent.tok,
        "",
        internal,
        max_len=int(agent.cfg.get("max_len", 512)),
        head_max_len=int(agent.cfg.get("head_max_len", 192)),
    )
    state_room = max(0, int(agent.cfg.get("max_len", 512)) - (len(empty_sequence) - 1) - 1)
    return {
        "state_input_chars": len(serialized),
        "state_input_bytes": len(serialized.encode("utf-8")),
        "state_input_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "raw_state_tokens": raw_tokens,
        "sequence_tokens": len(sequence),
        "marker_count": len(markers),
        "max_context_tokens": int(agent.cfg.get("max_len", 512)),
        "state_room_tokens": state_room,
        "retained_state_tokens": min(raw_tokens, state_room),
        "truncated": raw_tokens > state_room,
        "omitted_state_tokens": max(0, raw_tokens - state_room),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    import laya

    try:
        from adapter.laya_agent import build_action_question, encode_state
    except ImportError:
        from laya_agent import build_action_question, encode_state

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; prompt audit refuses CPU execution")

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=list(MODEL_FILES),
    )
    load_started = time.perf_counter()
    agent = laya.load(model_path, device="cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    if agent.device.type != "cuda":
        raise RuntimeError(f"Laya loaded on {agent.device}; CUDA is required")

    fixtures = build_fixtures(args.game_src)
    variants = (
        {
            "name": "baseline_json_canonical",
            "prompt_profile": "baseline",
            "state_encoding": "full_json",
            "candidate_order": GAME_ACTIONS,
        },
        {
            "name": "guided_semantic_canonical",
            "prompt_profile": "platformer_guided",
            "state_encoding": "semantic_v1",
            "candidate_order": GAME_ACTIONS,
        },
        {
            "name": "rules_v2_semantic_canonical",
            "prompt_profile": "platformer_rules_v2",
            "state_encoding": "semantic_v1",
            "candidate_order": GAME_ACTIONS,
        },
        {
            "name": "guided_semantic_jump_first",
            "prompt_profile": "platformer_guided",
            "state_encoding": "semantic_v1",
            "candidate_order": (
                "right_run_jump",
                "right_jump",
                "right_run",
                "right",
                "jump",
                "noop",
                "left",
            ),
        },
        {
            "name": "rules_v2_semantic_forward_binary",
            "prompt_profile": "platformer_rules_v2",
            "state_encoding": "semantic_v1",
            "candidate_order": ("right_run", "right_run_jump"),
        },
    )

    observations: dict[str, Any] = {}
    for variant in variants:
        question = {"action": build_action_question(variant["prompt_profile"], variant["candidate_order"])}
        rows = []
        for fixture_name, state in fixtures.items():
            state_payload = encode_state(state, variant["state_encoding"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = agent.predict(state_payload, question)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            answer = result["answers"]["action"]
            rows.append(
                {
                    "fixture": fixture_name,
                    "state_summary": {
                        "player_x": state["player"]["x"],
                        "grounded": state["player"]["grounded"],
                        "stalled_frames": state["episode"]["stalled_frames"],
                        "obstacle_ahead": state["terrain"]["obstacle_ahead"],
                        "obstacle_distance_tiles": state["terrain"]["obstacle_distance_tiles"],
                        "gap_ahead": state["terrain"]["gap_ahead"],
                        "gap_distance_tiles": state["terrain"]["gap_distance_tiles"],
                        "enemy_ahead": state["hazard"]["enemy_ahead"],
                    },
                    "choice": answer.get("choice"),
                    "probabilities": answer.get("probabilities", {}),
                    "confidence": answer.get("confidence"),
                    "input_tokens": result.get("usage", {}).get("input_tokens"),
                    "inference_ms": elapsed_ms,
                    "prompt_stats": prompt_stats(agent, state_payload, question["action"]),
                }
            )
        observations[variant["name"]] = {
            "prompt_profile": variant["prompt_profile"],
            "state_encoding": variant["state_encoding"],
            "candidate_order": list(variant["candidate_order"]),
            "fixtures": rows,
        }

    return {
        "status": "ok",
        "created_at_utc": utc_now(),
        "experiment": "jev-colab-lab/laya/jevdash-prompt-audit",
        "game": {"repository": "https://github.com/Sunwood-ai-labs/jevdash", "commit": GAME_COMMIT, "level": 1, "seed": 42},
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "laya_source_commit": LAYA_SOURCE_COMMIT,
            "actual_device": str(agent.device),
            "gpu": torch.cuda.get_device_name(0),
            "load_seconds": load_seconds,
            "packages": {name: package_version(name) for name in ("laya", "torch", "transformers", "huggingface-hub")},
        },
        "variants": observations,
        "note": "Real Laya CUDA prompt audit; no game control or fallback was applied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:1200],
                "traceback": traceback.format_exc()[-4000:],
            },
        }
        exit_code = 1
    else:
        exit_code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
