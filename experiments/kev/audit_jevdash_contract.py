"""Audit JevDash observations through the Kev serving contract.

This script is deliberately separate from gameplay.  It records the exact
state text, UTF-8 bytes, option order, token ids, and (when model paths are
provided) Kev probabilities for state-changing deterministic fixtures.  It
does not apply model actions and never provides a fallback action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
FPS = 60
DECISION_INTERVAL = 8
ACTION_ORDER = ("noop", "right", "right_run", "right_jump", "right_run_jump", "jump", "left")
ORDER_VARIANTS = {
    "canonical": ACTION_ORDER,
    "right_priority": ("right_run_jump", "right_jump", "right_run", "right", "noop", "jump", "left"),
    "reverse": tuple(reversed(ACTION_ORDER)),
}
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
        "Move right and start a normal jump with upward velocity -13.5. The jump starts with "
        "the normal upward impulse; "
        "a jump input cannot create a second jump while already airborne."
    ),
    "right_run_jump": (
        "Move right at running speed and start the stronger running jump with upward velocity -15.5 "
        "over an immediate pipe, "
        "gap, or enemy; an airborne jump input cannot create a second jump."
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def dump_observation(observation: Any) -> dict[str, Any]:
    return observation.model_dump(mode="json") if hasattr(observation, "model_dump") else dict(observation)


def compact_state(full: Mapping[str, Any]) -> dict[str, Any]:
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


def compact_policy_state(full: Mapping[str, Any]) -> dict[str, Any]:
    state = compact_state(full)
    state["control_hint"] = (
        "On clear ground advance right. If a gap, pipe, or enemy is ahead, use right_run_jump. "
        "Do not move left unless the state requires backtracking."
    )
    return state


def _step_game(player: Any, level: Any, action: str) -> None:
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


def collect_fixtures(game_root: Path) -> dict[str, dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    pygame.init()
    sys.path.insert(0, str(game_root / "src"))
    from jev_platformer.engine.entities import Player
    from jev_platformer.engine.world import Level
    from jev_platformer.telemetry.extractor import TelemetryExtractor

    fixtures: dict[str, dict[str, Any]] = {}
    paths = ("right_run", "right_run_jump")
    fixed_frames = {0, 8, 32, 64, 128, 256, 512}
    for path_action in paths:
        level = Level(1)
        player = Player(level.start_pos[0], level.start_pos[1])
        captured: dict[str, dict[str, Any]] = {}
        for frame in range(1800):
            observation = TelemetryExtractor.extract(player, level)
            full = dump_observation(observation)
            tags: list[str] = []
            if frame in fixed_frames:
                tags.append(f"{path_action}_frame_{frame}")
            if full["terrain"]["obstacle_ahead"] and "obstacle_ahead" not in captured:
                tags.append("obstacle_ahead")
            if full["terrain"]["gap_ahead"] and "gap_ahead" not in captured:
                tags.append("gap_ahead")
            if full["hazard"]["enemy_ahead"] and "enemy_ahead" not in captured:
                tags.append("enemy_ahead")
            if full["episode"]["stalled_frames"] >= 3 and "stalled" not in captured:
                tags.append("stalled")
            for tag in tags:
                if tag not in captured:
                    captured[tag] = {
                        "path_action": path_action,
                        "frame": frame,
                        "observation": full,
                    }
            _step_game(player, level, path_action)
            if player.is_dead or player.has_won:
                break
        for key, value in captured.items():
            fixture_name = key if key in {"obstacle_ahead", "gap_ahead", "enemy_ahead", "stalled"} else key
            fixtures.setdefault(fixture_name, value)
    pygame.quit()
    required = {"right_run_frame_0", "obstacle_ahead", "stalled", "gap_ahead", "enemy_ahead"}
    missing = sorted(required - fixtures.keys())
    if missing:
        raise RuntimeError(f"deterministic fixture collection missing {missing}; captured={sorted(fixtures)}")
    return fixtures


def make_request(state: Mapping[str, Any], question_variant: str, order: tuple[str, ...]) -> dict[str, Any]:
    from kev.api import SystemOneRequest, to_record

    request = {
        "state": state,
        "model": "kev-0.5b",
        "questions": {
            "action": {
                "type": "choice",
                "instructions": QUESTION_VARIANTS[question_variant],
                "criteria": {key: ACTION_DESCRIPTIONS[key] for key in order},
            }
        },
    }
    validated = SystemOneRequest.model_validate(request)
    record, meta = to_record(validated)
    return {"request": request, "record": record, "meta": meta}


def encode_audit(tokenizer: Any, state: Mapping[str, Any], question_variant: str, order: tuple[str, ...]) -> dict[str, Any]:
    from kev.model import encode

    material = make_request(state, question_variant, order)
    encoded = encode(tokenizer, material["record"])
    rendered_state = material["record"]["state"]
    options = material["record"]["questions"][0]["options"]
    encoded_ids = encoded["ids"]
    return {
        "question_variant": question_variant,
        "candidate_order": list(order),
        "mapping_matches_candidate_order": material["meta"][0]["keys"] == list(order),
        "rendered_state": rendered_state,
        "rendered_state_utf8_bytes": len(rendered_state.encode("utf-8")),
        "rendered_state_sha256": sha256_bytes(rendered_state.encode("utf-8")),
        "rendered_record_sha256": json_sha256(material["record"]),
        "options": options,
        "option_token_counts": [len(tokenizer(option, add_special_tokens=False).input_ids) for option in options],
        "encoded_token_count": len(encoded_ids),
        "state_token_count": sum(segment == 0 for segment in encoded["seg"]),
        "branch_token_count": len(encoded_ids) - sum(segment == 0 for segment in encoded["seg"]),
        "encoded_ids_sha256": json_sha256(encoded_ids),
        "encoded_decide_indices": encoded["decide_idx"],
        "encoded_option_indices": encoded["opt_idx"],
        "record": material["record"],
    }


class KevModelProbe:
    def __init__(self, upstream: Path, base: Path, adapter: Path):
        import torch
        from peft import PeftModel
        from kev.model import DecisionModel, load_tokenizer

        self.torch = torch
        self.device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("model audit requested without CUDA")
        self.tokenizer = load_tokenizer(str(base))
        model = DecisionModel(str(base), self.tokenizer, self.device, lora=None)
        model.lm = PeftModel.from_pretrained(model.lm, str(adapter)).to(self.device)
        head = torch.load(adapter / "head.pt", map_location="cpu", weights_only=False)
        model.head.load_state_dict(head["head"])
        model.eval()
        self.model = model
        self.upstream = str(upstream)

    def score(self, audit: dict[str, Any]) -> dict[str, Any]:
        from kev.api import to_record, SystemOneRequest
        from kev.model import encode

        request = SystemOneRequest.model_validate(audit["record_request"])
        record, meta = to_record(request)
        encoded = encode(self.tokenizer, record)
        started = time.perf_counter()
        with self.torch.inference_mode():
            probabilities = self.model.probs(encoded)[0].tolist()
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        probability_map = {key: float(value) for key, value in zip(meta[0]["keys"], probabilities)}
        return {
            "probabilities_by_candidate": probability_map,
            "argmax_action": max(probability_map, key=probability_map.get),
            "inference_ms": elapsed_ms,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--upstream",
        type=Path,
        required=True,
        help="Official Kev source checkout; required for the serving-contract encoder.",
    )
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if bool(args.base) != bool(args.adapter):
        raise SystemExit("--base and --adapter must be supplied together for a model audit")

    sys.path.insert(0, str(args.upstream)) if args.upstream else None
    from transformers import AutoTokenizer

    fixtures = collect_fixtures(args.game_root)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    probe = KevModelProbe(args.upstream, args.base, args.adapter) if args.base and args.adapter else None
    rows: list[dict[str, Any]] = []
    for fixture_name, fixture in fixtures.items():
        full = fixture["observation"]
        states = {
            "full_json": full,
            "compact_json": compact_state(full),
            "compact_policy_json": compact_policy_state(full),
        }
        for representation, state in states.items():
            for question_variant in QUESTION_VARIANTS:
                for order_name, order in ORDER_VARIANTS.items():
                    audit = encode_audit(tokenizer, state, question_variant, order)
                    request_material = make_request(state, question_variant, order)
                    audit["record_request"] = request_material["request"]
                    if probe:
                        audit["model"] = probe.score(audit)
                    rows.append(
                        {
                            "fixture": fixture_name,
                            "path_action": fixture["path_action"],
                            "fixture_frame": fixture["frame"],
                            "representation": representation,
                            "audit": audit,
                        }
                    )
    result = {
        "schema": "jev-colab-lab/kev-jevdash-contract-audit-v1",
        "game_commit": GAME_COMMIT,
        "seed": 42,
        "decision_interval_frames": DECISION_INTERVAL,
        "question_variants": QUESTION_VARIANTS,
        "order_variants": {name: list(order) for name, order in ORDER_VARIANTS.items()},
        "model_audit": bool(probe),
        "fixtures": {name: value for name, value in fixtures.items()},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fixture_count": len(fixtures), "row_count": len(rows), "model_audit": bool(probe)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
