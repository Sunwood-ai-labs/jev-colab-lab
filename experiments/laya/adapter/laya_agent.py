"""Synchronous, real Laya action adapter for the fixed JevDash observation schema."""

from __future__ import annotations

import math
import hashlib
import json
import os
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Mapping, Sequence


MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
LAYA_SOURCE_COMMIT = "d113dca2512fb3eaca313534bc54c7162d87c1d4"
GAME_ACTIONS = (
    "noop",
    "right",
    "right_run",
    "right_jump",
    "right_run_jump",
    "jump",
    "left",
)
PROMPT_PROFILES = ("baseline", "platformer_guided")
STATE_ENCODINGS = ("full_json", "semantic_v1")
MODEL_FILES = (
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
    "rl_agent_config.json",
)

_ACTION_CRITERIA = {
    "noop": (
        "Give no directional or jump input; slow down while grounded and keep "
        "existing horizontal momentum while airborne."
    ),
    "right": (
        "Move right at walking speed while grounded; while airborne, hold right "
        "with the game's full forward air-control speed."
    ),
    "right_run": (
        "Sprint right while grounded; while airborne, hold right with the game's "
        "full forward air-control speed."
    ),
    "right_jump": (
        "While grounded, move right and start a normal jump over a low obstacle or "
        "enemy; while airborne, hold right without starting another jump."
    ),
    "right_run_jump": (
        "While grounded, sprint right and start a strong forward jump over a pipe, "
        "pit, enemy, or blocked path; while airborne, hold right without starting "
        "another jump."
    ),
    "jump": "While grounded, start a vertical jump without directional movement; airborne input cannot double-jump.",
    "left": "Move left at walking speed; airborne left control reverses toward walking speed.",
}


def build_action_question(
    profile: str = "baseline",
    candidate_order: Sequence[str] = GAME_ACTIONS,
) -> Dict[str, Any]:
    """Build a deterministic Laya choice question for a candidate ordering."""

    if profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown Laya prompt profile: {profile!r}")
    order = tuple(candidate_order)
    if set(order) != set(GAME_ACTIONS) or len(order) != len(GAME_ACTIONS):
        raise ValueError("candidate_order must contain each JevDash action exactly once")
    if profile == "baseline":
        instructions = (
            "Choose exactly one action macro for the current observed game state. "
            "Use only the candidate action names provided below."
        )
    else:
        instructions = (
            "Choose exactly one action macro for a 60 FPS side-scrolling platformer. "
            "Preserve forward progress. If the grounded player has a nearby pipe, "
            "pit, enemy, or blocked path, choose right_run_jump for a strong forward "
            "jump; otherwise use right_run on clear ground. While airborne, keep "
            "rightward momentum. The obstacle and gap distances are coarse tile-column "
            "estimates, not exact front-edge distances. Use only the candidate action "
            "names provided below."
        )
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {name: _ACTION_CRITERIA[name] for name in order},
    }


ACTION_QUESTION: Dict[str, Any] = build_action_question("baseline")


def semantic_state_text(state: Mapping[str, Any]) -> str:
    """Render all JevObservation fields in a compact, stable semantic format.

    This is an input representation experiment, not a controller. It retains every
    field from ``JevObservation.model_dump()`` while making the field meanings and
    coarse distance semantics visible to the language encoder.
    """

    player = state["player"]
    hazard = state["hazard"]
    terrain = state["terrain"]
    episode = state["episode"]
    enemy = hazard.get("nearest_enemy")
    enemy_text = "none"
    if enemy is not None:
        enemy_text = (
            f"kind={enemy['kind']} distance_px={enemy['distance_pixels']} "
            f"vertical_offset_px={enemy['vertical_offset_pixels']} "
            f"relative_vx={enemy['relative_velocity_x']} "
            f"contact_frames={enemy['estimated_contact_frames']}"
        )
    radar = " / ".join(str(row) for row in state["local_grid"])
    return "\n".join(
        (
            f"OBJECTIVE: {state['objective']}",
            (
                "PLAYER: "
                f"x_px={player['x']} y_px={player['y']} vx_px_per_frame={player['vx']} "
                f"vy_px_per_frame={player['vy']} grounded={player['grounded']} "
                f"jumping={player['jumping']} airborne_frames={player['airborne_frames']} "
                f"running={player['running']}"
            ),
            (
                "HAZARD: "
                f"enemy_ahead={hazard['enemy_ahead']} nearest_enemy=({enemy_text}) "
                f"jump_must_start_now={hazard['jump_must_start_now']} "
                f"in_danger_zone={hazard['in_danger_zone']}"
            ),
            (
                "TERRAIN: "
                f"obstacle_ahead={terrain['obstacle_ahead']} "
                f"obstacle_distance_tiles={terrain['obstacle_distance_tiles']} "
                f"obstacle_height_tiles={terrain['obstacle_height_tiles']} "
                f"gap_ahead={terrain['gap_ahead']} "
                f"gap_distance_tiles={terrain['gap_distance_tiles']} "
                f"gap_width_tiles={terrain['gap_width_tiles']} "
                f"clear_forward_tiles={terrain['clear_forward_tiles']} "
                "(distances are coarse player-left tile-column estimates)"
            ),
            (
                "EPISODE: "
                f"score={episode['score']} coins={episode['coins']} lives={episode['lives']} "
                f"progress_px={episode['progress_pixels']} "
                f"goal_distance_px={episode['goal_distance_pixels']} "
                f"stalled_frames={episode['stalled_frames']} "
                f"is_dead={episode['is_dead']} has_won={episode['has_won']}"
            ),
            f"LOCAL_RADAR_7x11: {radar}",
        )
    )


def encode_state(state: Dict[str, Any], state_encoding: str) -> Any:
    """Return the exact object passed to Laya for a named representation."""

    if state_encoding == "full_json":
        return state
    if state_encoding == "semantic_v1":
        return semantic_state_text(state)
    raise ValueError(f"Unknown Laya state encoding: {state_encoding!r}")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class LayaDecision:
    action: str
    probabilities: Dict[str, float]
    confidence: float
    inference_ms: float
    input_tokens: int
    prompt_stats: Dict[str, Any]
    model_choice: str
    argmax_action: str
    choice_matches_argmax: bool
    raw_answer: Dict[str, Any]
    prompt_profile: str
    state_encoding: str


class LayaActionAdapter:
    """Load the pinned Laya checkpoint and synchronously choose one game action."""

    def __init__(
        self,
        device: str = "cuda",
        prompt_profile: str = "baseline",
        state_encoding: str = "full_json",
        candidate_order: Sequence[str] = GAME_ACTIONS,
    ):
        if device != "cuda":
            raise ValueError("This capture requires the real Laya model on CUDA; no CPU fallback is allowed")
        if prompt_profile not in PROMPT_PROFILES:
            raise ValueError(f"Unknown Laya prompt profile: {prompt_profile!r}")
        if state_encoding not in STATE_ENCODINGS:
            raise ValueError(f"Unknown Laya state encoding: {state_encoding!r}")
        self.prompt_profile = prompt_profile
        self.state_encoding = state_encoding
        self.candidate_order = tuple(candidate_order)
        self.action_question = build_action_question(prompt_profile, self.candidate_order)

        os.environ.setdefault("USE_TF", "0")
        import torch
        import laya
        from huggingface_hub import snapshot_download

        self.torch = torch
        self.laya = laya
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to run a non-GPU capture")

        download_started = time.perf_counter()
        self.model_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
        )
        self.snapshot_download_seconds = time.perf_counter() - download_started

        load_started = time.perf_counter()
        self.agent = laya.load(self.model_path, device="cuda")
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - load_started
        if self.agent.device.type != "cuda":
            raise RuntimeError(f"Laya loaded on {self.agent.device}; CUDA placement is required")

    def _prompt_stats(self, state_payload: Any, source_state: Dict[str, Any]) -> Dict[str, Any]:
        """Reproduce Laya's sequence budgeting to expose state truncation evidence."""

        from laya.common import build_sequence, render_options, serialize_state

        q = self.agent._to_internal(self.action_question)
        tok = self.agent.tok
        max_len = int(self.agent.cfg.get("max_len", 512))
        head_max_len = int(self.agent.cfg.get("head_max_len", 192))
        mask_tok = tok.mask_token
        if mask_tok is None:
            raise RuntimeError("Laya tokenizer has no mask token")

        options = render_options(q)
        raw_head_ids = tok(
            "%s question: %s" % (q["t"], q["ins"]),
            add_special_tokens=False,
        )["input_ids"]
        option_ids = []
        for option in options:
            option_ids.append(
                [tok.mask_token_id]
                + tok(" " + option.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
            )
        option_budget = head_max_len - sum(len(item) for item in option_ids)
        clipped_options = False
        if option_budget < 16:
            per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
            option_ids = [item[:per_option] for item in option_ids]
            option_budget = head_max_len - sum(len(item) for item in option_ids)
            clipped_options = True
        head_ids = raw_head_ids[: max(8, option_budget)]

        prefix_ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
        marker_positions = []
        for option in option_ids:
            marker_positions.append(len(prefix_ids))
            prefix_ids.extend(option)
        prefix_ids.append(tok.sep_token_id)

        serialized_state = serialize_state(state_payload)
        raw_state_ids = tok(
            serialized_state.replace(mask_tok, " "),
            add_special_tokens=False,
        )["input_ids"]
        state_room = max(0, max_len - len(prefix_ids) - 1)
        retained_state_tokens = min(len(raw_state_ids), state_room)
        sequence_ids, actual_markers = build_sequence(
            tok,
            state_payload,
            q,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        return {
            "representation": self.state_encoding,
            "state_fields": list(source_state.keys()),
            "omitted_state_fields": [],
            "max_context_tokens": max_len,
            "head_budget_tokens": head_max_len,
            "raw_state_tokens": len(raw_state_ids),
            "retained_state_tokens": retained_state_tokens,
            "truncated": len(raw_state_ids) > retained_state_tokens,
            "omitted_state_tokens": max(0, len(raw_state_ids) - retained_state_tokens),
            "question_head_tokens_before_clip": len(raw_head_ids),
            "question_head_tokens_retained": len(head_ids),
            "option_count": len(options),
            "option_tokens_clipped": clipped_options,
            "sequence_tokens": len(sequence_ids),
            "marker_count": len(actual_markers),
            "prefix_tokens_before_state": len(prefix_ids),
            "state_room_tokens": state_room,
            "candidate_order": list(self.candidate_order),
            "prompt_profile": self.prompt_profile,
            "state_input_bytes": len(serialized_state.encode("utf-8")),
            "state_input_chars": len(serialized_state),
            "state_input_sha256": hashlib.sha256(serialized_state.encode("utf-8")).hexdigest(),
        }

    def decide(self, observation: Any) -> LayaDecision:
        source_state = observation.model_dump(mode="json")
        state_payload = encode_state(source_state, self.state_encoding)
        started = time.perf_counter()
        self.torch.cuda.synchronize()
        result = self.agent.predict(state_payload, {"action": self.action_question})
        self.torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        answer = result.get("answers", {}).get("action", {})
        raw_probabilities = answer.get("probabilities", {})
        missing = [name for name in GAME_ACTIONS if name not in raw_probabilities]
        if missing:
            raise RuntimeError(f"Laya omitted required action candidates: {missing}")
        probabilities = {name: float(raw_probabilities[name]) for name in GAME_ACTIONS}
        if not all(math.isfinite(value) and value >= 0.0 for value in probabilities.values()):
            raise RuntimeError("Laya returned a non-finite or negative action probability")
        total = sum(probabilities.values())
        if total <= 0.0:
            raise RuntimeError("Laya returned zero total action probability")
        model_choice = str(answer.get("choice", ""))
        if model_choice not in GAME_ACTIONS:
            raise RuntimeError(f"Laya returned an invalid action choice: {model_choice!r}")
        argmax_action = max(
            self.candidate_order,
            key=lambda name: (probabilities[name], -self.candidate_order.index(name)),
        )

        usage = result.get("usage", {})
        prompt_stats = self._prompt_stats(state_payload, source_state)
        prompt_stats["usage_input_tokens"] = int(usage.get("input_tokens", 0))
        prompt_stats["usage_matches_reconstructed_sequence"] = (
            prompt_stats["usage_input_tokens"] == prompt_stats["sequence_tokens"]
        )
        return LayaDecision(
            action=model_choice,
            probabilities=probabilities,
            confidence=float(answer.get("confidence", 0.0)),
            inference_ms=inference_ms,
            input_tokens=int(usage.get("input_tokens", 0)),
            prompt_stats=prompt_stats,
            model_choice=model_choice,
            argmax_action=argmax_action,
            choice_matches_argmax=model_choice == argmax_action,
            raw_answer=_json_safe(answer),
            prompt_profile=self.prompt_profile,
            state_encoding=self.state_encoding,
        )

    def metadata(self) -> Dict[str, Any]:
        torch = self.torch
        gpu = {
            "available": bool(torch.cuda.is_available()),
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
        }
        return {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "laya_source_commit": LAYA_SOURCE_COMMIT,
            "prompt_profile": self.prompt_profile,
            "state_encoding": self.state_encoding,
            "candidate_order": list(self.candidate_order),
            "package_versions": {
                "laya": _package_version("laya"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "huggingface-hub": _package_version("huggingface-hub"),
                "safetensors": _package_version("safetensors"),
                "numpy": _package_version("numpy"),
            },
            "torch_cuda_version": torch.version.cuda,
            "gpu": gpu,
            "actual_device": str(self.agent.device),
            "parameter_dtype": str(next(self.agent.model.parameters()).dtype),
            "autocast_dtype": str(self.agent.dtype),
            "config": _json_safe(self.agent.cfg),
            "training_provenance": {
                "checkpoint_type": "official Laya decision checkpoint, fine-tuned from ModernBERT encoder",
                "config_training": _json_safe(self.agent.cfg.get("training", {})),
                "jevdash_uses_mock_or_live_jev": False,
                "weights_saved_in_artifacts": False,
            },
            "snapshot_download_seconds": self.snapshot_download_seconds,
            "model_load_seconds": self.load_seconds,
        }
