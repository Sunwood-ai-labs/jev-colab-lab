"""Synchronous, real Laya action adapter for the fixed JevDash observation schema."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Mapping


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
MODEL_FILES = (
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
    "rl_agent_config.json",
)

ACTION_QUESTION: Dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "Choose exactly one action macro for the current observed game state. "
        "Use only the candidate action names provided below."
    ),
    "criteria": {
        "noop": "Coast without a directional input.",
        "right": "Move right at walking speed.",
        "right_run": "Move right at running speed.",
        "right_jump": "Move right while starting a normal jump.",
        "right_run_jump": "Move right while starting a running jump.",
        "jump": "Start a vertical jump without directional movement.",
        "left": "Move left at walking speed.",
    },
}


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
    raw_answer: Dict[str, Any]


class LayaActionAdapter:
    """Load the pinned Laya checkpoint and synchronously choose one game action."""

    def __init__(self, device: str = "cuda"):
        if device != "cuda":
            raise ValueError("This capture requires the real Laya model on CUDA; no CPU fallback is allowed")

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

    def _prompt_stats(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Reproduce Laya's sequence budgeting to expose state truncation evidence."""

        from laya.common import build_sequence, render_options, serialize_state

        q = self.agent._to_internal(ACTION_QUESTION)
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

        raw_state_ids = tok(
            serialize_state(state).replace(mask_tok, " "),
            add_special_tokens=False,
        )["input_ids"]
        state_room = max(0, max_len - len(prefix_ids) - 1)
        retained_state_tokens = min(len(raw_state_ids), state_room)
        sequence_ids, actual_markers = build_sequence(
            tok,
            state,
            q,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        return {
            "representation": "full JevObservation.model_dump() JSON; no summary or control rule inserted",
            "state_fields": list(state.keys()),
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
        }

    def decide(self, observation: Any) -> LayaDecision:
        state = observation.model_dump(mode="json")
        started = time.perf_counter()
        self.torch.cuda.synchronize()
        result = self.agent.predict(state, {"action": ACTION_QUESTION})
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
        action = max(GAME_ACTIONS, key=lambda name: (probabilities[name], -GAME_ACTIONS.index(name)))
        model_choice = str(answer.get("choice", ""))
        if model_choice not in GAME_ACTIONS:
            raise RuntimeError(f"Laya returned an invalid action choice: {model_choice!r}")

        usage = result.get("usage", {})
        prompt_stats = self._prompt_stats(state)
        prompt_stats["usage_input_tokens"] = int(usage.get("input_tokens", 0))
        prompt_stats["usage_matches_reconstructed_sequence"] = (
            prompt_stats["usage_input_tokens"] == prompt_stats["sequence_tokens"]
        )
        return LayaDecision(
            action=action,
            probabilities=probabilities,
            confidence=float(answer.get("confidence", 0.0)),
            inference_ms=inference_ms,
            input_tokens=int(usage.get("input_tokens", 0)),
            prompt_stats=prompt_stats,
            model_choice=model_choice,
            raw_answer=_json_safe(answer),
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
