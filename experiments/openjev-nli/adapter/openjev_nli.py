"""Synchronous OpenJev NLI adapter for JevDash.

The adapter intentionally has no heuristic or mock fallback.  Each legal
JevDash action is represented by one NLI hypothesis, and the action with the
largest entailment probability is applied exactly as returned by the model.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from typing import Any, Mapping, Sequence


MODEL_ID = "AlexWortega/openjev"
MODEL_REVISION = "b32265f4700df7c02532933c9a4ff258a449d7ac"
MODEL_SUBFOLDER = "qwen3.5-4b-nli"
MODEL_LICENSE = "MIT"
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
ACTION_OPTIONS = (
    "noop",
    "right",
    "right_run",
    "right_jump",
    "right_run_jump",
    "jump",
    "left",
)
LABELS = ("contradiction", "entailment", "neutral")
ACTION_DESCRIPTIONS = {
    "noop": "keep the current horizontal momentum unchanged and do not start a jump",
    "right": "move right at walking speed without starting a jump",
    "right_run": "move right at running speed without starting a jump",
    "right_jump": "move right and start a normal jump",
    "right_run_jump": "move right at running speed and start a running jump",
    "jump": "start a normal jump without intentional horizontal movement",
    "left": "move left without starting a jump",
}

# The model card's zero-shot reranker uses short answer completions beginning
# with ``The correct answer is:``.  Keep these phrases close in length so the
# NLI score is less sensitive to a candidate's wording than the legacy
# verbose descriptions above.
CARD_ACTION_PHRASES = {
    "noop": "release horizontal control and do not jump",
    "right": "move right without jumping",
    "right_run": "run right without jumping",
    "right_jump": "move right and jump with normal strength",
    "right_run_jump": "run right and jump with full strength",
    "jump": "jump without horizontal input",
    "left": "move left without jumping",
}
CARD_ACTION_OPTIONS = tuple(ACTION_OPTIONS)
APPLICABILITY_HYPOTHESES = {
    "noop": "The player should release horizontal control and not jump because no immediate hazard requires a jump.",
    "right": "The player should move right without jumping because the forward path is clear.",
    "right_run": "The player should run right without jumping because the forward path is clear.",
    "right_jump": "The player should move right and start a normal jump to clear a nearby hazard.",
    "right_run_jump": "The player should run right and start a strong jump to clear a nearby hazard.",
    "jump": "The player should jump without horizontal input to clear a nearby hazard.",
    "left": "The player should move left without jumping because forward movement is unsafe.",
}
TWO_ACTION_OPTIONS = ("right_run", "right_run_jump")


def _get(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def _fmt(value: Any) -> str:
    """Render a scalar without adding prose or locale-dependent formatting."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def build_compact_premise(observation: Mapping[str, Any]) -> str:
    """Build a compact, human-readable state premise for zero-shot reranking.

    The legacy premise is intentionally kept in ``build_premise`` for exact
    reproduction of the first capture.  This version preserves every
    decision-relevant telemetry field while removing JSON punctuation and
    repeated schema names that consume context without changing the state.
    """

    enemy = _get(observation, "hazard", "nearest_enemy", default={}) or {}
    grid = _get(observation, "local_grid", default=[]) or []
    radar = "/".join(str(row) for row in grid)
    return (
        "Choose the next action in a deterministic 2D platformer. "
        "Reach the goal without dying. "
        "Player "
        f"x={_fmt(_get(observation, 'player', 'x'))} "
        f"y={_fmt(_get(observation, 'player', 'y'))} "
        f"vx={_fmt(_get(observation, 'player', 'vx'))} "
        f"vy={_fmt(_get(observation, 'player', 'vy'))} "
        f"grounded={_fmt(_get(observation, 'player', 'grounded'))} "
        f"jumping={_fmt(_get(observation, 'player', 'jumping'))} "
        f"airborne={_fmt(_get(observation, 'player', 'airborne_frames'))} "
        f"running={_fmt(_get(observation, 'player', 'running'))}. "
        "Hazard "
        f"enemy_ahead={_fmt(_get(observation, 'hazard', 'enemy_ahead'))} "
        f"enemy_kind={_fmt(_get(enemy, 'kind'))} "
        f"enemy_distance={_fmt(_get(enemy, 'distance_pixels'))} "
        f"enemy_vertical={_fmt(_get(enemy, 'vertical_offset_pixels'))} "
        f"enemy_contact={_fmt(_get(enemy, 'estimated_contact_frames'))} "
        f"jump_now={_fmt(_get(observation, 'hazard', 'jump_must_start_now'))}. "
        "Terrain "
        f"obstacle_ahead={_fmt(_get(observation, 'terrain', 'obstacle_ahead'))} "
        f"obstacle_distance={_fmt(_get(observation, 'terrain', 'obstacle_distance_tiles'))} "
        f"obstacle_height={_fmt(_get(observation, 'terrain', 'obstacle_height_tiles'))} "
        f"gap_ahead={_fmt(_get(observation, 'terrain', 'gap_ahead'))} "
        f"gap_distance={_fmt(_get(observation, 'terrain', 'gap_distance_tiles'))} "
        f"gap_width={_fmt(_get(observation, 'terrain', 'gap_width_tiles'))} "
        f"clear_forward={_fmt(_get(observation, 'terrain', 'clear_forward_tiles'))}. "
        "Episode "
        f"progress={_fmt(_get(observation, 'episode', 'progress_pixels'))} "
        f"goal_distance={_fmt(_get(observation, 'episode', 'goal_distance_pixels'))} "
        f"stalled={_fmt(_get(observation, 'episode', 'stalled_frames'))} "
        f"dead={_fmt(_get(observation, 'episode', 'is_dead'))} "
        f"won={_fmt(_get(observation, 'episode', 'has_won'))}. "
        f"Radar {radar}"
    )


def build_rules_v2_premise(observation: Mapping[str, Any]) -> str:
    """Add the fixed game's action physics and telemetry semantics to state."""

    return (
        build_compact_premise(observation)
        + " Physics rules: on ground, right moves at walking speed, right_run moves at running speed, "
        "right_jump starts a normal jump, and right_run_jump starts a stronger running jump. "
        "A normal jump starts with vertical velocity -13.5; a running jump starts with -15.5. "
        "While airborne, right actions set forward air velocity and pressing jump cannot start a second jump. "
        "Noop decelerates on ground but preserves horizontal air momentum. "
        "Obstacle and gap distances are coarse forward tile-scan distances from the player's tile, not pixel gaps. "
        "Grounded stalled frames mean forward motion has stopped; a nearby obstacle or gap may require taking off early."
    )


def build_card_hypotheses(actions: Sequence[str] | None = None) -> list[dict[str, str]]:
    """Build model-card-style answer hypotheses in a caller-specified order."""

    selected = tuple(actions or CARD_ACTION_OPTIONS)
    unknown = [action for action in selected if action not in CARD_ACTION_PHRASES]
    if unknown or len(selected) != len(CARD_ACTION_OPTIONS) or set(selected) != set(CARD_ACTION_OPTIONS):
        raise ValueError(f"candidate actions must contain exactly {CARD_ACTION_OPTIONS}: {selected}")
    return [
        {
            "action": action,
            "phrase": CARD_ACTION_PHRASES[action],
            "hypothesis": f"The correct answer is: {CARD_ACTION_PHRASES[action]}.",
        }
        for action in selected
    ]


def build_applicability_hypotheses(actions: Sequence[str] | None = None) -> list[dict[str, str]]:
    """Build fixed action-applicability statements for the NLI decision."""

    selected = tuple(actions or ACTION_OPTIONS)
    unknown = [action for action in selected if action not in APPLICABILITY_HYPOTHESES]
    if unknown or len(selected) != len(ACTION_OPTIONS) or set(selected) != set(ACTION_OPTIONS):
        raise ValueError(f"candidate actions must contain exactly {ACTION_OPTIONS}: {selected}")
    return [
        {
            "action": action,
            "hypothesis": APPLICABILITY_HYPOTHESES[action],
        }
        for action in selected
    ]


def build_conditioned_premise(observation: Mapping[str, Any]) -> str:
    """Normalize raw hazard flags into facts without selecting an action."""

    hazards: list[str] = []
    if _get(observation, "terrain", "gap_ahead"):
        hazards.append("gap")
    if _get(observation, "terrain", "obstacle_ahead"):
        hazards.append("obstacle")
    if _get(observation, "hazard", "enemy_ahead"):
        hazards.append("enemy")
    immediate_hazard = "+".join(hazards) if hazards else "none"
    path_clear = "no" if hazards else "yes"
    grounded = "yes" if _get(observation, "player", "grounded") else "no"
    stalled = "yes" if (_get(observation, "episode", "stalled_frames", default=0) or 0) >= 3 else "no"
    return (
        build_rules_v2_premise(observation)
        + f" Normalized facts: path_clear={path_clear}; immediate_hazard={immediate_hazard}; "
        f"grounded={grounded}; stalled={stalled}."
    )


def build_conditioned_two_action_hypotheses() -> list[dict[str, str]]:
    """Return the explicitly restricted run-versus-running-jump comparison."""

    return [
        {
            "action": "right_run",
            "hypothesis": "The forward path is clear, so the player should run right without jumping.",
        },
        {
            "action": "right_run_jump",
            "hypothesis": "A nearby gap, obstacle, or enemy is present, so the player should run right and start a strong jump.",
        },
    ]


def build_premise(observation: Mapping[str, Any]) -> str:
    """Build the one shared premise sent to every action hypothesis."""

    serialized = json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "You are choosing the next macro-action in a deterministic 2D platformer. "
        "The objective is to reach the Level 1 goal without dying. "
        "Use only the structured game state below; do not assume hidden state. "
        f"Structured game observation: {serialized}"
    )


def build_hypotheses() -> list[dict[str, str]]:
    """Return the fixed seven-choice mapping used for every model decision."""

    return [
        {
            "action": action,
            "hypothesis": f"The best next macro-action is {action}: {ACTION_DESCRIPTIONS[action]}.",
        }
        for action in ACTION_OPTIONS
    ]


class OpenJevNLIAdapter:
    """Load the pinned 4B checkpoint and synchronously score seven actions."""

    def __init__(self, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("OpenJev NLI requires CUDA; no GPU is available")

        self.torch = torch
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        self.max_length = max_length

        load_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            subfolder=MODEL_SUBFOLDER,
        )
        self.tokenizer.padding_side = "right"
        model_kwargs = {
            "revision": MODEL_REVISION,
            "subfolder": MODEL_SUBFOLDER,
            "device_map": {"": self.device},
            "low_cpu_mem_usage": True,
            "dtype": torch.bfloat16,
        }
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **model_kwargs)
        except TypeError as error:
            if "dtype" not in str(error):
                raise
            model_kwargs.pop("dtype")
            model_kwargs["torch_dtype"] = torch.bfloat16
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **model_kwargs)
        self.model.eval()
        torch.cuda.synchronize()
        self.load_elapsed_ms = round((time.perf_counter() - load_start) * 1000.0, 4)

        model_device = next(self.model.parameters()).device
        if model_device.type != "cuda":
            raise RuntimeError(f"OpenJev model was not placed on CUDA: {model_device}")

        self.nli_template = getattr(
            self.model.config,
            "nli_template",
            "Premise: {premise}\nHypothesis: {hypothesis}",
        )
        raw_labels = getattr(self.model.config, "id2label", {})
        self.label_indices = {
            str(label).lower(): int(index)
            for index, label in raw_labels.items()
        }
        missing = [label for label in LABELS if label not in self.label_indices]
        if missing:
            raise RuntimeError(f"OpenJev config is missing expected labels: {missing}")

        properties = torch.cuda.get_device_properties(0)
        self.gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "multi_processor_count": int(properties.multi_processor_count),
        }

    def environment(self) -> dict[str, Any]:
        package_versions: dict[str, str | None] = {}
        for package in ("torch", "transformers", "huggingface-hub", "accelerate", "safetensors", "tokenizers"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        return {
            "python": __import__("sys").version.split()[0],
            "torch_version": self.torch.__version__,
            "torch_cuda_version": self.torch.version.cuda,
            "package_versions": package_versions,
            "gpu": self.gpu,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
        }

    def decide(
        self,
        observation: Any,
        profile: str = "legacy",
        candidate_order: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Score all seven actions and return raw NLI probabilities.

        ``legacy`` exactly reproduces the first JevDash capture.  ``card``
        follows the fixed checkpoint's published reranking convention and is
        used by the improvement audit.
        """

        if hasattr(observation, "model_dump"):
            observation_dict = observation.model_dump(mode="json")
        else:
            observation_dict = dict(observation)

        if profile == "legacy":
            hypotheses = build_hypotheses()
            premise = build_premise(observation_dict)
        elif profile in ("card", "rules-v2", "applicability", "conditioned-two"):
            hypotheses = (
                build_applicability_hypotheses(candidate_order)
                if profile == "applicability"
                else (
                    build_conditioned_two_action_hypotheses()
                    if profile == "conditioned-two"
                    else build_card_hypotheses(candidate_order)
                )
            )
            if profile == "card":
                premise = build_compact_premise(observation_dict)
            elif profile == "conditioned-two":
                premise = build_conditioned_premise(observation_dict)
            else:
                premise = build_rules_v2_premise(observation_dict)
        else:
            raise ValueError(f"unknown OpenJev input profile: {profile}")
        pair_texts = [
            self.nli_template.format(premise=premise, hypothesis=item["hypothesis"])
            for item in hypotheses
        ]

        total_start = time.perf_counter()
        tokenize_start = time.perf_counter()
        encoded = self.tokenizer(
            pair_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        tokenize_ms = (time.perf_counter() - tokenize_start) * 1000.0

        self.torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**encoded)
        self.torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        rows = outputs.logits.float().softmax(dim=-1).detach().cpu().tolist()
        probabilities: list[dict[str, Any]] = []
        for item, row in zip(hypotheses, rows):
            class_probabilities = {
                label: round(float(row[self.label_indices[label]]), 8)
                for label in LABELS
            }
            probabilities.append(
                {
                    "action": item["action"],
                    "hypothesis": item["hypothesis"],
                    **({"phrase": item["phrase"]} if "phrase" in item else {}),
                    "probabilities": class_probabilities,
                    "entailment_score": class_probabilities["entailment"],
                }
            )

        chosen = max(probabilities, key=lambda item: item["entailment_score"])
        return {
            "action": chosen["action"],
            "probabilities": probabilities,
            "token_count": int(encoded["attention_mask"].sum().item()),
            "sequence_length": int(encoded["input_ids"].shape[-1]),
            "tokenization_ms": round(tokenize_ms, 4),
            "forward_ms": round(forward_ms, 4),
            "latency_ms": round(total_ms, 4),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_subfolder": MODEL_SUBFOLDER,
            "input_profile": profile,
            "premise": premise,
            "premise_char_count": len(premise),
            "premise_byte_count_utf8": len(premise.encode("utf-8")),
            "candidate_order": [item["action"] for item in hypotheses],
            "selection_rule": "argmax over NLI entailment probability; no heuristic fallback",
        }
