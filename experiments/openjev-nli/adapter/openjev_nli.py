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

    def decide(self, observation: Any) -> dict[str, Any]:
        """Score all seven actions and return the raw NLI probabilities."""

        if hasattr(observation, "model_dump"):
            observation_dict = observation.model_dump(mode="json")
        else:
            observation_dict = dict(observation)

        hypotheses = build_hypotheses()
        premise = build_premise(observation_dict)
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
            "selection_rule": "argmax over NLI entailment probability; no heuristic fallback",
        }
