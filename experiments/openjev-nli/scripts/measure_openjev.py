"""Measure the original AlexWortega/openjev 4B NLI checkpoint.

The script deliberately measures forward passes on already-tokenized inputs.
Model download/deserialization is reported separately as load time, while the
first forward pass and steady-state batches are reported independently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_ID = "AlexWortega/openjev"
MODEL_REVISION = "b32265f4700df7c02532933c9a4ff258a449d7ac"
MODEL_SUBFOLDER = "qwen3.5-4b-nli"
MODEL_LICENSE = "MIT"
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
LABELS = ("contradiction", "entailment", "neutral")
DEFAULT_OUTPUT = "/content/openjev-nli-result.json"
DEFAULT_REMOTE_DATASET = "/content/openjev-nli-fixture.json"
DEFAULT_LOCAL_DATASET = "data/fixture.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 4)


def summarize_timings(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min_ms": None, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min_ms": round(min(numeric), 4),
        "mean_ms": round(statistics.fmean(numeric), 4),
        "p50_ms": percentile(numeric, 0.50),
        "p95_ms": percentile(numeric, 0.95),
        "max_ms": round(max(numeric), 4),
    }


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("fixture must contain a non-empty items list")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("fixture items must be objects")
        hypotheses = item.get("hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError(f"fixture item {item.get('id', '<unknown>')} has no hypotheses")
        gold_index = item.get("gold_index")
        if not isinstance(gold_index, int) or not 0 <= gold_index < len(hypotheses):
            raise ValueError(f"fixture item {item.get('id', '<unknown>')} has invalid gold_index")
    return payload


def build_pairs(fixture: dict[str, Any], template: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for item in fixture["items"]:
        for candidate_index, hypothesis in enumerate(item["hypotheses"]):
            pairs.append(
                {
                    "question_id": item["id"],
                    "candidate_index": candidate_index,
                    "premise": item["premise"],
                    "hypothesis": hypothesis,
                    "text": template.format(premise=item["premise"], hypothesis=hypothesis),
                }
            )
    return pairs


def redact_error(message: str) -> str:
    redacted = str(message)
    redacted = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "<path>", redacted)
    redacted = re.sub(r"/home/[^\s'\"]+", "<path>", redacted)
    redacted = re.sub(r"/content/[^\s'\"]+", "<remote-path>", redacted)
    redacted = re.sub(r"(?i)(token|secret|password)=[^\s&]+", r"\1=<redacted>", redacted)
    return redacted[:1000]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def pick_device(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this experiment requires the requested Colab GPU")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def cuda_sync(torch: Any) -> None:
    torch.cuda.synchronize()


def cuda_memory(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def move_batch(batch: Any, device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def probs_from_logits(logits: Any) -> list[dict[str, float]]:
    probabilities = logits.float().softmax(dim=-1).detach().cpu().tolist()
    return [{label: round(float(row[index]), 8) for index, label in enumerate(LABELS)} for row in probabilities]


def make_environment(torch: Any, transformers: Any, hub: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "huggingface-hub", "accelerate", "safetensors", "tokenizers"):
        try:
            from importlib import metadata

            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "huggingface_hub_version": getattr(hub, "__version__", None),
        "package_versions": versions,
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "major": int(properties.major),
            "minor": int(properties.minor),
            "multi_processor_count": int(properties.multi_processor_count),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    remote_dataset = Path(DEFAULT_REMOTE_DATASET)
    dataset_default = str(remote_dataset if remote_dataset.exists() else Path(DEFAULT_LOCAL_DATASET))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=dataset_default)
    parser.add_argument("--output", default=DEFAULT_OUTPUT if remote_dataset.exists() else "results/openjev-nli-result.json")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--all-pairs-iterations", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args(argv)


def base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "running",
        "experiment": "openjev-nli-4b-l4",
        "created_at_utc": utc_now(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "subfolder": MODEL_SUBFOLDER,
            "license": MODEL_LICENSE,
            "base_model": BASE_MODEL_ID,
            "excluded": ["qwen3.5-4b-nli-v2", "qwen3.5-35b-a3b-nli"],
        },
        "measurement": {
            "dataset": str(args.dataset),
            "max_length": args.max_length,
            "warmup_iterations": args.warmup,
            "steady_state_iterations": args.iterations,
            "all_pairs_iterations": args.all_pairs_iterations,
            "timed_region": "forward_only_on_already_tokenized_tensors",
            "load_time_includes_model_download_if_cache_is_cold": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = base_result(args)
    fixture_path = Path(args.dataset)
    fixture = load_fixture(fixture_path)
    result["input"] = {
        "item_count": len(fixture["items"]),
        "pair_count": sum(len(item["hypotheses"]) for item in fixture["items"]),
        "labels": list(LABELS),
    }

    import torch
    import transformers
    from huggingface_hub import __version__ as hub_version
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class HubShim:
        __version__ = hub_version

    device = pick_device(torch)
    result["environment"] = make_environment(torch, transformers, HubShim)

    load_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        subfolder=MODEL_SUBFOLDER,
    )
    model_kwargs = {
        "revision": MODEL_REVISION,
        "subfolder": MODEL_SUBFOLDER,
        "device_map": {"": device},
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    }
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **model_kwargs)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    cuda_sync(torch)
    load_elapsed_ms = (time.perf_counter() - load_start) * 1000.0
    load_memory = cuda_memory(torch)
    result["load"] = {
        "elapsed_ms": round(load_elapsed_ms, 4),
        "memory": load_memory,
        "model_device": str(next(model.parameters()).device),
        "parameter_dtype": str(next(model.parameters()).dtype),
    }

    template = getattr(model.config, "nli_template", "Premise: {premise}\nHypothesis: {hypothesis}")
    pairs = build_pairs(fixture, template)
    tokenization_start = time.perf_counter()
    encoded = tokenizer(
        [pair["text"] for pair in pairs],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    tokenization_elapsed_ms = (time.perf_counter() - tokenization_start) * 1000.0
    encoded = move_batch(encoded, device)
    result["tokenization"] = {
        "elapsed_ms": round(tokenization_elapsed_ms, 4),
        "batch_size": len(pairs),
        "sequence_length": int(encoded["input_ids"].shape[-1]),
        "total_tokens": int(encoded["attention_mask"].sum().item()),
    }

    def forward(indices: Iterable[int]) -> Any:
        selected = list(indices)
        batch = {key: value[selected] for key, value in encoded.items()}
        with torch.inference_mode():
            return model(**batch).logits

    first_start = time.perf_counter()
    first_logits = forward([0])
    cuda_sync(torch)
    first_elapsed_ms = (time.perf_counter() - first_start) * 1000.0
    result["first_inference"] = {
        "elapsed_ms": round(first_elapsed_ms, 4),
        "memory": cuda_memory(torch),
        "pair_index": 0,
    }

    def benchmark(name: str, indices: Sequence[int], iterations: int) -> dict[str, Any]:
        for _ in range(args.warmup):
            forward(indices)
        cuda_sync(torch)
        torch.cuda.reset_peak_memory_stats()
        before = cuda_memory(torch)
        timings: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            forward(indices)
            cuda_sync(torch)
            timings.append((time.perf_counter() - start) * 1000.0)
        after = cuda_memory(torch)
        return {
            "name": name,
            "batch_size": len(indices),
            "iterations": iterations,
            "warmup_iterations": args.warmup,
            "timing_ms": summarize_timings(timings),
            "memory": {
                "allocated_before_bytes": before["allocated_bytes"],
                "reserved_before_bytes": before["reserved_bytes"],
                "peak_allocated_bytes": after["max_allocated_bytes"],
                "peak_reserved_bytes": after["max_reserved_bytes"],
                "additional_peak_allocated_bytes": max(
                    0, after["max_allocated_bytes"] - before["allocated_bytes"]
                ),
                "additional_peak_reserved_bytes": max(
                    0, after["max_reserved_bytes"] - before["reserved_bytes"]
                ),
            },
            "sequence_length": int(encoded["input_ids"][list(indices)].shape[-1]),
        }

    pair_count_per_item = len(fixture["items"][0]["hypotheses"])
    benchmarks = [
        benchmark("single_pair", [0], args.iterations),
        benchmark("one_question_three_hypotheses", list(range(pair_count_per_item)), args.iterations),
        benchmark("all_fixture_pairs", list(range(len(pairs))), args.all_pairs_iterations),
    ]
    result["benchmarks"] = benchmarks

    probabilities = probs_from_logits(forward(range(len(pairs))))
    pair_results: list[dict[str, Any]] = []
    for pair, probability in zip(pairs, probabilities):
        pair_results.append(
            {
                "question_id": pair["question_id"],
                "candidate_index": pair["candidate_index"],
                "premise": pair["premise"],
                "hypothesis": pair["hypothesis"],
                "probabilities": probability,
            }
        )
    result["predictions"] = {"pairs": pair_results}

    by_question: dict[str, list[dict[str, Any]]] = {}
    for pair_result in pair_results:
        by_question.setdefault(pair_result["question_id"], []).append(pair_result)
    decisions = []
    for item in fixture["items"]:
        candidates = by_question[item["id"]]
        scores = [candidate["probabilities"]["entailment"] for candidate in candidates]
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
        decisions.append(
            {
                "question_id": item["id"],
                "predicted_index": predicted_index,
                "gold_index": item["gold_index"],
                "correct": predicted_index == item["gold_index"],
                "entailment_scores": [round(float(score), 8) for score in scores],
            }
        )
    result["predictions"]["decisions"] = decisions
    result["predictions"]["accuracy"] = round(
        sum(1 for decision in decisions if decision["correct"]) / len(decisions),
        6,
    )
    result["status"] = "ok"
    result["finished_at_utc"] = utc_now()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    result = base_result(args)
    try:
        result = run(args)
    except Exception as error:  # record a sanitized, actionable blocker on the VM
        result["status"] = "error"
        result["finished_at_utc"] = utc_now()
        result["error"] = {
            "type": type(error).__name__,
            "message": redact_error(str(error)),
        }
        write_json(output_path, result)
        print(json.dumps({"status": "error", "error": result["error"]}, ensure_ascii=False))
        return 1
    write_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "gpu": result.get("environment", {}).get("gpu", {}).get("name"),
                "model_revision": MODEL_REVISION,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
