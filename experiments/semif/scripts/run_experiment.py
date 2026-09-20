"""Measure TheoLeeCJ/SemIf direct option-logit readout on one visible CUDA GPU.

The model implementation is imported from a pinned SemIf commit at runtime. This
runner records measurement provenance and refuses to label a non-L4 run as a
successful L4 result.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEMIF_REPOSITORY = "https://github.com/TheoLeeCJ/semif"
SEMIF_COMMIT = "ca3ba65f142967030ecb453346e94d6f476a69df"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_REPOSITORY = "https://huggingface.co/Qwen/Qwen3.5-4B"
CLI_VERSION = "google-colab-cli 0.6.0"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input fixture is empty")
    from semif_phase1.core import validate_row

    for row in rows:
        validate_row(row)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("input row ids must be unique")
    return rows


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def scrub_error(error: BaseException) -> str:
    """Keep failure evidence useful without persisting local paths or credentials."""
    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    message = re.sub(r"(?:[A-Za-z]:)?[/\\](?:[^\s'\"]+[/\\])*[^\s'\"]+", "<path>", message)
    return message[:1000] or error.__class__.__name__


def cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)


def memory_snapshot(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def device_metadata(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_count": int(torch.cuda.device_count()),
    }


def compact_output(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "option_ids",
        "probabilities",
        "option_logits",
        "input_tokens",
        "forward_seconds",
        "total_seconds",
        "prompt_sha256",
        "prompt_version",
        "readout",
        "probability_status",
    )
    return {key: result[key] for key in keys if key in result}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def base_payload(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": "jev-semif-l4-experiment-v1",
        "status": "starting",
        "started_at_utc": utc_now(),
        "runtime": {
            "label": "Google Colab GPU runtime",
            "colab_cli": CLI_VERSION,
            "session_name": "jev-semif",
            "expected_gpu": args.expected_gpu,
            "python_version": platform.python_version(),
            "platform": platform.platform(aliased=True),
            "executable": "python",
        },
        "source": {
            "implementation": "TheoLeeCJ/SemIf direct native next-token logit readout",
            "semif_repository": SEMIF_REPOSITORY,
            "semif_commit": SEMIF_COMMIT,
            "model_id": args.model,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": args.revision,
            "model_license": "Apache-2.0 (upstream model card)",
            "nli_comparison": "This is not AlexWortega/openjev; no NLI classifier head is used.",
        },
        "options": {
            "dtype": "bfloat16",
            "max_input_tokens": args.max_tokens,
            "warmup_iterations": args.warmup,
            "steady_repeats": args.repeats,
            "readout": "last-position full-vocabulary logits restricted to declared single-token A/B/C slots, then softmax",
            "probability_semantics": "conditional option score; uncalibrated as decision confidence",
        },
        "inputs": rows or [],
        "outputs": [],
        "metrics": {},
        "errors": [],
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    payload = base_payload(args)
    try:
        rows = load_rows(args.input)
        payload["inputs"] = rows

        import torch
        import transformers

        payload["runtime"].update(
            {
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this is not a GPU result")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}")
        device = device_metadata(torch)
        payload["runtime"]["gpu"] = device
        if args.expected_gpu and args.expected_gpu.lower() not in device["name"].lower():
            raise RuntimeError(f"expected GPU containing {args.expected_gpu!r}, got {device['name']!r}")

        from semif_phase1.core import load_causal_model
        from semif_phase1.direct import score

        load_started = time.perf_counter()
        model, tokenizer, loader_metadata = load_causal_model(args.model, args.revision)
        cuda_sync(torch)
        load_seconds = time.perf_counter() - load_started
        load_memory = memory_snapshot(torch)
        torch.cuda.reset_peak_memory_stats(0)

        first_started = time.perf_counter()
        first = score(model, tokenizer, rows[0], loader_metadata, args.max_tokens)
        cuda_sync(torch)
        first_wall_seconds = time.perf_counter() - first_started

        warmups: list[dict[str, float]] = []
        for index in range(args.warmup):
            result = score(model, tokenizer, rows[0], loader_metadata, args.max_tokens)
            warmups.append(
                {
                    "iteration": index + 1,
                    "wall_seconds": float(result["total_seconds"]),
                    "forward_seconds": float(result["forward_seconds"]),
                }
            )

        measured_rows: list[dict[str, Any]] = []
        rows_started = time.perf_counter()
        for row in rows:
            measured_rows.append(compact_output(score(model, tokenizer, row, loader_metadata, args.max_tokens)))
        cuda_sync(torch)
        rows_wall_seconds = time.perf_counter() - rows_started

        steady_batches: list[dict[str, float]] = []
        for repeat in range(args.repeats):
            batch_started = time.perf_counter()
            batch_rows = [score(model, tokenizer, row, loader_metadata, args.max_tokens) for row in rows]
            cuda_sync(torch)
            batch_seconds = time.perf_counter() - batch_started
            steady_batches.append(
                {
                    "repeat": repeat + 1,
                    "rows": len(batch_rows),
                    "wall_seconds": batch_seconds,
                    "per_decision_seconds": batch_seconds / len(batch_rows),
                }
            )

        inference_memory = memory_snapshot(torch)
        steady_seconds = [item["wall_seconds"] for item in steady_batches]
        steady_per_decision = [item["per_decision_seconds"] for item in steady_batches]
        payload.update(
            {
                "status": "success",
                "finished_at_utc": utc_now(),
                "outputs": measured_rows,
                "metrics": {
                    "model_load_seconds": load_seconds,
                    "first_inference_wall_seconds": first_wall_seconds,
                    "first_inference_reported_seconds": float(first["total_seconds"]),
                    "first_forward_seconds": float(first["forward_seconds"]),
                    "fixture_once_wall_seconds": rows_wall_seconds,
                    "warmup": warmups,
                    "steady_state": {
                        "repeats": args.repeats,
                        "rows_per_repeat": len(rows),
                        "batches": steady_batches,
                        "mean_batch_seconds": statistics.fmean(steady_seconds),
                        "median_batch_seconds": statistics.median(steady_seconds),
                        "mean_per_decision_seconds": statistics.fmean(steady_per_decision),
                        "median_per_decision_seconds": statistics.median(steady_per_decision),
                    },
                    "peak_vram": {
                        "after_load": load_memory,
                        "during_inference_after_reset": inference_memory,
                        "load_peak_allocated_bytes": load_memory["max_allocated_bytes"],
                        "load_peak_reserved_bytes": load_memory["max_reserved_bytes"],
                        "inference_peak_allocated_bytes": inference_memory["max_allocated_bytes"],
                        "inference_peak_reserved_bytes": inference_memory["max_reserved_bytes"],
                    },
                },
            }
        )
        return payload, 0
    except Exception as error:  # noqa: BLE001 - the failure artifact is part of the experiment contract
        payload.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "errors": [{"type": error.__class__.__name__, "message": scrub_error(error)}],
            }
        )
        return payload, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--expected-gpu", default="L4")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.max_tokens < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("max-tokens must be positive, warmup nonnegative, repeats positive")
    return args


def main() -> int:
    args = parse_args()
    payload, exit_code = run(args)
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output), "errors": payload["errors"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
