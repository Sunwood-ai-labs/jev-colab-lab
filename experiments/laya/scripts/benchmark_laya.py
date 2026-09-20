"""Run a reproducible Laya inference benchmark and write sanitized JSON.

The benchmark intentionally uses the public base ``convaiinnovations/laya``
checkpoint. It does not train, save model weights, or send private inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional


DEFAULT_MODEL_ID = "convaiinnovations/laya"
DEFAULT_MODEL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
DEFAULT_SOURCE_COMMIT = "d113dca2512fb3eaca313534bc54c7162d87c1d4"
MODEL_FILES = [
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
    "rl_agent_config.json",
]


def build_cases() -> Dict[str, Dict[str, Any]]:
    """Return synthetic, non-sensitive cases used for both benchmark paths."""

    state = {
        "ticket_id": "synthetic-001",
        "subject": "Duplicate charge on invoice 4411",
        "body": (
            "We were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan."
        ),
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "not urgent",
                "soon",
                "critical deadline or blocking issue",
            ],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the user explicitly request a refund?",
        },
    }
    return {
        "single": {"state": state, "questions": {"department": questions["department"]}},
        "batch": {"state": state, "questions": questions},
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    """Summarize positive millisecond measurements without numpy."""

    xs = [float(v) for v in values]
    if not xs:
        return {"count": 0}
    ordered = sorted(xs)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * q
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "count": len(xs),
        "mean_ms": statistics.fmean(xs),
        "min_ms": ordered[0],
        "median_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def json_safe(value: Any) -> Any:
    """Convert common runtime values to JSON without exposing arbitrary objects."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def error_record(exc: BaseException) -> Dict[str, str]:
    """Keep failure evidence concise and sanitized."""

    message = str(exc).replace("\r", " ").replace("\n", " ")
    return {"type": type(exc).__name__, "message": message[:1000]}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch_module.cuda.synchronize(device)


def _memory_snapshot(torch_module: Any, device: Any) -> Dict[str, float]:
    if getattr(device, "type", str(device)) != "cuda":
        return {}
    return {
        "allocated_mb": torch_module.cuda.memory_allocated(device) / 2**20,
        "reserved_mb": torch_module.cuda.memory_reserved(device) / 2**20,
        "max_allocated_mb": torch_module.cuda.max_memory_allocated(device) / 2**20,
        "max_reserved_mb": torch_module.cuda.max_memory_reserved(device) / 2**20,
    }


def _reset_peak(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch_module.cuda.reset_peak_memory_stats(device)


def _run_scenario(
    agent: Any,
    torch_module: Any,
    device: Any,
    case: Mapping[str, Any],
    warmup_count: int,
    repeat_count: int,
) -> Dict[str, Any]:
    state = case["state"]
    questions = case["questions"]

    def call() -> tuple[Dict[str, Any], float]:
        _synchronize(torch_module, device)
        started = time.perf_counter()
        result = agent.predict(state, questions)
        _synchronize(torch_module, device)
        return result, (time.perf_counter() - started) * 1000.0

    _reset_peak(torch_module, device)
    first_result, first_ms = call()
    baseline = _memory_snapshot(torch_module, device)

    warmup_ms: List[float] = []
    for _ in range(warmup_count):
        _, elapsed = call()
        warmup_ms.append(elapsed)

    _reset_peak(torch_module, device)
    steady_results: List[Dict[str, Any]] = []
    steady_ms: List[float] = []
    for _ in range(repeat_count):
        result, elapsed = call()
        steady_results.append(result)
        steady_ms.append(elapsed)

    peak = _memory_snapshot(torch_module, device)
    first_answers = json_safe(first_result.get("answers", {}))
    stable = all(json_safe(item.get("answers", {})) == first_answers for item in steady_results)
    return {
        "question_count": len(questions),
        "state": json_safe(state),
        "questions": json_safe(questions),
        "output": json_safe(first_result),
        "deterministic_answers_across_repeats": stable,
        "warmup_repeats": warmup_count,
        "steady_repeats": repeat_count,
        "latency_ms": {
            "first_inference": first_ms,
            "warmup": summarize(warmup_ms),
            "steady": summarize(steady_ms),
        },
        "vram": {
            "baseline_after_first_mb": baseline,
            "steady_peak_mb": peak,
            "steady_peak_delta_allocated_mb": max(
                0.0,
                peak.get("max_allocated_mb", 0.0) - baseline.get("allocated_mb", 0.0),
            ),
        },
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    # These imports are intentionally delayed so helper tests do not need a CUDA stack.
    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    import laya

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    requested_device = None if args.device == "auto" else args.device
    start_download = time.perf_counter()
    model_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        allow_patterns=MODEL_FILES,
    )
    download_seconds = time.perf_counter() - start_download

    cuda_before = torch.cuda.is_available()
    if cuda_before:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start_load = time.perf_counter()
    agent = laya.load(str(model_path), device=requested_device)
    device = agent.device
    _synchronize(torch, device)
    load_seconds = time.perf_counter() - start_load
    load_memory = _memory_snapshot(torch, device)

    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(f"Laya loaded on {device}, not CUDA; refusing to report a GPU result")

    parameter_dtype = None
    try:
        parameter_dtype = str(next(agent.model.parameters()).dtype)
    except (StopIteration, AttributeError):
        pass

    gpu: Dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        gpu.update(
            {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "current_index": torch.cuda.current_device(),
                "total_memory_mb": torch.cuda.get_device_properties(device).total_memory / 2**20,
            }
        )

    cases = build_cases()
    scenario_results = {}
    for name, case in cases.items():
        scenario_results[name] = _run_scenario(
            agent,
            torch,
            device,
            case,
            warmup_count=args.warmup,
            repeat_count=args.repeats,
        )

    return {
        "status": "ok",
        "created_at_utc": utc_now(),
        "experiment": "jev-colab-lab/laya",
        "execution": {
            "host_platform": platform.platform(),
            "python": sys.version.split()[0],
            "requested_device": args.device,
            "actual_device": str(device),
            "seed": args.seed,
            "warmup_repeats": args.warmup,
            "steady_repeats": args.repeats,
        },
        "source": {
            "upstream_repository": "https://github.com/NandhaKishorM/laya",
            "upstream_commit": args.source_commit,
            "model_id": args.model_id,
            "model_revision": args.revision,
            "model_files_fetched": MODEL_FILES,
            "license": "Apache-2.0",
        },
        "model": {
            "package": "laya",
            "config": json_safe(getattr(agent, "cfg", {})),
            "parameter_dtype": parameter_dtype,
            "autocast_dtype": str(getattr(agent, "dtype", None)),
            "model_class": type(getattr(agent, "model", None)).__name__,
        },
        "environment": {
            "packages": package_versions(
                ["laya", "torch", "transformers", "huggingface-hub", "safetensors", "numpy"]
            ),
            "torch_cuda_version": torch.version.cuda,
            "gpu": gpu,
        },
        "timing_seconds": {
            "snapshot_download": download_seconds,
            "agent_load": load_seconds,
        },
        "vram": {"load_peak_mb": load_memory},
        "scenarios": scenario_results,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="sanitized JSON output path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_benchmark(args)
    except Exception as exc:  # write failure evidence before returning non-zero
        result = {
            "status": "error",
            "created_at_utc": utc_now(),
            "experiment": "jev-colab-lab/laya",
            "execution": {
                "host_platform": platform.platform(),
                "python": sys.version.split()[0],
                "requested_device": args.device,
                "warmup_repeats": args.warmup,
                "steady_repeats": args.repeats,
            },
            "source": {
                "upstream_repository": "https://github.com/NandhaKishorM/laya",
                "upstream_commit": args.source_commit,
                "model_id": args.model_id,
                "model_revision": args.revision,
                "license": "Apache-2.0",
            },
            "error": error_record(exc),
        }
        exit_code = 1
    else:
        exit_code = 0
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
