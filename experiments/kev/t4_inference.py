"""Run the pinned kev-0.5b checkpoint on a Colab T4.

The upstream v0.1.0 code is cloned without modification.  Its model class is
given ``cuda`` explicitly because the release's automatic device selection
only chooses MPS or CPU.  The model, tokenizer, and adapter revisions are
downloaded by immutable Hub commit SHA.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


UPSTREAM_REPO = "https://github.com/jaredpalmer/kev.git"
UPSTREAM_REF = "v0.1.0"
UPSTREAM_COMMIT = "ac67bf4e52d7bdc8420d8024c396df5585915d8c"
ADAPTER_REPO = "jaredpalmer/kev-0.5b"
ADAPTER_REF = "edf1dc6d7f8d983c0adfd251e80a686e5539fc61"
BASE_REPO = "Qwen/Qwen2.5-0.5B"
BASE_REF = "060db6499f32faf8b98477b0a26969ef7d8b9987"
REQUESTED_GPU = "T4"
WARMUP_ITERATIONS = 2
STEADY_ITERATIONS = 10

ROOT = Path(os.environ.get("KEV_EXPERIMENT_ROOT", "/content/kev-experiment"))
UPSTREAM_DIR = ROOT / "upstream"
MODEL_ROOT = ROOT / "models"
ADAPTER_DIR = MODEL_ROOT / "kev-0.5b"
BASE_DIR = MODEL_ROOT / "qwen2.5-0.5b"
RESULT_PATH = Path(os.environ.get("KEV_RESULT_PATH", "/content/kev-t4-result.json"))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_command(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_manifest(root: Path, names: list[str]) -> list[dict[str, Any]]:
    out = []
    for name in names:
        path = root / name
        if path.is_file():
            out.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return out


def clone_upstream() -> dict[str, str]:
    UPSTREAM_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (UPSTREAM_DIR / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", UPSTREAM_REF, UPSTREAM_REPO, str(UPSTREAM_DIR)],
            check=True,
        )
    observed = run_command(["git", "rev-parse", "HEAD"], cwd=UPSTREAM_DIR)
    if observed != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: expected {UPSTREAM_COMMIT}, got {observed}")
    return {"repository": UPSTREAM_REPO, "ref": UPSTREAM_REF, "commit": observed}


def download_models() -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    adapter_info = HfApi().model_info(ADAPTER_REPO, revision=ADAPTER_REF)
    if adapter_info.sha != ADAPTER_REF:
        raise RuntimeError(f"adapter revision mismatch: expected {ADAPTER_REF}, got {adapter_info.sha}")
    adapter_path = snapshot_download(
        ADAPTER_REPO,
        revision=ADAPTER_REF,
        local_dir=str(ADAPTER_DIR),
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
            "added_tokens.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )
    base_info = HfApi().model_info(BASE_REPO, revision=BASE_REF)
    if base_info.sha != BASE_REF:
        raise RuntimeError(f"base revision mismatch: expected {BASE_REF}, got {base_info.sha}")
    base_path = snapshot_download(BASE_REPO, revision=BASE_REF, local_dir=str(BASE_DIR))
    return {
        "adapter": {"repo": ADAPTER_REPO, "revision": adapter_info.sha, "path": adapter_path},
        "base": {"repo": BASE_REPO, "revision": base_info.sha, "path": base_path},
        "adapter_files": files_manifest(
            ADAPTER_DIR,
            ["adapter_config.json", "adapter_model.safetensors", "head.pt", "tokenizer.json"],
        ),
        "base_files": files_manifest(
            BASE_DIR,
            ["config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json"],
        ),
    }


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_memory() -> dict[str, float]:
    import torch

    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 3),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 3),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 3),
    }


def timed_forward(model: Any, encoded: dict[str, Any]) -> tuple[list[Any], float]:
    sync_cuda()
    started = time.perf_counter()
    with __import__("torch").inference_mode():
        probabilities = model.probs(encoded)
    sync_cuda()
    return probabilities, time.perf_counter() - started


def seconds_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_s": sum(values) / len(values),
        "min_s": ordered[0],
        "median_s": ordered[len(ordered) // 2],
        "p95_s": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max_s": ordered[-1],
    }


def build_request() -> dict[str, Any]:
    return {
        "state": {
            "ticket": {
                "channel": "email",
                "body": "My shoes arrived two weeks late and in the wrong size. I also see two charges on my card.",
            }
        },
        "model": "kev-0.5b",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "returns": "Exchanges, refunds, wrong or damaged items",
                    "shipping": "Delivery status, delays, or lost packages",
                    "billing": "Charges, invoices, or payment problems",
                },
            },
            "escalate": {
                "type": "noul",
                "instructions": "Does this require urgent human attention?",
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    }


def package_versions() -> dict[str, str]:
    names = ["torch", "transformers", "peft", "accelerate", "huggingface_hub", "numpy"]
    versions = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            versions[name] = f"unavailable: {type(exc).__name__}"
    return versions


def execute() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this run cannot be reported as a T4 result")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"requested T4 but Colab reported {gpu_name!r}")

    source = clone_upstream()
    model_sources = download_models()
    sys.path.insert(0, str(UPSTREAM_DIR))
    from kev.api import SystemOneRequest, to_answers, to_record
    from kev.model import DecisionModel, encode, load_tokenizer
    from peft import PeftModel

    request = build_request()
    validated_request = SystemOneRequest.model_validate(request)
    record, meta = to_record(validated_request)
    encode_started = time.perf_counter()
    tokenizer = load_tokenizer(str(BASE_DIR))
    encoded = encode(tokenizer, record)
    encode_seconds = time.perf_counter() - encode_started

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = DecisionModel(str(BASE_DIR), tokenizer, "cuda", lora=None)
    model.lm = PeftModel.from_pretrained(model.lm, str(ADAPTER_DIR)).to("cuda")
    head_meta = torch.load(ADAPTER_DIR / "head.pt", map_location="cpu", weights_only=False)
    model.head.load_state_dict(head_meta["head"])
    model.eval()
    sync_cuda()
    load_seconds = time.perf_counter() - load_started
    load_memory = cuda_memory()
    torch.cuda.reset_peak_memory_stats()

    first_probabilities, first_seconds = timed_forward(model, encoded)
    warmup_seconds = []
    for _ in range(WARMUP_ITERATIONS):
        _, elapsed = timed_forward(model, encoded)
        warmup_seconds.append(elapsed)
    steady_seconds = []
    for _ in range(STEADY_ITERATIONS):
        _, elapsed = timed_forward(model, encoded)
        steady_seconds.append(elapsed)
    inference_memory = cuda_memory()

    packed = [p.tolist() for p in first_probabilities]
    separate = []
    separate_seconds = []
    for question_id in request["questions"]:
        one_request = dict(request)
        one_request["questions"] = {question_id: request["questions"][question_id]}
        one_record, _ = to_record(SystemOneRequest.model_validate(one_request))
        one_encoded = encode(tokenizer, one_record)
        one_probabilities, elapsed = timed_forward(model, one_encoded)
        separate.append(one_probabilities[0].tolist())
        separate_seconds.append(elapsed)
    max_abs_difference = max(
        abs(packed[q][i] - separate[q][i])
        for q in range(len(packed))
        for i in range(len(packed[q]))
    )

    return {
        "status": "success",
        "completed_at_utc": utc_now(),
        "requested_gpu": REQUESTED_GPU,
        "hardware": {
            "name": gpu_name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 3),
            "torch_cuda": torch.version.cuda,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "sources": {"upstream": source, "models": model_sources},
        "input": {
            "request": request,
            "rendered_state": record["state"],
            "question_count": len(record["questions"]),
            "encoded_tokens": len(encoded["ids"]),
            "state_tokens": encoded["seg"].count(0),
            "branch_token_counts": [len(options) for options in encoded["opt_idx"]],
        },
        "precision": "float32",
        "iterations": {
            "warmup": WARMUP_ITERATIONS,
            "steady": STEADY_ITERATIONS,
            "first_inference_s": first_seconds,
            "warmup": seconds_stats(warmup_seconds),
            "steady": seconds_stats(steady_seconds),
            "encode_s": encode_seconds,
            "load_s": load_seconds,
            "separate_question_s": seconds_stats(separate_seconds),
        },
        "memory": {"after_load": load_memory, "inference": inference_memory},
        "outputs": {
            "answers": to_answers(packed, meta),
            "probabilities": packed,
            "packed_vs_separate_max_abs_probability_difference": max_abs_difference,
            "separate_probabilities": separate,
        },
        "errors": [],
    }


def main() -> int:
    result: dict[str, Any] = {
        "schema": "jev-colab-lab/kev-t4-result-v1",
        "started_at_utc": utc_now(),
        "status": "started",
        "errors": [],
    }
    try:
        result.update(execute())
    except Exception as exc:  # persist a diagnostic even when setup/inference fails
        result.update(
            {
                "status": "error",
                "completed_at_utc": utc_now(),
                "errors": [
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=20),
                    }
                ],
            }
        )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
