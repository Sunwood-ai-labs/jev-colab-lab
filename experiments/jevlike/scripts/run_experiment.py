"""Run a small, measured Jevlike experiment on CPU or a Colab GPU.

The upstream implementation is fetched at a pinned commit and imported as-is.
This wrapper only supplies a reproducible synthetic-data run and measurements;
it does not reimplement the model architecture.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_URL = "https://github.com/vinnylarouge/jevlike.git"
UPSTREAM_COMMIT = "94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452"
UPSTREAM_LICENSE = "MIT"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run a command without echoing credentials or full subprocess output."""

    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        rendered = " ".join(command)
        raise RuntimeError(f"command failed with exit {result.returncode}: {rendered}")


def ensure_upstream(root: Path) -> tuple[Path, float]:
    """Fetch the official repository and verify the exact source revision."""

    started = time.perf_counter()
    root = root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and not (root / ".git").is_dir():
        raise RuntimeError(f"upstream path exists but is not a git checkout: {root}")
    if not root.exists():
        run_command(["git", "init", "--quiet", str(root)])
        run_command(["git", "-C", str(root), "remote", "add", "origin", UPSTREAM_URL])
    run_command([
        "git", "-C", str(root), "fetch", "--quiet", "--depth", "1",
        "origin", UPSTREAM_COMMIT,
    ])
    run_command([
        "git", "-C", str(root), "checkout", "--quiet", "--detach", UPSTREAM_COMMIT,
    ])
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream revision mismatch: expected {UPSTREAM_COMMIT}, got {actual}")
    return root, time.perf_counter() - started


def install_upstream(root: Path) -> float:
    """Install the pinned package through uv, using the active interpreter."""

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv was not found; install/use the official uv runtime")
    started = time.perf_counter()
    if sys.prefix != sys.base_prefix:
        command = [uv, "pip", "install", "--python", sys.executable, "--quiet", "-e", str(root)]
    else:
        command = [uv, "pip", "install", "--system", "--quiet", "-e", str(root)]
    run_command(command)
    importlib.invalidate_caches()
    return time.perf_counter() - started


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def vram_stats(torch: Any, device: Any) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def percentiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
    }


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but it is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info(torch: Any, device: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "jevlike": package_version("jevlike"),
        "uv": shutil.which("uv") is not None,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_total_memory_bytes": int(properties.total_memory),
        })
    else:
        info.update({"gpu_name": None, "gpu_compute_capability": None, "gpu_total_memory_bytes": None})
    return info


def train_and_measure(args: argparse.Namespace, upstream_root: Path) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from jevlike.data import ChoiceExample, JsonlDataset, write_synthetic
    from jevlike.model import load_checkpoint, make_system
    from jevlike.eval import metrics

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(torch, args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    checkpoint = output_dir / "tiny_scorer.pt"

    data_started = time.perf_counter()
    write_synthetic(
        data_dir,
        {"train": args.train_size, "validation": args.validation_size, "test": args.test_size},
        args.seed,
    )
    data_seconds = time.perf_counter() - data_started

    config = {
        "encoder": "tiny",
        "hf_model": "Qwen/Qwen2.5-0.5B",
        "width": args.width,
        "rank": args.rank,
        "context_tokens": args.context_tokens,
        "option_tokens": args.option_tokens,
    }
    model, collator = make_system(config, device)
    train_loader = DataLoader(
        JsonlDataset(data_dir / "train.jsonl"),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        JsonlDataset(data_dir / "validation.jsonl"),
        batch_size=args.batch_size,
        collate_fn=collator,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    epoch_logs: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        synchronize(torch, device)
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            loss = F.cross_entropy(model(batch), batch["labels"])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            total += float(loss.detach()) * batch["labels"].numel()
            count += batch["labels"].numel()
        model.eval()
        validation_total, validation_count = 0.0, 0
        with torch.no_grad():
            for host_batch in validation_loader:
                batch = move_batch(host_batch, device)
                validation_loss = F.cross_entropy(model(batch), batch["labels"], reduction="sum")
                validation_total += float(validation_loss)
                validation_count += batch["labels"].numel()
        validation_nll = validation_total / validation_count
        train_nll = total / count
        if validation_nll < best_loss:
            best_loss = validation_nll
            best_state = {
                name: parameter.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        epoch_logs.append({
            "epoch": epoch + 1,
            "train_nll": train_nll,
            "validation_nll": validation_nll,
            "device": str(device),
        })
    synchronize(torch, device)
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("training produced no checkpoint state")
    torch.save({"config": config, "state_dict": best_state}, checkpoint)
    training_vram = vram_stats(torch, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    loaded_model, loaded_collator, loaded_config = load_checkpoint(checkpoint, device)
    synchronize(torch, device)
    load_seconds = time.perf_counter() - load_started

    test_loader = DataLoader(
        JsonlDataset(data_dir / "test.jsonl"),
        batch_size=args.batch_size,
        collate_fn=loaded_collator,
    )
    loaded_model.eval()
    test_metrics = metrics(loaded_model, test_loader, device)
    shuffled_metrics = metrics(loaded_model, test_loader, device, shuffle_context=True)

    first_batch = move_batch(next(iter(test_loader)), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(torch, device)
    first_started = time.perf_counter()
    with torch.no_grad():
        first_logits = loaded_model(first_batch)
    synchronize(torch, device)
    first_inference_seconds = time.perf_counter() - first_started

    warmup_times: list[float] = []
    for _ in range(args.warmup):
        synchronize(torch, device)
        started = time.perf_counter()
        with torch.no_grad():
            loaded_model(first_batch)
        synchronize(torch, device)
        warmup_times.append((time.perf_counter() - started) * 1000.0)

    steady_times: list[float] = []
    for _ in range(args.repetitions):
        synchronize(torch, device)
        started = time.perf_counter()
        with torch.no_grad():
            loaded_model(first_batch)
        synchronize(torch, device)
        steady_times.append((time.perf_counter() - started) * 1000.0)
    inference_vram = vram_stats(torch, device)

    prediction_context = "Choose the exact badge amber badger. Badge: amber badger."
    prediction_options = ["azure crane", "amber badger", "gold heron"]
    prediction_batch = move_batch(
        loaded_collator([ChoiceExample(prediction_context, tuple(prediction_options), 1)]),
        device,
    )
    synchronize(torch, device)
    with torch.no_grad():
        probabilities = loaded_model(prediction_batch).softmax(-1)[0, :len(prediction_options)]
    probabilities = [float(value) for value in probabilities.detach().cpu()]

    return {
        "status": "success",
        "generated_at_utc": utc_now(),
        "source": {
            "repository": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "verified_checkout": "detached checkout verified at runtime",
        },
        "environment": environment_info(torch, device),
        "experiment": {
            "encoder": "tiny",
            "precision": "float32",
            "seed": args.seed,
            "train_examples": args.train_size,
            "validation_examples": args.validation_size,
            "test_examples": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "width": args.width,
            "rank": args.rank,
            "context_tokens": args.context_tokens,
            "option_tokens": args.option_tokens,
            "warmup_repetitions": args.warmup,
            "steady_repetitions": args.repetitions,
        },
        "timings": {
            "data_generation_seconds": data_seconds,
            "training_seconds": training_seconds,
            "checkpoint_load_seconds": load_seconds,
            "first_batch_inference_seconds": first_inference_seconds,
            "warmup_ms": percentiles(warmup_times),
            "steady_state_ms": percentiles(steady_times),
        },
        "peak_vram": {
            "training": training_vram,
            "inference": inference_vram,
        },
        "metrics": {
            "test": test_metrics,
            "shuffled_context_control": shuffled_metrics,
            "first_batch_shape": {
                "batch": int(first_logits.shape[0]),
                "options_with_padding": int(first_logits.shape[1]),
            },
        },
        "prediction": {
            "context": prediction_context,
            "options": [
                {"option": option, "probability": probability}
                for option, probability in zip(prediction_options, probabilities)
            ],
            "probability_sum": sum(probabilities),
        },
        "upstream_training_logs": epoch_logs,
        "artifacts": {
            "checkpoint": checkpoint.name,
            "data_directory": data_dir.name,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", default="results/run")
    parser.add_argument("--upstream-dir", default="")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=192)
    parser.add_argument("--option-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args, unknown = parser.parse_known_args()
    # colab exec evaluates a .py file inside the active Jupyter kernel. The
    # kernel launcher adds its own ``-f <kernel.json>`` pair to sys.argv; it is
    # not an experiment option and must not be forwarded to this parser.
    if unknown:
        is_kernel_arg = (
            len(unknown) == 2
            and unknown[0] == "-f"
            and unknown[1].endswith(".json")
        )
        if not is_kernel_arg:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def main() -> None:
    args = parse_args()
    upstream_dir = (
        Path(args.upstream_dir)
        if args.upstream_dir
        else Path(tempfile.gettempdir()) / "jevlike-upstream-pinned"
    )
    upstream_root, clone_seconds = ensure_upstream(upstream_dir)
    # A package installed into a running interpreter does not execute its new
    # .pth file until the next interpreter start. Import the verified checkout
    # directly so both local uv environments and Colab system Python use the
    # exact pinned source in this process.
    sys.path.insert(0, str(upstream_root))
    install_seconds = 0.0
    if not args.skip_install:
        install_seconds = install_upstream(upstream_root)
    result = train_and_measure(args, upstream_root)
    result["timings"]["upstream_checkout_seconds"] = clone_seconds
    result["timings"]["uv_install_seconds"] = install_seconds
    result["runtime_options"] = {
        "device_requested": args.device,
        "skip_install": args.skip_install,
    }
    result_path = Path(args.output_dir).expanduser().resolve() / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(f"RESULT_PATH={result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = {
            "status": "error",
            "generated_at_utc": utc_now(),
            "source": {"repository": UPSTREAM_URL, "commit": UPSTREAM_COMMIT},
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        print("RESULT_JSON=" + json.dumps(failure, ensure_ascii=False, sort_keys=True))
        raise
