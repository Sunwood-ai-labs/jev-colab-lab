"""Prepare a pinned JevDash checkout and run the Kev contract audit on a T4.

The detailed state/encoding/probability rows are produced by
``audit_jevdash_contract.py``.  This wrapper records only sanitized source,
hardware, timing, and result metadata around that audit; model weights and
Colab session metadata are never written to the repository artifact.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


GAME_REPO = "https://github.com/Sunwood-ai-labs/jevdash.git"
GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
GAME_DIR = Path("/content/kev-jevdash-contract/jevdash")
ROOT = Path("/content/kev-jevdash-contract")
RESULT_PATH = Path(os.environ.get("KEV_CONTRACT_RESULT_PATH", "/content/kev-jevdash-contract-audit.json"))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_command(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def clone_game() -> dict[str, str]:
    GAME_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (GAME_DIR / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", GAME_REPO, str(GAME_DIR)], check=True)
    observed = run_command(["git", "rev-parse", "HEAD"], cwd=GAME_DIR)
    if observed != GAME_COMMIT:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", GAME_COMMIT], cwd=GAME_DIR, check=True)
        subprocess.run(["git", "switch", "--detach", GAME_COMMIT], cwd=GAME_DIR, check=True)
        observed = run_command(["git", "rev-parse", "HEAD"], cwd=GAME_DIR)
    if observed != GAME_COMMIT:
        raise RuntimeError("pinned JevDash commit was not obtained")
    if run_command(["git", "status", "--porcelain"], cwd=GAME_DIR):
        raise RuntimeError("pinned JevDash checkout is dirty")
    return {"repository": GAME_REPO, "commit": observed}


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "accelerate", "huggingface_hub", "numpy", "pydantic", "pygame")
    versions: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "unavailable"
    return versions


def main() -> int:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": "jev-colab-lab/kev-jevdash-contract-audit-t4-v1",
        "started_at_utc": utc_now(),
        "status": "started",
        "game_commit": GAME_COMMIT,
        "requested_gpu": "T4",
        "hardware": {},
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": {}},
        "sources": {},
        "timings": {},
        "audit": None,
        "error": None,
    }
    phase = "hardware_check"
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        gpu_name = torch.cuda.get_device_name(0)
        if "T4" not in gpu_name.upper():
            raise RuntimeError("requested T4 but a different GPU was reported")
        result["hardware"] = {
            "name": gpu_name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 3),
            "torch_cuda": torch.version.cuda,
        }
        result["environment"]["packages"] = package_versions()

        phase = "prepare_sources"
        game_source = clone_game()
        sys.path.insert(0, "/content")
        import t4_inference

        kev_source = t4_inference.clone_upstream()
        model_sources = t4_inference.download_models()
        sys.path.insert(0, str(t4_inference.UPSTREAM_DIR))
        result["sources"] = {
            "game": game_source,
            "kev": {key: kev_source[key] for key in ("repository", "ref", "commit")},
            "adapter": {
                "repository": model_sources["adapter"]["repo"],
                "revision": model_sources["adapter"]["revision"],
            },
            "base": {
                "repository": model_sources["base"]["repo"],
                "revision": model_sources["base"]["revision"],
            },
        }

        phase = "run_contract_audit"
        from transformers import AutoTokenizer

        import audit_jevdash_contract as contract

        fixtures = contract.collect_fixtures(GAME_DIR)
        tokenizer = AutoTokenizer.from_pretrained(t4_inference.BASE_DIR)
        torch.cuda.reset_peak_memory_stats()
        model_load_started = time.perf_counter()
        probe = contract.KevModelProbe(t4_inference.UPSTREAM_DIR, t4_inference.BASE_DIR, t4_inference.ADAPTER_DIR)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_load_started
        audit_started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for fixture_name, fixture in fixtures.items():
            full = fixture["observation"]
            states = {
                "full_json": full,
                "compact_json": contract.compact_state(full),
                "compact_policy_json": contract.compact_policy_state(full),
            }
            for representation, state in states.items():
                for question_variant in contract.QUESTION_VARIANTS:
                    for order_name, order in contract.ORDER_VARIANTS.items():
                        audit = contract.encode_audit(tokenizer, state, question_variant, order)
                        request_material = contract.make_request(state, question_variant, order)
                        audit["record_request"] = request_material["request"]
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
        torch.cuda.synchronize()
        result["timings"] = {
            "model_load_seconds": model_load_seconds,
            "audit_seconds": time.perf_counter() - audit_started,
            "total_seconds": time.perf_counter() - started,
        }
        result["hardware"]["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
        result["hardware"]["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 3)
        result["audit"] = {
            "schema": "jev-colab-lab/kev-jevdash-contract-audit-v1",
            "seed": 42,
            "decision_interval_frames": contract.DECISION_INTERVAL,
            "question_variants": contract.QUESTION_VARIANTS,
            "order_variants": {name: list(order) for name, order in contract.ORDER_VARIANTS.items()},
            "model_audit": True,
            "fixtures": fixtures,
            "rows": rows,
        }
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = {
            "phase": phase,
            "type": type(exc).__name__,
            "message": "T4 contract audit failed; detailed exception text is omitted from the sanitized result.",
        }
        result["timings"]["total_seconds"] = time.perf_counter() - started

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "path": str(RESULT_PATH), "phase": phase}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
