"""Colab entrypoint for the explicitly candidate-limited Laya replay."""

from __future__ import annotations

import runpy
import sys
import zipfile
from pathlib import Path


if __name__ == "__main__":
    target = Path("/content/jevdash")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("/content/jevdash-eb2f926-src.zip") as zipped:
        zipped.extractall(target)
    sys.path.insert(0, "/content/jevdash/src")
    sys.path.insert(0, "/content")
    for module_name in ("runner", "laya_agent", "control"):
        sys.modules.pop(module_name, None)
    sys.argv = [
        "runner.py",
        "--video",
        "/content/laya-rules-v2-forward-binary-model-only.mp4",
        "--json",
        "/content/laya-rules-v2-forward-binary-model-only.json",
        "--prompt-profile",
        "platformer_rules_v2",
        "--state-encoding",
        "semantic_v1",
        "--control-mode",
        "model_only",
        "--candidate-order",
        "right_run_jump,right_run",
    ]
    runpy.run_path("/content/runner.py", run_name="__main__")
