"""Colab entrypoint for the uploaded fixed-game prompt audit."""

from __future__ import annotations

import runpy
import sys
import zipfile
from pathlib import Path

if __name__ == "__main__":
    archive = Path("/content/jevdash-eb2f926-src.zip")
    target = Path("/content/jevdash")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(target)
    sys.argv = [
        "prompt_audit.py",
        "--game-src",
        "/content/jevdash/src",
        "--output",
        "/content/laya-jevdash-prompt-audit.json",
    ]
    sys.modules.pop("laya_agent", None)
    sys.modules.pop("prompt_audit", None)
    runpy.run_path("/content/prompt_audit.py", run_name="__main__")
