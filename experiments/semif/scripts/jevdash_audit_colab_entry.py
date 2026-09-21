"""Thin entry point for the isolated JevDash SemIf audit session."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REMOTE_RUNNER = Path("/content/semif-jevdash-audit.py")
sys.argv = [
    "jevdash_audit.py",
    "--game-root",
    "/content/jevdash",
    "--game-commit-file",
    "/content/jevdash-commit.txt",
    "--output",
    "/content/semif-jevdash-audit.json",
    "--max-tokens",
    "4096",
]
runpy.run_path(str(REMOTE_RUNNER), run_name="__main__")
