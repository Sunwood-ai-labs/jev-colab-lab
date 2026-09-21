"""Thin entry point for the isolated Google Colab JevDash session."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REMOTE_RUNNER = Path("/content/semif-jevdash-runner.py")
sys.argv = [
    "jevdash_play.py",
    "--game-root",
    "/content/jevdash",
    "--game-commit-file",
    "/content/jevdash-commit.txt",
    "--output-video",
    "/content/semif-jevdash.mp4",
    "--output-json",
    "/content/semif-jevdash-episode.json",
    "--controller-prompt-version",
    "jev-dash-rules-v2",
    "--level",
    "1",
    "--seed",
    "42",
    "--fps",
    "60",
    "--frames-per-decision",
    "8",
    "--max-frames",
    "1800",
    "--static-terminal-frames",
    "120",
]
runpy.run_path(str(REMOTE_RUNNER), run_name="__main__")
