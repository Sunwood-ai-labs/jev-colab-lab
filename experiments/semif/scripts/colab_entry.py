"""Entry point sent by `colab exec` after the runner and fixture are uploaded."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REMOTE_RUNNER = Path("/content/semif-run-experiment.py")
sys.argv = [
    "run_experiment.py",
    "--input",
    "/content/semif-questions.jsonl",
    "--output",
    "/content/semif-l4-result.json",
    "--warmup",
    "2",
    "--repeats",
    "5",
    "--expected-gpu",
    "L4",
]
runpy.run_path(str(REMOTE_RUNNER), run_name="__main__")
