"""Install the experiment's non-Torch dependencies with uv on a Colab VM."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys


PACKAGE_SPECS = (
    "accelerate>=1.7,<2",
    "huggingface-hub>=1.5,<2",
    "numpy>=1.26,<3",
    "pygame>=2.6,<3",
    "pydantic>=2.7,<3",
    "safetensors>=0.5,<1",
    "transformers==5.15.0",
)


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print(json.dumps({"status": "blocked", "error": "uv is not installed on the Colab VM"}))
        return 2

    command = [uv, "pip", "install", "--system", *PACKAGE_SPECS]
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode != 0:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error": "uv dependency installation failed",
                    "returncode": completed.returncode,
                    "python": sys.version.split()[0],
                }
            )
        )
        return completed.returncode

    package_names = (
        "accelerate",
        "huggingface-hub",
        "numpy",
        "pygame",
        "pydantic",
        "safetensors",
        "transformers",
        "torch",
    )
    versions = {}
    for package in package_names:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    print(json.dumps({"status": "ok", "python": sys.version.split()[0], "packages": versions}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
