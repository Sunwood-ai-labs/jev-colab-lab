"""Colab entrypoint for the restricted run-vs-running-jump model-only test."""

import sys

sys.path.insert(0, "/content")

from run_jevdash_openjev import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--profile",
                "conditioned-two",
                "--execution-mode",
                "model-only",
                "--video",
                "/content/openjev-nli-jevdash-conditioned-two-model-only.mp4",
                "--json",
                "/content/openjev-nli-jevdash-conditioned-two-model-only.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-conditioned-two-model-only-frames",
            ]
        )
    )
