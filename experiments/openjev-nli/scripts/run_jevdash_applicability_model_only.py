"""Colab entrypoint for action-applicability OpenJev model-only execution."""

import sys

sys.path.insert(0, "/content")

from run_jevdash_openjev import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--profile",
                "applicability",
                "--execution-mode",
                "model-only",
                "--video",
                "/content/openjev-nli-jevdash-applicability-model-only.mp4",
                "--json",
                "/content/openjev-nli-jevdash-applicability-model-only.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-applicability-model-only-frames",
            ]
        )
    )
