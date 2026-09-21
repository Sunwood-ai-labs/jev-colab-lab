"""Colab entrypoint for rules-v2 OpenJev model-only execution."""

import sys

sys.path.insert(0, "/content")

from run_jevdash_openjev import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--profile",
                "rules-v2",
                "--execution-mode",
                "model-only",
                "--video",
                "/content/openjev-nli-jevdash-rules-model-only.mp4",
                "--json",
                "/content/openjev-nli-jevdash-rules-model-only.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-rules-model-only-frames",
            ]
        )
    )
