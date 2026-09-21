"""Colab entrypoint for the improved OpenJev card prompt without assistance."""

import sys

sys.path.insert(0, "/content")

from run_jevdash_openjev import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--profile",
                "card",
                "--execution-mode",
                "model-only",
                "--video",
                "/content/openjev-nli-jevdash-card-model-only.mp4",
                "--json",
                "/content/openjev-nli-jevdash-card-model-only.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-card-model-only-frames",
            ]
        )
    )
