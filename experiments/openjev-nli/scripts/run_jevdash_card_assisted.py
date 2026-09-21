"""Colab entrypoint for the improved OpenJev card prompt with a logged reflex."""

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
                "assisted",
                "--video",
                "/content/openjev-nli-jevdash-card-assisted.mp4",
                "--json",
                "/content/openjev-nli-jevdash-card-assisted.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-card-assisted-frames",
            ]
        )
    )
