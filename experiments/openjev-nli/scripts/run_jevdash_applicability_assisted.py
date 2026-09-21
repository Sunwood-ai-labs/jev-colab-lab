"""Colab entrypoint for action-applicability OpenJev with logged reflexes."""

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
                "assisted",
                "--video",
                "/content/openjev-nli-jevdash-applicability-assisted.mp4",
                "--json",
                "/content/openjev-nli-jevdash-applicability-assisted.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-applicability-assisted-frames",
            ]
        )
    )
