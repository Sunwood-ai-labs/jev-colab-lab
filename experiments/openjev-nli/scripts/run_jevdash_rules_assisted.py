"""Colab entrypoint for rules-v2 OpenJev execution with logged reflexes."""

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
                "assisted",
                "--video",
                "/content/openjev-nli-jevdash-rules-assisted.mp4",
                "--json",
                "/content/openjev-nli-jevdash-rules-assisted.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-rules-assisted-frames",
            ]
        )
    )
