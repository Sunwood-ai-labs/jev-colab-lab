"""Colab entrypoint for the restricted action test with logged reflexes."""

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
                "assisted",
                "--video",
                "/content/openjev-nli-jevdash-conditioned-two-assisted.mp4",
                "--json",
                "/content/openjev-nli-jevdash-conditioned-two-assisted.json",
                "--frame-dir",
                "/content/openjev-nli-jevdash-conditioned-two-assisted-frames",
            ]
        )
    )
