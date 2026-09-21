"""Launch the unassisted compact/rules-v2 Kev JevDash episode."""

import os


os.environ.update(
    {
        "KEV_REPRESENTATION": "compact_json",
        "KEV_QUESTION_VARIANT": "rules_v2",
        "KEV_ORDER_VARIANT": "canonical",
        "KEV_ASSIST": "none",
        "KEV_JEVDASH_OUTPUT": "/content/kev-jevdash-model-only-rules-v2",
    }
)

import run_jevdash_colab


raise SystemExit(run_jevdash_colab.main())
