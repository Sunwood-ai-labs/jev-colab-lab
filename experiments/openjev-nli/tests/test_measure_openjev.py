import json
import tempfile
import unittest
from pathlib import Path

import sys


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))

from measure_openjev import (  # noqa: E402
    LABELS,
    build_pairs,
    load_fixture,
    percentile,
    summarize_timings,
)


class MeasurementHelpersTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(percentile([], 0.5), None)

    def test_timing_summary_is_stable(self):
        summary = summarize_timings([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["p95_ms"], 3.85)

    def test_fixture_load_and_pair_template(self):
        fixture = {
            "items": [
                {
                    "id": "one",
                    "premise": "P",
                    "hypotheses": ["H1", "H2"],
                    "gold_index": 0,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            loaded = load_fixture(path)
        pairs = build_pairs(loaded, "Premise: {premise}\nHypothesis: {hypothesis}")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[1]["text"], "Premise: P\nHypothesis: H2")
        self.assertEqual(tuple(LABELS), ("contradiction", "entailment", "neutral"))


if __name__ == "__main__":
    unittest.main()
