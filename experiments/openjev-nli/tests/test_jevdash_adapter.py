import json
import sys
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from adapter.openjev_nli import ACTION_OPTIONS, ACTION_DESCRIPTIONS, build_hypotheses, build_premise  # noqa: E402


class OpenJevActionMappingTest(unittest.TestCase):
    def test_action_mapping_is_fixed_and_complete(self):
        hypotheses = build_hypotheses()
        self.assertEqual([item["action"] for item in hypotheses], list(ACTION_OPTIONS))
        self.assertEqual(len(hypotheses), 7)
        self.assertTrue(all(ACTION_DESCRIPTIONS[item["action"]] in item["hypothesis"] for item in hypotheses))

    def test_premise_contains_only_serialized_observation(self):
        observation = {"episode": {"progress_pixels": 12}, "local_grid": ["P.........."]}
        premise = build_premise(observation)
        self.assertIn("Structured game observation:", premise)
        self.assertIn(json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":")), premise)
        self.assertNotIn("api_key", premise.lower())
