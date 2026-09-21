import json
import sys
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from adapter.openjev_nli import (  # noqa: E402
    ACTION_OPTIONS,
    ACTION_DESCRIPTIONS,
    CARD_ACTION_OPTIONS,
    build_card_hypotheses,
    build_compact_premise,
    build_hypotheses,
    build_premise,
)


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

    def test_card_hypotheses_follow_model_card_answer_format(self):
        hypotheses = build_card_hypotheses()
        self.assertEqual([item["action"] for item in hypotheses], list(CARD_ACTION_OPTIONS))
        self.assertTrue(all(item["hypothesis"].startswith("The correct answer is: ") for item in hypotheses))
        self.assertTrue(all(item["phrase"] in item["hypothesis"] for item in hypotheses))

    def test_card_hypotheses_preserve_a_permutation_without_duplicates(self):
        order = tuple(reversed(CARD_ACTION_OPTIONS))
        self.assertEqual([item["action"] for item in build_card_hypotheses(order)], list(order))
        with self.assertRaises(ValueError):
            build_card_hypotheses(order[:-1])

    def test_compact_premise_preserves_decision_fields_and_is_shorter(self):
        observation = {
            "player": {"x": 96.0, "y": 540.0, "vx": 0.0, "vy": 0.0, "grounded": True,
                       "jumping": False, "airborne_frames": 0, "running": False},
            "hazard": {"enemy_ahead": False, "nearest_enemy": None, "jump_must_start_now": False},
            "terrain": {"obstacle_ahead": True, "obstacle_distance_tiles": 2.0,
                        "obstacle_height_tiles": 2, "gap_ahead": False, "gap_distance_tiles": None,
                        "gap_width_tiles": 0, "clear_forward_tiles": 1},
            "episode": {"progress_pixels": 96.0, "goal_distance_pixels": 4130.0,
                        "stalled_frames": 0, "is_dead": False, "has_won": False},
            "local_grid": ["..#..", "..P.."],
        }
        compact = build_compact_premise(observation)
        legacy = build_premise(observation)
        self.assertIn("obstacle_distance=2", compact)
        self.assertIn("Radar ..#../..P..", compact)
        self.assertIn("stalled=0", compact)
        self.assertLess(len(compact.encode("utf-8")), len(legacy.encode("utf-8")))
