import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from runner.run_jevdash_openjev import assisted_action, _count_actions, _count_reasons  # noqa: E402


def observation(*, grounded=True, gap=False, gap_distance=None, obstacle=False, obstacle_distance=None,
                enemy=False, enemy_distance=None, jump_now=False, stalled=0):
    nearest = None
    if enemy:
        nearest = SimpleNamespace(distance_pixels=enemy_distance or 100.0)
    return SimpleNamespace(
        player=SimpleNamespace(grounded=grounded),
        terrain=SimpleNamespace(
            gap_ahead=gap,
            gap_distance_tiles=gap_distance,
            obstacle_ahead=obstacle,
            obstacle_distance_tiles=obstacle_distance,
        ),
        hazard=SimpleNamespace(
            enemy_ahead=enemy,
            nearest_enemy=nearest,
            jump_must_start_now=jump_now,
        ),
        episode=SimpleNamespace(stalled_frames=stalled),
    )


class JevDashAssistanceTest(unittest.TestCase):
    def test_clear_state_preserves_raw_model_action(self):
        self.assertEqual(assisted_action(observation(), "right")[0], "right")
        self.assertIsNone(assisted_action(observation(), "right")[1])

    def test_grounded_gap_uses_logged_jump_reflex(self):
        action, reason = assisted_action(observation(gap=True, gap_distance=2.0), "right")
        self.assertEqual(action, "right_run_jump")
        self.assertEqual(reason, "gap_ahead")

    def test_airborne_near_gap_uses_airborne_reason(self):
        action, reason = assisted_action(
            observation(grounded=False, gap=True, gap_distance=1.0), "right"
        )
        self.assertEqual(action, "right_run_jump")
        self.assertEqual(reason, "airborne_gap")

    def test_multiple_triggers_are_preserved(self):
        action, reason = assisted_action(
            observation(obstacle=True, obstacle_distance=2.0, stalled=3), "right"
        )
        self.assertEqual(action, "right_run_jump")
        self.assertEqual(reason, "obstacle_ahead+stalled")

    def test_action_and_reason_counts_are_deterministic(self):
        rows = [
            {"raw_action": "right", "executed_action": "right", "overridden": False, "override_reason": None},
            {"raw_action": "right", "executed_action": "right_run_jump", "overridden": True, "override_reason": "gap_ahead"},
            {"raw_action": "right", "executed_action": "right_run_jump", "overridden": True, "override_reason": "gap_ahead"},
        ]
        self.assertEqual(_count_actions(rows, "raw_action"), {"right": 3})
        self.assertEqual(_count_actions(rows, "executed_action"), {"right": 1, "right_run_jump": 2})
        self.assertEqual(_count_reasons(rows), {"gap_ahead": 2})


if __name__ == "__main__":
    unittest.main()
