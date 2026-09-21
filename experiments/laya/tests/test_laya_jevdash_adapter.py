from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from adapter.laya_agent import (  # noqa: E402
    GAME_ACTIONS,
    build_action_question,
    semantic_state_text,
)
from adapter.control import select_executed_action  # noqa: E402


def _observation(**overrides):
    player_values = {"grounded": True, "airborne_frames": 0}
    player_values.update(overrides.pop("player", {}))
    player = SimpleNamespace(**player_values)
    terrain_values = {
        "gap_ahead": False,
        "gap_distance_tiles": None,
        "obstacle_ahead": False,
        "obstacle_distance_tiles": None,
    }
    terrain_values.update(overrides.pop("terrain", {}))
    terrain = SimpleNamespace(**terrain_values)
    nearest_enemy = overrides.pop("nearest_enemy", None)
    hazard_values = {
        "enemy_ahead": nearest_enemy is not None,
        "nearest_enemy": nearest_enemy,
        "jump_must_start_now": False,
    }
    hazard_values.update(overrides.pop("hazard", {}))
    hazard = SimpleNamespace(**hazard_values)
    episode_values = {"stalled_frames": 0}
    episode_values.update(overrides.pop("episode", {}))
    episode = SimpleNamespace(**episode_values)
    return SimpleNamespace(player=player, terrain=terrain, hazard=hazard, episode=episode)


def test_question_profiles_keep_all_actions_and_describe_strong_jump():
    question = build_action_question("platformer_guided")
    assert tuple(question["criteria"]) == GAME_ACTIONS
    assert "strong forward jump" in question["criteria"]["right_run_jump"]
    assert "coarse tile-column" in question["instructions"]


def test_semantic_state_retains_all_observation_sections():
    state = {
        "objective": "reach",
        "player": {
            "x": 1,
            "y": 2,
            "vx": 0,
            "vy": 0,
            "grounded": True,
            "jumping": False,
            "airborne_frames": 0,
            "running": False,
        },
        "hazard": {"enemy_ahead": False, "nearest_enemy": None, "jump_must_start_now": False, "in_danger_zone": False},
        "terrain": {"obstacle_ahead": True, "obstacle_distance_tiles": 1, "obstacle_height_tiles": 2, "gap_ahead": False, "gap_distance_tiles": None, "gap_width_tiles": 0, "clear_forward_tiles": 2},
        "episode": {"score": 0, "coins": 0, "lives": 3, "progress_pixels": 1, "goal_distance_pixels": 100, "stalled_frames": 4, "is_dead": False, "has_won": False},
        "local_grid": ["..P.."],
    }
    text = semantic_state_text(state)
    assert "PLAYER:" in text
    assert "HAZARD:" in text
    assert "TERRAIN:" in text
    assert "EPISODE:" in text
    assert "LOCAL_RADAR_7x11:" in text
    assert "stalled_frames=4" in text


def test_model_only_never_changes_the_model_action():
    observation = _observation(
        terrain={"obstacle_ahead": True, "obstacle_distance_tiles": 1.0},
        episode={"stalled_frames": 100},
    )
    assert select_executed_action(observation, "right_run", "model_only") == ("right_run", [])


def test_reflex_assistance_is_explicit_for_grounded_obstacle_and_stall():
    observation = _observation(
        terrain={"obstacle_ahead": True, "obstacle_distance_tiles": 1.0},
        episode={"stalled_frames": 100},
    )
    action, reasons = select_executed_action(observation, "right_run", "reflex_assisted")
    assert action == "right_run_jump"
    assert "obstacle_critical<=2.2_coarse_tiles" in reasons
    assert "stalled>=3_frames" in reasons


def test_reflex_assistance_protects_airborne_gap():
    observation = _observation(
        player={"grounded": False, "airborne_frames": 5},
        terrain={"gap_ahead": True, "gap_distance_tiles": 1.0},
    )
    action, reasons = select_executed_action(observation, "right_run", "reflex_assisted")
    assert action == "right_run_jump"
    assert reasons == ["airborne_gap<=1.5_coarse_tiles"]


def test_invalid_candidate_order_is_rejected():
    with pytest.raises(ValueError):
        build_action_question("baseline", GAME_ACTIONS[:-1])
