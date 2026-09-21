"""Explicit model-only versus narrow per-physics-frame reflex control."""

from __future__ import annotations

from typing import Any


CONTROL_MODES = ("model_only", "reflex_assisted")


def select_executed_action(observation: Any, raw_model_action: str, control_mode: str) -> tuple[str, list[str]]:
    """Apply the fixed game's narrow emergency reflex, keeping raw/model action visible.

    The reflex is evaluated every physics frame. It is deliberately separate from
    Laya's 8-frame decision cadence and cannot invent a model decision. A run using
    this mode is always reported as assisted, never as model-only.
    """

    if control_mode not in CONTROL_MODES:
        raise ValueError(f"Unknown control mode: {control_mode!r}")
    if control_mode == "model_only":
        return raw_model_action, []

    reasons: list[str] = []
    terrain = observation.terrain
    hazard = observation.hazard
    player = observation.player
    episode = observation.episode

    gap_critical = terrain.gap_ahead and (terrain.gap_distance_tiles or 99) <= 3.8
    obstacle_critical = terrain.obstacle_ahead and (terrain.obstacle_distance_tiles or 99) <= 2.2
    enemy_critical = (
        hazard.enemy_ahead
        and hazard.nearest_enemy is not None
        and (
            hazard.nearest_enemy.distance_pixels <= 130.0
            or hazard.jump_must_start_now
        )
    )
    stall_critical = episode.stalled_frames >= 3

    if player.grounded:
        if gap_critical:
            reasons.append("gap_critical<=3.8_coarse_tiles")
        if obstacle_critical:
            reasons.append("obstacle_critical<=2.2_coarse_tiles")
        if enemy_critical:
            reasons.append("enemy_critical<=130px_or_jump_now")
        if stall_critical:
            reasons.append("stalled>=3_frames")
    is_over_or_near_gap = (
        not player.grounded
        and terrain.gap_ahead
        and terrain.gap_distance_tiles is not None
        and terrain.gap_distance_tiles <= 1.5
    )
    if is_over_or_near_gap:
        reasons.append("airborne_gap<=1.5_coarse_tiles")

    if reasons:
        return "right_run_jump", reasons
    return raw_model_action, []
