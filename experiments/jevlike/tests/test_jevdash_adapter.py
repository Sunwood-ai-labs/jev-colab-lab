from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapter" / "jevdash_colab_runner.py"
REPLAY = ROOT / "adapter" / "jevdash_replay.py"
CLEAR = ROOT / "adapter" / "jevdash_clear_colab_runner.py"


def test_jevdash_adapter_is_syntax_valid_and_pinned():
    source = ADAPTER.read_text(encoding="utf-8")
    compile(source, str(ADAPTER), "exec")
    assert "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480" in source
    assert "94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452" in source
    assert "FRAMES_PER_DECISION = 8" in source
    assert "MAX_SIMULATION_FRAMES = 1800" in source
    assert "TERMINAL_HOLD_FRAMES = 120" in source


def test_jevdash_adapter_has_no_mock_or_live_controller_import():
    source = ADAPTER.read_text(encoding="utf-8")
    assert "MockJevAgent" not in source
    assert "JevLiveAgent" not in source
    assert "AsyncJevAgent" not in source
    assert "VercelGatewayJevClient" not in source


def test_jevdash_adapter_uses_fixed_action_space_and_truthful_clock_label():
    source = ADAPTER.read_text(encoding="utf-8")
    for action in ("noop", "right", "right_run", "right_jump", "right_run_jump", "jump", "left"):
        assert f'"{action}"' in source
    assert "SIMULATION TIME (inference waits omitted)" in source
    assert "60 FPS simulation;" in source
    assert "inference waits excluded" in source
    assert "video_time_seconds" in source
    assert "DANGER / URGENCY: NOT MEASURED" in source


def test_jevdash_replay_is_a_pinned_state_checked_presentation_correction():
    source = REPLAY.read_text(encoding="utf-8")
    compile(source, str(REPLAY), "exec")
    assert "recorded_trajectory_replay" in source
    assert "additional_model_inference" in source
    assert "source_trajectory_sha256" in source
    assert "state mismatch" in source


def test_clear_runner_is_pinned_and_keeps_model_only_separate():
    source = CLEAR.read_text(encoding="utf-8")
    compile(source, str(CLEAR), "exec")
    assert "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480" in source
    assert "94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452" in source
    assert "CONTEXT_TOKENS = 192" in source
    assert "gap_hazard = gap_distance is not None and gap_distance <= 400.0" in source
    assert 'mode not in {"model-only", "assisted"}' in source
    assert 'safety_reflex_evaluation": "every physics frame"' in source
    assert '"primary_model_only_success"' in source


def test_clear_runner_has_no_live_or_mock_controller_path():
    source = CLEAR.read_text(encoding="utf-8")
    assert "MockJevAgent" not in source
    assert "JevLiveAgent" not in source
    assert "AsyncJevAgent" not in source
    assert "VercelGatewayJevClient" not in source
    assert '"live_api_used": False' in source
    assert '"mock_used": False' in source
