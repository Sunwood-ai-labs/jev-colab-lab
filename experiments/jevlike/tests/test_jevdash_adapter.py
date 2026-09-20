from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapter" / "jevdash_colab_runner.py"


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
    assert "DANGER / URGENCY: NOT MEASURED" in source
