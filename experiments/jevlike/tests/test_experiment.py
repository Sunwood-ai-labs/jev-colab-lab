import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_runner_is_valid_python_and_pins_official_commit():
    source = (ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    compile(source, str(ROOT / "scripts" / "run_experiment.py"), "exec")
    metadata = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    assert metadata["commit"] in source
    assert metadata["repository"] in source


def test_notebook_is_valid_and_uses_cuda_runner():
    notebook = json.loads(
        (ROOT / "notebooks" / "jevlike_t4_experiment.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--device', 'cuda" in code
    assert "run_experiment.py" in code


def test_best_state_snapshot_clones_parameter_storage():
    import torch

    from scripts.run_experiment import snapshot_trainable_state

    model = torch.nn.Linear(2, 2)
    snapshot = snapshot_trainable_state(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    assert not torch.equal(snapshot["weight"], model.weight)


def test_first_post_load_measurement_precedes_evaluation_passes():
    source = (ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    assert source.index("first_started =") < source.index("test_metrics = metrics")
    assert "first_post_load_batch_inference_seconds" in source
    assert "not a process-cold startup measurement" in source
