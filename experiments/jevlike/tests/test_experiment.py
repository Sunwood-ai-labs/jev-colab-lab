import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
