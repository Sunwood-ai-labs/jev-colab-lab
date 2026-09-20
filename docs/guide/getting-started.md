# Getting started

This project separates model-free repository checks from optional model-backed runs. Start with the first path; it is fast and does not download weights.

## 1. Install the local tools

Install [uv](https://docs.astral.sh/uv/) and Node.js 20 or newer if you want to build the docs site locally. Python commands in this repository should run through <code>uv</code>.

~~~powershell
git clone https://github.com/Sunwood-ai-labs/jev-colab-lab.git
Set-Location jev-colab-lab
uv --version
~~~

## 2. Run model-free checks

~~~powershell
uv run --no-project --with pytest pytest experiments/laya/tests experiments/openjev-nli/tests experiments/semif/tests -q
uv run --project experiments/jevlike --extra dev pytest experiments/jevlike/tests -q
uv run --no-project python -m py_compile experiments/kev/t4_inference.py
~~~

These tests validate helper logic, public fixtures, notebook structure, and Python syntax. They do not prove that a GPU is available or that a model download will succeed.

## 3. Choose an experiment

Open the corresponding [experiment README](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments) and check its source manifest before installing anything. The experiment README is the authority for its own dependency constraints and result file.

| Target | Experiments | Intended use |
| --- | --- | --- |
| T4 | Laya, Kev-0.5B, Jevlike | lower-cost decision-model and tiny scorer paths |
| L4 | SemIf 4B, OpenJev NLI 4B | 4B direct-logit and NLI paths |

## 4. Run on Colab from Windows

The official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) currently supports Linux and macOS, not Windows. Use WSL Ubuntu on a Windows host:

~~~bash
uv tool install google-colab-cli
colab --help
~~~

Each run should have:

- a unique session name such as <code>jev-laya</code> or <code>jev-semif</code>
- a task-specific session state file outside the repository
- the requested GPU recorded separately from the actual GPU reported in the result
- an explicit download of sanitized results before stopping a manually created session

Authentication, quota, and entitlement failures are valid outcomes to record. Do not commit OAuth links, ADC files, session logs, or weights.

## 5. Read the evidence

Open the linked result JSON after a run. Treat the GPU as successful only when the result has a successful status and the actual GPU is recorded. See [Reproducibility](/guide/reproducibility) for the measurement vocabulary.
