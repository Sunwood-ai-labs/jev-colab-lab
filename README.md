<div align="center">
  <img src="https://raw.githubusercontent.com/Sunwood-ai-labs/jev-colab-lab/main/docs/public/jev-colab-lab-icon.svg" alt="Jev Colab Lab icon" width="96">
  <h1>Jev Colab Lab</h1>
  <p>Reproducible Google Colab GPU experiments for decision-model inference</p>
</div>

<p align="center">
  <a href="https://sunwood-ai-labs.github.io/jev-colab-lab/">Documentation</a> ·
  <a href="https://github.com/Sunwood-ai-labs/jev-colab-lab">Repository</a>
</p>

<p align="center">
  <a href="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/public-qa.yml"><img src="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/public-qa.yml/badge.svg" alt="Public QA"></a>
  <a href="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/docs.yml"><img src="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/docs.yml/badge.svg" alt="Docs build"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-0b1020.svg" alt="MIT license"></a>
</p>

<p align="center"><strong>English</strong> · <a href="README.ja.md">日本語</a></p>

## 🔭 What this repository is

This repository records small, reproducible experiments that run Jev-like decision-model paths on Google Colab GPUs. Each experiment keeps its runner, notebook, pinned upstream revisions, sanitized inputs, and result JSON together under <code>experiments/&lt;slug&gt;/</code>.

The current snapshot contains five integrated GPU result sets: Laya, Kev-0.5B, Jevlike tiny option-attention, SemIf 4B, and OpenJev NLI 4B. They target different model paths and fixtures, so their probabilities, accuracy, latency, and VRAM values are not a common benchmark.

## 🚀 Quick start

The repository is documentation-first: local checks do not download model weights.

~~~powershell
git clone https://github.com/Sunwood-ai-labs/jev-colab-lab.git
Set-Location jev-colab-lab

# Run the model-free helper and fixture tests with uv.
uv run --no-project --with pytest pytest experiments/laya/tests experiments/openjev-nli/tests experiments/semif/tests -q

# Run the Jevlike regression suite with its declared torch dependency.
uv run --project experiments/jevlike --extra dev pytest experiments/jevlike/tests -q

# Check the Kev runner syntax without installing its model dependencies.
uv run --no-project python -m py_compile experiments/kev/t4_inference.py
~~~

For a model-backed local smoke test, follow the experiment README and use that experiment's <code>uv</code> project. A local CPU run is never reported as a Colab GPU result.

## 🧪 Experiment catalog

| Experiment | Target | Measurement focus | Evidence |
| --- | --- | --- | --- |
| [Laya / ModernBERT](experiments/laya/README.md) | Colab T4 | choice, score, and noul decision outputs | [runner](experiments/laya/scripts/benchmark_laya.py) · [T4 JSON](experiments/laya/results/laya-t4-result.json) |
| [Kev-0.5B](experiments/kev/README.md) | Colab T4 | packed/separate option probabilities and latency | [runner](experiments/kev/t4_inference.py) · [T4 JSON](experiments/kev/results/kev-t4-result.json) |
| [Jevlike](experiments/jevlike/README.md) | Colab T4 | short synthetic training, held-out test, shuffled-context control | [runner](experiments/jevlike/scripts/run_experiment.py) · [T4 JSON](experiments/jevlike/results/colab-t4-result.json) |
| [SemIf / Qwen3.5-4B](experiments/semif/README.md) | Colab L4 | direct next-token option-logit readout | [runner](experiments/semif/scripts/run_experiment.py) · [L4 JSON](experiments/semif/results/semif-l4-result-20260921.json) |
| [OpenJev NLI 4B](experiments/openjev-nli/README.md) | Colab L4 | entailment, contradiction, and neutral scores | [runner](experiments/openjev-nli/scripts/measure_openjev.py) · [L4 JSON](experiments/openjev-nli/results/colab-l4.json) |

The checked-in JSON is the canonical evidence. Jevlike timing fields retain their measurement boundary in the result schema; this landing page does not use them as a cross-experiment comparison point.

## 🎮 JevDash capture set

The five model paths have a JevDash Level 1 capture set outside Git. This is an independent single-episode observation set, not a completed comparative game-ability evaluation; the Jevlike path has a confirmed input-truncation defect. The public experiment pages link to the runners and sanitized evidence that can be reviewed here; the MP4 files themselves have no public URL in this repository and are intentionally kept out of Git.

These are independent single-episode demonstrations, not a model ranking. Video time is simulation time with synchronous inference waits omitted. The Kev and Jevlike presentation replays preserve the recorded model trajectory and state, changing only the HUD rendering. See the [Japanese Colab summary](docs/ja/guide/jev-clone-colab.md) for the measured selection results and limitations.

Start from the [public capture evidence index](experiments/README.md#jevdash-capture-evidence) for the five experiment-specific conduits.

## 🧭 How to read the results

- <code>status=success</code> or <code>status=ok</code> is meaningful only together with the recorded GPU and runtime fields.
- Model loading, first inference, warmup, steady-state inference, and peak VRAM are separate measurements where the runner supports them.
- Conditional option scores are not automatically calibrated confidence, and a fixture accuracy is not a generalization claim.
- Failure JSON files remain useful evidence of dependency or authentication blockers; they are not successful GPU runs.
- Inputs are synthetic or sanitized. Credentials, OAuth links, session metadata, and model weights are intentionally excluded.

## 🛠️ Reproduce a Colab run

The official Colab CLI currently runs on Linux and macOS, not Windows. On a Windows host, use WSL Ubuntu and keep the CLI state outside this repository:

~~~bash
uv tool install google-colab-cli
colab --help
~~~

Each experiment README gives the target accelerator, a unique <code>jev-&lt;slug&gt;</code> session name, an isolated session-state file, the upload/execute/download flow, and the required stop or ephemeral-run cleanup. GPU quota, entitlement, and authentication can still block a run; record that condition once instead of retrying indefinitely.

## 🗂️ Repository map

~~~text
experiments/<slug>/
├── README.md                 # experiment-specific scope and reproduction
├── notebooks/                # Colab CLI notebooks where applicable
├── results/                  # sanitized measurements and blocker records
├── scripts/                  # runners and fixture validators
└── tests/                    # model-free helper tests where applicable
docs/                         # published bilingual guide
.github/workflows/            # docs deployment and public QA
NOTICE.md                     # upstream source and license boundaries
~~~

## 🧱 Scope and limitations

This is a research record, not the private TypeSafe Jev model, a production decision service, or a claim that the listed implementations are equivalent. Nimble 9B, OpenJev 35B, and DiffusionGemma remain deferred. See the [full guide](https://sunwood-ai-labs.github.io/jev-colab-lab/) for measurement semantics, source revisions, and troubleshooting.

## ⚖️ License and sources

Original repository glue code and documentation are released under the [MIT License](LICENSE). Upstream implementations, model weights, and their licenses remain separate; see [NOTICE.md](NOTICE.md) and each experiment's pinned source manifest. No model weights are stored in this repository.

## 📚 More documentation

- [English docs](https://sunwood-ai-labs.github.io/jev-colab-lab/)
- [日本語 docs](https://sunwood-ai-labs.github.io/jev-colab-lab/ja/)
- [Experiment index](experiments/README.md)
- [Contribution and operating rules](AGENTS.md)
