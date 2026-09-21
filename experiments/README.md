# Experiment catalog

This directory contains five independent, pinned experiments. Each folder owns its runner, notebook or fixture, sanitized result JSON, and local <code>uv</code> instructions.

| Slug | Path | Target | Current evidence |
| --- | --- | --- | --- |
| <code>laya</code> | [README](laya/README.md) | Colab T4 | [T4 result](laya/results/laya-t4-result.json) |
| <code>kev</code> | [README](kev/README.md) | Colab T4 | [T4 result](kev/results/kev-t4-result.json) · [initial failure](kev/results/kev-t4-attempt1-error.json) |
| <code>jevlike</code> | [README](jevlike/README.md) | Colab T4 | [T4 result](jevlike/results/colab-t4-result.json) · [blocker history](jevlike/results/colab-t4-blocker.json) |
| <code>semif</code> | [README](semif/README.md) | Colab L4 | [latest L4 result](semif/results/semif-l4-result-20260921.json) · [failure evidence](semif/results/semif-l4-failure-20260920.json) |
| <code>openjev-nli</code> | [README](openjev-nli/README.md) | Colab L4 | [L4 result](openjev-nli/results/colab-l4.json) |

## Evidence rules

- A GPU result must record the actual accelerator in its sanitized JSON. A requested accelerator alone is not success evidence.
- Loading, first inference, warmup, steady state, and peak VRAM are separate only when the runner records them separately.
- Local CPU smoke tests and model-free fixtures validate code paths; they do not validate Colab GPU performance.
- Synthetic or public fixtures are not representative accuracy benchmarks, and option probabilities are not automatically calibrated confidence.
- Upstream implementation revisions, model revisions, licenses, and the Colab CLI are recorded in <code>source.json</code>, <code>source-manifest.json</code>, or <code>references/verified-sources.json</code>.

## JevDash capture evidence

The five model paths have a completed JevDash Level 1 capture set outside Git. These links are the public review conduits for the capture code and checked-in evidence; the MP4 files themselves are intentionally external and this repository does not claim a public video URL.

| Model path | Public conduit | Evidence boundary |
| --- | --- | --- |
| Laya | [capture runner and README](laya/README.md) | Real T4 adapter path; fixed game revision and simulation-time labeling. |
| Kev | [capture README](kev/README.md) | T4 trajectory plus state-preserving presentation replay; sanitized manifests live in `kev/results/`. |
| Jevlike | [capture guide](jevlike/jevdash-capture.md) | Input truncation makes the game result invalid for ability evaluation; replay is HUD-only with no extra inference. |
| SemIf | [capture README](semif/README.md) | Real L4 option-logit control; checked-in runner and fixed game marker. |
| OpenJev NLI | [capture README and evidence](openjev-nli/README.md) | Real L4 NLI control; manifests, judgment log, representative frames, and decode evidence are checked in. |

Treat the five captures as independent single episodes, not a ranking. Their video clock is simulation time with synchronous inference waits omitted. Where a presentation replay exists, it preserves the recorded trajectory and state and changes only the HUD rendering.

Jevlike has a confirmed input-path defect: its 192-byte `ByteCollator` prefix is identical across all 27 decisions, as are the probability vector and `right_jump` action, before a 3.6-second death. The scratch scorer was trained only on synthetic badge-selection data. The other four paths are not confirmed to have the same defect, but their game connections are not a completed comparative ability evaluation.

## Deferred work

Nimble 9B, OpenJev 35B, and DiffusionGemma were outside the initial T4/L4 batch. They should receive their own folder and source manifest if they are resumed; do not silently mix them into an existing result.

Jevlike timing fields retain their measurement boundary in the result schema. Avoid comparing any one experiment's first-inference field with another experiment's field unless the runner definitions and setup boundaries are equivalent.
