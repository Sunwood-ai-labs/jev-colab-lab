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

## Deferred work

Nimble 9B, OpenJev 35B, and DiffusionGemma were outside the initial T4/L4 batch. They should receive their own folder and source manifest if they are resumed; do not silently mix them into an existing result.

The Jevlike timing correction is tracked in a separate review task. Until that change is integrated into <code>main</code>, avoid comparing its earlier first-inference value with the other experiments.
