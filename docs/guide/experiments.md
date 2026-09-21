# Experiments

Five experiments are integrated into <code>main</code>. They answer different questions and should be read independently.

| Experiment | GPU | What the runner does | Result |
| --- | --- | --- | --- |
| [Laya / ModernBERT](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/laya) | T4 | Runs choice, score, and noul decision outputs with synthetic cases. | [T4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/results/laya-t4-result.json) |
| [Kev-0.5B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/kev) | T4 | Uses the official decision model and adapter, then compares packed and separate reads. | [T4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/results/kev-t4-result.json) |
| [Jevlike](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/jevlike) | T4 | Trains a tiny scorer briefly, evaluates held-out synthetic data, and runs a shuffled-context control. | [T4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/results/colab-t4-result.json) |
| [SemIf / Qwen3.5-4B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/semif) | L4 | Reads next-token logits for declared single-token options and applies option-local softmax. | [latest L4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/results/semif-l4-result-20260921.json) |
| [OpenJev NLI 4B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/openjev-nli) | L4 | Scores fixture premise/hypothesis pairs with contradiction, entailment, and neutral outputs. | [L4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/results/colab-l4.json) |

## Evidence status

The Laya and OpenJev JSON use <code>status=ok</code>; Kev, Jevlike, and SemIf use <code>status=success</code>. In every case, inspect the adjacent runtime or hardware object for the actual GPU. A requested T4/L4 value by itself is not evidence of allocation.

The catalog also keeps failure records:

- Kev keeps the initial dependency mismatch beside its successful rerun.
- SemIf keeps the pre-fix optional-package import failure.
- Jevlike keeps the earlier authentication blocker.

Failures explain the path to a successful run. They are not extra successes.

## JevDash capture evidence

The five model paths have a completed JevDash Level 1 capture set outside Git. The experiment pages below are the public conduits for the runners and checked-in evidence; MP4 files are intentionally external and no public video URL is claimed here.

| Model path | Public conduit | What to verify |
| --- | --- | --- |
| Laya | [experiment README](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/README.md) | Fixed game revision, real T4 adapter path, and simulation-time labeling. |
| Kev | [experiment README](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/README.md) | T4 trajectory, state-preserving presentation replay, and sanitized manifests. |
| Jevlike | [capture guide](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/jevdash-capture.md) | Input truncation makes the game result invalid for ability evaluation; replay is HUD-only. |
| SemIf | [experiment README](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/README.md) | Real L4 option-logit control and fixed game marker. |
| OpenJev NLI | [experiment README and evidence](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/README.md) | L4 NLI control, manifests, judgment log, representative frames, and decode evidence. |

These are independent single episodes, not a ranking. Video time is simulation time with synchronous inference waits omitted. Kev and Jevlike presentation replays preserve the recorded trajectory and state and change only the HUD rendering.

Jevlike has a confirmed input-path defect: its 192-byte `ByteCollator` prefix is identical across all 27 decisions, as are the probability vector and `right_jump` action, before a 3.6-second death. The scratch scorer was trained only on synthetic badge-selection data. The other four paths are not confirmed to have the same defect, but their game connections are not a completed comparative ability evaluation.

## Comparison boundaries

Do not rank these rows by raw latency or probability. The models, prompt formats, option counts, batch shapes, precision, and fixture semantics differ. SemIf scores are conditional option scores, OpenJev outputs are NLI probabilities, and Jevlike accuracy is a short synthetic experiment with a shuffled-context control.

Jevlike timing fields retain their measurement boundary in the result schema. Do not compare any first-inference field across experiments unless the runner definitions and setup boundaries are equivalent.
