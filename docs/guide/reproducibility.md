# Reproducibility

Reproduction means preserving the source revision, runtime boundary, input fixture, measurement semantics, and cleanup path. It does not mean that a future Colab VM will have identical quota, package cache, or wall-clock latency.

## Source and environment

Each experiment records the upstream implementation and model revision in a local source manifest or in its result JSON. Keep the source revision fixed when comparing a rerun. Do not replace a pinned revision with a moving <code>main</code> URL in a notebook or script.

The intended Python workflow is <code>uv</code>:

~~~powershell
uv run --project experiments/laya --group dev pytest -q
uv run --project experiments/semif --group dev python scripts/validate_fixture.py data/questions.jsonl
~~~

The exact command can differ by experiment; its README is authoritative.

## Runtime isolation

Use one task-specific session name and one task-specific state file. Keep both outside Git:

~~~bash
CFG=/tmp/jev-semif-colab-session.json
COLAB=colab
$COLAB --config "$CFG" new --session jev-semif --gpu L4
~~~

Do not reuse another task's active session. Download the sanitized result and stop a manually created session even when the experiment fails. The ephemeral <code>colab run</code> path can release its own runtime after the script completes; follow the experiment README for its supported syntax.

## Measurement vocabulary

| Term | Meaning |
| --- | --- |
| Load | Model download/deserialization and device placement as defined by the runner. |
| First inference | The first measured inference region after the runner's stated setup boundary. It may include one-time kernel or cache effects. |
| Warmup | Deliberate unreported or separately reported repetitions before steady-state sampling. |
| Steady state | Repeated inference samples summarized with a mean/median/percentile where available. |
| Peak VRAM | The peak allocated/reserved CUDA memory reported by the runner, not total card capacity. |
| Fixture accuracy | Accuracy on the checked-in or synthetic fixture only; not a general benchmark. |

The result JSON is the source of truth for the exact boundary. Never infer a missing metric from another experiment's schema.

## Sanitization checklist

Before committing a result:

- remove model weights and large downloaded caches
- remove OAuth links, ADC credentials, tokens, cookies, and private session metadata
- replace private inputs with public or synthetic fixtures
- remove absolute local paths when they are not needed for interpretation
- preserve the GPU name, source revision, package versions, timings, memory, and errors

## Failure handling

Quota exhaustion, missing authentication, and dependency incompatibility are part of the experiment history. Record the condition once in a sanitized JSON file, release any runtime you created, and do not represent a local fallback as GPU success.
