# JevDash Jevlike r2 public verification

This folder contains the sanitized public evidence for the real Colab T4 r2 run. It intentionally excludes raw full logs, absolute local paths, session metadata, credentials, and model weights.

## Results

| Mode | GPU | Outcome | Simulation frames | Final progress | Actual overrides | Guard triggers |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [model-only](model-only-summary.json) | Tesla T4 | `has_won=true` | 647 | 4204.4 px | 0 | 0 |
| [assisted](assisted-summary.json) | Tesla T4 | `has_won=true` | 647 | 4204.4 px | 0 | 56 |

`actual_override` means `raw_action != executed_action`. A `guard_trigger` means the safety reflex returned a reason, even when it returned the same action. The r2 model-only trace selected `right_run_jump` for all 647 simulation frames; this is a constant-action clear and is not claimed as state-conditioned generalization.

## Evidence

- [model-only video](model-only.mp4) and [terminal frame](model-only-last.png)
- [assisted video](assisted.mp4) and [terminal frame](assisted-last.png)
- [verification manifest](verification.json), including artifact SHA256, ffprobe, full decode status, source revisions, model/training metadata, probabilities, sampled state trace, `has_won`, and VRAM measurements

## Revisions and contract

- JevDash: `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480`
- Jevlike: `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`
- Runner: `95381a0a76de8beb5e2f1b966dee12d23b75ae6d`
- Training variant: `r2-game-teacher`
- Level 1, seed 42, 60 FPS, decision every 8 physics frames, terminal hold 120 frames
- No mock controller, fallback controller, live API, or published model weights
