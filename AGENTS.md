# Colab experiments

- Python environments and commands must use uv. Keep dependencies scoped to experiments/<slug>/.
- Each task works on its own git worktree and branch; never modify another task's worktree or force-push.
- Scope files to experiments/<slug>/ (including notebooks, scripts, results, README). Do not modify root shared files from experiment tasks.
- Commit and push meaningful milestones to origin on your own branch. The public remote is Sunwood-ai-labs/jev-colab-lab. Verify the remote SHA after pushing.
- Publish only reviewed code and sanitized results. Never commit credentials, OAuth links/codes, session metadata, private input, raw user conversation, or model weights. Keep notebook outputs sanitized.
- Use the official Google Colab CLI https://github.com/googlecolab/google-colab-cli . It currently supports Linux/macOS, so use WSL on this Windows host. Do not claim Windows support.
- Runtime isolation: use a unique session name jev-<slug> and a task-specific --config directory outside git. Consult CLI help for exact syntax. Never stop/reuse another task's runtime. On quota exhaustion record the condition without repeatedly provisioning. Never purchase credits or upgrade plans.
- All tasks may develop concurrently; GPU execution depends on Colab's actual quota. Release your own runtime after downloading outputs, including on failure. Avoid unbounded training.
- Verify upstream repositories, model revisions, licenses and hardware needs from primary sources. User research is a hypothesis, not verified evidence.
- Target T4 for laya, kev, jevlike; L4 for semif and openjev-nli. Do not provision other GPU types without a user request.
- Save sanitized environment, source revisions, inputs/options, probabilities, timings, peak VRAM and errors. Separate warmup/loading from steady-state inference. Do not report local smoke tests as Colab GPU success.
- If authentication is required, finish all independent preparation, describe the precise user action, and do not expose credentials. Do not overwrite shared authentication or CLI installation concurrently.
- Completion report in Japanese: artifacts, tests, real GPU result or specific blocker, branch and commit.
