# Independent experiments

| Slug | Implementation | Initial target |
| --- | --- | --- |
| laya | Laya / ModernBERT | T4 |
| kev | Kev-0.5B | T4 |
| jevlike | Tiny option attention | T4 |
| semif | TheoLeeCJ / SemIf 4B | L4 |
| openjev-nli | AlexWortega / openjev NLI 4B | L4 |

Each experiment is developed in an independent worktree and pushed to its own branch. Hardware suitability remains unverified until execution. Nimble 9B, OpenJev 35B and DiffusionGemma are deferred from the initial T4/L4 batch.

Use uv for Python and the official Google Colab CLI via WSL. Authentication and account runtime limits can require user action or serialize GPU runs even when development proceeds in parallel.
