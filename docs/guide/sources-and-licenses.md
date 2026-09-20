# Sources and licenses

The repository's [MIT license](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/LICENSE) covers original experiment glue code and documentation. It does not relicense upstream implementations or model weights. The complete boundary note is in [<code>NOTICE.md</code>](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/NOTICE.md).

| Material | Source | Fixed revision | Upstream license |
| --- | --- | --- | --- |
| Laya implementation | [GitHub](https://github.com/NandhaKishorM/laya) | <code>d113dca2512fb3eaca313534bc54c7162d87c1d4</code> | Apache-2.0 |
| Laya checkpoint | [Hugging Face](https://huggingface.co/convaiinnovations/laya) | <code>1c5edc17a7acd8701df6fc341c0d179f1c62c982</code> | Apache-2.0 |
| Kev implementation | [GitHub](https://github.com/jaredpalmer/kev) | <code>ac67bf4e52d7bdc8420d8024c396df5585915d8c</code> | Apache-2.0 |
| Kev adapter | [Hugging Face](https://huggingface.co/jaredpalmer/kev-0.5b) | <code>edf1dc6d7f8d983c0adfd251e80a686e5539fc61</code> | Apache-2.0 |
| Kev base model | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B) | <code>060db6499f32faf8b98477b0a26969ef7d8b9987</code> | Apache-2.0 |
| Jevlike implementation | [GitHub](https://github.com/vinnylarouge/jevlike) | <code>94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452</code> | MIT |
| SemIf implementation | [GitHub](https://github.com/TheoLeeCJ/semif) | <code>ca3ba65f142967030ecb453346e94d6f476a69df</code> | MIT |
| SemIf and OpenJev base model | [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B) | <code>851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a</code> | Apache-2.0 |
| OpenJev NLI checkpoint | [Hugging Face](https://huggingface.co/AlexWortega/openjev) | <code>b32265f4700df7c02532933c9a4ff258a449d7ac</code> | MIT |
| Colab CLI | [GitHub](https://github.com/googlecolab/google-colab-cli) | CLI <code>0.6.0</code> in the recorded runs | Apache-2.0 |

## What is committed

The repository commits scripts, notebooks, public fixtures, sanitized JSON, and documentation. It does not commit model weights, downloaded caches, OAuth/ADC files, private session logs, or the original private discussion. The result JSON retains enough provenance to distinguish a real T4/L4 run from a CPU smoke test, fixture check, or failure record.

## Attribution boundary

The experiment runners may clone or install upstream code at runtime, but upstream source trees are not copied into this repository. Follow the license and model-card terms at the linked revision before redistributing a derived model output or rerunning with a different checkpoint.
