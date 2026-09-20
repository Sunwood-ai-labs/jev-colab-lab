# openJEV NLI 4B / L4 実験

`AlexWortega/openjev` の **初版4B NLI分類器**だけを対象に、Google Colab L4で推論・速度・VRAMを再現可能な条件で測定する実験です。SemIfの直接logit readoutとは別実装として扱い、35B-A3Bと後発v2チェックポイントは対象外にします。

## 固定した一次情報

| 項目 | 固定値 |
| --- | --- |
| Model | `AlexWortega/openjev` |
| Subfolder | `qwen3.5-4b-nli` |
| Revision | `b32265f4700df7c02532933c9a4ff258a449d7ac` |
| Architecture | `Qwen3_5ForSequenceClassification` |
| Labels | `contradiction`, `entailment`, `neutral` |
| Model-card license | MIT |
| Base model | `Qwen/Qwen3.5-4B`（Apache-2.0） |

revisionを固定した理由は、モデルカードのmainが更新されるとv2や35Bを含むためです。4B初版のモデルカード・ファイル履歴は[このrevision](https://huggingface.co/AlexWortega/openjev/commit/b32265f4700df7c02532933c9a4ff258a449d7ac)、モデル概要は[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev)で確認できます。Qwenのベースモデル情報は[公式モデルカード](https://huggingface.co/Qwen/Qwen3.5-4B)を参照しました。

## 成果物

- `scripts/measure_openjev.py`: CUDA必須の測定本体。ダウンロード込みのload、初回forward、warmup後の定常forwardを分離します。
- `scripts/install_colab_dependencies.py`: Colab VMで`uv pip install --system`を行う依存準備。
- `data/fixture.json`: 公開・合成の5問×3仮説。個人情報や元会話は含めません。
- `notebooks/openjev_nli_l4.ipynb`: Colab向けの実行手順。
- `results/`: 回収したsanitized JSON、実行ログ、阻害要因の記録。

結果JSONには、入力・各仮説の3クラス確率、entailmentによる選択、精度、warmup/反復条件、load/初回/定常時間、追加ピークVRAM、GPU名、精度(dtype)、依存バージョン、model revisionを保存します。モデル重みはコミットしません。

## ローカル検証

Python環境とコマンドは`uv`で管理します。ローカル検証はモデルをダウンロードせず、測定ヘルパーとfixtureの構造だけを確認します。

```powershell
cd C:\Users\makim\.codex\worktrees\f190\jev-colab-lab\experiments\openjev-nli
uv sync
uv run python -m unittest discover -s tests -p "test_*.py"
```

## WSLのGoogle Colab CLIでL4実行

Colab CLIはLinux/macOS向けで、WindowsホストではWSLから実行します。公式CLIは[`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli)です。`--config`はディレクトリではなく、セッション状態JSONファイルを指定します。

```bash
cd /mnt/c/Users/makim/.codex/worktrees/f190/jev-colab-lab/experiments/openjev-nli
COLAB=/home/makim/.local/bin/colab
AUTH=--auth=adc
STATE=/tmp/jev-openjev-nli-colab-state.json

$COLAB $AUTH --config "$STATE" new --session jev-openjev-nli --gpu L4
$COLAB $AUTH --config "$STATE" upload data/fixture.json /content/openjev-nli-fixture.json --session jev-openjev-nli
$COLAB $AUTH --config "$STATE" exec --session jev-openjev-nli --file scripts/install_colab_dependencies.py --timeout 600
$COLAB $AUTH --config "$STATE" exec --session jev-openjev-nli --file scripts/measure_openjev.py --timeout 1800
$COLAB $AUTH --config "$STATE" download /content/openjev-nli-result.json results/colab-l4.json --session jev-openjev-nli
$COLAB $AUTH --config "$STATE" stop --session jev-openjev-nli
```

`new`がquota/authで失敗した場合は、同じsessionを繰り返し作成しません。準備済みコードと阻害要因を記録し、課金やGPU変更は行いません。実GPU結果がない場合、ローカル検証をColab成功とは報告しません。

## 測定定義

- dtype: `bfloat16`（モデルconfigの指定に合わせる）。
- `load`: tokenizerとmodelの取得・ロード・GPU配置。Colabの新規VMではダウンロード時間を含む。
- `first_inference`: load直後の1ペアforward。
- `single_pair`: 1ペア、warmup後10回。
- `one_question_three_hypotheses`: 1 premiseに対する3仮説を1 batch、warmup後10回。
- `all_fixture_pairs`: 15ペアを1 batch、warmup後5回。
- timingは既にtokenize済みテンソルのforwardのみ。tokenization時間は別記録。

モデル出力の3クラス確率を候補間で再softmaxしません。各仮説の`entailment`確率が最大の候補をdecisionとして保存し、NLI確率と候補選択を分離します。

## 状態

2026-09-20に、専用session `jev-openjev-nli` で実L4実行を完了し、session停止後に結果を回収しました。[`results/colab-l4.json`](results/colab-l4.json)が唯一の実測JSONです。

| 測定 | 結果 |
| --- | ---: |
| GPU | NVIDIA L4（22.03 GiB） |
| dtype / device | bfloat16 / cuda:0 |
| load（取得・ロード込み） | 51,832.4028 ms |
| first inference | 1,290.0469 ms |
| single pair 定常平均 / p50 / p95 | 213.6877 / 213.5644 / 214.9477 ms |
| 3 hypotheses batch 定常平均 / p50 / p95 | 216.3447 / 213.4359 / 230.5303 ms |
| 15 pairs batch 定常平均 / p50 / p95 | 255.3123 / 254.4055 / 257.9754 ms |
| 定常ピークreserved VRAM（1 / 3 / 15 pairs） | 8.545 / 8.701 / 9.609 GiB |
| fixture decision accuracy | 5/5（1.0、合成fixture限定） |

実行環境はPython 3.13.15、Torch 2.11.0+cu128、Transformers 5.15.0、huggingface-hub 1.29.0、Accelerate 1.14.0です。初回実行ではColab UI外の`HF_TOKEN` secret取得警告が出ましたが、モデルは公開・非gatedであり、認証情報を使わず完了しています。

L4実行が成功したため、quota・認証・モデル互換性の未解決blockerはありません。fixtureは動作確認用の5問であり、公開ベンチマークや本家Jevとの性能比較ではありません。
