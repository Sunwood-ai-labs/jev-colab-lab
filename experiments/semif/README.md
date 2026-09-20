# SemIf / Qwen3.5-4B direct logit readout

TheoLeeCJ/SemIf（旧 OpenJev）の direct 実装を、AlexWortega/openjev の NLI classifier 実装と混同せずに Colab L4 で測定するための独立実験です。SemIf は Qwen3.5-4B の回答文字 `A/B/C...` に対応する次トークン logits だけを読み、候補内で softmax します。回答文の生成も NLI head も使いません。

この実験は候補確率を「校正済みの信頼度」とは扱いません。出力は、与えた候補に条件付けられた比較スコアです。

## 固定した一次情報

- SemIf: <https://github.com/TheoLeeCJ/semif>
  - 実験で参照する上流 commit: `ca3ba65f142967030ecb453346e94d6f476a69df`
  - プロジェクトコード: MIT
- Model: <https://huggingface.co/Qwen/Qwen3.5-4B>
  - 実験で参照する model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
  - model card のライセンス: Apache-2.0
- Colab CLI: <https://github.com/googlecolab/google-colab-cli>
  - Windows ホストから WSL Ubuntu 24.04 経由で使用
  - CLI `0.6.0` の公式仕様では GPU は `new --gpu L4`、状態は session state file で分離

上流のモデル重み、OAuth/ADC 情報、Colab session metadata はこのディレクトリに保存・コミットしません。

## 収録物

| パス | 内容 |
| --- | --- |
| `data/questions.jsonl` | 3件の公開・非機密 fixture。state、question、候補と option id を含む |
| `scripts/run_experiment.py` | load / first inference / warmup / steady-state / peak CUDA memory を測る runner |
| `scripts/validate_fixture.py` | モデルをロードしない fixture 検証 |
| `scripts/colab_entry.py` | CLI でアップロードした runner を実行する薄い entry point |
| `requirements-colab.txt` | Colab 用の uv 管理依存固定 |
| `notebooks/semif_l4_experiment.ipynb` | 同じ runner を使う再現用 notebook（出力なしで保存） |
| `results/semif-l4-result-20260921.json` | 全依存 version を含む canonical な L4 実測 |
| `results/semif-l4-failure-20260920.json` | optional package 不整合を修正する前の sanitized failure evidence |

結果 JSON には、モデル id/revision、SemIf commit、入力 fixture、候補 logits/probabilities、prompt hash、load/forward/warmup/steady の時間、GPU 名・精度・依存バージョン、peak allocated/reserved VRAM、エラーを保存します。秘密情報や model weights は含めません。

## ローカル fixture 検証

Python は `uv` で実行します。モデルをダウンロードしない検証は次の通りです。

```powershell
Set-Location experiments/semif
uv run --group dev python scripts/validate_fixture.py data/questions.jsonl
uv run --group dev pytest -q
```

GPU 実行用の依存解決・環境作成は、同じディレクトリで次の形式です。

```powershell
uv sync --extra gpu --group dev
uv run --extra gpu python scripts/run_experiment.py `
  --input data/questions.jsonl `
  --output results/semif-local.json `
  --expected-gpu L4
```

ローカルに CUDA L4 がない場合は成功結果になりません。CPU smoke test を L4 成功として扱わないでください。

## Colab CLI で L4 を測る

以下は Windows PowerShell から WSL の公式 CLI を呼ぶ例です。`--config` は git worktree 外の専用 state file、session 名は `jev-semif` に固定しています。既存 ADC を使う場合は `--auth=adc` を付けます。

```powershell
$Repo = (Get-Location).Path
$WslRepo = '/mnt/c' + ($Repo.Substring(2) -replace '\\','/')
$Config = '/mnt/c/Users/makim/.codex/jev-semif-colab-state.json'
$Colab = '/home/makim/.local/bin/colab'

wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config new --session jev-semif --gpu L4
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config install `
  --session jev-semif -r "$WslRepo/experiments/semif/requirements-colab.txt"
$Prepare = 'uv pip uninstall --system torchvision torchaudio librosa'
$Prepare | wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config console --session jev-semif
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config restart-kernel `
  --session jev-semif
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config upload `
  --session jev-semif "$WslRepo/experiments/semif/scripts/run_experiment.py" /content/semif-run-experiment.py
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config upload `
  --session jev-semif "$WslRepo/experiments/semif/data/questions.jsonl" /content/semif-questions.jsonl
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config exec `
  --session jev-semif -f "$WslRepo/experiments/semif/scripts/colab_entry.py" --timeout 3600
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config download `
  --session jev-semif /content/semif-l4-result.json "$Repo/experiments/semif/results/semif-l4-result.json"
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config stop --session jev-semif
```

`colab install` は公式 CLI が VM 上で `uv pip install` を使う経路です。L4 の quota/entitlement がない場合は別 GPU を勝手に要求せず、作成エラーを結果として記録します。実行後、結果をダウンロードできたことを確認してから必ず自分の `jev-semif` session だけを stop します。

Colab の base image に残っている text-only 推論不要の `torchvision`、`torchaudio`、`librosa` が、固定した `torch` と不整合な場合があります。上の isolated session 内の console でだけ `uv pip uninstall` し、kernel を再起動します。これは共有ホストや別 session には影響しません。

実際の GPU 成功条件は、JSON の `status == "success"`、`runtime.gpu.name` に `L4` が含まれること、`errors == []`、`metrics.peak_vram` が存在することです。`status == "failed"` の場合はエラーを保存しても GPU 成功とは報告しません。

## Notebook

`notebooks/semif_l4_experiment.ipynb` は、リポジトリを Colab VM に配置して `experiments/semif` を開いた状態で、`uv sync --extra gpu --group dev --frozen`、optional media package の除去、fixture 検証、runner の順に実行します。Notebook の生成 output はコミットせず、取得した sanitized JSON を一次結果として扱います。

## 既知の境界

- 実験は SemIf の direct option-logit readout の計測であり、TypeSafe の非公開 Jev の内部構造を証明しません。
- `Qwen3.5-4B` の logits を候補文字へ制限するため、候補集合に条件付いた値です。
- 時間と VRAM は Colab の実 GPU、driver、kernel、cache 状態に依存します。
- AlexWortega/openjev の `Qwen3.5 + 3-class NLI classifier` は別実装であり、この実験の dependency・結果・主張に含めません。
