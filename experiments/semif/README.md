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
| `scripts/jevdash_play.py` | 固定 JevDash を SemIf の7択 action logitsで同期操作し、動画とsanitized JSONを出す runner |
| `scripts/jevdash_colab_entry.py` | 専用 `jev-semif-jevdash` session 用の薄い entry point |
| `scripts/jevdash_audit.py` | 固定ゲームから作った実 telemetry で、入力表現・候補順・full-vocabulary readout を監査する runner |
| `scripts/jevdash_audit_colab_entry.py` | 専用 `jev-semif-jevdash-clear` session 用の監査 entry point |
| `data/jevdash-commit.txt` | 収録対象の固定 JevDash commit marker |
| `requirements-colab.txt` | Colab 用の uv 管理依存固定 |
| `requirements-semif-jevdash-colab.txt` | JevDash収録用のtorch / transformers / pygame / SemIf固定依存 |
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
$Config = '/tmp/jev-semif-colab-state.json'
$Colab = 'colab'

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

## JevDashを実モデルで収録した実測

固定した公開ゲーム <https://github.com/Sunwood-ai-labs/jevdash> の commit `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480` を `git archive` で Colab VM に配置し、ゲーム本体は変更せずに収録しました。モデルは SemIf commit `ca3ba65f142967030ecb453346e94d6f476a69df` の direct readout と、Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` です。

実行条件は Level 1、seed 42、simulation 60 FPS、8 frames/decision、最大1800 simulation frames、clear/death時の静止終端120 framesです。各decisionは現在の `JevObservation` をJSON stateとして7つの固定 action (`noop`, `right`, `right_run`, `right_jump`, `right_run_jump`, `jump`, `left`) に直接scoreし、`argmax`だけを次の8フレームへ同期適用します。mock agent、live gateway、fallback、ゲーム物理の変更はありません。動画時間はsimulation frameだけで数え、モデル推論待ち時間は動画時間に加えていません。

専用の Google Colab CLI session は `jev-semif-jevdash`、GPUはL4、状態ファイルはgit worktree外の専用state fileを使いました。実行後、結果をダウンロードしてから同sessionだけ停止済みです。再実行時の依存は次の通りです。

```powershell
$WslRepo = '/path/to/jev-colab-lab'
$Config = '/tmp/jev-semif-jevdash-colab-state.json'
$Colab = 'colab'
$VideoRoot = '/path/to/external/jevdash-videos/semif'
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config new --session jev-semif-jevdash --gpu L4
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config install `
  --session jev-semif-jevdash -r "$WslRepo/experiments/semif/requirements-semif-jevdash-colab.txt"
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config exec `
  --session jev-semif-jevdash -f "$WslRepo/experiments/semif/scripts/jevdash_colab_entry.py" --timeout 3600
```

今回の実測は次の通りです。

| 項目 | 実測 |
| --- | --- |
| GPU / Python / torch | NVIDIA L4 / 3.13.15 / 2.10.0+cu128 |
| runner status / outcome | `completed` / `timeout`（モデル・runnerエラーではない） |
| simulation / decision | 1800 frames / 225 decisions |
| 最終 progress / score / coins | 486 px / 0 / 0 |
| model load / inference wall | 57.9945 s / 42.2283 s |
| video | H.264 yuv420p、1280×720、60 FPS、1800 frames、30.000 s |
| terminal static frames | 0（clear/deathではなくtimeoutのため） |

sanitized JSON、MP4、ffprobe、全デコード証跡、代表PNG、検証manifestは、git外の次のディレクトリに保存しています。

`$VideoRoot`（git管理外）

代表フレーム（0、900、1799）を目視し、ゲームviewport、7択確率バー、L4表示、60 FPS/8F契約、`DANGER / URGENCY: NOT MEASURED` の表示を確認しました。文字切れ・重なりはありません。終端フレームの画面表示は29.98 s、JSONの1800フレーム時間は30.00 sで、60 FPSのフレーム境界として整合します。danger/urgencyはモデル出力として推測・表示していません。

## JevDash入力監査とmodel-onlyクリア

上記のtimeoutを、入力の意味・候補順・ゲーム物理の3層に分けて監査しました。監査は固定ゲームから `start`、`approach_pipe`、`blocked_pipe`、`first_gap`、`enemy_ahead`、`airborne` の6状態を作り、4 prompt variant × 4 action order = 96行を、同じ SemIf direct readout の実L4で測定したものです。JSON stateをそのまま渡したbaselineでは、`blocked_pipe` の canonical 順が `right_run`（確率0.384）を選びますが、候補順を変えると選択 action が変わりました。これは候補文字への条件付き softmaxだけの問題ではなく、候補列の意味づけをモデルが安定していないことを示します。

監査行の input tokenization では A〜G は単一 token (`32..38`) で、blocked baseline の full-vocabulary での候補7文字の確率質量は `0.9995927`、full-vocabulary argmax は `C` でした。したがって、このケースは候補外 token を隠した見かけの確信度ではありません。意味を明示した `rules_json` / `rules_compact` は、4つの候補順すべてで同じ action を選び、`blocked_pipe` は `right_run_jump`（確率約0.89）、`approach_pipe` は約0.65、`first_gap` は約0.76、`enemy_ahead` は約0.79になりました。監査の全行は [`semif-jevdash-input-audit-20260921.json`](results/semif-jevdash-input-audit-20260921.json) に保存しています。

ゲーム側の局所再生でも、固定物理では全フレーム `right_run` は x=486 付近でtimeoutし、全フレーム `right_run_jump` は647 simulation framesでclearしました。これは uv/Pygame のローカル物理確認であり、GPU推論の結果ではありません。実行runnerはこの結果をaction overrideには使わず、モデルのraw actionをそのまま適用します。

### 実L4での改良収録

監査後、compact telemetry（`grounded`、`stalled_frames`、粗い tile scan 距離、敵の pixel/contact 情報、airborne state）と、8フレーム macro の実物理を説明する候補文を使って `jev-dash-rules-v2` を収録しました。Level 1 / seed 42 / 60 FPS / 8 frames per decision、Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、SemIf `ca3ba65f142967030ecb453346e94d6f476a69df`、固定ゲーム commit `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480` です。

| 項目 | `jev-semif-jevdash-clear` の実測 |
| --- | --- |
| GPU / runtime | NVIDIA L4 / Python 3.13.15 / torch 2.10.0+cu128 |
| outcome / 最終状態 | `clear` / `has_won=true` / `is_dead=false` |
| simulation / decisions | 647 frames / 81 decisions |
| progress / score / coins | 4204.4 px / 400 / 8 |
| simulation action frames | `right_run` 16 / `right_run_jump` 631 |
| video | H.264 yuv420p / 1280×720 / 60 FPS / 767 frames |
| terminal static frames | 120 |
| inference wall / peak reserved VRAM | 18.7536 s / 8,663,334,912 bytes |
| control contract | `model_only` / raw=executed / override 0 / errors `[]` |

episode JSONの全647 frame trace、最後の `has_won=true`、MP4の全フレームdecode、開始・途中・`STAGE CLEAR!` の代表PNGを確認しました。成果物は [`results/jevdash-clear-20260921-v2/`](results/jevdash-clear-20260921-v2/) にあり、検証条件とSHA-256は同ディレクトリの `verification.json` に記録しています。

この成功は、固定Level 1 / seed 42で、モデルが少なくとも開始時のclear groundと障害付近で異なる選択をしたことを示します。一方、後半の実action framesは `right_run_jump` が大半で、複数seed・別初期状態・別levelに対する一般的なagent能力の証明ではありません。`right_run` と `right_run_jump` のどちらを選ぶかはモデル出力であり、clearを保証するhelperや実行時overrideではありません。

収録後、候補説明を固定ゲームの物理にさらに合わせ、`NOOP` の地上減速/空中速度保持、空中での右入力、grounded/coyote time時だけのjump開始、空中での再jump不可を明記した `jev-dash-rules-v3-physics` variantを追加しました。runnerの既定値とColab entry、notebookの既定値は、保存済みGPUログを再構成できる `jev-dash-rules-v2` のままです。v3は `--controller-prompt-version jev-dash-rules-v3-physics` で選べますが、追加GPU実行は行っていません。したがって、上表と同梱MP4が実証するのは明示的に `jev-dash-rules-v2` の収録であり、v3に同じ性能を帰属させません。

再現時は、Google Colab CLIの専用 `jev-semif-jevdash-clear` session（L4）で、まず `jevdash_audit_colab_entry.py` を実行してから `jevdash_colab_entry.py` を実行します。WindowsホストではWSLから、git worktree外の専用configを使います。

```powershell
$WslRepo = '/path/to/jev-colab-lab'
$Config = '/tmp/jev-semif-jevdash-clear-colab-state.json'
$Colab = 'colab'
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config new --session jev-semif-jevdash-clear --gpu L4
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config install `
  --session jev-semif-jevdash-clear -r "$WslRepo/experiments/semif/requirements-semif-jevdash-colab.txt"
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config upload `
  --session jev-semif-jevdash-clear "$WslRepo/experiments/semif/scripts/jevdash_audit.py" /content/semif-jevdash-audit.py
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config exec `
  --session jev-semif-jevdash-clear -f "$WslRepo/experiments/semif/scripts/jevdash_audit_colab_entry.py" --timeout 3600
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config upload `
  --session jev-semif-jevdash-clear "$WslRepo/experiments/semif/scripts/jevdash_play.py" /content/semif-jevdash-runner.py
wsl.exe -d Ubuntu-24.04 -- $Colab --auth=adc --config $Config exec `
  --session jev-semif-jevdash-clear -f "$WslRepo/experiments/semif/scripts/jevdash_colab_entry.py" --timeout 3600
```

実行後はJSON/動画をdownloadしてから、自分のsessionだけをstopします。notebookから同じ監査・収録を辿る導線は [`notebooks/semif_l4_experiment.ipynb`](notebooks/semif_l4_experiment.ipynb) の末尾に追加しています。

## Notebook

`notebooks/semif_l4_experiment.ipynb` は、リポジトリを Colab VM に配置して `experiments/semif` を開いた状態で、`uv sync --extra gpu --group dev --frozen`、optional media package の除去、fixture 検証、runner の順に実行します。Notebook の生成 output はコミットせず、取得した sanitized JSON を一次結果として扱います。

## 既知の境界

- 実験は SemIf の direct option-logit readout の計測であり、TypeSafe の非公開 Jev の内部構造を証明しません。
- `Qwen3.5-4B` の logits を候補文字へ制限するため、候補集合に条件付いた値です。
- 時間と VRAM は Colab の実 GPU、driver、kernel、cache 状態に依存します。
- AlexWortega/openjev の `Qwen3.5 + 3-class NLI classifier` は別実装であり、この実験の dependency・結果・主張に含めません。
