# Laya / ModernBERT のColab T4実験

Jev-likeな非自己回帰Decision ModelであるLayaを、Google ColabのT4で推論し、出力確率・レイテンシ・VRAMを再現可能な条件で記録する独立実験です。実験対象は公式リポジトリの汎用英語checkpointです。

## 固定した一次情報

- 実装: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) の `d113dca2512fb3eaca313534bc54c7162d87c1d4`
- 重み: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) の revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`
- パッケージ: `laya==0.3.4`
- backbone: ModernBERT-large、421M、英語、context 512
- ライセンス: Apache-2.0

`laya-typed-decisions` は4つの特定ワークフローに追加学習された別checkpointであり、汎用推論との混同を避けるため今回の主計測には使いません。原資料は未検証の参考情報であり、公開成果物には含めていません。

## 成果物

| パス | 内容 |
| --- | --- |
| `scripts/benchmark_laya.py` | pinned revisionの取得、推論、速度/VRAM計測、sanitized JSON保存 |
| `notebooks/laya_t4_benchmark.ipynb` | Colab CLIで実行する最小ノートブック |
| `tests/test_benchmark_laya.py` | モデル不要の入力・集計ヘルパーテスト |
| `references/verified-sources.json` | 一次情報と固定revisionの記録 |
| `results/` | 実行後に取得するJSON/ログ。重みは保存しない |

## ローカル検証

Python環境とコマンドはuvで管理します。

```powershell
uv sync --project experiments/laya --group dev
uv run --project experiments/laya pytest
uv run --project experiments/laya python experiments/laya/scripts/benchmark_laya.py --help
```

ローカルCPUでモデルを動かす場合は、GPU成功と混同しないよう `--device cpu` を明示し、結果はCPU smoke testとして扱ってください。実測の主結果はColab T4だけです。

## Colab CLIでのT4実測

WindowsホストからWSLの公式Colab CLIを使います。`--config` は他タスクと共有しない、git外のsession stateファイルです。セッション名も専用の `jev-laya` に固定します。

```powershell
$wslRepo = "/path/to/jev-colab-lab"
$wslScript = "$wslRepo/experiments/laya/scripts/benchmark_laya.py"
$wslResult = "$wslRepo/experiments/laya/results/laya-t4-result.json"
$colab = "colab"

wsl.exe -d Ubuntu-24.04 -- $colab --config /tmp/jev-laya-colab-session.json new --session jev-laya --gpu T4
wsl.exe -d Ubuntu-24.04 -- $colab --config /tmp/jev-laya-colab-session.json install --session jev-laya laya==0.3.4 'huggingface-hub>=0.20,<2' 'safetensors>=0.4,<1' 'transformers>=4.45,<6'
wsl.exe -d Ubuntu-24.04 -- $colab --config /tmp/jev-laya-colab-session.json exec --session jev-laya --file $wslScript --timeout 1800
wsl.exe -d Ubuntu-24.04 -- $colab --config /tmp/jev-laya-colab-session.json download --session jev-laya /content/laya-t4-result.json $wslResult
wsl.exe -d Ubuntu-24.04 -- $colab --config /tmp/jev-laya-colab-session.json stop --session jev-laya
```

`benchmark_laya.py` は `laya==0.3.4` などの依存がColab VMに存在することを前提にしています。quota不足や認証失敗で `new` が作れない場合は、再試行を繰り返さず、その条件を結果として記録します。

ノートブックを実行する場合は、先に `colab upload` で `benchmark_laya.py` を `/content/benchmark_laya.py` に送り、`notebooks/laya_t4_benchmark.ipynb` を `colab exec --file` で実行します。実行後は必ずJSONをダウンロードしてから `colab stop` で自分のセッションを解放します。

## 計測の読み方

出力JSONは次を分離します。

- `snapshot_download`: Hugging Faceからのモデル取得/cache時間
- `agent_load`: tokenizer・モデル構築とGPU配置
- `scenarios.single` / `scenarios.batch`: 1問と4問同時の初回、ウォームアップ、定常反復
- `vram.load_peak_mb` と各scenarioの `steady_peak_mb`: GPU名・CUDA版・モデル精度と併記したピーク値
- `output`: 選択肢、scoreの段階、noul確率を含むLayaの実出力

`--require-cuda` を指定するとGPUに配置できなかった場合はエラーJSONを残して非ゼロ終了します。CPU smoke test、quota不足、認証失敗をColab T4成功として報告しません。

## JevDash実モデル録画

`adapter/runner.py` は、`Sunwood-ai-labs/jevdash` の固定commit `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480` を変更せずに参照し、Level 1をseed 42で実行します。LayaはCUDA上で同期的に8シミュレーションフレームごとに1回だけ呼び、7候補のargmax actionだけをゲームへ渡します。モック、live Jev、ルールfallback、成功プレイの選別はありません。

Layaへ渡すstateは固定ゲームの `JevObservation.model_dump()` 全体です。各判断のJSONに観測、候補、全確率、入力token数、512-token context内の保持token数、truncation有無、推論時間を記録します。danger/urgencyはLayaのaction-only questionでは計測せず、HUD/JSONでも `UNMEASURED` と表示します。

動画はゲームの60 FPSシミュレーションフレームを録画します。Laya推論待ちを含むwall-clock実時間ではありません。死亡・クリア・1800フレーム上限の後に120フレームの静止終端を追加します。録画時の画面にはモデル名、GPU、`MODEL DECISION`、`SIM TIME ... inference waits omitted` を表示します。

実行は必ず独立したColab sessionで行い、ゲームclone・runner・動画・JSONをVM内で完結させてから回収します。session stateはgit外の専用ファイルにしてください。

```powershell
$cfg = "/tmp/jev-laya-jevdash-session.json"
$wslRepo = "/path/to/jev-colab-lab"
$runner = "$wslRepo/experiments/laya/adapter/runner.py"
$colab = "colab"

wsl.exe -d Ubuntu-24.04 -- $colab --auth adc --config $cfg new --session jev-laya-jevdash --gpu T4
# Upload the fixed game source archive and adapter files, then install with uv.
wsl.exe -d Ubuntu-24.04 -- $colab --auth adc --config $cfg install --session jev-laya-jevdash laya==0.3.4 pygame pydantic
# Execute the uploaded runner with SDL_VIDEODRIVER=dummy and --video/--json under /content.
# Download all files before stopping only this session.
wsl.exe -d Ubuntu-24.04 -- $colab --auth adc --config $cfg stop --session jev-laya-jevdash
```

FFmpegの `ffprobe`、全フレームdecode、代表3フレームの目視確認を実行結果に添えます。重み・認証情報・session metadataは成果物へコピーしません。
