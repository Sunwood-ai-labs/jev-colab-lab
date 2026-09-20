# JevDash 実モデル録画

固定commitの [Sunwood-ai-labs/jevdash](https://github.com/Sunwood-ai-labs/jevdash) を、Colab T4上の Jevlike TinyScorer で同期制御し、固定ゲームの録画と判断軌跡を保存する実験です。ゲームのcloneはgit外に置き、ゲーム本体は変更しません。

## 固定条件

- Game: `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480`
- Jevlike: `vinnylarouge/jevlike` / `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`
- Level: `1`
- seed: Python / NumPy / Torch `42`
- physics/video FPS: `60`
- decision interval: `8` simulation frames
- max gameplay frames: `1800`
- terminal hold: `120` static frames after death, clear, or timeout
- video target: `1280x720 MP4`
- action space: `noop`, `right`, `right_run`, `right_jump`, `right_run_jump`, `jump`, `left`

Jevlikeは事前学習済みの汎用モデルではありません。upstreamの `TinyScorer` をゼロから初期化し、`jevlike.data.write_synthetic` の合成badge選択JSONL（train 512 / validation 128 / test 128、4 epoch、seed 42）だけでColab実行時に学習し、checkpointを保存・再読込してからゲームへ接続します。ゲーム用データでの追加学習、ルール操縦、mock、live Jev API、fallbackはありません。したがって結果は、ゲーム知識を持たないJevlikeの実測失敗を含むものとして扱います。

## Colab CLI再現

WindowsホストではWSL Ubuntu-24.04の既存CLIを使います。認証済みのADCを使い、他タスクと異なるsession/configを指定します。

```bash
COLAB=/home/makim/.local/bin/colab
CFG=/tmp/jev-jevlike-jevdash-session.json
ROOT=/mnt/c/Users/makim/.codex/worktrees/7ce8/jev-colab-lab

$COLAB --auth adc --config "$CFG" sessions
$COLAB --auth adc --config "$CFG" new --session jev-jevlike-jevdash --gpu T4
$COLAB --auth adc --config "$CFG" exec --session jev-jevlike-jevdash \
  --file "$ROOT/experiments/jevlike/adapter/jevdash_colab_runner.py" --timeout 1800
$COLAB --auth adc --config "$CFG" download --session jev-jevlike-jevdash \
  /content/jevlike-jevdash-output/jevdash-jevlike-episode.json \
  /mnt/c/Prj/jev-colab-lab/.local/jevdash-videos/jevlike/jevdash-jevlike-episode.json
$COLAB --auth adc --config "$CFG" download --session jev-jevlike-jevdash \
  /content/jevlike-jevdash-output/jevdash-jevlike.mp4 \
  /mnt/c/Prj/jev-colab-lab/.local/jevdash-videos/jevlike/jevdash-jevlike.mp4
$COLAB --auth adc --config "$CFG" stop --session jev-jevlike-jevdash
```

`colab exec --file` はローカルWSLパスを読み、コードをremote kernelへ送ります。成果物回収後は必ずstopします。session logはendpoint等のmetadataを含み得るため、repoや納品ディレクトリへコピーしません。

## 記録と時間基準

各8 simulation frameの判断について、canonical observation JSON、候補7件、全確率、argmax action、forward/decision wall時間、simulation frame/timeをepisode JSONに保存します。`SIMULATION TIME (inference waits omitted)` は `frame / 60` であり、同期推論のwall-clock待ち時間を加算しません。JSONには実GPU、Torch/CUDA、checkpoint SHA256、game/model revision、学習出典、死亡/クリア/timeout、進行距離、判断回数、録画フレーム数を保存します。

HUDは `Jevlike TinyScorer` と表示し、TypeSafeのlive Jevとは表示しません。danger/urgencyはモデル出力として扱わず `NOT MEASURED` と表示します。

## 検証

納品時に次を実行し、MP4のmetadata、全フレームdecode、代表3フレームを保存します。

```bash
ffprobe -v error -show_streams -show_format -of json jevdash-jevlike.mp4
ffmpeg -v error -i jevdash-jevlike.mp4 -f null -
```

代表PNGは納品ディレクトリの `frames/` に置き、HUD・ゲーム画面・終端状態を目視確認します。
