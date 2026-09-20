# Jevlike tiny option-attention 実験

公式実装として [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) を採用し、`main` の確認済みコミット `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452` に固定した短時間実験です。MIT License の独立実装であり、非公開の TypeSafe Jev 本体や品質同等性を主張しません。

## 何を動かすか

上流の `TinyScorer` をそのまま使います。

- UTF-8 byte embedding と位置 embedding をスクラッチ学習
- 各 option を query にして context の key/value を attention で読む
- option ごとの logit を softmax し、候補確率を一括出力
- 合成バッジ選択データで短い学習を行い、held-out test と shuffled-context control を評価
- checkpoint 読み込み直後の first post-load 推論、warmup、定常推論、peak VRAM を分離計測

公式ソースの確認記録は [`source.json`](source.json)、計測本体は [`scripts/run_experiment.py`](scripts/run_experiment.py)、Colab用は [`notebooks/jevlike_t4_experiment.ipynb`](notebooks/jevlike_t4_experiment.ipynb) です。

## ローカルCPU smoke test

Python環境とパッケージ操作は `uv` を使います。上流コードは一時ディレクトリに固定コミットで取得し、実験フォルダへコピーしません。

```powershell
cd experiments/jevlike
$venv = Join-Path $env:TEMP 'jevlike-venv'
$venvPython = Join-Path $venv 'Scripts/python.exe'
uv venv --python 3.11 $venv
uv pip install --python $venvPython -e .
uv pip install --python $venvPython pytest
& $venvPython -m pytest -q tests
& $venvPython scripts/run_experiment.py --device cpu --output-dir results/local-cpu --skip-install
```

`--skip-install` は、同じ仮想環境へ先に上流パッケージをインストールした場合だけ指定します。通常は省略すれば、スクリプト自身が `uv pip install` を実行します。

## Google Colab T4 実行

Colab CLI は Windows ではなく WSL の Linux 側で実行します。CLIのセッション状態はリポジトリ外の専用ファイルに置き、他タスクのセッション名や状態を使いません。

```bash
cd /path/to/jev-colab-lab
mkdir -p /tmp/jevlike-colab
colab --auth adc --config /tmp/jevlike-colab/session.json \
  run --gpu T4 --session jev-jevlike --timeout 1800 \
  /path/to/jev-colab-lab/experiments/jevlike/scripts/run_experiment.py \
  --device cuda --output-dir /content/jevlike-results
```

`colab run` は成功・失敗を問わず通常はセッションを解放します。手動で `colab new` を使った場合は、結果回収後に必ず次を実行します。

```bash
colab --auth adc --config /tmp/jevlike-colab/session.json stop -s jev-jevlike
```

Notebookを実行する場合は [`jevlike_t4_experiment.ipynb`](notebooks/jevlike_t4_experiment.ipynb) を Colab CLI の `exec -f` に渡します。Notebookは同じ計測ラッパーをimmutableなgit commitから取得します。ネットワークやcommit取得ができない環境では上の `colab run` を使ってください。

### 実行状態

ローカルCPU smoke test と T4 実測の両方が成功済みです。T4結果は [`results/colab-t4-result.json`](results/colab-t4-result.json) に保存しています。Tesla T4 / compute capability 7.5 / float32 で、4 epoch学習、checkpoint再読込、初回推論、warmup、定常推論、peak VRAMを計測しました。

実測値は [`results/colab-t4-result.json`](results/colab-t4-result.json) に保存しています。学習、checkpoint再読込後の初回batch、warmup、定常推論、peak VRAM、test top-1、shuffled-context controlを、測定境界とともに記録しています。`first_post_load_batch_inference_seconds` は同一プロセス内のcheckpoint再読込後に測る値であり、Python起動・import・CUDA context初期化・依存インストールを含むprocess-cold測定ではありません。実験間の初回推論比較には使わないでください。`colab-t4-blocker.json` は前回のADC未認証試行の履歴として残しています。

## 結果の読み方

スクリプトは `result.json` と `RESULT_JSON=...` を出力します。結果には以下を保存します。

- upstream URL、固定コミット、ライセンス
- GPU名、compute capability、CUDA/PyTorch/NumPy/jevlike のバージョン
- seed、データ件数、epoch、batch、モデル幅、rank、入力長
- 学習・checkpoint読み込み・first post-load推論・warmup・定常推論の時間と測定条件
- 学習時・推論時の peak allocated/reserved VRAM
- test と shuffled-context control の top-1/top-3/ECE
- 固定質問・候補3件と各候補の確率、確率合計

GPUが割り当てられなかった場合は、CPU結果をGPU成功として扱わず、Colab CLIのエラーと割り当て条件を阻害要因として記録します。quota不足で同じGPUの再試行を繰り返しません。
