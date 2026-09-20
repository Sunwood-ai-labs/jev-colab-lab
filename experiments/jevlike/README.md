# Jevlike tiny option-attention 実験

公式実装として [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) を採用し、`main` の確認済みコミット `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452` に固定した短時間実験です。MIT License の独立実装であり、非公開の TypeSafe Jev 本体や品質同等性を主張しません。

## 何を動かすか

上流の `TinyScorer` をそのまま使います。

- UTF-8 byte embedding と位置 embedding をスクラッチ学習
- 各 option を query にして context の key/value を attention で読む
- option ごとの logit を softmax し、候補確率を一括出力
- 合成バッジ選択データで短い学習を行い、held-out test と shuffled-context control を評価
- checkpoint 読み込み、初回推論、warmup、定常推論、peak VRAM を分離計測

公式ソースの確認記録は [`source.json`](source.json)、計測本体は [`scripts/run_experiment.py`](scripts/run_experiment.py)、Colab用は [`notebooks/jevlike_t4_experiment.ipynb`](notebooks/jevlike_t4_experiment.ipynb) です。

## ローカルCPU smoke test

Python環境とパッケージ操作は `uv` を使います。上流コードは一時ディレクトリに固定コミットで取得し、実験フォルダへコピーしません。

```powershell
cd experiments/jevlike
uv venv --python 3.11 C:\Users\makim\AppData\Local\Temp\jevlike-venv-20260920
uv pip install --python C:\Users\makim\AppData\Local\Temp\jevlike-venv-20260920\Scripts\python.exe -e .
uv pip install --python C:\Users\makim\AppData\Local\Temp\jevlike-venv-20260920\Scripts\python.exe pytest
& C:\Users\makim\AppData\Local\Temp\jevlike-venv-20260920\Scripts\python.exe -m pytest -q tests
& C:\Users\makim\AppData\Local\Temp\jevlike-venv-20260920\Scripts\python.exe scripts/run_experiment.py --device cpu --output-dir results/local-cpu --skip-install
```

`--skip-install` は、同じ仮想環境へ先に上流パッケージをインストールした場合だけ指定します。通常は省略すれば、スクリプト自身が `uv pip install` を実行します。

## Google Colab T4 実行

Colab CLI は Windows ではなく WSL の Linux 側で実行します。CLIのセッション状態はリポジトリ外の専用ファイルに置き、他タスクのセッション名や状態を使いません。

```bash
cd /mnt/c/Users/makim/.codex/worktrees/7ce8/jev-colab-lab
mkdir -p /tmp/jevlike-colab
/home/makim/.local/bin/colab --config /tmp/jevlike-colab/session.json \
  run --gpu T4 --session jev-jevlike --timeout 1800 \
  /mnt/c/Users/makim/.codex/worktrees/7ce8/jev-colab-lab/experiments/jevlike/scripts/run_experiment.py \
  --device cuda --output-dir /content/jevlike-results
```

`colab run` は成功・失敗を問わず通常はセッションを解放します。手動で `colab new` を使った場合は、結果回収後に必ず次を実行します。

```bash
/home/makim/.local/bin/colab --config /tmp/jevlike-colab/session.json stop -s jev-jevlike
```

Notebookを実行する場合は [`jevlike_t4_experiment.ipynb`](notebooks/jevlike_t4_experiment.ipynb) を Colab CLI の `exec -f` に渡します。Notebookは公開実験ブランチから同じ計測ラッパーを取得するため、ブランチを取得できない環境では上の `colab run` を使ってください。

### 実行状態

ローカルCPU smoke test と T4 実測の両方が成功済みです。T4結果は [`results/colab-t4-result.json`](results/colab-t4-result.json) に保存しています。Tesla T4 / compute capability 7.5 / float32 で、4 epoch学習、checkpoint再読込、初回推論、warmup、定常推論、peak VRAMを計測しました。

実測値は、学習 **1.782秒**、checkpoint読み込み **0.0041秒**、初回batch推論 **0.00092秒**、定常推論 **p50 0.813ms / p95 0.928ms**、peak allocated VRAM **学習28.273MiB / 推論27.418MiB** です。test top-1 は **0.8125**、shuffled-context control は **0.1953** でした。`colab-t4-blocker.json` は前回のADC未認証試行の履歴として残しています。

## 結果の読み方

スクリプトは `result.json` と `RESULT_JSON=...` を出力します。結果には以下を保存します。

- upstream URL、固定コミット、ライセンス
- GPU名、compute capability、CUDA/PyTorch/NumPy/jevlike のバージョン
- seed、データ件数、epoch、batch、モデル幅、rank、入力長
- 学習・checkpoint読み込み・初回推論・warmup・定常推論の時間
- 学習時・推論時の peak allocated/reserved VRAM
- test と shuffled-context control の top-1/top-3/ECE
- 固定質問・候補3件と各候補の確率、確率合計

GPUが割り当てられなかった場合は、CPU結果をGPU成功として扱わず、Colab CLIのエラーと割り当て条件を阻害要因として記録します。quota不足で同じGPUの再試行を繰り返しません。
