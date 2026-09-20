# Kev-0.5B Colab/T4 実験

Jared Palmer の `kev-0.5b` を、公式実装・公式アダプタを固定して Google Colab の T4 で推論する実験です。Kev は TypeSafe Jev の公開 API 形状を参考にした研究用の decision model であり、文章を生成せず、1回の prefill と pointer readout から質問ごとの確率分布を返します。本家 Jev や本番用の判断器ではありません。

## 一次情報と固定revision

- 実装: [jaredpalmer/kev](https://github.com/jaredpalmer/kev) の `v0.1.0` / `ac67bf4e52d7bdc8420d8024c396df5585915d8c`
- アダプタとhead: [jaredpalmer/kev-0.5b](https://huggingface.co/jaredpalmer/kev-0.5b) の `v0.1` が指す commit `edf1dc6d7f8d983c0adfd251e80a686e5539fc61`
- backbone: [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) の確認時点の main commit `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Colab CLI: [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)。Windowsホストから WSL Ubuntu-24.04 の CLI を使用します。

固定値は [source-manifest.json](./source-manifest.json) にも記録しています。公式 `v0.1.0` の `serve.py` と `train.py` は自動選択が MPS またはCPUのみで、CUDAを自動選択しません。今回のT4実行では、公式の `DecisionModel`、`encode`、adapter/headの読み込みをそのまま使い、ランナーから `cuda` を明示しています。したがって「公式コードがCUDA/T4対応済み」とは表現せず、「CUDA実行経路を実験ランナーで検証」と扱います。

## ファイル

- `t4_inference.py`: 公式repoをcloneし、固定revisionのbase/adaptorを取得してT4推論、計測、sanitized JSON保存を行う。
- `kev_t4_inference.ipynb`: 上記スクリプトをColabセルから再実行する薄いノートブック。
- `pyproject.toml`: ローカル検証用のuv依存設定。
- `requirements-colab.txt`: Colab VMへ `uv pip` 経由で入れる依存設定。
- `source-manifest.json`: repo、model、license、revisionの確認記録。
- `results/`: 実行JSON・実行済みnotebook。重みは保存しない。

## ローカル検証

Python環境はuvに限定します。

```powershell
uv lock --project experiments/kev
uv run --project experiments/kev python -m py_compile experiments/kev/t4_inference.py
```

公式ソースの単体テストを実行する場合は、別の一時cloneで行います。実験フォルダへ公式repo全体をコピーしません。

## Colab CLIでのT4実行

CLIは共有設定と競合しない専用state fileを使います。`--config` はディレクトリではなくsession state fileです。まず WSL でCLIを確認します。

```bash
/home/makim/.local/bin/colab --help
/home/makim/.local/bin/colab --auth adc --config /tmp/jev-kev-colab-session.json sessions
```

専用セッションを作成し、依存を入れてスクリプトをuploadします。`--gpu T4` がquotaまたはentitlementで拒否された場合は、同じタスクで無限再試行しません。

```bash
COLAB=/home/makim/.local/bin/colab
CFG=/tmp/jev-kev-colab-session.json
$COLAB --auth adc --config "$CFG" new --session jev-kev --gpu T4
$COLAB --auth adc --config "$CFG" upload experiments/kev/requirements-colab.txt /content/requirements-colab.txt
$COLAB --auth adc --config "$CFG" install --session jev-kev --requirement /content/requirements-colab.txt
$COLAB --auth adc --config "$CFG" upload experiments/kev/t4_inference.py /content/t4_inference.py
$COLAB --auth adc --config "$CFG" exec --session jev-kev --file /content/t4_inference.py --timeout 1800
$COLAB --auth adc --config "$CFG" download --session jev-kev /content/kev-t4-result.json experiments/kev/results/kev-t4-result.json
# 生のsession logにはendpoint等のmetadataが入るため、repo外へ出力し公開しない
$COLAB --auth adc --config "$CFG" log --session jev-kev --output /tmp/kev-t4-session.ipynb
$COLAB --auth adc --config "$CFG" stop --session jev-kev
```

ノートブックを実行する場合は `t4_inference.py` を `/content/` にuploadした後、`kev_t4_inference.ipynb` を `colab exec --file` で送ります。最終的には結果回収後に必ず `stop` します。

## 記録する値

結果JSONには、実GPU名・compute capability・VRAM、CUDA/Python/依存version、公式repoとHub revision、入力state・質問・選択肢、候補確率、float32精度、load/encode/初回/warmup/定常/separate時間、peak allocated/reserved VRAM、packedとseparateの確率差、エラーを保存します。`status=success` であり、かつ `hardware.name` がT4のときだけ実GPU成功として扱います。

## 実測結果（2026-09-20）

`[kev-t4-result.json](./results/kev-t4-result.json)` はT4実測、`[kev-t4-attempt1-error.json](./results/kev-t4-attempt1-error.json)` は最初の依存不整合（Colab preinstallの `torchao 0.10.0` とPEFTの要求差）の記録です。`requirements-colab.txt` に `torchao>=0.16` を追加して再実行し、成功しました。成功時は Tesla T4 / capability 7.5 / float32、load 1.968秒、初回forward 0.703秒、定常forward平均 0.0501秒、推論peak allocated 1,946.746 MiB、packed/separateの最大確率差 `4.17e-7` でした。なお、Colabの実際のTorchは `2.11.0+cu128` で、公式pyprojectの `torch<2.9` 制約から外れているため、依存versionを結果JSONに残しています。これは「公式の依存ロックを完全再現」ではなく、「現行Colab T4上で公式モデル経路を実行できた」結果です。

このモデルカードの精度・ECEは、公式の学習データ由来のheld-out splitの値です。今回の固定サポート問い合わせは動作・確率出力・計測用で、公式ベンチマークの再計測ではありません。短い学習は、推論経路がT4で成立した後に別結果として実施し、元の重みやモデルカードの値を上書きしません。
