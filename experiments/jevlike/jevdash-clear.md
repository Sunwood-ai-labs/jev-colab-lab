# JevDash clear audit

この実験は、固定した JevDash Level 1 と upstream Jevlike TinyScorer を同じ入力・候補順・物理順序で比較するためのものです。実行コードは [jevdash_clear_colab_runner.py](adapter/jevdash_clear_colab_runner.py) です。

## 検証済みの基準実行

`r2-game-teacher` が検証済み T4 実行を再現する既定 variant です。教師追従 rollout のデータ分布を意図的に保持しており、T4 の実測では以下でした。

- GPU: Tesla T4、device: `cuda`
- game commit: `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480`
- Jevlike commit: `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`
- model-only: `has_won=true`, 647 simulation frames, 4204.4 px
- assisted: `has_won=true`, 647 simulation frames, guard trigger 56 physics frames、実 action override 0 frames
- model-only の raw action は全判断で `right_run_jump`。従って clear は再現できるが、状態条件を学習した成功とは主張しない

成果物は現在の worktree の `experiments/jevlike/results/jevdash-clear/` に保存しています。`*-model-only.{json,mp4}`、`*-assisted.{json,mp4}`、`*-verification.json` はローカル検証成果物として gitignore 対象です。

公開可能な r2 の動画・sanitized summary・verification は [公開証拠フォルダ](results/jevdash-clear-r2-public/README.md) に保存しています。ここには絶対パス、session metadata、credentials、raw full log、weightsを含めません。

## variant の選択

未検証 variant を既定値に置き換えないため、`r2-game-teacher` を既定にしています。

```powershell
uv run --no-project --python experiments/jevlike/.venv/Scripts/python.exe `
  experiments/jevlike/adapter/jevdash_clear_colab_runner.py `
  --training-variant r2-game-teacher
```

`balanced-probe` は、固定 Level 上の grounded probe と短い real physics arc を使って、各 split を `right_run`/`right_run_jump` の 50:50 にする改善案です。ローカル smoke では 100/100、compact context 最大177 bytesを確認していますが、Colab T4 での追加実行は `TooManyAssignmentsError / Precondition Failed` で割り当てられず、GPU成功とは扱いません。

```powershell
uv run --no-project --python experiments/jevlike/.venv/Scripts/python.exe `
  experiments/jevlike/adapter/jevdash_clear_colab_runner.py `
  --training-variant balanced-probe
```

## 入力監査と安全層

旧 prompt は最初の192 UTF-8 bytesが固定 prefix になり、変化するゲーム状態を TinyScorer に届けていませんでした。新しい state-first compact context は監査時177 bytes以内で、旧 prefix count 1 に対して30以上の状態 prefixを持ちます。`T` は粗い telemetry scan-column 距離、`X` は player の右端から固定 tile 左端までの物理 pixel clearanceです。grounded、airborne、敵の vertical offset、stalled も保存します。

model-only は safety reflex を無効にします。assisted は raw action、executed action、guard trigger reason、実際の raw→executed変更を別々に保存し、guard が同じ actionを返した場合を override と数えません。

ローカル audit は GPU結果ではありません。

```powershell
uv run --no-project --python experiments/jevlike/.venv/Scripts/python.exe `
  experiments/jevlike/adapter/jevdash_clear_colab_runner.py --audit-only `
  --game-root C:\path\to\jevdash-fixed `
  --jevlike-root C:\path\to\jevlike `
  --output experiments/jevlike/results/jevdash-clear/audit.json
```
