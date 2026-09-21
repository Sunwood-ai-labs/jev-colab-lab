<!--
note投稿用原稿
推奨タイトル: Jevクローン5種をGoogle Colabで検証した記録 (24文字)
アイキャッチ画像: colab_jev_banner.jpg
-->

更新日: 2026-09-21
対象リポジトリ: [Sunwood-ai-labs/jev-colab-lab](https://github.com/Sunwood-ai-labs/jev-colab-lab)

この記事は、Decision Model（選択肢の中から1つを決定するモデル）の公開実装を、Google ColabのGPU環境で実際に動かして検証した記録です。対象としたのは **Laya、Kev-0.5B、Jevlike、SemIf、OpenJev NLI** の5経路です。いずれも非公開である本家Jev（TypeSafe社）の再現ではなく、目的・入力形式・学習内容・確率の算出構造がそれぞれ異なる独立した実験です。

実測の結果、通常の選択推論については5件すべてで実GPU上の検証ログを正常に回収できました。一方で、推論の仕組み自体がモデルごとに根本から異なるため、測定された数値を1つの順位表（leaderboard）として単純比較することはできません。また、ゲーム接続検証（JevDash）については予備的な観測にとどまるため、次回の記事で独立して詳しく扱います。

## 先に結論

- **GPU実測**: Laya・Kev・JevlikeはColab T4、SemIf・OpenJev NLIはColab L4で動作を確認しました。結果JSONに記録された実行ステータスだけでなく、実際に割り当てられたGPU型番と稼働環境まで点検しています。
- **通常の選択推論**: 選択肢ごとの確率分布、NLIの含意スコア、モデルロード時間、初回推論時間、ウォームアップ、定常推論のレイテンシ、ピークVRAMを、各ランナーの測定基準に沿って記録しました。
- **比較の前提**: 表面上は同じ「選択」に見えても、Laya・Kev・Jevlikeは専用のdecision head、SemIfは候補文字に対する次トークンlogit、OpenJev NLIは3クラス分類モデルです。出力の意味が異なるため、横並びの性能比較には適しません。
- **ゲーム検証の扱い**: ゲーム環境（JevDash）を用いた制御実験については、Jevlikeにおける入力の切り詰め不具合の調査を含め、次回の記事で単独のテーマとして報告します。

## 1. 検証した5つのモデル

5つのモデルは、それぞれ設計方針も入力表現も異なります。表を用いず、各モデルの特徴と推論の形式を整理します。

### Laya (ModernBERT)
- **実行GPU**: Colab T4
- **推論の形**: 1回のdecision呼び出しで、選択肢（choice）・スコア（score）・拒絶（noul）を同時に出力
- **入力と学習の範囲**: ModernBERT-largeをベースとする公開英語チェックポイントを使用。今回のテストケースはサポート窓口の問い合わせ分類であり、公式ベンチマークの再計測ではありません。
- **リポジトリ**: [experiments/laya](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/laya)

### Kev-0.5B
- **実行GPU**: Colab T4
- **推論の形**: packed（連結）形式およびseparate（個別）形式によるポインタ読み出し
- **入力と学習の範囲**: 公式decision modelとadapterを使用。動作確認、確率の妥当性、計測境界の検証を目的としています。
- **リポジトリ**: [experiments/kev](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/kev)

### Jevlike
- **実行GPU**: Colab T4
- **推論の形**: TinyScorerによる選択肢のsoftmax計算
- **入力と学習の範囲**: バイトおよび位置の埋め込み表現を初期化し、合成されたバッジ選択データのみで4エポック学習。ゲーム制御の学習は含まれません。
- **リポジトリ**: [experiments/jevlike](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/jevlike)

### SemIf (Qwen3.5-4B)
- **実行GPU**: Colab L4
- **推論の形**: 事前に定義した選択肢（A/B/Cなど）に対応する単一トークンのlogitを抽出し、候補内のみでsoftmaxを計算
- **入力と学習の範囲**: 次トークンの直接読み出しを採用しており、文章生成やNLI headは利用していません。
- **リポジトリ**: [experiments/semif](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/semif)

### OpenJev NLI (4B)
- **実行GPU**: Colab L4
- **推論の形**: 前提と仮説のペアに対し、含意（entailment）・矛盾（contradiction）・中立（neutral）の3クラスに分類
- **入力と学習の範囲**: 初版の4B NLIチェックポイントを使用。5問の合成テストケースに対して含意確率が最大となる選択肢を選びます。
- **リポジトリ**: [experiments/openjev-nli](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/openjev-nli)

## 2. 通常推論の実測値

各モデルの推論結果と測定値を以下に示します。ここで得られた確率やレイテンシは、特定のテストケースと読み出し手法に限定された観測値です。モデル全体の汎化性能や、校正された信頼度を示すものではありません。

### Laya / Colab T4
- **推論結果**: 部署分類タスクにおいて `department=billing` を選択（確率: 0.9254）
- **レイテンシ**: 単一推論の定常平均 36.17 ms（20回試行）、モデルロード時間 22.96 秒
- **制限事項**: 公開されている英語チェックポイントによる合成サポートデータでの測定値です。出力された確率値を一般的な確信度として再解釈することはできません。
- **検証ログ**: [laya-t4-result.json](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/results/laya-t4-result.json)

### Kev-0.5B / Colab T4
- **推論結果**: `department=returns` を選択（確率: 0.85）
- **レイテンシ**: 定常順伝播の平均 50.12 ms（10回試行）、packed形式とseparate形式の出力差は最大 4.17e-7
- **制限事項**: Google ColabのPyTorch環境が公式リポジトリの指定範囲と一部異なるため、完全な依存ロック環境ではなく、固定モデルをT4で実行した際の測定値として扱います。
- **検証ログ**: [kev-t4-result.json](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/results/kev-t4-result.json)

### Jevlike / Colab T4
- **推論結果**: 合成タスクにおいて `amber badger` を選択（確率: 0.5839、テスト正答率: 81.25%、文脈シャッフル対照群: 19.53%）
- **レイテンシ**: 定常中央値 0.81 ms（20回試行）、4エポックの学習時間 1.37 秒
- **制限事項**: わずか512件の合成データセットによる小規模実験の数値です。プロセス起動時間を含めたコールドスタートの測定値ではありません。
- **検証ログ**: [colab-t4-result.json](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/results/colab-t4-result.json)

### SemIf / Colab L4
- **推論結果**: ルーティング課題において `route-1=account_access` を選択（条件付き確率: 0.99998）
- **レイテンシ**: モデルロード時間 55.76 秒、定常推論の平均 94.42 ms
- **制限事項**: あらかじめ絞り込んだ候補トークンの中でのみ計算した相対スコアであり、モデル全体の出力確率ではありません。
- **検証ログ**: [semif-l4-result-20260921.json](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/results/semif-l4-result-20260921.json)

### OpenJev NLI 4B / Colab L4
- **推論結果**: 5問の合成テストケースすべてにおいて正解の仮説を含意と判定（正答率: 5/5）
- **レイテンシ**: モデルロード時間 51.83 秒、15ペアのバッチ定常推論平均 255.31 ms
- **制限事項**: 含意確率が最大となる選択肢を採用するNLI分類です。5問の固定テストケースに対する結果であり、標準的なNLIベンチマークの網羅スコアではありません。
- **検証ログ**: [colab-l4.json](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/results/colab-l4.json)

## 3. 5冊のColabノートブック

各実験はGoogle Colabのノートブックとして整理されています。コードを読む場合はGitHubリンク、実際にブラウザ上で起動する場合は「Colabで開く」リンクを利用できます。ノートブックは薄いエントリーポイントとして設計されており、実行には同フォルダのランナーや設定ファイルも必要となります。

### Laya / Colab T4
- [GitHubでコードを見る](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/notebooks/laya_t4_benchmark.ipynb)
- [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/notebooks/laya_t4_benchmark.ipynb)
- **実行手順**: `benchmark_laya.py` を環境へ配置し、依存パッケージを導入したのちにノートブックを実行します。単一・バッチ推論のレイテンシとVRAMを計測します。

### Kev-0.5B / Colab T4
- [GitHubでコードを見る](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/kev_t4_inference.ipynb)
- [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/kev_t4_inference.ipynb)
- **実行手順**: `t4_inference.py` を配置し、packed形式とseparate形式の推論結果の整合性と実行速度を測定します。

### Jevlike / Colab T4
- [GitHubでコードを見る](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/notebooks/jevlike_t4_experiment.ipynb)
- [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/notebooks/jevlike_t4_experiment.ipynb)
- **実行手順**: 固定リビジョンのランナーを取得し、`uv` を用いて上流コードを導入したうえで、合成データを用いた短時間学習と推論を行います。

### SemIf / Colab L4
- [GitHubでコードを見る](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/notebooks/semif_l4_experiment.ipynb)
- [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/notebooks/semif_l4_experiment.ipynb)
- **実行手順**: リポジトリをColab VM内に配置し、`experiments/semif` を作業ディレクトリとして `uv sync` を実行後、テストケースの推論を実行します。

### OpenJev NLI 4B / Colab L4
- [GitHubでコードを見る](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/notebooks/openjev_nli_l4.ipynb)
- [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/notebooks/openjev_nli_l4.ipynb)
- **実行手順**: `experiments/openjev-nli` 配下で `uv` による依存環境を準備し、5問の合成テストケースに対する3クラス分類を実行します。

## 4. Windows環境からWSLでColab CLIを実行する

公式の [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) は Linux および macOS 向けに提供されており、Windows ネイティブ環境には対応していません。そのため、Windows ホストでは WSL（Ubuntu）を介して CLI を実行します。

WSL 内での基本的なセットアップは以下のとおりです。

```bash
uv tool install google-colab-cli
colab --help
colab version
```

独立したセッションを管理する場合は、実験ごとにセッション名と状態ファイル（state file）を分離します。

```bash
CFG=/tmp/jev-laya-colab-state.json
colab --config "$CFG" new --session jev-laya --gpu T4
# 各実験ディレクトリに用意したスクリプトを実行
colab --config "$CFG" stop --session jev-laya
```

セッション運用にあたっては、以下の規律を設けています。

- セッション名は `jev-<slug>` のように固有の名称とし、他の実験のランタイムを再利用しない。
- `--config` で指定する状態ファイルは Git 管理外に配置し、認証情報や接続情報をリポジトリに含めない。
- 結果 JSON を手元へ取得・確認したのち、自身が起動したセッションを必ず停止する。
- クォータ制限や認証エラーが発生した場合は、過度な再試行を行わず、制約事項として記録にとどめる。

## 5. 依存関係と作業ツリーを分離した理由

各実験の Python 依存パッケージは、ルートディレクトリではなく `experiments/<slug>/pyproject.toml` とロックファイルに閉じ込めて管理しています。また、Git の worktree 機能を活用し、実験ごとに独立したブランチと作業ツリーを展開しました。

この構造を採用したのは、単にコードを整理するためではなく、以下の混同を排除するためです。

- ローカル CPU での単体動作確認を、Colab GPU での推論成功と誤認しない。
- クラウドへ要求した GPU 型番と、実際にインスタンス内で認識された GPU 型番を厳密に区別する。
- モデルの初期ロード、初回推論、ウォームアップ、定常推論、ピーク VRAM といった、測定区間の異なる数値を無秩序に比較しない。
- 特定テストケースの正答率を、モデル全体の汎化性能として一般化しない。

## 6. 上流リポジトリとライセンスの境界

実験に用いたコードとモデルの固定リビジョンは、[NOTICE.md](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/NOTICE.md) および各マニフェストに記録しています。

- **Laya**: [実装 commit d113dca](https://github.com/NandhaKishorM/laya/tree/d113dca2512fb3eaca313534bc54c7162d87c1d4) / [チェックポイント revision 1c5edc](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982)（Apache-2.0）
- **Kev-0.5B**: [実装 commit ac67bf4](https://github.com/jaredpalmer/kev/tree/ac67bf4e52d7bdc8420d8024c396df5585915d8c) / [アダプター revision edf1dc](https://huggingface.co/jaredpalmer/kev-0.5b/tree/edf1dc6d7f8d983c0adfd251e80a686e5539fc61) / [ベースモデル Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987)（Apache-2.0）
- **Jevlike**: [実装 commit 94f5fd1](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452)（MIT）
- **SemIf**: [実装 commit ca3ba65](https://github.com/TheoLeeCJ/semif/tree/ca3ba65f142967030ecb453346e94d6f476a69df)（MIT） / [ベースモデル Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)（Apache-2.0）
- **OpenJev NLI**: [4B初版 revision b32265](https://huggingface.co/AlexWortega/openjev/tree/b32265f4700df7c02532933c9a4ff258a449d7ac)（モデルカード記載MIT / ベースモデルApache-2.0）

このように実装リポジトリもライセンス体系も異なるため、これらを「同一のモデル群」としてひとくくりに扱うことはできません。

## 7. 次回予告：JevDashによるゲーム制御検証

本プロジェクトでは、2Dアクションゲーム [jevdash](https://github.com/Sunwood-ai-labs/jevdash) を用いた同期推論テストも実施しました。

ただし、このゲーム接続テストは能力ランキングを算出するためのものではありません。予備検証の段階で、例えば Jevlike においては入力バイト列の先頭切り詰め仕様（`context_tokens=192`）に起因し、肝心のゲーム状態情報がモデルへ到達していなかった事実が判明しています。

ゲーム環境への接続方法、各モデルが選択した行動ログ、および入力経路の具体的な不具合については、**次回の独立した記事**で詳細に報告します。

## 8. 関連リンク

本実験の詳細な実装や環境構築手順については、以下のドキュメントを参照してください。

- [実験一覧ドキュメント](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/docs/ja/guide/experiments.md)
- [再現性ガイド](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/docs/ja/guide/reproducibility.md)
- [出典とライセンス一覧](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/docs/ja/guide/sources-and-licenses.md)
- [リポジトリトップ (GitHub)](https://github.com/Sunwood-ai-labs/jev-colab-lab)
