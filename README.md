# Jev Google Colab Lab

Jev系のDecision ModelをGoogle Colabで実行し、候補の確率を直接返す使い方と、通常のLLMによる文章生成との違いを検証する実験用フォルダ。

## 状態

2026-09-21: Laya / Kev / Jevlike はColab T4、SemIf / OpenJev NLI 4B はColab L4で実測を完了し、全5実験の成果物をmainへ統合済み。各実験は専用git worktreeとブランチ、uv環境で実施。

公開先: https://github.com/Sunwood-ai-labs/jev-colab-lab

実験成果物は `experiments/<slug>/` に保存し、各タスクで一区切りごとに検証・commit・push・mainへのマージまで行う。未統合の進行中成果物は各ブランチを参照。元の会話原文はローカル専用で公開対象外。

| 実験 | 実GPU検証 | 成果物 |
| --- | --- | --- |
| Laya | T4 成功 | [手順と実測](experiments/laya/README.md) |
| Kev-0.5B | T4 成功 | [手順と実測](experiments/kev/README.md) |
| SemIf 4B | L4 成功 | [手順と実測](experiments/semif/README.md) |
| OpenJev NLI 4B | L4 成功 | [手順と実測](experiments/openjev-nli/README.md) |
| Jevlike | T4 成功（短い学習・推論） | [手順と実測](experiments/jevlike/README.md) |

各実験の入力・精度・計測条件は異なり、速度の直接比較や汎化性能の評価ではない。統合時の軽量テストは計10件通過（Laya 3 / OpenJev NLI 3 / SemIf 2 / Jevlike 2）。GPU結果は各ディレクトリの保存済み実測JSONを参照。

Google Colab CLIはWSL Ubuntu-24.04の既存環境を使用する。Pythonはuvを使う。開発は並列、GPU実行の同時数はColabの利用枠に従う。詳しい作業規約はAGENTS.md、対象一覧はexperiments/README.mdを参照。

## 実験の順番（元資料に基づく暫定計画）

1. Laya
2. Kev-0.5B
3. Jevlike
4. SemIf / OpenJev NLI（別々の実装として扱う）
5. Bespoke Nimble
6. DiffusionGemma系

各実装の公式リポジトリ、ライセンス、利用可能な重み、依存関係、GPU要件を一次情報で確認してから実行する。元資料の性能値・VRAM目安・アーキテクチャ情報は、この作業では未検証。

## 比較したいこと

- 同じ入力・質問・選択肢に対する回答と候補確率
- 正解付き共通データでの正答率。確率の校正は正答率とは分けて評価
- モデル読み込み時間、初回推論、ウォームアップ後の推論時間
- GPUの種類、精度・量子化、入力長、バッチサイズごとのピークVRAM
- 単一質問と複数質問の挙動
- 通常のLLMによる回答生成との出力形式・処理時間・扱いやすさの違い

## フォルダ

| パス | 用途 |
| --- | --- |
| references/original-discussion.txt | ユーザー提供の調査・会話原文 |
| notebooks/ | Colab用ノートブック |
| data/ | 共通の入力・質問・選択肢・正解データ |
| results/ | 実行ログ、計測結果、出力例 |
| scripts/ | 実行・計測・集計の補助スクリプト |

## 最初の実験

Layaの公式情報と実行条件を確認し、利用可能なColab GPU上で最小の推論例を動かす。入力、質問、選択肢、出力、所要時間、ピークVRAM、環境情報を保存する。実行できない場合もエラーと条件を記録する。

各実行では日時、実装名、リポジトリURLとcommit、モデルIDとrevision、GPU、依存バージョン、精度、入力条件、ウォームアップ・反復回数を記録し、結果の再現性を確保する。
