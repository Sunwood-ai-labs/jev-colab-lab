<div align="center">
  <img src="https://raw.githubusercontent.com/Sunwood-ai-labs/jev-colab-lab/main/docs/public/jev-colab-lab-icon.svg" alt="Jev Colab Lab アイコン" width="96">
  <h1>Jev Colab Lab</h1>
  <p>Decision Model推論をGoogle Colab GPUで再現計測する実験リポジトリ</p>
</div>

<p align="center">
  <a href="https://sunwood-ai-labs.github.io/jev-colab-lab/ja/">ドキュメント</a> ·
  <a href="https://github.com/Sunwood-ai-labs/jev-colab-lab">リポジトリ</a>
</p>

<p align="center">
  <a href="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/public-qa.yml"><img src="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/public-qa.yml/badge.svg" alt="Public QA"></a>
  <a href="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/docs.yml"><img src="https://github.com/Sunwood-ai-labs/jev-colab-lab/actions/workflows/docs.yml/badge.svg" alt="Docs build"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-0b1020.svg" alt="MIT license"></a>
</p>

<p align="center"><a href="README.md">English</a> · <strong>日本語</strong></p>

## 🔭 このリポジトリについて

このリポジトリは、Jev系Decision Modelの推論経路をGoogle Colab GPUで小さく再現計測し、runner、notebook、固定した上流revision、sanitizedな入力、結果JSONを <code>experiments/&lt;slug&gt;/</code> にまとめる研究記録です。

現在は、Laya、Kev-0.5B、Jevlike tiny option-attention、SemIf 4B、OpenJev NLI 4Bの5実験をmainへ統合しています。モデル経路とfixtureが異なるため、確率・精度・速度・VRAMを共通ベンチマークとして直接比較しません。

## 🚀 最短で確認する

ローカルの確認はモデル重みをダウンロードしないmodel-freeテストです。

~~~powershell
git clone https://github.com/Sunwood-ai-labs/jev-colab-lab.git
Set-Location jev-colab-lab

# uvで補助関数・fixtureのテストを実行
uv run --no-project --with pytest pytest experiments/laya/tests experiments/openjev-nli/tests experiments/semif/tests -q

# Jevlikeの回帰テストは宣言済みtorch依存を含む環境で実行
uv run --project experiments/jevlike --extra dev pytest experiments/jevlike/tests -q

# Kev runnerの構文だけを確認（モデル依存はインストールしない）
uv run --no-project python -m py_compile experiments/kev/t4_inference.py
~~~

モデルを使うローカルsmoke testは各実験READMEの <code>uv</code> プロジェクトに従ってください。ローカルCPUの結果をColab GPU実測として扱いません。

## 🧪 実験カタログ

| 実験 | ターゲット | 計測対象 | 証跡 |
| --- | --- | --- | --- |
| [Laya / ModernBERT](experiments/laya/README.md) | Colab T4 | choice・score・noulの出力 | [runner](experiments/laya/scripts/benchmark_laya.py) · [T4 JSON](experiments/laya/results/laya-t4-result.json) |
| [Kev-0.5B](experiments/kev/README.md) | Colab T4 | packed/separate確率とレイテンシ | [runner](experiments/kev/t4_inference.py) · [T4 JSON](experiments/kev/results/kev-t4-result.json) |
| [Jevlike](experiments/jevlike/README.md) | Colab T4 | 短時間学習・held-out test・shuffled-context control | [runner](experiments/jevlike/scripts/run_experiment.py) · [T4 JSON](experiments/jevlike/results/colab-t4-result.json) |
| [SemIf / Qwen3.5-4B](experiments/semif/README.md) | Colab L4 | 次トークンlogitの直接option readout | [runner](experiments/semif/scripts/run_experiment.py) · [L4 JSON](experiments/semif/results/semif-l4-result-20260921.json) |
| [OpenJev NLI 4B](experiments/openjev-nli/README.md) | Colab L4 | entailment・contradiction・neutral | [runner](experiments/openjev-nli/scripts/measure_openjev.py) · [L4 JSON](experiments/openjev-nli/results/colab-l4.json) |

コミット済みJSONを各実験のcanonicalな証跡とします。Jevlikeのtiming fieldはresult schemaに計測境界を残していますが、このREADMEでは実験間比較の値として扱いません。

## 🎮 JevDash録画セット

5つのモデル経路でJevDash Level 1の録画セットをGit外に保存しています。これは単一episodeの観測記録であり、ゲーム能力の正常な比較評価が完了したことを意味しません。特にJevlikeは入力先頭の切り詰めで状態情報が欠落しており、ゲーム結果を能力評価に使えません。MP4本体はこのリポジトリに公開URLがなく、意図的にGitへ入れていません。

これはモデルランキングではなく、独立した単一episodeの実演です。動画時間は同期推論の待ち時間を除いたsimulation timeです。KevとJevlikeのpresentation replayは、記録済みのモデル軌跡と状態を維持し、HUD描画だけを補正しています。詳しい数値とJevlikeの不備は[日本語まとめ記事](docs/ja/guide/jev-clone-colab.md)にまとめています。

5実験の公開導線は[録画証跡インデックス](experiments/README.md#jevdash-capture-evidence)から確認できます。

## 🧭 結果の読み方

- <code>status=success</code> または <code>status=ok</code> は、結果JSONに実GPU名とruntime情報が記録されている場合に限って実測成功と読みます。
- モデル読み込み、初回推論、warmup、steady-state、peak VRAMは、runnerが分離記録している場合だけ別の値として扱います。
- 条件付きoption scoreは自動的に校正済み信頼度になるわけではなく、fixtureの正答率も汎化性能ではありません。
- 失敗JSONは依存関係や認証のblockerを残す証拠です。GPU成功結果ではありません。
- 入力はsyntheticまたはsanitizedです。認証情報、OAuthリンク、session metadata、モデル重みは公開対象から除外しています。

## 🛠️ Colabで再現する

公式Colab CLIは現在Linux/macOS向けで、Windowsをサポートしていません。WindowsホストではWSL Ubuntuを使い、CLIのstateをリポジトリ外に置きます。

~~~bash
uv tool install google-colab-cli
colab --help
~~~

各実験READMEに、対象GPU、固有の <code>jev-&lt;slug&gt;</code> session名、分離したsession state file、upload/execute/downloadの流れ、stopまたはephemeral runの後始末を記載しています。quota・entitlement・認証で実行できない場合は、無限再試行せず条件を一度記録します。

## 🗂️ リポジトリ構成

~~~text
experiments/<slug>/
├── README.md                 # 実験の範囲と再現手順
├── notebooks/                # 該当するColab CLI notebook
├── results/                  # sanitizedな実測値・blocker記録
├── scripts/                  # runnerとfixture validator
└── tests/                    # 該当するmodel-free helper test
docs/                         # 公開する英日ガイド
.github/workflows/            # docs公開とpublic QA
NOTICE.md                     # 上流ソースとライセンスの境界
~~~

## 🧱 範囲と制約

これはprivateなTypeSafe Jev本体でも、本番の判断サービスでも、列挙した実装が同等品質だという主張でもありません。Nimble 9B、OpenJev 35B、DiffusionGemmaは初回のT4/L4 batchから保留しています。計測の意味、source revision、トラブルシュートは[公開ガイド](https://sunwood-ai-labs.github.io/jev-colab-lab/ja/)を参照してください。

## ⚖️ ライセンスと出典

このリポジトリのオリジナルのglue codeとドキュメントは [MIT License](LICENSE) で公開します。上流実装・モデル重みとそのライセンスは別であり、[NOTICE.md](NOTICE.md) と各実験のsource manifestに記録しています。モデル重みはこのリポジトリに保存しません。

## 📚 追加ドキュメント

- [日本語 docs](https://sunwood-ai-labs.github.io/jev-colab-lab/ja/)
- [Jevクローン × Google Colab 日本語まとめ](docs/ja/guide/jev-clone-colab.md)
- [English docs](https://sunwood-ai-labs.github.io/jev-colab-lab/)
- [実験インデックス](experiments/README.md)
- [作業規約](AGENTS.md)
