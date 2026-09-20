# 実験一覧

5つの実験を <code>main</code> に統合しています。答える問いが異なるため、独立した実験として読みます。

| 実験 | GPU | runnerの処理 | 結果 |
| --- | --- | --- | --- |
| [Laya / ModernBERT](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/laya) | T4 | synthetic caseでchoice・score・noulのdecision outputを実行します。 | [T4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/results/laya-t4-result.json) |
| [Kev-0.5B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/kev) | T4 | 公式decision modelとadapterを使い、packedとseparate readを比較します。 | [T4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/results/kev-t4-result.json) |
| [Jevlike](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/jevlike) | T4 | tiny scorerを短時間学習し、held-out synthetic dataとshuffled-context controlを評価します。 | [T4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/results/colab-t4-result.json) |
| [SemIf / Qwen3.5-4B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/semif) | L4 | 宣言したsingle-token optionの次トークンlogitを読み、option内softmaxを適用します。 | [最新L4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/results/semif-l4-result-20260921.json) |
| [OpenJev NLI 4B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/openjev-nli) | L4 | fixtureのpremise/hypothesisペアをcontradiction・entailment・neutralでscoreします。 | [L4 JSON](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/results/colab-l4.json) |

## 証跡のstatus

LayaとOpenJevのJSONは <code>status=ok</code>、Kev・Jevlike・SemIfは <code>status=success</code> です。いずれも隣接するruntimeまたはhardware objectでactual GPUを確認します。requested T4/L4だけではallocationの証拠になりません。

catalogには失敗記録も残しています。

- Kevは、成功rerunの横に初回の依存関係不整合を保持します。
- SemIfは、optional package修正前のimport failureを保持します。
- Jevlikeは、以前の認証blockerを保持します。

失敗記録は成功までの経路を説明するもので、追加の成功結果ではありません。

## 比較の境界

raw latencyやprobabilityで順位付けしません。モデル、prompt形式、option数、batch shape、precision、fixtureの意味が異なるためです。SemIfはconditional option score、OpenJevはNLI probability、Jevlikeはshuffled-context controlを含む短いsynthetic experimentです。

Jevlikeの計測修正は別のreview taskで進行中です。修正版が統合されるまで、旧初回推論値は公開ガイドで繰り返しません。
