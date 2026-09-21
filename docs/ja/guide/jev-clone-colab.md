# Jevクローン × Google Colab：5経路を再現してわかったこと

更新日: 2026-09-21。対象リポジトリ: [Sunwood-ai-labs/jev-colab-lab](https://github.com/Sunwood-ai-labs/jev-colab-lab)

この記事は、Jevの考え方に近い公開Decision Modelの実装を、Google Colab GPUで動かした記録です。対象は **Laya、Kev-0.5B、Jevlike、SemIf、OpenJev NLI** の5経路。いずれも非公開のTypeSafe Jev本体ではなく、目的・入力形式・学習内容・確率の意味が異なる独立実験です。

結論を先に言うと、通常の選択推論は5件とも実GPUのsanitized JSONを回収できました。一方、JevDashの録画はゲーム能力の正常な比較評価ではありません。特にJevlikeは入力の先頭切り詰めで状態情報が欠落しており、ゲーム結果を能力の証拠として使えません。

## 先に結論

- **GPU実測**: Laya・Kev・JevlikeはColab T4、SemIf・OpenJev NLIはColab L4です。結果JSONのstatusだけでなく、actual GPU/runtimeも確認します。
- **通常の選択推論**: 選択肢の確率、NLIのentailment、load・初回・warmup・steady-state・peak VRAMを、各runnerの定義に従って記録しました。
- **比較の単位**: 同じ「選択」に見えても、Laya/Kev/Jevlikeはdecision head、SemIfは候補文字の次トークンlogit、OpenJev NLIは3クラス分類です。数値を横並びのleaderboardにはしません。
- **ゲーム録画**: Level 1・seed 42・60 FPS・8 simulation framesごとの同期判断です。推論待ち時間はsimulation timeに含まれません。単一seed・各1episodeなので順位は出せません。

## 1. 何を動かしたか

| 経路 | GPU | 推論の形 | 入力・学習の範囲 |
| --- | --- | --- | --- |
| [Laya / ModernBERT](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/laya) | T4 | choice・score・noulを1回のdecision呼び出しで出力 | 公開英語checkpoint。今回のfixtureはサポート問い合わせで、公式ベンチマークの再計測ではありません。 |
| [Kev-0.5B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/kev) | T4 | packedとseparateのpointer readout | 公式decision model＋adapter。今回のfixtureは動作・確率・計測用です。 |
| [Jevlike](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/jevlike) | T4 | TinyScorerのoption softmax | byte/position embeddingをスクラッチ学習し、合成badge選択だけで4 epoch。ゲーム学習はありません。 |
| [SemIf / Qwen3.5-4B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/semif) | L4 | 宣言したA/B/C等のsingle-token logitsを候補内softmax | 直接next-token readout。NLI headや回答文生成は使いません。 |
| [OpenJev NLI 4B](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments/openjev-nli) | L4 | contradiction・entailment・neutralの分類 | 初版4B NLI checkpointと、公開・合成の5問×3仮説fixture。35Bとv2は対象外です。 |

各実験のresult JSONがcanonicalな証跡です。`status=ok`/`success`だけではGPU成功とせず、JSON内の実GPU名、runtime、error、計測境界を一緒に読みます。

## 2. 通常の選択推論の実測結果

ここでいう確率・正答率は各fixtureとreadoutに条件付いた値です。一般性能、校正済み信頼度、本家Jevとの同等性を意味しません。

| 経路 | 選択推論の観測 | 計測の抜粋 | 読み方の制限 |
| --- | --- | --- | --- |
| Laya / T4 | `department=billing`、確率 **0.9254** | singleのsteady平均 **36.17 ms**（20回）、model load **22.96 s** | 公開英語checkpointのsynthetic support fixture。`confidence`を一般の信頼度として再解釈しない。 |
| Kev / T4 | `department=returns`、確率 **0.85** | steady forward平均 **50.12 ms**（10回）、packed/separate最大差 **4.17e-7** | 現行ColabのTorchは公式pyprojectの制約外なので、完全な依存ロック再現ではなく、固定モデル経路のT4実行記録。 |
| Jevlike / T4 | `amber badger`、確率 **0.5839**。test top-1 **81.25%**、shuffled-context control **19.53%** | steady p50 **0.81 ms**（20回）、4 epoch学習 **1.37 s** | 512/128/128件の合成badge選択だけ。ゲームの入力不備とは別に、これは小さなsynthetic実験の値。process-cold起動時間ではありません。 |
| SemIf / L4 | `route-1=account_access`、条件付き確率 **0.99998** | model load **55.76 s**、steady per-decision平均 **94.42 ms** | 候補文字に制限したoption scoreであり、校正済みdecision confidenceではありません。 |
| OpenJev NLI / L4 | 5問のfixture decision accuracy **5/5** | model load **51.83 s**（cache coldなら取得込み）、15 pair batchのsteady平均 **255.31 ms** | `entailment`最大で候補を選ぶNLI分類。5問の合成fixtureだけで、一般ベンチマークではありません。 |

詳細なload・初回・warmup・steady-state・peak VRAM、依存version、固定revisionは各JSONを参照してください。

- [Laya T4 result](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/results/laya-t4-result.json)
- [Kev T4 result](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/results/kev-t4-result.json)
- [Jevlike T4 result](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/results/colab-t4-result.json)
- [SemIf L4 result](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/results/semif-l4-result-20260921.json)
- [OpenJev NLI L4 result](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/results/colab-l4.json)

## 3. 5冊のColab notebookへのリンク

GitHubリンクはnotebookの中身を読む入口、Colabリンクはそのnotebookを開く入口です。Notebookは薄いentry pointであり、実験によっては同じフォルダのrunner・fixture・`pyproject.toml`も必要です。結果を再現したい場合は、まず各実験READMEの依存関係と固定revisionを確認してください。

| 経路 / GPU | Notebook | 実行方法 | 確認範囲と制限 |
| --- | --- | --- | --- |
| Laya / T4 | [GitHub](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/notebooks/laya_t4_benchmark.ipynb) · [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/laya/notebooks/laya_t4_benchmark.ipynb) | `benchmark_laya.py`を`/content/benchmark_laya.py`へuploadし、依存install後にnotebookを`colab exec`。 | single/batchのchoice・score・noul、速度・VRAMを測る。Layaの公式汎用英語checkpointのみ。 |
| Kev-0.5B / T4 | [GitHub](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/kev_t4_inference.ipynb) · [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/kev/kev_t4_inference.ipynb) | `t4_inference.py`を`/content/t4_inference.py`へuploadし、notebookから実行。 | packed/separate readoutの整合性とレイテンシ。モデル重みは保存しない。 |
| Jevlike / T4 | [GitHub](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/notebooks/jevlike_t4_experiment.ipynb) · [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/jevlike/notebooks/jevlike_t4_experiment.ipynb) | 固定した実験commitのrunnerをdownloadし、`uv`で上流をinstall、合成データを学習して実行。 | synthetic badge選択の短時間実験。JevDashゲーム結果は入力欠落のため能力評価に使えない。 |
| SemIf / L4 | [GitHub](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/notebooks/semif_l4_experiment.ipynb) · [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/semif/notebooks/semif_l4_experiment.ipynb) | repoをColab VMに配置し、`experiments/semif`を作業ディレクトリにして`uv sync`→fixture検証→runner。 | Qwen3.5-4Bの直接option-logit。候補集合に条件付いたscoreで、NLIではない。 |
| OpenJev NLI 4B / L4 | [GitHub](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/notebooks/openjev_nli_l4.ipynb) · [Colabで開く](https://colab.research.google.com/github/Sunwood-ai-labs/jev-colab-lab/blob/main/experiments/openjev-nli/notebooks/openjev_nli_l4.ipynb) | repoを`experiments/openjev-nli`で開き、`uv`依存準備→fixture→測定runner。 | 初版4B NLIの3クラス確率と5問fixture。v2/35B・公開ベンチマーク比較は含めない。 |

5冊すべてについて、ローカルファイルのJSON parse、パス、cell構造、対応READMEとのrunner名を確認済みです。Colab上の新しいruntimeでの再実行は、この記事作成のためには行っていません。

## 4. WindowsホストからWSLで実行する

公式の[Google Colab CLI](https://github.com/googlecolab/google-colab-cli)は、現時点でLinux/macOS向けでWindowsをサポートしていません。WindowsホストではWSL UbuntuをCLI実行環境にします。CLIの公式READMEにあるとおり、`uv tool install`、`colab new --gpu`、`colab exec`、`colab download`、`colab stop`の流れを使えます。

最初の確認はWSL側で行います。

```bash
uv tool install google-colab-cli
colab --help
colab version
```

手動sessionを使う場合は、実験ごとに名前とstate fileを分けます。

```bash
CFG=/tmp/jev-laya-colab-state.json
colab --config "$CFG" new --session jev-laya --gpu T4
# upload / install / exec / download は experiments/laya/README.md の手順を使う
colab --config "$CFG" stop --session jev-laya
```

実験ごとの原則は次のとおりです。

- session名は `jev-<slug>` のように一意にし、別タスクのruntimeを再利用しない。
- `--config`のstate fileはgit worktree外に置く。session log、endpoint、OAuth/ADC情報はcommitしない。
- 結果JSONをdownloadして内容を確認してから、自分が作ったsessionだけをstopする。
- `colab run`のephemeral実行を使う実験は、終了時にruntimeが解放される仕様でも、結果回収とstatus確認を行う。
- quota・entitlement・認証で作成できない場合は、同じGPUの無限再試行や課金をせず、blockerとして記録する。

## 5. uv・worktree・runtimeを分けた理由

各実験のPython依存は `experiments/<slug>/pyproject.toml` とlockfileに閉じ、コマンドは`uv run`またはColab VM内の`uv`経由にしました。実験タスクは独立worktree・独立branchで作業し、別タスクの未コミット変更やruntimeを共有しません。

これは同じモデルの性能を揃えるための仕組みではなく、次の混同を防ぐための実験運用です。

- ローカルCPU smoke testをColab GPU成功と扱わない。
- requested GPUと、result JSONに記録されたactual GPUを区別する。
- model load、初回推論、warmup、steady-state、peak VRAMの境界が違う値を横比較しない。
- fixture正答率を一般化性能と呼ばない。

## 6. 上流実装・学習内容・revision・license

実験で使った固定値は[NOTICE.md](https://github.com/Sunwood-ai-labs/jev-colab-lab/blob/main/NOTICE.md)と各source manifestに記録しています。主要な確認結果は次のとおりです。

| 対象 | 固定した一次資料 | 実験で確認した範囲 |
| --- | --- | --- |
| Laya | [実装 commit d113dca](https://github.com/NandhaKishorM/laya/tree/d113dca2512fb3eaca313534bc54c7162d87c1d4)、[checkpoint revision 1c5edc](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982) | `DecisionModel`、ModernBERT-large系の公開英語checkpoint、Apache-2.0。`laya-typed-decisions`は今回使っていません。 |
| Kev | [実装 commit ac67bf4](https://github.com/jaredpalmer/kev/tree/ac67bf4e52d7bdc8420d8024c396df5585915d8c)、[adapter revision edf1dc](https://huggingface.co/jaredpalmer/kev-0.5b/tree/edf1dc6d7f8d983c0adfd251e80a686e5539fc61)、[Qwen2.5-0.5B revision 060db6](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987) | 公式`DecisionModel`/`encode`/adapter headの経路を使用。実験runnerがCUDAを明示するT4検証であり、公式コードの自動CUDA選択を主張しません。Apache-2.0の境界は上流資料を参照。 |
| Jevlike | [実装 commit 94f5fd1](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452)、[`data.py`](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/jevlike/data.py#L45-L57)、[MIT license](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/LICENSE) | TinyScorerのbyte/position embeddingをスクラッチ初期化し、`write_synthetic`のbadge選択データだけで学習。上流MIT、ゲーム学習なし。 |
| SemIf | [実装 commit ca3ba65](https://github.com/TheoLeeCJ/semif/tree/ca3ba65f142967030ecb453346e94d6f476a69df)、[Qwen3.5-4B revision 851bf6](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | SemIfはMIT、QwenモデルカードはApache-2.0。宣言したsingle-token回答slotのlogitだけを読む直接readoutで、NLI classifierではありません。 |
| OpenJev NLI | [4B初版 revision b32265](https://huggingface.co/AlexWortega/openjev/tree/b32265f4700df7c02532933c9a4ff258a449d7ac)、[model card](https://huggingface.co/AlexWortega/openjev) | `qwen3.5-4b-nli`のみ。モデルカード側のMIT、base Qwen3.5-4BのApache-2.0を別々に扱い、v2/35Bは除外。 |

この区別があるため、「5つのJevクローンが同じモデル」「同じ学習をしている」「Jev本体と同等」とは書けません。

## 7. JevDash録画はゲーム能力の比較ではない

固定したゲームは[Sunwood-ai-labs/jevdashのcommit `eb2f926`](https://github.com/Sunwood-ai-labs/jevdash/tree/eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480)です。Level 1、seed 42、60 FPS、8 simulation framesごとに1判断、最大1800 simulation framesという同じ枠で、5本の録画をgit外に保存しました。ゲーム本体は変更していません。

ただし、これは同条件の単一episodeを残すための観測であり、能力評価の完了を意味しません。同期推論の待ち時間中はゲーム物理を進めず、動画・simulation timeにも推論待ちを入れていないため、到達xや30秒timeoutをGPU速度の順位として読めません。

| 経路 | 録画の観測 | 判定 |
| --- | --- | --- |
| Laya / T4 | 225判断、30秒上限でtimeout、x=2470（5本目の土管付近） | 実モデル経路の1episode。クリア性能の証明ではありません。 |
| Kev / T4 | 8判断すべて`left`、開始位置のx=96のまま左へ落下 | x=96は最大走行距離ではなく開始座標です。 |
| Jevlike / T4 | 27判断、3.6秒でdeath、全て`right_jump`、x=1454.6 | **ゲーム能力評価として無効**。入力欠落が確定しています。 |
| SemIf / L4 | 225判断、30秒timeout、x=486（最初の土管付近） | 1episodeの停止観測。 |
| OpenJev NLI / L4 | 225判断、30秒timeout、x=486（最初の土管付近） | 1episodeの停止観測。 |

### Jevlikeの入力欠落

Jevlike上流の`ByteCollator`は、コンテキストをUTF-8 byte列にして`[:context_tokens]`で先頭から切ります。今回のgame adapterは長い定型指示の後ろに`JevObservation`のJSONを連結し、`context_tokens=192`を指定していました。

作業時に確認したsanitized episode JSONでは、次の3点が揃っています。

- 全27判断で、モデルへ渡ったcontextの先頭192 bytesが同一。
- 27回すべて確率ベクトルが同一で、argmaxは`right_jump`。
- 物理216 frames、simulation time 3.6秒でdeath。TinyScorerは合成badge選択512例などで学習しており、ゲーム学習はありません。

従って、これは「ゲームを理解して常に同じ行動を選んだ」結果ではなく、状態情報が入力に届かなかった接続不備の記録です。presentation replayは元の軌跡とHUDを照合するだけで、この入力経路を修正したり追加推論したりしません。**今回の記事・変更でこの不備は修正済みとは扱いません。** Laya・Kev・SemIf・OpenJev NLIに同じ不具合があるとは確認していませんが、4件のゲーム接続妥当性を比較評価として完了したとも書きません。

## 8. 再利用する際の境界

- notebookは固定revisionやrunnerを確認する入口です。Colabで再実行した結果は、既存JSONを上書きせず別名で保存してください。
- `load`がdownload込みか、`first inference`がprocess-coldか、steady-stateが何を含むかは実験ごとに違います。
- local CPU、fixture test、notebookのJSON parseはコード経路の確認であり、Colab GPU成功の代わりにはなりません。
- 外部管理のMP4・代表フレーム・ゲームepisode JSONは、このリポジトリに公開URLを持たせていません。公開成果物はrunner、notebook、sanitized result、source manifestです。
- 認証情報、OAuth link、session metadata、private input、model weightsは公開しません。

次に読むなら、[実験一覧](experiments.md)、[再現性](reproducibility.md)、[出典とライセンス](sources-and-licenses.md)、または各実験のREADMEがおすすめです。
