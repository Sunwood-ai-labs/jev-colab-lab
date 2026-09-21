# JevDash 改良実験の検証記録

固定ゲームは `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480`、Level 1、60 Hz、モデル判断は8物理frameごと。推論待ちは物理を停止します。
ここでいう「モデル単独」は、モデルの選択を実行時の別ルールで変更しないという意味です。入力に記載したルールや教師学習まで存在しないという意味ではありません。

## 確認済みの改良結果

| 対象 | 旧ゲーム実験 | 改良結果 | 確認範囲 |
|---|---|---|---|
| SemIf / Qwen3.5-4B | 最初の土管、x486でtimeout | L4、rules-v2、モデル単独clear、647 frames | 81判断、走行2回・ダッシュジャンプ79回、実行上書き0 |
| Jevlike TinyScorer | 入力欠落、全right_jump、216 framesで死亡 | T4、r2ゲーム教師学習版、モデル単独clear、647 frames | 81判断すべてright_run_jump。状態適応の証明にはならない |
| Laya / ModernBERT | 7択ベースライン、通常ジャンプ偏りで死亡 | T4、2択制限 (`right_run`, `right_run_jump`) + reflex-assisted でclear、651 frames | 82判断、モデル単独（7択/2択）は未クリア、実行上書き60/651 frames (9.2%) |

上記クリア3件は担当タスクの確認に加え、統括側でもMP4を全フレームデコードし、クリア画面を目視確認しました。
さらに保存されたraw action列およびreflex補助ルールを独立した固定ゲームcloneで再生し、各エピソードの全frameの位置・速度・死活・クリア判定が一致することを確認しました。
この独立再生自体はローカルCPU検証であり、追加のColab推論ではありません。

SemIfでは、配布runnerから生成するstate/question/optionsが、保存GPUログ全81判断と完全一致することも確認しました。
実測済みrules-v2を既定とし、物理説明をさらに修正したv3は未GPU検証の任意variantとして区別します。

- [SemIfの実測JSON](../semif/results/jevdash-clear-20260921-v2/episode.json)
- [SemIfの実測動画](../semif/results/jevdash-clear-20260921-v2/episode.mp4)
- [Jevlikeの実験・variant説明](../jevlike/jevdash-clear.md)
- [Jevlike r2の公開動画・要約](../jevlike/results/jevdash-clear-r2-public/README.md)
- [Layaの実測manifest](../laya/results/jevdash-clear/manifest.json)
- [Layaのクリア結果JSON](../laya/results/jevdash-clear/laya-rules-v2-forward-binary-reflex-assisted.json)

独立再生の記録は [SemIf](results/semif-v2-independent-replay.json) / [Jevlike](results/jevlike-r2-independent-replay.json) / [Laya](results/laya-assisted-independent-replay.json) に保存しています。元episode全体のSHA256と検証環境を含みます。Jevlikeの公開要約は匿名化した別ファイルなので、元episodeのSHA256とは異なります。

```powershell
uv run --project experiments/control-audit python experiments/control-audit/verify_episode_replay.py --game <pinned-clean-game-checkout> --episode <full-episode-json> --output <verification-json>
```

検証器はモデル単独のraw/executed一致（SemIf/Jevlike）および補助適用時の事前観測・実行action（Laya）、frame連続性、全位置・速度・死活・クリア状態を照合します。SemIfの40番frameのxだけを1px変更した負の対照では、その不一致を検出してexit code 1となりました。

## 原因の切り分け

### 入力欠落は接続実装の不具合

旧Jevlikeは192byteの入力枠が固定指示文で埋まり、27判断で実入力が同一になっていました。これをモデル能力の失敗として扱うことはできません。
改良版は状態を先頭に置き、監査では177byte以内に収めて入力変化を確認しています。

### 入力の説明と候補順の影響

SemIfの実L4監査は6状態×4入力表現×4候補順の96条件です。旧表現では候補順で選択が変わりました。
物理・距離情報の意味を説明した表現では、監査した候補順で選択が安定し、障害物前でジャンプを選択しました。
旧blocked fixtureの候補token確率質量は約0.9996であり、この条件では候補内softmaxだけが見かけの高確率を作ったわけではありません。
モデルごとに採点方法が異なるため、この説明を他モデルへ自動的に一般化しません。

### 教師データの偏りとアクション・コラプス

Jevlike r2はゲーム教師データで再学習したvariantです。約97%がジャンプのデータで、実走の出力も全ダッシュジャンプになりました。
偏りを50:50にしたbalanced-probeを用意しましたが、追加T4は同時割当上限で確保できず、GPU性能は未検証です。
データ偏りを調整すれば状態適応が改善するとまでは、現時点では証明できません。

Kevでも同様に、rules-v2入力への改善を行いましたが、実T4でのモデル単独実行は225判断すべてが `jump` にコラプスし、初期位置 x96 で1800fタイムアウトとなりました。単なるプロンプト改善だけでは行動の固定化を解消できない事例です。

OpenJev NLIでも、L4実測のrules-model-only実行では全判断が `right` にコラプスし、最初の土管 (x486) で1800fタイムアウトとなりました。NLIの3クラス分類形式において「各行動の前提条件が真か」を問う設定でも、歩行偏りの壁が存在します。

### ゲーム側の補助とテレメトリー

- 本家は毎frameのジャンプ補助を持ちます。補助なしクローンとの録画比較は同条件ではありません。
- 固定right_run_jumpだけでも647 framesでclearできます。clearだけではゲーム理解を証明できません。
- 本家reflexも毎frameと8frameごとでは結果が異なります。[45条件の対照実験](README.md)を参照してください。
- 障害物距離はtile列差であり、前端の実距離ではありません。土管に接触したx486でも「距離1」となります。
- 浮遊床も障害物として数えられるため、接地状態・高さ・敵のvertical offset・停滞を併せて解釈する必要があります。
- noopは接地時に減速し、空中では慣性を保持します。right系は空中で6.6px/frameになります。通常ジャンプと走行ジャンプの初速はそれぞれ-13.5/-15.5です。
- OpenJev NLI rules-assistedの実験では、毎frame補助が37回発火したものの、モデルの基本行動が歩行 (`right`) であったため助走初速が足りず、388f (x2400.92) の穴で落下死亡しました。補助があっても基礎行動の速度が不足していればクリアできない物理的制約が確認されました。

### 補助の計数

実行上書きは `raw_action != executed_action` のframe数で数えます。補助判定が発火しても同じactionを返した場合は別のguard triggerです。
Jevlike r2 assistedはguard triggerが56 frameでも、実行上書きは0 frameでした。旧指標の混同を訂正しています。
Laya assistedでは、651 simulation frames中 60 frames（9.2%）が実際に `right_run` から `right_run_jump` へ上書き変更されました。

## 改良実験のまとめ

5つのJevクローンに対するGoogle Colab実機（T4/L4）での検証結果は以下の通りです：

1. **SemIf (L4)**: プロンプト（rules-v2）による状態・物理ルールの明確化により、**モデル単独でクリア達成**（647f, 81判断）。
2. **Jevlike (T4)**: 入力バイト枠欠落のバグ修正とゲーム教師学習（r2）により、**モデル単独でクリア達成**（647f, 81判断）。ただし全判断がダッシュジャンプにコラプスしており、状態適応の証明ではありません。
3. **Laya (T4)**: 候補を2択（`right_run`, `right_run_jump`）に制限し、毎frameのreflex補助を併用することで**クリア達成**（651f, 82判断, 60上書き）。7択およびモデル単独では未クリア。
4. **Kev (T4)**: プロンプト改善（rules-v2）を行っても全判断が `jump` にコラプスし、x96でタイムアウト未クリア。
5. **OpenJev NLI (L4)**: プロンプト改善を行っても全判断が `right` にコラプスしx486でタイムアウト。補助付きでも歩行速度不足によりx2400で落下死亡し未クリア。

固定地形での単一試行から未知ステージへの汎化やモデルの絶対的な性能順位を結論することはできません。
