# JevDash 改良実験の検証記録

固定ゲームは `eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480`、Level 1、60 Hz、モデル判断は8物理frameごと。推論待ちは物理を停止します。
ここでいう「モデル単独」は、モデルの選択を実行時の別ルールで変更しないという意味です。入力に記載したルールや教師学習まで存在しないという意味ではありません。

## 確認済みの改良結果

| 対象 | 旧ゲーム実験 | 改良結果 | 確認範囲 |
|---|---|---|---|
| SemIf / Qwen3.5-4B | 最初の土管、x486でtimeout | L4、rules-v2、モデル単独clear、647 frames | 81判断、走行2回・ダッシュジャンプ79回、実行上書き0 |
| Jevlike TinyScorer | 入力欠落、全right_jump、216 framesで死亡 | T4、r2ゲーム教師学習版、モデル単独clear、647 frames | 81判断すべてright_run_jump。状態適応の証明にはならない |

上記2件は担当タスクの確認に加え、統括側でもMP4を全フレームデコードし、クリア画面を目視確認しました。
さらに保存されたraw action列を独立した固定ゲームcloneで再生し、647 frameすべての位置・速度・死活・クリア判定が一致することを確認しました。
この独立再生自体はローカルCPU検証であり、追加のColab推論ではありません。

SemIfでは、配布runnerから生成するstate/question/optionsが、保存GPUログ全81判断と完全一致することも確認しました。
実測済みrules-v2を既定とし、物理説明をさらに修正したv3は未GPU検証の任意variantとして区別します。

- [SemIfの実測JSON](../semif/results/jevdash-clear-20260921-v2/episode.json)
- [SemIfの実測動画](../semif/results/jevdash-clear-20260921-v2/episode.mp4)
- [Jevlikeの実験・variant説明](../jevlike/jevdash-clear.md)
- [Jevlike r2の公開動画・要約](../jevlike/results/jevdash-clear-r2-public/README.md)

独立再生の記録は [SemIf](results/semif-v2-independent-replay.json) / [Jevlike](results/jevlike-r2-independent-replay.json) に保存しています。元episode全体のSHA256と検証環境を含みます。Jevlikeの公開要約は匿名化した別ファイルなので、元episodeのSHA256とは異なります。

```powershell
uv run --project experiments/control-audit python experiments/control-audit/verify_episode_replay.py --game <pinned-clean-game-checkout> --episode <full-episode-json> --output <verification-json>
```

検証器はモデル単独のraw/executed一致、frame連続性、全位置・速度・死活・クリア状態を照合します。SemIfの40番frameのxだけを1px変更した負の対照では、その不一致を検出してexit code 1となりました。

## 原因の切り分け

### 入力欠落は接続実装の不具合

旧Jevlikeは192byteの入力枠が固定指示文で埋まり、27判断で実入力が同一になっていました。これをモデル能力の失敗として扱うことはできません。
改良版は状態を先頭に置き、監査では177byte以内に収めて入力変化を確認しています。

### 入力の説明と候補順の影響

SemIfの実L4監査は6状態×4入力表現×4候補順の96条件です。旧表現では候補順で選択が変わりました。
物理・距離情報の意味を説明した表現では、監査した候補順で選択が安定し、障害物前でジャンプを選択しました。
旧blocked fixtureの候補token確率質量は約0.9996であり、この条件では候補内softmaxだけが見かけの高確率を作ったわけではありません。
モデルごとに採点方法が異なるため、この説明を他モデルへ自動的に一般化しません。

### 教師データの偏りは別の課題

Jevlike r2はゲーム教師データで再学習したvariantです。約97%がジャンプのデータで、実走の出力も全ダッシュジャンプになりました。
偏りを50:50にしたbalanced-probeを用意しましたが、追加T4は同時割当上限で確保できず、GPU性能は未検証です。
データ偏りを調整すれば状態適応が改善するとまでは、現時点では証明できません。

### ゲーム側の補助とテレメトリー

- 本家は毎frameのジャンプ補助を持ちます。補助なしクローンとの録画比較は同条件ではありません。
- 固定right_run_jumpだけでも647 framesでclearできます。clearだけではゲーム理解を証明できません。
- 本家reflexも毎frameと8frameごとでは結果が異なります。[45条件の対照実験](README.md)を参照してください。
- 障害物距離はtile列差であり、前端の実距離ではありません。土管に接触したx486でも「距離1」となります。
- 浮遊床も障害物として数えられるため、接地状態・高さ・敵のvertical offset・停滞を併せて解釈する必要があります。
- noopは接地時に減速し、空中では慣性を保持します。right系は空中で6.6px/frameになります。通常ジャンプと走行ジャンプの初速はそれぞれ-13.5/-15.5です。

### 補助の計数

実行上書きは `raw_action != executed_action` のframe数で数えます。補助判定が発火しても同じactionを返した場合は別のguard triggerです。
Jevlike r2 assistedはguard triggerが56 frameでも、実行上書きは0 frameでした。旧指標の混同を訂正しています。

## 残る検証

Laya・Kev・OpenJev NLIは改良版の実GPU検証中です。確認できていないクリア結果を上の表へ追加しません。
固定地形での単一試行から未知ステージへの汎化やモデルの性能順位は結論できません。
