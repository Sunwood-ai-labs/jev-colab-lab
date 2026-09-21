# JevDash 制御経路の原因監査

このディレクトリは **ローカルの決定的物理対照** です。Colab・GPU・モデル推論は使用しません。
既存ゲームと同じ更新順を使い、モデルの能力とゲーム側の補助制御の寄与を分離します。

## 確認結果

固定ゲーム [eb2f926](https://github.com/Sunwood-ai-labs/jevdash/tree/eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480)、Level 1、60 Hz、開始座標 x=96、上限1800 frames。

| 固定入力 | 補助 | 結果 | 終了frame | 最大x |
|---|---|---|---:|---:|
| right_run | なし | 最初の土管でtimeout | 1800 | 486 |
| right_jump | なし | 敵との衝突で死亡 | 216 | 1454.6 |
| left | なし | 左側へ落下 | 62 | 96 |
| right_run_jump | なし | **クリア** | 647 | 4204.4 |
| right_run | 本家reflexを毎frame | **クリア** | 651 | 4204.4 |
| right_run | 本家reflexを8frameごと | 死亡 | 343 | 2335.6 |

`right_run_jump` 固定と `right_run + 毎frame reflex` は、開始xを±16px動かした場合もクリアしました。
全45条件は [physics-controls.json](results/physics-controls.json) に記録しています。
同じステージの小さな開始位置変更であり、未知ステージへの汎化を示しません。
Level(1)の地形は固定なので、seedの変更だけでは異なる地形の評価になりません。

## 失敗原因について分かったこと

- 旧SemIf/OpenJevのx486停滞、旧Jevlikeの216frame死亡、旧Kevの62frame死亡は、それぞれの固定行動だけで再現できます。これで直接の状態遷移を説明できますが、モデルがその行動を選んだ根本原因まで証明するものではありません。
- `right_jump` と `right_run_jump` はジャンプ強度が異なり、同じ「右へジャンプ」でも結果が大きく変わります。
- 本家の [AsyncJevAgent.get_action](https://github.com/Sunwood-ai-labs/jevdash/blob/eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480/src/jev_platformer/controller/jev_agent.py) は障害物・敵・停滞などでモデル出力を上書きします。毎frame補助でのクリアはモデル単独の成功ではありません。
- 補助判定を8frameごとにまとめると死亡する対照があり、モデルの判断間隔と安全制御の間隔を混同できません。
- 常時ダッシュジャンプでもクリアできるため、`has_won=true`だけでは状態を理解した証拠になりません。
- Jevlikeの旧接続は192byteに入力を切り、状態情報を欠落させていました。この結果はゲーム能力の評価として無効です。各モデルの学習内容・接続不具合はそれぞれの修正実験で監査します。SemIfはQwen3.5-4B direct readoutであり、scratch badge学習はJevlikeだけです。

## 改良実験の完了条件

1. 最終byte/token入力に現在の状態が残り、候補が衝突・切断しないこと。
2. 同一状態の再現性、異なる状態での入力/確率/行動、候補順置換を記録すること。
3. rawモデル行動と実行行動を分離し、補助があれば理由・介入率を記録すること。
4. モデル単独・補助付き・モデルなしの固定行動対照を別物として報告すること。
5. Colab上の実モデル実行で内部`has_won`と動画を確認し、未達なら明記すること。
6. 教師学習を追加した場合は、その教師・データ・学習範囲を明示すること。

## 再実行

外部のクリーンなJevDash cloneを上記commitに固定してから実行します。

```powershell
uv run --project experiments/control-audit python experiments/control-audit/audit_controls.py --game <pinned-jevdash-directory> --output experiments/control-audit/results/physics-controls.json
```

Python依存は同ディレクトリのuv.lockで固定。ゲームHEADと未コミット変更を検査し、異なる実装の結果を混入させません。
reflex条件と物理更新順はJevDash（Copyright 2026 Sunwood AI Labs、MIT）から再現しています。ライセンス本文はリポジトリルートの [LICENSE](../../LICENSE) に同梱されています。
