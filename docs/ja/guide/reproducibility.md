# 再現性

再現とは、source revision、runtime境界、input fixture、計測の意味、cleanup手順をそろえることです。将来のColab VMでquota、package cache、wall-clock latencyまで同じになることを意味しません。

## Sourceと環境

各実験は、upstream implementationとmodel revisionをlocal source manifestまたはresult JSONに記録します。比較rerunではrevisionを固定してください。notebookやscriptのmoving <code>main</code> URLへ置き換えないでください。

Pythonは <code>uv</code> を使います。

~~~powershell
uv run --project experiments/laya --group dev pytest -q
uv run --project experiments/semif --group dev python scripts/validate_fixture.py data/questions.jsonl
~~~

正確なコマンドは実験ごとに異なるため、各READMEを正とします。

## Runtimeの分離

task専用のsession名とstate fileを1つずつ使い、Gitの外に置きます。

~~~bash
CFG=/tmp/jev-semif-colab-session.json
COLAB=colab
$COLAB --config "$CFG" new --session jev-semif --gpu L4
~~~

別タスクのactive sessionを再利用しません。実験が失敗してもsanitized resultを回収し、手動作成したsessionをstopします。ephemeralな <code>colab run</code> はscript完了後にruntimeを解放できますが、対応syntaxは実験READMEに従ってください。

## 計測用語

| 用語 | 意味 |
| --- | --- |
| Load | runnerが定義したモデルdownload/deserializationとdevice配置。 |
| First inference | setup境界の後に最初に測ったinference領域。一時的なkernel/cache効果を含む場合があります。 |
| Warmup | steady-state sampling前の意図的な反復。 |
| Steady state | mean/median/percentileなどで要約する反復inference。 |
| Peak VRAM | runnerが報告したallocated/reserved CUDA memoryであり、カード全体容量ではありません。 |
| Fixture accuracy | checked-inまたはsynthetic fixtureだけの正答率で、一般ベンチマークではありません。 |

正確な境界はresult JSONを正とします。別実験のschemaから欠けたmetricを推測しません。

## Sanitization checklist

結果をcommitする前に確認します。

- model weightsと大きなdownload cacheを除外する
- OAuth link、ADC credential、token、cookie、private session metadataを除外する
- private inputをpublicまたはsynthetic fixtureへ置き換える
- 解釈に不要な絶対パスを除外する
- GPU名、source revision、package version、timing、memory、errorは残す

## Failureの扱い

quota不足、認証不足、依存関係不整合は実験履歴の一部です。sanitized JSONに一度記録し、自分が作成したruntimeを解放し、local fallbackをGPU成功として表現しません。
