# はじめに

このプロジェクトでは、model-freeなリポジトリ確認と、任意のmodel-backed実行を分けています。まず前者を実行してください。高速で、モデル重みをダウンロードしません。

## 1. ローカルツールを入れる

docsをローカルbuildするなら、[uv](https://docs.astral.sh/uv/) とNode.js 20以降を用意します。このリポジトリのPythonコマンドは <code>uv</code> 経由で実行します。

~~~powershell
git clone https://github.com/Sunwood-ai-labs/jev-colab-lab.git
Set-Location jev-colab-lab
uv --version
~~~

## 2. model-free確認を実行する

~~~powershell
uv run --no-project --with pytest pytest experiments/laya/tests experiments/jevlike/tests experiments/openjev-nli/tests experiments/semif/tests -q
uv run --no-project python -m py_compile experiments/kev/t4_inference.py
~~~

これらはhelper logic、公開fixture、notebook構造、Python構文を確認します。GPUが使えることや、モデルのdownloadが成功することは証明しません。

## 3. 実験を選ぶ

対応する[実験README](https://github.com/Sunwood-ai-labs/jev-colab-lab/tree/main/experiments)を開き、インストール前にsource manifestを確認します。各実験の依存関係と結果ファイルは、その実験READMEを正とします。

| ターゲット | 実験 | 用途 |
| --- | --- | --- |
| T4 | Laya、Kev-0.5B、Jevlike | 低コストなdecision-model・tiny scorer経路 |
| L4 | SemIf 4B、OpenJev NLI 4B | 4B direct-logit・NLI経路 |

## 4. WindowsからColabを動かす

公式の[Google Colab CLI](https://github.com/googlecolab/google-colab-cli)は現在Linux/macOS向けで、Windowsをサポートしていません。WindowsホストではWSL Ubuntuを使います。

~~~bash
uv tool install google-colab-cli
colab --help
~~~

各実行には次を用意します。

- <code>jev-laya</code> や <code>jev-semif</code> のような固有のsession名
- リポジトリ外に置くtask専用のsession state file
- requested GPUと、結果に記録されたactual GPUを分けた記録
- 手動sessionなら、sanitized resultをdownloadしてからstopする手順

認証、quota、entitlementの失敗も記録可能な結果です。OAuthリンク、ADC file、session log、weightsをcommitしません。

## 5. 証跡を読む

実行後はリンクされたresult JSONを開きます。成功statusとactual GPUが両方記録されている場合だけGPU成功と扱います。計測用語は[再現性](/ja/guide/reproducibility)を参照してください。
