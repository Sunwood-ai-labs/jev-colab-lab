---
layout: home
hero:
  name: Jev Colab Lab
  text: Decision Model実験を再現可能に
  tagline: 固定したGPU実験、sanitizedな証跡、Colab再現手順をひとつの導線にまとめます。
  image:
    src: /jev-colab-lab-icon.svg
    alt: Jev Colab Labの幾何学的なdecision graphアイコン
  actions:
    - theme: brand
      text: ガイドを読む
      link: /ja/guide/getting-started
    - theme: alt
      text: English
      link: /
    - theme: alt
      text: ソースを見る
      link: https://github.com/Sunwood-ai-labs/jev-colab-lab
features:
  - icon: 🎯
    title: 固定した出典
    details: 上流リポジトリ、モデルrevision、ライセンス、想定GPUを各実験に記録します。
  - icon: 📏
    title: 分離した計測
    details: runnerが対応する範囲で、load・初回・warmup・steady state・peak VRAMを分けます。
  - icon: 🧼
    title: Sanitized成果物
    details: 認証情報、session metadata、private input、モデル重みを含めません。
---

## スナップショット

<div class="lab-grid">
  <div class="lab-card"><strong>T4 × 3</strong><p>Laya、Kev-0.5B、Jevlikeの短時間学習・推論。</p></div>
  <div class="lab-card"><strong>L4 × 2</strong><p>SemIfの直接option readoutとOpenJev NLI。</p></div>
  <div class="lab-card"><strong>結果セット × 5</strong><p>すべてchecked-in JSONと実験READMEへリンクします。</p></div>
</div>

5つの実験は意図的に異質です。このサイトはleaderboardではなく再現性のインデックスであり、fixture、readout、モデルサイズ、確率の意味が異なります。

## 次に読む

- [はじめに](/ja/guide/getting-started) — ローカル確認、uv、WSL、Colabの境界。
- [実験一覧](/ja/guide/experiments) — 各runnerが測るものと証跡の場所。
- [再現性](/ja/guide/reproducibility) — session分離、sanitized output、結果の解釈。
- [出典とライセンス](/ja/guide/sources-and-licenses) — upstream revisionとライセンスの対応。

<div class="tip custom-block">
  <p class="custom-block-title">現在の計測メモ</p>
  <p>Jevlikeの計測修正は別タスクでレビュー中です。旧初回推論時間は、ここで実験間比較の値として扱っていません。</p>
</div>
