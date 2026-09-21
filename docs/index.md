---
layout: home
hero:
  name: Jev Colab Lab
  text: Reproducible decision-model experiments
  tagline: Five pinned GPU experiments, sanitized evidence, and practical Colab reproduction paths.
  image:
    src: /jev-colab-lab-icon.svg
    alt: Jev Colab Lab geometric decision graph icon
  actions:
    - theme: brand
      text: Start with the guide
      link: /guide/getting-started
    - theme: alt
      text: 日本語
      link: /ja/
    - theme: alt
      text: Browse the source
      link: https://github.com/Sunwood-ai-labs/jev-colab-lab
features:
  - icon: 🎯
    title: Pinned sources
    details: Every experiment records upstream repositories, model revisions, licenses, and intended GPU target.
  - icon: 📏
    title: Separated measurements
    details: Loading, first inference, warmup, steady state, and peak VRAM stay distinct where the runner supports them.
  - icon: 🧼
    title: Sanitized artifacts
    details: Results exclude credentials, session metadata, private inputs, and model weights.
---

## Snapshot

<div class="lab-grid">
  <div class="lab-card"><strong>3 × T4</strong><p>Laya, Kev-0.5B, and Jevlike short training/inference.</p></div>
  <div class="lab-card"><strong>2 × L4</strong><p>SemIf direct option readout and OpenJev NLI.</p></div>
  <div class="lab-card"><strong>5 result sets</strong><p>Each result is linked to a checked-in JSON record and experiment README.</p></div>
  <div class="lab-card"><strong>5 JevDash captures</strong><p>Independent single-episode observations; not a completed ability comparison.</p></div>
</div>

The five experiments are intentionally heterogeneous. This site is a reproducibility index, not a leaderboard: fixtures, readouts, model sizes, and probability semantics differ.

## Continue

- [Getting started](/guide/getting-started) — local checks, uv, WSL, and Colab boundaries.
- [Experiments](/guide/experiments) — what each runner measures and where its evidence lives.
- [Reproducibility](/guide/reproducibility) — session isolation, sanitized outputs, and result interpretation.
- [Sources and licenses](/guide/sources-and-licenses) — upstream revision and license map.

<div class="tip custom-block">
  <p class="custom-block-title">Current measurement note</p>
  <p>Jevlike timing fields keep their measurement boundary in the result schema and are not used as cross-experiment comparison points.</p>
</div>
