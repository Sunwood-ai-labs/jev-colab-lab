import { defineConfig } from 'vitepress'
import type { DefaultTheme } from 'vitepress'

const englishNav: DefaultTheme.NavItem[] = [
  {
    text: 'Guide',
    items: [
      { text: 'Getting started', link: '/guide/getting-started' },
      { text: 'Experiments', link: '/guide/experiments' },
      { text: 'Reproducibility', link: '/guide/reproducibility' },
      { text: 'Sources and licenses', link: '/guide/sources-and-licenses' },
    ],
  },
  { text: '日本語', link: '/ja/' },
  { text: 'GitHub', link: 'https://github.com/Sunwood-ai-labs/jev-colab-lab' },
]

const japaneseNav: DefaultTheme.NavItem[] = [
  {
    text: 'ガイド',
    items: [
      { text: 'はじめに', link: '/ja/guide/getting-started' },
      { text: '実験一覧', link: '/ja/guide/experiments' },
      { text: '再現性', link: '/ja/guide/reproducibility' },
      { text: '出典とライセンス', link: '/ja/guide/sources-and-licenses' },
    ],
  },
  { text: 'English', link: '/' },
  { text: 'GitHub', link: 'https://github.com/Sunwood-ai-labs/jev-colab-lab' },
]

const englishSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: 'Guide',
    items: [
      { text: 'Getting started', link: '/guide/getting-started' },
      { text: 'Experiments', link: '/guide/experiments' },
      { text: 'Reproducibility', link: '/guide/reproducibility' },
      { text: 'Sources and licenses', link: '/guide/sources-and-licenses' },
    ],
  },
]

const japaneseSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: 'ガイド',
    items: [
      { text: 'はじめに', link: '/ja/guide/getting-started' },
      { text: '実験一覧', link: '/ja/guide/experiments' },
      { text: '再現性', link: '/ja/guide/reproducibility' },
      { text: '出典とライセンス', link: '/ja/guide/sources-and-licenses' },
    ],
  },
]

export default defineConfig({
  title: 'Jev Colab Lab',
  description: 'Reproducible Google Colab GPU experiments for decision-model inference.',
  lang: 'en-US',
  base: '/jev-colab-lab/',
  cleanUrls: true,
  head: [['link', { rel: 'icon', href: '/jev-colab-lab-icon.svg' }]],
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: { nav: englishNav, sidebar: { '/': englishSidebar } },
    },
    ja: {
      label: '日本語',
      lang: 'ja',
      link: '/ja/',
      themeConfig: {
        nav: japaneseNav,
        sidebar: { '/ja/': japaneseSidebar },
        outline: { label: 'このページ' },
        editLink: {
          pattern: 'https://github.com/Sunwood-ai-labs/jev-colab-lab/edit/main/docs/:path',
          text: 'GitHubでこのページを編集',
        },
        docFooter: { prev: '前のページ', next: '次のページ' },
      },
    },
  },
  themeConfig: {
    logo: '/jev-colab-lab-icon.svg',
    siteTitle: 'Jev Colab Lab',
    nav: englishNav,
    sidebar: { '/': englishSidebar, '/ja/': japaneseSidebar },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Sunwood-ai-labs/jev-colab-lab' },
    ],
    search: { provider: 'local' },
    editLink: {
      pattern: 'https://github.com/Sunwood-ai-labs/jev-colab-lab/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },
    footer: {
      message: 'Sanitized research records; not a production decision service.',
      copyright: 'Copyright © 2026 Sunwood-ai-labs',
    },
  },
})
