import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'TensorFlow 학습 가이드',
  tagline: '딥러닝의 핵심을 이해하는 완벽한 가이드',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://iCodePoet.github.io',
  baseUrl: '/tensorflow/',

  organizationName: 'iCodePoet',
  projectName: 'tensorflow',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'ko',
    locales: ['ko'],
  },

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/iCodePoet/tensorflow/tree/master/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/tensorflow-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },
    navbar: {
      title: 'TensorFlow 학습 가이드',
      logo: {
        alt: 'TensorFlow Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: '문서',
        },
        {
          href: 'https://github.com/tensorflow/tensorflow',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '문서',
          items: [
            {
              label: '시작하기',
              to: '/docs/intro',
            },
            {
              label: '아키텍처',
              to: '/docs/architecture/overview',
            },
            {
              label: '개념 이해',
              to: '/docs/concepts/glossary',
            },
          ],
        },
        {
          title: '커뮤니티',
          items: [
            {
              label: 'TensorFlow 공식 사이트',
              href: 'https://www.tensorflow.org/',
            },
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/tensorflow',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/tensorflow/tensorflow/discussions',
            },
          ],
        },
        {
          title: '더 보기',
          items: [
            {
              label: 'TensorFlow GitHub',
              href: 'https://github.com/tensorflow/tensorflow',
            },
            {
              label: 'TensorFlow Hub',
              href: 'https://tfhub.dev/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} TensorFlow Authors. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'cpp', 'protobuf', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
