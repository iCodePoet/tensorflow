import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: '개념 이해',
      items: [
        'concepts/glossary',
        'concepts/math-foundations',
      ],
    },
    {
      type: 'category',
      label: '아키텍처',
      items: [
        'architecture/overview',
      ],
    },
    {
      type: 'category',
      label: 'API 레퍼런스',
      items: [
        'api/python-api',
      ],
    },
    {
      type: 'category',
      label: '사용 가이드',
      items: [
        'usage/getting-started',
      ],
    },
    {
      type: 'category',
      label: '환경 설정',
      items: [
        'config/environment',
      ],
    },
    {
      type: 'category',
      label: '코드 분석',
      items: [
        'code-walkthrough/reading-order',
      ],
    },
  ],
};

export default sidebars;
