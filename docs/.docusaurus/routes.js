import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/tensorflow/docs',
    component: ComponentCreator('/tensorflow/docs', '982'),
    routes: [
      {
        path: '/tensorflow/docs',
        component: ComponentCreator('/tensorflow/docs', '260'),
        routes: [
          {
            path: '/tensorflow/docs',
            component: ComponentCreator('/tensorflow/docs', '137'),
            routes: [
              {
                path: '/tensorflow/docs/api/python-api',
                component: ComponentCreator('/tensorflow/docs/api/python-api', '3de'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tensorflow/docs/architecture/overview',
                component: ComponentCreator('/tensorflow/docs/architecture/overview', '466'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tensorflow/docs/code-walkthrough/reading-order',
                component: ComponentCreator('/tensorflow/docs/code-walkthrough/reading-order', 'b8c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tensorflow/docs/concepts/glossary',
                component: ComponentCreator('/tensorflow/docs/concepts/glossary', '84f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tensorflow/docs/concepts/math-foundations',
                component: ComponentCreator('/tensorflow/docs/concepts/math-foundations', 'b0b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tensorflow/docs/config/environment',
                component: ComponentCreator('/tensorflow/docs/config/environment', 'f11'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tensorflow/docs/intro',
                component: ComponentCreator('/tensorflow/docs/intro', '3eb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tensorflow/docs/usage/getting-started',
                component: ComponentCreator('/tensorflow/docs/usage/getting-started', 'c87'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/tensorflow/',
    component: ComponentCreator('/tensorflow/', '499'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
