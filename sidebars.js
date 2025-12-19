// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'introduction',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 Humanoid System',
      items: [
        'ros2/introduction-to-ros2',
        'ros2/ros2-communication-model',
        'ros2/robot-structure-urdf',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Additional Modules',
      items: [
        'placeholder',
      ],
      collapsed: false,
    },
  ],
};

export default sidebars;