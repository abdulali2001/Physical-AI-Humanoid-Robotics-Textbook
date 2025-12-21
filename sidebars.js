// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'introduction',
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 Humanoid System',
      items: [
        'ros2/introduction-to-ros2',
        'ros2/ros2-communication-model',
        'ros2/robot-structure-urdf',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin Simulation',
      items: [
        'module-2-digital-twin/physics-simulation-gazebo',
        'module-2-digital-twin/unity-rendering',
        'module-2-digital-twin/sensor-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-3-ai-robot-brain/isaac-sim-photorealistic-simulation',
        'module-3-ai-robot-brain/isaac-ros-vslam-navigation',
        'module-3-ai-robot-brain/nav2-humanoid-path-planning',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      collapsed: false,
      items: [
        'module-4-vision-language-action/voice-to-action-with-whisper',
        'module-4-vision-language-action/cognitive-planning-with-llms',
        'module-4-vision-language-action/capstone-project-autonomous-humanoid',
      ],
    },
  ],
};

export default sidebars;