# Quickstart Guide: NVIDIA Isaac AI Robotics Module

**Feature**: 003-isaac-ai-brain
**Created**: 2025-12-17

## Overview
This quickstart guide provides a fast path to getting started with NVIDIA Isaac technologies for AI robotics. It covers the essential setup and basic usage patterns for Isaac Sim, Isaac ROS, and Nav2 for humanoid robots.

## Prerequisites
- NVIDIA RTX GPU (minimum RTX 2060, recommended RTX 3080+)
- CUDA 11.8 or later
- ROS 2 Humble Hawksbill
- NVIDIA Omniverse account for Isaac Sim access

## Setup Steps

### 1. Isaac Sim Installation
1. Download and install NVIDIA Omniverse Launcher
2. Install Isaac Sim application through the launcher
3. Verify GPU compatibility and install appropriate drivers (535+)
4. Launch Isaac Sim and complete initial setup

### 2. Isaac ROS Setup
1. Install ROS 2 Humble Hawksbill
2. Add NVIDIA Isaac ROS package repositories:
   ```bash
   sudo apt update && sudo apt install wget
   sudo wget -O /usr/share/keyrings/nvidia-isaaclabs.gpg https://repo.download.nvidia.com/nvidia-isaaclabs.gpg
   echo "deb [signed-by=/usr/share/keyrings/nvidia-isaaclabs.gpg] https://repo.download.nvidia.com/ $(lsb_release -cs)/main" | sudo tee /etc/apt/sources.list.d/nvidia-isaaclabs.list
   ```
3. Install Isaac ROS packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-isaac-ros-visual-slam
   sudo apt install ros-humble-isaac-ros-stereo-image-proc
   ```

### 3. Nav2 Configuration
1. Install Nav2 packages:
   ```bash
   sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
   ```
2. Install humanoid-specific navigation packages if available
3. Configure robot-specific parameters

## Basic Usage Patterns

### Isaac Sim: Create First Environment
1. Launch Isaac Sim
2. Create a new scene
3. Add basic objects and lighting
4. Configure a virtual camera
5. Run simulation and capture synthetic data

### Isaac ROS: Visual SLAM
1. Launch camera drivers (real or simulated)
2. Run visual SLAM node:
   ```bash
   ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
   ```
3. Visualize results in RViz
4. Monitor performance metrics

### Nav2: Humanoid Path Planning
1. Load map (from Isaac Sim or real-world)
2. Set robot pose and goal
3. Configure humanoid-specific parameters
4. Execute navigation with balance constraints

## Troubleshooting Common Issues

### Isaac Sim Performance
- Reduce scene complexity if FPS is low
- Check GPU memory usage
- Verify CUDA version compatibility

### Isaac ROS VSLAM
- Ensure proper camera calibration
- Check image transport compression
- Verify GPU acceleration is enabled

### Nav2 Navigation
- Validate map quality and resolution
- Check robot kinematic constraints
- Verify costmap parameters

## Next Steps
1. Complete the Isaac Sim photorealistic simulation chapter
2. Implement Isaac ROS VSLAM examples
3. Configure Nav2 for your specific humanoid robot
4. Integrate the complete perception-to-navigation pipeline