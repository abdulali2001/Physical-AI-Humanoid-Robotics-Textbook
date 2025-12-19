# Quickstart Guide: Digital Twin Simulation (Gazebo & Unity)

## Overview
This guide provides a fast path to get started with digital twin simulation using Gazebo and Unity. Follow these steps to begin exploring physics simulation, rendering, and sensor integration.

## Prerequisites
- Basic understanding of robotics concepts
- Familiarity with command line tools
- Access to a computer capable of running Gazebo and Unity (minimum 8GB RAM recommended)

## Environment Setup

### Gazebo Installation
1. Install ROS Noetic or ROS2 (recommended for Gazebo integration)
2. Install Gazebo 11+ (or Garden/Harmonic for ROS2)
3. Verify installation: `gazebo --version`

### Unity Setup
1. Download and install Unity Hub
2. Install Unity 2022.3 LTS or newer
3. Install Unity Robotics Simulation package via Package Manager

### Required Tools
- Git for version control
- Text editor or IDE
- Terminal/command prompt access

## Quick Example: Simple Physics Simulation

### Step 1: Launch Basic Gazebo World
```bash
# Launch an empty world with physics enabled
gazebo --verbose worlds/empty.world
```

### Step 2: Add a Simple Robot Model
1. Create a basic robot model (or use a pre-built one like `rrbot`)
2. Spawn the robot in the simulation
3. Observe physics behavior with gravity and collisions

### Step 3: Configure Basic Sensors
1. Add a simple LiDAR sensor to the robot
2. Launch the robot with sensor plugins enabled
3. View sensor data in RViz or other visualization tools

## Quick Example: Unity Visualization

### Step 1: Import Unity Robotics Package
1. Open Unity Hub and create a new 3D project
2. Go to Window → Package Manager
3. Install "Unity Robotics Simulation" package

### Step 2: Create Basic Environment
1. Create a simple scene with ground plane and objects
2. Set up lighting and camera
3. Import robot model (or create simple representation)

### Step 3: Connect to Simulation Data
1. Use ROS# or Unity Robotics Hub to connect to ROS
2. Subscribe to robot state topics
3. Update Unity visualization based on simulation data

## Key Configuration Files

### Gazebo World File (example.world)
```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="quickstart_world">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Robot Model with Sensors (example_robot.sdf)
```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="quickstart_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry><box><size>0.5 0.5 0.2</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.5 0.5 0.2</size></box></geometry>
      </visual>
      <sensor name="lidar" type="ray">
        <ray><range><min>0.1</min><max>10.0</max></range></ray>
        <plugin name="lidar_controller" filename="libRayPlugin.so"/>
      </sensor>
    </link>
  </model>
</sdf>
```

## Common Commands

### Gazebo Commands
```bash
# Launch with specific world
gazebo my_world.world

# Launch with GUI only (no physics)
gzclient

# Launch server only (no GUI)
gzserver my_world.world
```

### Basic Troubleshooting
1. If Gazebo fails to launch, check graphics drivers
2. If sensors don't publish data, verify plugin configuration
3. If Unity doesn't connect to ROS, check network settings

## Next Steps

After completing this quickstart:

1. **Module 2, Chapter 1**: Deep dive into physics simulation concepts
2. **Module 2, Chapter 2**: Explore high-fidelity rendering in Unity
3. **Module 2, Chapter 3**: Master sensor simulation and integration

## Resources

- [Gazebo Tutorials](http://gazebosim.org/tutorials)
- [Unity Documentation](https://docs.unity3d.com/)
- [ROS Robotics Guide](http://wiki.ros.org/)

## Expected Outcomes

After completing the quickstart, you should be able to:
- Launch basic Gazebo simulations with physics
- Configure simple robot models with sensors
- Understand the connection between simulation and visualization
- Prepare for deeper exploration in the full module