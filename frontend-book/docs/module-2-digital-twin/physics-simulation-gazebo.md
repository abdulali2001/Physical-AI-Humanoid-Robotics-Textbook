---
sidebar_label: 'Physics Simulation in Gazebo'
---

# Physics Simulation in Gazebo

## Overview
This chapter covers physics simulation in Gazebo, including gravity, collisions, and physical interactions between robots and their environment. Gazebo is the standard physics simulation environment for robotics, particularly in the ROS ecosystem, providing realistic physics simulation with support for gravity, collision detection, and joint dynamics.

## Learning Objectives
- Understand Gazebo physics simulation capabilities
- Configure gravity and collision detection
- Create basic robot models with physical properties
- Simulate joint dynamics and interactions
- Implement practical examples of physics-based robot behaviors

## Table of Contents
- [Introduction to Gazebo Physics](#introduction)
- [Gravity Simulation in Detail](#gravity-simulation)
- [Collision Detection](#collision-detection)
- [Joint Dynamics and Physical Interactions](#joint-dynamics)
- [Practical Examples](#examples)
- [Troubleshooting Common Issues](#troubleshooting)
- [Configuration File Examples](#config-files)

## Introduction to Gazebo Physics {#introduction}

Gazebo provides realistic physics simulation with support for multiple physics engines including ODE, Bullet, and Simbody. This enables accurate modeling of robot behavior in realistic environments.

### Key Physics Concepts
- Gravity simulation
- Collision detection
- Joint dynamics
- Mass and inertia properties

### Supported Physics Engines
- **ODE (Open Dynamics Engine)**: Default engine, good for most applications
- **Bullet**: Supports more complex collision shapes
- **Simbody**: High-fidelity simulation for complex systems

## Gravity Simulation in Detail {#gravity-simulation}

Gravity simulation is fundamental to realistic robot behavior in Gazebo. The gravitational force affects all objects in the simulation and must be configured properly for accurate results.

### Gravity Configuration in World Files

Gravity is configured at the world level in SDF (Simulation Description Format) files:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="gravity_example">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include ground plane and other models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

The gravity vector is specified in meters per second squared (m/s²). The default Earth gravity is 9.8 m/s² in the negative Z direction (0 0 -9.8).

### Changing Gravity Settings

You can modify gravity to simulate different environments:

- **Earth**: `0 0 -9.8` m/s²
- **Moon**: `0 0 -1.62` m/s²
- **Mars**: `0 0 -3.71` m/s²
- **Zero Gravity**: `0 0 0` m/s² (for space simulation)

### Modifying Gravity During Simulation

Gravity can be changed during simulation using Gazebo services:

```bash
# Change gravity during runtime
gz physics -w default -x 0 -y 0 -z -5.0
```

### Practical Gravity Example

Create a simple world with different gravity values to observe the effects:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="variable_gravity">
    <physics type="ode">
      <gravity>0 0 -5.0</gravity> <!-- Reduced gravity -->
    </physics>

    <!-- Drop a sphere to observe slower fall -->
    <model name="falling_sphere">
      <pose>0 0 10 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.001</ixx>
            <iyy>0.001</iyy>
            <izz>0.001</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

## Collision Detection {#collision-detection}

Collision detection is crucial for realistic interactions between objects in Gazebo. It ensures objects don't pass through each other and enables proper contact physics.

### Collision Geometry Types

Gazebo supports several collision geometry types:

- **Box**: Rectangular collision volumes
- **Sphere**: Spherical collision volumes
- **Cylinder**: Cylindrical collision volumes
- **Capsule**: Capsule-shaped collision volumes
- **Mesh**: Arbitrary triangular mesh collision volumes
- **Plane**: Infinite plane collision surfaces

### Collision Configuration Example

```xml
<link name="collision_link">
  <!-- Visual properties (what you see) -->
  <visual name="visual">
    <geometry>
      <mesh><uri>meshes/robot_body.stl</uri></geometry>
    </visual>
  </visual>

  <!-- Collision properties (physics interaction) -->
  <collision name="collision">
    <geometry>
      <!-- Using simpler geometry for collision to improve performance -->
      <box><size>0.5 0.3 0.4</size></box>
    </geometry>
  </collision>
</link>
```

### Collision Detection Parameters

Key parameters affecting collision detection:

- **Contact Surface Parameters**: Bounce, friction, soft CFM
- **Collision Mesh Resolution**: Detail level affecting performance
- **Collision Detection Algorithm**: Different algorithms for different use cases

## Joint Dynamics and Physical Interactions {#joint-dynamics}

Joints define how robot parts move relative to each other, with constraints on movement and physical properties.

### Joint Types in Gazebo

- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint
- **Fixed**: No movement between links
- **Floating**: 6 DOF movement
- **Planar**: Movement in a plane

### Joint Configuration Example

```xml
<joint name="arm_joint" type="revolute">
  <parent>upper_arm</parent>
  <child>lower_arm</child>
  <axis>
    <xyz>0 1 0</xyz>
    <limit>
      <lower>-1.57</lower>  <!-- -90 degrees -->
      <upper>1.57</upper>   <!-- 90 degrees -->
      <effort>100</effort>
      <velocity>1.0</velocity>
    </limit>
    <dynamics>
      <damping>0.1</damping>
      <friction>0.0</friction>
    </dynamics>
  </axis>
  <physics>
    <ode>
      <limit>
        <cfm>0</cfm>
        <erp>0.2</erp>
      </limit>
      <suspension>
        <cfm>0</cfm>
        <erp>0.2</erp>
      </suspension>
    </ode>
  </physics>
</joint>
```

## Practical Examples {#examples}

### Example 1: Basic Gravity Test

Create a simple simulation to verify gravity is working correctly:

```bash
# Launch Gazebo with an empty world
gazebo --verbose worlds/empty.world

# Spawn a sphere model
gz model -f /usr/share/gazebo-11/models/sphere/model.sdf -m sphere_test -x 0 -y 0 -z 2

# Observe how the sphere falls due to gravity
```

### Example 2: Physics Playground

Create a simple playground world with multiple objects:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="physics_playground">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sphere -->
    <model name="sphere">
      <pose>0 0 2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><sphere><radius>0.1</radius></sphere></geometry>
        </collision>
        <visual name="visual">
          <geometry><sphere><radius>0.1</radius></sphere></geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.001</ixx>
            <iyy>0.001</iyy>
            <izz>0.001</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Box -->
    <model name="box">
      <pose>1 0 2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.001</ixx>
            <iyy>0.001</iyy>
            <izz>0.001</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

## Step-by-Step Tutorials {#tutorials}

### Tutorial 1: Creating Your First Physics Simulation

Follow these steps to create a basic physics simulation in Gazebo:

1. **Create a new world file** - Create a file named `basic_physics.world` in your Gazebo worlds directory
2. **Define the physics engine** - Set up ODE as your physics engine with Earth gravity
3. **Add a ground plane** - Include the standard ground plane model for collision
4. **Create a simple object** - Add a sphere model that will be affected by gravity
5. **Launch the simulation** - Run Gazebo with your new world file

Here's the complete world file to get started:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="basic_physics">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Test object -->
    <model name="test_sphere">
      <pose>0 0 2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>  <!-- Red color -->
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <iyy>0.0001</iyy>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

To run this simulation:
```bash
# Save the file as basic_physics.world, then run:
gazebo basic_physics.world
```

### Tutorial 2: Adding Joint Constraints

Learn how to create joint constraints between objects:

1. **Create a multi-link model** - Define two connected links
2. **Add a joint** - Connect the links with a revolute joint
3. **Configure joint limits** - Set minimum and maximum rotation angles
4. **Test the joint** - Launch the simulation and observe the constrained movement

This creates a simple pendulum-like structure with a rotating joint.

## Troubleshooting Common Issues {#troubleshooting}

### Gravity Not Working
- Check that physics engine is enabled in the world file
- Verify gravity vector is not zero
- Ensure models have proper mass and inertia values

### Objects Falling Through Ground
- Check that ground plane model is properly included
- Verify collision geometry is defined for all objects
- Check for coordinate system mismatches

### Performance Issues
- Simplify collision geometry (use boxes/cylinders instead of complex meshes)
- Reduce physics update rate if high precision isn't needed
- Limit the number of active objects in the simulation

## Configuration File Examples {#config-files}

### Complete Robot Model with Physics Properties

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="physics_demo_robot">
    <!-- Base link -->
    <link name="base_link">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.2</radius><length>0.1</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.2</radius><length>0.1</length></cylinder>
        </geometry>
      </visual>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.2</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Arm link -->
    <link name="arm_link">
      <collision name="collision">
        <geometry>
          <box><size>0.3 0.05 0.05</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.3 0.05 0.05</size></box>
        </geometry>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <iyy>0.01</iyy>
          <izz>0.005</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Joint connecting base and arm -->
    <joint name="base_to_arm" type="revolute">
      <parent>base_link</parent>
      <child>arm_link</child>
      <pose>0.2 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>10</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

## Cross-References to Related Topics

For additional information on related topics, see:

- [Robot Modeling with URDF](../ros2/robot-structure-urdf.md) - Learn how to create robot models that can be used in physics simulations
- [Unity Rendering](unity-rendering.md) - For high-fidelity visual rendering to complement physics simulation
- [Sensor Simulation](sensor-simulation.md) - For adding sensors to your physics-based robot models
- [Introduction to ROS 2](../ros2/introduction-to-ros2.md) - For understanding the robotics framework that works with Gazebo

## Summary

This chapter has covered the fundamentals of physics simulation in Gazebo, with special emphasis on gravity simulation. We've learned how to configure gravity parameters, set up collision detection, and create physically realistic models. These foundational concepts are essential for creating accurate simulation environments for robotics development and testing.