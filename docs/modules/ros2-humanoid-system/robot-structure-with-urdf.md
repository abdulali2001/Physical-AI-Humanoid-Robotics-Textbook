---
sidebar_label: 'Robot Structure with URDF'
---

# Robot Structure with URDF

## Understanding URDF

URDF (Unified Robot Description Format) is an XML-based format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including:

- Links: Rigid bodies with physical properties
- Joints: Connections between links with kinematic properties
- Materials: Visual appearance definitions
- Collision properties: Collision detection properties
- Inertial properties: Mass, center of mass, and inertia tensor

URDF is essential for humanoid robotics as it allows simulation, visualization, and control of complex robot structures.

## Basic URDF Structure

A basic URDF file follows this structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Define materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <!-- Define links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Define joints -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

## URDF Elements for Humanoid Robots

### Links

Links represent rigid bodies in the robot. For humanoid robots, typical links include:

- Torso/Body
- Head
- Arms (upper arm, lower arm, hand)
- Legs (thigh, shank, foot)
- Spine segments

Each link has three main components:

1. **Visual**: How the link appears in simulation and visualization
2. **Collision**: How the link interacts with other objects in collision detection
3. **Inertial**: Physical properties for dynamics simulation

### Joints

Joints define the connection between links. For humanoid robots, common joint types include:

- **revolute**: Rotational joint with limited range
- **continuous**: Rotational joint without limits
- **prismatic**: Linear sliding joint
- **fixed**: No movement between links
- **floating**: 6 DOF movement
- **planar**: Movement in a plane

Joint parameters include:
- Parent and child links
- Joint origin (position and orientation)
- Joint axis
- Limits (for revolute joints)
- Dynamics properties (damping, friction)

## Humanoid-Specific URDF Examples

### Simple Humanoid Torso

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <material name="light_grey">
    <color rgba="0.8 0.8 0.8 1.0"/>
  </material>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>
</robot>
```

### Arm Structure

```xml
<!-- Right upper arm -->
<link name="upper_arm_r">
  <visual>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
    <origin xyz="0 0 -0.15" rpy="0 1.57079632679 0"/>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
    <origin xyz="0 0 -0.15" rpy="0 1.57079632679 0"/>
  </collision>
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

<!-- Right shoulder joint -->
<joint name="shoulder_r_joint" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm_r"/>
  <origin xyz="0.2 0 0.2" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
</joint>

<!-- Right lower arm -->
<link name="lower_arm_r">
  <visual>
    <geometry>
      <cylinder length="0.25" radius="0.04"/>
    </geometry>
    <origin xyz="0 0 -0.125" rpy="0 1.57079632679 0"/>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.25" radius="0.04"/>
    </geometry>
    <origin xyz="0 0 -0.125" rpy="0 1.57079632679 0"/>
  </collision>
  <inertial>
    <mass value="1.5"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
  </inertial>
</link>

<!-- Right elbow joint -->
<joint name="elbow_r_joint" type="revolute">
  <parent link="upper_arm_r"/>
  <child link="lower_arm_r"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.35" effort="100" velocity="1.0"/>
</joint>
```

## Simulation Readiness Guidelines

### Visual and Collision Separation

For simulation efficiency, consider using simpler collision geometries than visual geometries:

```xml
<link name="complex_visual_simple_collision">
  <!-- Complex visual model -->
  <visual>
    <geometry>
      <mesh filename="package://robot_description/meshes/complex_shape.stl"/>
    </geometry>
  </visual>

  <!-- Simple collision model -->
  <collision>
    <geometry>
      <cylinder length="0.2" radius="0.1"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

Accurate inertial properties are crucial for realistic simulation:

```xml
<inertial>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <mass value="5.0"/>
  <!-- Diagonal elements of inertia tensor -->
  <inertia ixx="0.1" ixy="0.0" ixz="0.0"
           iyy="0.1" iyz="0.0"
           izz="0.1"/>
</inertial>
```

### Joint Limits and Safety

Always specify appropriate joint limits to prevent damage in simulation and real robots:

```xml
<joint name="safe_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <!-- Specify limits to protect the robot -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  <!-- Add safety margins -->
  <safety_controller k_position="20" k_velocity="400"
                    soft_lower_limit="-1.5" soft_upper_limit="1.5"/>
</joint>
```

## URDF Validation

To validate your URDF model:

1. **Parse Check**: Ensure the XML is well-formed
2. **Kinematic Chain**: Verify all links are connected through joints
3. **Inertial Properties**: Check that all links have proper inertial definitions
4. **Joint Limits**: Ensure all revolute joints have appropriate limits
5. **Collision Detection**: Verify collision properties are defined

## Tools for URDF Development

### Command Line Tools

- `check_urdf <urdf_file>`: Validate URDF structure
- `urdf_to_graphiz <urdf_file>`: Generate kinematic chain graph
- `rviz2`: Visualize URDF models in ROS 2

### Visualization

```bash
# Launch RViz to visualize your robot
ros2 run rviz2 rviz2
```

In RViz, add a RobotModel display and set the Robot Description parameter to your robot's description parameter name.

## Best Practices for Humanoid URDF

1. **Use Meaningful Names**: Use descriptive names for links and joints (e.g., `left_foot_joint` instead of `j1`)

2. **Follow Standard Conventions**: Use ROS-Industrial conventions when possible

3. **Consider Mass Distribution**: Distribute mass realistically across links

4. **Include Transmission Elements**: For real robots, include transmission definitions

5. **Use Xacro for Complex Models**: Xacro (XML Macros) can simplify complex URDFs

## Practical Exercises

To reinforce your understanding of URDF for humanoid robots, try these exercises:

1. **URDF Creation**: Create a simple URDF file for a 6-DOF robotic arm with a base, two links, and a simple gripper. Include proper inertial properties and joint limits.

2. **Model Validation**: Use the `check_urdf` command to validate a URDF file (you can use one of the examples from this chapter). Identify and fix any potential issues.

3. **Humanoid Extension**: Extend the simple humanoid torso example from this chapter by adding one arm with at least 2 joints. Consider appropriate joint limits and link properties for realistic movement.

## Summary

URDF provides the essential framework for describing robot structure in ROS 2. For humanoid robots, careful attention to link definitions, joint constraints, and inertial properties is crucial for successful simulation and control. Following best practices ensures your robot model is both accurate and simulation-ready.

## Next Steps

Return to the [Module 1 Overview](../../intro) to explore additional robotics concepts, or continue with other modules in the Physical AI Humanoid Robotics textbook.