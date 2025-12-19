---
sidebar_label: 'Nav2 for Humanoid Path Planning'
---

# Nav2 for Humanoid Path Planning

## Overview
This chapter covers configuring Nav2 for bipedal humanoid robots, focusing on path planning that accounts for unique kinematics and balance requirements.

## Learning Objectives
- Set up Nav2 for humanoid robot platforms
- Configure humanoid-specific navigation parameters
- Implement footstep planning algorithms
- Plan paths with humanoid kinematic constraints
- Consider center of mass and balance in navigation
- Handle obstacle avoidance for humanoid robots
- Integrate Nav2 with Isaac Sim for testing
- Troubleshoot humanoid navigation issues

## Table of Contents
- [Nav2 Setup for Humanoids](#setup)
- [Humanoid Configuration Parameters](#configuration)
- [Footstep Planning](#footstep-planning)
- [Path Planning with Constraints](#path-planning)
- [Center of Mass Considerations](#com-considerations)
- [Obstacle Avoidance](#obstacle-avoidance)
- [Isaac Sim Integration](#integration)
- [Troubleshooting](#troubleshooting)

## Nav2 Setup for Humanoids {#setup}

Navigation2 (Nav2) is ROS 2's state-of-the-art navigation framework, but it requires specific configuration to work effectively with humanoid robots. Unlike wheeled robots, humanoid robots have unique kinematic and dynamic constraints that must be considered in path planning and navigation.

### Prerequisites

Before configuring Nav2 for humanoid robots, ensure you have:

- **ROS 2**: Humble Hawksbill or later
- **Nav2**: Latest stable release (1.1.0+)
- **Robot Description**: URDF/SDF model with proper kinematic chains
- **Controllers**: Joint trajectory controllers for leg joints
- **Sensors**: IMU, force/torque sensors, and perception sensors
- **Computing**: Sufficient processing power for real-time footstep planning

### Installation

Install Nav2 following the standard installation process:

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-dwb-core ros-humble-dwb-critics ros-humble-dwb-plugins
sudo apt install ros-humble-nav2-rviz-plugins
sudo apt install ros-humble-slam-toolbox  # For mapping if needed
```

### Basic Humanoid Navigation Setup

```yaml
# config/humanoid_nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: False
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: True
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the default behavior tree for humanoid navigation
    default_nav_to_pose_bt_xml: "humanoid_nav_to_pose.xml"
    default_nav_through_poses_bt_xml: "humanoid_nav_through_poses.xml"

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    # Humanoid-specific controllers
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid path follower
    FollowPath:
      plugin: "nav2_mppi_controller::Controller"
      time_steps: 24
      control_freq: 20.0
      horizon: 1.5
      dt: 0.05
      discretization: 0.05
      # Humanoid-specific weights
      x_ref: [2.0, 0.0, 0.0]
      ctrl_coeffs: [50.0]
      input_penalty: [10.0, 10.0, 10.0]
      state_penalty: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      state_forward_weight: 1.0
      reference_heading_weight: 1.0
      transform_tolerance: 0.1
      velocity_scaling_tolerance: 0.5
      max_linear_speed: 0.5  # Reduced for humanoid stability
      max_angular_speed: 0.75

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Humanoid effective radius
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

## Humanoid Configuration Parameters {#configuration}

Humanoid robots require specialized configuration parameters that account for their unique kinematic and dynamic properties.

### Kinematic Constraints

Humanoid robots have several key constraints that affect navigation:

- **Step Length Limitations**: Maximum distance between consecutive footsteps
- **Turning Radius**: Limited by leg span and balance
- **Balance Requirements**: Center of Mass (CoM) must remain within support polygon
- **Foot Clearance**: Minimum height to avoid ground collision

```yaml
# humanoid_specific_params.yaml
controller_server:
  ros__parameters:
    # Humanoid-specific path following parameters
    FollowPath:
      # Maximum step length constraints
      max_step_length: 0.3  # meters
      max_step_width: 0.2   # meters (lateral step)
      max_step_rotation: 0.5  # radians (turning)

      # Balance constraints
      com_stability_margin: 0.1  # Safety margin for CoM
      support_polygon_buffer: 0.05  # Buffer around support polygon

      # Timing constraints for stable walking
      min_step_duration: 0.8  # Minimum time per step
      max_step_duration: 2.0  # Maximum time per step
      step_phase_duration: 0.4  # Duration of each phase (swing, stance)

local_costmap:
  local_costmap:
    ros__parameters:
      # Adjust robot radius for humanoid form
      robot_radius: 0.4  # Larger than typical wheeled robot

      # Specialized inflation for humanoid navigation
      inflation_layer:
        inflation_radius: 0.7  # Larger for safety
        cost_scaling_factor: 2.5  # Balanced cost scaling
```

### Balance Recovery Behaviors

```yaml
# balance_recovery_params.yaml
recovery_server:
  ros__parameters:
    use_sim_time: True
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    global_frame: odom
    robot_base_frame: base_link
    transform_timeout: 0.1
    recovery_plugins: ["spin", "backup", "wait", "balance_recovery"]
    spin:
      plugin: "nav2_recoveries/Spin"
      enabled: True
      frequency: 20.0
      add_to_blackboard: True
    backup:
      plugin: "nav2_recoveries/BackUp"
      enabled: True
      frequency: 20.0
      add_to_blackboard: True
    wait:
      plugin: "nav2_recoveries/Wait"
      enabled: True
      frequency: 20.0
      add_to_blackboard: True
    balance_recovery:
      plugin: "nav2_humanoid_recoveries/BalanceRecovery"
      enabled: True
      frequency: 20.0
      add_to_blackboard: True
      # Balance recovery specific parameters
      max_tilt_angle: 15.0  # Maximum acceptable tilt in degrees
      com_correction_speed: 0.1  # Speed of CoM correction
      recovery_timeout: 10.0
```

## Footstep Planning {#footstep-planning}

Footstep planning is the cornerstone of humanoid navigation, determining where and when each foot should be placed to maintain balance while following a path.

### Basic Footstep Planning

```cpp
// Example footstep planner implementation
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include <vector>

class HumanoidFootstepPlanner {
public:
    HumanoidFootstepPlanner() {
        // Initialize footstep planning parameters
        step_length_ = 0.3;      // Maximum step length
        step_width_ = 0.2;       // Default step width (stance width)
        max_turn_ = 0.5;         // Maximum turning angle per step
        nominal_height_ = 0.08;  // Nominal foot height above ground
    }

    // Plan footsteps along a given path
    std::vector<Footstep> planFootsteps(const nav_msgs::msg::Path& path,
                                       const geometry_msgs::msg::Pose& start_pose) {
        std::vector<Footstep> footsteps;

        // Start with current foot positions
        Footstep left_foot = getCurrentLeftFootPose();
        Footstep right_foot = getCurrentRightFootPose();

        // Determine initial support foot
        bool left_support = isLeftFootSupporting();

        for (size_t i = 0; i < path.poses.size(); ++i) {
            geometry_msgs::msg::Pose target_pose = path.poses[i];

            // Calculate desired foot placement
            auto desired_footstep = calculateDesiredFootstep(
                target_pose, left_foot, right_foot, left_support);

            // Validate footstep for collision and balance
            if (isValidFootstep(desired_footstep)) {
                footsteps.push_back(desired_footstep);

                // Update support foot
                left_support = !left_support;

                // Update foot poses for next iteration
                if (left_support) {
                    left_foot = desired_footstep;
                } else {
                    right_foot = desired_footstep;
                }
            }
        }

        return footsteps;
    }

private:
    struct Footstep {
        geometry_msgs::msg::Pose pose;
        bool is_left_foot;
        double time_from_start;
    };

    double step_length_;
    double step_width_;
    double max_turn_;
    double nominal_height_;

    Footstep calculateDesiredFootstep(const geometry_msgs::msg::Pose& target_pose,
                                    const Footstep& left_foot,
                                    const Footstep& right_foot,
                                    bool left_support) {
        Footstep desired_footstep;

        // Calculate step based on target and current support state
        // This is a simplified example - real implementation would be more complex
        if (left_support) {
            // Right foot needs to move toward target
            desired_footstep.is_left_foot = false;
            desired_footstep.pose.position.x = target_pose.position.x;
            desired_footstep.pose.position.y = target_pose.position.y - step_width_/2.0;
            desired_footstep.pose.position.z = nominal_height_;
        } else {
            // Left foot needs to move toward target
            desired_footstep.is_left_foot = true;
            desired_footstep.pose.position.x = target_pose.position.x;
            desired_footstep.pose.position.y = target_pose.position.y + step_width_/2.0;
            desired_footstep.pose.position.z = nominal_height_;
        }

        return desired_footstep;
    }

    bool isValidFootstep(const Footstep& footstep) {
        // Check for collisions with obstacles
        if (isFootstepInCollision(footstep)) {
            return false;
        }

        // Check for balance constraints
        if (!isBalanceMaintained(footstep)) {
            return false;
        }

        return true;
    }

    bool isFootstepInCollision(const Footstep& footstep) {
        // Check if footstep location collides with obstacles in local costmap
        // Implementation would check costmap values at footstep location
        return false; // Simplified for example
    }

    bool isBalanceMaintained(const Footstep& footstep) {
        // Check if CoM projection remains within support polygon
        // Implementation would calculate support polygon and CoM position
        return true; // Simplified for example
    }

    Footstep getCurrentLeftFootPose() {
        // Get current left foot pose from TF or joint states
        return Footstep{}; // Simplified for example
    }

    Footstep getCurrentRightFootPose() {
        // Get current right foot pose from TF or joint states
        return Footstep{}; // Simplified for example
    }

    bool isLeftFootSupporting() {
        // Determine which foot is currently supporting the robot
        return true; // Simplified for example
    }
};
```

### Advanced Footstep Planning with Terrain Adaptation

For more complex terrain, advanced footstep planning algorithms are needed:

```python
# Python example of terrain-adaptive footstep planning
import numpy as np
from scipy.spatial import KDTree
import math

class TerrainAdaptiveFootstepPlanner:
    def __init__(self):
        self.max_step_length = 0.3
        self.max_step_width = 0.2
        self.max_step_height_diff = 0.1  # Maximum height difference between steps
        self.foot_size = 0.15  # Approximate foot size for stability

    def plan_footsteps_terrain_adaptive(self, path, robot_pose, terrain_height_map):
        """
        Plan footsteps considering terrain height variations
        """
        footsteps = []

        # Start with current stance foot
        left_foot_pos, right_foot_pos = self.get_current_foot_positions()
        stance_foot = 'left' if self.is_left_supporting() else 'right'

        for i in range(len(path)):
            target_pos = path[i]

            # Calculate terrain height at target location
            target_height = self.get_terrain_height(target_pos, terrain_height_map)

            # Determine swing foot based on current stance
            swing_foot = 'right' if stance_foot == 'left' else 'left'

            # Plan next footstep considering terrain
            next_footstep = self.calculate_terrain_aware_footstep(
                target_pos, target_height, left_foot_pos, right_foot_pos,
                stance_foot, terrain_height_map
            )

            # Validate footstep
            if self.validate_terrain_footstep(next_footstep, terrain_height_map):
                footsteps.append(next_footstep)

                # Update stance foot for next step
                stance_foot = swing_foot

                # Update foot positions
                if swing_foot == 'left':
                    left_foot_pos = next_footstep
                else:
                    right_foot_pos = next_footstep

        return footsteps

    def calculate_terrain_aware_footstep(self, target_pos, target_height,
                                       left_pos, right_pos, stance_foot, terrain_map):
        """
        Calculate footstep considering terrain constraints
        """
        # Determine desired position based on path and current stance
        if stance_foot == 'left':
            # Right foot should move toward target
            desired_x = target_pos[0]
            desired_y = target_pos[1] - self.max_step_width/2
        else:
            # Left foot should move toward target
            desired_x = target_pos[0]
            desired_y = target_pos[1] + self.max_step_width/2

        # Get terrain height at desired position
        terrain_height = self.get_terrain_height([desired_x, desired_y], terrain_map)

        # Create footstep with terrain-appropriate height
        footstep = {
            'position': [desired_x, desired_y, terrain_height],
            'stance_foot': stance_foot,
            'swing_foot': 'right' if stance_foot == 'left' else 'left'
        }

        return footstep

    def validate_terrain_footstep(self, footstep, terrain_map):
        """
        Validate footstep considering terrain constraints
        """
        x, y, z = footstep['position']

        # Check if position is on traversable terrain
        if not self.is_traversable(x, y, terrain_map):
            return False

        # Check slope constraints
        slope = self.calculate_local_slope(x, y, terrain_map)
        if slope > math.radians(15):  # Maximum 15 degree slope
            return False

        # Check for obstacles at foot placement
        if self.has_obstacles_around(x, y, terrain_map):
            return False

        return True

    def get_terrain_height(self, pos, terrain_map):
        """
        Get terrain height at a specific position
        """
        # Interpolate terrain height from height map
        # Implementation would depend on terrain_map format
        return 0.0  # Simplified

    def is_traversable(self, x, y, terrain_map):
        """
        Check if terrain at (x,y) is traversable
        """
        # Check costmap or terrain properties
        return True  # Simplified

    def calculate_local_slope(self, x, y, terrain_map):
        """
        Calculate local terrain slope at position
        """
        # Calculate slope from height map gradients
        return 0.0  # Simplified

    def has_obstacles_around(self, x, y, terrain_map):
        """
        Check for obstacles around foot placement area
        """
        # Check for obstacles in foot area
        return False  # Simplified
```

## Path Planning with Constraints {#path-planning}

Path planning for humanoid robots must account for their unique kinematic constraints, including step limitations, balance requirements, and bipedal locomotion characteristics.

### Humanoid-Aware Global Planner

The global planner needs to generate paths that are feasible for humanoid locomotion:

```yaml
# config/humanoid_global_planner.yaml
global_costmap:
  global_costmap:
    ros__parameters:
      plugins: ["static_layer", "obstacle_layer", "inflation_layer", "humanoid_layer"]
      # ... other layers ...
      humanoid_layer:
        plugin: "nav2_humanoid_costmap_plugins::HumanoidLayer"
        enabled: True
        # Humanoid-specific inflation parameters
        step_length_limit: 0.3
        turning_radius_limit: 0.5
        support_polygon_buffer: 0.1
        balance_margin: 0.15

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5  # Increased tolerance for humanoid path feasibility
      use_astar: false
      allow_unknown: true
```

### Local Path Planning with Step Constraints

```cpp
// Example of humanoid-aware local planner
#include "nav2_core/local_planner.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_core/trajectory_generator.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

class HumanoidLocalPlanner : public nav2_core::LocalPlanner {
public:
    void initialize(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        const std::string & name,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override {

        parent_ = parent;
        name_ = name;
        costmap_ros_ = costmap_ros;

        // Initialize humanoid-specific parameters
        max_step_length_ = 0.3;
        max_step_width_ = 0.2;
        min_step_duration_ = 0.8;
        com_stability_margin_ = 0.1;
    }

    geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::PoseStamped & goal,
        const geometry_msgs::msg::Twist & velocity) override {

        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.frame_id = pose.header.frame_id;
        cmd_vel.header.stamp = node_->now();

        // Calculate humanoid-appropriate velocity based on step constraints
        auto desired_velocity = calculateHumanoidVelocity(pose, goal, velocity);

        cmd_vel.twist = desired_velocity;

        return cmd_vel;
    }

private:
    double max_step_length_;
    double max_step_width_;
    double min_step_duration_;
    double com_stability_margin_;
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
    std::string name_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;

    geometry_msgs::msg::Twist calculateHumanoidVelocity(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::PoseStamped & goal,
        const geometry_msgs::msg::Twist & current_vel) {

        geometry_msgs::msg::Twist result_vel;

        // Calculate distance to goal
        double dist_to_goal = calculateDistance(pose.pose, goal.pose);

        // If close to goal, slow down for precise positioning
        if (dist_to_goal < 0.5) {
            result_vel.linear.x = std::min(0.2, dist_to_goal * 0.5);
        } else {
            // Calculate max speed based on step constraints
            double max_speed = max_step_length_ / min_step_duration_;
            result_vel.linear.x = std::min(max_speed, 0.5); // Cap at 0.5 m/s for stability
        }

        // Calculate angular velocity with turning constraints
        double angle_to_goal = calculateAngleToGoal(pose.pose, goal.pose);
        double max_angular = max_step_width_ / (min_step_duration_ * 0.5); // Conservative turning
        result_vel.angular.z = std::max(std::min(angle_to_goal * 2.0, max_angular), -max_angular);

        return result_vel;
    }

    double calculateDistance(const geometry_msgs::msg::Pose & p1,
                           const geometry_msgs::msg::Pose & p2) {
        double dx = p2.position.x - p1.position.x;
        double dy = p2.position.y - p1.position.y;
        return sqrt(dx*dx + dy*dy);
    }

    double calculateAngleToGoal(const geometry_msgs::msg::Pose & current_pose,
                              const geometry_msgs::msg::Pose & goal_pose) {
        double desired_angle = atan2(goal_pose.position.y - current_pose.position.y,
                                   goal_pose.position.x - current_pose.position.x);
        double current_yaw = getCurrentYaw(current_pose);
        return normalizeAngle(desired_angle - current_yaw);
    }

    double getCurrentYaw(const geometry_msgs::msg::Pose & pose) {
        // Extract yaw from quaternion
        double siny_cosp = 2 * (pose.orientation.w * pose.orientation.z +
                               pose.orientation.x * pose.orientation.y);
        double cosy_cosp = 1 - 2 * (pose.orientation.y * pose.orientation.y +
                                   pose.orientation.z * pose.orientation.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
};
```

## Center of Mass Considerations {#com-considerations}

The Center of Mass (CoM) is critical for humanoid robot stability during navigation. Proper CoM management ensures the robot remains balanced while following paths.

### CoM Estimation and Control

```python
# Example of CoM estimation and control for humanoid navigation
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidCoMController:
    def __init__(self, robot_description):
        self.robot_description = robot_description
        self.com_history = []
        self.max_com_drift = 0.05  # Maximum allowable CoM drift from center
        self.com_buffer = 0.02     # Safety buffer

    def estimate_com(self, joint_states):
        """
        Estimate Center of Mass based on joint positions and link masses
        """
        total_mass = 0.0
        com_x = 0.0
        com_y = 0.0
        com_z = 0.0

        # Calculate CoM as weighted average of link CoMs
        for link_name, link_info in self.robot_description.links.items():
            if 'mass' in link_info and 'com_offset' in link_info:
                mass = link_info['mass']
                # Transform link CoM to world frame based on joint positions
                link_com_world = self.transform_to_world_frame(
                    link_name, link_info['com_offset'], joint_states)

                com_x += mass * link_com_world[0]
                com_y += mass * link_com_world[1]
                com_z += mass * link_com_world[2]
                total_mass += mass

        if total_mass > 0:
            com_x /= total_mass
            com_y /= total_mass
            com_z /= total_mass

        return np.array([com_x, com_y, com_z])

    def calculate_support_polygon(self, left_foot_pose, right_foot_pose):
        """
        Calculate the support polygon based on foot positions
        """
        # For bipedal robot, support polygon is typically a quadrilateral
        # between the two feet, but can be a triangle when stepping
        support_points = []

        # Define foot contact areas
        foot_size_x = 0.15  # Typical humanoid foot length
        foot_size_y = 0.08  # Typical humanoid foot width

        # Left foot contact points
        support_points.extend([
            [left_foot_pose[0] - foot_size_x/2, left_foot_pose[1] - foot_size_y/2],
            [left_foot_pose[0] + foot_size_x/2, left_foot_pose[1] - foot_size_y/2],
            [left_foot_pose[0] + foot_size_x/2, left_foot_pose[1] + foot_size_y/2],
            [left_foot_pose[0] - foot_size_x/2, left_foot_pose[1] + foot_size_y/2]
        ])

        # Right foot contact points
        support_points.extend([
            [right_foot_pose[0] - foot_size_x/2, right_foot_pose[1] - foot_size_y/2],
            [right_foot_pose[0] + foot_size_x/2, right_foot_pose[1] - foot_size_y/2],
            [right_foot_pose[0] + foot_size_x/2, right_foot_pose[1] + foot_size_y/2],
            [right_foot_pose[0] - foot_size_x/2, right_foot_pose[1] + foot_size_y/2]
        ])

        return self.convex_hull(support_points)

    def is_com_stable(self, com_position, support_polygon):
        """
        Check if CoM projection is within support polygon
        """
        com_2d = [com_position[0], com_position[1]]
        return self.point_in_polygon(com_2d, support_polygon)

    def calculate_balance_correction(self, com_position, support_polygon):
        """
        Calculate necessary corrections to maintain balance
        """
        com_2d = [com_position[0], com_position[1]]

        if not self.point_in_polygon(com_2d, support_polygon):
            # CoM is outside support polygon, calculate correction
            nearest_point = self.find_nearest_point_in_polygon(com_2d, support_polygon)
            correction_vector = np.array(nearest_point) - np.array(com_2d)

            # Scale correction to be conservative
            correction_magnitude = np.linalg.norm(correction_vector)
            if correction_magnitude > self.max_com_drift:
                correction_vector = correction_vector * (self.max_com_drift / correction_magnitude)

            return correction_vector
        else:
            # CoM is stable, return small adjustment toward center
            polygon_center = self.polygon_centroid(support_polygon)
            adjustment = 0.1 * (np.array(polygon_center) - np.array(com_2d))
            return adjustment

    def point_in_polygon(self, point, polygon):
        """
        Ray casting algorithm to check if point is in polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def convex_hull(self, points):
        """
        Simple convex hull calculation (Graham scan implementation)
        """
        # Simplified implementation - in practice, use scipy.spatial.ConvexHull
        return points  # Placeholder

    def polygon_centroid(self, polygon):
        """
        Calculate centroid of polygon
        """
        x_sum = sum(p[0] for p in polygon)
        y_sum = sum(p[1] for p in polygon)
        n = len(polygon)
        return [x_sum / n, y_sum / n]

    def find_nearest_point_in_polygon(self, point, polygon):
        """
        Find the nearest point in polygon to the given point
        """
        # For simplicity, return centroid
        return self.polygon_centroid(polygon)

# Integration with Nav2
class HumanoidBalanceController:
    def __init__(self):
        self.com_controller = HumanoidCoMController(self.load_robot_description())
        self.balance_threshold = 0.05  # Meters from support polygon edge
        self.recovery_active = False

    def load_robot_description(self):
        """
        Load robot description from URDF or similar
        """
        # Implementation would load robot description
        return {}

    def check_balance_and_adjust(self, current_state):
        """
        Check balance and return necessary adjustments
        """
        # Estimate current CoM
        current_com = self.com_controller.estimate_com(current_state.joint_states)

        # Get foot positions
        left_foot_pos = self.get_foot_position(current_state, 'left')
        right_foot_pos = self.get_foot_position(current_state, 'right')

        # Calculate support polygon
        support_polygon = self.com_controller.calculate_support_polygon(
            left_foot_pos, right_foot_pos)

        # Check if CoM is stable
        is_stable = self.com_controller.is_com_stable(current_com, support_polygon)

        if not is_stable:
            # Calculate balance correction
            correction = self.com_controller.calculate_balance_correction(
                current_com, support_polygon)

            # Apply correction through joint control or modify navigation commands
            self.apply_balance_correction(correction)
            self.recovery_active = True
        else:
            self.recovery_active = False

        return is_stable

    def get_foot_position(self, state, foot_name):
        """
        Get current foot position from robot state
        """
        # Implementation would get foot position from TF or kinematics
        return [0.0, 0.0]

    def apply_balance_correction(self, correction):
        """
        Apply balance correction through joint control
        """
        # Implementation would send commands to joints to adjust CoM
        pass
```

## Obstacle Avoidance {#obstacle-avoidance}

Obstacle avoidance for humanoid robots requires considering their unique movement patterns and balance constraints.

### Humanoid-Aware Obstacle Avoidance

```yaml
# config/humanoid_obstacle_avoidance.yaml
local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["voxel_layer", "inflation_layer", "obstacle_footprint_layer"]
      obstacle_footprint_layer:
        plugin: "nav2_costmap_2d::ObstacleFootprintLayer"
        enabled: True
        footprint_padding: 0.1  # Extra padding for humanoid stability
        max_obstacle_height: 1.8  # Consider obstacles up to humanoid height
        obstacle_range: 3.0
        raytrace_range: 4.0

controller_server:
  ros__parameters:
    # Humanoid-specific obstacle avoidance parameters
    FollowPath:
      plugin: "nav2_mppi_controller::Controller"
      # Adjust for obstacle avoidance while maintaining balance
      xy_goal_tolerance: 0.3  # Slightly larger for humanoid step constraints
      yaw_goal_tolerance: 0.3
      # Obstacle avoidance weights
      obstacle_weight: 50.0  # High weight for obstacle avoidance
      balance_weight: 30.0   # Maintain balance while avoiding
      smoothness_weight: 10.0 # Smooth transitions for stable walking
```

### Dynamic Obstacle Avoidance for Humanoids

```cpp
// Example of dynamic obstacle avoidance for humanoid robots
#include "rclcpp/rclcpp.hpp"
#include "nav2_core/controller.hpp"
#include "nav_2d_utils/parameters.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include <vector>

class HumanoidObstacleAvoidance {
public:
    HumanoidObstacleAvoidance() {
        max_linear_speed_ = 0.3;  // Conservative for humanoid stability
        max_angular_speed_ = 0.5;
        obstacle_threshold_ = 0.8;  // Detect obstacles early
        avoidance_distance_ = 0.5;  // Minimum distance to maintain
    }

    geometry_msgs::msg::Twist calculateAvoidanceVelocity(
        const geometry_msgs::msg::Twist& desired_velocity,
        const sensor_msgs::msg::LaserScan& scan_data) {

        geometry_msgs::msg::Twist result_velocity = desired_velocity;

        // Analyze scan data for obstacles
        auto obstacle_info = analyzeObstacles(scan_data);

        if (obstacle_info.closest_distance < obstacle_threshold_) {
            // Obstacles detected, calculate avoidance maneuver
            result_velocity = calculateAvoidanceManeuver(
                desired_velocity, obstacle_info);
        }

        // Apply humanoid-specific constraints
        result_velocity = applyHumanoidConstraints(result_velocity);

        return result_velocity;
    }

private:
    struct ObstacleInfo {
        double closest_distance;
        double closest_angle;
        std::vector<double> ranges;
        std::vector<double> angles;
    };

    double max_linear_speed_;
    double max_angular_speed_;
    double obstacle_threshold_;
    double avoidance_distance_;

    ObstacleInfo analyzeObstacles(const sensor_msgs::msg::LaserScan& scan) {
        ObstacleInfo info;
        info.closest_distance = std::numeric_limits<double>::max();
        info.closest_angle = 0.0;

        for (size_t i = 0; i < scan.ranges.size(); ++i) {
            double range = scan.ranges[i];
            double angle = scan.angle_min + i * scan.angle_increment;

            if (range < info.closest_distance && range > scan.range_min) {
                info.closest_distance = range;
                info.closest_angle = angle;
            }

            if (range < avoidance_distance_) {
                info.ranges.push_back(range);
                info.angles.push_back(angle);
            }
        }

        return info;
    }

    geometry_msgs::msg::Twist calculateAvoidanceManeuver(
        const geometry_msgs::msg::Twist& desired_vel,
        const ObstacleInfo& obstacle_info) {

        geometry_msgs::msg::Twist avoidance_vel;

        // Determine avoidance direction based on obstacle distribution
        double left_obstacle_density = countObstaclesInSector(obstacle_info, -M_PI/2, 0);
        double right_obstacle_density = countObstaclesInSector(obstacle_info, 0, M_PI/2);

        // Choose direction with fewer obstacles
        double avoidance_direction = (left_obstacle_density < right_obstacle_density) ? 1.0 : -1.0;

        // Reduce forward speed when avoiding
        avoidance_vel.linear.x = std::max(0.0, desired_vel.linear.x * 0.3);
        avoidance_vel.angular.z = avoidance_direction * max_angular_speed_ * 0.7;

        return avoidance_vel;
    }

    double countObstaclesInSector(const ObstacleInfo& info, double start_angle, double end_angle) {
        double count = 0.0;
        for (size_t i = 0; i < info.angles.size(); ++i) {
            if (info.angles[i] >= start_angle && info.angles[i] <= end_angle) {
                // Weight closer obstacles more heavily
                count += 1.0 / (info.ranges[i] + 0.1);
            }
        }
        return count;
    }

    geometry_msgs::msg::Twist applyHumanoidConstraints(
        const geometry_msgs::msg::Twist& input_vel) {

        geometry_msgs::msg::Twist constrained_vel = input_vel;

        // Apply humanoid-specific speed limits
        constrained_vel.linear.x = std::max(-max_linear_speed_,
                                          std::min(max_linear_speed_, input_vel.linear.x));
        constrained_vel.angular.z = std::max(-max_angular_speed_,
                                           std::min(max_angular_speed_, input_vel.angular.z));

        // Ensure minimum forward motion for stability when turning
        if (std::abs(input_vel.angular.z) > 0.2 && std::abs(input_vel.linear.x) < 0.1) {
            constrained_vel.linear.x = 0.1;  // Maintain minimum forward motion
        }

        return constrained_vel;
    }
};
```

## Isaac Sim Integration {#integration}

Integrating Nav2 with Isaac Sim allows for testing and validation of humanoid navigation in photorealistic environments.

### Isaac Sim Navigation Testing Setup

```python
# Example of integrating Nav2 with Isaac Sim for humanoid navigation testing
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
import carb

class IsaacSimHumanoidNavTester:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.humanoid_robot = None
        self.nav2_interface = None

    def setup_simulation_environment(self):
        """
        Set up Isaac Sim environment for humanoid navigation testing
        """
        # Add a test environment
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets path")
            return False

        # Add a simple office environment
        env_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=env_path, prim_path="/World/Room")

        # Set up lighting and physics
        self.world.scene.add_default_ground_plane()

        return True

    def setup_humanoid_robot(self, robot_usd_path):
        """
        Add humanoid robot to simulation
        """
        # Add humanoid robot to stage
        add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/Humanoid")

        # Initialize robot in simulation
        self.humanoid_robot = self.world.scene.add(
            # Robot initialization code here
        )

        return True

    def connect_to_nav2(self):
        """
        Connect Isaac Sim to Nav2 for navigation commands
        """
        # Initialize ROS 2 interface to communicate with Nav2
        import rclpy
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Path, Odometry
        from sensor_msgs.msg import LaserScan

        rclpy.init()

        # Create subscribers and publishers for Nav2 communication
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        return True

    def run_navigation_test(self, start_pose, goal_pose):
        """
        Run a navigation test in Isaac Sim with Nav2
        """
        # Send goal to Nav2
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = goal_pose[0]
        goal_msg.pose.position.y = goal_pose[1]
        goal_msg.pose.position.z = goal_pose[2]
        goal_msg.pose.orientation.w = 1.0  # Simplified orientation

        self.nav_goal_pub.publish(goal_msg)

        # Monitor navigation progress in simulation
        initial_distance = self.calculate_distance_to_goal()
        start_time = self.world.current_time

        while rclpy.ok():
            # Step simulation
            self.world.step(render=True)

            # Check if navigation is complete
            if self.is_navigation_complete():
                break

            # Check for timeout
            if self.world.current_time - start_time > 300.0:  # 5 minute timeout
                print("Navigation test timed out")
                break

        # Calculate results
        final_distance = self.calculate_distance_to_goal()
        execution_time = self.world.current_time - start_time

        return {
            'success': final_distance < 0.5,  # Within 0.5m of goal
            'time': execution_time,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'path_efficiency': self.calculate_path_efficiency()
        }

    def calculate_distance_to_goal(self):
        """
        Calculate distance from robot to goal in simulation
        """
        # Implementation would get robot position and calculate distance
        return 0.0

    def is_navigation_complete(self):
        """
        Check if navigation has completed successfully
        """
        # Implementation would check if robot reached goal
        return False

    def calculate_path_efficiency(self):
        """
        Calculate how efficiently the path was followed
        """
        # Implementation would compare planned vs executed path
        return 1.0

# Example usage
def main():
    tester = IsaacSimHumanoidNavTester()

    if not tester.setup_simulation_environment():
        print("Failed to set up simulation environment")
        return

    if not tester.setup_humanoid_robot("path/to/humanoid_robot.usd"):
        print("Failed to set up humanoid robot")
        return

    if not tester.connect_to_nav2():
        print("Failed to connect to Nav2")
        return

    # Define test scenario
    start_pose = [0.0, 0.0, 0.0]
    goal_pose = [5.0, 3.0, 0.0]

    # Run navigation test
    results = tester.run_navigation_test(start_pose, goal_pose)

    print(f"Navigation test results: {results}")

if __name__ == "__main__":
    main()
```

### Isaac Sim to Nav2 Configuration

```yaml
# config/isaac_sim_nav2_bridge.yaml
# Configuration for bridging Isaac Sim sensors to Nav2

isaac_ros_navigation_bridge:
  ros__parameters:
    # Map topic mappings from Isaac Sim to Nav2
    laser_scan_topic: "/isaac_sim/lidar_scan"
    imu_topic: "/isaac_sim/imu"
    joint_states_topic: "/isaac_sim/joint_states"
    tf_prefix: "humanoid_sim"

    # Navigation parameters specific to simulation
    simulation_scale: 1.0  # 1:1 scale for accurate testing
    time_scale: 1.0        # Real-time simulation

    # Sensor configuration for simulated humanoid
    sensor_config:
      lidar:
        topic: "/isaac_sim/lidar_scan"
        frame_id: "lidar_link"
        update_rate: 10.0
      imu:
        topic: "/isaac_sim/imu"
        frame_id: "imu_link"
        update_rate: 100.0
      odometry:
        topic: "/isaac_sim/odom"
        frame_id: "odom"
        child_frame_id: "base_link"
        update_rate: 50.0

# Nav2 configuration for simulation
bt_navigator:
  ros__parameters:
    # Use simulation-specific behavior tree
    default_nav_to_pose_bt_xml: "humanoid_sim_nav_to_pose.xml"

# Recovery behaviors for simulation
recovery_server:
  ros__parameters:
    # Additional recovery behaviors for simulation
    recovery_plugins: ["spin", "backup", "wait", "balance_recovery", "sim_reset_recovery"]
    sim_reset_recovery:
      plugin: "nav2_sim_recovery/SimResetRecovery"
      enabled: True
      frequency: 1.0
      add_to_blackboard: True
      reset_timeout: 5.0
```

## Troubleshooting {#troubleshooting}

### Common Setup Issues

**Nav2 Installation Problems**:
- Ensure ROS 2 Humble is properly installed
- Verify all Nav2 dependencies are installed
- Check that system meets minimum requirements

**Humanoid Model Issues**:
```bash
# Verify URDF is valid
check_urdf /path/to/humanoid.urdf

# Check joint limits and kinematics
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

### Navigation Problems

**Path Planning Failures**:
- Verify costmap inflation parameters are appropriate for humanoid size
- Check that global planner can find paths with humanoid constraints
- Ensure map resolution is fine enough for humanoid navigation

**Balance Issues During Navigation**:
- Reduce navigation speed for better stability
- Adjust step timing parameters
- Verify CoM estimation is accurate

**Footstep Planning Problems**:
- Check that footstep planner is receiving proper sensor data
- Verify terrain analysis is working correctly
- Ensure step constraints are appropriate for the robot

### Performance Optimization

**Computational Performance**:
```bash
# Monitor CPU usage during navigation
htop

# Monitor ROS 2 node performance
ros2 run top top_node

# Check for message delays
ros2 topic hz /odom
```

**Real-time Performance**:
- Use real-time kernel for consistent timing
- Optimize footstep planning algorithms
- Reduce sensor processing overhead where possible

### Debugging Tools

**Visualization**:
```bash
# Launch RViz with humanoid navigation configuration
ros2 launch nav2_bringup rviz_launch.py

# Use Nav2's built-in tools
ros2 run nav2_util show_parameters
```

**Logging**:
```bash
# Increase logging level for detailed debugging
ros2 param set /controller_server ros__parameters.log_level DEBUG

# Monitor specific topics
ros2 topic echo /humanoid_footsteps
ros2 topic echo /balance_status
```

**Simulation Debugging**:
- Use Isaac Sim's physics visualization tools
- Monitor joint torques and forces
- Visualize CoM trajectory and support polygons

## Cross-References to Related Topics

For additional information on related topics, see:

- [Isaac Sim for Photorealistic Simulation](isaac-sim-photorealistic-simulation.md) - For generating synthetic data to train perception models
- [Isaac ROS for VSLAM and Navigation](isaac-ros-vslam-navigation.md) - For implementing hardware-accelerated Visual SLAM with Isaac ROS packages
- [Physics Simulation in Gazebo](../module-2-digital-twin/physics-simulation-gazebo.md) - For understanding physics simulation fundamentals
- [ROS 2 Communication Model](../ros2/ros2-communication-model.md) - For understanding ROS 2 fundamentals that work with Nav2

## Summary

This chapter has covered the fundamentals of configuring Nav2 for bipedal humanoid robots, accounting for unique kinematics and balance requirements of two-legged locomotion. These capabilities complete the full perception-to-action pipeline for humanoid navigation.