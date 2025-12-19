---
sidebar_label: 'ROS 2 Communication Model'
---

# ROS 2 Communication Model

## Overview

The ROS 2 communication model is built on a distributed system architecture that enables flexible and robust robot software development. Understanding this model is crucial for developing effective humanoid robotics applications.

## Nodes

A **node** is a fundamental entity in ROS 2 that performs computation. Nodes are organized into packages to form the functionality of a ROS-based system. In humanoid robotics, different nodes might handle:

- Walking control
- Vision processing
- Arm control
- Sensor data processing
- High-level decision making

Nodes communicate with each other using various communication methods:

### Creating a Node with rclpy

Here's a basic example of creating a ROS 2 node using Python (rclpy):

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller node initialized')

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics and Publishers/Subscribers

**Topics** enable asynchronous, many-to-many communication through a publish/subscribe pattern. This is ideal for continuous data streams like sensor data or robot state information.

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Sensor reading: %d' % self.get_clock().now().nanoseconds
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    data_subscriber = DataSubscriber()

    try:
        rclpy.spin(data_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        data_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services

**Services** provide synchronous request/response communication, which is ideal for operations that need confirmation or return specific results. In humanoid robotics, services might be used for:

- Requesting specific robot poses
- Calibration procedures
- Emergency stops
- Configuration changes

### Service Example

Service definition (`AddTwoInts.srv`):
```
int64 a
int64 b
---
int64 sum
```

Service server:
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsService(Node):
    def __init__(self):
        super().__init__('add_two_ints_service')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(
            'Incoming request: %d + %d = %d' %
            (request.a, request.b, response.sum)
        )
        return response

def main(args=None):
    rclpy.init(args=args)
    add_two_ints_service = AddTwoIntsService()

    try:
        rclpy.spin(add_two_ints_service)
    except KeyboardInterrupt:
        pass
    finally:
        add_two_ints_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Service client:
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    add_two_ints_client = AddTwoIntsClient()
    response = add_two_ints_client.send_request(2, 3)
    add_two_ints_client.get_logger().info(
        'Result of %d + %d = %d' % (2, 3, response.sum)
    )

    add_two_ints_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

**Actions** provide goal-oriented communication with feedback and status. They're perfect for long-running tasks that might be preempted, such as:

- Navigation to a specific location
- Walking to a target position
- Manipulation tasks
- Complex trajectory execution

## Quality of Service (QoS)

QoS profiles allow fine-tuning communication behavior:

- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or transient local
- **History**: Keep last N messages or keep all
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: How to determine if a publisher is alive

## Message Flow Diagrams

Understanding the message flow between nodes is crucial for humanoid robotics:

### Publisher-Subscriber Pattern
```
Publisher Node           Topic              Subscriber Node
    │                      │                      │
    │ ────────────────────►│────────────────────► │
    │  Publish Message     │  Message Queue       │ Receive Message
    │                      │                      │ Process Data
    │ ◄─────────────────── │◄──────────────────── │
    │  (if ACK required)   │                      │
```

### Service Request-Response
```
Client Node              Service              Server Node
    │                       │                       │
    │ ──── Request ────────►│────────────────────►  │
    │                       │                       │ Process Request
    │                       │                       │
    │ ◄──────────────────── │◄────────── Response ─ │
    │    Response           │                       │
```

## Agent Controller Flow

In humanoid robotics, an agent controller typically follows this flow:

1. **Perception**: Subscribe to sensor topics to understand the environment
2. **Planning**: Use services or actions to plan movements
3. **Control**: Publish commands to actuators
4. **Monitoring**: Subscribe to feedback topics to monitor execution

### Example Agent Controller

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class HumanoidAgentController(Node):
    def __init__(self):
        super().__init__('humanoid_agent_controller')

        # Subscribe to sensor data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Publish commands
        self.command_publisher = self.create_publisher(
            String,
            'robot_commands',
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Humanoid Agent Controller initialized')

    def joint_state_callback(self, msg):
        # Process joint state data
        self.get_logger().info(f'Received joint states for {len(msg.name)} joints')

    def control_loop(self):
        # Implement control logic
        command_msg = String()
        command_msg.data = 'Move to target position'
        self.command_publisher.publish(command_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidAgentController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Exercises

To reinforce your understanding of the ROS 2 communication model, try these exercises:

1. **Node Implementation**: Create a simple ROS 2 node that publishes joint position commands and subscribes to sensor feedback. Test it with the examples provided in this chapter.

2. **Communication Pattern Analysis**: For a humanoid walking task, identify which communication pattern (topic, service, or action) would be most appropriate for:
   - Publishing joint angles to actuators
   - Requesting a specific walking gait
   - Executing a complex walking sequence with feedback

3. **QoS Configuration**: Research and explain how you would configure Quality of Service settings for safety-critical messages in a humanoid robot (e.g., emergency stop commands).

## Summary

The ROS 2 communication model provides a robust foundation for humanoid robotics applications. Understanding nodes, topics, services, and actions is essential for building distributed robot systems. The publish/subscribe pattern is ideal for continuous data streams, while services provide synchronous request/response communication, and actions handle long-running tasks with feedback.

## Next Steps

Continue to the next chapter to learn about [Robot Structure with URDF](./robot-structure-urdf), where we'll explore how to describe robot structure for humanoid robots with simulation readiness guidance.

[Previous: Introduction to ROS 2](./introduction-to-ros2) | [Next: Robot Structure with URDF](./robot-structure-urdf) | [Back to Module 1 Overview](../introduction)