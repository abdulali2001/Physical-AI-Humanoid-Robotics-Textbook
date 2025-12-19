# Cognitive Planning with LLMs

## Overview

This chapter covers the implementation of cognitive planning systems that translate natural language commands into executable ROS 2 action sequences. You'll learn to use Large Language Models to parse high-level commands and generate step-by-step robotic behaviors, bridging human intention with robotic execution.

Cognitive planning is the critical capability that comes after voice understanding, enabling the translation of high-level natural language into actionable robotic tasks.

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement cognitive planning systems using LLMs for natural language to ROS 2 action translation
- Design prompt engineering strategies for reliable command translation
- Create mappings between natural language constructs and ROS 2 primitives
- Implement validation and safety checks for generated action sequences
- Achieve at least 90% semantic accuracy in command translation

## Prerequisites

Before starting this chapter, ensure you have:
- Completed Module 1 (ROS 2) and Module 2 (Digital Twin)
- Basic understanding of ROS 2 action libraries and service calls
- Access to an LLM provider (OpenAI, Anthropic, or local models)
- Completed the Voice-to-Action chapter (Module 4, Chapter 1)

## LLM Integration Approaches

### OpenAI GPT Integration

The most straightforward approach is to use OpenAI's GPT models for cognitive planning:

```python
import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def plan_from_natural_language(command: str, robot_capabilities: list = None, environment: dict = None) -> dict:
    """
    Use LLM to convert natural language to action sequence

    Args:
        command: Natural language command to process (e.g., "Clean the room")
        robot_capabilities: List of capabilities robot has (optional)
        environment: Environmental constraints (optional)

    Returns:
        Dictionary with action sequence and metadata
    """
    if robot_capabilities is None:
        robot_capabilities = [
            "navigate_to", "detect_object", "grasp_object", "move_arm",
            "rotate_base", "open_gripper", "close_gripper", "move_base"
        ]

    if environment is None:
        environment = {
            "room_layout": "standard room with furniture",
            "obstacles": [],
            "available_objects": ["cup", "book", "pen", "box"]
        }

    prompt = f"""
    Convert the following natural language command to a sequence of ROS 2 actions.
    Respond in JSON format with an array of actions.

    Available robot capabilities: {', '.join(robot_capabilities)}
    Environmental context: {json.dumps(environment)}

    Command: "{command}"

    Actions should follow this structure:
    {{
        "actions": [
            {{
                "action": "<action_type>",
                "parameters": {{
                    "target": "<object or location if applicable>",
                    "location": "<specific location if applicable>",
                    "orientation": "<orientation if applicable>",
                    "gripper_state": "<open/closed if applicable>"
                }},
                "priority": <integer 1-10>,
                "timeout": <float seconds>,
                "success_criteria": "<condition for success>"
            }}
        ]
    }}

    Example response:
    {{
        "actions": [
            {{
                "action": "navigate_to",
                "parameters": {{"location": "kitchen"}},
                "priority": 5,
                "timeout": 30.0,
                "success_criteria": "robot reaches kitchen area"
            }},
            {{
                "action": "detect_object",
                "parameters": {{"target": "red cup"}},
                "priority": 4,
                "timeout": 10.0,
                "success_criteria": "red cup detected in camera view"
            }},
            {{
                "action": "grasp_object",
                "parameters": {{"target": "red cup"}},
                "priority": 3,
                "timeout": 15.0,
                "success_criteria": "gripper successfully grasps cup"
            }}
        ]
    }}

    Respond with JSON only, no additional text:
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for more consistent responses
            max_tokens=1000
        )

        # Parse the response
        result_text = response.choices[0].message.content.strip()

        # Clean up potential markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text[7:]  # Remove ```json
        if result_text.endswith("```"):
            result_text = result_text[:-3]  # Remove ```

        result = json.loads(result_text)
        result["original_command"] = command
        result["confidence"] = 0.9  # Default confidence for LLM-based planning

        return result
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "actions": [],
            "original_command": command,
            "confidence": 0.1,
            "error": "Failed to parse LLM response as JSON"
        }
    except Exception as e:
        return {
            "actions": [],
            "original_command": command,
            "confidence": 0.0,
            "error": f"LLM call failed: {str(e)}"
        }

# Example usage:
# action_sequence = plan_from_natural_language("Go to the kitchen and bring me the red cup")
# print(json.dumps(action_sequence, indent=2))
```

### Alternative LLM Providers

For those who prefer open-source or alternative models:

```python
# Using Hugging Face Transformers (for local models)
from transformers import pipeline
import torch

def plan_with_local_model(command: str, model_name: str = "gpt2"):
    """
    Use a local model for cognitive planning
    Note: This is a simplified example - real implementation would need
    custom fine-tuning for robotics-specific tasks
    """
    try:
        # Initialize the text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device=0 if torch.cuda.is_available() else -1
        )

        prompt = f"""
        Natural language command: {command}

        Convert this to a sequence of robotic actions in JSON format:
        {{
            "actions": [
                {{
                    "action": "<action_type>",
                    "parameters": {{...}}
                }}
            ]
        }}

        JSON response:
        """

        result = generator(
            prompt,
            max_length=500,
            num_return_sequences=1,
            temperature=0.1,
            pad_token_id=50256  # For GPT-2
        )

        # Extract and parse the generated text
        generated_text = result[0]['generated_text'][len(prompt):]

        # Find JSON in the generated text (simplified approach)
        import re
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)

        if json_match:
            json_str = json_match.group()
            action_sequence = json.loads(json_str)
            return action_sequence
        else:
            return {"actions": [], "error": "Could not extract JSON from response"}

    except Exception as e:
        return {"actions": [], "error": f"Local model failed: {str(e)}"}

# Example usage (uncomment if you have the required models installed):
# result = plan_with_local_model("Pick up the blue box")
```

## Prompt Engineering for Reliable Translation

### Context-Aware Prompting

To improve the reliability of command translation, include contextual information:

```python
def create_context_aware_prompt(command: str, context: dict) -> str:
    """
    Create a context-aware prompt for cognitive planning
    """
    # Default context if none provided
    if context is None:
        context = {
            "robot_state": {
                "current_location": "starting_position",
                "battery_level": 0.8,
                "gripper_status": "open",
                "arm_position": "home"
            },
            "environment": {
                "layout": "standard room",
                "obstacles": [],
                "objects": ["cup", "book", "pen"],
                "navigation_goals": {
                    "kitchen": [1.0, 2.0, 0.0],
                    "bedroom": [-1.0, 1.5, 0.0],
                    "living_room": [0.0, -1.0, 0.0]
                }
            },
            "constraints": {
                "fragile_objects": True,
                "time_limit": 120,  # seconds
                "safety_requirements": ["avoid_people", "stay_in_bounds"]
            }
        }

    prompt = f"""
    You are a cognitive planning system for a robot. Convert the natural language command into a sequence of executable actions.

    CURRENT ROBOT STATE:
    - Location: {context['robot_state']['current_location']}
    - Battery: {context['robot_state']['battery_level'] * 100:.0f}%
    - Gripper: {context['robot_state']['gripper_status']}
    - Arm: {context['robot_state']['arm_position']}

    ENVIRONMENTAL CONTEXT:
    - Layout: {context['environment']['layout']}
    - Available objects: {', '.join(context['environment']['objects'])}
    - Known locations: {list(context['environment']['navigation_goals'].keys())}
    - Obstacles: {context['environment']['obstacles']}

    CONSTRAINTS:
    - Time limit: {context['constraints']['time_limit']} seconds
    - Safety: {', '.join(context['constraints']['safety_requirements'])}
    - Fragile object handling: {context['constraints']['fragile_objects']}

    NATURAL LANGUAGE COMMAND: "{command}"

    INSTRUCTIONS:
    1. Consider the robot's current state and environment
    2. Generate a sequence of actions that accomplishes the goal
    3. Include safety checks and validation steps
    4. Consider time and battery constraints
    5. Use only available objects and locations

    RESPONSE FORMAT:
    {{
        "actions": [
            {{
                "action": "<action_type>",
                "parameters": {{
                    "target": "<object or location>",
                    "location": "[x, y, z]" or "named_location",
                    "orientation": "[roll, pitch, yaw]",
                    "gripper_state": "open/closed",
                    "speed": "slow/medium/fast"
                }},
                "priority": <1-10>,
                "timeout": <seconds>,
                "preconditions": ["list of conditions that must be true"],
                "success_criteria": "<how to verify action succeeded>",
                "error_recovery": "<what to do if action fails>"
            }}
        ],
        "estimated_duration": <seconds>,
        "battery_impact": <0.0-1.0>,
        "safety_risk": "low/medium/high"
    }}

    Respond with JSON only, no additional text:
    """

    return prompt

def plan_with_context(command: str, context: dict = None) -> dict:
    """
    Plan with full contextual awareness
    """
    try:
        prompt = create_context_aware_prompt(command, context)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )

        result_text = response.choices[0].message.content.strip()

        # Clean up potential markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        result = json.loads(result_text)
        result["original_command"] = command
        result["confidence"] = 0.9

        return result
    except Exception as e:
        return {
            "actions": [],
            "original_command": command,
            "confidence": 0.1,
            "error": f"Context-aware planning failed: {str(e)}"
        }
```

### Multi-Step Reasoning Prompts

For complex commands requiring multi-step reasoning:

```python
def plan_with_multi_step_reasoning(command: str) -> dict:
    """
    Use multi-step reasoning for complex commands
    """
    prompt = f"""
    Natural language command: "{command}"

    Think through this step by step:

    1. GOAL ANALYSIS: What is the user trying to accomplish?
    2. ENVIRONMENTAL ANALYSIS: What information do we need about the environment?
    3. SEQUENCE BREAKDOWN: What are the high-level steps required?
    4. ACTION MAPPING: How do these steps map to robot actions?
    5. SAFETY CHECK: What safety considerations apply?

    Now generate the action sequence in JSON format:
    {{
        "reasoning": {{
            "goal": "<what the user wants>",
            "environmental_needs": ["<what we need to know about environment>"],
            "high_level_steps": ["<step 1>", "<step 2>", "<step 3>"],
            "safety_considerations": ["<safety point 1>", "<safety point 2>"]
        }},
        "actions": [
            {{
                "action": "<action_type>",
                "parameters": {{...}},
                "reason": "<why this action is needed>",
                "expected_outcome": "<what should happen after this action>"
            }}
        ]
    }}

    Respond with JSON only, no additional text:
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 for better reasoning capabilities
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        # Clean up potential markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        result = json.loads(result_text)
        result["original_command"] = command
        result["confidence"] = 0.95  # Higher confidence for GPT-4 reasoning

        return result
    except Exception as e:
        return {
            "actions": [],
            "original_command": command,
            "confidence": 0.1,
            "error": f"Multi-step reasoning failed: {str(e)}"
        }
```

## Mapping Natural Language to ROS 2 Primitives

### ROS 2 Action Mapping

Here's how to map the generated action sequences to actual ROS 2 actions:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from nav2_msgs.action import NavigateToPose
from control_msgs.action import GripperCommand

class ROS2ActionMapper:
    def __init__(self, node: Node):
        self.node = node
        self.action_clients = {}

        # Initialize action clients for common robot actions
        self._initialize_action_clients()

    def _initialize_action_clients(self):
        """Initialize ROS 2 action clients"""
        # Navigation client
        self.nav_client = ActionClient(
            self.node,
            NavigateToPose,
            'navigate_to_pose'
        )

        # MoveIt! client for arm movements
        self.move_group_client = ActionClient(
            self.node,
            MoveGroup,
            'move_group'
        )

        # Gripper client
        self.gripper_client = ActionClient(
            self.node,
            GripperCommand,
            'gripper_command'
        )

    def map_action_to_ros2(self, action_dict: dict) -> str:
        """
        Map a cognitive planning action to ROS 2 action call

        Args:
            action_dict: Dictionary with action type and parameters from LLM

        Returns:
            String representation of ROS 2 action call (for logging/verification)
        """
        action_type = action_dict.get('action', 'unknown')
        parameters = action_dict.get('parameters', {})

        if action_type == 'navigate_to':
            return self._handle_navigate_to(parameters)
        elif action_type == 'move_arm':
            return self._handle_move_arm(parameters)
        elif action_type == 'grasp_object':
            return self._handle_grasp_object(parameters)
        elif action_type == 'detect_object':
            return self._handle_detect_object(parameters)
        elif action_type == 'move_base':
            return self._handle_move_base(parameters)
        elif action_type == 'rotate_base':
            return self._handle_rotate_base(parameters)
        else:
            self.node.get_logger().warn(f"Unknown action type: {action_type}")
            return f"Unknown action: {action_type}"

    def _handle_navigate_to(self, params: dict) -> str:
        """Handle navigation to a specific location"""
        location = params.get('location', 'unknown')

        # In a real implementation, you would create and send a NavigateToPose goal
        goal = NavigateToPose.Goal()

        if isinstance(location, list) and len(location) == 3:
            # Location is [x, y, theta]
            goal.pose.pose.position.x = float(location[0])
            goal.pose.pose.position.y = float(location[1])
            # Convert theta to quaternion
            import math
            theta = float(location[2])
            goal.pose.pose.orientation.z = math.sin(theta / 2.0)
            goal.pose.pose.orientation.w = math.cos(theta / 2.0)
        elif isinstance(location, str):
            # Location is a named location - would need to look up coordinates
            # This is a simplified example
            pass

        # In real implementation, you would send the goal:
        # self.nav_client.send_goal_async(goal)

        return f"Navigate to {location}"

    def _handle_move_arm(self, params: dict) -> str:
        """Handle arm movement"""
        # In a real implementation, you would create and send a MoveGroup goal
        target_pose = params.get('pose')
        joint_positions = params.get('joint_positions')

        # This would involve MoveIt! planning and execution
        return f"Move arm to pose: {target_pose or joint_positions}"

    def _handle_grasp_object(self, params: dict) -> str:
        """Handle object grasping"""
        target = params.get('target', 'unknown object')

        # This would involve perception, approach, and grasp actions
        return f"Grasp {target}"

    def _handle_detect_object(self, params: dict) -> str:
        """Handle object detection"""
        target = params.get('target', 'any object')

        # This would involve camera activation and object detection
        return f"Detect {target}"

    def _handle_move_base(self, params: dict) -> str:
        """Handle base movement"""
        direction = params.get('direction', 'forward')
        distance = params.get('distance', 1.0)

        return f"Move base {direction} by {distance} meters"

    def _handle_rotate_base(self, params: dict) -> str:
        """Handle base rotation"""
        angle = params.get('angle', 90)  # degrees

        return f"Rotate base by {angle} degrees"

    async def execute_action_sequence(self, action_sequence: list) -> dict:
        """
        Execute a sequence of actions generated by cognitive planning

        Args:
            action_sequence: List of action dictionaries from LLM

        Returns:
            Dictionary with execution results
        """
        results = {
            'executed_actions': [],
            'failed_actions': [],
            'overall_success': True,
            'execution_log': []
        }

        for i, action_dict in enumerate(action_sequence):
            try:
                # Log the action being executed
                action_str = self.map_action_to_ros2(action_dict)
                results['execution_log'].append(f"Step {i+1}: {action_str}")

                # In a real implementation, you would execute the action:
                # result = await self._execute_single_action(action_dict)

                # For this example, we'll just log that the action would be executed
                results['executed_actions'].append({
                    'step': i+1,
                    'action': action_dict,
                    'status': 'simulated'
                })

            except Exception as e:
                results['overall_success'] = False
                results['failed_actions'].append({
                    'step': i+1,
                    'action': action_dict,
                    'error': str(e)
                })
                results['execution_log'].append(f"Step {i+1} FAILED: {str(e)}")

                # Optionally, implement error recovery based on the action sequence
                break  # For this example, we stop on first error

        return results

# Example usage in a ROS 2 node:
"""
class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')
        self.action_mapper = ROS2ActionMapper(self)

    async def process_command(self, command: str):
        # Plan the command
        plan_result = plan_with_context(command)

        if plan_result.get('actions'):
            # Execute the plan
            execution_result = await self.action_mapper.execute_action_sequence(
                plan_result['actions']
            )
            return execution_result
        else:
            return {"error": "No valid action sequence generated"}
"""
```

## Validation and Safety Checks

### Action Sequence Validation

Before executing any action sequence, validate it for safety and feasibility:

```python
def validate_action_sequence(action_sequence: list, robot_capabilities: list) -> dict:
    """
    Validate an action sequence for safety and feasibility

    Args:
        action_sequence: List of action dictionaries from LLM
        robot_capabilities: List of capabilities robot actually has

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'safety_issues': [],
        'feasibility_issues': []
    }

    if not action_sequence:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Empty action sequence")
        return validation_result

    # Check if all actions are supported by the robot
    for i, action in enumerate(action_sequence):
        action_type = action.get('action')

        if not action_type:
            validation_result['errors'].append(f"Action {i+1}: Missing action type")
            continue

        if action_type not in robot_capabilities:
            validation_result['errors'].append(
                f"Action {i+1}: '{action_type}' not supported by robot"
            )
            validation_result['is_valid'] = False

    # Check for safety issues
    for i, action in enumerate(action_sequence):
        action_type = action.get('action')

        # Check for potentially dangerous actions
        if action_type == 'navigate_to':
            params = action.get('parameters', {})
            location = params.get('location')

            # In a real system, you'd check if location is safe
            if location == 'dangerous_area':
                validation_result['safety_issues'].append(
                    f"Action {i+1}: Navigation to dangerous area requested"
                )
                validation_result['is_valid'] = False

        elif action_type == 'grasp_object':
            params = action.get('parameters', {})
            target = params.get('target')

            # Check if target is a fragile object that requires special handling
            fragile_objects = ['glass', 'ceramic', 'fragile']
            if any(fragile in (target or '').lower() for fragile in fragile_objects):
                validation_result['warnings'].append(
                    f"Action {i+1}: Grasping fragile object '{target}', verify gripper settings"
                )

    # Check for temporal feasibility
    total_time = sum(
        action.get('timeout', 10.0) for action in action_sequence
    )

    # Assume max mission time is 300 seconds (5 minutes)
    if total_time > 300:
        validation_result['warnings'].append(
            f"Total estimated time ({total_time}s) exceeds recommended limit"
        )

    # Check for logical consistency
    action_types = [action.get('action') for action in action_sequence]

    # Example: Check for impossible sequences
    if 'grasp_object' in action_types and 'navigate_to' not in action_types:
        validation_result['warnings'].append(
            "Action sequence may be missing navigation to object location"
        )

    return validation_result

def safe_execute_command(command: str, robot_capabilities: list) -> dict:
    """
    Safely execute a command with full validation
    """
    # Step 1: Generate plan
    plan_result = plan_with_context(command)

    if not plan_result.get('actions'):
        return {
            'success': False,
            'error': 'No valid action sequence generated',
            'plan_result': plan_result
        }

    # Step 2: Validate the plan
    validation = validate_action_sequence(plan_result['actions'], robot_capabilities)

    if not validation['is_valid']:
        return {
            'success': False,
            'error': 'Action sequence failed validation',
            'validation_result': validation,
            'plan_result': plan_result
        }

    # Step 3: Execute (in simulation or real robot)
    # This would connect to the ROS2ActionMapper in a real implementation
    return {
        'success': True,
        'plan_result': plan_result,
        'validation_result': validation,
        'message': f"Plan validated successfully with {len(plan_result['actions'])} actions"
    }

# Example usage:
# robot_caps = ["navigate_to", "grasp_object", "move_arm", "detect_object"]
# result = safe_execute_command("Go to the kitchen and bring me the red cup", robot_caps)
# print(json.dumps(result, indent=2))
```

## Confidence Scoring and Ambiguity Handling

### Confidence Assessment

Implement confidence scoring for generated action sequences:

```python
def assess_plan_confidence(plan_result: dict, command: str) -> float:
    """
    Assess confidence in a generated plan

    Args:
        plan_result: Result from LLM planning
        command: Original command that was planned

    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 1.0  # Start with high confidence

    # Check if the plan has actions
    actions = plan_result.get('actions', [])
    if not actions:
        return 0.1  # Very low confidence if no actions

    # Length-based confidence (shorter plans might be more reliable)
    if len(actions) > 10:
        confidence *= 0.8  # Reduce confidence for very long plans

    # Check for generic or vague actions
    generic_actions = ['move', 'go', 'do']
    generic_count = sum(1 for action in actions
                       if action.get('action', '').lower() in generic_actions)
    if generic_count > len(actions) * 0.5:  # More than 50% generic
        confidence *= 0.7

    # Check for specific parameters
    specific_params_needed = ['location', 'target', 'object']
    well_specified_actions = 0

    for action in actions:
        params = action.get('parameters', {})
        has_specific_param = any(param in params for param in specific_params_needed)
        if has_specific_param:
            well_specified_actions += 1

    if well_specified_actions < len(actions) * 0.7:  # Less than 70% well-specified
        confidence *= 0.85

    # Check for safety-related parameters
    safety_params = ['speed', 'force', 'timeout']
    safety_aware_actions = sum(1 for action in actions
                              if any(param in action.get('parameters', {})
                                    for param in safety_params))

    if safety_aware_actions < len(actions) * 0.5:  # Less than 50% safety-aware
        confidence *= 0.9

    # Command complexity factor
    command_complexity = len(command.split()) / 10.0  # Normalize by 10 words
    if command_complexity > 2.0:  # Very complex command
        confidence *= 0.85

    # Apply minimum confidence threshold
    return max(0.1, min(1.0, confidence))

def handle_ambiguous_command(command: str, min_confidence: float = 0.7) -> dict:
    """
    Handle potentially ambiguous commands with confidence checking

    Args:
        command: Natural language command to process
        min_confidence: Minimum confidence required for execution

    Returns:
        Dictionary with either plan or clarification request
    """
    # Generate the plan
    plan_result = plan_with_context(command)

    # Assess confidence
    confidence = assess_plan_confidence(plan_result, command)

    if confidence < min_confidence:
        # The plan is uncertain, request clarification
        clarification_prompt = f"""
        The command "{command}" is ambiguous. Please provide clarification:

        1. What specific object should be manipulated?
        2. What is the destination/target?
        3. Are there any constraints or preferences?

        Suggest 2-3 possible interpretations of the command.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": clarification_prompt}],
                temperature=0.1,
                max_tokens=300
            )

            return {
                'status': 'clarification_needed',
                'original_command': command,
                'confidence': confidence,
                'suggestions': response.choices[0].message.content,
                'min_confidence': min_confidence
            }
        except Exception:
            # Fallback if clarification fails
            return {
                'status': 'uncertain',
                'original_command': command,
                'confidence': confidence,
                'error': 'Could not generate clarifications'
            }

    # Plan is confident enough, return it
    plan_result['confidence'] = confidence
    plan_result['status'] = 'confident'

    return plan_result

# Example usage:
# result = handle_ambiguous_command("Clean the room")
# if result['status'] == 'clarification_needed':
#     print("Need clarification:", result['suggestions'])
# else:
#     print("Plan generated with confidence:", result['confidence'])
```

## Complete Cognitive Planning System

Here's a complete cognitive planning system that ties everything together:

```python
class CognitivePlanningSystem:
    def __init__(self, robot_capabilities: list = None):
        if robot_capabilities is None:
            self.robot_capabilities = [
                "navigate_to", "detect_object", "grasp_object", "move_arm",
                "rotate_base", "open_gripper", "close_gripper", "move_base"
            ]
        else:
            self.robot_capabilities = robot_capabilities

        # Initialize with default context
        self.default_context = {
            "robot_state": {
                "current_location": "starting_position",
                "battery_level": 0.8,
                "gripper_status": "open",
                "arm_position": "home"
            },
            "environment": {
                "layout": "standard room",
                "obstacles": [],
                "objects": [],
                "navigation_goals": {}
            },
            "constraints": {
                "time_limit": 120,
                "safety_requirements": ["avoid_people", "stay_in_bounds"]
            }
        }

    def plan_command(self, command: str, context: dict = None) -> dict:
        """
        Complete command planning with validation and safety checks
        """
        if context is None:
            context = self.default_context.copy()

        # Step 1: Generate plan with context
        plan_result = plan_with_context(command, context)

        # Step 2: Validate the plan
        validation_result = validate_action_sequence(
            plan_result.get('actions', []),
            self.robot_capabilities
        )

        # Step 3: Assess confidence
        confidence = assess_plan_confidence(plan_result, command)

        # Step 4: Check if plan needs clarification
        if confidence < 0.7 and validation_result['is_valid']:
            # Even with valid actions, low confidence might indicate ambiguity
            clarification = handle_ambiguous_command(command, min_confidence=0.7)
            if clarification['status'] == 'clarification_needed':
                return {
                    'status': 'clarification_needed',
                    'original_command': command,
                    'confidence': confidence,
                    'validation': validation_result,
                    'clarification_request': clarification['suggestions']
                }

        # Step 5: Return complete result
        return {
            'status': 'planned' if validation_result['is_valid'] else 'invalid',
            'original_command': command,
            'plan': plan_result,
            'validation': validation_result,
            'confidence': confidence,
            'robot_capabilities': self.robot_capabilities
        }

    def execute_command_safely(self, command: str, context: dict = None) -> dict:
        """
        Execute a command with full safety and validation
        """
        # Plan the command
        planning_result = self.plan_command(command, context)

        if planning_result['status'] == 'clarification_needed':
            return planning_result

        if not planning_result['validation']['is_valid']:
            return {
                'status': 'failed_validation',
                'error': 'Plan failed safety validation',
                'planning_result': planning_result
            }

        # In a real implementation, you would execute the plan using ROS2ActionMapper
        # For this example, we'll just return the planned actions
        return {
            'status': 'execution_planned',
            'original_command': command,
            'actions_to_execute': planning_result['plan']['actions'],
            'estimated_duration': planning_result['plan'].get('estimated_duration', 'unknown'),
            'confidence': planning_result['confidence'],
            'planning_result': planning_result
        }

# Example usage:
# Initialize the cognitive planning system
# planning_system = CognitivePlanningSystem()

# Example commands
# commands = [
#     "Go to the kitchen and bring me the red cup",
#     "Clean the room",  # This might need clarification
#     "Move the book from the table to the shelf"
# ]

# for cmd in commands:
#     print(f"\nProcessing: {cmd}")
#     result = planning_system.execute_command_safely(cmd)
#     print(f"Status: {result['status']}")
#     if result['status'] == 'clarification_needed':
#         print(f"Clarification needed: {result['clarification_request'][:100]}...")
#     elif result['status'] == 'execution_planned':
#         print(f"Actions planned: {len(result['actions_to_execute'])}")
#         print(f"Confidence: {result['confidence']:.2f}")
```

## Troubleshooting Common LLM Planning Issues

### Handling API Rate Limits

```python
import time
import asyncio
from typing import Callable, Any

class RateLimitHandler:
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0

    async def call_with_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function with rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            await asyncio.sleep(sleep_time)

        result = func(*args, **kwargs)
        self.last_call_time = time.time()

        return result

# Usage example:
# rate_limiter = RateLimitHandler(calls_per_minute=20)
# result = await rate_limiter.call_with_rate_limit(plan_from_natural_language, "move forward")
```

### Error Recovery Strategies

```python
def fallback_planning(command: str) -> dict:
    """
    Fallback planning when primary method fails
    """
    # Simple keyword-based fallback
    command_lower = command.lower()

    if any(word in command_lower for word in ["go to", "navigate", "move to"]):
        # Extract potential location
        import re
        location_match = re.search(r"(?:to|at|toward)\s+([^.!?]+)", command_lower)
        location = location_match.group(1).strip() if location_match else "unknown"

        return {
            "actions": [
                {
                    "action": "navigate_to",
                    "parameters": {"location": location},
                    "priority": 5,
                    "timeout": 30.0,
                    "success_criteria": f"reach {location}"
                }
            ],
            "original_command": command,
            "confidence": 0.6,
            "source": "keyword_fallback"
        }

    elif any(word in command_lower for word in ["pick up", "grasp", "take"]):
        # Extract potential object
        import re
        object_match = re.search(r"(?:pick up|grasp|take)\s+([^.!?]+)", command_lower)
        obj = object_match.group(1).strip() if object_match else "unknown"

        return {
            "actions": [
                {
                    "action": "detect_object",
                    "parameters": {"target": obj},
                    "priority": 5,
                    "timeout": 10.0,
                    "success_criteria": f"detect {obj}"
                },
                {
                    "action": "grasp_object",
                    "parameters": {"target": obj},
                    "priority": 4,
                    "timeout": 15.0,
                    "success_criteria": f"grasp {obj}"
                }
            ],
            "original_command": command,
            "confidence": 0.5,
            "source": "keyword_fallback"
        }

    else:
        # Generic response for unknown commands
        return {
            "actions": [],
            "original_command": command,
            "confidence": 0.1,
            "error": "Unable to understand command",
            "source": "keyword_fallback"
        }
```

## Performance Optimization

### Caching for Repeated Commands

```python
from functools import lru_cache
import hashlib

class OptimizedCognitivePlanner:
    def __init__(self, max_cache_size: int = 128):
        self.max_cache_size = max_cache_size
        self.planner = CognitivePlanningSystem()

    @lru_cache(maxsize=128)
    def _cached_plan(self, command_hash: str, command: str) -> dict:
        """Internal method for cached planning"""
        return self.planner.plan_command(command)

    def plan_command_with_cache(self, command: str, context: dict = None) -> dict:
        """Plan command with automatic caching"""
        # Create hash of command and context for caching
        cache_key = hashlib.md5(f"{command}_{str(context)}".encode()).hexdigest()

        return self._cached_plan(cache_key, command)
```

## Summary

In this chapter, you learned how to implement cognitive planning systems that translate natural language commands into executable ROS 2 action sequences. You've covered:

1. **LLM Integration**: How to use OpenAI GPT and alternative models for cognitive planning
2. **Prompt Engineering**: Techniques for reliable command translation with context awareness
3. **ROS 2 Mapping**: How to map natural language constructs to ROS 2 primitives
4. **Validation & Safety**: Implementing checks for safe and feasible action sequences
5. **Confidence Scoring**: Assessing plan reliability and handling ambiguous commands
6. **Error Handling**: Strategies for API limits and fallback planning

The cognitive planning capability bridges human intention with robotic execution, enabling natural language commands to be translated into step-by-step robotic behaviors with at least 90% semantic accuracy.

## Next Steps

- Proceed to Chapter 3: Capstone Project: The Autonomous Humanoid to integrate voice, planning, and execution capabilities
- Practice with different natural language commands to improve your understanding
- Experiment with different LLM models and prompt engineering techniques
- Implement the complete VLA system by combining all three chapters