# Data Model: Vision-Language-Action (VLA) Module

## Overview
This document defines the key data structures and entities used in the Vision-Language-Action (VLA) module. These models represent the core concepts of voice processing, cognitive planning, and autonomous humanoid behavior.

## Core Entities

### Voice Command
Represents a natural language input processed by Whisper for actionable intent extraction.

```python
class VoiceCommand:
    id: str                    # Unique identifier for the command
    audio_data: bytes          # Raw audio data or path to audio file
    transcription: str         # Text transcription from Whisper
    confidence: float          # Confidence score from speech recognition (0.0-1.0)
    timestamp: datetime        # When the command was received
    language: str              # Detected language of the command
    intent: str                # Extracted intent from the command
    parameters: dict           # Additional parameters extracted from the command
    processed: bool            # Whether the command has been processed
    error: Optional[str]       # Error message if processing failed
```

### Cognitive Plan
Represents a structured sequence of ROS 2 actions generated from natural language commands using LLMs.

```python
class CognitivePlan:
    id: str                    # Unique identifier for the plan
    original_command: str      # The original natural language command
    action_sequence: List[dict] # Sequence of actions to execute
    status: str               # Current status (pending, executing, completed, failed)
    created_at: datetime      # When the plan was created
    estimated_duration: float  # Estimated time to complete the plan (seconds)
    confidence: float         # Confidence in the plan's correctness (0.0-1.0)
    error_recovery_plan: Optional[List[dict]] # Actions to take if errors occur
```

### Action
Represents a single ROS 2 action that can be executed by the robotic system.

```python
class Action:
    type: str                 # Type of action (e.g., "navigate_to", "detect_object", "grasp_object")
    parameters: dict          # Parameters for the action
    priority: int             # Execution priority (1-10, 10 being highest)
    timeout: float            # Maximum time to wait for action completion (seconds)
    dependencies: List[str]   # IDs of actions that must complete before this one
    success_criteria: dict    # Conditions that define success for this action
    error_handling: dict      # How to handle errors during execution
```

### Autonomous Humanoid
Represents the integrated system combining voice processing, cognitive planning, navigation, computer vision, and manipulation capabilities.

```python
class AutonomousHumanoid:
    id: str                   # Unique identifier for the humanoid instance
    status: str               # Current status (idle, processing_command, executing_plan, error)
    current_plan: Optional[CognitivePlan]  # The plan currently being executed
    voice_processor: VoiceCommandProcessor  # Voice processing component
    cognitive_planner: CognitivePlanner    # Planning component
    navigation_system: NavigationSystem    # Navigation component
    computer_vision: ComputerVisionSystem  # Vision component
    manipulation_system: ManipulationSystem # Manipulation component
    sensors: dict             # Current sensor data
    capabilities: List[str]   # List of capabilities (e.g., "voice_recognition", "navigation", "grasping")
```

### Navigation Goal
Represents a navigation target for the humanoid robot.

```python
class NavigationGoal:
    id: str                   # Unique identifier for the navigation goal
    position: dict            # Position coordinates (x, y, z)
    orientation: dict         # Orientation (quaternion: x, y, z, w)
    frame_id: str             # Coordinate frame (e.g., "map", "odom")
    description: str          # Human-readable description of the goal
    priority: int             # Navigation priority (1-10)
    constraints: dict         # Navigation constraints (e.g., avoid_obstacles, preferred_path)
```

### Object Detection
Represents an object detected by the computer vision system.

```python
class ObjectDetection:
    id: str                   # Unique identifier for the detection
    class_name: str           # Object class (e.g., "cup", "chair", "person")
    confidence: float         # Detection confidence (0.0-1.0)
    bounding_box: dict        # Bounding box coordinates (x, y, width, height)
    position_3d: dict         # 3D position relative to robot (x, y, z)
    color: Optional[str]      # Detected color of the object
    size: Optional[dict]      # Estimated size (width, height, depth)
    grasp_points: Optional[List[dict]]  # Suggested grasp points for manipulation
```

## Data Flow Relationships

### Voice Processing Flow
```
VoiceCommand -(transcription, intent)-> CognitivePlanner -(action_sequence)-> CognitivePlan
```

### Execution Flow
```
CognitivePlan -(action_sequence)-> Action -(execution)-> AutonomousHumanoid
```

### Sensor Integration
```
NavigationGoal -(position, orientation)-> AutonomousHumanoid -(sensors)-> ComputerVisionSystem -(detections)-> ObjectDetection
```

## State Models

### Humanoid State Machine
```python
class HumanoidState:
    current_state: str        # Current state (idle, listening, processing, planning, executing, error)
    previous_state: str       # Previous state for transition tracking
    state_entry_time: datetime # Time when current state was entered
    state_context: dict       # Contextual data for the current state
    available_transitions: List[str]  # Valid state transitions from current state
```

### Plan Execution State
```python
class PlanExecutionState:
    current_action_index: int # Index of currently executing action
    completed_actions: List[int]  # Indices of completed actions
    failed_actions: List[int]     # Indices of failed actions
    execution_log: List[dict]     # Log of all execution events
    remaining_time: float         # Estimated time remaining for plan completion
    success_probability: float    # Current probability of plan success
```

## Serialization Formats

### Action Sequence JSON Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "actions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {"type": "string"},
          "parameters": {"type": "object"},
          "priority": {"type": "integer", "minimum": 1, "maximum": 10},
          "timeout": {"type": "number", "minimum": 0}
        },
        "required": ["type", "parameters"]
      }
    }
  },
  "required": ["actions"]
}
```

### Voice Command Processing Result
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "command": {"type": "string"},
    "intent": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "parameters": {"type": "object"},
    "timestamp": {"type": "string", "format": "date-time"}
  },
  "required": ["command", "intent", "confidence"]
}
```

## Validation Rules

1. **Voice Command Validation**:
   - Confidence must be between 0.0 and 1.0
   - Transcription must not be empty
   - Timestamp must be within reasonable bounds

2. **Cognitive Plan Validation**:
   - Action sequence must not be empty
   - Each action must have a valid type
   - Dependencies must reference valid action IDs

3. **Action Validation**:
   - Timeout must be positive
   - Priority must be between 1 and 10
   - Required parameters must be present for each action type

4. **Object Detection Validation**:
   - Confidence must be between 0.0 and 1.0
   - Bounding box coordinates must be valid
   - 3D position must be in robot coordinate frame