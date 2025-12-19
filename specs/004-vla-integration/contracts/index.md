# Contracts: Vision-Language-Action (VLA) Module

## Overview
This document defines the API contracts, interfaces, and integration points for the Vision-Language-Action (VLA) module. These contracts ensure consistent interactions between different components and external systems.

## Voice Processing Contracts

### Whisper Integration API
```yaml
openapi: 3.0.0
info:
  title: VLA Voice Processing API
  version: 1.0.0
  description: API for voice command processing using OpenAI Whisper
paths:
  /voice/transcribe:
    post:
      summary: Transcribe audio to text using Whisper
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                  description: Audio file to transcribe
                language:
                  type: string
                  description: Language of the audio (optional)
                response_format:
                  type: string
                  enum: [json, text, srt, verbose_json, vtt]
                  default: json
      responses:
        '200':
          description: Successful transcription
          content:
            application/json:
              schema:
                type: object
                properties:
                  text:
                    type: string
                    description: Transcribed text
                  confidence:
                    type: number
                    description: Confidence score (0.0-1.0)
        '400':
          description: Invalid audio file or parameters
        '500':
          description: Transcription service error
```

### Voice Command Processor Interface
```python
class VoiceCommandProcessorInterface:
    def transcribe_audio(self, audio_data: bytes) -> dict:
        """
        Transcribe audio data to text

        Args:
            audio_data: Raw audio data to transcribe

        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        pass

    def extract_intent(self, text: str) -> dict:
        """
        Extract intent from transcribed text

        Args:
            text: Transcribed text from audio

        Returns:
            Dictionary with intent and parameters
        """
        pass

    def process_voice_command(self, audio_file_path: str) -> dict:
        """
        Complete voice command processing pipeline

        Args:
            audio_file_path: Path to audio file to process

        Returns:
            Dictionary with complete command processing results
        """
        pass
```

## Cognitive Planning Contracts

### LLM Planning API
```yaml
openapi: 3.0.0
info:
  title: VLA Cognitive Planning API
  version: 1.0.0
  description: API for generating action sequences from natural language using LLMs
paths:
  /plan/generate:
    post:
      summary: Generate action sequence from natural language command
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - command
              properties:
                command:
                  type: string
                  description: Natural language command to process
                context:
                  type: object
                  description: Context for planning (optional)
                  properties:
                    robot_capabilities:
                      type: array
                      items:
                        type: string
                      description: List of robot capabilities
                    environment:
                      type: object
                      description: Environmental constraints
      responses:
        '200':
          description: Generated action sequence
          content:
            application/json:
              schema:
                type: object
                properties:
                  actions:
                    type: array
                    items:
                      $ref: '#/components/schemas/Action'
                  confidence:
                    type: number
                    description: Confidence in plan (0.0-1.0)
        '400':
          description: Invalid command or parameters
        '500':
          description: Planning service error
components:
  schemas:
    Action:
      type: object
      properties:
        type:
          type: string
          description: Type of action to perform
        parameters:
          type: object
          description: Parameters for the action
        priority:
          type: integer
          description: Priority of the action (1-10)
```

### Cognitive Planner Interface
```python
class CognitivePlannerInterface:
    def generate_plan(self, command: str, context: dict = None) -> dict:
        """
        Generate action sequence from natural language command

        Args:
            command: Natural language command to process
            context: Additional context for planning

        Returns:
            Dictionary with action sequence and metadata
        """
        pass

    def validate_plan(self, plan: dict) -> dict:
        """
        Validate a cognitive plan for correctness and safety

        Args:
            plan: Plan to validate

        Returns:
            Dictionary with validation results
        """
        pass

    def execute_plan(self, plan: dict) -> dict:
        """
        Execute a cognitive plan

        Args:
            plan: Plan to execute

        Returns:
            Dictionary with execution results
        """
        pass
```

## ROS 2 Integration Contracts

### Action Mapping Interface
```python
class ROSActionMapperInterface:
    def map_action_to_ros(self, action: dict) -> str:
        """
        Map VLA action to ROS 2 action/service call

        Args:
            action: VLA action dictionary

        Returns:
            ROS 2 action/service call string
        """
        pass

    def execute_ros_action(self, ros_action: str) -> dict:
        """
        Execute ROS 2 action and return results

        Args:
            ros_action: ROS 2 action to execute

        Returns:
            Dictionary with execution results
        """
        pass

    def validate_ros_action(self, action: dict) -> bool:
        """
        Validate that the action can be executed in ROS 2 environment

        Args:
            action: Action to validate

        Returns:
            True if action is valid, False otherwise
        """
        pass
```

### Navigation Service Contract
```python
class NavigationServiceInterface:
    def navigate_to(self, goal: dict) -> dict:
        """
        Navigate to specified goal position

        Args:
            goal: Dictionary with position and orientation data

        Returns:
            Dictionary with navigation results
        """
        pass

    def get_robot_pose(self) -> dict:
        """
        Get current robot pose

        Returns:
            Dictionary with position and orientation
        """
        pass

    def cancel_navigation(self) -> bool:
        """
        Cancel current navigation goal

        Returns:
            True if cancellation was successful
        """
        pass
```

## Computer Vision Contracts

### Object Detection API
```yaml
openapi: 3.0.0
info:
  title: VLA Computer Vision API
  version: 1.0.0
  description: API for object detection and identification
paths:
  /vision/detect:
    post:
      summary: Detect objects in image
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                image:
                  type: string
                  format: binary
                  description: Image file to analyze
                target_objects:
                  type: array
                  items:
                    type: string
                  description: Specific objects to detect (optional)
      responses:
        '200':
          description: Detected objects
          content:
            application/json:
              schema:
                type: object
                properties:
                  objects:
                    type: array
                    items:
                      $ref: '#/components/schemas/DetectedObject'
        '400':
          description: Invalid image file
        '500':
          description: Vision service error
components:
  schemas:
    DetectedObject:
      type: object
      properties:
        class_name:
          type: string
          description: Object class name
        confidence:
          type: number
          description: Detection confidence (0.0-1.0)
        bounding_box:
          type: object
          properties:
            x: number
            y: number
            width: number
            height: number
```

### Vision Processing Interface
```python
class VisionProcessorInterface:
    def detect_objects(self, image_data: bytes, target_objects: list = None) -> list:
        """
        Detect objects in image data

        Args:
            image_data: Image data to analyze
            target_objects: Specific objects to look for (optional)

        Returns:
            List of detected objects with confidence scores
        """
        pass

    def identify_object_location(self, object_name: str) -> dict:
        """
        Identify location of a specific object in the environment

        Args:
            object_name: Name of object to locate

        Returns:
            Dictionary with object location data
        """
        pass

    def get_grasp_points(self, object_data: dict) -> list:
        """
        Get suggested grasp points for an object

        Args:
            object_data: Data about the object to grasp

        Returns:
            List of suggested grasp points
        """
        pass
```

## Autonomous Humanoid Contracts

### Main Control Interface
```python
class AutonomousHumanoidInterface:
    def process_voice_command(self, audio_file_path: str) -> dict:
        """
        Process a complete voice command from audio to action execution

        Args:
            audio_file_path: Path to audio file with command

        Returns:
            Dictionary with processing results
        """
        pass

    def execute_command_sequence(self, commands: list) -> dict:
        """
        Execute a sequence of commands

        Args:
            commands: List of commands to execute

        Returns:
            Dictionary with execution results
        """
        pass

    def get_status(self) -> dict:
        """
        Get current status of the humanoid

        Returns:
            Dictionary with current status information
        """
        pass

    def cancel_current_task(self) -> bool:
        """
        Cancel the currently executing task

        Returns:
            True if cancellation was successful
        """
        pass
```

## Error Handling Contracts

### Standard Error Response Format
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object",
    "timestamp": "string",
    "request_id": "string"
  }
}
```

### Error Codes
- `VOICE_RECOGNITION_FAILED`: Whisper transcription failed
- `INTENT_EXTRACTION_FAILED`: Could not extract intent from text
- `PLANNING_FAILED`: LLM could not generate valid action sequence
- `ACTION_EXECUTION_FAILED`: ROS action failed to execute
- `NAVIGATION_FAILED`: Navigation goal could not be reached
- `OBJECT_DETECTION_FAILED`: Vision system could not detect objects
- `INVALID_COMMAND`: Command could not be understood or processed

## Performance Contracts

### Response Time Requirements
- Voice transcription: <1.5 seconds for 5-second audio clip
- Intent extraction: <0.5 seconds
- Plan generation: <2 seconds for simple commands
- Action execution: <0.1 seconds for command dispatch
- Object detection: <1 second for single image

### Accuracy Requirements
- Voice recognition: ≥85% accuracy in quiet environments
- Intent extraction: ≥90% accuracy for simple commands
- Plan generation: ≥85% semantic accuracy
- Object detection: ≥80% accuracy for known objects

## Security Contracts

### API Authentication
All API endpoints require authentication via API key in the header:
```
Authorization: Bearer {API_KEY}
```

### Data Privacy
- Audio data is processed but not stored by default
- Transcription results may be logged for debugging
- Personal information in voice commands should be handled according to privacy policies