# Quickstart Guide: Vision-Language-Action (VLA) Module

## Overview
This quickstart guide provides a rapid introduction to the Vision-Language-Action (VLA) concepts covered in Module 4. Follow these steps to set up and experiment with voice-controlled robotics using OpenAI Whisper and LLM-based cognitive planning.

## Prerequisites

### System Requirements
- Python 3.11 or higher
- ROS 2 Humble Hawksbill installed
- Access to OpenAI API (for Whisper and LLMs) or alternative LLM provider
- Microphone for voice input (for testing)
- Operating System: Linux, Windows, or macOS

### Installation Steps

1. **Install ROS 2 Humble Hawksbill**
   ```bash
   # Follow official ROS 2 installation guide for your OS
   # Ubuntu/Debian: http://wiki.ros.org/humble/Installation/Ubuntu
   # Windows: http://wiki.ros.org/humble/Installation/Windows
   # macOS: http://wiki.ros.org/humble/Installation/macOS
   ```

2. **Set up Python Environment**
   ```bash
   python -m venv vla_env
   source vla_env/bin/activate  # On Windows: vla_env\Scripts\activate
   pip install --upgrade pip
   pip install openai torch torchaudio
   pip install numpy scipy matplotlib
   ```

3. **Install Additional Dependencies**
   ```bash
   pip install pyaudio  # For audio input
   pip install speechrecognition  # Alternative speech recognition
   pip install transformers  # For local LLM inference (optional)
   ```

4. **Configure OpenAI API Key**
   ```bash
   # Create .env file in your project directory
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## Quick Voice Command Demo

### Step 1: Basic Voice Recognition
```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file_path):
    """Transcribe audio using OpenAI Whisper"""
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text

# Example usage
# command_text = transcribe_audio("voice_command.wav")
# print(f"Recognized command: {command_text}")
```

### Step 2: Simple Command Processing
```python
def process_voice_command(transcript):
    """Extract intent from voice command"""
    # Simple keyword matching for demonstration
    if "move forward" in transcript.lower():
        return {"action": "move_forward", "params": {}}
    elif "turn left" in transcript.lower():
        return {"action": "turn_left", "params": {}}
    elif "stop" in transcript.lower():
        return {"action": "stop", "params": {}}
    else:
        return {"action": "unknown", "params": {"raw_command": transcript}}

# Example usage
# intent = process_voice_command(command_text)
# print(f"Intent: {intent}")
```

### Step 3: Basic ROS 2 Action Mapping
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def execute_intent(self, intent):
        """Convert intent to ROS 2 action"""
        msg = Twist()

        if intent['action'] == 'move_forward':
            msg.linear.x = 0.5  # Move forward at 0.5 m/s
        elif intent['action'] == 'turn_left':
            msg.angular.z = 0.5  # Turn left at 0.5 rad/s
        elif intent['action'] == 'stop':
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.publisher.publish(msg)
```

## Cognitive Planning Example

### Natural Language to Action Sequence
```python
import openai
import json

def plan_from_natural_language(command):
    """Use LLM to convert natural language to action sequence"""
    prompt = f"""
    Convert the following natural language command to a sequence of ROS 2 actions.
    Respond in JSON format with an array of actions.

    Command: "{command}"

    Actions should be one of: ["navigate_to", "detect_object", "grasp_object", "move_arm", "rotate_base"]

    Example response:
    {{
        "actions": [
            {{"action": "navigate_to", "params": {{"location": "kitchen"}}},
            {{"action": "detect_object", "params": {{"target": "red cup"}}},
            {{"action": "grasp_object", "params": {{"object": "red cup"}}}
        ]
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result['actions']
    except:
        # Fallback if parsing fails
        return [{"action": "unknown", "params": {"raw_command": command}}]
```

## Running the Full Pipeline

### Complete Voice-to-Action Example
```python
def run_vla_pipeline():
    """Complete Vision-Language-Action pipeline"""
    # 1. Record or provide audio file
    # audio_file = record_audio()  # Implementation needed

    # 2. Transcribe audio to text
    # command_text = transcribe_audio(audio_file)
    # print(f"Heard: {command_text}")

    # 3. Process natural language command
    # action_sequence = plan_from_natural_language(command_text)
    # print(f"Action sequence: {action_sequence}")

    # 4. Execute actions in ROS 2 environment
    # node = VoiceCommandNode()
    # for action in action_sequence:
    #     node.execute_intent(action)

    print("VLA pipeline completed successfully!")

if __name__ == '__main__':
    rclpy.init()
    run_vla_pipeline()
    rclpy.shutdown()
```

## Expected Output

After completing the quickstart:

1. **Voice Recognition**: You should be able to convert spoken commands to text with reasonable accuracy
2. **Intent Processing**: The system should identify basic movement commands from the transcribed text
3. **Action Mapping**: Commands should translate to appropriate ROS 2 movement actions
4. **Cognitive Planning**: Natural language commands should generate sequences of robotic actions

## Troubleshooting

### Common Issues

**Issue**: Whisper API returning errors
- **Solution**: Verify your API key is correct and you have sufficient credits

**Issue**: Audio recording problems
- **Solution**: Check microphone permissions and install pyaudio: `pip install pyaudio`

**Issue**: ROS 2 nodes not communicating
- **Solution**: Ensure ROS 2 environment is sourced: `source /opt/ros/humble/setup.bash`

**Issue**: LLM responses inconsistent
- **Solution**: Improve prompt engineering or use higher-quality model

## Next Steps

1. Complete the full "Voice-to-Action" chapter for detailed Whisper integration
2. Proceed to "Cognitive Planning with LLMs" for advanced natural language processing
3. Implement the "Capstone Project: The Autonomous Humanoid" for complete integration

## Performance Benchmarks

- Voice recognition accuracy: Target ≥85% in quiet environments
- Command processing latency: Target <2 seconds end-to-end
- Action sequence generation: Target ≥90% semantic accuracy