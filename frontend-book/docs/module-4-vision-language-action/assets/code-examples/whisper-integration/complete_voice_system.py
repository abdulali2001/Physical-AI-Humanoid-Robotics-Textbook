"""
Complete Voice Command System Example
This example demonstrates a complete voice-to-action system using OpenAI Whisper
"""

import openai
import os
import time
import threading
import queue
from dotenv import load_dotenv
import json
import requests
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class VoiceCommandSystem:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.is_running = False
        # Note: You'll need to implement or import HybridIntentExtractor and related functions
        # from the documentation examples
        # self.command_processor = HybridIntentExtractor()

    def process_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: Audio → Transcription → Intent → Result
        """
        try:
            # Step 1: Transcribe audio using Whisper API
            with open(audio_file_path, "rb") as audio_file:
                transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
                transcript = transcript_response.text

            # Step 2: Extract intent from transcription
            # Note: You'll need to implement or import the intent extraction functions
            # intent_result = self.command_processor.extract_intent(transcript)

            # For this example, we'll use a simple keyword-based approach
            intent_result = self.simple_intent_extraction(transcript)

            # Step 3: Validate intent
            available_actions = [
                "move_forward", "move_backward", "turn_left", "turn_right", "stop",
                "navigate_to", "detect_object", "grasp_object", "release_object"
            ]
            validated_result = self.validate_intent(intent_result, available_actions)

            return {
                "transcription": transcript,
                "intent_result": validated_result,
                "timestamp": time.time(),
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": time.time(),
                "status": "error"
            }

    def simple_intent_extraction(self, transcript: str) -> Dict[str, Any]:
        """Simple keyword-based intent extraction for demo purposes"""
        transcript_lower = transcript.lower().strip()

        # Define simple command patterns
        if any(word in transcript_lower for word in ["move forward", "go forward", "forward", "go ahead", "ahead"]):
            intent = "move_forward"
        elif any(word in transcript_lower for word in ["turn left", "left", "rotate left"]):
            intent = "turn_left"
        elif any(word in transcript_lower for word in ["turn right", "right", "rotate right"]):
            intent = "turn_right"
        elif any(word in transcript_lower for word in ["stop", "halt", "pause", "cease"]):
            intent = "stop"
        elif any(word in transcript_lower for word in ["go to", "navigate to", "move to", "head to"]):
            intent = "navigate_to"
            # Extract target from command
            for word in ["to", "toward", "at"]:
                if word in transcript_lower:
                    target_start = transcript_lower.find(word) + len(word) + 1
                    target = transcript_lower[target_start:].strip().split()[0] if transcript_lower[target_start:].strip().split() else "unknown"
                    return {
                        "intent": intent,
                        "original_command": transcript,
                        "confidence": 0.8,
                        "parameters": {"target": target}
                    }
        else:
            intent = "unknown"

        return {
            "intent": intent,
            "original_command": transcript,
            "confidence": 0.8 if intent != "unknown" else 0.1,
            "parameters": {}
        }

    def validate_intent(self, intent_result: Dict[str, Any], available_actions: list) -> Dict[str, Any]:
        """Validate that the extracted intent is valid and executable"""
        validated_result = intent_result.copy()

        # Check if intent is in available actions
        intent = intent_result.get("intent", "unknown")
        if intent != "unknown" and intent not in available_actions:
            validated_result["validation_error"] = f"Intent '{intent}' not available"
            validated_result["is_valid"] = False
            return validated_result

        validated_result["is_valid"] = True
        return validated_result

    def start_listening(self):
        """Start the voice command system"""
        self.is_running = True
        print("Voice command system started. Ready to process commands.")

    def stop_listening(self):
        """Stop the voice command system"""
        self.is_running = False
        print("Voice command system stopped.")

    def execute_command(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the recognized command (simulated in this example)
        """
        intent = intent_result.get("intent", "unknown")
        parameters = intent_result.get("parameters", {})

        print(f"Executing command: {intent} with parameters: {parameters}")

        # Simulate command execution
        if intent == "move_forward":
            print("Robot moving forward...")
            time.sleep(0.5)  # Simulate execution time
            return {"status": "completed", "action": "move_forward", "distance": "1 meter"}
        elif intent == "turn_left":
            print("Robot turning left...")
            time.sleep(0.5)
            return {"status": "completed", "action": "turn_left", "angle": "90 degrees"}
        elif intent == "turn_right":
            print("Robot turning right...")
            time.sleep(0.5)
            return {"status": "completed", "action": "turn_right", "angle": "90 degrees"}
        elif intent == "stop":
            print("Robot stopped.")
            return {"status": "completed", "action": "stop"}
        elif intent == "navigate_to":
            target = parameters.get("target", "unknown location")
            print(f"Robot navigating to {target}...")
            time.sleep(1.0)  # Simulate longer navigation
            return {"status": "completed", "action": "navigate_to", "target": target}
        else:
            print(f"Unknown or unsupported command: {intent}")
            return {"status": "failed", "action": intent, "reason": "unsupported"}

# Example usage
if __name__ == "__main__":
    # Initialize the voice command system
    vcs = VoiceCommandSystem()
    vcs.start_listening()

    print("This is a demo system. To use it:")
    print("1. Make sure you have an audio file to process")
    print("2. Update the audio file path in the code")
    print("3. Ensure your OpenAI API key is set in the .env file")
    print("\nFor a real implementation, replace 'sample_command.wav' with an actual audio file path")

    # Example: Process a sample audio file
    # Replace 'sample_command.wav' with an actual audio file path
    # result = vcs.process_audio_file("sample_command.wav")

    # if result["status"] == "success":
    #     print(f"Transcription: {result['transcription']}")
    #     print(f"Intent: {result['intent_result']['intent']}")
    #     print(f"Confidence: {result['intent_result'].get('confidence', 'N/A')}")

    #     # Execute the command if it's valid
    #     if result["intent_result"]["is_valid"]:
    #         execution_result = vcs.execute_command(result["intent_result"])
    #         print(f"Execution result: {execution_result}")
    #     else:
    #         print("Command validation failed:", result["intent_result"].get("validation_error"))
    # else:
    #     print(f"Processing failed: {result['error']}")

    vcs.stop_listening()