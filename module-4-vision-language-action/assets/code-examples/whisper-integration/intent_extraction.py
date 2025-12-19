"""
Intent Extraction Module
Advanced techniques for extracting structured intent from natural language
"""

import openai
import json
import os
import re
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_intent_from_text(transcript: str) -> Dict[str, Any]:
    """Extract intent from transcribed text using keyword matching"""
    transcript_lower = transcript.lower().strip()

    # Define command patterns with confidence scores
    command_patterns = {
        "move_forward": {
            "patterns": ["move forward", "go forward", "forward", "go ahead", "ahead", "move straight"],
            "confidence": 0.9
        },
        "move_backward": {
            "patterns": ["move backward", "go backward", "backward", "back", "reverse"],
            "confidence": 0.9
        },
        "turn_left": {
            "patterns": ["turn left", "left", "rotate left", "pivot left", "go left"],
            "confidence": 0.9
        },
        "turn_right": {
            "patterns": ["turn right", "right", "rotate right", "pivot right", "go right"],
            "confidence": 0.9
        },
        "stop": {
            "patterns": ["stop", "halt", "pause", "cease", "freeze", "stand still"],
            "confidence": 0.95
        },
        "move_up": {
            "patterns": ["move up", "up", "raise", "lift", "go up", "elevate"],
            "confidence": 0.85
        },
        "move_down": {
            "patterns": ["move down", "down", "lower", "go down", "descend"],
            "confidence": 0.85
        },
        "grip": {
            "patterns": ["grip", "grab", "pick up", "take", "hold", "grasp", "catch"],
            "confidence": 0.8
        },
        "release": {
            "patterns": ["release", "let go", "drop", "release grip", "loosen", "open gripper"],
            "confidence": 0.85
        },
        "navigate_to": {
            "patterns": ["go to", "navigate to", "move to", "head to", "go over to", "walk to"],
            "confidence": 0.8
        }
    }

    best_match = None
    best_confidence = 0
    matched_pattern = ""

    for intent, data in command_patterns.items():
        for pattern in data["patterns"]:
            if pattern in transcript_lower:
                # Calculate confidence based on pattern length (longer matches might be more specific)
                pattern_confidence = data["confidence"]
                if len(pattern) > len(matched_pattern):
                    matched_pattern = pattern

                if pattern_confidence > best_confidence:
                    best_confidence = pattern_confidence
                    best_match = intent

    if best_match:
        return {
            "intent": best_match,
            "original_command": transcript,
            "confidence": best_confidence,
            "matched_pattern": matched_pattern
        }

    # If no pattern matches, return unknown
    return {
        "intent": "unknown",
        "original_command": transcript,
        "confidence": 0.1,
        "matched_pattern": None
    }

def extract_intent_with_parameters(transcript: str) -> Dict[str, Any]:
    """Extract intent and parameters using regex patterns"""
    transcript_lower = transcript.lower().strip()

    # Define patterns that also capture parameters
    patterns = [
        # Navigate to location
        {
            "intent": "navigate_to",
            "pattern": r"(?:go to|navigate to|move to|head to|walk to)\s+(.+?)(?:\.|$)",
            "confidence": 0.9
        },
        # Move in direction with distance
        {
            "intent": "move_with_distance",
            "pattern": r"(move|go)\s+(forward|backward|left|right)\s+([0-9.]+)\s*(?:meters?|m|feet|ft)?",
            "confidence": 0.85
        },
        # Grasp specific object
        {
            "intent": "grasp_object",
            "pattern": r"(?:grasp|grab|pick up|take|hold)\s+(.+?)(?:\.|$)",
            "confidence": 0.8
        },
        # Turn to specific angle
        {
            "intent": "turn_to_angle",
            "pattern": r"(?:turn|rotate)\s+(left|right)\s+([0-9.]+)\s*(?:degrees?|deg)?",
            "confidence": 0.8
        }
    ]

    for pattern_config in patterns:
        match = re.search(pattern_config["pattern"], transcript_lower)
        if match:
            groups = match.groups()
            if len(groups) >= 1:
                return {
                    "intent": pattern_config["intent"],
                    "original_command": transcript,
                    "confidence": pattern_config["confidence"],
                    "parameters": {"target": groups[0] if len(groups) > 0 else None,
                                 "value": groups[1] if len(groups) > 1 else None},
                    "matched_pattern": pattern_config["pattern"]
                }

    # Fallback to simple keyword matching
    return extract_intent_from_text(transcript)

def extract_intent_with_llm(transcript: str, available_actions: List[str] = None) -> Dict[str, Any]:
    """
    Use LLM to extract structured intent from natural language

    Args:
        transcript: The transcribed text from voice input
        available_actions: List of available robot actions (optional)
    """
    if available_actions is None:
        available_actions = [
            "move_forward", "move_backward", "turn_left", "turn_right", "stop",
            "navigate_to", "detect_object", "grasp_object", "release_object",
            "move_arm_to", "rotate_base", "move_up", "move_down", "grip", "release"
        ]

    actions_str = ", ".join(available_actions)

    prompt = f"""
    Analyze this voice command and extract the intent and parameters.
    Return the result as valid JSON with the following structure:
    {{
        "intent": "<action_type>",
        "parameters": {{
            "target": "<object or location if applicable>",
            "direction": "<direction if applicable>",
            "distance": "<distance if applicable>",
            "speed": "<speed if applicable>",
            "location": "<specific location if applicable>"
        }},
        "confidence": <float between 0.0 and 1.0>,
        "action_sequence": ["<list of actions to perform>"]
    }}

    Available actions: {actions_str}

    Voice command: "{transcript}"

    Be as specific as possible. If the command is ambiguous, make reasonable assumptions based on context.
    If you cannot determine a clear intent, return "unknown" as the intent.
    Respond with JSON only, no additional text.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()

        # Clean up potential markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text[7:]  # Remove ```json
        if result_text.endswith("```"):
            result_text = result_text[:-3]  # Remove ```

        result = json.loads(result_text)
        result["original_command"] = transcript
        return result

    except json.JSONDecodeError:
        print(f"Failed to parse JSON response: {response.choices[0].message.content}")
        # Fallback to keyword matching if JSON parsing fails
        return {**extract_intent_from_text(transcript), "action_sequence": []}
    except Exception as e:
        print(f"LLM intent extraction failed: {e}")
        # Fallback to keyword matching if LLM call fails
        return {**extract_intent_from_text(transcript), "action_sequence": []}

class HybridIntentExtractor:
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.fallback_extractor = extract_intent_from_text
        self.llm_extractor = extract_intent_with_llm

    def extract_intent(self, transcript: str, use_llm_fallback: bool = True) -> Dict[str, Any]:
        """
        Extract intent using multiple methods with fallbacks
        """
        # First, try simple keyword matching for high-confidence, simple commands
        simple_result = extract_intent_with_parameters(transcript)

        # If confidence is high enough, return the simple result
        if simple_result["confidence"] >= self.confidence_threshold:
            return simple_result

        # If confidence is low, try LLM for more complex understanding
        if use_llm_fallback:
            llm_result = extract_intent_with_llm(transcript)

            # Return LLM result if it has higher confidence
            if llm_result.get("confidence", 0) > simple_result["confidence"]:
                return llm_result

        # Fallback to simple extraction
        return simple_result

def validate_intent(intent_result: Dict[str, Any], available_actions: List[str]) -> Dict[str, Any]:
    """
    Validate that the extracted intent is valid and executable

    Args:
        intent_result: Result from intent extraction
        available_actions: List of actions the robot can perform

    Returns:
        Validated intent result with validation status
    """
    validated_result = intent_result.copy()

    # Check if intent is in available actions
    intent = intent_result.get("intent", "unknown")
    if intent != "unknown" and intent not in available_actions:
        validated_result["validation_error"] = f"Intent '{intent}' not available"
        validated_result["is_valid"] = False
        return validated_result

    # Validate parameters if present
    parameters = intent_result.get("parameters", {})
    if parameters:
        # Add parameter validation logic here as needed
        # For example, check if target object exists, distance is reasonable, etc.
        pass

    validated_result["is_valid"] = True
    return validated_result

# Example usage and testing
if __name__ == "__main__":
    print("Intent Extraction Module")
    print("This module provides functions for:")
    print("1. Simple keyword-based intent extraction")
    print("2. Parameter extraction with regex")
    print("3. Advanced LLM-based intent extraction")
    print("4. Hybrid extraction with fallbacks")
    print("5. Intent validation")
    print("\nExample usage:")

    # Example 1: Simple keyword matching
    transcript1 = "Please move forward slowly"
    result1 = extract_intent_from_text(transcript1)
    print(f"\nTranscript: '{transcript1}'")
    print(f"Intent: {result1['intent']}, Confidence: {result1['confidence']:.2f}")

    # Example 2: Parameter extraction
    transcript2 = "Navigate to the kitchen area"
    result2 = extract_intent_with_parameters(transcript2)
    print(f"\nTranscript: '{transcript2}'")
    print(f"Intent: {result2['intent']}, Parameters: {result2.get('parameters', {})}")

    # Example 3: Hybrid extraction
    extractor = HybridIntentExtractor()
    transcript3 = "Turn the robot to the left by 90 degrees"
    result3 = extractor.extract_intent(transcript3)
    print(f"\nTranscript: '{transcript3}'")
    print(f"Intent: {result3['intent']}, Confidence: {result3['confidence']:.2f}")

    # Example 4: Validation
    available_actions = ["move_forward", "turn_left", "turn_right", "stop", "navigate_to"]
    validated_result = validate_intent(result3, available_actions)
    print(f"\nValidation result: {validated_result['is_valid']}")
    if not validated_result['is_valid']:
        print(f"Error: {validated_result.get('validation_error', 'Unknown error')}")