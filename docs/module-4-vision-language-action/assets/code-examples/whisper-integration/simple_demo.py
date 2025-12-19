"""
Simple Voice Command Demo
A minimal example to demonstrate Whisper integration
"""

import openai
import os
from dotenv import load_dotenv

# Load your API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def simple_voice_command_demo():
    """
    Simple demo that transcribes an audio file and prints the result
    """
    # Replace with your actual audio file path
    audio_file_path = "your_audio_file.wav"

    print("This is a demo function. To use it:")
    print("1. Replace 'your_audio_file.wav' with an actual audio file path")
    print("2. Ensure your OpenAI API key is set in the .env file")
    print("3. Run this script")

    # For demonstration, we'll show what would happen
    print("\nExpected behavior:")
    print("- The system would transcribe the audio file")
    print("- It would detect simple commands like 'move forward', 'turn left', etc.")
    print("- It would print the transcription and detected intent")

    # Uncomment the code below to use with a real audio file
    # try:
    #     # Transcribe the audio file
    #     with open(audio_file_path, "rb") as audio_file:
    #         result = openai.Audio.transcribe("whisper-1", audio_file)

    #     print("Transcription result:")
    #     print(result.text)

    #     # Simple intent detection
    #     text = result.text.lower()
    #     if "forward" in text or "ahead" in text:
    #         print("Detected intent: Move forward")
    #     elif "left" in text:
    #         print("Detected intent: Turn left")
    #     elif "right" in text:
    #         print("Detected intent: Turn right")
    #     elif "stop" in text or "halt" in text:
    #         print("Detected intent: Stop")
    #     else:
    #         print("Unknown command")

    # except Exception as e:
    #     print(f"Error processing audio: {e}")

# Run the simple demo
if __name__ == "__main__":
    simple_voice_command_demo()