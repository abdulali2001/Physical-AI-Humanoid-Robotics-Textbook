# Voice-to-Action with OpenAI Whisper

## Overview

This chapter covers the implementation of voice command understanding using OpenAI Whisper for converting spoken natural language commands into structured text that can be processed by robotic systems. You'll learn to set up the Whisper model, process audio input, and extract actionable intents from voice commands.

The Voice-to-Action capability is foundational for natural human-robot interaction through voice, which is essential for subsequent cognitive planning and autonomous behavior capabilities.

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up OpenAI Whisper for voice command processing
- Process audio input and convert speech to text
- Extract actionable intents from transcribed text
- Implement error handling for voice recognition failures
- Achieve at least 85% accuracy in voice command recognition

## Prerequisites

Before starting this chapter, ensure you have:
- Python 3.11 or higher installed
- Access to OpenAI API (with sufficient credits)
- A microphone for testing voice commands
- Basic understanding of audio processing concepts
- Completed Module 1 (ROS 2) and Module 2 (Digital Twin) basics

## Whisper Integration Setup

### System Requirements

Before installing Whisper, ensure your system meets the following requirements:

- **Operating System**: Linux, Windows, or macOS
- **Python**: Version 3.8 or higher (Python 3.11 recommended)
- **Memory**: At least 4GB RAM (8GB+ recommended for larger models)
- **Storage**: 2-10GB available space depending on model size
- **Microphone**: For testing voice commands

### Installation Options

You have two options for using Whisper: via OpenAI's API or with local deployment.

#### Option 1: OpenAI API (Recommended for beginners)

This approach uses OpenAI's API for transcription, which is simpler to set up:

```bash
pip install openai python-dotenv
```

Create a `.env` file in your project directory:

```bash
OPENAI_API_KEY=your-api-key-here
```

#### Option 2: Local Deployment (Recommended for production)

For local deployment, you can install Whisper directly:

```bash
# Install PyTorch first
pip install torch torchvision torchaudio

# Install Whisper
pip install git+https://github.com/openai/whisper.git

# Or for a specific version
pip install openai-whisper
```

Additional dependencies for audio processing:
```bash
pip install python-dotenv pyaudio sounddevice numpy scipy
```

### OpenAI API Configuration

If using the OpenAI API, configure your API key:

```python
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
```

### Local Whisper Setup

If using local Whisper, install and set up the model:

```python
import whisper

# Load different model sizes based on your needs and resources:
# - tiny: Fastest, lowest accuracy (74M parameters)
# - base: Good balance (145M parameters)
# - small: Better accuracy (444M parameters)
# - medium: High accuracy, slower (769M parameters)
# - large: Highest accuracy, slowest (1550M parameters)

model = whisper.load_model("base")  # Choose appropriate size
```

### Complete Implementation Examples

Here are complete implementations for both API and local approaches:

#### OpenAI API Implementation

```python
import openai
import os
from dotenv import load_dotenv
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_with_api(audio_file_path, language=None, response_format="json"):
    """
    Transcribe audio using OpenAI's Whisper API

    Args:
        audio_file_path: Path to the audio file to transcribe
        language: Language of the audio (optional, e.g., 'en', 'es', 'fr')
        response_format: Format of the response ('json', 'text', 'srt', 'verbose_json', 'vtt')

    Returns:
        Transcription result in the specified format
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format=response_format
            )
        return transcript
    except Exception as e:
        print(f"API transcription failed: {e}")
        return None

# Example usage
# result = transcribe_with_api("my_audio.wav", language="en")
# print(result)
```

#### Local Whisper Implementation

```python
import whisper
import torch

def transcribe_with_local(audio_file_path, model_size="base", language=None, device="cpu"):
    """
    Transcribe audio using local Whisper model

    Args:
        audio_file_path: Path to the audio file to transcribe
        model_size: Size of the model ('tiny', 'base', 'small', 'medium', 'large')
        language: Language of the audio (optional, e.g., 'en', 'es', 'fr')
        device: Device to run the model on ('cpu' or 'cuda')

    Returns:
        Transcription result
    """
    try:
        # Load the model
        model = whisper.load_model(model_size, device=device)

        # Transcribe the audio
        result = model.transcribe(
            audio_file_path,
            language=language,
            verbose=False  # Set to True for progress output
        )

        return result
    except Exception as e:
        print(f"Local transcription failed: {e}")
        return None

# Example usage
# result = transcribe_with_local("my_audio.wav", model_size="base", device="cuda" if torch.cuda.is_available() else "cpu")
# print(result["text"])
```

### Comparison of Approaches

| Aspect | OpenAI API | Local Deployment |
|--------|------------|------------------|
| **Setup Complexity** | Simple (just API key) | Complex (model download, dependencies) |
| **Cost** | $0.006/minute of audio | Free after initial download |
| **Latency** | Higher (network + processing time) | Lower (local processing) |
| **Privacy** | Audio sent to OpenAI | Audio stays local |
| **Reliability** | Depends on OpenAI service availability | Independent of external services |
| **Model Updates** | Automatic | Manual updates required |
| **Customization** | Limited | Full control over model |
| **Resource Usage** | Minimal | High (RAM, storage, CPU/GPU) |

### When to Use Each Approach

#### Use OpenAI API when:
- Getting started quickly
- Privacy is not a primary concern
- Audio volume is relatively low
- You want minimal infrastructure management
- You need guaranteed availability

#### Use Local Deployment when:
- Privacy is critical
- High volume of transcriptions
- Need to minimize latency
- Working in offline environments
- Want to customize the model
- Long-term cost optimization is important

### API Integration Best Practices

When using the OpenAI API approach, follow these best practices:

```python
import openai
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperAPIClient:
    def __init__(self, max_retries=3, retry_delay=1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def transcribe_with_retry(self, audio_file_path, **kwargs):
        """Transcribe audio with retry logic for API calls"""
        for attempt in range(self.max_retries):
            try:
                with open(audio_file_path, "rb") as audio_file:
                    result = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file,
                        **kwargs
                    )
                logger.info(f"Successfully transcribed {audio_file_path}")
                return result
            except openai.error.RateLimitError:
                logger.warning(f"Rate limit exceeded on attempt {attempt + 1}, retrying...")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
            except openai.error.AuthenticationError:
                logger.error("Authentication failed. Check your API key.")
                return None
            except openai.error.APIError as e:
                logger.error(f"API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.retry_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.retry_delay)

        return None

# Example usage
# client = WhisperAPIClient()
# result = client.transcribe_with_retry("audio.wav", language="en", response_format="verbose_json")
```

### Local Deployment Best Practices

For local Whisper deployments, consider these best practices:

```python
import whisper
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalWhisperProcessor:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the Whisper model with appropriate device selection"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading {self.model_size} model on {device}")
            self.model = whisper.load_model(self.model_size, device=device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def transcribe(self, audio_file_path, **kwargs):
        """Transcribe audio with local model"""
        try:
            result = self.model.transcribe(audio_file_path, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Local transcription failed: {e}")
            return None

    def transcribe_with_options(self, audio_file_path, language=None, temperature=0.0, compression_ratio_threshold=2.4):
        """Transcribe with additional options for better accuracy"""
        try:
            result = self.model.transcribe(
                audio_file_path,
                language=language,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold
            )
            return result
        except Exception as e:
            logger.error(f"Advanced transcription failed: {e}")
            return None

# Example usage
# processor = LocalWhisperProcessor(model_size="base")
# result = processor.transcribe("audio.wav")
```

### Basic Whisper Implementation

Here's a simple implementation to get started with Whisper:

```python
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API key
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

## Audio Preprocessing and Enhancement

Audio preprocessing is crucial for achieving high recognition accuracy, especially in noisy environments. This section covers various techniques to enhance audio quality before feeding it to Whisper.

### Required Libraries

First, install the necessary libraries for audio processing:

```bash
pip install numpy scipy librosa pydub noisereduce
```

### Complete Audio Preprocessing Pipeline

Here's a comprehensive audio preprocessing pipeline that addresses multiple audio quality issues:

```python
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import librosa
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os

class AudioPreprocessor:
    def __init__(self, target_sample_rate=16000, target_channels=1):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

    def preprocess_audio(self, input_path, output_path=None):
        """
        Complete audio preprocessing pipeline

        Args:
            input_path: Path to input audio file
            output_path: Path for processed output (optional, auto-generated if None)

        Returns:
            Path to processed audio file
        """
        if output_path is None:
            # Create temporary file if no output path specified
            output_path = f"processed_{os.path.basename(input_path)}"

        # Load audio
        audio, sample_rate = self._load_audio(input_path)

        # Convert to target sample rate
        if sample_rate != self.target_sample_rate:
            audio = self._resample_audio(audio, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate

        # Ensure mono
        if self.target_channels == 1 and len(audio.shape) > 1:
            audio = self._to_mono(audio)

        # Apply noise reduction
        audio = self._reduce_noise(audio, sample_rate)

        # Normalize audio
        audio = self._normalize_audio(audio)

        # Apply high-pass filter to remove low-frequency rumble
        audio = self._apply_high_pass_filter(audio, sample_rate)

        # Save processed audio
        sf.write(output_path, audio, sample_rate)

        return output_path

    def _load_audio(self, file_path):
        """Load audio file using librosa (supports many formats)"""
        audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
        return audio, sample_rate

    def _resample_audio(self, audio, original_sr, target_sr):
        """Resample audio to target sample rate"""
        return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

    def _to_mono(self, audio):
        """Convert stereo audio to mono"""
        if len(audio.shape) > 1:
            return np.mean(audio, axis=0)
        return audio

    def _reduce_noise(self, audio, sample_rate):
        """Apply noise reduction using spectral gating"""
        # For simple noise reduction, we'll use a spectral gate approach
        # This is a simplified version - for production, consider using noisereduce library

        # Calculate noise threshold (simple approach)
        noise_threshold = np.std(audio) * 0.1

        # Apply basic noise gating
        audio_clean = np.where(np.abs(audio) > noise_threshold, audio, 0)

        return audio_clean

    def _normalize_audio(self, audio):
        """Normalize audio to -1 to 1 range"""
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            return audio / max_amplitude
        return audio

    def _apply_high_pass_filter(self, audio, sample_rate):
        """Apply high-pass filter to remove low-frequency rumble"""
        # Cutoff frequency for high-pass filter (typically 80-100 Hz for voice)
        cutoff_freq = 100.0
        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist

        # Create 4th order Butterworth high-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)

        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)

        return filtered_audio

# Example usage
preprocessor = AudioPreprocessor()
processed_file = preprocessor.preprocess_audio("input_audio.wav")
print(f"Processed audio saved to: {processed_file}")
```

### Advanced Noise Reduction Techniques

For more sophisticated noise reduction, especially in challenging environments, you can use the `noisereduce` library:

```python
import noisereduce as nr
import librosa
import numpy as np

def advanced_noise_reduction(audio_path, noise_sample_duration=0.5):
    """
    Apply advanced noise reduction using spectral subtraction

    Args:
        audio_path: Path to the audio file
        noise_sample_duration: Duration in seconds of initial audio to use as noise profile

    Returns:
        Path to noise-reduced audio file
    """
    # Load audio
    audio, sample_rate = librosa.load(audio_path, sr=None)

    # Estimate noise profile from the first few seconds of audio
    noise_sample_length = int(noise_sample_duration * sample_rate)
    noise_profile = audio[:noise_sample_length]

    # Apply noise reduction
    reduced_noise_audio = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        y_noise=noise_profile,
        prop_decrease=1.0,  # Maximum noise reduction
        stationary=False    # Non-stationary noise assumption
    )

    # Save the processed audio
    output_path = f"noise_reduced_{os.path.basename(audio_path)}"
    sf.write(output_path, reduced_noise_audio, sample_rate)

    return output_path

# Example usage
# reduced_file = advanced_noise_reduction("noisy_audio.wav")
```

### Audio Format Conversion and Optimization

Whisper works best with specific audio formats. Here's a utility to convert various formats to the optimal format:

```python
from pydub import AudioSegment
import os

def convert_to_whisper_format(input_path, output_path=None):
    """
    Convert audio to optimal format for Whisper processing

    Optimal format: 16kHz, mono, WAV format
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_whisper_optimized.wav"

    # Load audio file (pydub supports many formats)
    audio = AudioSegment.from_file(input_path)

    # Convert to optimal format for Whisper:
    # - 16kHz sample rate (Whisper was trained on this)
    # - Mono channel (simplifies processing)
    # - 16-bit depth (standard)
    audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
    audio = audio.set_channels(1)        # Convert to mono
    audio = audio.set_sample_width(2)    # 16-bit depth

    # Export as WAV
    audio.export(output_path, format="wav")

    return output_path

# Example usage
# optimized_file = convert_to_whisper_format("input_audio.mp3")
```

### Real-time Audio Preprocessing

For real-time applications, you might need to preprocess audio chunks:

```python
import queue
import threading
import numpy as np
from scipy import signal

class RealTimeAudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1024  # Process audio in chunks of 1024 samples
        self.audio_queue = queue.Queue()

        # Precompute filter coefficients for high-pass filter
        cutoff_freq = 100.0
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        self.b, self.a = signal.butter(4, normalized_cutoff, btype='high', analog=False)

        # Initialize filter state
        self.filter_state = signal.lfilter_zi(self.b, self.a)

    def preprocess_chunk(self, audio_chunk):
        """Preprocess a single chunk of audio in real-time"""
        # Apply high-pass filter with preserved state between chunks
        filtered_chunk, self.filter_state = signal.lfilter(
            self.b, self.a, audio_chunk, zi=self.filter_state
        )

        # Normalize chunk
        max_val = np.max(np.abs(filtered_chunk))
        if max_val > 0:
            filtered_chunk = filtered_chunk / max_val

        return filtered_chunk

    def start_processing(self, audio_generator):
        """Start real-time processing of audio chunks"""
        def process_thread():
            for chunk in audio_generator:
                processed_chunk = self.preprocess_chunk(chunk)
                self.audio_queue.put(processed_chunk)

        thread = threading.Thread(target=process_thread)
        thread.start()
        return thread

# Example usage would be with a real-time audio input source
```

### Audio Quality Assessment

It's important to assess the quality of your preprocessing:

```python
import librosa
import numpy as np

def assess_audio_quality(audio_path):
    """
    Assess various quality metrics of an audio file
    """
    audio, sample_rate = librosa.load(audio_path, sr=None)

    # Calculate metrics
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    rms_energy = np.sqrt(np.mean(audio**2))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    # Estimate signal-to-noise ratio (simplified)
    # This is a basic estimation - real SNR calculation is more complex
    signal_power = np.mean(audio**2)
    noise_power = np.var(audio - np.mean(audio))  # Approximate noise as variations around mean
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))  # Add small value to avoid division by zero

    quality_metrics = {
        'duration_seconds': duration,
        'sample_rate': sample_rate,
        'rms_energy': rms_energy,
        'zero_crossing_rate': zero_crossing_rate,
        'estimated_snr_db': snr,
        'is_silent': rms_energy < 0.001,  # Threshold for silence detection
        'is_too_loud': rms_energy > 0.8   # Threshold to detect clipping risk
    }

    return quality_metrics

# Example usage
# metrics = assess_audio_quality("processed_audio.wav")
# print(f"Audio quality metrics: {metrics}")
```

### Complete Processing Workflow

Here's how to combine all preprocessing steps into a complete workflow:

```python
def complete_audio_preprocessing_workflow(input_file_path):
    """
    Complete workflow: convert format → preprocess → assess quality
    """
    print(f"Processing audio file: {input_file_path}")

    # Step 1: Convert to optimal format
    print("Step 1: Converting to Whisper-optimized format...")
    optimized_file = convert_to_whisper_format(input_file_path)

    # Step 2: Apply advanced preprocessing
    print("Step 2: Applying noise reduction...")
    processed_file = advanced_noise_reduction(optimized_file)

    # Step 3: Assess quality
    print("Step 3: Assessing audio quality...")
    quality_metrics = assess_audio_quality(processed_file)

    print(f"Quality assessment: {quality_metrics}")

    # Check if audio quality is sufficient for Whisper
    if quality_metrics['estimated_snr_db'] < 10:
        print("Warning: Low SNR detected. Recognition accuracy may be affected.")
    if quality_metrics['is_silent']:
        print("Warning: Audio appears to be silent.")

    return processed_file

# Example usage
# final_file = complete_audio_preprocessing_workflow("input_audio.mp3")
```

### Noise Reduction

For better recognition accuracy, especially in noisy environments, implement audio preprocessing:

## Intent Extraction and Command Understanding

Intent extraction is the process of understanding what action the user wants the robot to perform based on the transcribed voice command. This section covers various approaches to extract structured intent from natural language.

### Simple Keyword Matching Approach

For basic intent extraction, you can start with keyword matching:

```python
def extract_intent_from_text(transcript):
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
```

### Advanced Pattern Matching with Parameters

For more sophisticated extraction that also captures parameters:

```python
import re

def extract_intent_with_parameters(transcript):
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
```

### Advanced Intent Extraction with LLMs

For more sophisticated intent extraction, you can use an LLM call:

```python
import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_intent_with_llm(transcript, available_actions=None):
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
        "intent": "&lt;action_type&gt;",
        "parameters": {{
            "target": "&lt;object or location if applicable&gt;",
            "direction": "&lt;direction if applicable&gt;",
            "distance": "&lt;distance if applicable&gt;",
            "speed": "&lt;speed if applicable&gt;",
            "location": "&lt;specific location if applicable&gt;"
        }},
        "confidence": &lt;float between 0.0 and 1.0&gt;,
        "action_sequence": ["&lt;list of actions to perform&gt;"]
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
```

### Intent Classification with Machine Learning

For a more scalable approach, you can use machine learning for intent classification:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        self.is_trained = False
        self.intent_labels = []

    def train(self, training_data):
        """
        Train the intent classifier

        Args:
            training_data: List of tuples (text, intent)
        """
        texts, intents = zip(*training_data)
        self.intent_labels = list(set(intents))

        self.pipeline.fit(texts, intents)
        self.is_trained = True

    def predict(self, text):
        """Predict intent for given text"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        predicted_intent = self.pipeline.predict([text])[0]
        confidence_scores = self.pipeline.predict_proba([text])[0]
        max_confidence = np.max(confidence_scores)

        return {
            "intent": predicted_intent,
            "confidence": float(max_confidence),
            "original_command": text
        }

# Example training data (in practice, you'd have much more data)
training_data = [
    ("move forward", "move_forward"),
    ("go forward", "move_forward"),
    ("move backward", "move_backward"),
    ("go backward", "move_backward"),
    ("turn left", "turn_left"),
    ("turn right", "turn_right"),
    ("stop", "stop"),
    ("go to kitchen", "navigate_to"),
    ("navigate to bedroom", "navigate_to"),
    ("grab the cup", "grasp_object"),
    ("pick up the red ball", "grasp_object")
]

# Example usage:
# classifier = IntentClassifier()
# classifier.train(training_data)
# result = classifier.predict("move forward slowly")
```

### Hybrid Intent Extraction System

A robust system often combines multiple approaches:

```python
class HybridIntentExtractor:
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.fallback_extractor = extract_intent_from_text
        self.llm_extractor = extract_intent_with_llm

    def extract_intent(self, transcript, use_llm_fallback=True):
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

# Example usage
extractor = HybridIntentExtractor()
result = extractor.extract_intent("Please navigate to the kitchen and grab the red cup")
print(f"Extracted intent: {result}")
```

### Intent Validation and Error Handling

It's important to validate extracted intents before execution:

```python
def validate_intent(intent_result, available_actions):
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

# Example usage
available_robot_actions = [
    "move_forward", "move_backward", "turn_left", "turn_right", "stop",
    "navigate_to", "detect_object", "grasp_object", "release_object"
]

# result = extractor.extract_intent("move forward")
# validated_result = validate_intent(result, available_robot_actions)
```

### Complete Intent Processing Pipeline

Here's how to combine intent extraction with the rest of the voice processing pipeline:

```python
def process_voice_command_complete(audio_file_path, use_local_whisper=False, model_size="base"):
    """
    Complete pipeline: Audio → Transcription → Intent Extraction → Validation
    """
    try:
        # Step 1: Transcribe audio
        if use_local_whisper:
            import whisper
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_file_path)
            transcript = result["text"]
        else:
            # Use OpenAI API transcription
            with open(audio_file_path, "rb") as audio_file:
                transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
                transcript = transcript_response.text

        print(f"Transcription: {transcript}")

        # Step 2: Extract intent
        extractor = HybridIntentExtractor()
        intent_result = extractor.extract_intent(transcript)

        # Step 3: Validate intent
        available_actions = [
            "move_forward", "move_backward", "turn_left", "turn_right", "stop",
            "navigate_to", "detect_object", "grasp_object", "release_object"
        ]
        validated_result = validate_intent(intent_result, available_actions)

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

# Example usage:
# result = process_voice_command_complete("command.wav")
# if result["status"] == "success" and result["intent_result"]["is_valid"]:
#     print(f"Executing intent: {result['intent_result']['intent']}")
# else:
#     print(f"Error processing command: {result.get('error', 'Invalid intent')}")
```

### Advanced Intent Extraction with LLMs

## Performance Benchmarking and Optimization

Achieving the target performance metrics for voice recognition is crucial for a responsive and effective voice-to-action system. This section covers how to benchmark, measure, and optimize your Whisper-based system.

### Performance Metrics

The key performance metrics for voice recognition in robotics applications are:

1. **Accuracy**: Percentage of correctly recognized voice commands
2. **Latency**: Time from audio input to action execution
3. **Throughput**: Number of commands processed per unit time
4. **Resource Usage**: CPU, memory, and network consumption
5. **Robustness**: Performance under various environmental conditions

### Benchmarking Framework

Here's a comprehensive benchmarking framework to measure your system's performance:

```python
import time
import statistics
import json
from typing import List, Dict, Any
import os
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    accuracy: float
    avg_latency: float
    min_latency: float
    max_latency: float
    std_deviation: float
    throughput: float  # commands per second
    total_commands: int
    correct_commands: int
    test_duration: float

class VoiceRecognitionBenchmark:
    def __init__(self, system_under_test):
        self.sut = system_under_test
        self.results = []

    def run_accuracy_test(self, test_cases: List[Dict[str, str]]) -> BenchmarkResult:
        """
        Run accuracy benchmark with known input-output pairs

        Args:
            test_cases: List of dicts with 'audio_file' and 'expected_text' keys
        """
        start_time = time.time()
        latencies = []
        correct_count = 0
        total_count = len(test_cases)

        for test_case in test_cases:
            start_proc = time.time()

            # Process the audio file
            result = self.sut.process_audio_file(test_case['audio_file'])

            end_proc = time.time()
            latency = end_proc - start_proc
            latencies.append(latency)

            # Compare result with expected output
            if self._compare_transcriptions(result['transcription'], test_case['expected_text']):
                correct_count += 1

        test_duration = time.time() - start_time
        accuracy = correct_count / total_count if total_count > 0 else 0
        avg_latency = statistics.mean(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        throughput = total_count / test_duration if test_duration > 0 else 0

        return BenchmarkResult(
            accuracy=accuracy,
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            std_deviation=std_dev,
            throughput=throughput,
            total_commands=total_count,
            correct_commands=correct_count,
            test_duration=test_duration
        )

    def _compare_transcriptions(self, actual: str, expected: str, threshold: float = 0.8) -> bool:
        """
        Compare two transcriptions using a similarity metric
        """
        # Simple word-based similarity (you might want to use more sophisticated methods)
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())

        if not actual_words and not expected_words:
            return True
        if not actual_words or not expected_words:
            return False

        intersection = actual_words.intersection(expected_words)
        union = actual_words.union(expected_words)
        similarity = len(intersection) / len(union)

        return similarity >= threshold

    def run_latency_test(self, audio_file: str, iterations: int = 10) -> Dict[str, float]:
        """
        Measure processing latency over multiple iterations
        """
        latencies = []

        for _ in range(iterations):
            start_time = time.time()
            self.sut.process_audio_file(audio_file)
            end_time = time.time()
            latencies.append(end_time - start_time)

        return {
            'avg_latency': statistics.mean(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'percentile_95': self._percentile(latencies, 95) if latencies else 0
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight

# Example usage of the benchmarking framework
# benchmark = VoiceRecognitionBenchmark(your_system)
# test_cases = [
#     {"audio_file": "test1.wav", "expected_text": "move forward"},
#     {"audio_file": "test2.wav", "expected_text": "turn left"},
#     # Add more test cases...
# ]
# results = benchmark.run_accuracy_test(test_cases)
# print(f"Accuracy: {results.accuracy:.2%}")
# print(f"Average latency: {results.avg_latency:.3f}s")
```

### Real-time Performance Monitoring

For monitoring performance during actual operation:

```python
import threading
import time
from collections import deque
import logging

class RealTimePerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.accuracies = deque(maxlen=window_size)
        self.throughput_samples = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.start_time = time.time()

    def record_transaction(self, latency: float, accuracy: float = None):
        """Record a single transaction for monitoring"""
        with self.lock:
            self.latencies.append(latency)
            if accuracy is not None:
                self.accuracies.append(accuracy)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        with self.lock:
            if not self.latencies:
                return {}

            recent_latencies = list(self.latencies)
            current_metrics = {
                'avg_latency': sum(recent_latencies) / len(recent_latencies),
                'min_latency': min(recent_latencies),
                'max_latency': max(recent_latencies),
                'count': len(recent_latencies)
            }

            if self.accuracies:
                recent_accuracies = list(self.accuracies)
                current_metrics['avg_accuracy'] = sum(recent_accuracies) / len(recent_accuracies)

            # Calculate throughput (commands per second)
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                current_metrics['throughput'] = len(self.latencies) / elapsed_time

            return current_metrics

    def start_monitoring(self, interval: float = 1.0):
        """Start periodic monitoring output"""
        def monitor_loop():
            while True:
                time.sleep(interval)
                metrics = self.get_current_metrics()
                logging.info(f"Performance metrics: {metrics}")

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread

# Example usage
# monitor = RealTimePerformanceMonitor()
# monitor.start_monitoring()
```

### Environmental Performance Testing

Test your system under various environmental conditions:

```python
import os
import subprocess

class EnvironmentalPerformanceTester:
    def __init__(self, base_system):
        self.system = base_system

    def test_with_noise_levels(self, base_audio: str, noise_levels: List[float]):
        """
        Test performance with different background noise levels
        """
        results = {}

        for noise_level in noise_levels:
            # Add noise to audio file (this is a simplified example)
            noisy_audio = self._add_noise(base_audio, noise_level)

            # Test the system with noisy audio
            result = self.system.process_audio_file(noisy_audio)

            results[f"noise_{noise_level}"] = result
            os.remove(noisy_audio)  # Clean up temporary file

        return results

    def test_with_different_accents(self, audio_samples_by_accent: Dict[str, str]):
        """
        Test performance with different speaker accents
        """
        results = {}

        for accent, audio_file in audio_samples_by_accent.items():
            result = self.system.process_audio_file(audio_file)
            results[accent] = result

        return results

    def _add_noise(self, audio_file: str, noise_level: float) -> str:
        """
        Add background noise to an audio file (implementation would use audio processing libraries)
        """
        # This is a placeholder - in practice you'd use libraries like pydub or librosa
        # to add noise to the audio file
        noisy_file = f"noisy_{noise_level}_{os.path.basename(audio_file)}"

        # Example using sox if available (audio processing tool)
        # subprocess.run(['sox', audio_file, noisy_file, 'noiseprof', 'noise.prof'])
        # subprocess.run(['sox', audio_file, noisy_file, 'noisered', 'noise.prof', f'{noise_level}'])

        return noisy_file
```

### Optimization Strategies

Here are several optimization strategies to improve performance:

#### 1. Caching for Frequently Used Commands

```python
from functools import lru_cache
import hashlib

class OptimizedWhisperProcessor:
    def __init__(self, whisper_model):
        self.model = whisper_model
        # Cache with size limit to prevent memory issues
        self._transcribe_with_cache = lru_cache(maxsize=128)(self._transcribe_impl)

    def _transcribe_impl(self, audio_hash: str, audio_path: str):
        """Internal transcription method that gets cached"""
        return self.model.transcribe(audio_path)

    def transcribe(self, audio_path: str):
        """Transcribe with automatic caching"""
        # Create hash of audio file content
        with open(audio_path, 'rb') as f:
            audio_hash = hashlib.md5(f.read()).hexdigest()

        return self._transcribe_with_cache(audio_hash, audio_path)
```

#### 2. Asynchronous Processing

```python
import asyncio
import concurrent.futures
from typing import List

class AsyncWhisperProcessor:
    def __init__(self, model, max_workers=4):
        self.model = model
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def transcribe_async(self, audio_path: str):
        """Asynchronously transcribe audio"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.model.transcribe,
            audio_path
        )

    async def transcribe_batch_async(self, audio_paths: List[str]):
        """Transcribe multiple files concurrently"""
        tasks = [self.transcribe_async(path) for path in audio_paths]
        return await asyncio.gather(*tasks)
```

#### 3. Model Optimization

```python
import whisper
import torch

class OptimizedModelLoader:
    @staticmethod
    def load_optimized_model(model_size: str = "base", device: str = None):
        """
        Load Whisper model with optimizations
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model with specific optimizations
        model = whisper.load_model(
            model_size,
            device=device,
            download_root="./models"  # Specify download location
        )

        # Set to evaluation mode
        model.eval()

        # If using CUDA, enable optimizations
        if device == "cuda":
            model = model.half()  # Use half precision for faster inference

        return model
```

### Performance Testing Script

Here's a complete script to run performance tests:

```python
#!/usr/bin/env python3
"""
Performance testing script for Whisper-based voice recognition
"""
import os
import sys
import argparse
import json
from datetime import datetime

def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark"""
    print("Starting comprehensive performance benchmark...")

    # Initialize system
    model = whisper.load_model("base")
    system = VoiceRecognitionSystem(model)  # Assuming you have this class

    # Initialize benchmark
    benchmark = VoiceRecognitionBenchmark(system)

    # Define test cases (you would load these from a file in practice)
    test_cases = [
        {"audio_file": "test_move_forward.wav", "expected_text": "move forward"},
        {"audio_file": "test_turn_left.wav", "expected_text": "turn left"},
        {"audio_file": "test_stop.wav", "expected_text": "stop"},
        # Add more test cases...
    ]

    # Run accuracy test
    print("Running accuracy test...")
    accuracy_results = benchmark.run_accuracy_test(test_cases)

    # Run latency test
    print("Running latency test...")
    latency_results = benchmark.run_latency_test(test_cases[0]['audio_file'], iterations=20)

    # Compile results
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "accuracy_results": {
            "accuracy": accuracy_results.accuracy,
            "avg_latency": accuracy_results.avg_latency,
            "min_latency": accuracy_results.min_latency,
            "max_latency": accuracy_results.max_latency,
            "std_deviation": accuracy_results.std_deviation,
            "throughput": accuracy_results.throughput,
            "total_commands": accuracy_results.total_commands,
            "correct_commands": accuracy_results.correct_commands,
            "test_duration": accuracy_results.test_duration
        },
        "latency_results": latency_results
    }

    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Accuracy: {benchmark_results['accuracy_results']['accuracy']:.2%}")
    print(f"Average Latency: {benchmark_results['accuracy_results']['avg_latency']:.3f}s")
    print(f"Min Latency: {benchmark_results['accuracy_results']['min_latency']:.3f}s")
    print(f"Max Latency: {benchmark_results['accuracy_results']['max_latency']:.3f}s")
    print(f"Throughput: {benchmark_results['accuracy_results']['throughput']:.2f} commands/sec")
    print(f"Test Duration: {benchmark_results['accuracy_results']['test_duration']:.2f}s")

    # Save results to file
    with open(f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nResults saved to: benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    return benchmark_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Whisper performance benchmarks")
    parser.add_argument("--model-size", choices=["tiny", "base", "small", "medium", "large"],
                       default="base", help="Whisper model size to benchmark")
    parser.add_argument("--test-file", help="Single audio file for quick latency test")

    args = parser.parse_args()

    if args.test_file:
        # Run quick test on single file
        print(f"Running quick latency test on {args.test_file}...")
        # Implementation for single file test
    else:
        # Run comprehensive benchmark
        run_comprehensive_benchmark()
```

### Meeting Performance Targets

To meet the target performance of 85% accuracy and sub-2-second latency:

1. **For Accuracy (Target: ≥85%)**:
    - Use appropriate model size (base or larger for better accuracy)
    - Implement proper audio preprocessing
    - Consider domain-specific fine-tuning
    - Use confidence thresholding to filter low-confidence results

2. **For Latency (Target: &lt;2 seconds)**:
    - Use local Whisper models instead of API calls when possible
    - Implement caching for repeated commands
    - Optimize audio preprocessing pipeline
    - Use GPU acceleration if available
    - Consider using smaller models for real-time applications

### Performance Monitoring Dashboard

For ongoing monitoring, consider implementing a simple dashboard:

```python
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PerformanceDashboard:
    def __init__(self):
        self.metrics_history = []

    def add_metrics(self, metrics: Dict[str, float]):
        """Add new metrics to history"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

    def plot_performance_trends(self, days_back: int = 7):
        """Plot performance trends over time"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        # Filter metrics from the specified time period
        filtered_data = [m for m in self.metrics_history
                        if m['timestamp'] >= start_time]

        if not filtered_data:
            print("No data available for the specified time period")
            return

        timestamps = [m['timestamp'] for m in filtered_data]
        latencies = [m['metrics'].get('avg_latency', 0) for m in filtered_data]
        accuracies = [m['metrics'].get('avg_accuracy', 0) for m in filtered_data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot latency
        ax1.plot(timestamps, latencies, 'b-', label='Average Latency')
        ax1.axhline(y=2.0, color='r', linestyle='--', label='Target (2s)')
        ax1.set_ylabel('Latency (seconds)')
        ax1.set_title('Voice Recognition Latency Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(timestamps, accuracies, 'g-', label='Average Accuracy')
        ax2.axhline(y=0.85, color='r', linestyle='--', label='Target (85%)')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Time')
        ax2.set_title('Voice Recognition Accuracy Over Time')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('performance_trends.png')
        plt.show()
```

## Troubleshooting Common Whisper Integration Issues

This section covers common issues you might encounter when integrating Whisper for voice-to-action systems and their solutions.

### Audio Format and Quality Issues

#### Issue: "Unsupported audio format" error
**Symptoms**: Whisper rejects audio files with format errors
**Causes**: Audio file in unsupported format or encoding
**Solutions**:
1. Convert audio to supported formats: MP3, MP4, M4A, WAV, MPEG, MPGA, WEBM, or FLAC
2. Use audio processing libraries like `pydub` for format conversion:

```python
from pydub import AudioSegment

def convert_audio_format(input_path, output_path, output_format="wav"):
    """Convert audio to a supported format"""
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format=output_format)
    return output_path

# Example usage
# convert_audio_format("input.ogg", "output.wav")
```

#### Issue: Poor transcription quality
**Symptoms**: Low accuracy, garbled text output
**Causes**: Low audio quality, background noise, incorrect sample rate
**Solutions**:
1. Apply audio preprocessing as described in the Audio Preprocessing section
2. Ensure audio sample rate is appropriate (16kHz recommended for Whisper)
3. Check that microphone is positioned correctly and audio levels are appropriate

#### Issue: Audio too long for processing
**Symptoms**: API errors or extremely long processing times
**Causes**: Audio files longer than Whisper's recommended duration (30 seconds for API)
**Solutions**:
1. Split long audio files into shorter segments:

```python
def split_audio_file(file_path, max_duration=25):
    """Split audio file into smaller segments"""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # pydub uses milliseconds

    if duration_seconds <= max_duration:
        return [file_path]  # No need to split

    segments = []
    segment_length = max_duration * 1000  # Convert to milliseconds

    for i in range(0, len(audio), int(segment_length)):
        segment = audio[i:i + int(segment_length)]
        segment_path = f"{file_path}_segment_{i//int(segment_length)}.wav"
        segment.export(segment_path, format="wav")
        segments.append(segment_path)

    return segments
```

### API and Network Issues

#### Issue: Rate limit exceeded errors
**Symptoms**: HTTP 429 errors from OpenAI API
**Causes**: Exceeding API rate limits
**Solutions**:
1. Implement exponential backoff retry logic:

```python
import time
import random

def transcribe_with_backoff(audio_file, max_retries=5):
    """Transcribe with exponential backoff for rate limits"""
    for attempt in range(max_retries):
        try:
            with open(audio_file, "rb") as af:
                result = openai.Audio.transcribe("whisper-1", af)
            return result
        except openai.error.RateLimitError:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
```

#### Issue: Authentication failures
**Symptoms**: HTTP 401 errors, "Incorrect API key" messages
**Causes**: Invalid or missing API key
**Solutions**:
1. Verify your API key is correct and has sufficient permissions
2. Check that the `.env` file is properly loaded:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

openai.api_key = api_key
```

### Local Whisper Issues

#### Issue: Out of memory errors
**Symptoms**: CUDA out of memory errors when using local Whisper
**Causes**: Model size too large for available GPU memory
**Solutions**:
1. Use a smaller model size (tiny, base instead of large)
2. Use CPU instead of GPU for processing
3. Process audio in smaller chunks

```python
def safe_transcribe(audio_path, model_size="base", device="cpu"):
    """Safely transcribe with fallback options"""
    import torch

    try:
        # Try with specified device
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_path)
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory exceeded, falling back to CPU")
            model = whisper.load_model(model_size, device="cpu")
            result = model.transcribe(audio_path)
            return result
        else:
            raise
```

#### Issue: Slow processing times
**Symptoms**: Long transcription times (&gt;30 seconds for short audio)
**Causes**: CPU-only processing, large model size, insufficient system resources
**Solutions**:
1. Ensure GPU acceleration is enabled if available
2. Use smaller model sizes for real-time applications
3. Optimize system resources (close other applications)

### Audio Input Issues

#### Issue: No audio input detected
**Symptoms**: Silent or empty transcription results
**Causes**: Microphone not properly configured, audio input issues
**Solutions**:
1. Test microphone independently using system tools
2. Verify audio input permissions in your application
3. Use pyaudio to capture and save audio for testing:

```python
import pyaudio
import wave

def record_audio(duration=5, filename="temp_recording.wav", sample_rate=16000):
    """Record audio from microphone for testing"""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()

    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk
    )

    print("Recording...")
    frames = []

    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename
```

### Intent Extraction Issues

#### Issue: Low confidence in intent recognition
**Symptoms**: High number of "unknown" intents, poor command recognition
**Causes**: Insufficient training data, complex commands, audio quality issues
**Solutions**:
1. Improve audio preprocessing to enhance transcription quality
2. Expand keyword matching patterns
3. Use LLM-based intent extraction for complex commands
4. Implement confidence thresholding:

```python
def process_with_confidence_threshold(transcript, min_confidence=0.5):
    """Process transcript with confidence thresholding"""
    intent_result = extract_intent_with_llm(transcript)

    if intent_result.get("confidence", 0) < min_confidence:
        return {
            "intent": "uncertain",
            "original_command": transcript,
            "confidence": intent_result.get("confidence", 0),
            "suggested_alternatives": ["Could you repeat that?", "Please speak more clearly"]
        }

    return intent_result
```

### Performance Issues

#### Issue: High latency in voice processing
**Symptoms**: Delay &gt;2 seconds from voice input to action execution
**Causes**: Network latency, large model size, inefficient pipeline
**Solutions**:
1. Use local Whisper models instead of API calls
2. Implement caching for common commands
3. Optimize audio preprocessing pipeline
4. Use asynchronous processing where possible

#### Issue: Inconsistent performance
**Symptoms**: Performance varies significantly between runs
**Causes**: System resource fluctuations, network variability, model caching
**Solutions**:
1. Monitor system resources during processing
2. Implement proper benchmarking as described in the Performance section
3. Use consistent model loading and warm-up procedures

### Debugging Strategies

#### Enable detailed logging
```python
import logging

# Set up logging for Whisper integration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whisper_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('whisper_vla')
```

#### Test with known audio samples
Always test your integration with audio samples of known content to verify accuracy:

```python
def test_integration():
    """Test the complete integration with known samples"""
    test_samples = {
        "move_forward.wav": "move forward",
        "turn_left.wav": "turn left",
        "stop.wav": "stop"
    }

    results = {}
    for audio_file, expected_text in test_samples.items():
        try:
            result = process_voice_command_complete(audio_file)
            results[audio_file] = {
                'expected': expected_text,
                'transcribed': result.get('transcription', ''),
                'intent': result.get('intent_result', {}).get('intent', 'unknown'),
                'success': expected_text.lower() in result.get('transcription', '').lower()
            }
        except Exception as e:
            results[audio_file] = {
                'error': str(e),
                'success': False
            }

    return results
```

### Getting Help

If you encounter issues not covered here:

1. Check the official OpenAI documentation for API-related issues
2. Review Whisper's GitHub repository for local installation issues
3. Verify your Python environment and dependencies
4. Test components in isolation to identify the source of problems
5. Consider using the troubleshooting guide template created for this module

## Reading Level Validation

This documentation is designed to be accessible to students and developers with a Flesch-Kincaid grade level of 9-11 (typically corresponding to ages 14-16). To ensure the content remains accessible:

### Complexity Guidelines Applied

1. **Sentence Structure**: Sentences are kept to an average of 15-20 words where possible
2. **Vocabulary**: Technical terms are defined when first used and explained in context
3. **Paragraph Length**: Paragraphs are broken into digestible chunks with clear headings
4. **Active Voice**: Content is written primarily in active voice for clarity
5. **Technical Jargon**: Minimized and explained when necessary

### Example of Complex vs. Accessible Language

**Complex**: "The implementation of the Whisper-based voice recognition system requires the instantiation of a neural network model with specific architectural parameters."

**Accessible**: "To use the Whisper voice recognition system, you need to load a neural network model with the right settings."

### Validation Approach

While the Flesch-Kincaid score is calculated automatically by many tools, you can validate the accessibility of your content by:

1. Reading sentences aloud to check for natural flow
2. Ensuring technical concepts are explained step-by-step
3. Using examples that relate to common experiences
4. Breaking down complex processes into smaller, numbered steps

## Practical Examples and Code Snippets

This section provides complete, practical examples that you can download and run to experiment with voice-to-action systems.

### Complete Voice Command System Example

Here's a complete example that combines all the concepts we've covered into a working voice command system:

```python
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
        self.command_processor = HybridIntentExtractor()  # From previous examples

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
            intent_result = self.command_processor.extract_intent(transcript)

            # Step 3: Validate intent
            available_actions = [
                "move_forward", "move_backward", "turn_left", "turn_right", "stop",
                "navigate_to", "detect_object", "grasp_object", "release_object"
            ]
            validated_result = validate_intent(intent_result, available_actions)

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

    # Example: Process a sample audio file
    # Replace 'sample_command.wav' with an actual audio file path
    result = vcs.process_audio_file("sample_command.wav")

    if result["status"] == "success":
        print(f"Transcription: {result['transcription']}")
        print(f"Intent: {result['intent_result']['intent']}")
        print(f"Confidence: {result['intent_result'].get('confidence', 'N/A')}")

        # Execute the command if it's valid
        if result["intent_result"]["is_valid"]:
            execution_result = vcs.execute_command(result["intent_result"])
            print(f"Execution result: {execution_result}")
        else:
            print("Command validation failed:", result["intent_result"].get("validation_error"))
    else:
        print(f"Processing failed: {result['error']}")

    vcs.stop_listening()
```

### Simple Voice Command Demo

For beginners, here's a simplified example to get started quickly:

```python
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

    try:
        # Transcribe the audio file
        with open(audio_file_path, "rb") as audio_file:
            result = openai.Audio.transcribe("whisper-1", audio_file)

        print("Transcription result:")
        print(result.text)

        # Simple intent detection
        text = result.text.lower()
        if "forward" in text or "ahead" in text:
            print("Detected intent: Move forward")
        elif "left" in text:
            print("Detected intent: Turn left")
        elif "right" in text:
            print("Detected intent: Turn right")
        elif "stop" in text or "halt" in text:
            print("Detected intent: Stop")
        else:
            print("Unknown command")

    except Exception as e:
        print(f"Error processing audio: {e}")

# Run the simple demo
# simple_voice_command_demo()
```

### Downloadable Code Examples

The following code examples are available for download in the assets directory:

1. **complete_voice_system.py** - Full voice command system with all features
2. **simple_demo.py** - Basic example for beginners
3. **audio_preprocessing.py** - Complete audio preprocessing pipeline
4. **intent_extraction.py** - Advanced intent extraction techniques
5. **benchmarking_tools.py** - Performance testing utilities

These examples can be downloaded from the course materials or cloned from the repository.

### Example Project Structure

When implementing your own voice-to-action system, consider this project structure:

```
voice-to-action-project/
├── main.py                 # Main application entry point
├── audio_utils.py          # Audio processing functions
├── intent_extractor.py     # Intent extraction logic
├── command_executor.py     # Command execution simulation
├── config.py               # Configuration settings
├── tests/                  # Unit tests
│   ├── test_audio.py
│   ├── test_intent.py
│   └── test_end_to_end.py
├── samples/                # Audio sample files for testing
│   ├── move_forward.wav
│   ├── turn_left.wav
│   └── stop.wav
├── requirements.txt        # Python dependencies
└── .env                   # Environment variables (API keys, etc.)
```

### Running the Examples

To run these examples:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

4. Run the example:
   ```bash
   python complete_voice_system.py
   ```

### Troubleshooting the Examples

If you encounter issues with the examples:

1. Verify your OpenAI API key is correct and has sufficient credits
2. Check that all required packages are installed
3. Ensure your audio files are in the correct format (WAV, MP3, etc.)
4. Confirm your internet connection for API calls
5. Check the troubleshooting section for common issues

## Voice Command Processing Pipeline

Here's a complete voice command processing pipeline:

```python
    def __init__(self):
        self.command_queue = queue.Queue()
        self.is_listening = False

    def start_listening(self):
        """Start listening for voice commands in a separate thread"""
        self.is_listening = True
        listener_thread = threading.Thread(target=self._listen_for_commands)
        listener_thread.start()

    def _listen_for_commands(self):
        """Internal method to listen for and process voice commands"""
        while self.is_listening:
            # This is a simplified example - in practice you'd use
            # pyaudio or another library to record from microphone
            print("Listening for voice command...")

            # Simulate receiving a command
            # In practice, this would come from audio recording
            command_file = self._record_audio()

            if command_file:
                # Process the command
                result = self.process_command_file(command_file)
                self.command_queue.put(result)

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    def process_command_file(self, audio_file_path):
        """Process an audio file through the complete pipeline"""
        try:
            # 1. Preprocess audio
            processed_file = preprocess_audio(audio_file_path)

            # 2. Transcribe using Whisper
            transcription = transcribe_audio(processed_file)

            # 3. Extract intent
            intent_result = extract_intent_with_llm(transcription)

            # 4. Return complete result
            return {
                "transcription": transcription,
                "intent_result": intent_result,
                "timestamp": time.time(),
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": time.time(),
                "status": "error"
            }

    def _record_audio(self):
        """Record audio from microphone (simplified placeholder)"""
        # In a real implementation, this would record from the microphone
        # For now, return None to indicate no recording functionality
        return None

    def stop_listening(self):
        """Stop listening for voice commands"""
        self.is_listening = False

# Example usage:
# processor = VoiceCommandProcessor()
# processor.start_listening()
#
# # In another thread or at a later time:
# try:
#     result = processor.command_queue.get(timeout=5.0)
#     print(f"Processed command: {result}")
# except queue.Empty:
#     print("No command received")
#
# processor.stop_listening()
```

## Performance Optimization

### Caching for Faster Response

To reduce API calls and improve response time, implement caching:

```python
import hashlib
from functools import lru_cache

class WhisperCache:
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize

    def get(self, audio_file_path):
        """Get cached transcription if available"""
        # Create a hash of the file content
        with open(audio_file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        return self.cache.get(file_hash)

    def set(self, audio_file_path, transcription):
        """Cache a transcription"""
        with open(audio_file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # Remove oldest entry if cache is full
        if len(self.cache) >= self.maxsize:
            # Remove first item (oldest in insertion order)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[file_hash] = transcription

# Global cache instance
whisper_cache = WhisperCache()

def transcribe_audio_with_cache(audio_file_path):
    """Transcribe audio with caching"""
    cached_result = whisper_cache.get(audio_file_path)
    if cached_result:
        print("Using cached transcription")
        return cached_result

    # Not in cache, so transcribe and cache the result
    result = transcribe_audio(audio_file_path)
    whisper_cache.set(audio_file_path, result)
    return result
```

## Error Handling and Recovery

### Handling Whisper Failures

Implement robust error handling for Whisper API failures:

```python
import time
from typing import Optional

def transcribe_audio_with_retry(audio_file_path, max_retries=3):
    """Transcribe audio with retry logic"""
    for attempt in range(max_retries):
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
            return transcript.text
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded, waiting before retry {attempt + 1}/{max_retries}")
            time.sleep(2 ** attempt)  # Exponential backoff
        except openai.error.AuthenticationError:
            print("Authentication failed. Check your API key.")
            return None
        except openai.error.APIError as e:
            print(f"API error occurred: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
        except FileNotFoundError:
            print(f"Audio file not found: {audio_file_path}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)

    return None  # All retries failed
```

## Testing Your Implementation

### Accuracy Testing

To test the accuracy of your voice-to-action system:

```python
def test_voice_accuracy():
    """Test the accuracy of voice recognition with known commands"""
    test_cases = [
        {"audio_file": "test_move_forward.wav", "expected_intent": "move_forward"},
        {"audio_file": "test_turn_left.wav", "expected_intent": "turn_left"},
        {"audio_file": "test_stop.wav", "expected_intent": "stop"},
        # Add more test cases...
    ]

    correct_predictions = 0
    total_tests = len(test_cases)

    for test_case in test_cases:
        result = process_command_file(test_case["audio_file"])
        if result["status"] == "success":
            predicted_intent = result["intent_result"]["intent"]
            if predicted_intent == test_case["expected_intent"]:
                correct_predictions += 1

    accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_tests})")

    return accuracy >= 0.85  # Target 85% accuracy

# Run the test
# success = test_voice_accuracy()
# print(f"Test {'PASSED' if success else 'FAILED'}")
```

## Real-World Considerations

### Microphone Quality and Positioning

- Use a quality microphone positioned appropriately for voice capture
- Consider using a headset microphone for consistent audio quality
- Position the microphone to minimize background noise

### Environmental Factors

- Background noise can significantly impact recognition accuracy
- Consider implementing noise suppression algorithms
- Test your system in various acoustic environments

### Privacy and Data Handling

- Be mindful of privacy when processing voice data
- Consider using local Whisper models for sensitive applications
- Implement proper data handling and retention policies

## Summary

In this chapter, you learned how to implement voice command understanding using OpenAI Whisper. You've set up the basic infrastructure for converting speech to text, extracting actionable intents, and handling errors gracefully.

The voice-to-action capability forms the foundation for natural human-robot interaction and is essential for the more advanced cognitive planning and autonomous behavior capabilities covered in subsequent chapters.

## Next Steps

- Proceed to Chapter 2: Cognitive Planning with LLMs to learn how to translate natural language commands into ROS 2 action sequences
- Practice with different voice commands to improve your understanding
- Experiment with different Whisper model sizes to balance accuracy and performance