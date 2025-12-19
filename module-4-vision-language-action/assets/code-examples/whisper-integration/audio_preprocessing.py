"""
Audio Preprocessing Pipeline
Complete audio preprocessing for Whisper-based voice recognition
"""

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
    # - 16-bit depth
    audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
    audio = audio.set_channels(1)        # Convert to mono
    audio = audio.set_sample_width(2)    # 16-bit depth

    # Export as WAV
    audio.export(output_path, format="wav")

    return output_path

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
    preprocessor = AudioPreprocessor()
    processed_file = preprocessor.preprocess_audio(optimized_file)

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
if __name__ == "__main__":
    print("Audio Preprocessing Pipeline")
    print("This module provides functions for:")
    print("1. Converting audio to Whisper-optimized format")
    print("2. Applying noise reduction and normalization")
    print("3. Assessing audio quality")
    print("4. Complete preprocessing workflow")
    print("\nTo use with a real audio file, call the functions with your file path:")
    print("# processed_file = complete_audio_preprocessing_workflow('your_audio_file.mp3')")