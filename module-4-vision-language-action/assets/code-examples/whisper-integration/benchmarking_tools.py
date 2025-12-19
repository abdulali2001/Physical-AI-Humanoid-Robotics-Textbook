"""
Benchmarking Tools for Voice Recognition
Performance testing and optimization utilities
"""

import time
import statistics
import json
from typing import List, Dict, Any
import os
from dataclasses import dataclass
import whisper
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.latencies = []
        self.accuracies = []
        self.throughput_samples = []

    def record_transaction(self, latency: float, accuracy: float = None):
        """Record a single transaction for monitoring"""
        self.latencies.append(latency)
        if accuracy is not None:
            self.accuracies.append(accuracy)

        # Keep lists within window size
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
        if len(self.accuracies) > self.window_size:
            self.accuracies.pop(0)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if not self.latencies:
            return {}

        current_metrics = {
            'avg_latency': sum(self.latencies) / len(self.latencies),
            'min_latency': min(self.latencies),
            'max_latency': max(self.latencies),
            'count': len(self.latencies)
        }

        if self.accuracies:
            current_metrics['avg_accuracy'] = sum(self.accuracies) / len(self.accuracies)

        return current_metrics

def run_simple_latency_test(audio_file_path: str, iterations: int = 5) -> Dict[str, float]:
    """
    Simple function to test latency of Whisper API
    """
    latencies = []

    for i in range(iterations):
        start_time = time.time()
        try:
            with open(audio_file_path, "rb") as audio_file:
                result = openai.Audio.transcribe("whisper-1", audio_file)
            end_time = time.time()
            latencies.append(end_time - start_time)
            print(f"Iteration {i+1}/{iterations}: {end_time - start_time:.3f}s")
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")

    if latencies:
        return {
            'avg_latency': statistics.mean(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'total_time': sum(latencies),
            'iterations': len(latencies)
        }
    else:
        return {'error': 'No successful iterations'}

def run_simple_accuracy_test(test_cases: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Simple function to test accuracy of Whisper API
    """
    correct = 0
    total = len(test_cases)
    latencies = []

    for test_case in test_cases:
        start_time = time.time()
        try:
            with open(test_case['audio_file'], "rb") as audio_file:
                result = openai.Audio.transcribe("whisper-1", audio_file)
            end_time = time.time()

            latencies.append(end_time - start_time)

            # Simple comparison (case-insensitive, word-based)
            actual_words = set(result.text.lower().split())
            expected_words = set(test_case['expected_text'].lower().split())
            intersection = actual_words.intersection(expected_words)
            union = actual_words.union(expected_words)

            if union:
                similarity = len(intersection) / len(union)
                if similarity >= 0.8:  # 80% word overlap threshold
                    correct += 1
        except Exception as e:
            print(f"Error processing {test_case['audio_file']}: {e}")

    accuracy = correct / total if total > 0 else 0
    avg_latency = statistics.mean(latencies) if latencies else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_latency': avg_latency,
        'latencies': latencies
    }

def print_performance_report(results: Dict[str, Any], test_type: str = "General"):
    """
    Print a formatted performance report
    """
    print(f"\n{'='*50}")
    print(f"{test_type} PERFORMANCE REPORT")
    print(f"{'='*50}")

    if 'accuracy' in results:
        print(f"Accuracy: {results['accuracy']:.2%} ({results.get('correct', 0)}/{results.get('total', 0)})")

    if 'avg_latency' in results:
        print(f"Average Latency: {results['avg_latency']:.3f}s")

    if 'min_latency' in results:
        print(f"Min Latency: {results['min_latency']:.3f}s")
        print(f"Max Latency: {results['max_latency']:.3f}s")

    if 'std_dev' in results:
        print(f"Std Deviation: {results['std_dev']:.3f}s")

    if 'throughput' in results:
        print(f"Throughput: {results['throughput']:.2f} commands/sec")

    if 'test_duration' in results:
        print(f"Test Duration: {results['test_duration']:.2f}s")

    if 'error' in results:
        print(f"Error: {results['error']}")

    print(f"{'='*50}\n")

# Example usage and testing
if __name__ == "__main__":
    print("Benchmarking Tools for Voice Recognition")
    print("This module provides functions for:")
    print("1. Accuracy testing with known audio samples")
    print("2. Latency measurement and analysis")
    print("3. Performance monitoring")
    print("4. Comprehensive benchmarking reports")
    print("\nTo run actual tests, provide audio files and expected transcriptions:")

    # Example of how to use the benchmarking tools (commented out to avoid errors without actual files)
    """
    # Example test cases (you would provide actual audio files)
    test_cases = [
        {"audio_file": "test_move_forward.wav", "expected_text": "move forward"},
        {"audio_file": "test_turn_left.wav", "expected_text": "turn left"},
        {"audio_file": "test_stop.wav", "expected_text": "stop"},
    ]

    # Run accuracy test
    accuracy_results = run_simple_accuracy_test(test_cases)
    print_performance_report(accuracy_results, "Accuracy")

    # Run latency test
    if os.path.exists("test_sample.wav"):  # Only run if test file exists
        latency_results = run_simple_latency_test("test_sample.wav", iterations=3)
        print_performance_report(latency_results, "Latency")
    """

    print("\nFor actual benchmarking, create test cases with:")
    print("1. Audio files in common formats (WAV, MP3, etc.)")
    print("2. Expected transcriptions for accuracy comparison")
    print("3. Properly configured OpenAI API credentials")