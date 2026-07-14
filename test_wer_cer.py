#!/usr/bin/env python3
"""Test script for WER/CER evaluation service."""

import sys
import os
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import tempfile
import numpy as np
import soundfile as sf
import httpx
from aiperf.tts.wer_cer_service import WERCERService


def get_test_audio() -> str:
    """Get the test audio file path.

    Returns:
        Path to the test audio file
    """
    # Use the provided test audio file
    audio_path = os.path.join(os.path.dirname(__file__), "hello how are you.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Test audio file not found: {audio_path}")
    return audio_path


def test_wer_cer_service():
    """Test the WER/CER service with a simple example."""
    print("Testing WER/CER Service...")

    # Check for GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        device = "cuda" if has_gpu else "cpu"
        print(f"GPU available: {has_gpu}, using device: {device}")
    except ImportError:
        device = "cpu"
        print("PyTorch not available, using CPU")

    # Create service instance
    try:
        service = WERCERService(model_size="tiny", device=device)
        print("✓ WERCERService created successfully")
    except Exception as e:
        print(f"✗ Failed to create WERCERService: {e}")
        print("  (This is expected if Whisper is not installed)")
        print("  Install with: pip install openai-whisper")
        return False

    # Get test audio
    try:
        audio_path = get_test_audio()
        print(f"✓ Test audio loaded: {audio_path}")
    except Exception as e:
        print(f"✗ Failed to load test audio: {e}")
        return False

    # Test evaluation
    try:
        reference_text = "hello how are you"
        result = service.evaluate(audio_path, reference_text, language="en")

        print(f"✓ Evaluation successful:")
        print(f"  - WER: {result.wer:.2f}%")
        print(f"  - CER: {result.cer:.2f}%")
        print(f"  - Transcription: '{result.transcription}'")
        print(f"  - Reference: '{result.reference_text}'")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n✓ Direct service test passed!")
    return True


def test_server_via_http():
    """Test the WER/CER service via HTTP server."""
    print("\nTesting WER/CER Service via HTTP...")

    # Start server in background
    import subprocess
    import time

    server_process = None
    try:
        # Start the server
        print("Starting WER/CER server...")
        server_process = subprocess.Popen(
            [sys.executable, "-m", "aiperf.tts.wer_cer_service"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        time.sleep(5)

        # Test health endpoint
        print("Testing health endpoint...")
        try:
            response = httpx.get("http://localhost:8001/health", timeout=5.0)
            response.raise_for_status()
            print(f"✓ Health check: {response.json()}")
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False

        # Test evaluation via JSON endpoint
        print("Testing evaluation via JSON endpoint...")
        try:
            audio_path = get_test_audio()
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            payload = {
                "audio_data": audio_b64,
                "audio_format": "wav",
                "reference_text": "hello how are you",
                "language": "en",
            }

            response = httpx.post(
                "http://localhost:8001/evaluate_json",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

            print(f"✓ Evaluation successful:")
            print(f"  - WER: {result['wer']:.2f}%")
            print(f"  - CER: {result['cer']:.2f}%")
            print(f"  - Transcription: '{result['transcription']}'")
            print(f"  - Reference: '{result['reference_text']}'")
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("✓ HTTP server test passed!")
        return True

    except Exception as e:
        print(f"✗ Server test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up server process
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            server_process.wait(timeout=5)


def test_record_processor():
    """Test the WER/CER record processor."""
    print("\nTesting WER/CER Record Processor...")

    try:
        from aiperf.tts.wer_cer_record_processor import WERCERRecordProcessor
        print("✓ WERCERRecordProcessor imported successfully")
    except Exception as e:
        print(f"✗ Failed to import WERCERRecordProcessor: {e}")
        return False

    print("✓ Record processor test skipped (requires full AIPerf config)")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("WER/CER Evaluation Service Test")
    print("=" * 60)
    print()

    # Test via HTTP server first (to avoid model loading conflicts)
    server_ok = test_server_via_http()

    # Test service directly
    service_ok = test_wer_cer_service()

    # Test record processor
    processor_ok = test_record_processor()

    # Clean up test audio file
    try:
        audio_path = get_test_audio()
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"✓ Cleaned up test audio file")
    except Exception as e:
        print(f"⚠ Failed to clean up test audio: {e}")

    print()
    print("=" * 60)
    if service_ok and server_ok and processor_ok:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
