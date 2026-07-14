# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone WER/CER evaluation service using Whisper and jiwer.

This service provides HTTP endpoints for evaluating TTS audio quality
by computing Word Error Rate (WER) and Character Error Rate (CER).
Uses OpenAI Whisper for transcription and jiwer for WER/CER calculation.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import jiwer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False
    _log.warning("jiwer not installed. Install with: pip install jiwer")

try:
    import whisper
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False
    _log.warning("whisper not installed. Install with: pip install openai-whisper")


class WERCERRequest(BaseModel):
    """Request model for WER/CER evaluation."""

    reference_text: str = Field(description="Reference/transcript text to compare against")
    language: str = Field(default="en", description="Language code for Whisper (e.g., 'en', 'ja', 'zh')")


class WERCERResponse(BaseModel):
    """Response model for WER/CER evaluation."""

    wer: float = Field(description="Word Error Rate (0-100)")
    cer: float = Field(description="Character Error Rate (0-100)")
    transcription: str = Field(description="Transcribed text from Whisper")
    reference_text: str = Field(description="Reference text that was compared")
    language: str = Field(description="Language used for transcription")


class WERCERService:
    """WER/CER evaluation service using Whisper and jiwer."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """Initialize the WER/CER service.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run Whisper on (cpu, cuda)
        """
        if not _HAS_WHISPER:
            raise RuntimeError(
                "whisper not installed. Install with: pip install openai-whisper"
            )
        if not _HAS_JIWER:
            raise RuntimeError("jiwer not installed. Install with: pip install jiwer")

        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy load Whisper model."""
        if self._model is None:
            _log.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            self._model = whisper.load_model(self.model_size, device=self.device)
            _log.info("Whisper model loaded successfully")

    def _transcribe_audio(self, audio_path: str, language: str) -> str:
        """Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code for Whisper

        Returns:
            Transcribed text
        """
        self._load_model()

        # Transcribe with specified language
        result = self._model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            fp16=False if self.device == "cpu" else True,
        )
        return result["text"].strip()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for WER/CER calculation.

        Args:
            text: Input text

        Returns:
            Normalized text (lowercase, no punctuation)
        """
        import re

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate using jiwer.

        Args:
            reference: Reference text
            hypothesis: Hypothesis (transcribed) text

        Returns:
            WER as percentage (0-100)
        """
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 100.0
        if not hypothesis:
            return 100.0

        # Normalize texts
        ref_norm = self._normalize_text(reference)
        hyp_norm = self._normalize_text(hypothesis)

        # Calculate WER using jiwer
        wer = jiwer.wer(ref_norm, hyp_norm)
        return wer * 100  # Convert to percentage

    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate using jiwer.

        Args:
            reference: Reference text
            hypothesis: Hypothesis (transcribed) text

        Returns:
            CER as percentage (0-100)
        """
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 100.0
        if not hypothesis:
            return 100.0

        # Normalize texts and remove spaces for CER
        ref_norm = self._normalize_text(reference).replace(" ", "")
        hyp_norm = self._normalize_text(hypothesis).replace(" ", "")

        # Calculate CER using jiwer
        cer = jiwer.cer(ref_norm, hyp_norm)
        return cer * 100  # Convert to percentage

    def evaluate(
        self, audio_path: str, reference_text: str, language: str = "en"
    ) -> WERCERResponse:
        """Evaluate audio file against reference text.

        Args:
            audio_path: Path to audio file
            reference_text: Reference/transcript text
            language: Language code for Whisper

        Returns:
            WERCERResponse with WER, CER, and transcription
        """
        # Transcribe audio
        transcription = self._transcribe_audio(audio_path, language)

        # Calculate WER/CER
        wer = self._calculate_wer(reference_text, transcription)
        cer = self._calculate_cer(reference_text, transcription)

        return WERCERResponse(
            wer=wer,
            cer=cer,
            transcription=transcription,
            reference_text=reference_text,
            language=language,
        )


# Global service instance
_service: WERCERService | None = None


def get_service(model_size: str = "base", device: str = "cpu") -> WERCERService:
    """Get or create the global WER/CER service instance.

    Args:
        model_size: Whisper model size
        device: Device to run on

    Returns:
        WERCERService instance
    """
    global _service
    if _service is None:
        _service = WERCERService(model_size=model_size, device=device)
    return _service


# FastAPI app
app = FastAPI(
    title="WER/CER Evaluation Service",
    description="Standalone service for evaluating TTS audio quality using Whisper and jiwer",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "wer-cer-evaluation"}


@app.post("/evaluate", response_model=WERCERResponse)
async def evaluate_audio(
    audio_file: UploadFile = File(description="Audio file to evaluate (WAV, MP3, etc.)"),
    reference_text: str = Form(description="Reference/transcript text"),
    language: str = Form(default="en", description="Language code for Whisper"),
):
    """Evaluate audio file against reference text.

    Args:
        audio_file: Uploaded audio file
        reference_text: Reference text to compare against
        language: Language code for Whisper transcription

    Returns:
        WERCERResponse with WER, CER, and transcription
    """
    # Get model size and device from environment
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")

    service = get_service(model_size=model_size, device=device)

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(audio_file.filename or "audio").suffix
    ) as temp_file:
        temp_path = temp_file.name
        content = await audio_file.read()
        temp_file.write(content)

    try:
        # Evaluate audio
        result = service.evaluate(temp_path, reference_text, language)
        return result
    except Exception as e:
        _log.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()


@app.post("/evaluate_json")
async def evaluate_audio_json(request: dict):
    """Evaluate audio from JSON payload with base64-encoded audio.

    Expected JSON format:
    {
        "audio_data": "base64-encoded-audio-bytes",
        "audio_format": "wav",
        "reference_text": "reference text",
        "language": "en"
    }
    """
    import base64

    # Parse request
    audio_data_b64 = request.get("audio_data")
    audio_format = request.get("audio_format", "wav")
    reference_text = request.get("reference_text")
    language = request.get("language", "en")

    if not audio_data_b64 or not reference_text:
        raise HTTPException(
            status_code=400, detail="Missing audio_data or reference_text"
        )

    # Decode base64 audio
    try:
        audio_bytes = base64.b64decode(audio_data_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {e}")

    # Save to temp file
    suffix = f".{audio_format}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        temp_file.write(audio_bytes)

    try:
        # Get model size and device from environment
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        device = os.getenv("WHISPER_DEVICE", "cpu")

        service = get_service(model_size=model_size, device=device)

        # Evaluate audio
        result = service.evaluate(temp_path, reference_text, language)
        return orjson.loads(result.model_dump_json())
    except Exception as e:
        _log.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def main():
    """Run the WER/CER service standalone."""
    import uvicorn

    host = os.getenv("WER_CER_HOST", "0.0.0.0")
    port = int(os.getenv("WER_CER_PORT", "8001"))
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")

    _log.info(f"Starting WER/CER service on {host}:{port}")
    _log.info(f"Whisper model: {model_size}, device: {device}")

    # Pre-load model
    get_service(model_size=model_size, device=device)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
