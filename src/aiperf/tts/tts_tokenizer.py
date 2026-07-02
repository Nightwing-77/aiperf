# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 TTS tokenizer wrapper for audio codec token encoding."""

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from aiperf.common.exceptions import NotInitializedError, TokenizerError
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class TTSTokenizer(AIPerfLoggerMixin):
    """Qwen3 TTS tokenizer for converting audio duration to codec tokens.

    Wraps Qwen3TTSTokenizer to encode audio and count codec tokens.
    Used during TTS trace replay to convert audio_duration_ms → codec token count.
    """

    def __init__(self, run: "BenchmarkRun", **kwargs):
        super().__init__(**kwargs)
        self.run = run
        self._tokenizer = None
        self._device = "cuda" if self._cuda_available() else "cpu"
        self._sample_rate = 24000  # Qwen3 TTS default sample rate
        self._lock = asyncio.Lock()

    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def initialize(self, tokenizer_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz") -> None:
        """Initialize the Qwen3 TTS tokenizer.

        Args:
            tokenizer_name: HuggingFace model name for the tokenizer.
        """
        if self._tokenizer is not None:
            return

        async with self._lock:
            if self._tokenizer is not None:
                return

            self.info(f"Loading TTS tokenizer: {tokenizer_name} on {self._device}")

            try:
                from qwen_tts import Qwen3TTSTokenizer

                self._tokenizer = await asyncio.to_thread(
                    Qwen3TTSTokenizer.from_pretrained,
                    tokenizer_name,
                    device_map=self._device,
                )
                self.info(f"TTS tokenizer loaded successfully on {self._device}")
            except ImportError as e:
                raise TokenizerError(
                    "qwen-tts package not installed. Install with: pip install qwen-tts"
                ) from e
            except Exception as e:
                raise TokenizerError(f"Failed to load TTS tokenizer: {e}") from e

    async def duration_to_codec_tokens(self, duration_ms: float) -> int:
        """Convert audio duration to codec token count.

        Generates silent audio of the specified duration, encodes it using
        the TTS tokenizer, and returns the number of codec tokens.

        Args:
            duration_ms: Audio duration in milliseconds.

        Returns:
            Number of codec tokens for the specified duration.
        """
        if self._tokenizer is None:
            raise NotInitializedError(
                "TTSTokenizer not initialized. Call initialize() first."
            )

        self.debug(f"Converting audio duration: {duration_ms}ms to codec tokens")

        # Generate silent audio of specified duration
        audio_samples = self._generate_silent_audio(duration_ms)

        # Save to temporary file for encoding
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Write silent audio to temp file
            sf.write(str(temp_path), audio_samples, self._sample_rate)

            # Encode audio to get codec tokens
            encoder_output = await asyncio.to_thread(
                self._tokenizer.encode, str(temp_path)
            )

            # Count tokens in audio_codes
            # encoder_output.audio_codes is a tensor of shape [batch, channels, tokens]
            audio_codes = encoder_output.audio_codes
            token_count = audio_codes.shape[-1]  # Last dimension is token sequence length

            return token_count
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def _generate_silent_audio(self, duration_ms: float) -> np.ndarray:
        """Generate silent audio of specified duration.

        Args:
            duration_ms: Audio duration in milliseconds.

        Returns:
            NumPy array of silent audio samples.
        """
        duration_sec = duration_ms / 1000
        num_samples = int(duration_sec * self._sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

    async def shutdown(self) -> None:
        """Shutdown the tokenizer and release resources."""
        if self._tokenizer is not None:
            self.info("Shutting down TTS tokenizer")
            self._tokenizer = None
