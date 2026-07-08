# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TTS trace loader with codec token encoding using Qwen3 TTS tokenizer."""

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from pydantic import ValidationError

from aiperf.common.models import Turn
from aiperf.dataset.loader.base_loader import BaseLoader, LoaderProbeData
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.tts.tts_tokenizer import TTSTokenizer

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class TTSTraceDatasetLoader(BaseLoader):
    """TTS trace loader that encodes audio duration to codec tokens.

    Loads MooncakeTrace format with audio_duration_ms and converts it to
    codec token counts using the Qwen3 TTS tokenizer during replay.
    """

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        For TTS trace data, validate against MooncakeTrace model AND check for
        audio_duration_ms field to distinguish from regular mooncake traces.
        """
        if data is None:
            return False

        # Must have audio_duration_ms to be a TTS trace
        if "audio_duration_ms" not in data:
            return False

        # Still validate against MooncakeTrace model
        try:
            MooncakeTrace.model_validate(data)
            return True
        except ValidationError:
            return False

    def __init__(
        self,
        *,
        filename: Union[str, Path, None] = None,
        run: Optional["BenchmarkRun"] = None,
        **kwargs,
    ):
        super().__init__(filename=filename, run=run, **kwargs)
        self.tts_tokenizer = TTSTokenizer(run=run)

    def load_dataset(self) -> list:
        """Load dataset from file and convert to conversations.

        Returns:
            List of Conversation objects.
        """
        import json
        from aiperf.common.models import Conversation

        # Initialize TTS tokenizer
        self.tts_tokenizer.initialize()

        # Load and parse traces from file
        traces = []
        with open(self.filename, "r") as f:
            for line in f:
                if line.strip():
                    trace = MooncakeTrace.model_validate(json.loads(line))
                    traces.append(trace)

        # Convert audio_duration_ms to codec tokens
        total_traces = len(traces)
        converted_traces = 0

        for trace in traces:
            if trace.audio_duration_ms and not trace.output_length:
                # Use estimation for performance - accurate enough for max_tokens
                trace.output_length = self._estimate_codec_tokens_from_duration(
                    trace.audio_duration_ms
                )
                converted_traces += 1

        self.info(
            f"Converted {converted_traces}/{total_traces} traces from audio duration "
            f"to codec tokens using TTS tokenizer"
        )

        # Build conversations (one per trace for fixed schedule)
        conversations = []
        for i, trace in enumerate(traces):
            # Generate a simple prompt based on input_length
            prompt = "Hello world" * (trace.input_length // 11 + 1)
            prompt = prompt[:trace.input_length]

            # Build turn with raw_payload
            turn = self._build_turn(trace, prompt)

            # Create conversation
            conv = Conversation(
                session_id=f"tts_session_{i}",
                turns=[turn],
            )
            conversations.append(conv)

        self.tts_tokenizer.shutdown()
        return conversations

    def _estimate_codec_tokens_from_duration(self, duration_ms: float) -> int:
        """Estimation of codec tokens from duration.

        Based on Qwen3 TTS codec parameters (12Hz, ~75 tokens/sec).

        Args:
            duration_ms: Audio duration in milliseconds.

        Returns:
            Estimated codec token count.
        """
        # Qwen3 TTS: 12Hz codec = 12 tokens per second
        tokens_per_second = 12
        duration_sec = duration_ms / 1000
        return int(duration_sec * tokens_per_second)

    def _build_turn(self, trace: MooncakeTrace, prompt: str) -> Turn:
        """Build a Turn with raw_payload for TTS raw endpoint.

        Args:
            trace: MooncakeTrace with audio_duration_ms converted to output_length.
            prompt: Generated text prompt.

        Returns:
            Turn with raw_payload set for TTS request.
        """
        # Construct raw payload for TTS request
        # vLLM TTS API expects OpenAI-compatible format
        raw_payload = {
            "input": prompt,
            "voice": "aiden",  # Default voice, can be overridden via trace.extra
        }

        # Add max_tokens if specified (controls output length)
        if trace.output_length:
            raw_payload["max_tokens"] = trace.output_length

        # Add extra fields if present
        if trace.extra:
            raw_payload.update(trace.extra)

        return Turn(
            timestamp=trace.timestamp,
            delay=trace.delay,
            raw_payload=raw_payload,
        )
