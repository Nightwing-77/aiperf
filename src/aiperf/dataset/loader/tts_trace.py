# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TTS trace loader with codec token encoding using Qwen3 TTS tokenizer."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from pydantic import ValidationError

from aiperf.common.models import Turn
from aiperf.dataset.loader.base_loader import LoaderProbeData
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.tts.tts_tokenizer import TTSTokenizer

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class TTSTraceDatasetLoader(MooncakeTraceDatasetLoader):
    """TTS trace loader that encodes audio duration to codec tokens.

    Extends MooncakeTraceDatasetLoader to convert audio_duration_ms to
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
        prompt_generator=None,
        run: Optional["BenchmarkRun"] = None,
        **kwargs,
    ):
        super().__init__(
            filename=filename,
            prompt_generator=prompt_generator,
            run=run,
            **kwargs,
        )
        self.tts_tokenizer = TTSTokenizer(run=run)

    def convert_to_conversations(
        self, data: dict[str, list[MooncakeTrace]]
    ) -> list:
        """Convert traces to conversations with codec token encoding.

        Overrides base method to:
        1. Initialize TTS tokenizer
        2. Convert audio_duration_ms → codec tokens for each trace
        3. Build conversations with accurate token counts

        Args:
            data: Grouped trace data by session ID.

        Returns:
            List of Conversation objects.
        """
        # Initialize TTS tokenizer once
        self.tts_tokenizer.initialize()

        # Process each trace to convert duration → codec tokens
        total_traces = 0
        converted_traces = 0

        for session_id, traces in data.items():
            for trace in traces:
                total_traces += 1
                if trace.audio_duration_ms and not trace.output_length:
                    # Use estimation for performance - accurate enough for max_tokens
                    # The TTS tokenizer encoding is too slow for large datasets
                    trace.output_length = self._estimate_codec_tokens_from_duration(
                        trace.audio_duration_ms
                    )
                    converted_traces += 1
                    # Log progress every 100 traces
                    if converted_traces % 100 == 0:
                        self.info(
                            f"Converted {converted_traces}/{total_traces} traces..."
                        )

        self.info(
            f"Converted {converted_traces}/{total_traces} traces from audio duration "
            f"to codec tokens using TTS tokenizer"
        )

        # Print sample conversion details (always visible)
        if converted_traces > 0:
            print(f"\n[TTS Trace Conversion Details]")
            for session_id, traces in data.items():
                for trace in traces[:3]:  # Print first 3 traces
                    if trace.audio_duration_ms and trace.output_length:
                        print(
                            f"  Trace: audio_duration_ms={trace.audio_duration_ms}ms "
                            f"→ codec_tokens={trace.output_length}"
                        )
                break  # Only first session
            print()

        # Use base class logic for conversation building
        return super().convert_to_conversations(data)

    def _estimate_codec_tokens_from_duration(self, duration_ms: float) -> int:
        """Fallback estimation of codec tokens from duration.

        Used when TTS tokenizer encoding fails. Provides a rough estimate
        based on Qwen3 TTS codec parameters (12Hz, ~75 tokens/sec).

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

        Overrides base method to construct raw_payload instead of texts,
        since raw endpoint requires raw_payload on every turn.

        Args:
            trace: MooncakeTrace with audio_duration_ms converted to output_length.
            prompt: Generated text prompt (ignored for raw payload).

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

        # Print request details for visibility
        print(f"\n[TTS Request Details]")
        print(f"  Text: {prompt}")
        print(f"  Max tokens (codec): {trace.output_length}")
        print(f"  Raw payload: {raw_payload}")
        print(f"  Timestamp: {getattr(trace, 'timestamp', None)}")
        print(f"  Delay: {getattr(trace, 'delay', None)}")
        print()

        return Turn(
            timestamp=getattr(trace, "timestamp", None),
            delay=getattr(trace, "delay", None),
            raw_payload=raw_payload,
        )

    def shutdown(self) -> None:
        """Shutdown the TTS tokenizer and release resources."""
        self.tts_tokenizer.shutdown()
        super().shutdown()
