# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TTS trace loader with codec token encoding using estimation."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from pydantic import ValidationError

from aiperf.common.models import Turn
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_loader import BaseFileLoader, LoaderProbeData
from aiperf.dataset.loader.models import MooncakeTrace

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class TTSTraceDatasetLoader(BaseFileLoader):
    """TTS trace loader that encodes audio duration to codec tokens.

    Loads MooncakeTrace format with audio_duration_ms and converts it to
    codec token counts using estimation (12 tokens/second for Qwen3 TTS).
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
        # Initialize prompt generator for synthetic text generation
        # This requires a tokenizer, so --use-server-token-count cannot be used
        from aiperf.common.tokenizer import Tokenizer

        tokenizer_config = self.run.cfg.tokenizer
        model_name = self.run.cfg.get_model_names()[0]
        tokenizer_name = tokenizer_config.get_tokenizer_name_for_model(model_name)

        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=tokenizer_config.trust_remote_code,
            revision=tokenizer_config.revision,
            resolve_alias=tokenizer_config.should_resolve_alias,
        )

        # Create prompt generator with default prompts conf
        from aiperf.config.dataset.content import (
            AudioConfig,
            ImageConfig,
            PrefixPromptConfig,
            PromptConfig,
            RankingsConfig,
        )

        prompts_config = PromptConfig(isl=128)
        self.prompt_generator = PromptGenerator(
            prompts=prompts_config,
            prefix_prompts=None,
            tokenizer=self.tokenizer,
        )

    def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        """Load dataset from file and return traces grouped by session.

        Returns:
            Dict mapping session IDs to lists of MooncakeTrace objects.
        """
        import json

        # Load and parse traces from file
        traces = []
        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trace = MooncakeTrace.model_validate(json.loads(line))
                    traces.append(trace)

        # Convert audio_duration_ms to codec tokens using estimation
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
            f"to codec tokens using estimation"
        )

        # Group traces by session (one per trace for fixed schedule)
        data = {}
        for i, trace in enumerate(traces):
            session_id = f"tts_session_{i}"
            data[session_id] = [trace]

        return data

    def convert_to_conversations(
        self, custom_data: dict[str, list[MooncakeTrace]]
    ) -> list:
        """Convert traces to conversations.

        Args:
            custom_data: Dict mapping session IDs to lists of MooncakeTrace objects.

        Returns:
            List of Conversation objects.
        """
        from aiperf.common.models import Conversation

        conversations = []
        for session_id, traces in custom_data.items():
            # Generate synthetic text using prompt generator
            trace = traces[0]
            prompt = self.prompt_generator.generate(mean=trace.input_length)

            # Build turn with raw_payload
            turn = self._build_turn(trace, prompt)

            # Create conversation
            conv = Conversation(
                session_id=session_id,
                turns=[turn],
            )
            conversations.append(conv)

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
