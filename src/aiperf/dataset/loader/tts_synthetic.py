# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic TTS workload generator for automated deployment testing.

Generates N random TTS requests with random text lengths, audio durations,
and timestamps within a configurable duration window. Produces trace-like
data that is replayed via the existing FixedScheduleStrategy.

Unlike ``TTSTraceDatasetLoader`` which reads a trace file, this loader
generates synthetic traces on-the-fly from inline config records. The text
content is sampled from the corpus via ``PromptGenerator`` (same as TTS
trace replay), and audio duration is converted to codec tokens using the
same 12 tokens/second estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common import random_generator as rng
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_loader import BaseFileLoader, LoaderProbeData
from aiperf.dataset.loader.models import MooncakeTrace

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_TOKENS_PER_SECOND = 12


class TTSSyntheticLoader(BaseFileLoader):
    """Synthetic TTS workload generator for automated deployment testing.

    Generates random TTS requests with:
    - Random text length (input tokens) from a configurable range
    - Random audio duration (output codec tokens) from a configurable range
    - Random timestamps uniformly distributed across a duration window

    Uses inline records for configuration (no file needed). The first
    inline record contains the generation parameters:
    - duration_sec: Length of the time window in seconds
    - num_requests: Number of TTS requests to generate
    - input_length_min / input_length_max: Range for text token count
    - audio_duration_min_ms / audio_duration_max_ms: Range for audio output
    - voice: TTS voice to use (default: "aiden")
    """

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        """This loader is not auto-detectable; use explicit format: tts_synthetic."""
        return False

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        inline_records: list[dict[str, Any]] | None = None,
        run: BenchmarkRun | None = None,
        **kwargs,
    ):
        super().__init__(
            filename=filename, inline_records=inline_records, run=run, **kwargs
        )

        config = self._parse_config()
        self._duration_sec: float = config["duration_sec"]
        self._num_requests: int = config["num_requests"]
        self._input_length_min: int = config["input_length_min"]
        self._input_length_max: int = config["input_length_max"]
        self._audio_duration_min_ms: float = config["audio_duration_min_ms"]
        self._audio_duration_max_ms: float = config["audio_duration_max_ms"]
        self._voice: str = config.get("voice", "aiden")

        self._timestamp_rng = rng.derive("tts_synthetic.timestamp")
        self._input_length_rng = rng.derive("tts_synthetic.input_length")
        self._audio_duration_rng = rng.derive("tts_synthetic.audio_duration")

        from aiperf.common.tokenizer import Tokenizer
        from aiperf.config.dataset.content import PromptConfig

        tokenizer_config = self.run.cfg.tokenizer
        model_name = self.run.cfg.get_model_names()[0]
        tokenizer_name = tokenizer_config.get_tokenizer_name_for_model(model_name)

        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=tokenizer_config.trust_remote_code,
            revision=tokenizer_config.revision,
            resolve_alias=tokenizer_config.should_resolve_alias,
        )

        prompts_config = PromptConfig(isl=self._input_length_max)
        self.prompt_generator = PromptGenerator(
            prompts=prompts_config,
            prefix_prompts=None,
            tokenizer=self.tokenizer,
        )

    def _parse_config(self) -> dict[str, Any]:
        """Parse generation config from inline records."""
        if self.inline_records is None or not self.inline_records:
            raise ValueError(
                "TTSSyntheticLoader requires inline_records with generation config. "
                "Provide a record with: duration_sec, num_requests, "
                "input_length_min, input_length_max, "
                "audio_duration_min_ms, audio_duration_max_ms."
            )

        config = self.inline_records[0]

        required = [
            "duration_sec",
            "num_requests",
            "input_length_min",
            "input_length_max",
            "audio_duration_min_ms",
            "audio_duration_max_ms",
        ]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(
                f"TTSSyntheticLoader config missing required fields: {missing}"
            )

        if config["input_length_min"] > config["input_length_max"]:
            raise ValueError(
                "input_length_min must be <= input_length_max, "
                f"got {config['input_length_min']} > {config['input_length_max']}"
            )
        if config["audio_duration_min_ms"] > config["audio_duration_max_ms"]:
            raise ValueError(
                "audio_duration_min_ms must be <= audio_duration_max_ms, "
                f"got {config['audio_duration_min_ms']} > {config['audio_duration_max_ms']}"
            )

        return config

    def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        """Generate synthetic TTS traces with random parameters.

        Returns:
            Dict mapping session IDs to lists of MooncakeTrace objects.
        """
        duration_ms = self._duration_sec * 1000
        traces: list[MooncakeTrace] = []

        for _ in range(self._num_requests):
            input_length = self._input_length_rng.randint(
                self._input_length_min, self._input_length_max
            )
            audio_duration_ms = self._audio_duration_rng.uniform(
                self._audio_duration_min_ms, self._audio_duration_max_ms
            )
            timestamp = self._timestamp_rng.uniform(0, duration_ms)

            output_length = self._estimate_codec_tokens_from_duration(audio_duration_ms)

            trace = MooncakeTrace(
                input_length=input_length,
                output_length=output_length,
                audio_duration_ms=audio_duration_ms,
                timestamp=timestamp,
            )
            traces.append(trace)

        self.info(
            f"Generated {len(traces)} synthetic TTS traces: "
            f"duration={self._duration_sec}s, "
            f"input_length=[{self._input_length_min}, {self._input_length_max}], "
            f"audio_duration=[{self._audio_duration_min_ms}, {self._audio_duration_max_ms}]ms"
        )

        data: dict[str, list[MooncakeTrace]] = {}
        for i, trace in enumerate(traces):
            session_id = f"tts_synthetic_session_{i}"
            data[session_id] = [trace]

        return data

    def convert_to_conversations(
        self, custom_data: dict[str, list[MooncakeTrace]]
    ) -> list[Conversation]:
        """Convert generated traces to conversations with synthetic text.

        Args:
            custom_data: Dict mapping session IDs to lists of MooncakeTrace objects.

        Returns:
            List of Conversation objects with raw_payload for TTS requests.
        """
        conversations = []
        for session_id, traces in custom_data.items():
            trace = traces[0]
            input_length = trace.input_length if trace.input_length is not None else 128
            prompt = self.prompt_generator.generate(mean=input_length, stddev=0)
            turn = self._build_turn(trace, prompt)
            conv = Conversation(
                session_id=session_id,
                turns=[turn],
            )
            conversations.append(conv)

        return conversations

    def _estimate_codec_tokens_from_duration(self, duration_ms: float) -> int:
        """Estimate codec tokens from audio duration.

        Based on Qwen3 TTS codec parameters (12Hz, 12 tokens/second).

        Args:
            duration_ms: Audio duration in milliseconds.

        Returns:
            Estimated codec token count.
        """
        duration_sec = duration_ms / 1000
        return int(duration_sec * _TOKENS_PER_SECOND)

    def _build_turn(self, trace: MooncakeTrace, prompt: str) -> Turn:
        """Build a Turn with raw_payload for TTS raw endpoint.

        Args:
            trace: MooncakeTrace with audio_duration_ms converted to output_length.
            prompt: Generated text prompt.

        Returns:
            Turn with raw_payload set for TTS request.
        """
        raw_payload: dict[str, Any] = {
            "input": prompt,
            "voice": self._voice,
        }

        # vLLM-Omni's /v1/audio/speech accepts "max_new_tokens", not the OpenAI-chat
        # "max_tokens" field; sending the latter is silently dropped and the server
        # falls back to its default max_new_tokens (2048), massively overrunning
        # short TTS requests.
        if trace.output_length:
            raw_payload["max_new_tokens"] = trace.output_length

        if trace.extra:
            raw_payload.update(trace.extra)

        return Turn(
            timestamp=trace.timestamp if trace.timestamp is not None else 0,
            delay=trace.delay if trace.delay is not None else 0,
            raw_payload=raw_payload,
        )
