# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class RTFxMetric(BaseRecordMetric[float]):
    """Inverse Real-Time Factor (RTFx) for ASR and TTS benchmarks.

    Formula:
        RTFx = audio_duration_seconds / request_latency_seconds

    Higher is better; expressed as "Nx faster than real-time." This is the
    industry-standard ASR throughput metric (HuggingFace Open ASR Leaderboard
    requires it; NVIDIA Riva and NeMo use it as headline metric). For TTS,
    the same formula applies to the generated (output) audio duration instead
    of the input audio duration.

    Example:
        10s of input audio transcribed with 1s request latency -> RTFx = 10.0
        ("10x faster than real-time"). RTFx < 1.0 means the server is slower
        than real-time and not suitable for live transcription/synthesis.

    Uses the input audio duration (ASR: turn.audio_duration_seconds) if present,
    otherwise falls back to the generated output audio duration (TTS: decoded
    from the response's WAV header, see ``__audio_duration_seconds`` in
    ``ParsedResponse.metadata``). Extracted directly from the record rather than
    from ``AudioDurationMetric`` / ``OutputAudioDurationMetric``'s computed values,
    since the two are mutually exclusive per-record and ``required_metrics`` only
    supports strict AND ordering, not an OR fallback. Requires
    ``RequestLatencyMetric`` to be computed first. Non-audio requests yield no
    metric value.

    Raises:
        NoMetricValue: when no audio duration can be determined, or the
            measured request latency is non-positive.
    """

    tag = "rtfx"
    header = "Inverse Real-Time Factor (RTFx)"
    short_header = "RTFx"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    display_order = 850
    flags = MetricFlags.SUPPORTS_AUDIO_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {RequestLatencyMetric.tag}

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        audio_duration = self._input_audio_duration(record)
        if audio_duration is None:
            audio_duration = self._output_audio_duration(record)
        if audio_duration is None:
            raise NoMetricValue(
                "No input or output audio duration available for this record; "
                "RTFx applies to ASR/TTS (audio) requests only."
            )

        latency_seconds = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, MetricTimeUnit.SECONDS
        )
        if latency_seconds <= 0:
            raise NoMetricValue(
                f"Request latency is non-positive ({latency_seconds}s); "
                "RTFx undefined. Likely an upstream measurement bug."
            )

        return audio_duration / latency_seconds

    @staticmethod
    def _input_audio_duration(record: ParsedResponseRecord) -> float | None:
        turns = record.request.turns
        if not turns:
            return None
        duration = turns[0].audio_duration_seconds
        return duration if duration and duration > 0 else None

    @staticmethod
    def _output_audio_duration(record: ParsedResponseRecord) -> float | None:
        for response in record.content_responses:
            duration = (response.metadata or {}).get("__audio_duration_seconds")
            if duration is not None and duration > 0:
                return duration
        return None
