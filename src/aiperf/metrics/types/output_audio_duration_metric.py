# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricConsoleGroup, MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class OutputAudioDurationMetric(BaseRecordMetric[float]):
    """Per-request generated audio duration in seconds, for TTS benchmarks.

    Complements ``AudioDurationMetric`` (input audio duration, for ASR) with the
    output-side equivalent: the duration of audio the model *generated*. Decoded
    from the WAV header by the transport layer before the raw bytes are discarded
    (see ``aiperf.transports.aiohttp_client._decode_audio_duration_seconds``), and
    propagated through ``ParsedResponse.metadata['__audio_duration_seconds']``.

    Raises:
        NoMetricValue: when no content response carries a decoded audio duration
            (non-audio responses, or non-WAV audio whose duration could not be
            determined from the header).
    """

    tag = "output_audio_duration"
    header = "Output Audio Duration"
    short_header = "Out Audio Dur"
    unit = MetricTimeUnit.SECONDS
    display_order = 871
    flags = MetricFlags.SUPPORTS_AUDIO_ONLY
    console_group = MetricConsoleGroup.NONE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        for response in record.content_responses:
            duration = (response.metadata or {}).get("__audio_duration_seconds")
            if duration is not None and duration > 0:
                return duration

        raise NoMetricValue(
            "No response carried a decoded output audio duration; "
            "output_audio_duration applies to TTS (audio-generating) requests only."
        )
