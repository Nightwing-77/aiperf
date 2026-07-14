# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WER/CER evaluation configuration for TTS audio quality assessment.

Hosts `WERCERConfig` — the optional configuration block that enables
WER/CER evaluation of TTS audio responses using a standalone evaluation service.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig


class WERCERConfig(BaseConfig):
    """Configuration for WER/CER evaluation of TTS audio.

    When enabled, evaluates TTS audio responses by computing Word Error Rate (WER)
    and Character Error Rate (CER) using a standalone Whisper-based evaluation service.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable WER/CER evaluation for TTS audio responses. "
        "When True, the WERCERRecordProcessor will evaluate audio quality.",
    )

    service_url: str = Field(
        default="http://localhost:8001",
        description="URL of the standalone WER/CER evaluation service. "
        "The service should expose an /evaluate_json endpoint accepting "
        "base64-encoded audio data and reference text.",
    )

    language: str = Field(
        default="en",
        description="Language code for Whisper transcription (e.g., 'en', 'ja', 'zh'). "
        "Passed to the evaluation service for language-specific transcription.",
    )

    timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout in seconds for HTTP requests to the WER/CER evaluation service. "
        "Should be set based on expected audio length and Whisper model size.",
    )
