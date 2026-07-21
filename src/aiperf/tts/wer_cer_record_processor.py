# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WER/CER record processor for TTS audio evaluation.

This processor evaluates TTS audio responses by computing Word Error Rate (WER)
and Character Error Rate (CER) using a standalone WER/CER evaluation service.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from pydantic import Field

from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_log = logging.getLogger(__name__)


class WERCERRecordProcessor(AIPerfLifecycleMixin):
    """Record processor for WER/CER evaluation of TTS audio.

    Extracts audio from TTS response payloads, sends it to the WER/CER evaluation
    service, and adds audio.wer and audio.cer metrics to records.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the WER/CER record processor.

        Args:
            run: Benchmark run configuration
            service_id: Service identifier
            **kwargs: Additional arguments

        Raises:
            PostProcessorDisabled: If WER/CER evaluation is not configured
        """
        # Check if WER/CER evaluation is enabled
        wer_cer_config = getattr(run.cfg, "wer_cer", None)
        if wer_cer_config is None or not wer_cer_config.enabled:
            raise PostProcessorDisabled(
                "WER/CER record processor is disabled: wer_cer mode is not enabled"
            )

        super().__init__(service_id=service_id, **kwargs)
        self.run = run
        self.service_url = wer_cer_config.service_url
        self.language = wer_cer_config.language
        self.timeout = wer_cer_config.timeout

        # Initialize HTTP client
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict:
        """Process a TTS audio response record.

        Extracts audio from the response, evaluates it against reference text,
        and adds WER/CER metrics to the record.

        Args:
            record: Parsed response record containing TTS audio
            metadata: Record metadata

        Returns:
            MetricRecordDict with WER/CER metrics
        """
        record_metrics = MetricRecordDict()

        # Extract audio and reference text from record
        audio_data, reference_text = self._extract_audio_and_reference(record)

        if audio_data is None or reference_text is None:
            self.debug(
                lambda: f"No audio data or reference text found in record, skipping WER/CER evaluation"
            )
            # Set default max error values
            record_metrics["audio.wer"] = 100.0
            record_metrics["audio.cer"] = 100.0
            return record_metrics

        try:
            # Evaluate audio using WER/CER service
            wer, cer = await self._evaluate_audio(audio_data, reference_text)

            record_metrics["audio.wer"] = wer
            record_metrics["audio.cer"] = cer

            self.info(
                lambda: f"WER/CER evaluation complete: WER={wer:.2f}%, CER={cer:.2f}%"
            )
        except Exception as e:
            self.warning(
                lambda: f"WER/CER evaluation failed: {e}, setting max error values"
            )
            record_metrics["audio.wer"] = 100.0
            record_metrics["audio.cer"] = 100.0

        return record_metrics

    def _extract_audio_and_reference(
        self, record: ParsedResponseRecord
    ) -> tuple[bytes | None, str | None]:
        """Extract audio data and reference text from response record.

        Args:
            record: Parsed response record

        Returns:
            Tuple of (audio_data, reference_text) or (None, None) if not found
        """
        audio_data = None
        reference_text = None

        # Try to extract audio from response content. NOTE: raw TTS audio bytes
        # are discarded by the transport before crossing the ZMQ boundary (only
        # __audio_duration_seconds survives, for RTFx); this path only fires for
        # endpoints that embed audio directly in a structured JSON payload.
        for resp in record.content_responses:
            if not resp.data:
                continue
            if hasattr(resp, "raw_response") and resp.raw_response:
                raw_resp = resp.raw_response
                if isinstance(raw_resp, dict):
                    for field in ["audio", "audio_data", "audio_bytes", "audio_content"]:
                        if field in raw_resp:
                            audio_data = raw_resp[field]
                            if isinstance(audio_data, str):
                                import base64

                                audio_data = base64.b64decode(audio_data)
                            break
                    break

        # Reference text: the TTS input prompt lives in the last request turn.
        for turn in reversed(record.request.turns):
            if turn.raw_payload:
                for field in ["input", "text", "prompt"]:
                    if field in turn.raw_payload:
                        reference_text = turn.raw_payload[field]
                        break
            if reference_text is None and turn.texts:
                text = turn.texts[0]
                if text.contents:
                    reference_text = text.contents[0]
            if reference_text is None and turn.raw_messages:
                for msg in reversed(turn.raw_messages):
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if content:
                        reference_text = content
                        break
            if reference_text is not None:
                break

        return audio_data, reference_text

    async def _evaluate_audio(
        self, audio_data: bytes, reference_text: str
    ) -> tuple[float, float]:
        """Evaluate audio using the WER/CER service.

        Args:
            audio_data: Audio data as bytes
            reference_text: Reference text to compare against

        Returns:
            Tuple of (wer, cer) as percentages

        Raises:
            Exception: If evaluation fails
        """
        import base64

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)

        try:
            # Prepare request payload
            import base64

            audio_b64 = base64.b64encode(audio_data).decode("utf-8")

            payload = {
                "audio_data": audio_b64,
                "audio_format": "wav",
                "reference_text": reference_text,
                "language": self.language,
            }

            # Send request to WER/CER service
            response = await self._client.post(
                f"{self.service_url}/evaluate_json",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            response.raise_for_status()
            result = response.json()

            wer = result.get("wer", 100.0)
            cer = result.get("cer", 100.0)

            return wer, cer
        finally:
            # Clean up temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    async def shutdown(self) -> None:
        """Shutdown the processor and close HTTP client."""
        await self._client.aclose()
        await super().shutdown()
