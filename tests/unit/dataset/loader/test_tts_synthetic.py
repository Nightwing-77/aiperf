# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TTSSyntheticLoader."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aiperf.dataset.loader.models import MooncakeTrace


def _make_tts_synthetic_config(
    *,
    duration_sec: float = 60,
    num_requests: int = 10,
    input_length_min: int = 10,
    input_length_max: int = 128,
    audio_duration_min_ms: float = 500,
    audio_duration_max_ms: float = 5000,
    voice: str = "aiden",
) -> list[dict]:
    """Build inline records config for TTSSyntheticLoader."""
    return [
        {
            "duration_sec": duration_sec,
            "num_requests": num_requests,
            "input_length_min": input_length_min,
            "input_length_max": input_length_max,
            "audio_duration_min_ms": audio_duration_min_ms,
            "audio_duration_max_ms": audio_duration_max_ms,
            "voice": voice,
        }
    ]


def _make_loader(
    config: list[dict] | None = None,
):
    """Create a TTSSyntheticLoader with FakeTokenizer patched in."""
    from aiperf.common import random_generator as rng
    from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader
    from tests.harness.fake_tokenizer import FakeTokenizer

    if config is None:
        config = _make_tts_synthetic_config()

    rng.reset()
    rng.init(42)

    with patch(
        "aiperf.common.tokenizer.Tokenizer.from_pretrained",
        return_value=FakeTokenizer(),
    ):
        loader = TTSSyntheticLoader(
            inline_records=config,
            run=None,
        )
    return loader


class TestTTSSyntheticLoaderCanLoad:
    """Tests for can_load classmethod."""

    def test_can_load_returns_false_for_none_data(self):
        from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader

        assert TTSSyntheticLoader.can_load(data=None) is False

    def test_can_load_returns_false_for_dict_data(self):
        from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader

        assert (
            TTSSyntheticLoader.can_load(data={"duration_sec": 60})
            is False
        )


class TestTTSSyntheticLoaderConfig:
    """Tests for config parsing and validation."""

    def test_missing_inline_records_raises(self):
        from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader

        with pytest.raises(ValueError, match="inline_records"):
            with patch(
                "aiperf.common.tokenizer.Tokenizer.from_pretrained",
                return_value=None,
            ):
                TTSSyntheticLoader(
                    inline_records=None,
                    run=None,
                )

    def test_empty_inline_records_raises(self):
        from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader

        with pytest.raises(ValueError, match="inline_records"):
            with patch(
                "aiperf.common.tokenizer.Tokenizer.from_pretrained",
                return_value=None,
            ):
                TTSSyntheticLoader(
                    inline_records=[],
                    run=None,
                )

    def test_missing_required_fields_raises(self):
        from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader

        with pytest.raises(ValueError, match="missing required fields"):
            with patch(
                "aiperf.common.tokenizer.Tokenizer.from_pretrained",
                return_value=None,
            ):
                TTSSyntheticLoader(
                    inline_records=[{"duration_sec": 60}],
                    run=None,
                )

    def test_input_length_min_gt_max_raises(self):
        from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader

        config = _make_tts_synthetic_config(
            input_length_min=200, input_length_max=100
        )
        with pytest.raises(ValueError, match="input_length_min"):
            with patch(
                "aiperf.common.tokenizer.Tokenizer.from_pretrained",
                return_value=None,
            ):
                TTSSyntheticLoader(
                    inline_records=config,
                    run=None,
                )

    def test_audio_duration_min_gt_max_raises(self):
        from aiperf.dataset.loader.tts_synthetic import TTSSyntheticLoader

        config = _make_tts_synthetic_config(
            audio_duration_min_ms=10000, audio_duration_max_ms=1000
        )
        with pytest.raises(ValueError, match="audio_duration_min_ms"):
            with patch(
                "aiperf.common.tokenizer.Tokenizer.from_pretrained",
                return_value=None,
            ):
                TTSSyntheticLoader(
                    inline_records=config,
                    run=None,
                )


class TestTTSSyntheticLoaderGeneration:
    """Tests for synthetic trace generation."""

    def test_load_dataset_generates_correct_count(self):
        loader = _make_loader(
            _make_tts_synthetic_config(num_requests=20)
        )
        data = loader.load_dataset()

        assert len(data) == 20
        for session_id, traces in data.items():
            assert session_id.startswith("tts_synthetic_session_")
            assert len(traces) == 1
            assert isinstance(traces[0], MooncakeTrace)

    def test_timestamps_within_duration_window(self):
        loader = _make_loader(
            _make_tts_synthetic_config(duration_sec=30, num_requests=15)
        )
        data = loader.load_dataset()

        max_ts = 30 * 1000
        for traces in data.values():
            trace = traces[0]
            assert trace.timestamp is not None
            assert 0 <= trace.timestamp <= max_ts

    def test_input_lengths_within_range(self):
        loader = _make_loader(
            _make_tts_synthetic_config(
                input_length_min=10, input_length_max=50, num_requests=20
            )
        )
        data = loader.load_dataset()

        for traces in data.values():
            trace = traces[0]
            assert trace.input_length is not None
            assert 10 <= trace.input_length <= 50

    def test_audio_durations_within_range(self):
        loader = _make_loader(
            _make_tts_synthetic_config(
                audio_duration_min_ms=1000,
                audio_duration_max_ms=5000,
                num_requests=20,
            )
        )
        data = loader.load_dataset()

        for traces in data.values():
            trace = traces[0]
            assert trace.audio_duration_ms is not None
            assert 1000 <= trace.audio_duration_ms <= 5000

    def test_codec_tokens_estimated_from_duration(self):
        loader = _make_loader(
            _make_tts_synthetic_config(
                audio_duration_min_ms=2000,
                audio_duration_max_ms=2000,
                num_requests=1,
            )
        )
        data = loader.load_dataset()

        traces = list(data.values())[0]
        trace = traces[0]
        assert trace.output_length is not None
        assert trace.output_length == int(2.0 * 12)

    def test_convert_to_conversations_produces_raw_payload(self):
        loader = _make_loader(
            _make_tts_synthetic_config(num_requests=5)
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 5
        for conv in conversations:
            assert len(conv.turns) == 1
            turn = conv.turns[0]
            assert turn.raw_payload is not None
            assert "input" in turn.raw_payload
            assert turn.raw_payload["voice"] == "Vivian"
            assert "max_new_tokens" in turn.raw_payload
            assert turn.timestamp is not None

    def test_voice_override(self):
        loader = _make_loader(
            _make_tts_synthetic_config(voice="jessica", num_requests=3)
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            assert conv.turns[0].raw_payload["voice"] == "jessica"

    def test_reproducible_with_same_seed(self):
        from aiperf.common import random_generator as rng

        rng.reset()
        rng.init(42)
        loader1 = _make_loader(
            _make_tts_synthetic_config(num_requests=10)
        )
        data1 = loader1.load_dataset()

        rng.reset()
        rng.init(42)
        loader2 = _make_loader(
            _make_tts_synthetic_config(num_requests=10)
        )
        data2 = loader2.load_dataset()

        ts1 = [t[0].timestamp for t in data1.values()]
        ts2 = [t[0].timestamp for t in data2.values()]
        assert ts1 == ts2
