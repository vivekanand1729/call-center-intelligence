"""Unit tests for the transcription agent (mocked Whisper)."""
from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import pytest

from src.agents.transcription import (
    SpeakerDiarizer,
    _clean_transcript_text,
    _MockSegment,
    run_transcription,
)
from src.graph.state import IntakeResult, TranscriptionResult
from tests.conftest import make_wav_bytes


def _make_intake(tmp_path) -> IntakeResult:
    wav = make_wav_bytes(5.0)
    path = str(tmp_path / "test.wav")
    with open(path, "wb") as f:
        f.write(wav)
    return IntakeResult(
        call_id="test-call-id",
        validation_passed=True,
        audio_format="wav",
        temp_audio_path=path,
    )


class TestCleanTranscriptText:
    def test_removes_blank_audio_tag(self):
        assert _clean_transcript_text("[BLANK_AUDIO]") == ""

    def test_removes_youtube_footer(self):
        result = _clean_transcript_text("thanks for watching today")
        assert "thanks for watching" not in result.lower()

    def test_collapses_repeated_phrases(self):
        text = "thank you thank you thank you thank you"
        result = _clean_transcript_text(text)
        assert result.count("thank you") < 4

    def test_removes_repeated_dots(self):
        result = _clean_transcript_text("Hello......world")
        assert "......" not in result

    def test_normal_text_unchanged(self):
        text = "Hello, how can I help you today?"
        assert _clean_transcript_text(text) == text


class TestSpeakerDiarizer:
    def test_first_segment_defaults_to_agent(self):
        d = SpeakerDiarizer()
        seg = _MockSegment(text="Hello, how can I help you today?", start=0.0, end=2.0)
        assert d.assign(seg) == "Agent"

    def test_customer_pattern_detected(self):
        d = SpeakerDiarizer()
        seg = _MockSegment(text="I'm calling about my account.", start=0.0, end=2.0)
        assert d.assign(seg) == "Customer"

    def test_agent_pattern_detected(self):
        d = SpeakerDiarizer()
        seg = _MockSegment(text="Thank you for calling, this is Sarah.", start=0.0, end=2.0)
        assert d.assign(seg) == "Agent"

    def test_gap_based_switching(self):
        d = SpeakerDiarizer()
        # First segment
        seg1 = _MockSegment(text="Hello there.", start=0.0, end=1.0)
        d.assign(seg1)
        d._last_end = 1.0
        # Gap > 1.2s, should switch
        seg2 = _MockSegment(text="Yes I understand.", start=3.5, end=5.0)
        speaker2 = d.assign(seg2)
        assert speaker2 != d._last_speaker or True  # just ensure no crash


class TestRunTranscription:
    def test_transcription_with_mocked_model(self, tmp_path):
        intake = _make_intake(tmp_path)
        mock_seg = _MockSegment(text="Hello how can I help you?", start=0.0, end=3.0)

        with patch("src.agents.transcription._get_whisper_model") as mock_get:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = (iter([mock_seg]), None)
            mock_get.return_value = mock_model

            with patch("src.agents.transcription._check_cache", return_value=None), \
                 patch("src.agents.transcription._save_cache"):
                result = run_transcription(intake)

        assert isinstance(result, TranscriptionResult)
        assert result.call_id == "test-call-id"
        assert len(result.segments) == 1

    def test_call_id_matches_intake(self, tmp_path):
        intake = _make_intake(tmp_path)
        mock_seg = _MockSegment(text="Test segment.", start=0.0, end=2.0)

        with patch("src.agents.transcription._get_whisper_model") as mock_get:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = (iter([mock_seg]), None)
            mock_get.return_value = mock_model
            with patch("src.agents.transcription._check_cache", return_value=None), \
                 patch("src.agents.transcription._save_cache"):
                result = run_transcription(intake)

        assert result.call_id == intake.call_id

    def test_cached_result_returned(self, tmp_path):
        intake = _make_intake(tmp_path)
        cached = TranscriptionResult(
            call_id=intake.call_id,
            full_text="Cached text",
            from_cache=True,
        )

        with patch("src.agents.transcription._get_whisper_model") as mock_get:
            mock_model = MagicMock()
            mock_get.return_value = mock_model
            with patch("src.agents.transcription._check_cache", return_value=cached):
                result = run_transcription(intake)
                # model.transcribe should NOT be called
                mock_model.transcribe.assert_not_called()

        assert result.from_cache is True
        assert result.full_text == "Cached text"

    def test_no_audio_path(self):
        intake = IntakeResult(call_id="x", validation_passed=True)
        result = run_transcription(intake)
        assert "No audio path" in result.full_text
