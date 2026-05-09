"""Unit tests for audio utilities."""
from __future__ import annotations

import io
import wave

import pytest

from src.utils.audio import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_FORMATS,
    detect_audio_format,
    extract_audio_properties,
    validate_audio_file,
)
from tests.conftest import make_wav_bytes


class TestDetectAudioFormat:
    def test_wav_detected(self):
        data = make_wav_bytes(1.0)
        assert detect_audio_format(data) == "wav"

    def test_mp3_id3_detected(self):
        data = b"ID3" + b"\x00" * 20
        assert detect_audio_format(data) == "mp3"

    def test_mp3_sync_bits_detected(self):
        data = b"\xff\xfb\x90\x00" + b"\x00" * 100
        assert detect_audio_format(data) == "mp3"

    def test_flac_detected(self):
        data = b"fLaC" + b"\x00" * 20
        assert detect_audio_format(data) == "flac"

    def test_m4a_detected(self):
        data = b"\x00\x00\x00\x20" + b"ftyp" + b"\x00" * 20
        assert detect_audio_format(data) == "m4a"

    def test_unknown_format(self):
        data = b"\x00" * 20
        assert detect_audio_format(data) == "unknown"

    def test_ogg_unknown(self):
        data = b"OggS" + b"\x00" * 20
        assert detect_audio_format(data) == "unknown"

    def test_short_data(self):
        data = b"\xff"
        assert detect_audio_format(data) == "unknown"


class TestValidateAudioFile:
    def test_valid_wav(self, wav_bytes_5s):
        result = validate_audio_file(wav_bytes_5s, "call.wav")
        assert result.is_valid is True
        assert result.error is None

    def test_empty_file_rejected(self):
        result = validate_audio_file(b"", "empty.wav")
        assert result.is_valid is False
        assert result.error is not None

    def test_unsupported_format_rejected(self):
        data = b"OggS" + b"\x00" * 20
        result = validate_audio_file(data, "audio.ogg")
        assert result.is_valid is False
        assert "Unsupported" in result.error

    def test_file_too_large_rejected(self):
        large = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * (MAX_FILE_SIZE_BYTES + 100)
        # Override format detection: real WAV bytes start
        large_wav = make_wav_bytes(1.0)
        big_data = large_wav + b"\x00" * (MAX_FILE_SIZE_BYTES + 100)
        # Create a file just over 50MB (not a valid WAV beyond header)
        oversized = b"ID3" + b"\x00" * (MAX_FILE_SIZE_BYTES + 1)
        result = validate_audio_file(oversized, "big.mp3")
        assert result.is_valid is False
        assert "exceeds maximum" in result.error

    def test_long_wav_rejected(self, wav_bytes_long):
        result = validate_audio_file(wav_bytes_long, "long.wav")
        assert result.is_valid is False
        assert "duration" in result.error.lower()

    def test_wav_duration_error_before_size(self, wav_bytes_long):
        result = validate_audio_file(wav_bytes_long, "long.wav")
        assert "duration" in result.error.lower()


class TestExtractAudioProperties:
    def test_wav_properties(self, wav_bytes_5s):
        props = extract_audio_properties(wav_bytes_5s, "wav")
        assert 4.9 <= props.duration_seconds <= 5.1
        assert props.sample_rate == 16000
        assert props.channels == 1
        assert props.format == "wav"

    def test_wav_1channel(self):
        data = make_wav_bytes(2.0, sample_rate=44100, channels=1)
        props = extract_audio_properties(data, "wav")
        assert 1.9 <= props.duration_seconds <= 2.1
        assert props.sample_rate == 44100
