"""Audio format detection, validation, and property extraction."""
from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import Optional

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DURATION_SECONDS = 3600  # 60 minutes
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a"}


class AudioValidationError(Exception):
    pass


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None


@dataclass
class AudioProperties:
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str


def detect_audio_format(data: bytes) -> str:
    """Detect audio format by inspecting the first 12 bytes (magic bytes)."""
    if len(data) < 12:
        return "unknown"
    header = data[:12]
    # WAV: RIFF....WAVE
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    # MP3: ID3 header or sync bits 0xFF 0xEx
    if header[:3] == b"ID3":
        return "mp3"
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "mp3"
    # FLAC: fLaC
    if header[:4] == b"fLaC":
        return "flac"
    # M4A: ftyp at bytes 4-7
    if header[4:8] == b"ftyp":
        return "m4a"
    return "unknown"


def validate_audio_file(data: bytes, filename: str) -> ValidationResult:
    """Validate audio file by format, size, and duration."""
    if not data:
        return ValidationResult(is_valid=False, error="Empty file provided")

    fmt = detect_audio_format(data)
    if fmt not in SUPPORTED_FORMATS:
        return ValidationResult(
            is_valid=False,
            error=f"Unsupported format '{fmt}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS)).upper()}",
        )

    # For WAV, check duration before size to give the more specific error
    if fmt == "wav":
        try:
            props = _extract_wav_properties(data)
            if props.duration_seconds > MAX_DURATION_SECONDS:
                return ValidationResult(
                    is_valid=False,
                    error=f"Audio duration {props.duration_seconds:.1f}s exceeds maximum {MAX_DURATION_SECONDS}s (60 minutes)",
                )
        except AudioValidationError as e:
            return ValidationResult(is_valid=False, error=str(e))

    if len(data) > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            is_valid=False,
            error=f"File size {len(data)} bytes exceeds maximum {MAX_FILE_SIZE_BYTES} bytes (50 MB)",
        )

    return ValidationResult(is_valid=True)


def extract_audio_properties(data: bytes, fmt: str) -> AudioProperties:
    """Extract duration, sample rate, and channels from audio data."""
    if fmt == "wav":
        return _extract_wav_properties(data)
    return _extract_non_wav_properties(data, fmt)


def _extract_wav_properties(data: bytes) -> AudioProperties:
    try:
        buf = io.BytesIO(data)
        with wave.open(buf) as w:
            frames = w.getnframes()
            rate = w.getframerate()
            channels = w.getnchannels()
            duration = frames / rate if rate > 0 else 0.0
        return AudioProperties(
            duration_seconds=round(duration, 4),
            sample_rate=rate,
            channels=channels,
            format="wav",
        )
    except wave.Error as e:
        raise AudioValidationError(f"Cannot read WAV file: {e}") from e
    except Exception as e:
        raise AudioValidationError(f"Corrupt or unreadable WAV file: {e}") from e


def _extract_non_wav_properties(data: bytes, fmt: str) -> AudioProperties:
    """Extract properties for MP3, FLAC, M4A using mutagen if available."""
    try:
        import mutagen
        import io as _io
        buf = _io.BytesIO(data)
        audio = mutagen.File(buf)
        if audio is not None and audio.info is not None:
            return AudioProperties(
                duration_seconds=round(audio.info.length, 4),
                sample_rate=getattr(audio.info, "sample_rate", 44100),
                channels=getattr(audio.info, "channels", 2),
                format=fmt,
            )
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: return default properties
    return AudioProperties(
        duration_seconds=0.0,
        sample_rate=44100,
        channels=2,
        format=fmt,
    )
