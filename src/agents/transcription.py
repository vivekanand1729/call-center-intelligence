"""Transcription agent with SHA-256 caching and heuristic speaker diarization.

Uses faster-whisper when available; falls back to OpenAI Whisper API.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import wave
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from src.graph.state import IntakeResult, TranscriptionResult, TranscriptionSegment

# Singleton Whisper model
_model: Any = None
_model_size: str = ""

# Agent/Customer content patterns for diarization
_AGENT_PATTERNS = re.compile(
    r"(?i)(thank you for calling|how (?:can|may) i (?:help|assist)|this is|my name is|i can help|let me (?:check|look)|i'll transfer|please hold|is there anything else)",
    re.IGNORECASE,
)
_CUSTOMER_PATTERNS = re.compile(
    r"(?i)(i(?:'m| am) calling|i need|i want|i have a (?:problem|question|issue)|my (?:account|order|bill)|can you help|i was charged|i didn't|i can't)",
    re.IGNORECASE,
)

_WHISPER_ARTIFACTS = re.compile(
    r"\[BLANK_AUDIO\]|thanks for watching|thank you for watching|\[music\]|\[applause\]|\[noise\]|\[laughter\]",
    re.IGNORECASE,
)
_REPEATED_DOTS = re.compile(r"\.{4,}")
_REPEATED_PHRASE = re.compile(r"\b(\w+(?:\s+\w+){0,3})\b(?:\s+\1){3,}", re.IGNORECASE)


def _detect_device() -> tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def _get_whisper_model(model_size: str = "tiny") -> Any:
    global _model, _model_size
    if _model is not None and _model_size == model_size:
        return _model
    try:
        from faster_whisper import WhisperModel
        device, compute_type = _detect_device()
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _model_size = model_size
        return _model
    except ImportError:
        _model = _OpenAIWhisperAdapter()
        _model_size = model_size
        return _model


def _clean_transcript_text(text: str) -> str:
    text = _WHISPER_ARTIFACTS.sub("", text)
    text = _REPEATED_DOTS.sub("...", text)
    text = _REPEATED_PHRASE.sub(r"\1", text)
    return text.strip()


@dataclass
class _MockSegment:
    text: str
    start: float
    end: float
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.05
    words: list = None

    def __post_init__(self):
        if self.words is None:
            self.words = []


class _OpenAIWhisperAdapter:
    """Fallback transcription using OpenAI Whisper API."""

    def transcribe(self, audio_path: str, **kwargs) -> tuple[Iterator[_MockSegment], Any]:
        import os
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    segments.append(
                        _MockSegment(
                            text=seg.text,
                            start=float(seg.start),
                            end=float(seg.end),
                            avg_logprob=-0.2,
                            no_speech_prob=0.05,
                        )
                    )
            else:
                # Single segment fallback
                segments.append(_MockSegment(text=response.text, start=0.0, end=30.0))
            return iter(segments), None
        except Exception as e:
            segments = [_MockSegment(text=f"[Transcription error: {e}]", start=0.0, end=1.0)]
            return iter(segments), None


class SpeakerDiarizer:
    """Heuristic speaker diarization using content patterns and gap-based switching."""

    def __init__(self):
        self._last_speaker = "Agent"
        self._last_end = 0.0

    def assign(self, segment: _MockSegment, prev_text: str = "") -> str:
        text = segment.text.strip()
        gap = segment.start - self._last_end

        # Content pattern takes priority
        if _AGENT_PATTERNS.search(text):
            speaker = "Agent"
        elif _CUSTOMER_PATTERNS.search(text):
            speaker = "Customer"
        # Gap-based switching
        elif gap > 1.2:
            speaker = "Customer" if self._last_speaker == "Agent" else "Agent"
        # Question-answer switching
        elif prev_text.strip().endswith("?"):
            speaker = "Customer" if self._last_speaker == "Agent" else "Agent"
        # Short affirmation after long segment
        elif len(text.split()) <= 3 and len(prev_text.split()) > 10:
            speaker = "Customer" if self._last_speaker == "Agent" else "Agent"
        else:
            speaker = self._last_speaker

        self._last_speaker = speaker
        self._last_end = segment.end
        return speaker


def _compute_audio_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _check_cache(audio_hash: str, call_id: str) -> Optional[TranscriptionResult]:
    try:
        from src.database.connection import session_scope
        from src.database.models import TranscriptionCache
        with session_scope() as session:
            row = session.query(TranscriptionCache).filter_by(audio_hash=audio_hash).first()
            if row:
                result = TranscriptionResult.model_validate_json(row.transcription_json)
                result = result.model_copy(update={"call_id": call_id, "from_cache": True})
                return result
    except Exception:
        pass
    return None


def _save_cache(audio_hash: str, result: TranscriptionResult) -> None:
    try:
        from src.database.connection import session_scope
        from src.database.models import TranscriptionCache
        with session_scope() as session:
            existing = session.query(TranscriptionCache).filter_by(audio_hash=audio_hash).first()
            if not existing:
                row = TranscriptionCache(
                    audio_hash=audio_hash,
                    transcription_json=result.model_dump_json(),
                )
                session.add(row)
    except Exception:
        pass


def run_transcription(intake: IntakeResult, model_size: str = "tiny") -> TranscriptionResult:
    if not intake.temp_audio_path:
        return TranscriptionResult(
            call_id=intake.call_id,
            full_text="[No audio path available]",
            segments=[],
        )

    # Check SHA-256 cache first
    audio_hash = _compute_audio_hash(intake.temp_audio_path)
    cached = _check_cache(audio_hash, intake.call_id)
    if cached:
        return cached

    model = _get_whisper_model(model_size)
    segments_iter, info = model.transcribe(
        intake.temp_audio_path,
        beam_size=1,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    diarizer = SpeakerDiarizer()
    result_segments: list[TranscriptionSegment] = []
    prev_text = ""
    total_conf = 0.0

    for seg in segments_iter:
        text = _clean_transcript_text(seg.text)
        if not text:
            continue
        speaker = diarizer.assign(seg, prev_text)
        logprob_conf = max(0.0, min(1.0, 1.0 + seg.avg_logprob))
        speech_conf = 1.0 - seg.no_speech_prob
        conf = round(logprob_conf * 0.7 + speech_conf * 0.3, 4)
        result_segments.append(
            TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=text,
                speaker=speaker,
                confidence=conf,
            )
        )
        total_conf += conf
        prev_text = text

    if not result_segments:
        full_text = ""
        avg_conf = 1.0
    else:
        full_text = " ".join(s.text for s in result_segments)
        avg_conf = round(total_conf / len(result_segments), 4)

    result = TranscriptionResult(
        call_id=intake.call_id,
        full_text=full_text,
        segments=result_segments,
        avg_confidence=avg_conf,
        low_confidence=avg_conf < 0.4,
    )

    _save_cache(audio_hash, result)
    return result
