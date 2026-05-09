"""Intake agent: validates audio, scans metadata for PII."""
from __future__ import annotations

import re
import tempfile
import uuid
from typing import Optional

from src.graph.state import AudioInput, AudioProperties, IntakeResult, PIIScanResult
from src.utils.audio import (
    MAX_DURATION_SECONDS,
    AudioValidationError,
    AudioProperties as AudioPropsUtil,
    detect_audio_format,
    extract_audio_properties,
    validate_audio_file,
)

# PII patterns for metadata fields
_META_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"), "credit_card"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "phone"),
]

_EMPTY_AUDIO_PROPS = AudioProperties()
_EMPTY_PII = PIIScanResult()


def _make_failed_result(call_id: str, error: str) -> IntakeResult:
    return IntakeResult(
        call_id=call_id,
        validation_passed=False,
        validation_error=error,
        audio_properties=_EMPTY_AUDIO_PROPS,
        pii_scan=_EMPTY_PII,
    )


def _scan_metadata_pii(fields: dict[str, Optional[str]]) -> PIIScanResult:
    affected: list[str] = []
    for field_name, value in fields.items():
        if not value:
            continue
        for pattern, _ in _META_PII_PATTERNS:
            if pattern.search(value):
                affected.append(field_name)
                break
    return PIIScanResult(pii_detected=bool(affected), affected_fields=affected)


def run_intake(audio_input: AudioInput) -> IntakeResult:
    call_id = str(uuid.uuid4())

    # Validate
    result = validate_audio_file(audio_input.audio_data, audio_input.filename)
    if not result.is_valid:
        return _make_failed_result(call_id, result.error or "Validation failed")

    fmt = detect_audio_format(audio_input.audio_data)

    # Extract properties
    try:
        props_util = extract_audio_properties(audio_input.audio_data, fmt)
        audio_props = AudioProperties(
            duration_seconds=props_util.duration_seconds,
            sample_rate=props_util.sample_rate,
            channels=props_util.channels,
            format=fmt,
        )
    except AudioValidationError as e:
        return _make_failed_result(call_id, str(e))

    # Duration guard for non-WAV (WAV already checked in validate)
    if fmt != "wav" and audio_props.duration_seconds > MAX_DURATION_SECONDS:
        return _make_failed_result(
            call_id,
            f"Audio duration {audio_props.duration_seconds:.1f}s exceeds maximum {MAX_DURATION_SECONDS}s (60 minutes)",
        )

    # Scan metadata PII
    pii_scan = _scan_metadata_pii(
        {
            "caller_id": audio_input.caller_id,
            "department": audio_input.department,
        }
    )

    # Write to temp file
    suffix = f".{fmt}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_input.audio_data)
    tmp.flush()
    tmp.close()

    return IntakeResult(
        call_id=call_id,
        validation_passed=True,
        audio_format=fmt,
        audio_properties=audio_props,
        pii_scan=pii_scan,
        temp_audio_path=tmp.name,
    )
