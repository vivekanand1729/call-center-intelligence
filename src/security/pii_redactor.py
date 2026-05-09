"""PII redactor: SSN, credit card, email, phone — right-to-left replacement."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple

from src.graph.state import TranscriptionResult, TranscriptionSegment


class _PIIPattern(NamedTuple):
    pattern: re.Pattern
    placeholder: str
    name: str


PII_PATTERNS: list[_PIIPattern] = [
    _PIIPattern(
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
        "SSN",
    ),
    _PIIPattern(
        re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
        "[REDACTED_CREDIT_CARD]",
        "CREDIT_CARD",
    ),
    _PIIPattern(
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
        "EMAIL",
    ),
    _PIIPattern(
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        "[REDACTED_PHONE]",
        "PHONE",
    ),
]


@dataclass
class RedactionResult:
    redacted_text: str
    pii_found: bool
    types_found: list[str] = field(default_factory=list)


@dataclass
class _Match:
    start: int
    end: int
    placeholder: str
    name: str


def _collect_matches(text: str) -> list[_Match]:
    matches: list[_Match] = []
    for pii in PII_PATTERNS:
        for m in pii.pattern.finditer(text):
            matches.append(_Match(m.start(), m.end(), pii.placeholder, pii.name))
    # Sort descending so right-to-left replacement preserves offsets
    matches.sort(key=lambda x: x.start, reverse=True)
    # Deduplicate overlapping matches (keep earlier start)
    deduped: list[_Match] = []
    for m in matches:
        if deduped and m.end > deduped[-1].start:
            # overlapping — keep the one with smaller start (earlier)
            if m.start < deduped[-1].start:
                deduped[-1] = m
        else:
            deduped.append(m)
    return deduped


def redact_pii(text: str) -> RedactionResult:
    matches = _collect_matches(text)
    if not matches:
        return RedactionResult(redacted_text=text, pii_found=False)

    result = text
    types_found: set[str] = set()
    for m in matches:
        result = result[: m.start] + m.placeholder + result[m.end :]
        types_found.add(m.name)

    return RedactionResult(redacted_text=result, pii_found=True, types_found=sorted(types_found))


def redact_transcription(transcript: TranscriptionResult) -> TranscriptionResult:
    """Apply PII redaction to full_text and every segment."""
    full_result = redact_pii(transcript.full_text)
    redacted_segments: list[TranscriptionSegment] = []
    for seg in transcript.segments:
        seg_result = redact_pii(seg.text)
        redacted_segments.append(seg.model_copy(update={"text": seg_result.redacted_text}))

    return transcript.model_copy(
        update={
            "full_text": full_result.redacted_text,
            "segments": redacted_segments,
        }
    )
