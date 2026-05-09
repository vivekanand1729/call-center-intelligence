"""All 14 Pydantic data models and PipelineState TypedDict."""
from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AudioInput(BaseModel):
    audio_data: bytes
    filename: str
    caller_id: Optional[str] = None
    department: Optional[str] = None
    timestamp: Optional[str] = None


class AudioProperties(BaseModel):
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    format: str = ""


class PIIScanResult(BaseModel):
    pii_detected: bool = False
    affected_fields: list[str] = Field(default_factory=list)


class IntakeResult(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    validation_passed: bool = False
    validation_error: Optional[str] = None
    audio_format: str = ""
    audio_properties: AudioProperties = Field(default_factory=AudioProperties)
    pii_scan: PIIScanResult = Field(default_factory=PIIScanResult)
    temp_audio_path: Optional[str] = None


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = "Unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class TranscriptionResult(BaseModel):
    call_id: str
    full_text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    language: str = "en"
    avg_confidence: float = 1.0
    low_confidence: bool = False
    from_cache: bool = False


class ResolutionStatus(StrEnum):
    resolved = "resolved"
    unresolved = "unresolved"
    escalated = "escalated"


class ActionItem(BaseModel):
    description: str
    owner: str = "Unknown"
    deadline: Optional[str] = None


class Entity(BaseModel):
    name: str
    entity_type: str


class SummaryResult(BaseModel):
    call_id: str = ""
    call_purpose: str
    key_discussion_points: list[str] = Field(min_length=1)
    action_items: list[ActionItem] = Field(default_factory=list)
    resolution_status: ResolutionStatus
    sentiment_trajectory: str
    entities: list[Entity] = Field(default_factory=list)


class QADimensionScore(BaseModel):
    dimension: str
    score: int = Field(ge=1, le=5)
    justification: str


class ComplianceFlag(BaseModel):
    description: str
    severity: str  # low / medium / high / critical
    timestamp_reference: Optional[str] = None


class QAScoreResult(BaseModel):
    call_id: str = ""
    professionalism: QADimensionScore
    empathy: QADimensionScore
    problem_resolution: QADimensionScore
    compliance: QADimensionScore
    communication_clarity: QADimensionScore
    overall_score: float = Field(ge=1.0, le=5.0)
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)


class CallReport(BaseModel):
    call_id: str
    audio_filename: str = ""
    intake: Optional[IntakeResult] = None
    transcription: Optional[TranscriptionResult] = None
    summary: Optional[SummaryResult] = None
    qa_scores: Optional[QAScoreResult] = None
    status: str = "completed"
    processed_at: Optional[str] = None
    trace_id: Optional[str] = None


# LangGraph shared state
from typing import TypedDict


class PipelineState(TypedDict, total=False):
    audio_input: AudioInput
    intake: IntakeResult
    transcription: TranscriptionResult
    summary: SummaryResult
    qa_scores: QAScoreResult
    report: CallReport
    error: str
    status: str
