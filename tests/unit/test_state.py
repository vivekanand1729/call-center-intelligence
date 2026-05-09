"""Unit tests for Pydantic data models and PipelineState."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.graph.state import (
    ActionItem,
    AudioInput,
    AudioProperties,
    CallReport,
    ComplianceFlag,
    Entity,
    IntakeResult,
    PIIScanResult,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


class TestTranscriptionSegment:
    def test_valid_segment(self):
        seg = TranscriptionSegment(start=0.0, end=5.0, text="Hello", speaker="Agent", confidence=0.9)
        assert seg.confidence == 0.9

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionSegment(start=0.0, end=5.0, text="Hello", confidence=1.5)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionSegment(start=0.0, end=5.0, text="Hello", confidence=-0.1)

    def test_confidence_boundary_values(self):
        seg0 = TranscriptionSegment(start=0.0, end=1.0, text="x", confidence=0.0)
        seg1 = TranscriptionSegment(start=0.0, end=1.0, text="x", confidence=1.0)
        assert seg0.confidence == 0.0
        assert seg1.confidence == 1.0


class TestQADimensionScore:
    def test_valid_score(self):
        d = QADimensionScore(dimension="Empathy", score=3, justification="ok")
        assert d.score == 3

    def test_score_zero_raises(self):
        with pytest.raises(ValidationError):
            QADimensionScore(dimension="Empathy", score=0, justification="bad")

    def test_score_six_raises(self):
        with pytest.raises(ValidationError):
            QADimensionScore(dimension="Empathy", score=6, justification="bad")

    def test_score_boundary_1(self):
        d = QADimensionScore(dimension="X", score=1, justification="low")
        assert d.score == 1

    def test_score_boundary_5(self):
        d = QADimensionScore(dimension="X", score=5, justification="high")
        assert d.score == 5


class TestQAScoreResult:
    def _make_qa(self, overall=3.0):
        dim = QADimensionScore(dimension="X", score=3, justification="ok")
        return QAScoreResult(
            professionalism=dim,
            empathy=dim,
            problem_resolution=dim,
            compliance=dim,
            communication_clarity=dim,
            overall_score=overall,
        )

    def test_valid_overall(self):
        qa = self._make_qa(3.5)
        assert qa.overall_score == 3.5

    def test_overall_below_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_qa(0.5)

    def test_overall_above_5_raises(self):
        with pytest.raises(ValidationError):
            self._make_qa(5.5)


class TestResolutionStatus:
    def test_enum_values(self):
        assert ResolutionStatus.resolved == "resolved"
        assert ResolutionStatus.unresolved == "unresolved"
        assert ResolutionStatus.escalated == "escalated"


class TestAudioInput:
    def test_required_fields(self):
        ai = AudioInput(audio_data=b"data", filename="test.wav")
        assert ai.audio_data == b"data"
        assert ai.caller_id is None

    def test_optional_fields(self):
        ai = AudioInput(audio_data=b"d", filename="f.wav", caller_id="123", department="sales")
        assert ai.caller_id == "123"


class TestComplianceFlag:
    def test_flag_creation(self):
        f = ComplianceFlag(description="No ID verification", severity="high")
        assert f.severity == "high"
