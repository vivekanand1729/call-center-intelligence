"""Unit tests for display formatters."""
from __future__ import annotations

import pytest

from src.utils.formatters import format_qa, format_summary, format_transcript, secs_to_mmss
from src.graph.state import (
    ActionItem,
    ComplianceFlag,
    Entity,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


class TestSecsToMmss:
    def test_zero(self):
        assert secs_to_mmss(0.0) == "00:00"

    def test_90_seconds(self):
        assert secs_to_mmss(90.0) == "01:30"

    def test_one_minute(self):
        assert secs_to_mmss(60.0) == "01:00"

    def test_two_hours(self):
        assert secs_to_mmss(7200.0) == "120:00"

    def test_negative_returns_zero(self):
        assert secs_to_mmss(-5.0) == "00:00"

    def test_fractional_seconds(self):
        result = secs_to_mmss(90.7)
        assert result == "01:31"


def _make_summary():
    return SummaryResult(
        call_id="x",
        call_purpose="Billing dispute",
        key_discussion_points=["Customer charged twice", "Agent issued refund"],
        action_items=[ActionItem(description="Send confirmation email", owner="Agent", deadline="2024-01-15")],
        resolution_status=ResolutionStatus.resolved,
        sentiment_trajectory="Frustrated → Satisfied",
        entities=[Entity(name="John Smith", entity_type="person")],
    )


def _make_dim(score: int = 4) -> QADimensionScore:
    return QADimensionScore(dimension="TestDim", score=score, justification="Good handling at 01:30")


def _make_qa(flags=None):
    dim = _make_dim()
    return QAScoreResult(
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        overall_score=4.0,
        compliance_flags=flags or [],
    )


class TestFormatSummary:
    def test_contains_purpose(self):
        s = _make_summary()
        md = format_summary(s)
        assert "Billing dispute" in md

    def test_contains_resolution(self):
        md = format_summary(_make_summary())
        assert "resolved" in md.lower()

    def test_contains_sentiment(self):
        md = format_summary(_make_summary())
        assert "Frustrated → Satisfied" in md

    def test_contains_action_items(self):
        md = format_summary(_make_summary())
        assert "Send confirmation email" in md

    def test_contains_entities(self):
        md = format_summary(_make_summary())
        assert "John Smith" in md

    def test_none_returns_placeholder(self):
        assert "No summary" in format_summary(None)


class TestFormatQA:
    def test_contains_overall_score(self):
        md = format_qa(_make_qa())
        assert "4.0" in md

    def test_no_flags_message(self):
        md = format_qa(_make_qa())
        assert "No compliance issues detected" in md

    def test_with_compliance_flags(self):
        flags = [ComplianceFlag(description="No ID check", severity="high")]
        md = format_qa(_make_qa(flags=flags))
        assert "No ID check" in md
        assert "HIGH" in md

    def test_critical_flag_shown(self):
        flags = [ComplianceFlag(description="Data breach risk", severity="critical")]
        md = format_qa(_make_qa(flags=flags))
        assert "CRITICAL" in md

    def test_none_returns_placeholder(self):
        assert "No QA scores" in format_qa(None)


class TestFormatTranscript:
    def test_formats_with_timestamps(self):
        tr = TranscriptionResult(
            call_id="x",
            full_text="Hello",
            segments=[TranscriptionSegment(start=0.0, end=5.0, text="Hello", speaker="Agent", confidence=0.9)],
        )
        result = format_transcript(tr)
        assert "00:00" in result
        assert "Agent" in result
        assert "Hello" in result

    def test_low_confidence_marker(self):
        tr = TranscriptionResult(
            call_id="x",
            full_text="Test",
            segments=[TranscriptionSegment(start=0.0, end=5.0, text="Test", speaker="Agent", confidence=0.2)],
        )
        result = format_transcript(tr)
        assert "[LOW CONF]" in result
