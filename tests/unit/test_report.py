"""Unit tests for report agent."""
from __future__ import annotations

import pytest

from src.agents.report import (
    compile_report,
    generate_report_json,
    generate_report_pdf,
)
from src.graph.state import (
    ActionItem,
    ComplianceFlag,
    Entity,
    IntakeResult,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def _make_full_state():
    intake = IntakeResult(call_id="report-test", validation_passed=True, audio_format="wav")
    transcript = TranscriptionResult(
        call_id="report-test",
        full_text="Hello how can I help?",
        segments=[TranscriptionSegment(start=0.0, end=3.0, text="Hello", speaker="Agent", confidence=0.9)],
    )
    summary = SummaryResult(
        call_id="report-test",
        call_purpose="Billing issue",
        key_discussion_points=["Charge dispute"],
        resolution_status=ResolutionStatus.resolved,
        sentiment_trajectory="Neutral",
    )
    dim = QADimensionScore(dimension="X", score=4, justification="Good")
    qa = QAScoreResult(
        call_id="report-test",
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        overall_score=4.0,
    )
    return intake, transcript, summary, qa


class TestCompileReport:
    def test_call_id_propagated(self):
        intake, tr, sm, qa = _make_full_state()
        report = compile_report("report-test", intake=intake, transcription=tr, summary=sm, qa_scores=qa)
        assert report.call_id == "report-test"

    def test_status_default_completed(self):
        intake, tr, sm, qa = _make_full_state()
        report = compile_report("test-id", intake=intake)
        assert report.status == "completed"

    def test_flagged_status(self):
        report = compile_report("x", status="flagged_for_review")
        assert report.status == "flagged_for_review"


class TestGenerateReportJson:
    def test_json_contains_call_id(self):
        intake, tr, sm, qa = _make_full_state()
        report = compile_report("report-test", intake=intake, transcription=tr, summary=sm, qa_scores=qa)
        json_str = generate_report_json(report)
        assert '"call_id"' in json_str
        assert '"summary"' in json_str

    def test_json_is_valid(self):
        import json
        report = compile_report("x")
        json_str = generate_report_json(report)
        parsed = json.loads(json_str)
        assert parsed["call_id"] == "x"


class TestGenerateReportPdf:
    def test_returns_bytes(self):
        intake, tr, sm, qa = _make_full_state()
        report = compile_report("report-test", intake=intake, summary=sm, qa_scores=qa)
        pdf_bytes = generate_report_pdf(report)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_pdf_starts_with_pdf_header_or_text(self):
        report = compile_report("x")
        pdf_bytes = generate_report_pdf(report)
        # Either real PDF (%PDF-) or our text fallback
        assert pdf_bytes[:5] == b"%PDF-" or b"CALL CENTER" in pdf_bytes
