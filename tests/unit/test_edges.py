"""Unit tests for LangGraph routing edge functions."""
from __future__ import annotations

import pytest

from src.graph.edges import (
    route_after_injection,
    route_after_intake,
    route_after_qa,
    route_after_transcription,
)
from src.graph.state import (
    ComplianceFlag,
    IntakeResult,
    QADimensionScore,
    QAScoreResult,
)


def _make_dim():
    return QADimensionScore(dimension="X", score=3, justification="ok")


def _make_qa(flags=None) -> QAScoreResult:
    dim = _make_dim()
    return QAScoreResult(
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        overall_score=3.0,
        compliance_flags=flags or [],
    )


class TestRouteAfterIntake:
    def test_valid_intake_routes_to_transcribe(self):
        intake = IntakeResult(call_id="x", validation_passed=True)
        state = {"intake": intake}
        assert route_after_intake(state) == "transcribe"

    def test_failed_intake_routes_to_error(self):
        intake = IntakeResult(call_id="x", validation_passed=False, validation_error="Bad file")
        state = {"intake": intake}
        assert route_after_intake(state) == "error"

    def test_no_intake_routes_to_error(self):
        assert route_after_intake({}) == "error"


class TestRouteAfterTranscription:
    def test_always_summarize(self):
        assert route_after_transcription({}) == "summarize"


class TestRouteAfterInjection:
    def test_injection_detected_routes_to_error(self):
        state = {"status": "injection_detected"}
        assert route_after_injection(state) == "error"

    def test_clean_routes_to_pii_redact(self):
        state = {"status": "injection_check_passed"}
        assert route_after_injection(state) == "pii_redact"


class TestRouteAfterQA:
    def test_no_flags_routes_to_report(self):
        state = {"qa_scores": _make_qa()}
        assert route_after_qa(state) == "report"

    def test_low_flag_routes_to_report(self):
        flags = [ComplianceFlag(description="Minor issue", severity="low")]
        state = {"qa_scores": _make_qa(flags=flags)}
        assert route_after_qa(state) == "report"

    def test_high_flag_routes_to_report(self):
        flags = [ComplianceFlag(description="Serious issue", severity="high")]
        state = {"qa_scores": _make_qa(flags=flags)}
        assert route_after_qa(state) == "report"

    def test_critical_flag_routes_to_supervisor(self):
        flags = [ComplianceFlag(description="Critical violation", severity="critical")]
        state = {"qa_scores": _make_qa(flags=flags)}
        assert route_after_qa(state) == "supervisor_review"

    def test_error_status_routes_to_error(self):
        state = {"status": "error"}
        assert route_after_qa(state) == "error"

    def test_no_qa_routes_to_report(self):
        assert route_after_qa({}) == "report"
