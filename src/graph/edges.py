"""LangGraph conditional routing edge functions."""
from __future__ import annotations

from src.graph.state import IntakeResult, PipelineState, QAScoreResult, TranscriptionResult


def route_after_intake(state: PipelineState) -> str:
    intake: IntakeResult | None = state.get("intake")
    if intake and intake.validation_passed:
        return "transcribe"
    return "error"


def route_after_transcription(state: PipelineState) -> str:
    return "summarize"


def route_after_injection(state: PipelineState) -> str:
    if state.get("status") == "injection_detected":
        return "error"
    return "pii_redact"


def route_after_qa(state: PipelineState) -> str:
    if state.get("status") == "error":
        return "error"
    qa: QAScoreResult | None = state.get("qa_scores")
    if qa and qa.compliance_flags:
        for flag in qa.compliance_flags:
            if flag.severity.lower() == "critical":
                return "supervisor_review"
    return "report"
