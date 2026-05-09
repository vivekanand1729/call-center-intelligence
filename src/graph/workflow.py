"""LangGraph StateGraph workflow: 7 pipeline stages + error and supervisor nodes."""
from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, StateGraph

from src.graph.edges import (
    route_after_injection,
    route_after_intake,
    route_after_qa,
    route_after_transcription,
)
from src.graph.state import PipelineState
from src.security.audit import AuditLogger

_audit: AuditLogger = AuditLogger()


# ──────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────

def intake_step(state: PipelineState) -> dict:
    from src.agents.intake import run_intake
    audio_input = state["audio_input"]
    try:
        intake = run_intake(audio_input)
        _audit.log(intake.call_id, "intake_completed", details={"validation_passed": intake.validation_passed})
        if not intake.validation_passed:
            return {"intake": intake, "status": "failed", "error": intake.validation_error or "Intake failed"}
        return {"intake": intake, "status": "intake_complete"}
    except Exception as e:
        return {"status": "failed", "error": f"Intake error: {e}"}


def transcription_step(state: PipelineState) -> dict:
    from src.agents.transcription import run_transcription
    from src.utils.config import load_config
    intake = state["intake"]
    cfg = load_config()
    try:
        result = run_transcription(intake, model_size=cfg.whisper_model_size)
        _audit.log(intake.call_id, "transcription_completed", details={"from_cache": result.from_cache})
        return {"transcription": result, "status": "transcription_complete"}
    except Exception as e:
        _audit.log(intake.call_id, "transcription_failed", details={"error": str(e)})
        return {"status": "failed", "error": f"Transcription error: {e}"}


def injection_check_step(state: PipelineState) -> dict:
    from src.security.injection_detector import detect_injection
    intake = state.get("intake")
    transcription = state.get("transcription")
    call_id = intake.call_id if intake else "unknown"
    if transcription is None:
        return {"status": "failed", "error": "No transcription to check"}
    result = detect_injection(transcription.full_text)
    if result.injection_detected:
        _audit.log(call_id, "injection_detected", details={"patterns": result.matched_patterns})
        return {"status": "injection_detected", "error": f"Prompt injection detected: {result.matched_patterns}"}
    _audit.log(call_id, "injection_check_passed")
    return {"status": "injection_check_passed"}


def pii_redaction_step(state: PipelineState) -> dict:
    from src.security.pii_redactor import redact_transcription
    intake = state.get("intake")
    transcription = state.get("transcription")
    call_id = intake.call_id if intake else "unknown"
    if transcription is None:
        return {"status": "failed", "error": "No transcription for PII redaction"}
    redacted = redact_transcription(transcription)
    _audit.log(call_id, "pii_redaction_completed")
    return {"transcription": redacted, "status": "pii_redacted"}


def summarize_and_qa_step(state: PipelineState) -> dict:
    from src.agents.qa_scoring import QAScoringError, run_qa_scoring
    from src.agents.summarization import SummarizationError, run_summarization
    intake = state.get("intake")
    transcription = state.get("transcription")
    call_id = intake.call_id if intake else "unknown"
    if transcription is None:
        return {"status": "failed", "error": "No transcription for analysis"}
    try:
        summary = run_summarization(transcription)
        _audit.log(call_id, "summarization_completed")
    except SummarizationError as e:
        _audit.log(call_id, "summarization_failed", details={"error": str(e)})
        return {"status": "failed", "error": str(e)}
    try:
        qa_scores = run_qa_scoring(transcription, summary=summary)
        _audit.log(call_id, "qa_scoring_completed", details={"overall_score": qa_scores.overall_score})
        return {"summary": summary, "qa_scores": qa_scores, "status": "analysis_complete"}
    except QAScoringError as e:
        _audit.log(call_id, "qa_scoring_failed", details={"error": str(e)})
        return {"summary": summary, "status": "failed", "error": str(e)}


def report_step(state: PipelineState) -> dict:
    from src.agents.report import compile_report, persist_report
    intake = state.get("intake")
    call_id = intake.call_id if intake else "unknown"
    try:
        report = compile_report(
            call_id=call_id,
            intake=intake,
            transcription=state.get("transcription"),
            summary=state.get("summary"),
            qa_scores=state.get("qa_scores"),
            status="completed",
        )
        persist_report(report)
        _audit.log(call_id, "completed")
        return {"report": report, "status": "completed"}
    except Exception as e:
        _audit.log(call_id, "report_failed", details={"error": str(e)})
        return {"status": "failed", "error": f"Report error: {e}"}


def supervisor_review_step(state: PipelineState) -> dict:
    from src.agents.report import compile_report, persist_report
    intake = state.get("intake")
    call_id = intake.call_id if intake else "unknown"
    try:
        report = compile_report(
            call_id=call_id,
            intake=intake,
            transcription=state.get("transcription"),
            summary=state.get("summary"),
            qa_scores=state.get("qa_scores"),
            status="flagged_for_review",
        )
        persist_report(report)
        _audit.log(call_id, "flagged_for_review", details={"reason": "critical_compliance_flag"})
        return {"report": report, "status": "flagged_for_review"}
    except Exception as e:
        return {"status": "failed", "error": f"Supervisor review error: {e}"}


def error_step(state: PipelineState) -> dict:
    intake = state.get("intake")
    call_id = intake.call_id if intake else "unknown"
    error_msg = (
        state.get("error")
        or (intake.validation_error if intake else None)
        or "Pipeline failed"
    )
    _audit.log(call_id, "pipeline_failed", details={"error": error_msg})
    # Persist a failed call record
    try:
        from src.agents.report import compile_report, persist_report
        report = compile_report(
            call_id=call_id,
            intake=intake,
            transcription=state.get("transcription"),
            status="failed",
        )
        persist_report(report)
    except Exception:
        pass
    return {"status": "failed", "error": error_msg}


# ──────────────────────────────────────────────
# Graph compilation
# ──────────────────────────────────────────────

def compile_workflow(config: Any = None, db_engine: Any = None) -> Any:
    from src.database.connection import get_engine, init_db
    engine = db_engine or get_engine()
    init_db(engine)

    graph = StateGraph(PipelineState)

    graph.add_node("intake_step", intake_step)
    graph.add_node("transcribe_step", transcription_step)
    graph.add_node("injection_check_step", injection_check_step)
    graph.add_node("pii_redact_step", pii_redaction_step)
    graph.add_node("summarize_and_qa_step", summarize_and_qa_step)
    graph.add_node("report_step", report_step)
    graph.add_node("supervisor_step", supervisor_review_step)
    graph.add_node("error_step", error_step)

    graph.set_entry_point("intake_step")

    graph.add_conditional_edges(
        "intake_step",
        route_after_intake,
        {"transcribe": "transcribe_step", "error": "error_step"},
    )
    graph.add_edge("transcribe_step", "injection_check_step")
    graph.add_conditional_edges(
        "injection_check_step",
        route_after_injection,
        {"pii_redact": "pii_redact_step", "error": "error_step"},
    )
    graph.add_edge("pii_redact_step", "summarize_and_qa_step")
    graph.add_conditional_edges(
        "summarize_and_qa_step",
        route_after_qa,
        {"report": "report_step", "supervisor_review": "supervisor_step", "error": "error_step"},
    )
    graph.add_edge("report_step", END)
    graph.add_edge("supervisor_step", END)
    graph.add_edge("error_step", END)

    return graph.compile()
