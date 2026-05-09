"""Report agent: compile, persist, and generate PDF/JSON reports."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Optional

from src.graph.state import (
    CallReport,
    IntakeResult,
    QAScoreResult,
    SummaryResult,
    TranscriptionResult,
)


def compile_report(
    call_id: str,
    intake: Optional[IntakeResult] = None,
    transcription: Optional[TranscriptionResult] = None,
    summary: Optional[SummaryResult] = None,
    qa_scores: Optional[QAScoreResult] = None,
    status: str = "completed",
    trace_id: Optional[str] = None,
) -> CallReport:
    audio_filename = intake.temp_audio_path or "" if intake else ""
    # Use original filename if available
    return CallReport(
        call_id=call_id,
        audio_filename=audio_filename,
        intake=intake,
        transcription=transcription,
        summary=summary,
        qa_scores=qa_scores,
        status=status,
        processed_at=datetime.now(timezone.utc).isoformat(),
        trace_id=trace_id,
    )


def persist_report(report: CallReport, engine=None) -> None:
    from src.database.connection import session_scope
    from src.database.models import CallRecord

    with session_scope(engine) as session:
        existing = session.query(CallRecord).filter_by(call_id=report.call_id).first()
        if existing:
            existing.status = report.status
            existing.report_json = generate_report_json(report)
            if report.transcription:
                existing.transcript_text = report.transcription.full_text
            if report.summary:
                existing.summary_json = report.summary.model_dump_json()
            if report.qa_scores:
                existing.qa_scores_json = report.qa_scores.model_dump_json()
        else:
            record = CallRecord(
                call_id=report.call_id,
                status=report.status,
                audio_filename=report.audio_filename or "",
                transcript_text=report.transcription.full_text if report.transcription else None,
                summary_json=report.summary.model_dump_json() if report.summary else None,
                qa_scores_json=report.qa_scores.model_dump_json() if report.qa_scores else None,
                report_json=generate_report_json(report),
                trace_id=report.trace_id,
            )
            session.add(record)


def generate_report_json(report: CallReport) -> str:
    return report.model_dump_json(indent=2)


def generate_report_pdf(report: CallReport) -> bytes:
    """Generate a PDF report using ReportLab. Falls back to plain text bytes."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"Call Center Analysis Report", styles["Title"]))
        story.append(Paragraph(f"Call ID: {report.call_id}", styles["Normal"]))
        story.append(Paragraph(f"Status: {report.status}", styles["Normal"]))
        if report.processed_at:
            story.append(Paragraph(f"Processed: {report.processed_at}", styles["Normal"]))
        story.append(Spacer(1, 12))

        if report.summary:
            story.append(Paragraph("Summary", styles["Heading2"]))
            story.append(Paragraph(f"Purpose: {report.summary.call_purpose}", styles["Normal"]))
            story.append(Paragraph(f"Resolution: {report.summary.resolution_status.value}", styles["Normal"]))
            story.append(Paragraph(f"Sentiment: {report.summary.sentiment_trajectory}", styles["Normal"]))
            story.append(Spacer(1, 8))

        if report.qa_scores:
            story.append(Paragraph("QA Scores", styles["Heading2"]))
            story.append(Paragraph(f"Overall Score: {report.qa_scores.overall_score:.1f}/5.0", styles["Normal"]))
            for dim in [
                report.qa_scores.professionalism,
                report.qa_scores.empathy,
                report.qa_scores.problem_resolution,
                report.qa_scores.compliance,
                report.qa_scores.communication_clarity,
            ]:
                story.append(Paragraph(f"{dim.dimension}: {dim.score}/5 — {dim.justification}", styles["Normal"]))
            story.append(Spacer(1, 8))
            if report.qa_scores.compliance_flags:
                story.append(Paragraph("Compliance Flags", styles["Heading3"]))
                for flag in report.qa_scores.compliance_flags:
                    story.append(Paragraph(f"[{flag.severity.upper()}] {flag.description}", styles["Normal"]))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        # Fallback: generate a text-based "report" as bytes
        lines = [
            f"CALL CENTER ANALYSIS REPORT",
            f"Call ID: {report.call_id}",
            f"Status: {report.status}",
            f"Processed: {report.processed_at or 'N/A'}",
            "",
        ]
        if report.summary:
            lines += [
                "SUMMARY",
                f"Purpose: {report.summary.call_purpose}",
                f"Resolution: {report.summary.resolution_status.value}",
                f"Sentiment: {report.summary.sentiment_trajectory}",
                "",
            ]
        if report.qa_scores:
            lines += [
                "QA SCORES",
                f"Overall: {report.qa_scores.overall_score:.1f}/5.0",
            ]
            for dim in [
                report.qa_scores.professionalism,
                report.qa_scores.empathy,
                report.qa_scores.problem_resolution,
                report.qa_scores.compliance,
                report.qa_scores.communication_clarity,
            ]:
                lines.append(f"  {dim.dimension}: {dim.score}/5")
        return "\n".join(lines).encode("utf-8")
