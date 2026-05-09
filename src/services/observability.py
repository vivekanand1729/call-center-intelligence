"""Observability service: pipeline health metrics and audit dashboard."""
from __future__ import annotations

from typing import Any


def get_observability_dashboard() -> tuple[str, str, list[list[str]]]:
    """Return (metrics_md, langsmith_md, audit_rows)."""
    try:
        from src.database.connection import session_scope
        from src.database.models import AuditLogEntry, CallRecord
        from sqlalchemy import func

        with session_scope() as session:
            total = session.query(func.count(CallRecord.id)).scalar() or 0
            completed = session.query(func.count(CallRecord.id)).filter(CallRecord.status == "completed").scalar() or 0
            failed = session.query(func.count(CallRecord.id)).filter(CallRecord.status == "failed").scalar() or 0
            flagged = session.query(func.count(CallRecord.id)).filter(CallRecord.status == "flagged_for_review").scalar() or 0

            # Average QA score from qa_scores_json
            import json
            qa_rows = session.query(CallRecord.qa_scores_json).filter(
                CallRecord.status == "completed",
                CallRecord.qa_scores_json.isnot(None),
            ).all()
            scores = []
            for (qa_json,) in qa_rows:
                try:
                    d = json.loads(qa_json)
                    scores.append(float(d.get("overall_score", 0)))
                except Exception:
                    pass
            avg_qa = round(sum(scores) / len(scores), 2) if scores else 0.0

            # Compliance flags
            total_flags = 0
            for (qa_json,) in qa_rows:
                try:
                    d = json.loads(qa_json)
                    total_flags += len(d.get("compliance_flags", []))
                except Exception:
                    pass

            # Recent audit events
            events = (
                session.query(AuditLogEntry)
                .order_by(AuditLogEntry.timestamp.desc())
                .limit(20)
                .all()
            )
            audit_rows = [
                [
                    e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else "",
                    e.call_id[:8] + "…" if e.call_id else "",
                    e.action,
                    e.details or "",
                ]
                for e in events
            ]

            total_audit = session.query(func.count(AuditLogEntry.id)).scalar() or 0

        success_rate = round(completed / total * 100, 1) if total > 0 else 0.0

        metrics_md = f"""### Pipeline Health

| Metric | Value |
|--------|-------|
| Total Calls Processed | {total} |
| Completed | {completed} |
| Failed | {failed} |
| Flagged for Review | {flagged} |
| Success Rate | {success_rate}% |
| Average QA Score | {avg_qa}/5.0 |
| Total Compliance Flags | {total_flags} |
| Total Audit Events | {total_audit} |
"""

    except Exception as e:
        metrics_md = f"_Error loading metrics: {e}_"
        audit_rows = []

    import os
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "call-center-intelligence")
    if langsmith_enabled:
        langsmith_md = f"**LangSmith:** ✅ Enabled — Project: `{project}`"
    else:
        langsmith_md = "**LangSmith:** ⚠️ Disabled — Set `LANGCHAIN_TRACING_V2=true` to enable tracing"

    return metrics_md, langsmith_md, audit_rows
