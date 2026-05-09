"""Display formatters: secs_to_mmss, format_summary, format_qa."""
from __future__ import annotations

from typing import Any


def secs_to_mmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(round(seconds))
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


def format_summary(summary: Any) -> str:
    if summary is None:
        return "_No summary available._"
    lines = ["## Call Summary", ""]
    lines.append(f"**Purpose:** {summary.call_purpose}")
    lines.append(f"**Resolution:** {summary.resolution_status.value}")
    lines.append(f"**Sentiment Trajectory:** {summary.sentiment_trajectory}")
    lines.append("")
    lines.append("### Key Discussion Points")
    for point in summary.key_discussion_points:
        lines.append(f"- {point}")
    if summary.action_items:
        lines.append("")
        lines.append("### Action Items")
        for item in summary.action_items:
            deadline = f" _(by {item.deadline})_" if item.deadline else ""
            lines.append(f"- **{item.owner}**: {item.description}{deadline}")
    if summary.entities:
        lines.append("")
        lines.append("### Named Entities")
        for e in summary.entities:
            lines.append(f"- `{e.name}` ({e.entity_type})")
    return "\n".join(lines)


_SEVERITY_ICONS = {
    "low": "ℹ️",
    "medium": "⚠️",
    "high": "\U0001f536",
    "critical": "🔴",
}


def format_qa(qa: Any) -> str:
    if qa is None:
        return "_No QA scores available._"
    lines = ["## QA Scorecard", ""]
    lines.append(f"**Overall Score: {qa.overall_score:.1f} / 5.0**")
    lines.append("")
    lines.append("| Dimension | Score | Justification |")
    lines.append("|-----------|-------|---------------|")
    dims = [
        qa.professionalism,
        qa.empathy,
        qa.problem_resolution,
        qa.compliance,
        qa.communication_clarity,
    ]
    for d in dims:
        lines.append(f"| {d.dimension} | {d.score}/5 | {d.justification} |")
    lines.append("")
    if qa.compliance_flags:
        lines.append("### Compliance Flags")
        for flag in qa.compliance_flags:
            icon = _SEVERITY_ICONS.get(flag.severity.lower(), "⚠️")
            ts = f" @ {flag.timestamp_reference}" if flag.timestamp_reference else ""
            lines.append(f"- {icon} **{flag.severity.upper()}**{ts}: {flag.description}")
    else:
        lines.append("No compliance issues detected.")
    return "\n".join(lines)


def format_transcript(transcription: Any) -> str:
    if transcription is None:
        return "_No transcript available._"
    lines = []
    for seg in transcription.segments:
        ts = f"[{secs_to_mmss(seg.start)}-{secs_to_mmss(seg.end)}]"
        low_conf = " [LOW CONF]" if seg.confidence < 0.4 else ""
        lines.append(f"{ts} {seg.speaker}: {seg.text}{low_conf}")
    return "\n".join(lines) if lines else transcription.full_text
