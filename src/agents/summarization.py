"""Summarization agent with structured Pydantic output and exponential backoff."""
from __future__ import annotations

import time
from typing import Any

from src.graph.state import SummaryResult, TranscriptionResult
from src.utils.formatters import secs_to_mmss
from src.utils.llm_factory import get_llm


class SummarizationError(Exception):
    pass


_SYSTEM_PROMPT = """You are an expert call center analyst. Analyze the provided call transcript and extract structured information.

Be precise and ground all observations in the actual transcript content.
Do not fabricate information not present in the transcript."""


def _format_transcript_for_llm(transcript: TranscriptionResult) -> str:
    lines = []
    for seg in transcript.segments:
        ts = f"[{secs_to_mmss(seg.start)}-{secs_to_mmss(seg.end)}]"
        lines.append(f"{ts} {seg.speaker}: {seg.text}")
    return "\n".join(lines) if lines else transcript.full_text


def run_summarization(
    transcript: TranscriptionResult,
    provider: str | None = None,
    max_retries: int = 3,
) -> SummaryResult:
    llm = get_llm(provider=provider)
    formatted = _format_transcript_for_llm(transcript)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analyze this call center transcript and return structured JSON:\n\n{formatted}"
            ),
        },
    ]
    structured_llm = llm.with_structured_output(SummaryResult)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: SummaryResult = structured_llm.invoke(messages)
            result = result.model_copy(update={"call_id": transcript.call_id})
            return result
        except Exception as e:
            last_exc = e
            wait = min(2**attempt, 10)
            time.sleep(wait)
    raise SummarizationError(
        f"Summarization failed after {max_retries} attempts: {last_exc}"
    ) from last_exc
