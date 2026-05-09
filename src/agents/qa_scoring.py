"""QA Scoring agent with deterministic weighted score recomputation."""
from __future__ import annotations

import time
from typing import Any

from src.graph.state import QAScoreResult, SummaryResult, TranscriptionResult
from src.utils.formatters import secs_to_mmss
from src.utils.llm_factory import get_llm

DIMENSION_WEIGHTS = {
    "professionalism": 0.15,
    "empathy": 0.20,
    "problem_resolution": 0.30,
    "compliance": 0.20,
    "communication_clarity": 0.15,
}


class QAScoringError(Exception):
    pass


_SYSTEM_PROMPT = """You are a call center quality assurance specialist. Score the agent on five dimensions.

## Scoring Philosophy
- Score 3 is the BASELINE for competent, adequate handling
- Scores of 4 and 5 must be EARNED with specific evidence
- Score 1 means critical failure; score 2 means below expectations
- Do NOT inflate scores — most calls should score 2-4

## Dimension Rubrics

### Professionalism (15%)
1: Rude, unprofessional language, poor greeting/closing
2: Inconsistent professionalism, minor interruptions
3: Adequate professional conduct, standard greeting
4: Consistently professional, composed under pressure
5: Exemplary conduct, warm and polished throughout

### Empathy (20%)
1: Dismissive, shows no understanding of customer feelings
2: Minimal acknowledgment
3: Acknowledges customer concern appropriately
4: Active listening, personalizes responses
5: Excellent rapport, highly personalized empathetic responses

### Problem Resolution (30%)
1: Issue not addressed at all
2: Partial resolution, wrong root cause
3: Issue addressed adequately
4: Clear root cause identification, confirmed customer understanding
5: Thorough resolution, proactive follow-up

### Compliance (20%)
1: Critical compliance violations (no identity verification, data exposure)
2: Multiple procedural lapses
3: Required disclosures met
4: All procedures followed correctly
5: Exemplary compliance, exceeds requirements

### Communication Clarity (15%)
1: Confusing, excessive jargon, no comprehension check
2: Unclear at times, difficult to follow
3: Generally clear, minimal jargon
4: Structured and clear explanations, comprehension confirmed
5: Exceptional clarity, proactive comprehension checks

## Compliance Flag Guidance
Only flag GENUINE procedural violations, not style preferences.
Severity levels: low (minor lapse), medium (procedural), high (regulatory), critical (identity/data breach risk)

## Justifications
Cite specific transcript timestamps (MM:SS format) and behave like a real QA coach.
Short calls are efficient, not deficient."""


def _recompute_overall_score(result: QAScoreResult) -> float:
    score = (
        result.professionalism.score * DIMENSION_WEIGHTS["professionalism"]
        + result.empathy.score * DIMENSION_WEIGHTS["empathy"]
        + result.problem_resolution.score * DIMENSION_WEIGHTS["problem_resolution"]
        + result.compliance.score * DIMENSION_WEIGHTS["compliance"]
        + result.communication_clarity.score * DIMENSION_WEIGHTS["communication_clarity"]
    )
    return round(score, 2)


def run_qa_scoring(
    transcript: TranscriptionResult,
    summary: SummaryResult | None = None,
    provider: str | None = None,
    max_retries: int = 3,
) -> QAScoreResult:
    llm = get_llm(provider=provider)

    formatted_transcript = "\n".join(
        f"[{secs_to_mmss(s.start)}-{secs_to_mmss(s.end)}] {s.speaker}: {s.text}"
        for s in transcript.segments
    ) or transcript.full_text

    summary_context = ""
    if summary:
        summary_context = (
            f"\n\nCall Summary Context:\n"
            f"Purpose: {summary.call_purpose}\n"
            f"Resolution: {summary.resolution_status.value}\n"
            f"Sentiment: {summary.sentiment_trajectory}"
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Score this call center interaction:\n\n{formatted_transcript}{summary_context}"
            ),
        },
    ]

    structured_llm = llm.with_structured_output(QAScoreResult)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: QAScoreResult = structured_llm.invoke(messages)
            # Always recompute — discard LLM's overall_score
            recomputed = _recompute_overall_score(result)
            result = result.model_copy(
                update={"call_id": transcript.call_id, "overall_score": recomputed}
            )
            return result
        except Exception as e:
            last_exc = e
            wait = min(2**attempt, 10)
            time.sleep(wait)

    raise QAScoringError(
        f"QA scoring failed after {max_retries} attempts: {last_exc}"
    ) from last_exc
