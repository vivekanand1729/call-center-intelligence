"""Unit tests for summarization agent (mocked LLM)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.summarization import SummarizationError, run_summarization
from src.graph.state import (
    ActionItem,
    Entity,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def _make_transcript(text: str = "Hello how can I help?") -> TranscriptionResult:
    return TranscriptionResult(
        call_id="test-call",
        full_text=text,
        segments=[
            TranscriptionSegment(start=0.0, end=3.0, text=text, speaker="Agent", confidence=0.9)
        ],
    )


def _make_summary_result() -> SummaryResult:
    return SummaryResult(
        call_id="",
        call_purpose="Account inquiry",
        key_discussion_points=["Customer asked about billing"],
        action_items=[ActionItem(description="Send invoice", owner="Agent")],
        resolution_status=ResolutionStatus.resolved,
        sentiment_trajectory="Neutral → Satisfied",
        entities=[Entity(name="John Smith", entity_type="person")],
    )


class TestRunSummarization:
    def test_returns_summary_result(self):
        transcript = _make_transcript()
        mock_result = _make_summary_result()

        with patch("src.agents.summarization.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_result
            mock_llm.with_structured_output.return_value = mock_structured
            mock_get_llm.return_value = mock_llm

            result = run_summarization(transcript)

        assert isinstance(result, SummaryResult)
        assert result.call_id == "test-call"
        assert result.call_purpose == "Account inquiry"

    def test_call_id_set_from_transcript(self):
        transcript = _make_transcript()
        mock_result = _make_summary_result()

        with patch("src.agents.summarization.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_result
            mock_get_llm.return_value = mock_llm
            result = run_summarization(transcript)

        assert result.call_id == transcript.call_id

    def test_raises_after_max_retries(self):
        transcript = _make_transcript()

        with patch("src.agents.summarization.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API error")
            mock_get_llm.return_value = mock_llm

            with patch("src.agents.summarization.time.sleep"):
                with pytest.raises(SummarizationError):
                    run_summarization(transcript, max_retries=3)

    def test_retry_called_exactly_max_times(self):
        transcript = _make_transcript()
        call_count = 0

        def raise_exc(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("fail")

        with patch("src.agents.summarization.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.side_effect = raise_exc
            mock_get_llm.return_value = mock_llm

            with patch("src.agents.summarization.time.sleep"):
                with pytest.raises(SummarizationError):
                    run_summarization(transcript, max_retries=3)

        assert call_count == 3
