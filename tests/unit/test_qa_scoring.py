"""Unit tests for QA scoring agent (mocked LLM)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.qa_scoring import QAScoringError, _recompute_overall_score, run_qa_scoring
from src.graph.state import (
    ComplianceFlag,
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def _make_transcript() -> TranscriptionResult:
    return TranscriptionResult(
        call_id="qa-test",
        full_text="I want to help you resolve this issue.",
        segments=[
            TranscriptionSegment(start=0.0, end=3.0, text="I want to help you.", speaker="Agent", confidence=0.9)
        ],
    )


def _make_dim(score: int) -> QADimensionScore:
    return QADimensionScore(dimension="X", score=score, justification="ok")


def _make_qa_result(scores: list[int] = None, flags: list = None) -> QAScoreResult:
    scores = scores or [3, 3, 3, 3, 3]
    return QAScoreResult(
        professionalism=_make_dim(scores[0]),
        empathy=_make_dim(scores[1]),
        problem_resolution=_make_dim(scores[2]),
        compliance=_make_dim(scores[3]),
        communication_clarity=_make_dim(scores[4]),
        overall_score=3.0,  # will be overridden
        compliance_flags=flags or [],
    )


class TestRecomputeOverallScore:
    def test_all_fives_gives_five(self):
        qa = _make_qa_result([5, 5, 5, 5, 5])
        assert _recompute_overall_score(qa) == 5.0

    def test_all_ones_gives_one(self):
        qa = _make_qa_result([1, 1, 1, 1, 1])
        assert _recompute_overall_score(qa) == 1.0

    def test_weighted_calculation(self):
        qa = _make_qa_result([4, 4, 4, 4, 4])
        # 4 * (0.15 + 0.20 + 0.30 + 0.20 + 0.15) = 4.0
        assert _recompute_overall_score(qa) == pytest.approx(4.0, abs=0.01)

    def test_llm_score_overridden(self):
        """LLM returns overall_score=3.0 but all dims are 5."""
        qa = _make_qa_result([5, 5, 5, 5, 5])
        qa = qa.model_copy(update={"overall_score": 3.0})
        recomputed = _recompute_overall_score(qa)
        assert recomputed == 5.0


class TestRunQAScoring:
    def test_returns_qa_result(self):
        transcript = _make_transcript()
        mock_qa = _make_qa_result([4, 4, 4, 4, 4])

        with patch("src.agents.qa_scoring.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_qa
            mock_get_llm.return_value = mock_llm

            result = run_qa_scoring(transcript)

        assert isinstance(result, QAScoreResult)
        assert result.call_id == "qa-test"

    def test_overall_score_recomputed_not_llm_value(self):
        """Critical: LLM returns overall=3.0 but dimensions all score 5."""
        transcript = _make_transcript()
        mock_qa = _make_qa_result([5, 5, 5, 5, 5])
        mock_qa = mock_qa.model_copy(update={"overall_score": 3.0})

        with patch("src.agents.qa_scoring.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_qa
            mock_get_llm.return_value = mock_llm

            result = run_qa_scoring(transcript)

        assert result.overall_score == pytest.approx(5.0, abs=0.01)

    def test_call_id_set(self):
        transcript = _make_transcript()
        mock_qa = _make_qa_result()

        with patch("src.agents.qa_scoring.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_qa
            mock_get_llm.return_value = mock_llm

            result = run_qa_scoring(transcript)

        assert result.call_id == transcript.call_id

    def test_raises_after_max_retries(self):
        transcript = _make_transcript()

        with patch("src.agents.qa_scoring.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API error")
            mock_get_llm.return_value = mock_llm

            with patch("src.agents.qa_scoring.time.sleep"):
                with pytest.raises(QAScoringError):
                    run_qa_scoring(transcript, max_retries=3)
