"""
Unit tests for locisimiles.pipeline._types module.
Tests type definitions and utility functions.
"""

import sys
from io import StringIO

from locisimiles.document import TextSegment
from locisimiles.pipeline._types import (
    Candidate,
    CandidateGeneratorOutput,
    CandidateJudge,
    CandidateJudgeOutput,
    pretty_print,
)


class TestTypeStructures:
    """Tests for type structure definitions."""

    def test_candidate_generator_output_structure(self, sample_segments):
        """Test CandidateGeneratorOutput matches expected format."""
        output: CandidateGeneratorOutput = {
            "q1": [
                Candidate(segment=sample_segments[0], score=0.95),
                Candidate(segment=sample_segments[1], score=0.85),
            ],
            "q2": [
                Candidate(segment=sample_segments[2], score=0.75),
            ],
        }
        # Verify structure
        assert isinstance(output, dict)
        for qid, candidates in output.items():
            assert isinstance(qid, str)
            assert isinstance(candidates, list)
            for cand in candidates:
                assert isinstance(cand, Candidate)
                assert isinstance(cand.segment, TextSegment)
                assert isinstance(cand.score, float)

    def test_judge_output_structure(self, sample_segments):
        """Test CandidateJudgeOutput matches expected format."""
        output: CandidateJudgeOutput = {
            "q1": [
                CandidateJudge(
                    segment=sample_segments[0], candidate_score=0.95, judgment_score=0.88
                ),
                CandidateJudge(
                    segment=sample_segments[1], candidate_score=0.85, judgment_score=0.45
                ),
            ],
            "q2": [
                CandidateJudge(
                    segment=sample_segments[2], candidate_score=0.75, judgment_score=0.32
                ),
            ],
        }
        # Verify structure
        assert isinstance(output, dict)
        for qid, judgments in output.items():
            assert isinstance(qid, str)
            assert isinstance(judgments, list)
            for j in judgments:
                assert isinstance(j, CandidateJudge)
                assert isinstance(j.segment, TextSegment)
                assert isinstance(j.candidate_score, (float, type(None)))
                assert isinstance(j.judgment_score, float)

    def test_judgment_with_none_candidate_score(self, sample_segments):
        """Test CandidateJudge can have None for candidate_score (classification-only pipeline)."""
        j = CandidateJudge(segment=sample_segments[0], candidate_score=None, judgment_score=0.88)
        assert j.candidate_score is None
        assert j.judgment_score == 0.88


class TestPrettyPrint:
    """Tests for the pretty_print utility function."""

    def test_pretty_print_output(self, sample_segments):
        """Test pretty_print produces formatted output."""
        judge_output: CandidateJudgeOutput = {
            "query_1": [
                CandidateJudge(
                    segment=sample_segments[0], candidate_score=0.95, judgment_score=0.88
                ),
                CandidateJudge(
                    segment=sample_segments[1], candidate_score=0.75, judgment_score=0.45
                ),
            ],
        }

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        pretty_print(judge_output)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Verify output contains expected elements
        assert "query_1" in output
        assert "seg1" in output
        assert "candidate=" in output
        assert "judgment=" in output

    def test_pretty_print_multiple_queries(self, sample_segments):
        """Test pretty_print handles multiple queries."""
        judge_output: CandidateJudgeOutput = {
            "q1": [
                CandidateJudge(segment=sample_segments[0], candidate_score=0.9, judgment_score=0.8)
            ],
            "q2": [
                CandidateJudge(segment=sample_segments[1], candidate_score=0.7, judgment_score=0.6)
            ],
            "q3": [
                CandidateJudge(segment=sample_segments[2], candidate_score=0.5, judgment_score=0.4)
            ],
        }

        captured_output = StringIO()
        sys.stdout = captured_output

        pretty_print(judge_output)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "q1" in output
        assert "q2" in output
        assert "q3" in output

    def test_pretty_print_none_candidate_score(self, sample_segments):
        """Test pretty_print handles None candidate scores."""
        judge_output: CandidateJudgeOutput = {
            "q1": [
                CandidateJudge(
                    segment=sample_segments[0], candidate_score=None, judgment_score=0.88
                )
            ],
        }

        captured_output = StringIO()
        sys.stdout = captured_output

        pretty_print(judge_output)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Should show "N/A" for None candidate_score
        assert "N/A" in output

    def test_pretty_print_empty_dict(self):
        """Test pretty_print handles empty CandidateJudgeOutput."""
        judge_output: CandidateJudgeOutput = {}

        captured_output = StringIO()
        sys.stdout = captured_output

        pretty_print(judge_output)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Should produce no output (or just newlines)
        assert output.strip() == ""

    def test_pretty_print_empty_results(self):
        """Test pretty_print handles query with no candidates."""
        judge_output: CandidateJudgeOutput = {
            "q1": [],
        }

        captured_output = StringIO()
        sys.stdout = captured_output

        pretty_print(judge_output)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "q1" in output

    def test_pretty_print_formatting(self, sample_segments):
        """Test pretty_print uses correct number formatting."""
        judge_output: CandidateJudgeOutput = {
            "q1": [
                CandidateJudge(
                    segment=sample_segments[0], candidate_score=0.123456, judgment_score=0.987654
                )
            ],
        }

        captured_output = StringIO()
        sys.stdout = captured_output

        pretty_print(judge_output)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check formatting (3 decimal places for candidate and judgment)
        assert "+0.123" in output or "0.123" in output
        assert "0.988" in output  # Rounded


class TestScoreType:
    """Tests for ScoreT type."""

    def test_score_is_float(self):
        """Test that scores are floats."""
        from locisimiles.pipeline._types import ScoreT

        score: ScoreT = 0.95
        assert isinstance(score, float)

    def test_score_range(self):
        """Test scores can be in valid ranges."""
        from locisimiles.pipeline._types import ScoreT

        # Probability scores [0, 1]
        prob_score: ScoreT = 0.5
        assert 0.0 <= prob_score <= 1.0

        # Similarity scores (cosine) [-1, 1]
        sim_score: ScoreT = -0.3
        assert -1.0 <= sim_score <= 1.0
